#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import warnings
import json  # <--- NUOVO IMPORT
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from ecmwf.opendata import Client
import pytz
import boto3
import mimetypes

warnings.filterwarnings("ignore")

# ==============================================================================
# ⚙️ CONFIGURAZIONE PATH
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, "mappe_output")
os.makedirs(OUTDIR, exist_ok=True)

print(f"📂 Output directory: {OUTDIR}")

# ==================== RUN ECMWF ====================
now = datetime.utcnow()
if now.hour < 8:
    run_hour = 12
    run_date = now.date() - timedelta(days=1)
elif now.hour < 20:
    run_hour = 0
    run_date = now.date()
else:
    run_hour = 12
    run_date = now.date()

run_dt = datetime(run_date.year, run_date.month, run_date.day, run_hour)

DATE_STR = run_dt.strftime("%Y-%m-%d")
DATE_TAG = run_dt.strftime("%Y%m%d")
RUN_STR = f"{run_hour:02d}z"
RUN_ID_STRING = f"{DATE_TAG}{run_hour:02d}"

# Formato data per il JSON (compatibile con app Android: yyyyMMdd_HHmm)
JSON_RUN_DATE = run_dt.strftime("%Y%m%d_%H%M") 

print(f"🚀 Inizio elaborazione run {DATE_TAG} {RUN_STR} (ID: {RUN_ID_STRING})")

# ==================== GENERA FILE TXT INFO RUN (Legacy) ====================
# Lo manteniamo per sicurezza, ma il JSON sarà il principale
txt_path = os.path.join(OUTDIR, "datarun.txt")
try:
    with open(txt_path, "w") as f:
        f.write(RUN_ID_STRING)
except Exception as e:
    pass

# ==================== LISTA PER IL JSON ====================
# Qui accumuleremo tutti i file generati
generated_files = [] 

# ==================== TIMEZONE SETUP ====================
tz_rome = pytz.timezone('Europe/Rome')

# ==================== NOMI FILE DATI ====================
FILES = {
    "pl_850":     os.path.join(OUTDIR, f"pl_850_{DATE_TAG}_{RUN_STR}.grib2"),
    "pl_700":     os.path.join(OUTDIR, f"pl_700_{DATE_TAG}_{RUN_STR}.grib2"),
    "pl_500":     os.path.join(OUTDIR, f"pl_500_{DATE_TAG}_{RUN_STR}.grib2"),
    "pl_300":     os.path.join(OUTDIR, f"pl_300_{DATE_TAG}_{RUN_STR}.grib2"),
    "sfc_thermo": os.path.join(OUTDIR, f"sfc_thermo_{DATE_TAG}_{RUN_STR}.grib2"),
    "sfc_prec":   os.path.join(OUTDIR, f"sfc_prec_{DATE_TAG}_{RUN_STR}.grib2"),
}

# ==================== CLEAN OLD RUNS ====================
def clean_old_runs(outdir, date_tag, current_run):
    if not os.path.exists(outdir): return
    for fname in os.listdir(outdir):
        if fname.endswith(".grib2"):
            if date_tag not in fname or current_run not in fname:
                try:
                    os.remove(os.path.join(outdir, fname))
                except Exception:
                    pass

clean_old_runs(OUTDIR, DATE_TAG, RUN_STR)

# ==================== STEPS ====================
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))
print(f"▶ Steps totali richiesti: {len(steps)}")

# ==================== DOWNLOAD DATA ====================
client = Client(source="ecmwf")

def check_and_download(key, params, levels=None, levtype="pl"):
    fpath = FILES[key]
    if os.path.exists(fpath):
        print(f"▶ {key} OK (file esistente)")
        return

    print(f"⬇️ Scarico {key}...")
    request = {
        "date": DATE_STR,
        "time": run_hour,
        "step": steps,
        "stream": "oper",
        "type": "fc",
        "param": params,
        "levtype": levtype,
        "target": fpath
    }
    if levels is not None:
        request["levelist"] = levels

    try:
        client.retrieve(**request)
        print(f"✅ {key} scaricato!")
    except Exception as e:
        print(f"❌ Errore download {key}: {e}")
        if "sfc" in key:
            sys.exit(1)

check_and_download("pl_850", ["t","gh"], [850])
check_and_download("pl_700", ["u","v","r", "gh"], [700])
check_and_download("pl_500", ["t","gh","u","v"], [500])
check_and_download("pl_300", ["u","v","gh"], [300])
check_and_download("sfc_thermo", ["2t","2d","msl"], None, levtype="sfc")
check_and_download("sfc_prec", ["tp"], None, levtype="sfc")


# ==================== PHYSICS ====================
def calculate_wet_bulb_stull(t_celsius, rh_percent):
    tw = t_celsius * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) + \
         np.arctan(t_celsius + rh_percent) - \
         np.arctan(rh_percent - 1.676331) + \
         0.00391838 * (rh_percent ** 1.5) * np.arctan(0.023101 * rh_percent) - \
         4.686035
    return tw

def dewpoint_to_rh(t_celsius, td_celsius):
    a, b = 17.27, 237.7
    es = 6.112 * np.exp((a * t_celsius) / (b + t_celsius))
    e = 6.112 * np.exp((a * td_celsius) / (b + td_celsius))
    rh = 100 * (e / es)
    return np.clip(rh, 0, 100)

# ==================== DATASETS ====================
ds = {}
try:
    ds[850] = xr.open_dataset(FILES["pl_850"], engine="cfgrib")
    ds[700] = xr.open_dataset(FILES["pl_700"], engine="cfgrib")
    ds[500] = xr.open_dataset(FILES["pl_500"], engine="cfgrib")
    ds[300] = xr.open_dataset(FILES["pl_300"], engine="cfgrib")
    sfc_t = xr.open_dataset(FILES["sfc_thermo"], engine="cfgrib")
    sfc_p = xr.open_dataset(FILES["sfc_prec"], engine="cfgrib")
    ds["sfc"] = xr.merge([sfc_t, sfc_p], compat='override')
except Exception as e:
    print(f"❌ Errore apertura file grib: {e}")
    sys.exit()

# ==================== DOMINIO & COLORMAPS ====================
nord, sud, ovest, est = 70, 28, -28, 48

# (COLORMAPS INVARIATE - Tagliate per brevità, mantieni le tue originali)
# ... T850, PRECIP, NEVE, RH, VENTO ...
# Assicurati di copiare le tue definizioni di colormap qui
# -----------------------------------------------------------------
colors_t = ["#ad99ad", "#948094", "#7a667a", "#614D61", "#473347", "#3D1A57", "#330066", "#460073", "#59007f", "#6C00BF", "#7f00ff", "#4040FF", "#007fff", "#00A6FF", "#00ccff", "#00E6FF", "#00ffff", "#13F2CC", "#26e599", "#56C943", "#66bf26", "#93D226", "#bfe526", "#EFF969", "#ffff7f", "#FFFF5C", "#ffff00", "#FFEC00", "#ffd900", "#FFC500", "#ffb000", "#FF9100", "#ff7200", "#FF3900", "#ff0000", "#E60000", "#cc0000", "#A60016", "#7f002c", "#A61F4D", "#cc3d6e", "#E61FB7", "#ff00ff", "#FF40FF", "#ff7fff", "#ffbfff"]
boundaries_t = np.arange(-46, 48, 2)
cmap_t = ListedColormap(colors_t)
norm_t = BoundaryNorm(boundaries_t, cmap_t.N)

colors_p = ["#ffffff", "#C2E7FF", "#47BFFF", "#0055ff", "#0000aa", "#32cd32", "#008000", "#ffff00", "#ff9900", "#b30000", "#ff00ff", "#4b0082"]
boundaries_p = [0,0.1,0.5,1,3,5,7,10,15,20,30,40,50]
cmap_p = ListedColormap(colors_p)
norm_p = BoundaryNorm(boundaries_p, cmap_p.N, clip=False)

colors_snow = ["#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#e0e0e0", "#c0c0c0", "#a0a0a0", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177", "#49006a"]
boundaries_snow = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300]
cmap_snow = ListedColormap(colors_snow)
cmap_snow.set_under('none')
norm_snow = BoundaryNorm(boundaries_snow, cmap_snow.N, clip=False)

colors_rh = ["#32CD32", "#87CEFA", "#0000CD"]
boundaries_rh = [65, 80, 95, 100]
cmap_rh = ListedColormap(colors_rh)
cmap_rh.set_under('none')
norm_rh = BoundaryNorm(boundaries_rh, cmap_rh.N, clip=False)

colors_w = ["#ffffff","#e0f8ff","#b3ecff","#80dfff","#00ccff","#0099ff","#0066ff","#0033ff","#33cc33","#33ff33", "#ffff00","#ffcc00","#ff9900","#ff6600","#ff3300","#cc0000","#800000"]
boundaries_w = [0,15,30,45,60,75,90,105,120,135,150,180,210,240,270,300,330]
cmap_w = ListedColormap(colors_w)
norm_w = BoundaryNorm(boundaries_w, cmap_w.N)


# ==================== MAP SETUP ====================
def clip_lon_lat(data):
    return data.sel(latitude=slice(nord, sud), longitude=slice(ovest, est))

def setup_map():
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidths=0.4)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.4)
    return fig, ax

def add_title(ax, main_title, valid_dt, lead_str):
    valid_str = valid_dt.strftime("%d/%m/%Y")
    valid_str_2 = valid_dt.strftime("%H")
    valid_txt = rf"\bf{{{valid_str}}}"
    valid_txt_2 = rf"\bf{{{valid_str_2} {lead_str}}}"
    title = (rf"$\bf{{{main_title}}}$"
             f"\n$\\bf{{ECMWF\\ IFS}}$ run: {run_dt.strftime('%d/%m/%Y')} {run_hour:02d}z "
             f"| validità: ${valid_txt}$ ${valid_txt_2}$")
    ax.set_title(title)

# ==================== MAIN LOOP ====================
prev_tp = None
snowpack = None
prev_step_h = 0
ref_steps = ds[850].step.values 

for idx, step_td in enumerate(ref_steps):
    step_h = int(step_td / np.timedelta64(1, "h"))
    
    valid_dt_utc = run_dt + timedelta(hours=step_h)
    valid_dt = pytz.utc.localize(valid_dt_utc).astimezone(tz_rome)
    lead_str = f"(+{step_h}h)"
    
    if idx == 0:
        dt_hours = step_h
    else:
        dt_hours = step_h - prev_step_h
    prev_step_h = step_h

    # --- LETTURA DATI ---
    try:
        t850 = clip_lon_lat(ds[850].t.sel(step=step_td, method="nearest") - 273.15)
        gh500 = clip_lon_lat(ds[500].gh.sel(step=step_td, method="nearest") / 10)
        t500 = clip_lon_lat(ds[500].t.sel(step=step_td, method="nearest") - 273.15)
        u500 = clip_lon_lat(ds[500].u.sel(step=step_td, method="nearest"))
        v500 = clip_lon_lat(ds[500].v.sel(step=step_td, method="nearest"))
        ws500 = np.sqrt(u500**2 + v500**2) * 3.6
        
        u700 = clip_lon_lat(ds[700].u.sel(step=step_td, method="nearest"))
        v700 = clip_lon_lat(ds[700].v.sel(step=step_td, method="nearest"))
        gh700 = clip_lon_lat(ds[700].gh.sel(step=step_td, method="nearest") / 10)
        r700 = clip_lon_lat(ds[700].r.sel(step=step_td, method="nearest")).where(lambda x: x <= 100, 99.99)
        
        u300 = clip_lon_lat(ds[300].u.sel(step=step_td, method="nearest"))
        v300 = clip_lon_lat(ds[300].v.sel(step=step_td, method="nearest"))
        ws300 = np.sqrt(u300**2 + v300**2) * 3.6
        gh300 = clip_lon_lat(ds[300].gh.sel(step=step_td, method="nearest") / 10)
        
        msl = clip_lon_lat(ds["sfc"].msl.sel(step=step_td, method="nearest") / 100)
        t2m = clip_lon_lat(ds["sfc"].t2m.sel(step=step_td, method="nearest") - 273.15)
        d2m = clip_lon_lat(ds["sfc"].d2m.sel(step=step_td, method="nearest") - 273.15)
    except KeyError as e:
        print(f"⚠️ Dati mancanti step +{step_h}h: {e}")
        continue

    # --- CALCOLO PRECIP E NEVE ---
    prec = None
    if "tp" in ds["sfc"].data_vars:
        try:
            tp_curr = ds["sfc"].tp.sel(step=step_td, method="nearest").fillna(0) * 1000
            if prev_tp is None:
                prec = tp_curr.copy()
            else:
                prec = (tp_curr - prev_tp).clip(min=0)
            prev_tp = tp_curr
            prec = clip_lon_lat(prec)

            rh = dewpoint_to_rh(t2m.values, d2m.values)
            tw = calculate_wet_bulb_stull(t2m.values, rh)
            
            if snowpack is None:
                snowpack = xr.zeros_like(prec)
            
            new_snow = prec.copy()
            new_snow.values = np.where(tw < 0.5, prec.values, 0.0)
            
            melt_rate = 0.7
            tw_excess = tw - 0.5
            melt = xr.zeros_like(prec)
            melt.values = np.maximum(tw_excess, 0) * melt_rate * dt_hours
            
            snowpack = (snowpack + new_snow - melt).clip(min=0)
        except Exception as e:
            prec = None

    # --- PLOT MAPS & ADD TO JSON LIST ---
    
    # 1. T850
    fname = f"T850_GH500_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(t850.longitude, t850.latitude, t850, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    cs_t = ax.contour(t850.longitude, t850.latitude, t850, levels=np.arange(-48,49,4), colors='dimgray', linewidths=0.01)
    ax.clabel(cs_t, fmt='%d', fontsize=4)
    cs_t2 = ax.contour(t850.longitude, t850.latitude, t850, levels=np.arange(-48,49,8), colors='dimgray', linewidths=0.4)
    ax.clabel(cs_t2, fmt='%d', fontsize=4)
    gh_levels = np.arange(460,600,4)
    gh_lw = [1.2 if (abs(l-544)%16==0) else 0.6 for l in gh_levels]
    cs_gh = ax.contour(gh500.longitude, gh500.latitude, gh500, levels=gh_levels, colors='black', linewidths=gh_lw)
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, "Temperatura\ 850hPa\ -\ Altezza\ di\ Geopotenziale\ 500\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.8, label="Temperatura (°C)", ticks=np.arange(-44,45,4))
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    
    generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    # 2. T500
    fname = f"T500_GH500_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(t500.longitude, t500.latitude, t500, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    cs_t = ax.contour(t500.longitude, t500.latitude, t500, levels=np.arange(-48, 49, 4), colors='dimgray', linewidths=0.01)
    ax.clabel(cs_t, fmt='%d', fontsize=4)
    cs_t2 = ax.contour(t500.longitude, t500.latitude, t500, levels=np.arange(-48, 49, 8), colors='dimgray', linewidths=0.4)
    ax.clabel(cs_t2, fmt='%d', fontsize=4)
    cs_gh = ax.contour(gh500.longitude, gh500.latitude, gh500, levels=gh_levels, colors='black', linewidths=gh_lw)
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, "Temperatura\ 500hPa\ -\ Altezza\ di\ Geopotenziale\ 500\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.7, label="Temperatura (°C)", ticks=np.arange(-44,45,4))
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()

    generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    # 3. WIND500
    fname = f"WIND500_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(ws500.longitude, ws500.latitude, ws500, levels=boundaries_w, cmap=cmap_w, norm=norm_w, extend='max')
    cs_gh = ax.contour(gh500.longitude, gh500.latitude, gh500, levels=gh_levels, colors='black', linewidths=gh_lw)
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    ax.quiver(u500.longitude[::6], u500.latitude[::6], u500[::6, ::6], v500[::6, ::6], scale=1200, width=0.001, headwidth=3, headlength=4, color='#333333', alpha=0.7)
    add_title(ax, "Vento\ 500hPa\ -\ Altezza\ di\ Geopotenziale\ 500\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.7, label="Intensità del vento (km/h)")
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()

    generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    # 4. RH700
    fname = f"RH700_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf_rh = ax.contourf(r700.longitude, r700.latitude, r700, levels=boundaries_rh, cmap=cmap_rh, norm=norm_rh, extend='both')
    gh700_levs = np.arange(200, 530, 4)
    gh_lw700 = [1.2 if (abs(l - 316) % 20 == 0) else 0.6 for l in gh700_levs]
    cs_gh = ax.contour(gh700.longitude, gh700.latitude, gh700, levels=gh700_levs, colors='black', linewidths=gh_lw700)
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    ws700 = np.sqrt(u700**2 + v700**2) * 3.6
    ax.quiver(u700.longitude[::7], v700.latitude[::7], u700[::7, ::7], v700[::7, ::7], ws700[::7, ::7], scale=1000, width=0.0015, headwidth=3, headlength=4, cmap=cmap_w, alpha=0.9)
    add_title(ax, "Umidità\ Relativa\ 700hPa\ -\ Vento\ 700hPa\ -\ Altezza\ di\ Geopotenziale\ 700hPa", valid_dt, lead_str)
    cax_rh = fig.add_axes([0.17, 0.2, 0.3, 0.02])
    cax_wind = fig.add_axes([0.55, 0.2, 0.3, 0.02])
    cbar_rh = fig.colorbar(cf_rh, cax=cax_rh, orientation='horizontal', label="Umidità relativa (%)", ticks=[65, 80, 95, 100])
    cbar_rh.ax.tick_params(labelsize=8)
    from matplotlib.cm import ScalarMappable
    sm_wind = ScalarMappable(norm=norm_w, cmap=cmap_w)
    sm_wind.set_array([])
    cbar_wind = fig.colorbar(sm_wind, cax=cax_wind, orientation='horizontal', label="Intensità del vento (km/h)")
    cbar_wind.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()

    generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    # 5. JET300
    fname = f"JET300_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(ws300.longitude, ws300.latitude, ws300, levels=boundaries_w, cmap=cmap_w, norm=norm_w, extend='max')
    gh300_levs = np.arange(800, 1200, 4)
    gh_lw300 = [1.2 if (abs(l - 1000) % 20 == 0) else 0.6 for l in gh300_levs]
    cs_gh = ax.contour(gh300.longitude, gh300.latitude, gh300, levels=gh300_levs, colors='black', linewidths=gh_lw300)
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, "Jet\ Stream\ 300hPa\ -\ Altezza\ di\ Geopotenziale\ 300\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.7, label="Intensità del vento (km/h)")
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()

    generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    # 6. PREC_MSL
    if idx > 0 and prec is not None:
        fname = f"PREC_MSL_{step_h:03d}.webp"
        fig, ax = setup_map()
        cf = ax.contourf(prec.longitude, prec.latitude, prec, levels=boundaries_p, cmap=cmap_p, norm=norm_p, extend='max')
        msl_levels = np.arange(np.floor(msl.min()/5)*5, np.ceil(msl.max()/5)*5+1, 5)
        msl_lw = [1 if (abs(l-1010)%20==0) else 0.6 for l in msl_levels]
        cs_msl = ax.contour(msl.longitude, msl.latitude, msl, levels=msl_levels, colors='black', linewidths=msl_lw)
        ax.clabel(cs_msl, fmt='%d', fontsize=5)
        add_title(ax, "Precipitazione\ Totale\ 3h\ -\ Pressione\ al\ livello\ del\ mare", valid_dt, lead_str)
        tp_ticks = boundaries_p
        tp_labels = [str(t) if t < 1 else str(int(t)) for t in tp_ticks]
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.6, label="Precipitazione Totale 3h (mm)", ticks=tp_ticks)
        cbar.ax.set_xticklabels(tp_labels)
        cbar.ax.tick_params(labelsize=8)
        plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
        plt.close()

        generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    # 7. SNOWPACK
    if idx > 0 and snowpack is not None:
        fname = f"SNOWPACK_MSL_{step_h:03d}.webp"
        fig, ax = setup_map()
        cf = ax.contourf(snowpack.longitude, snowpack.latitude, snowpack, levels=boundaries_snow, cmap=cmap_snow, norm=norm_snow, extend='max')
        msl_levels = np.arange(np.floor(msl.min()/5)*5, np.ceil(msl.max()/5)*5+1, 5)
        msl_lw = [1 if (abs(l-1010)%20==0) else 0.6 for l in msl_levels]
        cs_msl = ax.contour(msl.longitude, msl.latitude, msl, levels=msl_levels, colors='black', linewidths=msl_lw)
        ax.clabel(cs_msl, fmt='%d', fontsize=5)
        add_title(ax, "Neve\ fresca\ al\ suolo\ -\ Pressione\ al\ livello\ del\ mare", valid_dt, lead_str)
        snow_ticks = boundaries_snow
        snow_labels = [str(t) if t < 1 else str(int(t)) for t in snow_ticks]
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.6, label="Neve fresca accumulata da inizio run (cm)", ticks=snow_ticks)
        cbar.ax.set_xticklabels(snow_labels)
        cbar.ax.tick_params(labelsize=8)
        plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
        plt.close()

        generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA

    print(f"✅ Step +{step_h}h completato")
    
    # 8. ITALIA
    def clip_italy(data):
        return data.sel(latitude=slice(48.9, 33.7), longitude=slice(3, 22))

    fname = f"italia_{step_h:03d}.webp"
    t850_it = clip_italy(t850)
    gh500_it = clip_italy(gh500)
    prec_it = clip_italy(prec) if prec is not None else None
    msl_it = clip_italy(msl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})

    # PANNELLO 1
    ax = axes[0]
    cf = ax.contourf(t850_it.longitude, t850_it.latitude, t850_it, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    cs_t = ax.contour(t850_it.longitude, t850_it.latitude, t850_it, levels=np.arange(-48, 49, 4), colors='gray', linewidths=0.6)
    ax.clabel(cs_t, fmt='%d°C', fontsize=5)
    cs_t2 = ax.contour(t850_it.longitude, t850_it.latitude, t850_it, levels=np.arange(-48, 49, 2), colors='gray', linewidths=0.01)
    gh_levels_it = np.arange(500, 600, 4)
    gh_lw_it = [1.4 if (abs(l - 544) % 16 == 0) else 0.8 for l in gh_levels_it]
    cs_gh = ax.contour(gh500_it.longitude, gh500_it.latitude, gh500_it, levels=gh_levels_it, colors='black', linewidths=gh_lw_it)
    ax.clabel(cs_gh, fmt='%d', fontsize=8)
    ax.coastlines(linewidths=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.5)
    ax.set_title(r"$\bf{Temperatura\ 850\ hPa - Altezza\ di\ Geopotenziale\ 500\ hPa}$", fontsize=12)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, label=r"Temperatura (°C)", ticks=np.arange(-32, 33, 4))
    cbar.ax.tick_params(labelsize=8)

    # PANNELLO 2
    ax = axes[1]
    if prec_it is not None:
        cf = ax.contourf(prec_it.longitude, prec_it.latitude, prec_it, levels=boundaries_p, cmap=cmap_p, norm=norm_p, extend="max")
        tp_ticks_labels = [str(t) if t < 1 else str(int(t)) for t in boundaries_p]
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, label=r"Precipitazione Totale (mm)", ticks=boundaries_p)
        cbar.ax.set_xticklabels(tp_ticks_labels)
        cbar.ax.tick_params(labelsize=8)
    
    msl_levels_it = np.arange(np.floor(msl_it.min() / 5) * 5, np.ceil(msl_it.max() / 5) * 5 + 1, 5)
    msl_lw_it = [1.4 if (abs(l - 1010) % 20 == 0) else 1 for l in msl_levels_it]
    cs_msl = ax.contour(msl_it.longitude, msl_it.latitude, msl_it, levels=msl_levels_it, colors='black', linewidths=msl_lw_it)
    ax.clabel(cs_msl, fmt='%d', fontsize=8)
    ax.coastlines(linewidths=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.5)
    ax.set_title(r"$\bf{Precipitazione\ 3h - Pressione\ al\ livello\ del\ mare}$", fontsize=12)

    valid_str = run_dt.strftime("%d/%m/%Y")
    valid_str_2 = valid_dt.strftime("%H") + f" (+{step_h}h)"
    timestep_date = valid_dt.strftime("%d/%m/%Y")

    fig.suptitle(
        fr"$\bf{{ECMWF\\ IFS}}$ run: {valid_str} {run_hour:02d}z | validità: {timestep_date} $\bf{{{valid_str_2}}}$",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    
    generated_files.append({"name": fname, "step": step_h}) # <--- AGGIUNTA
    print(f"✅ Mappa Italia +{step_h}h completata")

# ==================== SALVATAGGIO JSON FINALE ====================
print("💾 Generazione file status_ecmwf.json...")
json_data = {
    "run_date": JSON_RUN_DATE,  # es: "20260130_1200"
    "base_path": "ECMWF",
    "files": generated_files
}

json_path = os.path.join(OUTDIR, "status_ecmwf.json")
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)
print("✅ JSON Generato")

# ==================== UPLOAD SU CLOUDFLARE R2 ====================

def upload_to_r2():
    print("\n☁️ Avvio upload su Cloudflare R2...", flush=True)
    
    R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL") 
    R2_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
    R2_SECRET = os.environ.get("R2_SECRET_ACCESS_KEY")
    BUCKET_NAME = "mappe" 
    REMOTE_FOLDER = "ECMWF"

    if not all([R2_ENDPOINT, R2_KEY_ID, R2_SECRET, BUCKET_NAME]):
        print("⚠️ Credenziali R2 mancanti. Salto l'upload.", flush=True)
        return

    try:
        s3 = boto3.client(
            service_name='s3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_KEY_ID,
            aws_secret_access_key=R2_SECRET,
            region_name='auto'
        )

        # Includiamo anche il .json nella lista
        files = [f for f in os.listdir(OUTDIR) 
                 if os.path.isfile(os.path.join(OUTDIR, f)) 
                 and f.endswith(('.webp', '.txt', '.json'))]
        
        for filename in files:
            local_path = os.path.join(OUTDIR, filename)
            remote_key = f"{REMOTE_FOLDER}/{filename}"
            
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            # GESTIONE CACHE DIFFERENZIATA
            if filename.endswith(".json"):
                # Il JSON deve essere sempre fresco per segnalare nuovi run
                cache_control = "no-cache, no-store, must-revalidate"
            else:
                # Le immagini possono essere cachate (verranno invalidate dal cambio signature nel JSON)
                cache_control = "max-age=300"

            print(f"   ⬆️ Upload: {filename} -> {remote_key} [CC: {cache_control}]", end="... ", flush=True)
            
            s3.upload_file(
                local_path, 
                BUCKET_NAME, 
                remote_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'CacheControl': cache_control
                }
            )
            print("OK", flush=True)
            
        print("✅ Upload completato con successo!", flush=True)

    except Exception as e:
        print(f"\n❌ Errore durante l'upload R2: {e}", flush=True)

if __name__ == "__main__":
    upload_to_r2()


print("🏁 Elaborazione terminata!")
