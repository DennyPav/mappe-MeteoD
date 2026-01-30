#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import warnings
import json
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
# ⚙️ CONFIGURAZIONE PATH E R2 (SETUP INIZIALE)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, "mappe_output")
os.makedirs(OUTDIR, exist_ok=True)

print(f"📂 Output directory: {OUTDIR}")

# --- SETUP R2 CLIENT (Lo inizializziamo SUBITO) ---
R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL") 
R2_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET = os.environ.get("R2_SECRET_ACCESS_KEY")
BUCKET_NAME = "mappe" 
REMOTE_FOLDER = "ECMWF"

s3_client = None
if all([R2_ENDPOINT, R2_KEY_ID, R2_SECRET, BUCKET_NAME]):
    try:
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_KEY_ID,
            aws_secret_access_key=R2_SECRET,
            region_name='auto'
        )
        print("☁️ Client R2 inizializzato per upload incrementale.")
    except Exception as e:
        print(f"⚠️ Errore init R2: {e}. L'upload sarà saltato.")
else:
    print("⚠️ Credenziali R2 mancanti. Upload disabilitato.")

# Funzione Helper per Upload Singolo
def upload_single_file(filename):
    if s3_client is None: return
    
    local_path = os.path.join(OUTDIR, filename)
    remote_key = f"{REMOTE_FOLDER}/{filename}"
    
    content_type, _ = mimetypes.guess_type(local_path)
    if content_type is None: content_type = 'application/octet-stream'
    
    # Cache control differenziato
    cc = "no-cache, no-store, must-revalidate" if filename.endswith(".json") else "max-age=300"
    
    try:
        # print(f"   ⬆️ Uploading {filename}...", end=" ", flush=True) # Commentato per pulizia log
        s3_client.upload_file(
            local_path, 
            BUCKET_NAME, 
            remote_key,
            ExtraArgs={'ContentType': content_type, 'CacheControl': cc}
        )
        # print("OK") 
    except Exception as e:
        print(f"❌ Upload fallito per {filename}: {e}")


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
JSON_RUN_DATE = run_dt.strftime("%Y%m%d_%H%M") 

print(f"🚀 Inizio elaborazione run {DATE_TAG} {RUN_STR} (ID: {RUN_ID_STRING})")

generated_files = [] 
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
                try: os.remove(os.path.join(outdir, fname))
                except: pass

clean_old_runs(OUTDIR, DATE_TAG, RUN_STR)

# ==================== STEPS & DOWNLOAD ====================
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))
client = Client(source="ecmwf")

def check_and_download(key, params, levels=None, levtype="pl"):
    fpath = FILES[key]
    if os.path.exists(fpath): return

    print(f"⬇️ Scarico {key}...")
    request = {
        "date": DATE_STR, "time": run_hour, "step": steps, "stream": "oper",
        "type": "fc", "param": params, "levtype": levtype, "target": fpath
    }
    if levels: request["levelist"] = levels

    try:
        client.retrieve(**request)
    except Exception as e:
        print(f"❌ Errore download {key}: {e}")
        if "sfc" in key: sys.exit(1)

check_and_download("pl_850", ["t","gh"], [850])
check_and_download("pl_700", ["u","v","r", "gh"], [700])
check_and_download("pl_500", ["t","gh","u","v"], [500])
check_and_download("pl_300", ["u","v","gh"], [300])
check_and_download("sfc_thermo", ["2t","2d","msl"], None, levtype="sfc")
check_and_download("sfc_prec", ["tp"], None, levtype="sfc")

# ==================== PHYSICS & DATASETS ====================
def calculate_wet_bulb_stull(t, rh):
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331) + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh) - 4.686035)

def dewpoint_to_rh(t, td):
    return np.clip(100 * (6.112 * np.exp((17.27 * td) / (237.7 + td))) / (6.112 * np.exp((17.27 * t) / (237.7 + t))), 0, 100)

ds = {}
try:
    ds[850] = xr.open_dataset(FILES["pl_850"], engine="cfgrib")
    ds[700] = xr.open_dataset(FILES["pl_700"], engine="cfgrib")
    ds[500] = xr.open_dataset(FILES["pl_500"], engine="cfgrib")
    ds[300] = xr.open_dataset(FILES["pl_300"], engine="cfgrib")
    ds["sfc"] = xr.merge([xr.open_dataset(FILES["sfc_thermo"], engine="cfgrib"), xr.open_dataset(FILES["sfc_prec"], engine="cfgrib")], compat='override')
except Exception as e:
    print(f"❌ Errore grib: {e}"); sys.exit()

# ==================== MAP SETUP ====================
nord, sud, ovest, est = 70, 28, -28, 48

# COLORMAPS (Ridotte per brevità - mantieni le tue complete)
colors_t = ["#ad99ad", "#948094", "#7a667a", "#614D61", "#473347", "#3D1A57", "#330066", "#460073", "#59007f", "#6C00BF", "#7f00ff", "#4040FF", "#007fff", "#00A6FF", "#00ccff", "#00E6FF", "#00ffff", "#13F2CC", "#26e599", "#56C943", "#66bf26", "#93D226", "#bfe526", "#EFF969", "#ffff7f", "#FFFF5C", "#ffff00", "#FFEC00", "#ffd900", "#FFC500", "#ffb000", "#FF9100", "#ff7200", "#FF3900", "#ff0000", "#E60000", "#cc0000", "#A60016", "#7f002c", "#A61F4D", "#cc3d6e", "#E61FB7", "#ff00ff", "#FF40FF", "#ff7fff", "#ffbfff"]
boundaries_t = np.arange(-46, 48, 2); cmap_t = ListedColormap(colors_t); norm_t = BoundaryNorm(boundaries_t, cmap_t.N)
colors_p = ["#ffffff", "#C2E7FF", "#47BFFF", "#0055ff", "#0000aa", "#32cd32", "#008000", "#ffff00", "#ff9900", "#b30000", "#ff00ff", "#4b0082"]
boundaries_p = [0,0.1,0.5,1,3,5,7,10,15,20,30,40,50]; cmap_p = ListedColormap(colors_p); norm_p = BoundaryNorm(boundaries_p, cmap_p.N, clip=False)
colors_snow = ["#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#e0e0e0", "#c0c0c0", "#a0a0a0", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177", "#49006a"]
boundaries_snow = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300]; cmap_snow = ListedColormap(colors_snow); cmap_snow.set_under('none'); norm_snow = BoundaryNorm(boundaries_snow, cmap_snow.N, clip=False)
colors_rh = ["#32CD32", "#87CEFA", "#0000CD"]; boundaries_rh = [65, 80, 95, 100]; cmap_rh = ListedColormap(colors_rh); cmap_rh.set_under('none'); norm_rh = BoundaryNorm(boundaries_rh, cmap_rh.N, clip=False)
colors_w = ["#ffffff","#e0f8ff","#b3ecff","#80dfff","#00ccff","#0099ff","#0066ff","#0033ff","#33cc33","#33ff33", "#ffff00","#ffcc00","#ff9900","#ff6600","#ff3300","#cc0000","#800000"]
boundaries_w = [0,15,30,45,60,75,90,105,120,135,150,180,210,240,270,300,330]; cmap_w = ListedColormap(colors_w); norm_w = BoundaryNorm(boundaries_w, cmap_w.N)

def clip_lon_lat(data): return data.sel(latitude=slice(nord, sud), longitude=slice(ovest, est))
def setup_map():
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidths=0.4)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.4)
    return fig, ax

def add_title(ax, main_title_bold, valid_dt, lead_str):
    valid_str = valid_dt.strftime("%d/%m/%Y"); valid_str_2 = valid_dt.strftime("%H")
    title = (r"$\bf{" + main_title_bold + r"}$" + "\n" + r"$\bf{ECMWF\ IFS}$ run: " + run_dt.strftime('%d/%m/%Y') + f" {run_hour:02d}z | validità: " + r"$\bf{" + valid_str + r"}$" + " " + r"$\bf{" + valid_str_2 + " " + lead_str + r"}$")
    ax.set_title(title)

# ==================== MAIN LOOP ====================
prev_tp = None
snowpack = None
prev_step_h = 0
ref_steps = ds[850].step.values 

print("🎨 Inizio generazione mappe con Upload Incrementale...")

for idx, step_td in enumerate(ref_steps):
    step_h = int(step_td / np.timedelta64(1, "h"))
    valid_dt = pytz.utc.localize(run_dt + timedelta(hours=step_h)).astimezone(tz_rome)
    lead_str = f"(+{step_h}h)"
    dt_hours = step_h if idx == 0 else step_h - prev_step_h
    prev_step_h = step_h

    # --- DATI ---
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
    except KeyError: continue

    # --- CALCOLI DERIVATI ---
    prec = None
    if "tp" in ds["sfc"].data_vars:
        try:
            tp_curr = ds["sfc"].tp.sel(step=step_td, method="nearest").fillna(0) * 1000
            prec = (tp_curr if prev_tp is None else (tp_curr - prev_tp).clip(min=0))
            prev_tp = tp_curr
            prec = clip_lon_lat(prec)
            
            # Neve
            rh = dewpoint_to_rh(t2m.values, d2m.values)
            tw = calculate_wet_bulb_stull(t2m.values, rh)
            if snowpack is None: snowpack = xr.zeros_like(prec)
            new_snow = prec.copy(); new_snow.values = np.where(tw < 0.5, prec.values, 0.0)
            melt = xr.zeros_like(prec); melt.values = np.maximum(tw - 0.5, 0) * 0.7 * dt_hours
            snowpack = (snowpack + new_snow - melt).clip(min=0)
        except: prec = None

    # --- PLOT, SAVE, UPLOAD ---
    
    # 1. T850
    fname = f"T850_GH500_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(t850.longitude, t850.latitude, t850, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    ax.contour(t850.longitude, t850.latitude, t850, levels=np.arange(-48,49,4), colors='dimgray', linewidths=0.01)
    ax.contour(t850.longitude, t850.latitude, t850, levels=np.arange(-48,49,8), colors='dimgray', linewidths=0.4)
    cs_gh = ax.contour(gh500.longitude, gh500.latitude, gh500, levels=np.arange(460,600,4), colors='black', linewidths=[1.2 if (abs(l-544)%16==0) else 0.6 for l in np.arange(460,600,4)])
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, r"Temperatura\ 850hPa\ -\ Altezza\ di\ Geopotenziale\ 500\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.8, label="Temperatura (°C)", ticks=np.arange(-44,45,4)); cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    upload_single_file(fname) # <--- UPLOAD IMMEDIATO
    generated_files.append({"name": fname, "step": step_h})

    # 2. T500
    fname = f"T500_GH500_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(t500.longitude, t500.latitude, t500, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    ax.contour(t500.longitude, t500.latitude, t500, levels=np.arange(-48, 49, 4), colors='dimgray', linewidths=0.01)
    ax.contour(t500.longitude, t500.latitude, t500, levels=np.arange(-48, 49, 8), colors='dimgray', linewidths=0.4)
    cs_gh = ax.contour(gh500.longitude, gh500.latitude, gh500, levels=np.arange(460,600,4), colors='black', linewidths=[1.2 if (abs(l-544)%16==0) else 0.6 for l in np.arange(460,600,4)])
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, r"Temperatura\ 500hPa\ -\ Altezza\ di\ Geopotenziale\ 500\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.7, label="Temperatura (°C)", ticks=np.arange(-44,45,4)); cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    upload_single_file(fname) # <--- UPLOAD IMMEDIATO
    generated_files.append({"name": fname, "step": step_h})

    # 3. WIND500
    fname = f"WIND500_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(ws500.longitude, ws500.latitude, ws500, levels=boundaries_w, cmap=cmap_w, norm=norm_w, extend='max')
    cs_gh = ax.contour(gh500.longitude, gh500.latitude, gh500, levels=np.arange(460,600,4), colors='black', linewidths=[1.2 if (abs(l-544)%16==0) else 0.6 for l in np.arange(460,600,4)])
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    ax.quiver(u500.longitude[::6], u500.latitude[::6], u500[::6, ::6], v500[::6, ::6], scale=1200, width=0.001, headwidth=3, headlength=4, color='#333333', alpha=0.7)
    add_title(ax, r"Vento\ 500hPa\ -\ Altezza\ di\ Geopotenziale\ 500\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.7, label="Intensità del vento (km/h)"); cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    upload_single_file(fname) # <--- UPLOAD IMMEDIATO
    generated_files.append({"name": fname, "step": step_h})

    # 4. RH700
    fname = f"RH700_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf_rh = ax.contourf(r700.longitude, r700.latitude, r700, levels=boundaries_rh, cmap=cmap_rh, norm=norm_rh, extend='both')
    gh700_levs = np.arange(200, 530, 4)
    cs_gh = ax.contour(gh700.longitude, gh700.latitude, gh700, levels=gh700_levs, colors='black', linewidths=[1.2 if (abs(l - 316) % 20 == 0) else 0.6 for l in gh700_levs])
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    ws700 = np.sqrt(u700**2 + v700**2) * 3.6
    ax.quiver(u700.longitude[::7], v700.latitude[::7], u700[::7, ::7], v700[::7, ::7], ws700[::7, ::7], scale=1000, width=0.0015, headwidth=3, headlength=4, cmap=cmap_w, alpha=0.9)
    add_title(ax, r"Umidità\ Relativa\ 700hPa\ -\ Vento\ 700hPa\ -\ Altezza\ di\ Geopotenziale\ 700hPa", valid_dt, lead_str)
    cax_rh = fig.add_axes([0.17, 0.2, 0.3, 0.02]); cax_wind = fig.add_axes([0.55, 0.2, 0.3, 0.02])
    cbar_rh = fig.colorbar(cf_rh, cax=cax_rh, orientation='horizontal', label="Umidità relativa (%)", ticks=[65, 80, 95, 100]); cbar_rh.ax.tick_params(labelsize=8)
    from matplotlib.cm import ScalarMappable; sm_wind = ScalarMappable(norm=norm_w, cmap=cmap_w); sm_wind.set_array([])
    cbar_wind = fig.colorbar(sm_wind, cax=cax_wind, orientation='horizontal', label="Intensità del vento (km/h)"); cbar_wind.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    upload_single_file(fname) # <--- UPLOAD IMMEDIATO
    generated_files.append({"name": fname, "step": step_h})

    # 5. JET300
    fname = f"JET300_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(ws300.longitude, ws300.latitude, ws300, levels=boundaries_w, cmap=cmap_w, norm=norm_w, extend='max')
    gh300_levs = np.arange(800, 1200, 4)
    cs_gh = ax.contour(gh300.longitude, gh300.latitude, gh300, levels=gh300_levs, colors='black', linewidths=[1.2 if (abs(l - 1000) % 20 == 0) else 0.6 for l in gh300_levs])
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, r"Jet\ Stream\ 300hPa\ -\ Altezza\ di\ Geopotenziale\ 300\ hPa", valid_dt, lead_str)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.7, label="Intensità del vento (km/h)"); cbar.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    upload_single_file(fname) # <--- UPLOAD IMMEDIATO
    generated_files.append({"name": fname, "step": step_h})

    # 6. PREC_MSL
    if idx > 0 and prec is not None:
        fname = f"PREC_MSL_{step_h:03d}.webp"
        fig, ax = setup_map()
        cf = ax.contourf(prec.longitude, prec.latitude, prec, levels=boundaries_p, cmap=cmap_p, norm=norm_p, extend='max')
        msl_levels = np.arange(np.floor(msl.min()/5)*5, np.ceil(msl.max()/5)*5+1, 5)
        cs_msl = ax.contour(msl.longitude, msl.latitude, msl, levels=msl_levels, colors='black', linewidths=[1 if (abs(l-1010)%20==0) else 0.6 for l in msl_levels])
        ax.clabel(cs_msl, fmt='%d', fontsize=5)
        add_title(ax, r"Precipitazione\ Totale\ 3h\ -\ Pressione\ al\ livello\ del\ mare", valid_dt, lead_str)
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.6, label="Precipitazione Totale 3h (mm)", ticks=boundaries_p)
        cbar.ax.set_xticklabels([str(t) if t < 1 else str(int(t)) for t in boundaries_p]); cbar.ax.tick_params(labelsize=8)
        plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
        plt.close()
        upload_single_file(fname) # <--- UPLOAD IMMEDIATO
        generated_files.append({"name": fname, "step": step_h})

    # 7. SNOWPACK
    if idx > 0 and snowpack is not None:
        fname = f"SNOWPACK_MSL_{step_h:03d}.webp"
        fig, ax = setup_map()
        cf = ax.contourf(snowpack.longitude, snowpack.latitude, snowpack, levels=boundaries_snow, cmap=cmap_snow, norm=norm_snow, extend='max')
        msl_levels = np.arange(np.floor(msl.min()/5)*5, np.ceil(msl.max()/5)*5+1, 5)
        cs_msl = ax.contour(msl.longitude, msl.latitude, msl, levels=msl_levels, colors='black', linewidths=[1 if (abs(l-1010)%20==0) else 0.6 for l in msl_levels])
        ax.clabel(cs_msl, fmt='%d', fontsize=5)
        add_title(ax, r"Neve\ fresca\ al\ suolo\ -\ Pressione\ al\ livello\ del\ mare", valid_dt, lead_str)
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.6, label="Neve fresca accumulata da inizio run (cm)", ticks=boundaries_snow)
        cbar.ax.set_xticklabels([str(t) if t < 1 else str(int(t)) for t in boundaries_snow]); cbar.ax.tick_params(labelsize=8)
        plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
        plt.close()
        upload_single_file(fname) # <--- UPLOAD IMMEDIATO
        generated_files.append({"name": fname, "step": step_h})

    # 8. ITALIA
    fname = f"italia_{step_h:03d}.webp"
    def clip_italy(data): return data.sel(latitude=slice(48.9, 33.7), longitude=slice(3, 22))
    t850_it = clip_italy(t850); gh500_it = clip_italy(gh500); prec_it = clip_italy(prec) if prec is not None else None; msl_it = clip_italy(msl)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Pannello 1
    ax = axes[0]
    cf = ax.contourf(t850_it.longitude, t850_it.latitude, t850_it, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    ax.contour(t850_it.longitude, t850_it.latitude, t850_it, levels=np.arange(-48, 49, 4), colors='gray', linewidths=0.6)
    ax.contour(t850_it.longitude, t850_it.latitude, t850_it, levels=np.arange(-48, 49, 2), colors='gray', linewidths=0.01)
    gh_levels_it = np.arange(500, 600, 4)
    cs_gh = ax.contour(gh500_it.longitude, gh500_it.latitude, gh500_it, levels=gh_levels_it, colors='black', linewidths=[1.4 if (abs(l - 544) % 16 == 0) else 0.8 for l in gh_levels_it])
    ax.clabel(cs_gh, fmt='%d', fontsize=8); ax.coastlines(linewidths=0.5); ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.5)
    ax.set_title(r"$\bf{Temperatura\ 850\ hPa - Altezza\ di\ Geopotenziale\ 500\ hPa}$", fontsize=12)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, label=r"Temperatura (°C)", ticks=np.arange(-32, 33, 4)); cbar.ax.tick_params(labelsize=8)

    # Pannello 2
    ax = axes[1]
    if prec_it is not None:
        cf = ax.contourf(prec_it.longitude, prec_it.latitude, prec_it, levels=boundaries_p, cmap=cmap_p, norm=norm_p, extend="max")
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, label=r"Precipitazione Totale (mm)", ticks=boundaries_p)
        cbar.ax.set_xticklabels([str(t) if t < 1 else str(int(t)) for t in boundaries_p]); cbar.ax.tick_params(labelsize=8)
    msl_levels_it = np.arange(np.floor(msl_it.min() / 5) * 5, np.ceil(msl_it.max() / 5) * 5 + 1, 5)
    cs_msl = ax.contour(msl_it.longitude, msl_it.latitude, msl_it, levels=msl_levels_it, colors='black', linewidths=[1.4 if (abs(l - 1010) % 20 == 0) else 1 for l in msl_levels_it])
    ax.clabel(cs_msl, fmt='%d', fontsize=8); ax.coastlines(linewidths=0.5); ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.5)
    ax.set_title(r"$\bf{Precipitazione\ 3h - Pressione\ al\ livello\ del\ mare}$", fontsize=12)

    valid_str = run_dt.strftime("%d/%m/%Y"); valid_str_2 = valid_dt.strftime("%H") + f" (+{step_h}h)"; timestep_date = valid_dt.strftime("%d/%m/%Y")
    title_latex = (r"$\bf{ECMWF\ IFS}$" + f" run: {valid_str} {run_hour:02d}z | validità: {timestep_date} " + r"$\bf{" + valid_str_2 + r"}$")
    fig.suptitle(title_latex, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTDIR, fname), dpi=120, bbox_inches='tight', pil_kwargs={'quality': 65})
    plt.close()
    upload_single_file(fname) # <--- UPLOAD IMMEDIATO
    generated_files.append({"name": fname, "step": step_h})

    print(f"✅ Step +{step_h}h completato e caricato")

# ==================== JSON FINALE ====================
print("💾 Generazione status_ecmwf.json...")
json_data = {"run_date": JSON_RUN_DATE, "base_path": "ECMWF", "files": generated_files}
json_path = os.path.join(OUTDIR, "status_ecmwf.json")
with open(json_path, 'w') as f: json.dump(json_data, f, indent=2)

upload_single_file("status_ecmwf.json") # <--- UPLOAD JSON FINALE
print("✅ JSON Finale Caricato. Elaborazione terminata!")
