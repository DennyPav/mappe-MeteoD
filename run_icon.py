#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import matplotlib

# --- 1. CONFIGURAZIONE BACKEND PER GITHUB ACTIONS ---
# Forza backend non interattivo per evitare crash "headless"
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import requests
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
import pytz

# Filtra warning non critici per pulire il log di GitHub
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== CONFIGURAZIONE PATH ====================
# Percorsi relativi alla posizione dello script (funziona ovunque)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.join(BASE_DIR, "icon_data")
OUTDIR = os.path.join(BASE_DIR, "icon_output")
SHP_PATH = os.path.join(BASE_DIR, "Reg01012025_g_WGS84.shp")

os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

print(f"📂 Output directory: {OUTDIR}", flush=True)
print(f"🗺️ Shapefile path: {SHP_PATH}", flush=True)

# PARAMETRI FISICI
DT_HOURS = 1.0          # Timestep modello
MELT_RATE = 0.7         # mm/h fusione neve per °C > 0
SNOW_TW_THRESH = 0.5    # Soglia wet-bulb passaggio pioggia/neve
MAP_EXTENT = (5, 20, 35, 48)

# TIMEZONE (Gestisce automaticamente Ora Solare/Legale)
TZ_ROME = pytz.timezone('Europe/Rome')

# ==================== PALETTE COLORI ====================

# 1. TEMPERATURA 2M
colors_t = [
    "#ad99ad", "#948094", "#7a667a", "#614D61", "#473347", "#3D1A57", "#330066", 
    "#460073", "#59007f", "#6C00BF", "#7f00ff", "#4040FF", "#007fff", "#00A6FF", 
    "#00ccff", "#00E6FF", "#00ffff", "#13F2CC", "#26e599", "#56C943", "#66bf26", 
    "#93D226", "#bfe526", "#EFF969", "#ffff7f", "#FFFF5C", "#ffff00", "#FFEC00", 
    "#ffd900", "#FFC500", "#ffb000", "#FF9100", "#ff7200", "#FF3900", "#ff0000", 
    "#E60000", "#cc0000", "#A60016", "#7f002c", "#A61F4D", "#cc3d6e", "#E61FB7", 
    "#ff00ff", "#FF40FF", "#ff7fff", "#ffbfff"
]
boundaries_t = np.arange(-46, 48, 2)
cmap_t = ListedColormap(colors_t)
norm_t = BoundaryNorm(boundaries_t, cmap_t.N)
ticks_t_lines = [b for b in boundaries_t if b % 4 == 0]

# 2. PRECIPITAZIONE ISTANTANEA
colors_p = ["#ffffff","#bfe7f9","#7ed1f3","#00a6e6","#003f7b","#f4f89f","#e6ed3b",
            "#ffd800","#ff9500","#ff2f00","#b40a00","#840000","#dd007f"]
boundaries_p = [0,0.1,0.5,1,3,5,7,10,15,20,30,40,50]
cmap_p = ListedColormap(colors_p)
cmap_p.set_under("none")
norm_p = BoundaryNorm(boundaries_p, cmap_p.N, clip=False)

# 3. PRECIPITAZIONE CUMULATA
colors_p_cum = [
    "#ffffff", "#8fd3ff", "#00a6ff", "#0055ff", "#0000aa", "#32cd32", "#008000", 
    "#ffff00", "#ffcc00", "#ff9900", "#ff4500", "#ff0000", "#b30000", "#ff1493", 
    "#ff00ff", "#9400d3", "#4b0082", "#dadada", "#909090", "#505050", "#000000"
]
boundaries_p_cum = [0,1,5,10,15,20,30,40,50,75,100,125,150,200,250,300,400,500,650,800,1000]
cmap_p_cum = ListedColormap(colors_p_cum)
cmap_p_cum.set_under("none")
norm_p_cum = BoundaryNorm(boundaries_p_cum, cmap_p_cum.N, clip=False)

# 4A. NEVE ORARIA
colors_snow = ["#ffffff","#a1d99b","#74c476","#41ab5d","#e0e0e0","#c0c0c0","#a0a0a0",
            "#fde0dd","#fa9fb5","#dd3497","#7a0177","#49006a"]
boundaries_snow = [0,0.1,0.5,1,3,5,7,10,15,20,30,40,50]
cmap_snow = ListedColormap(colors_snow)
cmap_snow.set_under('none')
norm_snow = BoundaryNorm(boundaries_snow, cmap_snow.N, clip=False)

# 4B. NEVE CUMULATA
colors_snow_cum = [
    "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", 
    "#e0e0e0", "#c0c0c0", "#a0a0a0",
    "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1",
    "#dd3497", "#ae017e", "#7a0177", "#49006a"
]
boundaries_snow_cum = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300]
cmap_snow_cum = ListedColormap(colors_snow_cum)
cmap_snow_cum.set_under('none')
norm_snow_cum = BoundaryNorm(boundaries_snow_cum, cmap_snow_cum.N, clip=False)

# 6. RH
colors_rh = ["#ff0000", "#ff8c00", "#ffff00", "#32CD32", "#87CEFA", "#0000CD"]
boundaries_rh = [0, 20, 40, 60, 80, 95, 100]
cmap_rh = ListedColormap(colors_rh)
norm_rh = BoundaryNorm(boundaries_rh, cmap_rh.N, clip=False)

# 7. VENTO
colors_w = [
    "#ffffff", "#e0f8ff", "#b3ecff", "#80dfff", "#00ccff", "#0099ff", "#0066ff", "#0033ff", 
    "#33cc33", "#33ff33", "#ffff00", "#ffcc00", "#ff9900", "#ff6600", "#ff3300", "#cc0000", "#800000"
]
boundaries_w = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120]
cmap_w = ListedColormap(colors_w)
norm_w = BoundaryNorm(boundaries_w, cmap_w.N, clip=False)

# RAFFICA
colors_g = colors_w + ["#5a0000", "#3d0000", "#2b002b", "#000000"]
boundaries_g = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250]
cmap_g = ListedColormap(colors_g)
norm_g = BoundaryNorm(boundaries_g, cmap_g.N, clip=False)

# 5. CAPE
colors_cape = [
    "#ffffff", "#0272fc", "#02bafc", "#02fcf2", "#05b2a5", "#02fc35", 
    "#c4fc02", "#fc9302", "#fc1302", "#b2097c", "#ff00ff", "#7f02fc"
]
boundaries_cape = [50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000, 9000]
cmap_cape = ListedColormap(colors_cape)
cmap_cape.set_under("none")
norm_cape = BoundaryNorm(boundaries_cape, cmap_cape.N, clip=False)

# 8. ZERO TERMICO
colors_zt = [
    "#ffffff", "#8221ed", "#5300bb", "#1d09a8", "#0e2dac", "#1660a1", "#2489a5", "#2c9b95", 
    "#17d4ce", "#00d65d", "#3ac303", "#aaed00", "#f8e300", "#efa700", "#e26901", 
    "#d22c00", "#c40111", "#aa0259", "#a70086", "#f3129e", "#ed70aa", "#f6c1d3"
]
boundaries_zt = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 
                 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 6000]
cmap_zt = ListedColormap(colors_zt)
cmap_zt.set_under("none")
norm_zt = BoundaryNorm(boundaries_zt, cmap_zt.N, clip=False)


# ==================== GESTIONE TEMPO RUN ====================
def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 4:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 16:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

run_date, run_hour = get_run_datetime_now_utc()
GRIB_DIR = os.path.join(WORKDIR, f"{run_date}{run_hour}")
os.makedirs(GRIB_DIR, exist_ok=True)
print(f"🔵 Run: {run_date} {run_hour} UTC", flush=True)

run_datetime_obj = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H").replace(tzinfo=timezone.utc)

# ==================== DOWNLOAD ====================
def download_icon(var, filter_str=None):
    base = f"https://meteohub.agenziaitaliameteo.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS/{run_date}{run_hour}/{var}/"
    print(f"🔍 {var}...", end=" ", flush=True)
    try:
        r = requests.get(base, timeout=30)
        if r.status_code != 200:
            print(f"⚠️ HTTP {r.status_code}", flush=True)
            return
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a") if a.get("href", "").endswith(".grib")]
        if filter_str: links = [l for l in links if filter_str in l]
        if not links: print("⚠️ Nessun file", flush=True); return
        
        c = 0
        for l in links:
            local_name = f"{var}_{l}"
            out = os.path.join(GRIB_DIR, local_name)
            if not os.path.exists(out) or os.path.getsize(out) == 0:
                with requests.get(base + l, stream=True, timeout=60) as rr:
                    rr.raise_for_status()
                    with open(out, "wb") as f:
                        for chunk in rr.iter_content(8192): f.write(chunk)
                c += 1
        print(f"✅ {len(links)} file ({c} nuovi)", flush=True)
    except Exception as e: print(f"❌ {e}", flush=True)


dl_list = [("T_2M", None), ("TD_2M", None), ("TOT_PREC", None), ("PMSL", None),
           ("U_10M", None), ("V_10M", None), ("CAPE_ML", None), ("HZEROCL", None),
           ("OMEGA", "700"), ("FI", "500"), ("U", "700"), ("V", "700"), ("VMAX_10M", None)]


print("\n⬇️ Download...", flush=True)
for v, f in dl_list: download_icon(v, f)

# ==================== UTILS CARICAMENTO XARRAY ====================
def list_files(token):
    prefix = f"{token}_"
    flist = [os.path.join(GRIB_DIR, f) for f in os.listdir(GRIB_DIR) 
             if f.startswith(prefix) and f.endswith(".grib")]
    flist.sort()
    return flist


def to_forecast_time(da):
    if "valid_time" in da.coords:
        if "time" in da.dims or "time" in da.coords: da = da.rename({"time": "run_time"})
        return da.swap_dims({"step": "valid_time"}).rename({"valid_time": "time"})
    if "time" in da.dims and "step" in da.dims and da.sizes["time"] == 1:
        valid_times = [da.time.values[0] + s for s in da.step.values.astype("timedelta64[ns]")]
        da = da.isel(time=0, drop=True)
        return da.assign_coords(time=("step", valid_times)).swap_dims({"step": "time"})
    return da

def open_grib_safe(token, short_name_hint, extra_filter=None):
    files = list_files(token)
    if not files: print(f"❌ Manca {token}", flush=True); sys.exit(1)
    filt = extra_filter if extra_filter else {}
    try:
        ds = xr.open_mfdataset(files, engine="cfgrib", combine="by_coords", backend_kwargs={"filter_by_keys": filt, "indexpath": ""})
        if short_name_hint in ds: return ds[short_name_hint]
        for c in ["prmsl", "pmsl"]:
            if c in ds: print(f"⚠️ Uso '{c}' per {token}", flush=True); return ds[c]
        if len(ds.data_vars) == 1: return ds[list(ds.data_vars)[0]]
        print(f"❌ Variabile {short_name_hint} non trovata in {token}", flush=True); sys.exit(1)
    except Exception as e: print(f"❌ Errore {token}: {e}", flush=True); sys.exit(1)

# ==================== CARICAMENTO DATI ====================
print("\n📂 Caricamento e Conversione Unità...", flush=True)
t2m = to_forecast_time(open_grib_safe("T_2M", "2t")) - 273.15
d2m = to_forecast_time(open_grib_safe("TD_2M", "2d")) - 273.15
tp_raw = to_forecast_time(open_grib_safe("TOT_PREC", "tp"))
tp_cum = tp_raw.clip(min=0)
tp = tp_cum.diff(dim="time") 
msl = to_forecast_time(open_grib_safe("PMSL", "msl")) / 100.0
u10 = to_forecast_time(open_grib_safe("U_10M", "10u"))
v10 = to_forecast_time(open_grib_safe("V_10M", "10v"))
vmax_10m = to_forecast_time(open_grib_safe("VMAX_10M", "fg10")) * 3.6 
cape = to_forecast_time(open_grib_safe("CAPE_ML", "cape"))
zt = to_forecast_time(open_grib_safe("HZEROCL", "hzcl"))
om700 = to_forecast_time(open_grib_safe("OMEGA", "w", {"typeOfLevel": "isobaricInhPa"}))
z500 = to_forecast_time(open_grib_safe("FI", "z", {"typeOfLevel": "isobaricInhPa"})) / 9.80665
u700 = to_forecast_time(open_grib_safe("U", "u", {"typeOfLevel": "isobaricInhPa", "level": 700}))
v700 = to_forecast_time(open_grib_safe("V", "v", {"typeOfLevel": "isobaricInhPa", "level": 700}))

# ==================== UTILS MAPPE E PLOT ====================
print("🗺️ Caricamento shapefile...", flush=True)
regions_geom = None
if os.path.exists(SHP_PATH):
    try:
        reg_df = gpd.read_file(SHP_PATH).explode(index_parts=False).to_crs(epsg=4326)
        regions_geom = reg_df.geometry
        # Semplificazione per evitare segfault con Cartopy su GitHub Actions
        regions_geom = regions_geom.simplify(tolerance=0.01, preserve_topology=True)
        print("✅ Shapefile caricato e semplificato!", flush=True)
    except Exception as e:
        print(f"⚠️ Errore caricamento shapefile: {e}", flush=True)
else:
    print(f"⚠️ Shapefile non trovato: {SHP_PATH}", flush=True)

def setup_map():
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    if regions_geom is not None:
        ax.add_geometries(regions_geom, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    else:
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    return fig, ax

def add_mslp(ax, msl_da):
    # Controllo se è una mappa 2D (ha lat/lon)
    if "latitude" not in msl_da.coords or "longitude" not in msl_da.coords:
        if "lat" not in msl_da.coords or "lon" not in msl_da.coords:
            return # Skip if no coords
            
    if "longitude" in msl_da.coords:
        x, y = msl_da.longitude, msl_da.latitude
    elif "lon" in msl_da.coords:
        x, y = msl_da.lon, msl_da.lat
    else:
        return

    mn = np.floor(msl_da.min() / 2) * 2
    mx = np.ceil(msl_da.max() / 2) * 2 + 1
    levels = np.arange(mn, mx, 2)
    lws = [1 if (abs(l - 1000) % 8 == 0) else 0.6 for l in levels]
    cs = ax.contour(x, y, msl_da, 
                    levels=levels, colors="k", linewidths=lws, alpha=0.9)
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True, colors='k')

def save_plot(name): 
    plt.savefig(name, dpi=120, bbox_inches="tight")
    plt.close()

def wetbulb_stull(t, rh):
    rh = rh.clip(min=0, max=100)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + 
            np.arctan(t + rh) - np.arctan(rh - 1.676331) + 
            0.00391838 * rh**1.5 * np.arctan(0.023101 * rh) - 4.686035)

def get_rh(t, td):
    return 100 * np.exp((17.625*td)/(243.04+td)) / np.exp((17.625*t)/(243.04+t))

def round_prec_data(da):
    vals = da.values.copy()
    mask_decimals = (vals >= 0.1) & (vals <= 0.9)
    vals_rounded = np.round(vals)
    vals_rounded[mask_decimals] = np.round(vals[mask_decimals], 1)
    return xr.DataArray(vals_rounded, coords=da.coords, dims=da.dims)

def finalize_plot(fig, ax, cf, run_dt, valid_dt, title_bold, title_normal, cbar_label, explicit_ticks=None):
    kw_args = {}
    if explicit_ticks is not None:
        kw_args['ticks'] = explicit_ticks
        
    cbar = plt.colorbar(cf, orientation="horizontal", pad=0.03, shrink=0.8, aspect=35, **kw_args)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=7, rotation=0)
    
    # --- CONVERSIONE ORARIO LOCALE (con gestione DST automatica) ---
    if run_dt.tzinfo is None: run_dt = run_dt.replace(tzinfo=timezone.utc)
    if isinstance(valid_dt, np.datetime64): 
        valid_dt = pd.to_datetime(valid_dt).replace(tzinfo=timezone.utc)
    elif isinstance(valid_dt, pd.Timestamp):
        if valid_dt.tz is None: valid_dt = valid_dt.replace(tzinfo=timezone.utc)
    
    # Questo converte automaticamente tenendo conto della data (Estate=+2, Inverno=+1)
    run_dt_loc = run_dt.astimezone(TZ_ROME)
    valid_dt_loc = valid_dt.astimezone(TZ_ROME)

    run_str = run_dt.strftime("%d/%m/%Y %H UTC")
    
    if "Giornaliera" in title_normal:
        valid_str_final = valid_dt_loc.strftime(r"%d/%m/%Y")
    else:
        diff_hours = int((valid_dt - run_dt).total_seconds() / 3600)
        day_str = valid_dt_loc.strftime(r"%d/%m/%Y")
        hour_str = valid_dt_loc.strftime(r"%H")
        valid_str_final = fr"{day_str}\ {hour_str}\ (+{diff_hours}h)"

    title_bold_escaped = title_bold.replace(" ", r"\ ")
    full_title = fr"$\bf{{{title_bold_escaped}}}$ {title_normal}"
    subtitle = fr"$\bf{{ICON-2I}}$ Run: {run_str} | Validità: $\bf{{{valid_str_final}}}$"
    
    fig.suptitle(full_title, fontsize=13, y=0.96)
    ax.set_title(subtitle, fontsize=9, loc='center')
    plt.subplots_adjust(top=0.91, bottom=0.05)


# ==================== MAIN LOOP (PLOT ORARI) ====================
print("\n🎨 Generazione Plot Orari...", flush=True)
times = t2m.time.values
prev_tp = None
snowpack = xr.zeros_like(tp.isel(time=0))

for idx, t_val in enumerate(times):
    # NOME FILE BASATO SU STEP (000, 001...) PER SOVRASCRITTURA
    step_str = f"{idx:03d}"
    print(f"   [{idx+1}/{len(times)}] Elaborazione Step {step_str}...", flush=True)
    
    valid_dt_obj = pd.to_datetime(t_val).replace(tzinfo=timezone.utc)
    
    # SELEZIONE ROBUSTA (method='nearest')
    t2 = t2m.sel(time=t_val, method='nearest')
    td = d2m.sel(time=t_val, method='nearest')
    msl_curr = msl.sel(time=t_val, method='nearest')
    
    if "latitude" in msl_curr.coords and "longitude" in msl_curr.coords:
        msl_s = xr.DataArray(gaussian_filter(msl_curr, 4.0), coords=msl_curr.coords, dims=msl_curr.dims)
    else:
        msl_s = msl_curr
    
    try:
        prec = tp.sel(time=t_val, method='nearest').fillna(0)
    except KeyError:
        prec = xr.zeros_like(t2)

    prec_plot = round_prec_data(prec)
    rh = get_rh(t2, td)
    tw = wetbulb_stull(t2, rh)
    new_snow = prec.where(tw < SNOW_TW_THRESH, 0.0)
    melt = (np.maximum(tw - SNOW_TW_THRESH, 0) * MELT_RATE * DT_HOURS)
    snowpack = (snowpack + new_snow - melt).clip(min=0)
    new_snow_plot = round_prec_data(new_snow)
    
    u = u10.sel(time=t_val, method='nearest')
    v = v10.sel(time=t_val, method='nearest')
    ws10 = np.sqrt(u**2 + v**2) * 3.6 
    c_val = cape.sel(time=t_val, method='nearest')
    zt_val = zt.sel(time=t_val, method='nearest')
    u7 = u700.sel(time=t_val, method='nearest')
    v7 = v700.sel(time=t_val, method='nearest')

    # SALVA I PLOT USANDO step_str
    
    # PLOT 1: T2M
    fig, ax = setup_map()
    cf = ax.contourf(t2.longitude, t2.latitude, t2, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend="both")
    add_mslp(ax, msl_s)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Temperatura 2m", "- MSLP", "Temperatura (°C)", explicit_ticks=ticks_t_lines)
    save_plot(os.path.join(OUTDIR, f"T2M_{step_str}.png"))

    # PLOT 2: PREC
    fig, ax = setup_map()
    cf = ax.contourf(prec_plot.longitude, prec_plot.latitude, prec_plot, levels=boundaries_p, cmap=cmap_p, norm=norm_p, extend="max")
    add_mslp(ax, msl_s)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Precipitazione Oraria", "- MSLP", "Precipitazione Totale (mm)", boundaries_p)
    save_plot(os.path.join(OUTDIR, f"PREC_{step_str}.png"))

    # PLOT 3: SNOW
    fig, ax = setup_map()
    cf = ax.contourf(new_snow_plot.longitude, new_snow_plot.latitude, new_snow_plot, levels=boundaries_snow, cmap=cmap_snow, norm=norm_snow, extend="max")
    add_mslp(ax, msl_s)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Neve Oraria", "- MSLP", "Neve (cm)", boundaries_snow)
    save_plot(os.path.join(OUTDIR, f"SNOW_{step_str}.png"))

    # PLOT 3-BIS: SNOWPACK
    snowpack_plot = round_prec_data(snowpack)
    fig, ax = setup_map()
    cf = ax.contourf(snowpack_plot.longitude, snowpack_plot.latitude, snowpack_plot, levels=boundaries_snow_cum, cmap=cmap_snow_cum, norm=norm_snow_cum, extend="max")
    add_mslp(ax, msl_s)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Neve fresca cumulata", "- MSLP", "Neve al suolo (cm)", boundaries_snow_cum)
    save_plot(os.path.join(OUTDIR, f"SNOWPACK_{step_str}.png"))

    # PLOT 4: CAPE
    fig, ax = setup_map()
    cf = ax.contourf(c_val.longitude, c_val.latitude, c_val, levels=boundaries_cape, cmap=cmap_cape, norm=norm_cape, extend="max")
    add_mslp(ax, msl_s)
    ax.quiver(u7.longitude[::12], u7.latitude[::12], u7[::12,::12], v7[::12,::12], scale=1000, width=0.0012, color="#333333", alpha=0.9)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "CAPE", "- MSLP - Vento 700hPa", "CAPE (J/kg)", boundaries_cape)
    save_plot(os.path.join(OUTDIR, f"CAPE_{step_str}.png"))

    # PLOT 5: RH
    fig, ax = setup_map()
    cf = ax.contourf(rh.longitude, rh.latitude, rh, levels=boundaries_rh, cmap=cmap_rh, norm=norm_rh, extend="max")
    add_mslp(ax, msl_s)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Umidità Relativa", "- MSLP", "Umidità relativa (%)", boundaries_rh)
    save_plot(os.path.join(OUTDIR, f"RH_{step_str}.png"))

    # PLOT 6: WIND
    fig, ax = setup_map()
    cf = ax.contourf(ws10.longitude, ws10.latitude, ws10, levels=boundaries_w, cmap=cmap_w, norm=norm_w, extend="max")
    add_mslp(ax, msl_s)
    ax.quiver(u.longitude[::12], u.latitude[::12], u[::12,::12], v[::12,::12], scale=600, width=0.0015, color="k", alpha=0.8)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Vento 10m", "+ MSLP", "Intensità del vento (km/h)", boundaries_w)
    save_plot(os.path.join(OUTDIR, f"WIND_{step_str}.png"))

    # PLOT 8: ZERO
    fig, ax = setup_map()
    cf = ax.contourf(zt_val.longitude, zt_val.latitude, zt_val, levels=boundaries_zt, cmap=cmap_zt, norm=norm_zt, extend="max")
    add_mslp(ax, msl_s)
    finalize_plot(fig, ax, cf, run_datetime_obj, valid_dt_obj, "Altezza Zero Termico", "- MSLP", "Altitudine (m)", boundaries_zt)
    save_plot(os.path.join(OUTDIR, f"ZERO_{step_str}.png"))


# ==================== PLOT FINALI CUMULATI ====================
print("\n📊 Generazione Cumulate Finali...", flush=True)

last_time_val = t2m.time.values[-1]
last_time_dt = pd.to_datetime(last_time_val).replace(tzinfo=timezone.utc)

# 1. CUMULATA TOTALE
tp_total_run = tp.sum(dim="time")
tp_total_plot = round_prec_data(tp_total_run)
fig, ax = setup_map()
cf = ax.contourf(tp_total_plot.longitude, tp_total_plot.latitude, tp_total_plot.squeeze(), levels=boundaries_p_cum, cmap=cmap_p_cum, norm=norm_p_cum, extend="max")
finalize_plot(fig, ax, cf, run_datetime_obj, last_time_dt, "Precipitazione Totale Cumulata", "", "Precipitazione Totale (mm)", boundaries_p_cum)
save_plot(os.path.join(OUTDIR, "PREC_CUM_TOTALE_RUN.png"))

# 2. SNOWPACK TOTALE
snowpack_tot_plot = round_prec_data(snowpack)
fig, ax = setup_map()
cf = ax.contourf(snowpack_tot_plot.longitude, snowpack_tot_plot.latitude, snowpack_tot_plot.squeeze(), levels=boundaries_snow_cum, cmap=cmap_snow_cum, norm=norm_snow_cum, extend="max")
finalize_plot(fig, ax, cf, run_datetime_obj, last_time_dt, "Neve fresca cumulata", "", "Neve al suolo (cm)", boundaries_snow_cum)
save_plot(os.path.join(OUTDIR, "SNOWPACK_TOT.png"))

# 3. GIORNALIERI
daily_prec = tp.resample(time="1D").sum()
for idx, t_day in enumerate(daily_prec.time.values):
    day_idx_str = f"{idx:02d}"
    ts = pd.to_datetime(t_day).replace(tzinfo=timezone.utc)
    prec_day_data = daily_prec.sel(time=t_day)
    if prec_day_data.max() < 0.1: continue

    prec_day_plot = round_prec_data(prec_day_data)
    fig, ax = setup_map()
    cf = ax.contourf(prec_day_plot.longitude, prec_day_plot.latitude, prec_day_plot.squeeze(), levels=boundaries_p_cum, cmap=cmap_p_cum, norm=norm_p_cum, extend="max")
    finalize_plot(fig, ax, cf, run_datetime_obj, ts, "Precipitazione Totale Cumulata", "Giornaliera", "Precipitazione Totale giornaliera (mm)", boundaries_p_cum)
    save_plot(os.path.join(OUTDIR, f"PREC_CUM_DAY_{day_idx_str}.png"))

# 4. TMIN / TMAX / RAFFICA
days = np.unique(t2m.time.dt.floor("D"))
for idx, day in enumerate(days):
    day_idx_str = f"{idx:02d}"
    ts_day = pd.to_datetime(day).replace(tzinfo=timezone.utc)
    t_day = t2m.sel(time=slice(day, day + np.timedelta64(1,'D') - np.timedelta64(1,'ns')))
    
    if t_day.sizes['time'] > 0:
        # Definizione livelli contorni (ogni 4 gradi)
        levs_lines = np.arange(-48, 52, 4)

        # --- TMIN ---
        t_min_val = t_day.min("time")
        fig, ax = setup_map()
        cf = ax.contourf(t_min_val.longitude, t_min_val.latitude, t_min_val, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend="both")
        
        # Salva l'oggetto contour in 'cs' per poterlo etichettare
        cs = ax.contour(t_min_val.longitude, t_min_val.latitude, t_min_val, levels=levs_lines, colors="#555555", linewidths=0.3)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d') # Aggiunge etichette piccole
        
        finalize_plot(fig, ax, cf, run_datetime_obj, ts_day, "Temperatura Minima", "Giornaliera", "Temperatura (°C)", explicit_ticks=ticks_t_lines)
        save_plot(os.path.join(OUTDIR, f"TMIN_DAY_{day_idx_str}.png"))
        
        # --- TMAX ---
        t_max_val = t_day.max("time")
        fig, ax = setup_map()
        cf = ax.contourf(t_max_val.longitude, t_max_val.latitude, t_max_val, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend="both")
        
        # Salva l'oggetto contour in 'cs' per poterlo etichettare
        cs = ax.contour(t_max_val.longitude, t_max_val.latitude, t_max_val, levels=levs_lines, colors="#555555", linewidths=0.3)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d') # Aggiunge etichette piccole
        
        finalize_plot(fig, ax, cf, run_datetime_obj, ts_day, "Temperatura Massima", "Giornaliera", "Temperatura (°C)", explicit_ticks=ticks_t_lines)
        save_plot(os.path.join(OUTDIR, f"TMAX_DAY_{day_idx_str}.png"))

        
        # RAFFICA
        vmax_day = vmax_10m.sel(time=slice(day, day + np.timedelta64(1,'D') - np.timedelta64(1,'ns')))
        if vmax_day.sizes['time'] > 0:
            g_max_val = vmax_day.max("time")
            fig, ax = setup_map()
            cf = ax.contourf(g_max_val.longitude, g_max_val.latitude, g_max_val, levels=boundaries_g, cmap=cmap_g, norm=norm_g, extend="max")
            ax.contour(g_max_val.longitude, g_max_val.latitude, g_max_val, levels=[50, 100], colors="black", linewidths=0.3, alpha=0.5)
            finalize_plot(fig, ax, cf, run_datetime_obj, ts_day, "Raffica di vento massima", "Giornaliera", "Intensità della raffica (km/h)", explicit_ticks=boundaries_g)
            save_plot(os.path.join(OUTDIR, f"GUST_MAX_DAY_{day_idx_str}.png"))

print("\n✅ Finito!", flush=True)
