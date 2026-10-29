#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import warnings
import json
from datetime import datetime, timedelta, timezone
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Esecuzione senza interfaccia grafica (server/Databricks)
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
CACHE_DIR = os.path.join(OUTDIR, "grib_0p1")  # Cache separata dai vecchi GRIB 0.25
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Dominio delle mappe: Europa/Atlantico (coordinate geografiche)
NORTH, SOUTH, WEST, EAST = 70.0, 28.0, -28.0, 48.0
RESOLUTION = "0p1"  # ECMWF IFS, griglia 0.1 gradi (circa 9 km)
# Per provare un ciclo preciso, impostare ECMWF_RUN=2026102812 (UTC)
RUN_OVERRIDE = os.environ.get("ECMWF_RUN", "").strip()

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
# Il prodotto 0.1 esce circa due ore dopo il prodotto 0.25.
# Le 10:00/22:00 UTC sono soglie conservative, NON una verifica di disponibilita.
now = datetime.now(timezone.utc)
if RUN_OVERRIDE:
    try:
        run_dt = datetime.strptime(RUN_OVERRIDE, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("ECMWF_RUN deve avere il formato YYYYMMDDHH") from exc
    if run_dt.hour not in (0, 12):
        raise ValueError("Sono supportati i run ECMWF 00 e 12 UTC")
elif now.hour < 9:
    yesterday = (now - timedelta(days=1)).date()
    run_dt = datetime(yesterday.year, yesterday.month, yesterday.day, 12)
elif now.hour < 21:
    run_dt = datetime(now.year, now.month, now.day, 0)
else:
    run_dt = datetime(now.year, now.month, now.day, 12)

run_hour = run_dt.hour

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
    "pl_850":     os.path.join(CACHE_DIR, f"pl_850_{DATE_TAG}_{RUN_STR}_{RESOLUTION}.grib2"),
    "pl_700":     os.path.join(CACHE_DIR, f"pl_700_{DATE_TAG}_{RUN_STR}_{RESOLUTION}.grib2"),
    "pl_500":     os.path.join(CACHE_DIR, f"pl_500_{DATE_TAG}_{RUN_STR}_{RESOLUTION}.grib2"),
    "pl_300":     os.path.join(CACHE_DIR, f"pl_300_{DATE_TAG}_{RUN_STR}_{RESOLUTION}.grib2"),
    "sfc_thermo": os.path.join(CACHE_DIR, f"sfc_thermo_{DATE_TAG}_{RUN_STR}_{RESOLUTION}.grib2"),
    "sfc_prec":   os.path.join(CACHE_DIR, f"sfc_prec_{DATE_TAG}_{RUN_STR}_{RESOLUTION}.grib2"),
}

# ==================== CLEAN OLD RUNS ====================
def clean_old_runs(outdir, date_tag, current_run):
    """Elimina solo GRIB 0.1 di cicli precedenti, non immagini e altri dati."""
    for fname in os.listdir(outdir):
        if fname.endswith(".grib2") and (date_tag not in fname or current_run not in fname):
            try:
                os.remove(os.path.join(outdir, fname))
            except OSError as exc:
                print(f"⚠️ Pulizia cache {fname}: {exc}")

clean_old_runs(CACHE_DIR, DATE_TAG, RUN_STR)

# ==================== STEPS & DOWNLOAD ====================
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))
client = Client(source="azure", model="ifs", resol=RESOLUTION)

def check_and_download(key, params, levels=None, levtype="pl"):
    fpath = FILES[key]
    # Non riutilizzare un file nullo o incompleto.
    if os.path.isfile(fpath) and os.path.getsize(fpath) > 1000:
        print(f"✅ In cache: {key} ({os.path.getsize(fpath) / 1e6:.1f} MB)")
        return

    tmp_path = fpath + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"⬇️ Scarico {key} a {RESOLUTION}...", flush=True)
    request = {
        "date": DATE_STR, "time": run_hour, "step": steps, "stream": "oper",
        "type": "fc", "param": params, "levtype": levtype, "target": tmp_path,
    }
    if levels is not None:
        request["levelist"] = levels

    try:
        client.retrieve(**request)
        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 1000:
            raise RuntimeError("File mancante o troppo piccolo")
        os.replace(tmp_path, fpath)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Download ECMWF {key} fallito: {exc}") from exc

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

def crop_europe(dataset):
    """Taglia la griglia in RAM dopo il download globale, gestendo lon 0..360."""
    lats = np.asarray(dataset.latitude.values)
    lons = np.asarray(dataset.longitude.values)
    lons_signed = (lons + 180.0) % 360.0 - 180.0

    # Indici ordinati: lat decrescente, lon crescente da -28 a +48.
    lat_i = np.flatnonzero((lats >= SOUTH - 0.01) & (lats <= NORTH + 0.01))
    lon_i = np.flatnonzero((lons_signed >= WEST - 0.01) & (lons_signed <= EAST + 0.01))
    if len(lat_i) < 2 or len(lon_i) < 2:
        raise ValueError("Griglia GRIB non copre il dominio Europa configurato")
    lat_i = lat_i[np.argsort(lats[lat_i])[::-1]]
    lon_i = lon_i[np.argsort(lons_signed[lon_i])]

    # isel conserva il caricamento differito, senza convertire l'intero run in NetCDF.
    cut = dataset.isel(latitude=lat_i, longitude=lon_i)
    return cut.assign_coords(longitude=("longitude", lons_signed[lon_i]))


def open_grib_subset(path):
    # cache=False evita di conservare in memoria i grandi campi globali una volta letti.
    data = xr.open_dataset(
        path, engine="cfgrib", cache=False,
        backend_kwargs={"indexpath": ""},
    )
    return crop_europe(data)


ds = {}
try:
    ds[850] = open_grib_subset(FILES["pl_850"])
    ds[700] = open_grib_subset(FILES["pl_700"])
    ds[500] = open_grib_subset(FILES["pl_500"])
    ds[300] = open_grib_subset(FILES["pl_300"])
    # TP e termodinamica separati: evitano possibili conflitti GRIB instant/accum.
    ds["thermo"] = open_grib_subset(FILES["sfc_thermo"])
    ds["prec"] = open_grib_subset(FILES["sfc_prec"])

    expected = set(steps)
    for name, dataset in ds.items():
        available = {int(h / np.timedelta64(1, "h")) for h in dataset.step.values}
        # Alcuni cicli non forniscono TP a +0h: il cumulato iniziale e' zero.
        required = expected - {0} if name == "prec" else expected
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"{name}: timestep mancanti (primi 10): {missing[:10]}")
    if "tp" not in ds["prec"].data_vars:
        raise RuntimeError("Variabile TP mancante nel GRIB delle precipitazioni")
except Exception as e:
    for dataset in ds.values():
        dataset.close()
    raise RuntimeError(f"Errore apertura GRIB 0.1: {e}") from e

# ==================== MAP SETUP ====================
nord, sud, ovest, est = NORTH, SOUTH, WEST, EAST

# COLORMAPS (Ridotte per brevità - mantieni le tue complete)
colors_t = ["#ad99ad", "#948094", "#7a667a", "#614D61", "#473347", "#3D1A57", "#330066", "#460073", "#59007f", "#6C00BF", "#7f00ff", "#4040FF", "#007fff", "#00A6FF", "#00ccff", "#00E6FF", "#00ffff", "#13F2CC", "#26e599", "#56C943", "#66bf26", "#93D226", "#bfe526", "#EFF969", "#ffff7f", "#FFFF5C", "#ffff00", "#FFEC00", "#ffd900", "#FFC500", "#ffb000", "#FF9100", "#ff7200", "#FF3900", "#ff0000", "#E60000", "#cc0000", "#A60016", "#7f002c", "#A61F4D", "#cc3d6e", "#E61FB7", "#ff00ff", "#FF40FF", "#ff7fff", "#ffbfff"]
boundaries_t = np.arange(-46, 48, 2); cmap_t = ListedColormap(colors_t); norm_t = BoundaryNorm(boundaries_t, cmap_t.N)
colors_p = ["#ffffff", "#C2E7FF", "#47BFFF", "#0055ff", "#0000aa", "#32cd32", "#008000", "#ffff00", "#ff9900", "#b30000", "#4b0082", "#ff00ff"]
boundaries_p = [0,0.1,0.5,1,3,5,7,10,15,20,30,40,50]; cmap_p = ListedColormap(colors_p); norm_p = BoundaryNorm(boundaries_p, cmap_p.N, clip=False)
colors_snow = ["#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#e0e0e0", "#c0c0c0", "#a0a0a0", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177", "#49006a"]
boundaries_snow = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300]; cmap_snow = ListedColormap(colors_snow); cmap_snow.set_under('none'); norm_snow = BoundaryNorm(boundaries_snow, cmap_snow.N, clip=False)
colors_rh = ["#32CD32", "#87CEFA", "#0000CD"]; boundaries_rh = [65, 80, 95, 100]; cmap_rh = ListedColormap(colors_rh); cmap_rh.set_under('none'); norm_rh = BoundaryNorm(boundaries_rh, cmap_rh.N, clip=False)
colors_w = ["#ffffff","#e0f8ff","#b3ecff","#80dfff","#00ccff","#0099ff","#0066ff","#0033ff","#33cc33","#33ff33", "#ffff00","#ffcc00","#ff9900","#ff6600","#ff3300","#cc0000","#800000"]
boundaries_w = [0,15,30,45,60,75,90,105,120,135,150,180,210,240,270,300,330]; cmap_w = ListedColormap(colors_w); norm_w = BoundaryNorm(boundaries_w, cmap_w.N)

def clip_lon_lat(data):
    # I dataset sono gia ritagliati e hanno longitude in -180..180.
    return data
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
        t850 = clip_lon_lat(ds[850].t.sel(step=step_td) - 273.15)
        gh850 = clip_lon_lat(ds[850].gh.sel(step=step_td) / 10)
        gh500 = clip_lon_lat(ds[500].gh.sel(step=step_td) / 10)
        t500 = clip_lon_lat(ds[500].t.sel(step=step_td) - 273.15)
        u500 = clip_lon_lat(ds[500].u.sel(step=step_td))
        v500 = clip_lon_lat(ds[500].v.sel(step=step_td))
        ws500 = np.sqrt(u500**2 + v500**2) * 3.6
        u700 = clip_lon_lat(ds[700].u.sel(step=step_td))
        v700 = clip_lon_lat(ds[700].v.sel(step=step_td))
        gh700 = clip_lon_lat(ds[700].gh.sel(step=step_td) / 10)
        r700 = clip_lon_lat(ds[700].r.sel(step=step_td)).where(lambda x: x <= 100, 99.99)
        u300 = clip_lon_lat(ds[300].u.sel(step=step_td))
        v300 = clip_lon_lat(ds[300].v.sel(step=step_td))
        ws300 = np.sqrt(u300**2 + v300**2) * 3.6
        gh300 = clip_lon_lat(ds[300].gh.sel(step=step_td) / 10)
        msl = clip_lon_lat(ds["thermo"].msl.sel(step=step_td) / 100)
        t2m = clip_lon_lat(ds["thermo"].t2m.sel(step=step_td) - 273.15)
        d2m = clip_lon_lat(ds["thermo"].d2m.sel(step=step_td) - 273.15)
    except (KeyError, AttributeError) as exc:
        raise RuntimeError(f"Campo meteorologico mancante al timestep +{step_h}h: {exc}") from exc

    # --- CALCOLI DERIVATI ---
    prec = None
    if "tp" in ds["prec"].data_vars:
        try:
            if step_h == 0 and 0 not in {
                int(h / np.timedelta64(1, "h")) for h in ds["prec"].step.values
            }:
                # TP puo' iniziare a +3h; la mappa del passo zero non mostra pioggia.
                prec = None
            else:
                tp_curr = ds["prec"].tp.sel(step=step_td).fillna(0) * 1000
                prec = (tp_curr if prev_tp is None else (tp_curr - prev_tp).clip(min=0))
                prev_tp = tp_curr
                prec = clip_lon_lat(prec)
            
            # Neve
            if prec is not None:
                rh = dewpoint_to_rh(t2m.values, d2m.values)
                tw = calculate_wet_bulb_stull(t2m.values, rh)
                if snowpack is None: snowpack = xr.zeros_like(prec)
                new_snow = prec.copy(); new_snow.values = np.where(tw < 0.5, prec.values, 0.0)
                melt = xr.zeros_like(prec); melt.values = np.maximum(tw - 0.5, 0) * 0.7 * dt_hours
                snowpack = (snowpack + new_snow - melt).clip(min=0)
        except Exception as e:
            raise RuntimeError(f"Errore calcolo precipitazioni al passo +{step_h}h: {e}") from e

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

    # 2. T850gh850
    fname = f"T850_GH850_{step_h:03d}.webp"
    fig, ax = setup_map()
    cf = ax.contourf(t850.longitude, t850.latitude, t850, levels=boundaries_t, cmap=cmap_t, norm=norm_t, extend='both')
    ax.contour(t850.longitude, t850.latitude, t850, levels=np.arange(-48, 49, 4), colors='dimgray', linewidths=0.01)
    ax.contour(t850.longitude, t850.latitude, t850, levels=np.arange(-48, 49, 8), colors='dimgray', linewidths=0.4)
    cs_gh = ax.contour(gh850.longitude, gh850.latitude, gh850, 
                   levels=np.arange(120, 180, 4), 
                   colors='black', 
                   linewidths=[1.2 if (abs(l-144)%16==0) else 0.6 for l in np.arange(120, 180, 4)])
    ax.clabel(cs_gh, fmt='%d', fontsize=5)
    add_title(ax, r"Temperatura\ 850hPa\ -\ Altezza\ di\ Geopotenziale\ 850\ hPa", valid_dt, lead_str)
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
    ax.quiver(u500.longitude[::15], u500.latitude[::15], u500[::15, ::15], v500[::15, ::15], scale=1200, width=0.001, headwidth=3, headlength=4, color='#333333', alpha=0.7)
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
    ax.quiver(u700.longitude[::18], v700.latitude[::18], u700[::18, ::18], v700[::18, ::18], ws700[::18, ::18], scale=1000, width=0.0015, headwidth=3, headlength=4, cmap=cmap_w, alpha=0.9)
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
        add_title(ax, rf"Precipitazione\ Totale\ {dt_hours}h\ -\ Pressione\ al\ livello\ del\ mare", valid_dt, lead_str)
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.01, shrink=0.6, label=f"Precipitazione Totale {dt_hours}h (mm)", ticks=boundaries_p)
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
    ax.set_title(rf"$\bf{{Precipitazione\ {dt_hours}h - Pressione\ al\ livello\ del\ mare}}$", fontsize=12)

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

# Chiudi i backend GRIB dopo la generazione.
for dataset in ds.values():
    dataset.close()
