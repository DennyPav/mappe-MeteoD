import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
import datetime
import pytz
import os
import boto3
import sys
import shutil
import zipfile
import pandas as pd
import geopandas as gpd  # <--- NUOVO IMPORT

# ================= CONFIGURAZIONE =================

DOWNLOAD_DIR = "cams_data"
OUTDIR = "mappe_output"
SHP_PATH = "Reg01012025_g_WGS84.shp"  # <--- FILE SHAPEFILE

if os.path.exists(DOWNLOAD_DIR):
    shutil.rmtree(DOWNLOAD_DIR)
if os.path.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Area Europa Full Domain
NORD, SUD, OVEST, EST = 56, 32, 0, 24

# --- ADS ---
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"

raw_key = os.environ.get("CDS_API_KEY", "")
if ":" in raw_key:
    CDS_KEY = raw_key.split(":", 1)[1]
else:
    CDS_KEY = raw_key

# --- R2 ---
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
BUCKET_NAME = "mappe"
R2_FOLDER = "CAMS"

TZ_ROME = pytz.timezone("Europe/Rome")
TZ_UTC = datetime.timezone.utc

VAR_CONFIG = {
    "pm2p5": {"tag": "PM25", "title": "Particolato Fine PM2.5"},
    "pm10": {"tag": "PM10", "title": "Particolato PM10"},
    "no2": {"tag": "NO", "title": r"Biossido di Azoto NO$_2$"},
    "o3": {"tag": "O", "title": r"Ozono O$_3$"},
}

# ================= FUNZIONI DI SUPPORTO =================

def get_aqi_colormap(pollutant):
    unit_label = "Concentrazione (µg/m³)"
    
    if pollutant == "no2":
        levels = [0, 20, 40, 90, 120, 230, 340, 1000]
        colors = ["#009966", "#ffde33", "#ff9933", "#cc0033", "#660099", "#7e0023", "#000000"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm, levels, unit_label

    if pollutant == "o3":
        levels = [0, 50, 80, 100, 120, 140, 160, 180, 200, 240, 300]
        colors = ["#009966", "#33cc33", "#ccff33", "#ffff00", "#ffcc00", "#ff6600", "#ff0000", "#cc0000", "#990099", "#660066"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm, levels, unit_label

    if pollutant in ["pm2p5", "pm10"]:
        if pollutant == "pm2p5":
            levels = [0, 5, 10, 15, 20, 25, 35, 50, 75, 100]
        else:
            levels = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        n_intervals = len(levels) - 1
        cmap_base = plt.get_cmap("Spectral_r", n_intervals)
        norm = BoundaryNorm(levels, ncolors=cmap_base.N, clip=True)
        return cmap_base, norm, levels, unit_label

    return plt.cm.viridis, None, None, unit_label

def clip_lon_lat(data):
    if "latitude" in data.coords:
        data = data.rename({"latitude": "lat", "longitude": "lon"})
    data = data.sortby(["lat", "lon"])
    lat_min, lat_max = min(NORD, SUD), max(NORD, SUD)
    lon_min, lon_max = min(OVEST, EST), max(OVEST, EST)
    return data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

def setup_map(regions_geom=None):
    """
    Configura la mappa.
    regions_geom: Geometrie (Series o List) delle regioni da disegnare.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([OVEST, EST, SUD, NORD], crs=ccrs.PlateCarree())
    
    # 1. Aggiungi Ocean/Land per sfondo pulito
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e0f7fa")

    # 2. Aggiungi Regioni (Sottili)
    # Vengono disegnate PRIMA dei confini nazionali in modo che se si sovrappongono, 
    # quelli nazionali (più spessi) prevalgano visivamente, oppure sopra Land ma sotto Borders.
    if regions_geom is not None:
        ax.add_geometries(
            regions_geom, 
            crs=ccrs.PlateCarree(), 
            facecolor='none', 
            edgecolor='black', 
            linewidth=0.5
        )

    # 3. Aggiungi Coste e Confini Nazionali (Più spessi)
    ax.coastlines(linewidths=1.0, resolution="10m", color="black")
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.0)

    return fig, ax

def add_title(ax, var_key, valid_dt_loc, run_dt_utc, lead_hours):
    full_name = VAR_CONFIG[var_key]["title"]
    
    # Run Date (UTC)
    run_date_str = run_dt_utc.strftime("%d/%m/%Y")
    
    # Validity Date (Local Rome Time)
    valid_str = valid_dt_loc.strftime('%d/%m/%Y %H:%M')
    
    # Titolo Variabile
    ax.text(0.5, 1.06, full_name, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Sottotitolo Dati
    subtitle = f"CAMS Run: {run_date_str} 00z  |  Validità: {valid_str} (+{lead_hours}h)"
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=11)
    

def upload_to_r2(file_path, object_name):
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        return
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )
    try:
        s3_client.upload_file(
            file_path,
            BUCKET_NAME,
            object_name,
            ExtraArgs={"ContentType": "image/webp"},
        )
        print(f"✅ Upload R2 OK: {object_name}")
    except Exception as e:
        print(f"❌ Errore upload R2: {e}")

def identify_variable(var_name):
    v = var_name.lower()
    if "pm2p5" in v or "2.5um" in v: return "pm2p5"
    if "pm10" in v or "10um" in v: return "pm10"
    if "no2" in v or "nitrogen_dioxide" in v: return "no2"
    if "o3" in v or "ozone" in v or "go3" in v: return "o3"
    return None

# ================= MAIN LOOP =================

def run_job():
    print(f"--- Start CAMS Processing: {datetime.datetime.now()} ---")

    # --- CARICAMENTO SHAPEFILE REGIONI (Tuo codice) ---
    print("🗺️ Caricamento shapefile...", flush=True)
    regions_geom = None
    if os.path.exists(SHP_PATH):
        try:
            reg_df = gpd.read_file(SHP_PATH).explode(index_parts=False).to_crs(epsg=4326)
            # Semplificazione per rendere il plotting più veloce e leggero
            regions_geom = reg_df.geometry.simplify(tolerance=0.01, preserve_topology=True)
            print("✅ Shapefile OK!", flush=True)
        except Exception as e:
            print(f"⚠️ Errore shapefile: {e}", flush=True)
    else:
        print(f"⚠️ Shapefile non trovato: {SHP_PATH}", flush=True)
    # --------------------------------------------------

    if not CDS_KEY:
        print("❌ CDS_API_KEY non trovata.")
        sys.exit(1)

    try:
        client = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
        
        # 1. Definizione Data Run: OGGI 00:00 UTC
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        today_date = now_utc.date()
        
        run_dt_utc = datetime.datetime(
            today_date.year, today_date.month, today_date.day, 
            0, 0, 0, tzinfo=datetime.timezone.utc
        )
        
        date_query = f"{today_date}/{today_date}"
        print(f"📅 Run Reference Date (UTC): {run_dt_utc}")

        file_zip = os.path.join(DOWNLOAD_DIR, "cams.zip")
        file_nc = os.path.join(DOWNLOAD_DIR, "data.nc")
        
        leadtimes = [str(i) for i in range(0, 97)]

        request = {
            "variable": ["nitrogen_dioxide", "ozone", "particulate_matter_2.5um", "particulate_matter_10um"],
            "model": ["ensemble"],
            "level": ["0"],
            "date": [date_query],
            "type": ["forecast"],
            "time": ["00:00"],
            "leadtime_hour": leadtimes,
            "data_format": "netcdf_zip",
            "area": [NORD, OVEST, SUD, EST],
        }

        print(f"⬇️ Download {ADS_URL}...")
        client.retrieve("cams-europe-air-quality-forecasts", request).download(file_zip)

        with zipfile.ZipFile(file_zip, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
            extracted = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".nc")]
            os.rename(os.path.join(DOWNLOAD_DIR, extracted[0]), file_nc)

        ds = xr.open_dataset(file_nc)

        if "leadtime" in ds.coords:
            time_dim = "leadtime"
            steps_values = ds.leadtime.values
        elif "step" in ds.coords:
            time_dim = "step"
            steps_values = ds.step.values
        elif "time" in ds.coords:
            time_dim = "time"
            steps_values = ds.time.values
        else:
            print("❌ Dimensione temporale mancante")
            sys.exit(1)

        # Loop Temporale
        for i, val in enumerate(steps_values):
            
            if np.issubdtype(val.dtype, np.timedelta64):
                hours_added = int(val / np.timedelta64(1, 'h'))
            elif np.issubdtype(val.dtype, np.datetime64):
                hours_added = i 
            else:
                hours_added = i

            valid_dt_utc = run_dt_utc + datetime.timedelta(hours=hours_added)
            valid_dt_loc = valid_dt_utc.astimezone(TZ_ROME)
            
            timestep_str = f"{hours_added:02d}"
            
            print(f"   Processing +{hours_added}h -> Valid (IT): {valid_dt_loc}")

            for var_nc_name in ds.data_vars:
                short_name = identify_variable(var_nc_name)
                if not short_name: continue

                da = ds[var_nc_name].isel({time_dim: i})

                if "level" in da.dims: da = da.sel(level=da.level.min())
                if "model" in da.dims: da = da.mean("model")
                if "ensemble" in da.dims: da = da.mean("ensemble")

                data = clip_lon_lat(da)
                if data.ndim != 2: continue

                cmap, norm, levels, unit = get_aqi_colormap(short_name)
                
                # Setup mappa PASSANDO LE REGIONI
                fig, ax = setup_map(regions_geom)
                
                if levels:
                    cf = ax.contourf(data.lon, data.lat, data.values, levels=levels, cmap=cmap, norm=norm, extend="max")
                else:
                    cf = ax.contourf(data.lon, data.lat, data.values, cmap=cmap, extend="max")

                add_title(ax, short_name, valid_dt_loc, run_dt_utc, hours_added)
                
                cbar = plt.colorbar(cf, orientation="horizontal", pad=0.02, shrink=0.7, label=unit)
                cbar.ax.tick_params(labelsize=8)

                tag = VAR_CONFIG[short_name]["tag"]
                filename = f"{tag}_{timestep_str}.webp"
                filepath = os.path.join(OUTDIR, filename)

                plt.savefig(filepath, dpi=100, bbox_inches="tight", format="webp", pil_kwargs={"quality": 70})
                plt.close(fig)
                upload_to_r2(filepath, f"{R2_FOLDER}/{filename}")

                ds.close()
        
        # --- GENERAZIONE STATUS.JSON ---
        import json
        
        # Formattiamo la data del run come stringa (es. 202604150000)
        run_date_str = run_dt_utc.strftime("%Y%m%d%H%M")
        
        status_files = []
        # Registriamo tutti i timestep (da 0 a 96) che sono stati processati
        for lead in leadtimes:
            # Salviamo un riferimento per ogni step temporale. 
            # Non serve elencare tutte le singole variabili, basta lo step per generare l'array in Kotlin
            step_int = int(lead)
            timestep_str = f"{step_int:02d}"
            # Usiamo un prefisso generico, l'app sa già come comporre il nome finale combinandolo con il prefisso della variabile
            status_files.append({
                "name": f"{timestep_str}.webp",
                "step": step_int
            })
            
        status_data = {
            "rundate": run_date_str,
            "files": status_files
        }
        
        status_json_path = os.path.join(OUTDIR, "statuscams.json")
        with open(status_json_path, "w") as f:
            json.dump(status_data, f, indent=4)
            
        # Carica il JSON su R2 nella root CAMS
        upload_to_r2(status_json_path, f"{R2_FOLDER}/statuscams.json")
        
        print("✅ Status JSON generato e caricato.")
        # -------------------------------
        
        print("✅ Job completato.")

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_job()
