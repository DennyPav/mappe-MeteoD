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

# ================= CONFIGURAZIONE AMBIENTE =================

# Cartelle temporanee
DOWNLOAD_DIR = "cams_data"
OUTDIR = "mappe_output"

# Pulizia iniziale
if os.path.exists(DOWNLOAD_DIR): shutil.rmtree(DOWNLOAD_DIR)
if os.path.exists(OUTDIR): shutil.rmtree(OUTDIR)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Area Geografica (Italia/Europa allargata)
NORD, SUD, OVEST, EST = 54, 31, -10, 31 

# Recupero Credenziali (nomi esatti forniti da te)
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
BUCKET_NAME = "mappe"
R2_FOLDER = "CAMS"

CDS_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")
CDS_KEY = os.environ.get("CDS_API_KEY")

TZ_ROME = pytz.timezone('Europe/Rome')

# Configurazione Variabili (Mapping nomi interni -> Nomi file/Titoli)
VAR_CONFIG = {
    'pm2p5': {
        'file_tag': 'PM25', 
        'title': 'Particolato Fine (PM2.5)'
    },
    'pm10': {
        'file_tag': 'PM10', 
        'title': 'Particolato (PM10)'
    },
    'no2': {
        'file_tag': 'NO',   # Richiesto "NO" nel nome file
        'title': 'Biossido di Azoto (NO₂)'
    },
    'o3': {
        'file_tag': 'O',    # Richiesto "O" nel nome file
        'title': 'Ozono (O₃)'
    }
}

# ================= FUNZIONI GRAFICHE =================

def get_aqi_colormap(pollutant):
    # Definisce colori e livelli in base al tipo di inquinante
    if pollutant == 'no2': 
        # Scala NO2 µg/m3
        levels = [0, 20, 40, 90, 120, 230, 340, 1000]
        # Verde -> Giallo -> Arancio -> Rosso -> Viola -> Marrone
        cmap = ListedColormap(['#009966', '#ffde33', '#ff9933', '#cc0033', '#660099', '#7e0023'])
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm, levels, "µg/m³"

    elif pollutant == 'o3':
        # Scala Ozono µg/m3
        levels = [0, 50, 80, 100, 120, 140, 160, 180, 200, 240, 300]
        colors = ['#009966', '#33cc33', '#ccff33', '#ffff00', '#ffcc00', '#ff6600', '#ff0000', '#cc0000', '#990099', '#660066']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm, levels, "µg/m³"

    elif pollutant in ['pm2p5', 'pm10']:
        # Scala PM µg/m3
        if pollutant == 'pm2p5':
            levels = [0, 5, 10, 15, 20, 25, 35, 50, 75, 100]
        else:
            levels = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        
        # Spettro invertito (Verde basso, Rosso alto)
        cmap_base = plt.get_cmap('Spectral_r', len(levels)-1)
        norm = BoundaryNorm(levels, ncolors=cmap_base.N, clip=True)
        return cmap_base, norm, levels, "µg/m³"
    
    return plt.cm.viridis, None, None, ""

def clip_lon_lat(data):
    if 'latitude' in data.coords:
        return data.sel(latitude=slice(NORD, SUD), longitude=slice(OVEST, EST))
    elif 'lat' in data.coords:
        return data.sel(lat=slice(NORD, SUD), lon=slice(OVEST, EST))
    return data

def setup_map():
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidths=0.6, resolution='10m')
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax

def add_title(ax, var_key, valid_dt, run_dt, lead_hours):
    # Recupera il nome completo in italiano
    full_name = VAR_CONFIG[var_key]['title']
    
    valid_str = valid_dt.strftime("%d/%m/%Y")
    valid_hour = valid_dt.strftime("%H:%M")
    
    # Titolo formattato
    # Esempio: "Biossido di Azoto (NO2)"
    # Sottotitolo: Run e Validità
    title = (f"{full_name}\n"
             f"Run: {run_dt.strftime('%d/%m/%Y %H')}z | "
             f"Validità: {valid_str} {valid_hour} LT (+{lead_hours}h)")
    
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.text(0.99, 0.01, 'Data: CAMS/Copernicus - Processing: Python', transform=ax.transAxes, 
            fontsize=8, color='gray', ha='right', va='bottom')

# ================= UPLOAD R2 =================

def upload_to_r2(file_path, object_name):
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        print("⚠️ Credenziali R2 mancanti. Salto upload.")
        return

    s3_client = boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )
    try:
        # ExtraArgsContentType serve per visualizzare correttamente i webp nel browser
        s3_client.upload_file(
            file_path, 
            BUCKET_NAME, 
            object_name,
            ExtraArgs={'ContentType': 'image/webp'}
        )
        print(f"✅ Upload OK: {object_name}")
    except Exception as e:
        print(f"❌ Errore upload R2: {e}")

# ================= HELPER DATI =================

def identify_variable(var_name):
    v = var_name.lower()
    if 'pm2p5' in v or '2.5um' in v: return 'pm2p5'
    if 'pm10' in v or '10um' in v: return 'pm10'
    if 'no2' in v or 'nitrogen_dioxide' in v: return 'no2'
    if 'o3' in v or 'ozone' in v or 'go3' in v: return 'o3'
    return None

def pd_timestamp_to_datetime(ts):
    return datetime.datetime.utcfromtimestamp(ts.astype('O')/1e9)

# ================= MAIN =================

def run_job():
    print(f"--- Start CAMS Processing: {datetime.datetime.now()} ---")
    
    if not CDS_KEY:
        print("❌ Errore: CDS_API_KEY mancante.")
        sys.exit(1)

    # 1. Download
    try:
        client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
        today = datetime.datetime.now(datetime.timezone.utc).date()
        date_str = f"{today}/{today}"
        
        file_zip = os.path.join(DOWNLOAD_DIR, "cams.zip")
        file_nc = os.path.join(DOWNLOAD_DIR, "data.nc")
        
        request = {
            "variable": ["nitrogen_dioxide", "ozone", "particulate_matter_2.5um", "particulate_matter_10um"],
            "model": ["ensemble"],
            "level": ["0"],
            "date": [date_str],
            "type": ["forecast"],
            "time": ["00:00"],
            # Scarica fino a +96 ore
            "leadtime_hour": [str(i) for i in range(0, 97)],
            "data_format": "netcdf_zip",
            "area": [NORD, OVEST, SUD, EST]
        }
        
        print("⬇️ Download dati CAMS...")
        client.retrieve("cams-europe-air-quality-forecasts", request).download(file_zip)
        
        import zipfile
        with zipfile.ZipFile(file_zip, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
            extracted = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.nc')]
            if extracted:
                os.rename(os.path.join(DOWNLOAD_DIR, extracted[0]), file_nc)
        
    except Exception as e:
        print(f"❌ Errore download CDS: {e}")
        sys.exit(1)

    # 2. Plotting
    try:
        ds = xr.open_dataset(file_nc)
        run_dt = datetime.datetime.strptime(str(ds.time.values)[0:19], '%Y-%m-%dT%H:%M:%S')
        times = ds.time.values if 'time' in ds.coords else ds.step.values
        
        for i, t in enumerate(times):
            valid_dt_utc = pd_timestamp_to_datetime(t)
            lead_hours = int((valid_dt_utc - run_dt).total_seconds() / 3600)
            valid_dt_loc = pytz.utc.localize(valid_dt_utc).astimezone(TZ_ROME)
            
            # Stringa timestep a due cifre (00, 01, ... 96)
            timestep_str = f"{lead_hours:02d}"

            for var_nc_name in ds.data_vars:
                short_name = identify_variable(var_nc_name)
                if not short_name: continue
                
                # Conversione in µg/m3
                data = clip_lon_lat(ds[var_nc_name].isel(time=i)) * 1e9
                
                cmap, norm, levels, unit_label = get_aqi_colormap(short_name)
                
                fig, ax = setup_map()
                
                # Plot Contourf
                if levels:
                    cf = ax.contourf(data.longitude, data.latitude, data, levels=levels, cmap=cmap, norm=norm, extend='max')
                else:
                    cf = ax.contourf(data.longitude, data.latitude, data, cmap=cmap, extend='max')
                
                # Titolo e Colorbar
                add_title(ax, short_name, valid_dt_loc, run_dt, lead_hours)
                cbar = plt.colorbar(cf, orientation='horizontal', pad=0.02, shrink=0.8, label=unit_label)
                cbar.ax.tick_params(labelsize=8)
                
                # Salvataggio WEBP
                # Nome file richiesto: variabile_timestep.webp (es. PM25_00.webp)
                file_tag = VAR_CONFIG[short_name]['file_tag']
                filename = f"{file_tag}_{timestep_str}.webp"
                filepath = os.path.join(OUTDIR, filename)
                
                plt.savefig(filepath, dpi=100, bbox_inches='tight', format='webp')
                plt.close()
                
                # Upload R2
                # Path richiesto: CAMS/variabile_timestep.webp
                r2_object_path = f"{R2_FOLDER}/{filename}"
                upload_to_r2(filepath, r2_object_path)

        ds.close()
        print("✅ Job completato.")

    except Exception as e:
        print(f"❌ Errore elaborazione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_job()
