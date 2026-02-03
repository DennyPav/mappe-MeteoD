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

# ================= CONFIGURAZIONE =================

# Cartelle di lavoro
DOWNLOAD_DIR = "cams_data"
OUTDIR = "mappe_output"

# Pulizia preventiva
if os.path.exists(DOWNLOAD_DIR): shutil.rmtree(DOWNLOAD_DIR)
if os.path.exists(OUTDIR): shutil.rmtree(OUTDIR)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Coordinate Ritaglio Mappa (Italia/Europa)
# Nord, Sud, Ovest, Est
NORD, SUD, OVEST, EST = 54, 31, -10, 31 

# --- CREDENZIALI & API ---

# 1. Cloudflare R2 (Bucket Mappe)
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
BUCKET_NAME = "mappe"
R2_FOLDER = "CAMS"

# 2. Copernicus ADS (Atmosphere Data Store)
# IMPORTANTE: Forziamo l'URL di ADS per evitare l'errore 404 del CDS
ADS_URL = "https://ads.atmosphere.copernicus.eu/api/v2"
CDS_KEY = os.environ.get("CDS_API_KEY")

# Timezone per i titoli
TZ_ROME = pytz.timezone('Europe/Rome')

# Configurazione Nomi e Titoli
VAR_CONFIG = {
    'pm2p5': { 'tag': 'PM25', 'title': 'Particolato Fine (PM2.5)' },
    'pm10':  { 'tag': 'PM10', 'title': 'Particolato (PM10)' },
    'no2':   { 'tag': 'NO',   'title': 'Biossido di Azoto (NO₂)' }, # Tag richiesto: NO
    'o3':    { 'tag': 'O',    'title': 'Ozono (O₃)' }             # Tag richiesto: O
}

# ================= FUNZIONI DI SUPPORTO =================

def get_aqi_colormap(pollutant):
    """Restituisce la colormap e i livelli per ogni inquinante"""
    
    if pollutant == 'no2': 
        # Scala NO2 µg/m3 (Verde -> Giallo -> Rosso -> Viola)
        levels = [0, 20, 40, 90, 120, 230, 340, 1000]
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
        # Scala PM µg/m3 (Invertita: Verde bassi valori, Rosso alti)
        if pollutant == 'pm2p5':
            levels = [0, 5, 10, 15, 20, 25, 35, 50, 75, 100]
        else:
            levels = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        
        cmap_base = plt.get_cmap('Spectral_r', len(levels)-1)
        norm = BoundaryNorm(levels, ncolors=cmap_base.N, clip=True)
        return cmap_base, norm, levels, "µg/m³"
    
    return plt.cm.viridis, None, None, ""

def clip_lon_lat(data):
    """Ritaglia l'array sulla zona di interesse"""
    if 'latitude' in data.coords:
        return data.sel(latitude=slice(NORD, SUD), longitude=slice(OVEST, EST))
    elif 'lat' in data.coords:
        return data.sel(lat=slice(NORD, SUD), lon=slice(OVEST, EST))
    return data

def setup_map():
    """Crea la base cartografica"""
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidths=0.6, resolution='10m')
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidths=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    return fig, ax

def add_title(ax, var_key, valid_dt, run_dt, lead_hours):
    """Aggiunge titoli formattati in italiano"""
    full_name = VAR_CONFIG[var_key]['title']
    valid_str = valid_dt.strftime("%d/%m/%Y")
    valid_hour = valid_dt.strftime("%H:%M")
    
    title = (f"{full_name}\n"
             f"Run: {run_dt.strftime('%d/%m/%Y %H')}z | "
             f"Validità: {valid_str} {valid_hour} LT (+{lead_hours}h)")
    
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.text(0.99, 0.01, 'Data: CAMS/Copernicus - Processing: Python', 
            transform=ax.transAxes, fontsize=8, color='gray', ha='right', va='bottom')

def upload_to_r2(file_path, object_name):
    """Carica su Cloudflare R2"""
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        print("⚠️ Credenziali R2 mancanti. Salto upload.")
        return

    s3_client = boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )
    try:
        s3_client.upload_file(
            file_path, 
            BUCKET_NAME, 
            object_name,
            ExtraArgs={'ContentType': 'image/webp'} # Importante per visualizzazione web
        )
        print(f"✅ Upload R2 OK: {object_name}")
    except Exception as e:
        print(f"❌ Errore upload R2: {e}")

def identify_variable(var_name):
    """Mappa i nomi del NetCDF nei nostri codici interni"""
    v = var_name.lower()
    if 'pm2p5' in v or '2.5um' in v: return 'pm2p5'
    if 'pm10' in v or '10um' in v: return 'pm10'
    if 'no2' in v or 'nitrogen_dioxide' in v: return 'no2'
    if 'o3' in v or 'ozone' in v or 'go3' in v: return 'o3'
    return None

def pd_to_dt(ts):
    """Converte Timestamp numpy/pandas in datetime standard"""
    return datetime.datetime.utcfromtimestamp(ts.astype('O')/1e9)

# ================= MAIN LOOP =================

def run_job():
    print(f"--- Start CAMS Processing: {datetime.datetime.now()} ---")
    
    if not CDS_KEY:
        print("❌ Errore: CDS_API_KEY non trovata nelle variabili d'ambiente.")
        sys.exit(1)

    # 1. DOWNLOAD
    try:
        # Istanziamo il client forzando l'URL ADS
        client = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
        
        today = datetime.datetime.now(datetime.timezone.utc).date()
        date_query = f"{today}/{today}"
        
        file_zip = os.path.join(DOWNLOAD_DIR, "cams.zip")
        file_nc = os.path.join(DOWNLOAD_DIR, "data.nc")
        
        # Lista leadtimes da 0 a 96
        leadtimes = [str(i) for i in range(0, 97)]

        request = {
            "variable": [
                "nitrogen_dioxide",
                "ozone",
                "particulate_matter_2.5um",
                "particulate_matter_10um"
            ],
            "model": ["ensemble"],
            "level": ["0"],
            "date": [date_query],
            "type": ["forecast"],
            "time": ["00:00"],
            "leadtime_hour": leadtimes,
            "data_format": "netcdf_zip",
            # Aggiungo area per ridurre il download ed evitare memory leak
            "area": [NORD, OVEST, SUD, EST] 
        }
        
        print(f"⬇️ Richiesta API verso {ADS_URL}...")
        client.retrieve("cams-europe-air-quality-forecasts", request).download(file_zip)
        
        print("📦 Estrazione ZIP...")
        with zipfile.ZipFile(file_zip, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
            extracted = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.nc')]
            if extracted:
                os.rename(os.path.join(DOWNLOAD_DIR, extracted[0]), file_nc)
            else:
                raise FileNotFoundError("Nessun file .nc trovato nello zip CAMS")
        
    except Exception as e:
        print(f"❌ Errore Critico Download: {e}")
        # Info debug utile se fallisce ancora
        print("Suggerimento: Verifica di aver accettato la licenza su ads.atmosphere.copernicus.eu")
        sys.exit(1)

    # 2. ELABORAZIONE GRAFICA
    try:
        print("🎨 Inizio generazione mappe...")
        ds = xr.open_dataset(file_nc)
        
        # Determina tempo di run
        if 'time' in ds.coords and ds.time.size == 1:
             run_dt = pd_to_dt(ds.time.values[0])
             # Se time è fisso, usiamo 'step' o 'leadtime' per iterare
             steps = ds.step.values if 'step' in ds.coords else ds.leadtime.values
             time_iter = False
        else:
             # Struttura classica: time contiene tutti gli step futuri
             run_dt = pd_to_dt(ds.time.values[0])
             steps = ds.time.values
             time_iter = True
        
        # Loop temporale (0h -> 96h)
        for i, val in enumerate(steps):
            
            # Calcolo date
            if time_iter:
                valid_dt_utc = pd_to_dt(val)
                diff_hours = (valid_dt_utc - run_dt).total_seconds() / 3600
                lead_hours = int(round(diff_hours))
            else:
                # val è un timedelta (nanosecondi)
                hours_added = int(val.astype('timedelta64[h]').astype(int))
                valid_dt_utc = run_dt + datetime.timedelta(hours=hours_added)
                lead_hours = hours_added

            valid_dt_loc = pytz.utc.localize(valid_dt_utc).astimezone(TZ_ROME)
            timestep_str = f"{lead_hours:02d}" # es. "05", "12"

            print(f"   Processing +{lead_hours}h ...")

            # Loop variabili
            for var_nc_name in ds.data_vars:
                short_name = identify_variable(var_nc_name)
                if not short_name: continue # Salta variabili ausiliarie
                
                # Slicing e Conversione (kg/m3 -> µg/m3)
                if time_iter:
                    data_slice = ds[var_nc_name].sel(time=val)
                else:
                    data_slice = ds[var_nc_name].isel(step=i) if 'step' in ds.coords else ds[var_nc_name].isel(time=0)

                data = clip_lon_lat(data_slice) * 1e9
                
                # Configurazione Stile
                cmap, norm, levels, unit = get_aqi_colormap(short_name)
                
                # Creazione Plot
                fig, ax = setup_map()
                
                if levels:
                    cf = ax.contourf(data.longitude, data.latitude, data, 
                                     levels=levels, cmap=cmap, norm=norm, extend='max')
                else:
                    cf = ax.contourf(data.longitude, data.latitude, data, 
                                     cmap=cmap, extend='max')
                
                # Decorazioni
                add_title(ax, short_name, valid_dt_loc, run_dt, lead_hours)
                cbar = plt.colorbar(cf, orientation='horizontal', pad=0.02, shrink=0.8, label=unit)
                cbar.ax.tick_params(labelsize=8)
                
                # Salvataggio e Upload
                tag = VAR_CONFIG[short_name]['tag']
                filename = f"{tag}_{timestep_str}.webp"
                filepath = os.path.join(OUTDIR, filename)
                
                plt.savefig(filepath, dpi=100, bbox_inches='tight', format='webp')
                plt.close(fig) # Importante: chiudere la figura per liberare memoria
                
                # Upload: CAMS/PM25_00.webp
                upload_to_r2(filepath, f"{R2_FOLDER}/{filename}")

        ds.close()
        print("✅ Job completato con successo.")

    except Exception as e:
        print(f"❌ Errore elaborazione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_job()
