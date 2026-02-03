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

DOWNLOAD_DIR = "cams_data"
OUTDIR = "mappe_output"

if os.path.exists(DOWNLOAD_DIR):
    shutil.rmtree(DOWNLOAD_DIR)
if os.path.exists(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Area Europa (dataset CAMS: N, S, W, E)[web:16][web:11]
NORD, SUD, OVEST, EST = 54, 31, -10, 31

# --- ADS (nuovo endpoint) ---
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

VAR_CONFIG = {
    "pm2p5": {"tag": "PM25", "title": "Particolato Fine (PM2.5)"},
    "pm10": {"tag": "PM10", "title": "Particolato (PM10)"},
    "no2": {"tag": "NO", "title": "Biossido di Azoto (NO₂)"},
    "o3": {"tag": "O", "title": "Ozono (O₃)"},
}

# ================= FUNZIONI DI SUPPORTO =================

def get_aqi_colormap(pollutant):
    """Restituisce colormap, norm, livelli e unità (µg/m³)."""
    if pollutant == "no2":
        levels = [0, 20, 40, 90, 120, 230, 340, 1000]
        colors = [
            "#009966",  # 0-20
            "#ffde33",  # 20-40
            "#ff9933",  # 40-90
            "#cc0033",  # 90-120
            "#660099",  # 120-230
            "#7e0023",  # 230-340
            "#000000",  # 340-1000
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm, levels, "µg/m³"

    if pollutant == "o3":
        levels = [0, 50, 80, 100, 120, 140, 160, 180, 200, 240, 300]
        colors = [
            "#009966",  # 0-50
            "#33cc33",  # 50-80
            "#ccff33",  # 80-100
            "#ffff00",  # 100-120
            "#ffcc00",  # 120-140
            "#ff6600",  # 140-160
            "#ff0000",  # 160-180
            "#cc0000",  # 180-200
            "#990099",  # 200-240
            "#660066",  # 240-300
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm, levels, "µg/m³"

    if pollutant in ["pm2p5", "pm10"]:
        if pollutant == "pm2p5":
            levels = [0, 5, 10, 15, 20, 25, 35, 50, 75, 100]
        else:
            levels = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        n_intervals = len(levels) - 1
        cmap_base = plt.get_cmap("Spectral_r", n_intervals)
        norm = BoundaryNorm(levels, ncolors=cmap_base.N, clip=True)
        return cmap_base, norm, levels, "µg/m³"

    return plt.cm.viridis, None, None, "µg/m³"

def clip_lon_lat(data):
    """Standardizza nomi lat/lon, ordina e ritaglia area."""
    if "latitude" in data.coords:
        data = data.rename({"latitude": "lat", "longitude": "lon"})
    data = data.sortby(["lat", "lon"])
    lat_min, lat_max = min(NORD, SUD), max(NORD, SUD)
    lon_min, lon_max = min(OVEST, EST), max(OVEST, EST)
    return data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

def setup_map():
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidths=0.6, resolution="10m")
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e0f7fa")
    return fig, ax

def add_title(ax, var_key, valid_dt, run_dt, lead_hours):
    full_name = VAR_CONFIG[var_key]["title"]
    valid_str = valid_dt.strftime("%d/%m/%Y")
    valid_hour = valid_dt.strftime("%H:%M")
    title = (
        f"{full_name}\n"
        f"Run: {run_dt.strftime('%d/%m/%Y %H')}z | "
        f"Validità: {valid_str} {valid_hour} LT (+{lead_hours}h)"
    )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.text(
        0.99,
        0.01,
        "Data: CAMS/Copernicus - Processing: Python",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
    )

def upload_to_r2(file_path, object_name):
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        print("⚠️ Credenziali R2 mancanti. Salto upload.")
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
    if "pm2p5" in v or "2.5um" in v:
        return "pm2p5"
    if "pm10" in v or "10um" in v:
        return "pm10"
    if "no2" in v or "nitrogen_dioxide" in v:
        return "no2"
    if "o3" in v or "ozone" in v or "go3" in v:
        return "o3"
    return None

def pd_to_dt(ts):
    return datetime.datetime.utcfromtimestamp(ts.astype("O") / 1e9)

# ================= MAIN LOOP =================

def run_job():
    print(f"--- Start CAMS Processing: {datetime.datetime.now()} ---")

    if not CDS_KEY:
        print("❌ Errore: CDS_API_KEY non trovata o vuota.")
        sys.exit(1)

    # -------- DOWNLOAD --------
    try:
        client = cdsapi.Client(url=ADS_URL, key=CDS_KEY)
        today = datetime.datetime.now(datetime.timezone.utc).date()
        date_query = f"{today}/{today}"

        file_zip = os.path.join(DOWNLOAD_DIR, "cams.zip")
        file_nc = os.path.join(DOWNLOAD_DIR, "data.nc")

        leadtimes = [str(i) for i in range(0, 97)]

        request = {
            "variable": [
                "nitrogen_dioxide",
                "ozone",
                "particulate_matter_2.5um",
                "particulate_matter_10um",
            ],
            "model": ["ensemble"],
            "level": ["0"],
            "date": [date_query],
            "type": ["forecast"],
            "time": ["00:00"],
            "leadtime_hour": leadtimes,
            "data_format": "netcdf_zip",
            "area": [NORD, OVEST, SUD, EST],
        }

        print(f"⬇️ Richiesta API verso {ADS_URL}...")
        client.retrieve("cams-europe-air-quality-forecasts", request).download(file_zip)

        print("📦 Estrazione ZIP...")
        with zipfile.ZipFile(file_zip, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
            extracted = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".nc")]
            if not extracted:
                raise FileNotFoundError("Nessun file .nc trovato nello zip")
            os.rename(os.path.join(DOWNLOAD_DIR, extracted[0]), file_nc)

    except Exception as e:
        print(f"❌ Errore Critico Download: {e}")
        sys.exit(1)

    # -------- ELABORAZIONE --------
    try:
        print("🎨 Inizio generazione mappe...")
        ds = xr.open_dataset(file_nc)

        # time/run handling: CAMS Europe ha forecast orario 0–96h dal run 00 UTC[web:11]
        if "time" in ds.coords and ds.time.size > 1:
            run_dt = pd_to_dt(ds.time.values[0])
            steps = ds.time.values
            time_iter = True
        else:
            if "time" in ds.coords:
                val_t = ds.time.values if np.ndim(ds.time.values) == 0 else ds.time.values[0]
                run_dt = pd_to_dt(val_t)
            else:
                run_dt = datetime.datetime.now(datetime.timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
            steps = ds.step.values if "step" in ds.coords else ds.leadtime.values
            time_iter = False

        for i, val in enumerate(steps):
            if time_iter:
                valid_dt_utc = pd_to_dt(val)
                diff_hours = (valid_dt_utc - run_dt).total_seconds() / 3600
                lead_hours = int(round(diff_hours))
            else:
                hours_added = int(val.astype("timedelta64[h]").astype(int))
                valid_dt_utc = run_dt + datetime.timedelta(hours=hours_added)
                lead_hours = hours_added

            valid_dt_loc = pytz.utc.localize(valid_dt_utc).astimezone(TZ_ROME)
            timestep_str = f"{lead_hours:02d}"

            print(f"   Processing +{lead_hours}h ...")

            for var_nc_name in ds.data_vars:
                short_name = identify_variable(var_nc_name)
                if not short_name:
                    continue

                # selezione temporale
                if time_iter:
                    da = ds[var_nc_name].sel(time=val)
                else:
                    idx_dict = (
                        {"step": i}
                        if "step" in ds.coords
                        else {"leadtime": i}
                        if "leadtime" in ds.coords
                        else {"time": i}
                    )
                    da = ds[var_nc_name].isel(**idx_dict)

                # riduci dimensioni: tieni solo lat/lon (2D)
                if "level" in da.dims:
                    da = da.sel(level=da.level.min())
                if "model" in da.dims:
                    da = da.mean("model")
                if "ensemble" in da.dims:
                    da = da.mean("ensemble")

                data = clip_lon_lat(da) * 1e9  # kg/m3 -> µg/m3

                # ora data deve essere 2D (lat, lon)
                if data.ndim != 2:
                    print(f"⚠️ Skip {short_name} +{lead_hours}h: data.ndim={data.ndim}")
                    continue

                cmap, norm, levels, unit = get_aqi_colormap(short_name)

                fig, ax = setup_map()
                if levels:
                    cf = ax.contourf(
                        data.lon,
                        data.lat,
                        data.values,
                        levels=levels,
                        cmap=cmap,
                        norm=norm,
                        extend="max",
                    )
                else:
                    cf = ax.contourf(
                        data.lon,
                        data.lat,
                        data.values,
                        cmap=cmap,
                        extend="max",
                    )

                add_title(ax, short_name, valid_dt_loc, run_dt, lead_hours)
                cbar = plt.colorbar(
                    cf, orientation="horizontal", pad=0.02, shrink=0.8, label=unit
                )
                cbar.ax.tick_params(labelsize=8)

                tag = VAR_CONFIG[short_name]["tag"]
                filename = f"{tag}_{timestep_str}.webp"
                filepath = os.path.join(OUTDIR, filename)

                plt.savefig(
                    filepath,
                    dpi=100,
                    bbox_inches="tight",
                    format="webp",
                    pil_kwargs={"quality": 70},
                )
                plt.close(fig)

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
