# ============================================================
# IMPORT
# ============================================================
import os
from datetime import datetime
import calendar
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cdsapi

# ============================================================
# REGOLA DEL 6 DEL MESE
# ============================================================
def get_target_month(now=None):
    if now is None:
        now = datetime.utcnow()
    delta = 1 if now.day >= 6 else 2
    year = now.year
    month = now.month - delta
    if month <= 0:
        month += 12
        year -= 1
    mesi_italiani = [
        "Gennaio","Febbraio","Marzo","Aprile","Maggio","Giugno",
        "Luglio","Agosto","Settembre","Ottobre","Novembre","Dicembre"
    ]
    return year, month, mesi_italiani[month-1]

year, month, mese = get_target_month()
year_str = str(year)
month_str = f"{month:02d}"

# ============================================================
# DOMINIO
# ============================================================
WEST, EAST = 6.0, 19.0
SOUTH, NORTH = 35.0, 48.0

# ============================================================
# PERCORSI - Adattati per GitHub Actions
# ============================================================
# Directory di lavoro GitHub Actions
base_dir = os.path.join(os.getcwd(), "data")
os.makedirs(base_dir, exist_ok=True)

nc_file_obs  = os.path.join(base_dir, f"Italia_{year_str}_{month_str}.nc")
# Italia_climatologia.nc deve essere nella root del repository
nc_file_clim = os.path.join(os.getcwd(), "Italia_climatologia.nc")
out_dir = os.path.join(base_dir, "plot_mensili_era5_only", f"{year_str}_{month_str}")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# DOWNLOAD ERA5-LAND (se necessario)
# ============================================================
def download_era5_land(year, month, out_nc):
    # Le credenziali vengono lette automaticamente da environment variables
    # impostate tramite GitHub Secrets
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["2m_temperature", "total_precipitation"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [NORTH, WEST, SOUTH, EAST],
        },
        out_nc
    )

if not os.path.exists(nc_file_obs):
    print(f"Download ERA5-Land: {mese} {year}")
    download_era5_land(year, month, nc_file_obs)
else:
    print("ERA5-Land già presente")

# ============================================================
# CARICAMENTO DATI
# ============================================================
ds_obs  = xr.open_dataset(nc_file_obs)
ds_clim = xr.open_dataset(nc_file_clim)

# ============================================================
# ERA5 E CLIMATOLOGIA
# ============================================================
num_days = calendar.monthrange(year, month)[1]

time_dim = 'valid_time' if 'valid_time' in ds_obs.dims else 'time'
time_dim_clim = 'valid_time' if 'valid_time' in ds_clim.dims else 'time'

temp_obs = ds_obs["t2m"].isel({time_dim: 0}) - 273.15
prec_obs = ds_obs["tp"].isel({time_dim: 0}) * 1000 * num_days

month_index = month - 1
all_idx = np.arange(month_index, ds_clim["t2m"].sizes[time_dim_clim], 12)

temp_clim = ds_clim["t2m"].isel({time_dim_clim: all_idx}).mean(time_dim_clim) - 273.15
prec_clim = ds_clim["tp"].isel({time_dim_clim: all_idx}).mean(time_dim_clim) * 1000 * num_days

temp_anom = temp_obs - temp_clim
prec_anom = prec_obs - prec_clim

# ============================================================
# COLORMAP PERSONALIZZATE
# ============================================================

# --- 1. TEMPERATURA 2M (ASSOLUTA) ---
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

# --- 2. PRECIPITAZIONE (ASSOLUTA) ---
boundaries_prec = np.arange(0, 295, 5)  # Include il limite superiore
colors_prec = [
    "#ffffff", "#eaf7fd", "#d5effb", "#bfe7f9", "#aae0f7", "#94d8f5", # Bianco -> Celestino chiaro
    "#7ed1f3", "#68caf1", "#52c3ef", "#3dbbec", "#27b4ea", "#12ade8", # Celestino chiaro -> Celestino medio
    "#00a6e6", "#0095d4", "#0084c3", "#0072b1", "#00619f", "#00508d", # Celestino medio -> Blu
    "#003f7b", "#002e69", "#001d57", "#f7fbb3", "#f4f89f", "#f1f68b", # Blu -> Giallo chiaro
    "#eff377", "#ecf163", "#e9ef4f", "#e6ed3b", "#e3eb27", "#e0e913", # Giallo chiaro -> Giallo intenso
    "#ffd800", "#ffc700", "#ffb700", "#ffa600", "#ff9500", "#ff8400", # Giallo intenso -> Arancione
    "#ff7300", "#ff6200", "#ff5100", "#ff4000", "#ff2f00", "#ff1e00", # Arancione -> Rosso chiaro
    "#e51700", "#cc1000", "#b40a00", "#9c0400", "#840000", "#9a0038", # Rosso chiaro -> Rosso scuro
    "#b00050", "#c60068", "#dd007f", "#f30096", "#ff19ac", "#ff33c2", # Rosso scuro -> Viola rosato
    "#ff4dd8", "#ff66ef", "#ff80f5", "#ff99fb", "#ffb3ff", "#ffe5ff", # Viola rosato -> Rosa chiaro
]

# Creare la colormap personalizzata
cmap_p_cum = mcolors.ListedColormap(colors_prec)
norm_p_cum = mcolors.BoundaryNorm(boundaries_prec, cmap_p_cum.N, extend='both')

# --- 3. ANOMALIA TEMPERATURA ---
boundaries_anom_temp = np.arange(-4.3, 4.5, 0.3)
colors_anom_temp = [
    "#002473","#00287f","#113b8c","#234d99","#3560a6","#4772b3","#5984c0",
    "#6b96cd","#7da8da","#8fbbe7","#a1cdf4","#b3dfff","#c5f1ff","#d7ffff","#ffffff",
    "#ffffff","#ffffff","#ffebe8","#ffd6d0","#ffc1b8","#ffaca0","#ff9788","#ff8270",
    "#ff6d58","#ff583f","#ff4327","#ff2e0f","#e32000","#c81800","#a41200","#7f0000","#640000"
]
cmap_anom_temp = mcolors.ListedColormap(colors_anom_temp)
norm_anom_temp = mcolors.BoundaryNorm(boundaries_anom_temp, cmap_anom_temp.N, extend="both")

# --- 4. ANOMALIA PRECIPITAZIONE ---

boundaries_anom_prec = np.arange(-100,101,10)
colors_anom_prec = [
    "#6b3f2a","#844c2d","#9b5930","#b26633","#c97336","#df8142","#eaa974",
    "#f2c9a8","#f9e1cf","#fdf1e7","#ffffff","#ffffff","#dbf5db","#c6edc6","#b0e5b0",
    "#99dd99","#83d483","#6dcc6d","#56b456","#3e9c3e","#267426","#005500"
]
cmap_anom_prec = mcolors.ListedColormap(colors_anom_prec)
norm_anom_prec = mcolors.BoundaryNorm(boundaries_anom_prec, cmap_anom_prec.N, extend="both")

# ============================================================
# CONFINI REGIONALI - Download se necessario
# ============================================================
shapefile_path = os.path.join(base_dir, "Reg01012025_g_WGS84.shp")

# Se lo shapefile non esiste, devi scaricarlo o includerlo nel repository
# Per semplicità, assumo che sia incluso nel repository in una cartella 'shapefiles'
shapefile_repo = os.path.join(os.getcwd(), "shapefiles", "Reg01012025_g_WGS84.shp")

if os.path.exists(shapefile_repo):
    reg_df = gpd.read_file(shapefile_repo).explode(index_parts=False)
    if reg_df.crs.to_string() != "EPSG:4326":
        reg_df = reg_df.to_crs(epsg=4326)
else:
    print("Warning: Shapefile non trovato, plot senza confini regionali")
    reg_df = None

# ============================================================
# FUNZIONE PLOT (2 PANNELLI: ASSOLUTO + ANOMALIA)
# ============================================================
def plot_combo(data_abs, data_anom, 
               cmap_abs, norm_abs, 
               cmap_anom, norm_anom, 
               title_abs, title_anom, suptitle,
               unit_abs, unit_anom, 
               out_file):

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True
    )
    
    extent = [WEST, EAST, SOUTH, NORTH]
    
    # --- PANNELLO 1: ASSOLUTO ---
    ax0 = axes[0]
    pcm0 = ax0.pcolormesh(
        ds_obs["longitude"], ds_obs["latitude"], data_abs,
        cmap=cmap_abs, norm=norm_abs, shading="auto",
        transform=ccrs.PlateCarree()
    )
    ax0.set_extent(extent, crs=ccrs.PlateCarree())
    ax0.coastlines(linewidth=1)
    ax0.add_feature(cfeature.BORDERS, linewidth=0.8)
    
    if reg_df is not None:
        ax0.add_geometries(
            reg_df.geometry, ccrs.PlateCarree(),
            edgecolor="black", facecolor="none", linewidth=0.3
        )
    
    ax0.set_title(title_abs, fontsize=12, fontweight="bold", pad=8)
    
    cbar0 = fig.colorbar(
        pcm0, ax=ax0,
        orientation="horizontal",
        fraction=0.05, pad=0.03,
        extend="both"
    )
    cbar0.set_label(unit_abs, fontsize=8)
    cbar0.ax.tick_params(labelsize=6)

    # --- PANNELLO 2: ANOMALIA ---
    ax1 = axes[1]
    pcm1 = ax1.pcolormesh(
        ds_obs["longitude"], ds_obs["latitude"], data_anom,
        cmap=cmap_anom, norm=norm_anom, shading="auto",
        transform=ccrs.PlateCarree()
    )
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.coastlines(linewidth=1)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.8)
    
    if reg_df is not None:
        ax1.add_geometries(
            reg_df.geometry, ccrs.PlateCarree(),
            edgecolor="black", facecolor="none", linewidth=0.3
        )
    
    ax1.set_title(title_anom, fontsize=12, fontweight="bold", pad=8)

    cbar1 = fig.colorbar(
        pcm1, ax=ax1,
        orientation="horizontal",
        fraction=0.05, pad=0.03,
        extend="both"
    )
    cbar1.set_label(unit_anom, fontsize=8)
    cbar1.ax.tick_params(labelsize=6)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.03)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print("Salvato:", out_file)
                 
# ============================================================
# ESECUZIONE PLOT
# ============================================================

# 1. Temperatura
temp_filename = f"temperatura_{year_str}_{month_str}.png"
plot_combo(
    data_abs=temp_obs, 
    data_anom=temp_anom,
    cmap_abs=cmap_t, 
    norm_abs=norm_t,
    cmap_anom=cmap_anom_temp, 
    norm_anom=norm_anom_temp,
    title_abs="Temperatura media [ERA5-Land]",
    title_anom="Anomalia di temperatura",
    suptitle=f"{mese} {year}",
    unit_abs="Temperatura (°C)",
    unit_anom="Anomalia rispetto alla climatologia [1991-2020] (°C)",
    out_file=os.path.join(out_dir, temp_filename)
)

# 2. Precipitazione
prec_filename = f"precipitazione_{year_str}_{month_str}.png"
plot_combo(
    data_abs=prec_obs, 
    data_anom=prec_anom,
    cmap_abs=cmap_p_cum, 
    norm_abs=norm_p_cum,
    cmap_anom=cmap_anom_prec, 
    norm_anom=norm_anom_prec,
    title_abs="Precipitazione totale [ERA5-Land]",
    title_anom="Anomalia di precipitazione",
    suptitle=f"{mese} {year}",
    unit_abs="Precipitazione (mm)",
    unit_anom="Anomalia rispetto alla climatologia [1991-2020] (mm)",
    out_file=os.path.join(out_dir, prec_filename)
)

print(f"\nFile generati:")
print(f"- {temp_filename}")
print(f"- {prec_filename}")
