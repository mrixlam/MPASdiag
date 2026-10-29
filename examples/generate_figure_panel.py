# Standard libraries
import numpy as np
from pathlib import Path
from datetime import datetime

# Matplotlib + Cartopy build and arrange the figure environment
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# Load relevant MPASdiag classes for data loading, processing, and remapping
from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.visualization.surface import MPASSurfacePlotter

# Sample data bundled in the repository
gridPath = '../data/grids/x1.10242.static.nc'
dataDir  = '../data/u240k/diag'

VAR = 't2m'                       # 2-m temperature (a 2-D diagnostic field)
N_ROWS, N_COLS = 2, 3             # 2 rows x 3 columns = 6 panels
TINDICES = [0, 1, 2, 3, 4, 5]     # one forecast time per panel

# Forecast initialization time, used to label each panel's lead time
initTime = '20240917T00'
init64 = np.datetime64(datetime.strptime(initTime, '%Y%m%dT%H'))
init_time_disp = str(np.datetime64(init64, 'h')).replace('T', ' ') + ':00 UTC'

# Map extent shared by every panel (global)
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX =  -90.0,  90.0
data_crs = ccrs.PlateCarree()

# Shared filled-contour levels [deg C] and colormap so panels are comparable
custom_levels = list(range(-40, 44, 4))      # -40, -36, ..., 40
colormap = 'RdYlBu_r'
norm = BoundaryNorm(custom_levels, ncolors=plt.get_cmap(colormap).N, clip=True)

# Resolution of the regular grid the cells are remapped onto [deg]
GRID_RES = 0.25

# Output directory
OUTPUT_DIR = Path('./output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the unstructured t2m field; the mesh is shared by all times
processor = MPAS2DProcessor(grid_file=gridPath)
processor.load_2d_data(dataDir)

# A surface plotter instance, used here only for its gridding helper
plotter = MPASSurfacePlotter(verbose=False, dpi=300)

# Create the 2 x 3 grid of Cartopy GeoAxes
fig, axes = plt.subplots(
    N_ROWS, N_COLS, figsize=(18, 8), dpi=300,
    subplot_kw={'projection': data_crs},
)

# Flatten the 2-D array of axes so we can loop over it with a single index
axes_flat = axes.flat

# Initialize variables to hold the cell coordinates and the mappable from the last panel
lon = lat = None 
cs = None

for i, ax in enumerate(axes_flat):
    # Hide any trailing axes if there are fewer times than panels
    if i >= len(TINDICES):
        ax.set_visible(False)
        continue
    
    # Get the time index for this panel
    tindex = TINDICES[i]

    # Pull t2m for this time and convert Kelvin -> Celsius
    field = processor.get_2d_variable_data(VAR, tindex)

    # Extract the cell coordinates only once, since they are shared by all times
    if lon is None:
        lon, lat = processor.extract_2d_coordinates_for_variable(VAR, field)

    # Convert from Kelvin to Celsius for plotting
    t2m_c = field.values.flatten() - 273.15

    # Remap the unstructured cells onto a regular lat/lon grid for shading
    lon_mesh, lat_mesh, data_interp = plotter._interpolate_to_grid(
        lon, lat, t2m_c,
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
        grid_resolution=GRID_RES,
    )

    # Filled contours with identical levels/norm in every panel
    cs = ax.contourf(
        lon_mesh, lat_mesh, data_interp, levels=custom_levels,
        cmap=colormap, norm=norm, transform=data_crs, extend='both',
    )

    # Coastlines/borders and the fixed shared extent
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray', alpha=0.6)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=data_crs)

    # Edge tick labels only: latitudes on the left column, longitudes on the bottom row
    row, col = divmod(i, N_COLS)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (col == 0)
    gl.bottom_labels = (row == N_ROWS - 1)

    # Panel index label (a), (b), (c), ...
    ax.text(0.01, 0.94, f'({chr(97 + i)})',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='left', zorder=6,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.8, edgecolor='gray', linewidth=0.6))

    # Per-panel title: valid time and forecast lead time
    valtime = processor.dataset['Time'][tindex].values
    valtime_str = str(valtime.astype('datetime64[h]')).replace('-', '')
    lead_minutes = int((valtime - init64) / np.timedelta64(1, 'm'))
    lead_h, lead_m = divmod(lead_minutes, 60)
    ax.set_title(f'{valtime_str} UTC (+{lead_h:02d}:{lead_m:02d})', fontsize=11)

# Make room at the bottom, then add one horizontal colorbar shared by all panels
fig.subplots_adjust(wspace=0.08, hspace=0.18, bottom=0.13, top=0.90)
cax = fig.add_axes([0.25, 0.08, 0.50, 0.02])

# Draw the colorbar 
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal',
                    ticks=custom_levels[::2], extend='both')
cbar.set_label('2-m Temperature [°C]')

# Overall figure title
fig.suptitle(f'2-m Temperature | Init: {init_time_disp}',
             fontsize=16, fontweight='bold')

out_file = OUTPUT_DIR / f'{VAR}_panel_2x3_{TINDICES[0]}-{TINDICES[-1]}.png'
fig.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_file}')