import sys
from pathlib import Path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

from mpasdiag.processing.processors_3d import MPAS3DProcessor

DATA_DIR = str(Path(__file__).parent.parent / 'data' / 'u15k')
GRID_FILE = str(Path(__file__).parent.parent / 'data' / 'grids' / 'x1.2621442.init.nc')

proc = MPAS3DProcessor(GRID_FILE, verbose=True)
proc.load_3d_data(DATA_DIR)
vars3d = proc.get_available_3d_variables()
print('Available 3D vars:', vars3d[:20])
var = 'theta' if 'theta' in vars3d else vars3d[0]
print('Using variable:', var)

# Get vertical levels (pressure)
try:
    v_pressure = proc.get_vertical_levels(var, return_pressure=True, time_index=0)
    import numpy as np
    v_arr = np.array(v_pressure)
    print('Vertical levels (pressure) dtype:', v_arr.dtype)
    print('min,max,finite:', np.nanmin(v_arr), np.nanmax(v_arr), np.all(np.isfinite(v_arr)))
except Exception as e:
    print('Error getting pressure levels:', e)

# Print presence of related variables
for name in ['pressure_p','pressure_base','fzp','hyam','hybm','surface_pressure']:
    print(name, 'in dataset?', name in proc.dataset)
    if name in proc.dataset:
        da = proc.dataset[name]
        try:
            td = da.isel({'Time':0}).values
        except Exception:
            td = da.values
        import numpy as np
        print('  shape:', np.shape(td), 'min,max,finite:', np.nanmin(td), np.nanmax(td), np.all(np.isfinite(td)))

# Also print get_vertical_levels(return_pressure=False)
v_indices = proc.get_vertical_levels(var, return_pressure=False, time_index=0)
print('Vertical levels (indices) sample:', v_indices[:10])

print('Done')
