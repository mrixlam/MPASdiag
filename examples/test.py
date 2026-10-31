import mpasdiag as md

grid = "../data/grids/x1.10242.static.nc"
data = "../data/u240k/diag"
var = "t2m"

proc = md.MPAS2DProcessor(
    grid_file=grid, verbose=False,
).load_2d_data(data)

field = proc.dataset[var].isel(Time=0)
lon, lat = proc.extract_2d_coordinates_for_variable(
    var, field,
)

area = md.GeographicBounds(-180, 180, -90, 90)
viz = md.MPASSurfacePlotter(verbose=False)
viz.create_surface_map(
    lon=lon, lat=lat, data=field.values,
    var_name=var, bounds=area,
)
viz.save_plot("t2m_map")
viz.close_plot()
