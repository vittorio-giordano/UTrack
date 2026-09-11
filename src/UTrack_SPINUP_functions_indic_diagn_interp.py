"""UTrack: Lagrangian Atmospheric Moisture Tracking Model.

Core numerical routines and data management classes for global forward and backward
atmospheric moisture tracking.

"""

from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from numba import njit, prange

# UTILITY AND INTERPOLATION ROUTINES
# -----------------------------------------


@njit
def get_grid_indices_regular(lat, lon, lat0, dlat, lon0, dlon, nlat, nlon):
    """Compute grid indices for a regular latitude/longitude grid.

    Parameters
    ----------
    lat : float
        latitude in degrees.
    lon : float
        longitude in degrees (-180 to 180 or 0 to 360).
    lat0 : float
        Origin latitude (first grid point) in degrees.
    dlat : float
        Grid latitude resolution in degrees (can be negative).
    lon0 : float
        Origin longitude (first grid point) in degrees.
    dlon : float
        Grid longitude resolution in degrees.
    nlat : int
        Total number of latitude grid points.
    nlon : int
        Total number of longitude grid points.

    Returns
    -------
    lat_idx : int
        Clamped latitude grid index.
    lon_idx : int
        Wrapped longitude grid index.
    """
    # Latitude
    r_lat = (lat - lat0) / dlat
    lat_idx = int(np.round(r_lat))
    if lat_idx < 0:
        lat_idx = 0
    elif lat_idx >= nlat:
        lat_idx = nlat - 1

    # Longitude
    r_lon = (lon - lon0) / dlon
    lon_idx = int(np.round(r_lon))
    lon_idx = lon_idx % nlon
    if lon_idx < 0:
        lon_idx += nlon

    return lat_idx, lon_idx


@njit
def get_nearest_level_index(target_lev, levels):
    """Find the index of the nearest vertical pressure level."""
    return np.abs(levels - target_lev).argmin()


@njit
def get_starting_level_numba(
    q_profile, levels, dp, sp_val, q2_val, current_level=-999.0
):
    """Select a vertical level sampled probabilistically from the moisture profile.

    Probability density is proportional to layer water vapor mass (q * dp).
    Levels intersecting or below the surface pressure are discarded.

    Parameters
    ----------
    q_profile : np.ndarray
        1D array of specific humidity profile [kg/kg] sorted from surface to top.
    levels : np.ndarray
        1D array of pressure levels [hPa].
    dp : np.ndarray
        1D array of vertical layer thicknesses [hPa].
    sp_val : float
        Local surface pressure [hPa].
    q2_val : float
        2-meter surface specific humidity [kg/kg].
    current_level : float, optional
        Current parcel pressure level [hPa]. If -999.0, represents parcel birth or dry column fallback.

    Returns
    -------
    float
        Selected pressure level [hPa].
    """
    valid_mass = np.zeros(len(q_profile) + 1)
    lowest_valid_idx = -1

    for k in range(len(q_profile)):
        if not np.isnan(q_profile[k]) and q_profile[k] > 0 and levels[k] <= sp_val:
            if lowest_valid_idx == -1:
                lowest_valid_idx = k

        P_top = levels[k] - (dp[k] / 2.0)
        P_bot = levels[k] + (dp[k] / 2.0)
        actual_P_bot = min(sp_val, P_bot)

        if actual_P_bot > P_top:
            valid_mass[k] = q_profile[k] * (actual_P_bot - P_top)

    if lowest_valid_idx != -1:
        P_bot_lowest = levels[lowest_valid_idx] + (dp[lowest_valid_idx] / 2.0)
        if sp_val > P_bot_lowest:
            valid_mass[-1] = q2_val * (sp_val - P_bot_lowest)

    elif lowest_valid_idx == -1:
        # Fallback for completely dry column
        valid_mass[-1] = q2_val * 10.0

    q_sum = np.sum(valid_mass)

    # Fallback if total moisture is zero
    if q_sum == 0:
        if current_level != -999.0:
            return current_level
        else:
            return (
                float(levels[lowest_valid_idx])
                if lowest_valid_idx != -1
                else float(levels[0])
            )

    r = np.random.random() * q_sum
    cumsum = 0.0

    # surface level
    if valid_mass[-1] > 0:
        cumsum += valid_mass[-1]
        if cumsum >= r:
            return float(sp_val)

    # ERA5 pressure levels
    for i in range(len(levels)):
        val = valid_mass[i]
        if val > 0:
            cumsum += val
            if cumsum >= r:
                return float(levels[i])

    # Fallback
    if current_level != -999.0:
        return current_level
    return (
        float(levels[lowest_valid_idx]) if lowest_valid_idx != -1 else float(levels[0])
    )

@njit
def trilinear_interpolate_uvw(
    lat,
    lon,
    level,
    lat0,
    dlat,
    lon0,
    dlon,
    nlat,
    nlon,
    u_grid,
    v_grid,
    w_grid,
    levels,
    sp_grid,
    u10_grid,
    v10_grid,
    w10_grid,
):
    """Interpolate 3D wind components (u, v, w) trilinearly in space.

    Handles terrain intersections by interpolating between 10-meter
    surface winds and the first above-ground pressure level.

    Parameters
    ----------
    lat : float
        Parcel latitude [degrees north].
    lon : float
        Parcel longitude [degrees east].
    level : float
        Parcel pressure level [hPa].
    lat0, dlat, lon0, dlon : float
        Grid coordinates and grid cell resolutions [degrees].
    nlat, nlon : int
        Grid dimensions.
    u_grid, v_grid, w_grid : np.ndarray
        3D wind arrays (level, lat, lon). w_grid is in [Pa/s].
    levels : np.ndarray
        1D array of pressure levels [hPa].
    sp_grid : np.ndarray
        2D surface pressure array [hPa].
    u10_grid, v10_grid, w10_grid : np.ndarray
        2D surface wind component arrays [m/s].

    Returns
    -------
    u_interp, v_interp, w_interp : tuple of float
        Interpolated wind components (u [m/s], v [m/s], w [Pa/s]).
    """
    r_lat = (lat - lat0) / dlat
    r_lon = (lon - lon0) / dlon

    lat_idx0 = int(np.floor(r_lat))
    lon_idx0 = int(np.floor(r_lon))
    dy = r_lat - lat_idx0
    dx = r_lon - lon_idx0

    lat_idx1 = lat_idx0 + 1
    lon_idx1 = lon_idx0 + 1

    if lat_idx0 < 0:
        lat_idx0 = 0
    if lat_idx1 < 0:
        lat_idx1 = 0
    if lat_idx0 >= nlat:
        lat_idx0 = nlat - 1
    if lat_idx1 >= nlat:
        lat_idx1 = nlat - 1

    lon_idx0 = lon_idx0 % nlon
    if lon_idx0 < 0:
        lon_idx0 += nlon
    lon_idx1 = lon_idx1 % nlon
    if lon_idx1 < 0:
        lon_idx1 += nlon

    lev_idx0 = 0
    lev_idx1 = 0
    dz = 0.0

    if level >= levels[0]:
        lev_idx0, lev_idx1, dz = 0, 0, 0.0
    elif level <= levels[-1]:
        lev_idx0, lev_idx1, dz = len(levels) - 1, len(levels) - 1, 0.0
    else:
        for k in range(len(levels) - 1):
            if levels[k] >= level >= levels[k + 1]:
                lev_idx0, lev_idx1 = k, k + 1
                dz = (level - levels[lev_idx0]) / (levels[lev_idx1] - levels[lev_idx0])
                break

    u000 = u_grid[lev_idx0, lat_idx0, lon_idx0]
    u010 = u_grid[lev_idx0, lat_idx0, lon_idx1]
    u100 = u_grid[lev_idx0, lat_idx1, lon_idx0]
    u110 = u_grid[lev_idx0, lat_idx1, lon_idx1]
    u001 = u_grid[lev_idx1, lat_idx0, lon_idx0]
    u011 = u_grid[lev_idx1, lat_idx0, lon_idx1]
    u101 = u_grid[lev_idx1, lat_idx1, lon_idx0]
    u111 = u_grid[lev_idx1, lat_idx1, lon_idx1]

    has_nan = (
        np.isnan(u000)
        or np.isnan(u010)
        or np.isnan(u100)
        or np.isnan(u110)
        or np.isnan(u001)
        or np.isnan(u011)
        or np.isnan(u101)
        or np.isnan(u111)
    )

    if has_nan:
        nearest_lat = int(np.round(r_lat))
        nearest_lon = int(np.round(r_lon))
        nearest_lat = min(max(nearest_lat, 0), nlat - 1)
        nearest_lon = nearest_lon % nlon

        u_bot, u_top = (
            u_grid[lev_idx0, nearest_lat, nearest_lon],
            u_grid[lev_idx1, nearest_lat, nearest_lon],
        )
        v_bot, v_top = (
            v_grid[lev_idx0, nearest_lat, nearest_lon],
            v_grid[lev_idx1, nearest_lat, nearest_lon],
        )
        w_bot, w_top = (
            w_grid[lev_idx0, nearest_lat, nearest_lon],
            w_grid[lev_idx1, nearest_lat, nearest_lon],
        )

        local_sp = sp_grid[nearest_lat, nearest_lon]
        # Case 1: Both bounding vertical levels are above ground
        if not np.isnan(u_bot) and not np.isnan(u_top):
            return (
                u_bot * (1.0 - dz) + u_top * dz,
                v_bot * (1.0 - dz) + v_top * dz,
                w_bot * (1.0 - dz) + w_top * dz,
            )
        # Case 2: Lower level is subterranean, upper level is in the atmosphere
        if np.isnan(u_bot) and not np.isnan(u_top):
            if level <= local_sp:
                u_10, v_10, w_10 = (
                    u10_grid[nearest_lat, nearest_lon],
                    v10_grid[nearest_lat, nearest_lon],
                    w10_grid[nearest_lat, nearest_lon],
                )

                if local_sp <= levels[lev_idx1]:
                    return u_10, v_10, w_10

                dz_pbl = (local_sp - level) / (local_sp - levels[lev_idx1])
                return (
                    u_10 * (1.0 - dz_pbl) + u_top * dz_pbl,
                    v_10 * (1.0 - dz_pbl) + v_top * dz_pbl,
                    w_10 * (1.0 - dz_pbl) + w_top * dz_pbl,
                )
            return np.nan, np.nan, np.nan  # Trigger Push-To-Surface

        # Case 3: Both levels are below the surface
        if level <= local_sp:
            for k_up in range(lev_idx1, len(levels)):
                if not np.isnan(u_grid[k_up, nearest_lat, nearest_lon]):
                    u_val, v_val, w_val = (
                        u_grid[k_up, nearest_lat, nearest_lon],
                        v_grid[k_up, nearest_lat, nearest_lon],
                        w_grid[k_up, nearest_lat, nearest_lon],
                    )
                    u_10, v_10, w_10 = (
                        u10_grid[nearest_lat, nearest_lon],
                        v10_grid[nearest_lat, nearest_lon],
                        w10_grid[nearest_lat, nearest_lon],
                    )

                    if local_sp <= levels[k_up]:
                        return u_10, v_10, w_10

                    dz_pbl = (local_sp - level) / (local_sp - levels[k_up])
                    return (
                        u_10 * (1.0 - dz_pbl) + u_val * dz_pbl,
                        v_10 * (1.0 - dz_pbl) + v_val * dz_pbl,
                        w_10 * (1.0 - dz_pbl) + w_val * dz_pbl,
                    )
        return np.nan, np.nan, np.nan

    u0 = (u000 * (1.0 - dx) + u010 * dx) * (1.0 - dy) + (
        u100 * (1.0 - dx) + u110 * dx
    ) * dy
    u1 = (u001 * (1.0 - dx) + u011 * dx) * (1.0 - dy) + (
        u101 * (1.0 - dx) + u111 * dx
    ) * dy

    v0 = (
        v_grid[lev_idx0, lat_idx0, lon_idx0] * (1.0 - dx)
        + v_grid[lev_idx0, lat_idx0, lon_idx1] * dx
    ) * (1.0 - dy) + (
        v_grid[lev_idx0, lat_idx1, lon_idx0] * (1.0 - dx)
        + v_grid[lev_idx0, lat_idx1, lon_idx1] * dx
    ) * dy
    v1 = (
        v_grid[lev_idx1, lat_idx0, lon_idx0] * (1.0 - dx)
        + v_grid[lev_idx1, lat_idx0, lon_idx1] * dx
    ) * (1.0 - dy) + (
        v_grid[lev_idx1, lat_idx1, lon_idx0] * (1.0 - dx)
        + v_grid[lev_idx1, lat_idx1, lon_idx1] * dx
    ) * dy

    w0 = (
        w_grid[lev_idx0, lat_idx0, lon_idx0] * (1.0 - dx)
        + w_grid[lev_idx0, lat_idx0, lon_idx1] * dx
    ) * (1.0 - dy) + (
        w_grid[lev_idx0, lat_idx1, lon_idx0] * (1.0 - dx)
        + w_grid[lev_idx0, lat_idx1, lon_idx1] * dx
    ) * dy
    w1 = (
        w_grid[lev_idx1, lat_idx0, lon_idx0] * (1.0 - dx)
        + w_grid[lev_idx1, lat_idx0, lon_idx1] * dx
    ) * (1.0 - dy) + (
        w_grid[lev_idx1, lat_idx1, lon_idx0] * (1.0 - dx)
        + w_grid[lev_idx1, lat_idx1, lon_idx1] * dx
    ) * dy

    return (
        u0 * (1.0 - dz) + u1 * dz,
        v0 * (1.0 - dz) + v1 * dz,
        w0 * (1.0 - dz) + w1 * dz,
    )


@njit
def compute_specific_humidity_surface(dewpoint, pressure_pa):
    """Compute surface specific humidity from 2-meter dew point temperature.

    Incorporates mixed-phase saturation vapor pressure
    following the ECMWF Integrated Forecasting System (IFS) formulation.

    Parameters
    ----------
    dewpoint : float
        2-meter dew point temperature [Kelvin].
    pressure_pa : float
        Surface atmospheric pressure [Pascals].

    Returns
    -------
    float
        Specific humidity [kg/kg].
    """
    Rd = 287.0597
    Rv = 461.5250
    t0 = 273.16
    t_i = 250.16

    # Liquid phase saturation vapor pressure
    e_water = 611.21 * np.exp(17.502 * (dewpoint - t0) / (dewpoint - 32.19))
    # Ice phase saturation vapor pressure
    e_ice = 611.21 * np.exp(22.587 * (dewpoint - t0) / (dewpoint - (-0.7)))

    alpha = (dewpoint - t_i) / (t0 - t_i)
    alpha = np.minimum(np.maximum(alpha, 0.0), 1.0)
    alpha = alpha**2

    e_actual = alpha * e_water + (1.0 - alpha) * e_ice
    q = (Rd / Rv) * e_actual / (pressure_pa - ((1.0 - (Rd / Rv)) * e_actual))
    return q


@njit
def insert_new_parcels_in_place(parcels, new_p):
    """Insert newly emitted parcels into preallocated static memory array.
    Overwrites rows where parcel status equals 0.0 (inactive slots).
    """
    num_new = new_p.shape[0]
    max_p = parcels.shape[0]
    n_inserted = 0

    if num_new == 0:
        return True

    for i in range(max_p):
        if parcels[i, 11] == 0.0:
            for col in range(parcels.shape[1]):
                parcels[i, col] = new_p[n_inserted, col]
            n_inserted += 1
            if n_inserted == num_new:
                return True

    return False


@njit(parallel=True)
def run_global_cleanup_step(parcels, tracking_duration_days):
    """Inactivate parcels that exceed maximum age or are depleted of moisture.

    Operates in-place with zero memory allocation.

    Parameters
    ----------
    parcels : np.ndarray
        Global 2D parcel state array of shape (N, 14).
    tracking_duration_days : float
        Maximum allowed tracking duration in days.
    """
    n = parcels.shape[0]
    MST, ORIG_MST, TIME, START_TIME, STATUS = 3, 4, 5, 6, 11
    max_age_sec = tracking_duration_days * 86400.0

    for i in prange(n):
        if parcels[i, STATUS] == 0.0:
            continue

        age = np.abs(parcels[i, TIME] - parcels[i, START_TIME])
        orig_mst = parcels[i, ORIG_MST]

        # Inactivate if remaining moisture is less than 1% of initial moisture
        if orig_mst > 0.0:
            dry = (parcels[i, MST] / orig_mst) < 0.01
        else:
            dry = True

        if age > max_age_sec or dry:
            parcels[i, STATUS] = 0.0


# CORE TRACKING FUNCTIONS
# -----------------------------------------


def create_parcels_mask(
    mask,
    parcel_start_time,
    delta_t_sec,
    m,
    parcels_per_mm,
    forward_tracking,
    parcel_data_columns,
    current_id_counter=None,
    start_from_surface=False,
    single_point_coords=None,
):
    """Generate and initialize new parcels.
    Implements a Monte Carlo spatial allocation scheme where parcel emission
    probability is proportional to local cell moisture volume.

    Parameters
    ----------
    mask : np.ndarray
        2D spatial binary mask [lat, lon] defining the release region.
    parcel_start_time : datetime
        Emission timestamp.
    delta_t_sec : float
        Emission time interval in seconds (typically 3600.0).
    m : Meteo
        Atmospheric data container holding active meteorological fields.
    parcels_per_mm : float
        Parcel release density per mm of globally averaged flux depth.
    forward_tracking : bool
        True for forward tracking (forced by Evaporation); False for backward (Precipitation).
    parcel_data_columns : list of str
        Definitions of parcel columns.
    current_id_counter : int, optional
        Starting unique integer ID for tracking.
    start_from_surface : bool, optional
        If True, initializes parcels at local surface pressure; otherwise,
        samples vertically based on specific humidity.
    single_point_coords : tuple of float, optional
        Coordinates (lat, lon) if simulating an explicit point source.

    Returns
    -------
    np.ndarray
        2D array of newly emitted parcels of shape (num_parcels, 14).
    """

    moisture_flux = m.E if forward_tracking else m.P

    if m.surface_areas is None:
        raise ValueError("m.surface_areas not computed.")

    weighted_prob = moisture_flux * mask * m.surface_areas  # [mm* m^2 / s]

    total_prob = np.sum(weighted_prob)
    if total_prob == 0:
        return np.empty((0, len(parcel_data_columns)), dtype=np.float32)

    total_volume_released = np.sum(weighted_prob) * delta_t_sec
    total_mask_area = np.sum(mask * m.surface_areas)

    if total_mask_area > 0:
        average_depth_mm = total_volume_released / total_mask_area
    else:
        average_depth_mm = 0

    num_parcels = int(average_depth_mm * parcels_per_mm)
    if num_parcels == 0:
        return np.empty((0, len(parcel_data_columns)), dtype=np.float32)

    norm_mask = (weighted_prob / total_prob).flatten()
    flat_indices = np.random.choice(norm_mask.size, num_parcels, p=norm_mask)
    lat_indices, lon_indices = np.unravel_index(flat_indices, weighted_prob.shape)

    if single_point_coords is not None:
        # TRUE POINT SOURCE
        start_lats = np.full(num_parcels, single_point_coords[0], dtype=np.float32)
        start_lons = np.full(num_parcels, single_point_coords[1], dtype=np.float32)
    else:
        # AREA SOURCE
        dlat_cell = np.abs(m.dlat_deg)
        dlon_cell = np.abs(m.dlon_deg)
        start_lats = m.lats[lat_indices] + (
            (np.random.random(num_parcels) - 0.5) * dlat_cell
        )
        start_lons = m.lons[lon_indices] + (
            (np.random.random(num_parcels) - 0.5) * dlon_cell
        )

    start_levels = np.zeros(num_parcels)
    if forward_tracking and start_from_surface:
        start_levels = m.sp[lat_indices, lon_indices]
    else:
        for i in range(num_parcels):
            q_profile = m.q[:, lat_indices[i], lon_indices[i]]
            sp_local = m.sp[lat_indices[i], lon_indices[i]]
            q2_local = m.q2[lat_indices[i], lon_indices[i]]
            start_levels[i] = get_starting_level_numba(
                q_profile, m.levels, m.dp, sp_local, q2_local, current_level=-999.0
            )

    parcel_moisture = total_volume_released / num_parcels

    parcels = np.zeros(
        (num_parcels, len(parcel_data_columns)), dtype=np.float32
    )  # Parcel State Vector Definition (Array shape: [N, 14])
    parcels[:, 0] = start_lats  # Latitude [degrees north]
    parcels[:, 1] = start_lons  # Longitude [degrees east, -180 to 180]
    parcels[:, 2] = start_levels  # Pressure level [hPa]
    parcels[:, 3] = parcel_moisture  # Current moisture content [mm * m^2 or kg]
    parcels[:, 4] = parcel_moisture  # Initial Moisture at release
    parcels[:, 5] = parcel_start_time.timestamp()  # Current tracking timestamp
    parcels[:, 6] = parcel_start_time.timestamp()  # Release timestamp
    parcels[:, 7] = start_lats  # Initial release latitude [degrees north]
    parcels[:, 8] = start_lons  # Initial release longitude [degrees east]
    parcels[:, 9] = lat_indices  # Current grid latitude index [integer as float]
    parcels[:, 10] = lon_indices  # Current grid longitude index [integer as float]
    parcels[:, 11] = 1.0  # Parcel status [1.0 = Active, 0.0 = Inactive/Terminated]

    if len(parcel_data_columns) >= 13 and current_id_counter is not None:
        new_ids = np.arange(
            current_id_counter, current_id_counter + num_parcels, dtype=np.float32
        )
        parcels[:, 12] = new_ids  # Unique parcel identifier [float ID]
        # Column 13: Moisture allocated during current time step [mm * m^2 or kg]

    return parcels


@njit(parallel=True)
def run_advection_step(
    parcels,
    dt_sec,
    u,
    u10_grid,
    v,
    v10_grid,
    w,
    w10_grid,
    q,
    q2_grid,
    sp_grid,
    levels,
    dp,
    dlat_m,
    dlon_m,
    forward,
    lat0,
    dlat_deg,
    lon0,
    dlon_deg,
    nlat,
    nlon,
    use_mixing,
    mixing_timescale,
):
    """Advect active parcels using 3D wind fields and apply stochastic vertical mixing.

    Parameters
    ----------

    dt_sec : float
        Advection integration time step [seconds].
    dlat_m, dlon_m : np.ndarray
        Metric conversion arrays for latitude/longitude cell dimensions [meters].
    forward : bool
        True for forward advection; False for backward advection (winds inverted).
    use_mixing : bool
        Whether to enable stochastic vertical mixing.
    mixing_timescale : float
        Characteristic vertical mixing timescale [seconds] (e.g., 86400.0 for daily).

    """
    n = parcels.shape[0]
    LAT, LON, LEV, TIME, STATUS = 0, 1, 2, 5, 11

    for i in prange(n):
        if parcels[i, STATUS] == 0.0:
            continue

        lat = parcels[i, LAT]
        lon = parcels[i, LON]
        level = parcels[i, LEV]

        lat_idx, lon_idx = get_grid_indices_regular(
            lat, lon, lat0, dlat_deg, lon0, dlon_deg, nlat, nlon
        )

        u_p, v_p, w_p = trilinear_interpolate_uvw(
            lat,
            lon,
            level,
            lat0,
            dlat_deg,
            lon0,
            dlon_deg,
            nlat,
            nlon,
            u,
            v,
            w,
            levels,
            sp_grid,
            u10_grid,
            v10_grid,
            w10_grid,
        )

        #  push to surface if parcel is below terrain
        if np.isnan(u_p):
            for k in range(len(levels)):
                if not np.isnan(u[k, lat_idx, lon_idx]):
                    level = float(levels[k])

                    u_p, v_p, w_p = trilinear_interpolate_uvw(
                        lat,
                        lon,
                        level,
                        lat0,
                        dlat_deg,
                        lon0,
                        dlon_deg,
                        nlat,
                        nlon,
                        u,
                        v,
                        w,
                        levels,
                        sp_grid,
                        u10_grid,
                        v10_grid,
                        w10_grid,
                    )
                    break

        if np.isnan(u_p):
            u_p = 0.0
        if np.isnan(v_p):
            v_p = 0.0
        if np.isnan(w_p):
            w_p = 0.0

        if not forward:
            u_p = -u_p
            v_p = -v_p
            w_p = -w_p

        # Advection
        lon += u_p * dt_sec / dlon_m[lat_idx]
        lat += v_p * dt_sec / dlat_m[lat_idx]
        level += w_p * dt_sec / 100.0

        if lat > 90:
            lat = 180 - lat
            lon += 180
        if lat < -90:
            lat = -180 - lat
            lon += 180
        lon = ((lon + 180) % 360) - 180

        if level < 50:
            level = 50.0

        li_new, loi_new = get_grid_indices_regular(
            lat, lon, lat0, dlat_deg, lon0, dlon_deg, nlat, nlon
        )

        local_sp = sp_grid[li_new, loi_new]
        if level > local_sp:
            level = local_sp  # Scivola sul terreno

        if use_mixing:
            if np.random.random() < (dt_sec / mixing_timescale):
                level = float(
                    get_starting_level_numba(
                        q[:, li_new, loi_new],
                        levels,
                        dp,
                        sp_grid[li_new, loi_new],
                        q2_grid[li_new, loi_new],
                        current_level=level,
                    )
                )

        parcels[i, LAT] = lat
        parcels[i, LON] = lon
        parcels[i, LEV] = level
        parcels[i, TIME] += dt_sec

    return


@njit
def aggregate_cell_moisture(parcels, lat0, dlat_deg, lon0, dlon_deg, nlat, nlon):
    """Aggregate total moisture of all active parcels over each grid column.

    Reconstructs the 2D Lagrangian Total Column Water (TCW).

    Returns
    -------
    np.ndarray
        2D grid of aggregated moisture [nlat, nlon].
    """
    n = parcels.shape[0]
    LAT, LON, MST, STATUS = 0, 1, 3, 11
    sum_moisture = np.zeros((nlat, nlon), dtype=np.float64)

    for i in range(n):
        if parcels[i, STATUS] == 0.0:
            continue

        lat = parcels[i, LAT]
        lon = parcels[i, LON]
        moisture = parcels[i, MST]
        li, loi = get_grid_indices_regular(
            lat, lon, lat0, dlat_deg, lon0, dlon_deg, nlat, nlon
        )
        sum_moisture[li, loi] += moisture

    return sum_moisture


@njit(parallel=True)
def run_budget_step(parcels, fraction_grid, lat0, dlat_deg, lon0, dlon_deg, nlat, nlon):
    """Apply the Dirmeyer and Brubaker (1999) moisture allocation scheme.

    Fractionally partitions moisture based on local surface flux ratios.

    Parameters
    ----------
    fraction_grid : np.ndarray
        2D array of allocation fractions (e.g., P / TCW or ET / TCW).

    """
    n = parcels.shape[0]
    LAT, LON, MST, CLAT, CLON, STATUS, ALLOC = 0, 1, 3, 9, 10, 11, 13

    for i in prange(n):
        if parcels[i, STATUS] == 0.0:
            continue

        lat = parcels[i, LAT]
        lon = parcels[i, LON]
        moisture = parcels[i, MST]

        li, loi = get_grid_indices_regular(
            lat, lon, lat0, dlat_deg, lon0, dlon_deg, nlat, nlon
        )

        frac = fraction_grid[li, loi]

        allocated = moisture * frac
        moisture -= allocated

        parcels[i, MST] = moisture
        parcels[i, ALLOC] = allocated
        parcels[i, CLAT] = li
        parcels[i, CLON] = loi
    return


@njit
def accumulate_deposition(parcels, output_map):
    """Accumulate allocated moisture from active parcels onto the output grid.

    Parameters
    ----------
    output_map : np.ndarray
        2D output accumulator array [nlat, nlon].

    """
    n = parcels.shape[0]
    CLAT, CLON, STATUS, ALLOC = 9, 10, 11, 13

    for i in range(n):
        status = parcels[i, STATUS]
        if status == 0.0:
            continue

        li = int(parcels[i, CLAT])
        loi = int(parcels[i, CLON])
        amount = parcels[i, ALLOC]

        if amount > 0:
            if status == 1.0:
                output_map[li, loi] += amount
            parcels[i, ALLOC] = 0.0


# METEO CLASS for atmospheric data handling
# -----------------------------------------
class Meteo:
    """Atmospheric forcing manager loading ERA5 from the public Google Cloud ARCO Zarr store.

    Handles asynchronous daily buffering, spatial grid metric calculations,
    linear sub-hourly interpolation, and flux accumulation conventions.

    Parameters
    ----------
    use_surface_pressure : bool, optional
        Whether to load and evaluate 2D surface pressure for terrain masking.
        Default is False (assumes 1000 hPa nominal surface).
    """

    def __init__(self, use_surface_pressure=False):
        self.ds_arco = None
        self.use_sp = use_surface_pressure

        # 24-hour daily memory buffers
        self.u_buffer, self.v_buffer, self.w_buffer, self.q_buffer = (
            None,
            None,
            None,
            None,
        )
        self.E_buffer, self.P_buffer, self.PW_buffer = None, None, None
        self.sp_buffer = None
        self.buffer_date = None

        self.u10_buffer, self.v10_buffer, self.d2m_buffer = None, None, None

        # Active time step interpolated arrays
        self.u, self.v, self.w, self.q = None, None, None, None
        self.E, self.P, self.PW = None, None, None
        self.sp = None

        self.u10, self.v10, self.w10, self.q2 = None, None, None, None

        # Grid coordinate definitions
        self.levels, self.lats, self.lons = None, None, None
        self.dlat_m, self.dlon_m, self.surface_areas, self.dp = None, None, None, None

        self.lat0, self.dlat_deg = None, None
        self.lon0, self.dlon_deg = None, None
        self.nlat, self.nlon = None, None

        self.pressure_levels_to_load = [
            1000,
            975,
            950,
            925,
            900,
            875,
            850,
            825,
            800,
            775,
            750,
            700,
            650,
            600,
            550,
            500,
            450,
            400,
            350,
            300,
            250,
            200,
            150,
            100,
            50,
        ]  # can be extended to 37

    def initialize(self):
        """Establish lazy connection to ARCO-ERA5 Zarr store on Google Cloud Storage."""
        print("Initializing connection to ARCO ERA5 (Google Cloud)...")
        try:
            self.ds_arco = xr.open_zarr(
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                chunks={"time": 1},
                storage_options=dict(token="anon"),
            )
            print("   -> Connection established successfully.")
        except Exception as e:
            print(f"FATAL: Failed to connect to ARCO-ERA5: {e}")
            print("Verify network connectivity and ensure 'gcsfs' is installed.")
            raise e

    def _load_daily_buffer(self, date):
        """Load a 24-hour meteorological block into memory.

        Parameters
        ----------
        date : datetime
            Day to load.
        """
        print(f"--- Loading 24-hour meteorological buffer for {date.date()} ---")
        start = datetime(date.year, date.month, date.day, 0, 0)
        end = start + timedelta(hours=24)

        def load_var(var_name, is_3d=False):
            ds = self.ds_arco.sel(time=slice(start, end))
            if is_3d:
                ds = ds.sel(level=self.pressure_levels_to_load)
            if ds.longitude.max() > 180:
                ds = ds.assign_coords(
                    longitude=(((ds.longitude + 180) % 360) - 180)
                ).sortby("longitude")
            if ds.latitude[0] < ds.latitude[-1]:
                ds = ds.sortby("latitude", ascending=False)
            return ds[var_name].compute().values, ds

        self.q_buffer, ds_ref = load_var("specific_humidity", True)
        self.u_buffer, _ = load_var("u_component_of_wind", True)
        self.v_buffer, _ = load_var("v_component_of_wind", True)
        self.w_buffer, _ = load_var("vertical_velocity", True)
        self.E_buffer, _ = load_var("evaporation", False)
        self.P_buffer, _ = load_var("total_precipitation", False)
        self.PW_buffer, _ = load_var("total_column_water", False)
        self.u10_buffer, _ = load_var("10m_u_component_of_wind", False)
        self.v10_buffer, _ = load_var("10m_v_component_of_wind", False)
        self.d2m_buffer, _ = load_var("2m_dewpoint_temperature", False)

        if self.use_sp:
            print("    -> Downloading Surface Pressure...")
            self.sp_buffer, _ = load_var("surface_pressure", False)

        if self.lats is None:
            self.lats = ds_ref.latitude.values
            self.lons = ds_ref.longitude.values
            self.levels = ds_ref.level.values
            self.compute_grid_info()

        self.buffer_date = date.date()

    def _set_active_hour(self, h, m_frac, forward=True):
        """Interpolate variables between consecutive hours.

        Parameters
        ----------
        h : int
            Hour index (0 to 23).
        m_frac : float
            Fractional hour [0.0 to 1.0].
        forward : bool, optional
            Tracking direction (governs flux accumulation alignment).
        """
        w0, w1 = (1.0 - m_frac), m_frac

        if self.use_sp:
            sp_raw = self.sp_buffer[h] * w0 + self.sp_buffer[h + 1] * w1
            self.sp = sp_raw / 100.0
        else:
            self.sp = np.full((self.nlat, self.nlon), 1000.0, dtype=np.float32)

        self.u = self.u_buffer[h] * w0 + self.u_buffer[h + 1] * w1
        self.v = self.v_buffer[h] * w0 + self.v_buffer[h + 1] * w1
        self.w = self.w_buffer[h] * w0 + self.w_buffer[h + 1] * w1
        self.q = self.q_buffer[h] * w0 + self.q_buffer[h + 1] * w1
        self.PW = self.PW_buffer[h] * w0 + self.PW_buffer[h + 1] * w1

        self.u10 = self.u10_buffer[h] * w0 + self.u10_buffer[h + 1] * w1
        self.v10 = self.v10_buffer[h] * w0 + self.v10_buffer[h + 1] * w1
        self.w10 = np.zeros_like(self.sp)

        d2m_interp = self.d2m_buffer[h] * w0 + self.d2m_buffer[h + 1] * w1
        self.q2 = compute_specific_humidity_surface(d2m_interp, self.sp * 100.0)

        # Align with ERA5 hourly accumulation convention
        if forward:
            flux_idx = h + 1
        else:
            flux_idx = h if m_frac == 0.0 else h + 1

        # Rectify evaporation: flip sign (upward positive) and mask condensation
        self.E = (
            np.abs(np.minimum(self.E_buffer[flux_idx], 0.0)) * 1000.0 / 3600.0
        )  # m to mm/s
        self.P = np.maximum(self.P_buffer[flux_idx], 0.0) * 1000.0 / 3600.0  # m to mm/s

        if self.use_sp:
            mask_underground = self.levels[:, None, None] > self.sp[None, :, :]
            self.q[mask_underground] = np.nan
            self.u[mask_underground] = np.nan
            self.v[mask_underground] = np.nan
            self.w[mask_underground] = np.nan

    def load_data(self, date, forward=True):
        """Load meteorological fields for the requested timestamp.
        Automatically triggers buffer updates across day boundaries.
        """
        logical_buffer_id = date.date()

        if logical_buffer_id != self.buffer_date:
            self._load_daily_buffer(date)
            self.buffer_date = logical_buffer_id

        m_frac = date.minute / 60.0
        self._set_active_hour(date.hour, m_frac, forward)

    def compute_grid_info(self):
        """Compute grid metric dimensions, cell surface areas, and layer thicknesses."""
        print("Computing grid metrics and layer thicknesses")

        self.nlat = len(self.lats)
        self.nlon = len(self.lons)
        self.lat0 = float(self.lats[0])
        self.lon0 = float(self.lons[0])
        self.dlat_deg = float(self.lats[1] - self.lats[0])
        self.dlon_deg = float(self.lons[1] - self.lons[0])

        lat_rad = np.radians(self.lats)
        self.dlat_m = (
            111132.92
            - 559.82 * np.cos(2 * lat_rad)
            + 1.175 * np.cos(4 * lat_rad)
            - 0.0023 * np.cos(6 * lat_rad)
        ).astype(np.float32)
        self.dlon_m = (
            111412.84 * np.cos(lat_rad)
            - 93.5 * np.cos(3 * lat_rad)
            + 0.118 * np.cos(5 * lat_rad)
        ).astype(np.float32)

        # Areas
        R = 6371000.0
        lat_b = np.zeros(self.nlat + 1)
        lat_b[1:-1] = 0.5 * (self.lats[:-1] + self.lats[1:])
        lat_b[0] = self.lats[0] - 0.5 * self.dlat_deg
        lat_b[-1] = self.lats[-1] + 0.5 * self.dlat_deg
        lat_b = np.clip(lat_b, -90, 90)
        lon_b = np.zeros(self.nlon + 1)
        lon_b[1:-1] = 0.5 * (self.lons[:-1] + self.lons[1:])
        lon_b[0] = self.lons[0] - 0.5 * self.dlon_deg
        lon_b[-1] = self.lons[-1] + 0.5 * self.dlon_deg

        sin_diff = np.abs(
            np.sin(np.radians(lat_b[1:])) - np.sin(np.radians(lat_b[:-1]))
        )
        lon_diff = np.abs(np.radians(lon_b[1:]) - np.radians(lon_b[:-1]))
        self.surface_areas = (R**2 * sin_diff[:, None] * lon_diff[None, :]).astype(
            np.float32
        )

        # Levels thickness (dp)
        bounds = np.zeros(len(self.levels) + 1)
        bounds[1:-1] = (self.levels[:-1] + self.levels[1:]) / 2.0
        bounds[0] = self.levels[0] + (self.levels[0] - self.levels[1]) / 2.0
        bounds[-1] = self.levels[-1] - (self.levels[-2] - self.levels[-1]) / 2.0

        self.dp = np.abs(bounds[:-1] - bounds[1:]).astype(np.float32)

        print("   -> Grid metrics successfully computed.")

    def close(self):
        """Close connection to ARCO-ERA5 Zarr store."""
        if self.ds_arco:
            self.ds_arco.close()
            print("ARCO-ERA5 connection closed.")
