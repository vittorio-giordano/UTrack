"""UTrack: Lagrangian Atmospheric Moisture Tracking Model

This module provides the primary execution pipeline for running global or regional
forward and backward atmospheric moisture tracking simulations using ERA5 atmospheric
forcing.

Usage
-----
Run directly from the command line by supplying a YAML configuration file, simulation
year, and month, or a .job file for SLURM job submission. Example command:

    python UTrack_SPINUP_tracking_indic_diagn_interp.py --config config.yml --year 2024 --month 5
            UTrack_SPINUP_tracking_indic_diagn_interp.py config.yml job.job

"""

import argparse
import calendar
import os
import random
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from UTrack_SPINUP_functions_indic_diagn_interp import (
    Meteo,
    accumulate_deposition,
    aggregate_cell_moisture,
    create_parcels_mask,
    insert_new_parcels_in_place,
    run_advection_step,
    run_budget_step,
    run_global_cleanup_step,
)


# CLI arguments
# ------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="UTrack Lagrangian Model")
    parser.add_argument(
        "--config", type=str, required=True, help=".yml configuration file path"
    )
    parser.add_argument("--year", type=int, required=True, help="Year of simulation")
    parser.add_argument("--month", type=int, required=True, help="Month of simulation")
    return parser.parse_args()


def main():
    args = parse_arguments()
    YEAR = args.year
    MONTH = args.month

    # Parameters from .yml config file
    # ------------------------------
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    # Domain configuration
    global_mask = config["domain"].get("global_mask", False)
    singlepoint = config["domain"]["singlepoint"]
    single_point_coords = config["domain"]["single_point_coords"]
    LSM_FILE = config["domain"]["mask_file"]

    # Tracking parameters
    forward_tracking = config["tracking"]["forward_tracking"]
    spin_up = config["tracking"]["spin_up_mode"]
    tracking_duration_days = config["tracking"]["tracking_duration_days"]

    # Physics parameters
    START_FROM_SURFACE = config["physics"]["start_from_surface"]
    USE_SURFACE_PRESSURE = config["physics"]["use_surface_pressure"]
    USE_MIXING = config["physics"]["use_mixing"]
    MIXING_TIMESCALE = float(config["physics"]["mixing_timescale"])

    # Numerical settings
    parcels_per_mm = config["numerics"]["parcels_per_mm"]
    dt_hours = float(config["numerics"]["dt_hours"])
    MAX_PARCELS = config["numerics"]["max_parcels"]

    # System outputs
    output_dir = config["system"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Parcels array columns
    cols = [
        "lat",
        "lon",
        "lev",
        "mst",
        "orig_mst",
        "time",
        "start_time",
        "slat",
        "slon",
        "clat",
        "clon",
        "status",
        "id",
        "allocated",
    ]

    # temporal setup
    dt = timedelta(hours=dt_hours)
    dt_sec = dt.total_seconds()

    _, last_day = calendar.monthrange(YEAR, MONTH)
    target_start = datetime(YEAR, MONTH, 1, 0, 0)
    target_end = datetime(YEAR, MONTH, last_day, 23, 0)

    if spin_up:
        print(f"Spin-up enabled: Validation targeting {target_start.strftime('%Y-%m')}")
        spinup_days = tracking_duration_days

        if forward_tracking:
            curr_date = target_start - timedelta(days=spinup_days)
            simulation_end_date = target_end  # Stop at the end of the target period
            release_limit_date = (
                target_end  # Release parcels until the end of the target period
            )
            time_step = dt  # Positive

        else:
            curr_date = target_end + timedelta(days=spinup_days)
            simulation_end_date = (
                target_start  # Stop at the beginning of the target period
            )
            release_limit_date = target_start
            time_step = -dt  # Negative

    else:
        # Standard tracking mode
        print("Standard Tracking Mode (no initialization)")

        if forward_tracking:
            time_step = dt
            curr_date = target_start
            release_limit_date = target_end
            simulation_end_date = release_limit_date + timedelta(
                days=tracking_duration_days
            )
        else:
            time_step = -dt
            curr_date = target_end
            release_limit_date = target_start
            simulation_end_date = release_limit_date - timedelta(
                days=tracking_duration_days
            )

    print(f"Simulation Start:               {curr_date}")
    print(f"Simulation End:                 {simulation_end_date}")
    print(f"Moisture Identification Window: {target_start} <-> {target_end}")
    sim_start_date = curr_date

    # Initialize Meteorological Data Manager
    # ------------------------------
    m = Meteo(use_surface_pressure=USE_SURFACE_PRESSURE)
    m.initialize()
    m.load_data(curr_date, forward_tracking)

    output_map = np.zeros((len(m.lats), len(m.lons)), dtype=np.float64)

    print(f"Pre-allocating memory for {MAX_PARCELS} parcels (Object Pool)...")
    parcels = np.zeros((MAX_PARCELS, len(cols)), dtype=np.float32)

    reference_map_output = np.zeros((len(m.lats), len(m.lons)), dtype=np.float64)
    reference_map_input = np.zeros((len(m.lats), len(m.lons)), dtype=np.float64)

    tot_lagrangian_tcw = np.zeros((len(m.lats), len(m.lons)), dtype=np.float64)
    tot_era5_tcw = np.zeros((len(m.lats), len(m.lons)), dtype=np.float64)

    # random snapshots scheduling for diagnostics
    # ------------------------------
    num_snapshots = 5

    if forward_tracking:
        boundary_date = target_start
    else:
        boundary_date = target_end

    total_hours = int((target_end - target_start).total_seconds() / 3600.0)
    all_possible_dates = [
        target_start + timedelta(hours=i) for i in range(total_hours + 1)
    ]

    if boundary_date in all_possible_dates:
        all_possible_dates.remove(boundary_date)

    random_dates = random.sample(all_possible_dates, num_snapshots - 1)
    if forward_tracking:
        snapshot_dates = [boundary_date] + sorted(random_dates)
    else:
        snapshot_dates = [boundary_date] + sorted(random_dates, reverse=True)

    print(f"Snapshots: {[d.strftime('%Y-%m-%d %H:%M') for d in snapshot_dates]}")

    snap_lag_tcw = np.zeros((num_snapshots, len(m.lats), len(m.lons)), dtype=np.float64)
    snap_era5_tcw = np.zeros(
        (num_snapshots, len(m.lats), len(m.lons)), dtype=np.float64
    )
    snap_era5_flux = np.zeros(
        (num_snapshots, len(m.lats), len(m.lons)), dtype=np.float64
    )
    snap_lag_flux = np.zeros(
        (num_snapshots, len(m.lats), len(m.lons)), dtype=np.float64
    )

    particle_id_counter = 0 if singlepoint else None
    parcels_history_list = []

    # Directory for trajectories
    if singlepoint:
        traj_dir = os.path.join(output_dir, "trajectories", f"{YEAR}_{MONTH:02d}_SP")
        os.makedirs(traj_dir, exist_ok=True)

    # Release domain definition
    # ------------------------------
    print("Defining release domain")
    if singlepoint:
        print(f"MODE: Single Point Release at {single_point_coords}")
        mask_array = np.zeros((len(m.lats), len(m.lons)), dtype=np.float32)
        lat_idx = (np.abs(m.lats - single_point_coords[0])).argmin()
        lon_idx = (np.abs(m.lons - single_point_coords[1])).argmin()
        mask_array[lat_idx, lon_idx] = 1.0
        print(f"  -> grid indexes: [{lat_idx}, {lon_idx}]")

    elif global_mask:
        print("MODE: Global Release domain")
        mask_array = np.ones((len(m.lats), len(m.lons)), dtype=np.float32)
        print(f"  -> Global Mask active: {m.nlat * m.nlon} source cells.")

    else:
        print(f"MODE: Custom Regional Mask from {LSM_FILE}")
        with xr.open_dataset(LSM_FILE) as ds_lsm:
            rename_dict = {}
            if "lat" in ds_lsm.coords and "latitude" not in ds_lsm.coords:
                rename_dict["lat"] = "latitude"
            if "lon" in ds_lsm.coords and "longitude" not in ds_lsm.coords:
                rename_dict["lon"] = "longitude"

            if rename_dict:
                ds_lsm = ds_lsm.rename(rename_dict)
                print(f"  -> Coordinate rinominate: {rename_dict}")

            ignore_vars = ["number", "valid_time", "expver"]
            valid_vars = [v for v in ds_lsm.data_vars if v not in ignore_vars]

            if not valid_vars:
                raise ValueError(f"No valid data variables found in: {LSM_FILE}")

            mask_var_name = valid_vars[-1]
            print(f"  -> Mask variable detected: '{mask_var_name}'")
            land_mask_xr = ds_lsm[mask_var_name].load().squeeze(drop=True)

        if land_mask_xr.longitude.max() > 180:
            land_mask_xr = land_mask_xr.assign_coords(
                longitude=(((land_mask_xr.longitude + 180) % 360) - 180)
            ).sortby("longitude")

        mask_array_xr = land_mask_xr.interp(
            latitude=m.lats, longitude=m.lons, method="nearest"
        )
        mask_array = mask_array_xr.values
        mask_array = np.nan_to_num(mask_array, nan=0.0)
        mask_array = np.where(mask_array > 0.0, 1.0, 0.0)

        num_celle_attive = int(np.sum(mask_array))
        print(f"  -> Regional Mask active: {num_celle_attive} source cells found.")

    # Main Simulation Loop
    # ------------------------------
    total_released = 0.0
    is_first_accum_step = True

    # Profiling Timers
    time_acc_load = 0.0
    time_acc_create = 0.0
    time_acc_concat = 0.0
    time_acc_adv = 0.0
    time_acc_agg = 0.0
    time_acc_budg = 0.0
    time_acc_clean = 0.0
    profiling_records = []

    print("Starting trajectory integration loop...")

    while (forward_tracking and curr_date <= simulation_end_date) or (
        not forward_tracking and curr_date >= simulation_end_date
    ):
        t0 = time.time()
        m.load_data(curr_date, forward_tracking)
        time_acc_load += time.time() - t0

        active_mask = parcels[:, 11] > 0
        num_active = np.count_nonzero(active_mask)

        if spin_up:
            do_accumulate = target_start <= curr_date <= target_end
        else:
            do_accumulate = True

        if do_accumulate and is_first_accum_step:
            if num_active > 0:
                valid_parcels_mask = parcels[:, 11] == 1.0
                input_from_spinup = np.sum(parcels[valid_parcels_mask, 3])
                total_released += input_from_spinup
                print(
                    f"  -> Accumulation Start: Added {input_from_spinup/10**3:.2e} m3 from active parcels from initialization period."
                )
            is_first_accum_step = False

        if do_accumulate:
            if forward_tracking:
                current_flux = m.P
                input_flux = m.E
            else:
                current_flux = m.E
                input_flux = m.P

            reference_map_output += current_flux * dt_sec * m.surface_areas
            reference_map_input += input_flux * dt_sec * m.surface_areas

        if forward_tracking:
            in_release_window = curr_date <= release_limit_date
        else:
            in_release_window = curr_date >= release_limit_date

        # Hourly parcels release
        if curr_date.minute == 0 and in_release_window:
            t0 = time.time()

            sp_coords_pass = single_point_coords if singlepoint else None

            new_p = create_parcels_mask(
                mask_array,
                curr_date,
                3600.0,
                m,
                parcels_per_mm,
                forward_tracking,
                cols,
                particle_id_counter,
                START_FROM_SURFACE,
                sp_coords_pass,
            )
            time_acc_create += time.time() - t0

            num_new = len(new_p)
            if num_new > 0:
                t0 = time.time()
                success = insert_new_parcels_in_place(parcels, new_p)
                if not success:
                    raise MemoryError(
                        f"Parcel pool exhausted! Preallocated MAX_PARCELS={MAX_PARCELS} "
                        f"is too small for current simulation density."
                    )
                time_acc_concat += time.time() - t0

                if do_accumulate:
                    total_released += np.sum(new_p[:, 4])

                if singlepoint:
                    particle_id_counter += len(new_p)

                active_mask = parcels[:, 11] > 0
                num_active = np.count_nonzero(active_mask)

        if num_active > 0:
            flux_grid = m.P if forward_tracking else m.E
            pw_safe = np.maximum(m.PW, 1e-6)
            fraction_grid = (flux_grid * dt_sec) / pw_safe
            fraction_grid = np.minimum(fraction_grid, 1.0)

            t0 = time.time()
            run_advection_step(
                parcels,
                dt_sec,
                m.u,
                m.u10,
                m.v,
                m.v10,
                m.w,
                m.w10,
                m.q,
                m.q2,
                m.sp,
                m.levels,
                m.dp,
                m.dlat_m,
                m.dlon_m,
                forward_tracking,
                m.lat0,
                m.dlat_deg,
                m.lon0,
                m.dlon_deg,
                m.nlat,
                m.nlon,
                USE_MIXING,
                MIXING_TIMESCALE,
            )
            time_acc_adv += time.time() - t0

            if curr_date.minute == 0:
                t0 = time.time()
                sum_moisture = aggregate_cell_moisture(
                    parcels, m.lat0, m.dlat_deg, m.lon0, m.dlon_deg, m.nlat, m.nlon
                )
                time_acc_agg += time.time() - t0

                if do_accumulate:
                    tot_lagrangian_tcw += sum_moisture
                    tot_era5_tcw += m.PW * m.surface_areas

                if curr_date in snapshot_dates:
                    idx = snapshot_dates.index(curr_date)
                    grid_volumes_requested = (
                        flux_grid * dt_sec * m.surface_areas
                    )  # ERA5
                    step_deposited = sum_moisture * fraction_grid

                    snap_lag_tcw[idx, :, :] = sum_moisture
                    snap_era5_tcw[idx, :, :] = m.PW * m.surface_areas
                    snap_era5_flux[idx, :, :] = grid_volumes_requested
                    snap_lag_flux[idx, :, :] = step_deposited
                    print(
                        f"  -> [DIAGN] Snapshot {idx + 1}/{num_snapshots} taken: {curr_date}"
                    )

            t0 = time.time()
            run_budget_step(
                parcels,
                fraction_grid,
                m.lat0,
                m.dlat_deg,
                m.lon0,
                m.dlon_deg,
                m.nlat,
                m.nlon,
            )

            if do_accumulate:
                accumulate_deposition(parcels, output_map)
            time_acc_budg += time.time() - t0

            t0 = time.time()
            if not singlepoint:
                run_global_cleanup_step(parcels, tracking_duration_days)

            else:
                active_indices = np.where(active_mask)[0]
                if len(active_indices) > 0:
                    active_parcels = parcels[active_indices]
                    age_in_seconds = np.abs(active_parcels[:, 5] - active_parcels[:, 6])
                    old_mask_active = age_in_seconds > (
                        tracking_duration_days * 86400.0
                    )

                    orig_mst = active_parcels[:, 4]
                    safe_orig = np.where(orig_mst > 0, orig_mst, 1.0)
                    dry_mask_active = (active_parcels[:, 3] / safe_orig) < 0.01
                    dry_mask_active[orig_mst == 0] = True
                    dead_mask_active = dry_mask_active | old_mask_active

                    dead_indices = active_indices[dead_mask_active]

                    if len(dead_indices) > 0:
                        dead_parcels = parcels[dead_indices].copy()
                        dead_parcels[:, 11] = 0.0

                        old_dead_local_mask = old_mask_active[dead_mask_active]
                        dead_parcels[old_dead_local_mask, 11] = -1.0
                        parcels_history_list.append(dead_parcels)

                    parcels[dead_indices, 11] = 0.0

                time_acc_clean += time.time() - t0

        if singlepoint and curr_date.minute == 0 and curr_date.hour % 6 == 0:
            if num_active > 0:
                traj_data = parcels[active_mask][:, [0, 1, 2, 3, 5, 12]]
                df_traj = pd.DataFrame(
                    traj_data, columns=["lat", "lon", "lev", "mst", "time", "id"]
                )
                df_traj["time"] = pd.to_datetime(df_traj["time"], unit="s")

                fname = f"traj_{curr_date.strftime('%Y%m%d_%H%M')}.parquet"
                df_traj.to_parquet(os.path.join(traj_dir, fname), engine="pyarrow")

        if curr_date.minute == 0:
            status = (
                "REC"
                if (spin_up and do_accumulate)
                else ("SPIN" if spin_up else "STD")
            )

            current_active = np.count_nonzero(parcels[:, 11])
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {curr_date} [{status}] | Active: {current_active} | Rel: {total_released/10**3:.2e}"
            )

            profiling_records.append(
                {
                    "Date": curr_date,
                    "Active_Parcels": current_active,
                    "Load_s": time_acc_load,
                    "Create_s": time_acc_create,
                    "Concat_s": time_acc_concat,
                    "Advect_s": time_acc_adv,
                    "Aggreg_s": time_acc_agg,
                    "Budget_s": time_acc_budg,
                    "Clean_s": time_acc_clean,
                }
            )

            time_acc_load = 0.0
            time_acc_create = 0.0
            time_acc_concat = 0.0
            time_acc_adv = 0.0
            time_acc_agg = 0.0
            time_acc_budg = 0.0
            time_acc_clean = 0.0

        curr_date += time_step

    m.close()
    print("Simulation completed.")

    # Account for residual airborne moisture at simulation end
    if spin_up and num_active > 0:
        valid_parcels_mask = parcels[:, 11] == 1.0
        residual_moisture = np.sum(parcels[valid_parcels_mask, 3])
        total_released -= residual_moisture
        print(
            f"  -> Accumulation End: Subtracted {residual_moisture/10**3:.2e} m3 (still in the atmosphere)."
        )

    if total_released < 0:
        total_released = 0.0

    # Output generation
    # ------------------------------
    if singlepoint:
        print("Exporting parcel history...")
        final_active_mask = parcels[:, 11] > 0
        if np.any(final_active_mask):
            parcels[final_active_mask, 11] = 1.0
            parcels_history_list.append(parcels[final_active_mask])

        if len(parcels_history_list) > 0:
            full_history = np.concatenate(parcels_history_list, axis=0)
            df_hist = pd.DataFrame(full_history, columns=cols)
            out_name = os.path.join(
                output_dir, f"UTrack_parcels_{YEAR}_{MONTH:02d}_SP.parquet"
            )
            df_hist.to_parquet(out_name, engine="pyarrow")
            print(f"Parquet saved: {out_name}")

    output_raw = output_map
    output_reference = reference_map_output
    input_reference = reference_map_input

    sum_raw = np.sum(output_raw)
    scaling_factor = 1.0

    if total_released > 0 and sum_raw > 0:
        scaling_factor = sum_raw / total_released
        print(f"Identification Factor (Global Balance): {scaling_factor:.4f}")
    else:
        print("Warning: Identification factor set to 1.0 (Zero release or deposition)")

    # Average TCW over the accumulation period
    total_accumulation_hours = int((target_end - target_start).total_seconds() / 3600.0) + 1
    
    if total_accumulation_hours > 0:
        mean_lagrangian_tcw = tot_lagrangian_tcw / total_accumulation_hours
        mean_era5_tcw = tot_era5_tcw / total_accumulation_hours
        print(f"Averaged TCW over {total_accumulation_hours} accumulation hours.")
    else:
        mean_lagrangian_tcw = tot_lagrangian_tcw
        mean_era5_tcw = tot_era5_tcw

    dir_str = "fwd" if forward_tracking else "bwd"

    if singlepoint:
        type_str = f"SP_{single_point_coords[0]}_{single_point_coords[1]}"
    elif global_mask:
        type_str = "Global"
    else:
        type_str = "Regional"

    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    output_filename = os.path.join(
        output_dir,
        f"UTrack_{YEAR}_{MONTH:02d}_{dir_str}_{type_str}_{config_basename}_interp.nc",
    )

    snap_time_coords = [np.datetime64(d) for d in snapshot_dates]

    ds_out = xr.Dataset(
        {
            "raw_identified": (("lat", "lon"), output_raw),
            "reference_output_ERA5": (("lat", "lon"), output_reference),
            "reference_input_ERA5": (("lat", "lon"), input_reference),
            "mean_lagrangian_tcw": (("lat", "lon"), mean_lagrangian_tcw),
            "mean_era5_tcw": (("lat", "lon"), mean_era5_tcw),
            "snap_lag_tcw": (("snapshot", "lat", "lon"), snap_lag_tcw),
            "snap_era5_tcw": (("snapshot", "lat", "lon"), snap_era5_tcw),
            "snap_era5_flux": (("snapshot", "lat", "lon"), snap_era5_flux),
            "snap_lag_flux": (("snapshot", "lat", "lon"), snap_lag_flux),
        },
        coords={
            "lat": ("lat", m.lats),
            "lon": ("lon", m.lons),
            "snapshot": ("snapshot", snap_time_coords),
        },
    )

    ds_out.lat.attrs["units"] = "degrees_north"
    ds_out.lon.attrs["units"] = "degrees_east"

    ds_out.raw_identified.attrs["long_name"] = (
        "Diagnosed moisture volume"
    )
    ds_out.raw_identified.attrs["units"] = "mm m2"

    ds_out.reference_output_ERA5.attrs["long_name"] = (
        "Total Reference output (ERA5 P if fwd, ET if bwd) diagnosed during identification period"
    )
    ds_out.reference_output_ERA5.attrs["units"] = "mm m2"

    ds_out.reference_input_ERA5.attrs["long_name"] = (
        "Total Reference input (ERA5 ET if fwd, P if bwd) diagnosed during identification period"
    )
    ds_out.reference_input_ERA5.attrs["units"] = "mm m2"

    ds_out.mean_lagrangian_tcw.attrs["long_name"] = (
        "Hourly average Lagrangian Total Column Water"
    )
    ds_out.mean_lagrangian_tcw.attrs["units"] = "mm m2"

    ds_out.mean_era5_tcw.attrs["long_name"] = (
        "Hourly average ERA5 Total Column Water"
    )
    ds_out.mean_era5_tcw.attrs["units"] = "mm m2"

    ds_out.snap_lag_tcw.attrs["long_name"] = "Lagrangian TCW Snapshot"
    ds_out.snap_lag_tcw.attrs["units"] = "mm m2"

    ds_out.snap_era5_tcw.attrs["long_name"] = "ERA5 TCW Snapshot"
    ds_out.snap_era5_tcw.attrs["units"] = "mm m2"

    ds_out.snap_era5_flux.attrs["long_name"] = (
        "Instantaneous ERA5 reference volume (P if fwd, E if bwd)"
    )
    ds_out.snap_era5_flux.attrs["units"] = "mm m2"

    ds_out.snap_lag_flux.attrs["long_name"] = "Instantaneous Model diagnosed volume"
    ds_out.snap_lag_flux.attrs["units"] = "mm m2"

    ds_out.attrs["Simulation_Year"] = YEAR
    ds_out.attrs["Simulation_Month"] = MONTH

    if forward_tracking:
        ds_out.attrs["Release_Period"] = f"{sim_start_date} to {release_limit_date}"
        ds_out.attrs["Tracking_Period"] = f"{sim_start_date} to {simulation_end_date}"
    else:
        ds_out.attrs["Release_Period"] = (
            f"{release_limit_date} to {sim_start_date} (Backward)"
        )
        ds_out.attrs["Tracking_Period"] = (
            f"{simulation_end_date} to {sim_start_date} (Backward)"
        )

    ds_out.attrs["Mode"] = "Single Point" if singlepoint else "Global Mask"
    ds_out.attrs["Total_Net_Released_mm_m2"] = total_released
    ds_out.attrs["Total_Diagnosed_mm_m2"] = sum_raw
    ds_out.attrs["Identification_Factor"] = scaling_factor

    if singlepoint:
        ds_out.attrs["Point_Coords"] = f"{single_point_coords}"

    ds_out.to_netcdf(output_filename)
    print(f"Output NetCDF salvato: {output_filename}")

    # Exporting profiling times
    if len(profiling_records) > 0:
        df_prof = pd.DataFrame(profiling_records)
        prof_filename = os.path.join(
            output_dir,
            f"UTrack_Profiling_{YEAR}_{MONTH:02d}_{dir_str}_{type_str}_{config_basename}_interp.csv",
        )
        df_prof.to_csv(prof_filename, index=False)
        print(f"Dati di profiling salvati in: {prof_filename}")


if __name__ == "__main__":
    main()
