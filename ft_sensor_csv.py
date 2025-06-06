#!/usr/bin/env python3
"""
Demonstrates how to visualize Force-Torque sensor data using Rerun.

This script reads force-torque data from a CSV file and displays it using
TimeSeriesView with separate plots for force and torque components.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import rerun as rr
from rerun import blueprint as rrb

# Data file path
DATA_FILE = Path(__file__).parent / "data" / "sensor_data.csv"

# Axis names and colors for force and torque
XYZ_AXIS_NAMES = ["x", "y", "z"]
XYZ_AXIS_COLORS = [[231, 76, 60], [39, 174, 96], [52, 120, 219]]  # Red, Green, Blue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizes Force-Torque sensor data using the Rerun SDK."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=float("inf"),
        help="If specified, limits the number of seconds logged",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9999,
        help="Port for the web viewer (default: 9999)",
    )
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="Start web server to serve the visualization",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()  # Create blueprint with force and torque plots
    blueprint = rrb.Vertical(
        rrb.TimeSeriesView(
            origin="force",
            name="Force (N)",
            overrides={
                "force/x": rr.SeriesLine(color=[231, 76, 60], name="fx"),
                "force/y": rr.SeriesLine(color=[39, 174, 96], name="fy"),
                "force/z": rr.SeriesLine(color=[52, 120, 219], name="fz"),
            },
        ),
        rrb.TimeSeriesView(
            origin="torque",
            name="Torque (Nm)",
            overrides={
                "torque/x": rr.SeriesLine(color=[231, 76, 60], name="tx"),
                "torque/y": rr.SeriesLine(color=[39, 174, 96], name="ty"),
                "torque/z": rr.SeriesLine(color=[52, 120, 219], name="tz"),
            },
        ),
        row_shares=[0.5, 0.5],
    )

    if args.serve_web:
        # Initialize Rerun for web serving
        rr.init("rerun_ft_sensor_web", default_blueprint=blueprint)
        print("Rerun initialized for web serving")

        # Start the gRPC server
        url = rr.serve_grpc()
        print(f"gRPC server started at: {url}")

        # Log the data
        _log_ft_data(args.seconds)

        # Start the web viewer
        rr.serve_web_viewer(web_port=args.web_port, connect_to=url)
        print(f"Web viewer started at http://localhost:{args.web_port}")

        try:
            print("Server running... Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")
    else:
        # Use standard script setup for desktop viewer
        rr.script_setup(args, "rerun_ft_sensor", default_blueprint=blueprint)
        _log_ft_data(args.seconds)
        rr.script_teardown(args)


def _log_ft_data(max_time_sec: float) -> None:
    """Log force-torque sensor data to Rerun."""
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    # Read the CSV data
    ft_data = pd.read_csv(DATA_FILE)

    print(f"Loaded {len(ft_data)} samples from {DATA_FILE}")
    print(f"Data columns: {list(ft_data.columns)}")

    # Filter data based on time limit
    if max_time_sec != float("inf"):
        start_time = ft_data["timestamp"].iloc[0]
        max_timestamp = start_time + max_time_sec
        ft_data = ft_data[ft_data["timestamp"] <= max_timestamp]
        print(f"Filtered to {len(ft_data)} samples within {max_time_sec} seconds")

    # Convert timestamps to datetime format
    timestamps = pd.to_datetime(ft_data["timestamp"], unit="s")
    # times = rr.TimeColumn("timestamp", timestamp=timestamps)    # Log force data (fx, fy, fz) - separate components
    force_data = ft_data[["fx", "fy", "fz"]].to_numpy()
    for i, (timestamp, force_values) in enumerate(zip(timestamps, force_data)):
        rr.set_time_nanos("timestamp", int(timestamp.timestamp() * 1e9))
        rr.log("force/x", rr.Scalar(force_values[0]))
        rr.log("force/y", rr.Scalar(force_values[1]))
        rr.log("force/z", rr.Scalar(force_values[2]))

    # Log torque data (tx, ty, tz) - separate components
    torque_data = ft_data[["tx", "ty", "tz"]].to_numpy()
    for i, (timestamp, torque_values) in enumerate(zip(timestamps, torque_data)):
        rr.set_time_nanos("timestamp", int(timestamp.timestamp() * 1e9))
        rr.log("torque/x", rr.Scalar(torque_values[0]))
        rr.log("torque/y", rr.Scalar(torque_values[1]))
        rr.log("torque/z", rr.Scalar(torque_values[2]))

    print("Force-Torque data logged successfully!")
    print(
        f"Force range - X: [{force_data[:, 0].min():.3f}, {force_data[:, 0].max():.3f}] N"
    )
    print(
        f"Force range - Y: [{force_data[:, 1].min():.3f}, {force_data[:, 1].max():.3f}] N"
    )
    print(
        f"Force range - Z: [{force_data[:, 2].min():.3f}, {force_data[:, 2].max():.3f}] N"
    )
    print(
        f"Torque range - X: [{torque_data[:, 0].min():.3f}, {torque_data[:, 0].max():.3f}] Nm"
    )
    print(
        f"Torque range - Y: [{torque_data[:, 1].min():.3f}, {torque_data[:, 1].max():.3f}] Nm"
    )
    print(
        f"Torque range - Z: [{torque_data[:, 2].min():.3f}, {torque_data[:, 2].max():.3f}] Nm"
    )


if __name__ == "__main__":
    main()
