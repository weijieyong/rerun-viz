#!/usr/bin/env python3
"""
Demonstrates how to visualize Force-Torque sensor data using Rerun.

This script reads real-time streaming force-torque data and displays it using
TimeSeriesView with separate plots for force and torque components.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from queue import Queue
from datetime import datetime

import rerun as rr
from rerun import blueprint as rrb

# Axis names and colors for force and torque
XYZ_AXIS_NAMES = ["x", "y", "z"]
XYZ_AXIS_COLORS = [[231, 76, 60], [39, 174, 96], [52, 120, 219]]  # Red, Green, Blue

# Queue for streaming data
data_queue = Queue()

# Pattern to parse the streaming data
# F:  99Hz | Force: [   0.02,    0.14,   -0.11] N | Torque: [ 0.000, -0.000, -0.001] Nm
DATA_PATTERN = re.compile(
    r"F:\s*(\d+)Hz\s*\|\s*Force:\s*\[\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\]\s*N\s*\|\s*Torque:\s*\[\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\]\s*Nm"
)


def parse_streaming_data(line: str) -> dict | None:
    """Parse a line of streaming data into structured format."""
    match = DATA_PATTERN.match(line.strip())
    if match:
        frequency, fx, fy, fz, tx, ty, tz = match.groups()
        return {
            "timestamp": datetime.now(),
            "frequency": float(frequency),
            "force": [float(fx), float(fy), float(fz)],
            "torque": [float(tx), float(ty), float(tz)],
        }
    return None


def stream_reader():
    """Read streaming data from stdin and put it in the queue."""
    print("Starting stream reader... Reading from stdin")
    print(
        "Expected format: F:  99Hz | Force: [   0.02,    0.14,   -0.11] N | Torque: [ 0.000, -0.000, -0.001] Nm"
    )

    try:
        for line in sys.stdin:
            if line.strip():
                data = parse_streaming_data(line)
                if data:
                    data_queue.put(data)
                else:
                    print(f"Warning: Could not parse line: {line.strip()}")
    except KeyboardInterrupt:
        print("Stream reader stopped by user")
    except Exception as e:
        print(f"Error in stream reader: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizes real-time Force-Torque sensor data using the Rerun SDK."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float("inf"),
        help="If specified, limits the duration of logging in seconds",
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
        help="Start web viewer server",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    # Create blueprint with force and torque plots
    blueprint = rrb.Vertical(
        rrb.TimeSeriesView(
            origin="force",
            name="Force (N)",
            time_ranges=[
                # Sliding window showing last 100 samples for real-time visualization
                rrb.VisibleTimeRange(
                    "timestamp",
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-10.0),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                ),
            ],
            overrides={
                "/force": rr.SeriesLines.from_fields(
                    names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                ),
            },
        ),
        rrb.TimeSeriesView(
            origin="torque",
            name="Torque (Nm)",
            time_ranges=[
                # Sliding window showing last 100 samples for real-time visualization
                rrb.VisibleTimeRange(
                    "timestamp",
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-10.0),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                ),
            ],
            overrides={
                "/torque": rr.SeriesLines.from_fields(
                    names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                ),
            },
        ),
    )

    rr.script_setup(args, "rerun_ft_sensor_stream", default_blueprint=blueprint)

    # Set up web server if requested
    web_url = None
    if args.serve_web:
        # Start gRPC server
        grpc_url = rr.serve_grpc()
        print(f"gRPC server started at: {grpc_url}")

        # Start web viewer
        rr.serve_web_viewer(web_port=args.web_port, connect_to=grpc_url)
        web_url = f"http://localhost:{args.web_port}"
        print(f"Web viewer started at: {web_url}")
        print(f"Open your browser to view the live data stream.")

    # Start the stream reader in a separate thread
    reader_thread = threading.Thread(target=stream_reader, daemon=True)
    reader_thread.start()

    try:
        _log_streaming_data(args.duration)
    except KeyboardInterrupt:
        print("Logging stopped by user")
    finally:
        rr.script_teardown(args)


def _log_streaming_data(max_duration_sec: float) -> None:
    """Log real-time force-torque sensor data to Rerun."""
    print("Starting real-time data logging...")
    print("Waiting for streaming data...")

    start_time = time.time()
    sample_count = 0

    # Statistics tracking
    force_stats = {"min": [float("inf")] * 3, "max": [float("-inf")] * 3}
    torque_stats = {"min": [float("inf")] * 3, "max": [float("-inf")] * 3}

    while True:
        # Check if we've exceeded the maximum duration
        if time.time() - start_time > max_duration_sec:
            print(f"Reached maximum duration of {max_duration_sec} seconds")
            break

        try:
            # Get data from queue with timeout
            data = data_queue.get(timeout=1.0)

            # Set the timestamp
            timestamp = data["timestamp"]
            rr.set_time("timestamp", timestamp=timestamp)

            # Log force data
            force = data["force"]
            rr.log("/force", rr.Scalars(force))

            # Log torque data
            torque = data["torque"]
            rr.log("/torque", rr.Scalars(torque))

            # Log frequency data
            frequency = data["frequency"]
            rr.log("/frequency", rr.Scalars([frequency]))

            # Update statistics
            for i in range(3):
                force_stats["min"][i] = min(force_stats["min"][i], force[i])
                force_stats["max"][i] = max(force_stats["max"][i], force[i])
                torque_stats["min"][i] = min(torque_stats["min"][i], torque[i])
                torque_stats["max"][i] = max(torque_stats["max"][i], torque[i])

            sample_count += 1

            # Print periodic updates
            # if sample_count % 100 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"Logged {sample_count} samples in {elapsed:.1f}s (avg: {sample_count/elapsed:.1f} Hz)")
            #     print(f"Force ranges - X: [{force_stats['min'][0]:.3f}, {force_stats['max'][0]:.3f}] N")
            #     print(f"Torque ranges - X: [{torque_stats['min'][0]:.3f}, {torque_stats['max'][0]:.3f}] Nm")

        except Exception as e:
            if "timeout" in str(e).lower():
                # No data received in timeout period
                continue
            else:
                print(f"Error processing data: {e}")
                break

    elapsed = time.time() - start_time
    print(f"\nLogging completed!")
    print(f"Total samples: {sample_count}")
    print(f"Total time: {elapsed:.1f}s")
    if sample_count > 0:
        print(f"Average frequency: {sample_count / elapsed:.1f} Hz")
        print(f"Final Force ranges:")
        print(f"  X: [{force_stats['min'][0]:.3f}, {force_stats['max'][0]:.3f}] N")
        print(f"  Y: [{force_stats['min'][1]:.3f}, {force_stats['max'][1]:.3f}] N")
        print(f"  Z: [{force_stats['min'][2]:.3f}, {force_stats['max'][2]:.3f}] N")
        print(f"Final Torque ranges:")
        print(f"  X: [{torque_stats['min'][0]:.3f}, {torque_stats['max'][0]:.3f}] Nm")
        print(f"  Y: [{torque_stats['min'][1]:.3f}, {torque_stats['max'][1]:.3f}] Nm")
        print(f"  Z: [{torque_stats['min'][2]:.3f}, {torque_stats['max'][2]:.3f}] Nm")


if __name__ == "__main__":
    main()
