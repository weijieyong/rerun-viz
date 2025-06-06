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
from collections import deque
from typing import List, Optional
import numpy as np

import rerun as rr
from rerun import blueprint as rrb

# Axis names and colors for force and torque
XYZ_AXIS_NAMES = ["x", "y", "z"]
XYZ_AXIS_COLORS = [[231, 76, 60], [39, 174, 96], [52, 120, 219]]  # Red, Green, Blue

# Queue for streaming data
data_queue = Queue()

# Smoothing configuration
class SmoothingConfig:
    """Configuration for data smoothing."""
    def __init__(self, method: str = "moving_average", window_size: int = 5, alpha: float = 0.3):
        self.method = method  # "moving_average", "exponential", "savgol"
        self.window_size = window_size
        self.alpha = alpha  # For exponential smoothing

# Global smoothing state
smoothing_config = SmoothingConfig()

class DataSmoother:
    """Handles smoothing of force and torque data."""
    
    def __init__(self, config: SmoothingConfig):
        self.config = config
        self.force_history = deque(maxlen=config.window_size)
        self.torque_history = deque(maxlen=config.window_size)
        self.force_ema = None
        self.torque_ema = None
    
    def smooth_data(self, force: List[float], torque: List[float]) -> tuple[List[float], List[float]]:
        """Apply smoothing to force and torque data."""
        self.force_history.append(force)
        self.torque_history.append(torque)
        
        if self.config.method == "moving_average":
            return self._moving_average()
        elif self.config.method == "exponential":
            return self._exponential_smoothing(force, torque)
        elif self.config.method == "savgol":
            return self._savitzky_golay()
        else:
            return force, torque  # No smoothing
    
    def _moving_average(self) -> tuple[List[float], List[float]]:
        """Apply moving average smoothing."""
        if len(self.force_history) == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        
        # Calculate average for each axis
        force_avg = [0.0, 0.0, 0.0]
        torque_avg = [0.0, 0.0, 0.0]
        
        for i in range(3):
            force_avg[i] = sum(f[i] for f in self.force_history) / len(self.force_history)
            torque_avg[i] = sum(t[i] for t in self.torque_history) / len(self.torque_history)
        
        return force_avg, torque_avg
    
    def _exponential_smoothing(self, force: List[float], torque: List[float]) -> tuple[List[float], List[float]]:
        """Apply exponential moving average smoothing."""
        if self.force_ema is None:
            self.force_ema = force[:]
            self.torque_ema = torque[:]
            return force, torque
        
        # Apply EMA: new_value = alpha * current + (1-alpha) * previous
        alpha = self.config.alpha
        for i in range(3):
            self.force_ema[i] = alpha * force[i] + (1 - alpha) * self.force_ema[i]
            self.torque_ema[i] = alpha * torque[i] + (1 - alpha) * self.torque_ema[i]
        
        return self.force_ema[:], self.torque_ema[:]
    
    def _savitzky_golay(self) -> tuple[List[float], List[float]]:
        """Apply Savitzky-Golay filter smoothing."""
        try:
            from scipy.signal import savgol_filter

            if len(self.force_history) < 3:
                # Not enough data for Savitzky-Golay, use moving average
                return self._moving_average()
            
            # Convert to numpy arrays for filtering
            force_array = np.array(list(self.force_history))
            torque_array = np.array(list(self.torque_history))
            
            # Apply Savitzky-Golay filter
            window_length = min(len(self.force_history), 5)
            if window_length % 2 == 0:
                window_length -= 1  # Must be odd
            
            if window_length >= 3:
                force_smooth = savgol_filter(force_array, window_length, 2, axis=0)
                torque_smooth = savgol_filter(torque_array, window_length, 2, axis=0)
                return force_smooth[-1].tolist(), torque_smooth[-1].tolist()
            else:
                return self._moving_average()
                
        except ImportError:
            print("Warning: scipy not available, falling back to moving average")
            return self._moving_average()

# Global smoother instance
data_smoother = DataSmoother(smoothing_config)

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
    parser.add_argument(
        "--smoothing",
        type=str,
        choices=["none", "moving_average", "exponential", "savgol"],
        default="moving_average",
        help="Smoothing method to apply to the data (default: moving_average)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for moving average and Savitzky-Golay smoothing (default: 5)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.3,
        help="Alpha parameter for exponential smoothing (0-1, default: 0.3)",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    
    # Configure smoothing based on arguments
    global smoothing_config, data_smoother
    if args.smoothing != "none":
        smoothing_config = SmoothingConfig(
            method=args.smoothing,
            window_size=args.smoothing_window,
            alpha=args.smoothing_alpha
        )
        data_smoother = DataSmoother(smoothing_config)
        print(f"Smoothing enabled: {args.smoothing} (window={args.smoothing_window}, alpha={args.smoothing_alpha})")
    else:
        print("Smoothing disabled")
    
    # Create blueprint with force and torque plots (both raw and smoothed)
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
                "/force/raw": rr.SeriesLines.from_fields(
                    names=XYZ_AXIS_NAMES, colors=[[200, 200, 200], [180, 180, 180], [160, 160, 160]]
                ),
                "/force/smoothed": rr.SeriesLines.from_fields(
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
                "/torque/raw": rr.SeriesLines.from_fields(
                    names=XYZ_AXIS_NAMES, colors=[[200, 200, 200], [180, 180, 180], [160, 160, 160]]
                ),
                "/torque/smoothed": rr.SeriesLines.from_fields(
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

            # Get raw force and torque data
            force_raw = data["force"]
            torque_raw = data["torque"]

            # Apply smoothing if enabled
            if smoothing_config.method != "none":
                force_smooth, torque_smooth = data_smoother.smooth_data(force_raw, torque_raw)
                
                # Log both raw and smoothed data
                rr.log("/force/raw", rr.Scalars(force_raw))
                rr.log("/force/smoothed", rr.Scalars(force_smooth))
                rr.log("/torque/raw", rr.Scalars(torque_raw))
                rr.log("/torque/smoothed", rr.Scalars(torque_smooth))
                
                # Use smoothed data for statistics
                force = force_smooth
                torque = torque_smooth
            else:
                # Log only raw data when smoothing is disabled
                rr.log("/force/raw", rr.Scalars(force_raw))
                rr.log("/torque/raw", rr.Scalars(torque_raw))
                
                # Use raw data for statistics
                force = force_raw
                torque = torque_raw

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
