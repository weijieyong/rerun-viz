#!/usr/bin/env python3
"""
Simulates force-torque sensor data streaming for testing the ft_sensor_stream.py script.

Usage:
    python simulate_ft_stream.py | python ft_sensor_stream.py
"""

import time
import random
import math


def generate_ft_data():
    """Generate simulated force-torque data."""
    t = 0
    while True:
        # Simulate some realistic force patterns
        fx = 0.1 * math.sin(t * 0.1) + random.uniform(-0.05, 0.05)
        fy = 0.15 * math.cos(t * 0.08) + random.uniform(-0.08, 0.08)
        fz = -0.1 + 0.05 * math.sin(t * 0.05) + random.uniform(-0.03, 0.03)

        # Simulate some torque (usually smaller values)
        tx = 0.001 * math.sin(t * 0.12) + random.uniform(-0.002, 0.002)
        ty = 0.001 * math.cos(t * 0.07) + random.uniform(-0.002, 0.002)
        tz = random.uniform(-0.001, 0.001)

        # Simulate varying frequency around 100Hz
        frequency = 100 + random.randint(-2, 0)
        # Format output to match expected format
        print(
            f"F: {frequency:3d}Hz | "
            f"Force: [{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] N | "
            f"Torque: [{tx:6.3f}, {ty:6.3f}, {tz:6.3f}] Nm",
            flush=True,
        )
        # print(f"F: {frequency:3d}Hz | Force: [{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] N | Torque: [{tx:6.3f}, {ty:6.3f}, {tz:6.3f}] Nm", flush=True)

        # Sleep to simulate real-time streaming (approximately 100Hz)
        time.sleep(0.01)
        t += 1


if __name__ == "__main__":
    try:
        generate_ft_data()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
