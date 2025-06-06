#!/usr/bin/env python3
"""Test script to verify the regex pattern works correctly."""

import re

# Updated pattern
DATA_PATTERN = re.compile(
    r"F:\s*(\d+)Hz\s*\|\s*Force:\s*\[\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\]\s*N\s*\|\s*Torque:\s*\[\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\]\s*Nm"
)

# Test samples from the simulator output
test_lines = [
    "F: 100Hz | Force: [  -0.05,    0.21,   -0.12] N | Torque: [ 0.001,  0.001, -0.000] Nm",
    "F:  98Hz | Force: [  -0.04,    0.07,   -0.07] N | Torque: [ 0.000,  0.000, -0.000] Nm",
    "F:  99Hz | Force: [  -0.02,    0.15,   -0.11] N | Torque: [-0.001,  0.001,  0.001] Nm",
    "F: 100Hz | Force: [   0.05,    0.21,   -0.12] N | Torque: [-0.001,  0.003,  0.000] Nm",
]


def parse_streaming_data(line: str) -> dict | None:
    """Parse a line of streaming data into structured format."""
    match = DATA_PATTERN.match(line.strip())
    if match:
        frequency, fx, fy, fz, tx, ty, tz = match.groups()
        return {
            "frequency": float(frequency),
            "force": [float(fx), float(fy), float(fz)],
            "torque": [float(tx), float(ty), float(tz)],
        }
    return None


# Test the regex
for i, line in enumerate(test_lines):
    print(f"Test {i + 1}: {line}")
    result = parse_streaming_data(line)
    if result:
        print(f"  ✓ Parsed successfully:")
        print(f"    Frequency: {result['frequency']} Hz")
        print(f"    Force: {result['force']} N")
        print(f"    Torque: {result['torque']} Nm")
    else:
        print("  ✗ Failed to parse")
    print()
