# IBVS Configuration - 2026-01-23_00-00-41

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = 46.1  # pixels
DESIRED_OFFSET_X = 9.5     # pixels (+ = right)
DESIRED_OFFSET_Y = 123.5     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width=640, height=480):
    """Generate desired feature corners."""
    cx_img = width / 2 + 9.5
    cy_img = height / 2 + 123.5
    half = 46.1 / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array([[298.0, 352.0], [354.0, 339.0], [359.0, 375.0], [307.0, 388.0]])

# Robot state at this configuration
JOINT_POSITIONS = [-0.0011555210268736663, 0.20092298918584037, 0.18645496681941182, 1.3279286100258934, 0.0002134380722141342]
GRIPPER_HEIGHT = 0.115  # meters
