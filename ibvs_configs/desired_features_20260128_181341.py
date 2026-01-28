# IBVS Configuration - 2026-01-28_18-13-41

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = 45.9  # pixels
DESIRED_OFFSET_X = 14.0     # pixels (+ = right)
DESIRED_OFFSET_Y = 121.2     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width=640, height=480):
    """Generate desired feature corners."""
    cx_img = width / 2 + 14.0
    cy_img = height / 2 + 121.2
    half = 45.9 / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array([[303.0, 348.0], [359.0, 338.0], [363.0, 375.0], [311.0, 384.0]])

# Robot state at this configuration
JOINT_POSITIONS = [-0.00022104617209665485, 0.19097110093855715, 0.19295568597479248, 1.3056507239687865, 6.964249398201361e-05]
GRIPPER_HEIGHT = 0.115  # meters
