# IBVS Configuration - 2026-01-26_00-12-59

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = 38.7  # pixels
DESIRED_OFFSET_X = -1.5     # pixels (+ = right)
DESIRED_OFFSET_Y = 93.5     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width=640, height=480):
    """Generate desired feature corners."""
    cx_img = width / 2 + -1.5
    cy_img = height / 2 + 93.5
    half = 38.7 / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array([[295.0, 315.0], [342.0, 320.0], [340.0, 352.0], [297.0, 347.0]])

# Robot state at this configuration
JOINT_POSITIONS = [-0.15294902556709275, 0.16775851144270007, 0.08734280836295108, 1.1461596313191924, 3.7566772217639813e-07]
GRIPPER_HEIGHT = 0.134  # meters
