# IBVS Configuration - 2026-01-25_22-14-14

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = 44.8  # pixels
DESIRED_OFFSET_X = 18.0     # pixels (+ = right)
DESIRED_OFFSET_Y = 108.2     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width=640, height=480):
    """Generate desired feature corners."""
    cx_img = width / 2 + 18.0
    cy_img = height / 2 + 108.2
    half = 44.8 / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array([[308.0, 334.0], [362.0, 326.0], [366.0, 363.0], [316.0, 370.0]])

# Robot state at this configuration
JOINT_POSITIONS = [-0.0009317734468461022, 0.18869451029497483, 0.1414053504286783, 1.3418885892238845, 9.621998165600564e-05]
GRIPPER_HEIGHT = 0.122  # meters
