# IBVS Configuration - 2026-01-22_14-26-03

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = 44.7  # pixels
DESIRED_OFFSET_X = -14.0     # pixels (+ = right)
DESIRED_OFFSET_Y = 111.2     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width=640, height=480):
    """Generate desired feature corners."""
    cx_img = width / 2 + -14.0
    cy_img = height / 2 + 111.2
    half = 44.7 / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array([[275.0, 339.0], [329.0, 328.0], [335.0, 364.0], [285.0, 374.0]])

# Robot state at this configuration
JOINT_POSITIONS = [-5.26317566901881e-07, 0.13078370465770986, 0.220390241579916, 1.3259208124383934, 9.285309168433019e-07]
GRIPPER_HEIGHT = 0.122  # meters
