# IBVS Configuration - 2026-01-25_22-13-34

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = 43.6  # pixels
DESIRED_OFFSET_X = 13.0     # pixels (+ = right)
DESIRED_OFFSET_Y = 99.2     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width=640, height=480):
    """Generate desired feature corners."""
    cx_img = width / 2 + 13.0
    cy_img = height / 2 + 99.2
    half = 43.6 / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array([[304.0, 327.0], [356.0, 316.0], [360.0, 352.0], [312.0, 362.0]])

# Robot state at this configuration
JOINT_POSITIONS = [-0.00031327262186253555, 0.16372325844577537, 0.13499027544447315, 1.3944531201767072, -3.983830562768386e-06]
GRIPPER_HEIGHT = 0.128  # meters
