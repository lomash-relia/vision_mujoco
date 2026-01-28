# Do not use this code. Use v2 python script.


# """
# Image-Based Visual Servoing (IBVS) for SO101 Robot with Eye-in-Hand Camera

# Implementation based on:
# - Chaumette & Hutchinson (2006) "Visual Servo Control I: Basic Approaches" IEEE RAM
# - Corke (2017) "Robotics, Vision and Control" Chapter 15

# This implementation correctly handles:
# 1. Interaction matrix with proper focal length scaling
# 2. Camera-to-end-effector frame transformation
# 3. Position control integration from velocity commands
# 4. Consistent ArUco corner ordering
# 5. MuJoCo depth buffer conversion
# """

# import mujoco
# import mujoco.viewer
# import numpy as np
# import cv2
# import cv2.aruco as aruco
# import os
# import time
# import xml.etree.ElementTree as ET
# from pathlib import Path
# from enum import IntEnum

# # ==========================================
# # CONFIGURATION
# # ==========================================
# ROBOT_XML = r"model\so101_new_calib.xml"
# # Scene selection is done after feature flags are defined (see USE_CUBE_SCENE)
# SCENE_XML_FLAT = r"model\scene_ibvs.xml"     # Original flat ArUco marker
# SCENE_XML_CUBE = r"model\scene_ibvs_cube.xml"  # Cube with ArUco on top
# TEMP_ROBOT_XML = r"model\so101_with_cam.xml"
# TEMP_SCENE_XML = r"model\scene_ibvs_with_cam.xml"
# IMG_DIR = "img"

# # Camera Parameters
# CAM_NAME = "eye_in_hand_cam"
# CAM_MOUNT_BODY = "gripper"
# CAM_POS = "0.02 -0.12 0.03"    # Moved further from gripper (-0.09 -> -0.12) and slightly up
# CAM_QUAT = "0.05 0 -0.26 0.964"  # Slight tilt away from arm
# CAM_FOVY = 70  # degrees
# CAM_WIDTH = 640
# CAM_HEIGHT = 480

# # IBVS Parameters
# IBVS_LAMBDA = 0.2          # Control gain (conservative)
# IBVS_DT = 0.02             # Control timestep
# MAX_JOINT_VEL = 0.5        # Maximum joint velocity (rad/s)
# CONVERGENCE_THRESHOLD = 5.0  # Pixel error threshold for convergence
# DESIRED_MARKER_SIZE = 70   # Smaller size = arm stops further from marker
# DESIRED_OFFSET_Y = -40      # Offset marker target downward in image (reduced to shift target up)

# # Orientation IBVS Parameters
# MARKER_SIZE = 0.07                  # Physical marker size in meters
# IBVS_LAMBDA_ORIENT = 0.3            # Orientation gain (higher than position for faster alignment)
# ENABLE_ORIENTATION_CONTROL = True   # Toggle orientation control on/off

# # ==========================================
# # FEATURE FLAGS (Toggle new features)
# # ==========================================
# ENABLE_SEARCH = True                # Enable automatic marker search
# ENABLE_GRASPABILITY_CHECK = True    # Enable graspability validation before grasp
# ENABLE_GRASP_EXECUTION = True       # Enable full grasp pipeline
# ENABLE_JOINT_WEIGHTING = True       # Enable singularity-aware joint weighting

# # Scene Selection
# USE_CUBE_SCENE = True               # True = cube with ArUco, False = flat marker (original)
# SCENE_XML = SCENE_XML_CUBE if USE_CUBE_SCENE else SCENE_XML_FLAT

# # ==========================================
# # SEARCH PARAMETERS
# # ==========================================
# SEARCH_PAN_RANGE = [-0.5, 0.5]      # Joint 1 sweep range (±30° in radians)
# SEARCH_PAN_STEP = 0.26              # 15° steps in radians
# SEARCH_PITCH_RANGE = [-0.35, 0.35]  # Joint 4 sweep range (±20° in radians)
# SEARCH_PITCH_STEP = 0.17            # 10° steps in radians
# SEARCH_DWELL_FRAMES = 10            # Frames to wait at each search position
# MARKER_LOST_THRESHOLD = 30          # Frames without detection before starting search
# SEARCH_LIFT_ANGLE = 1.2             # Shoulder lift angle during search (higher = arm up)
# SEARCH_ELBOW_ANGLE = 1.3            # Elbow angle during search (higher = more bent)
# SEARCH_WRIST_ANGLE = 0.6            # Wrist flex angle during search (camera looks down)

# # ==========================================
# # GRASPABILITY PARAMETERS
# # ==========================================
# MARKER_STABILITY_FRAMES = 5         # Consecutive detections required before grasp
# MAX_GRASP_DISTANCE = 0.50           # Maximum distance from base (meters)
# MIN_MARKER_HEIGHT = 0.02            # Minimum marker Z-height (meters)
# GRIPPER_FINGER_LENGTH = 0.06        # Gripper finger length (meters)
# MIN_MANIPULABILITY = 0.01           # Yoshikawa manipulability threshold
# PRE_GRASP_HEIGHT = 0.10             # Pre-grasp pose height above marker (meters)

# # ==========================================
# # GRASP EXECUTION PARAMETERS
# # ==========================================
# CUBE_SIZE = 0.04                    # 4cm cube
# ORIENTATION_ALIGN_THRESHOLD = 5.0   # Degrees tolerance for gripper alignment
# DESCENT_SPEED = 0.005               # Vertical descent speed (m/step)
# GRASP_HEIGHT_OFFSET = 0.020         # Final gripper height above ground
# GRIPPER_CLOSE_POSITION = 1.0        # Gripper closed position (radians) for 4cm cube
# GRIPPER_OPEN_POSITION = 0.0         # Gripper open position (radians)
# GRIPPER_CLOSE_SPEED = 0.05          # Gripper closing speed (rad/step)
# LIFT_HEIGHT = 0.05                  # Lift height for grasp verification (meters)
# AUTO_RETRY_ON_FAILURE = False       # Manual retry mode (wait for 'G' key)
# PRE_GRASP_DISTANCE = 0.15           # Depth threshold to trigger grasp (meters)

# # ==========================================
# # JOINT WEIGHTING PARAMETERS
# # ==========================================
# WRIST_SINGULARITY_ZONE = 20.0       # Degrees from 0° or 180° considered near-singular
# BASE_WEIGHT_FINE_THRESHOLD = 50.0   # Pixel error threshold for reduced base weight
# MIN_MANIPULABILITY_WARNING = 0.05   # Log warning if manipulability below this

# # ==========================================
# # 1. CAMERA INTRINSICS
# # ==========================================
# def compute_camera_intrinsics(fovy, width, height):
#     """Compute camera intrinsic parameters from FOV and resolution."""
#     fy = height / (2 * np.tan(np.radians(fovy / 2)))
#     fx = fy  # Square pixels assumed
#     cx = width / 2
#     cy = height / 2
#     K = np.array([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1]
#     ])
#     return K, fx, fy, cx, cy

# # Global camera intrinsics
# K, fx, fy, cx, cy = compute_camera_intrinsics(CAM_FOVY, CAM_WIDTH, CAM_HEIGHT)

# # ==========================================
# # 2. FRAME TRANSFORMATION UTILITIES
# # ==========================================
# def quat_to_rotation_matrix(quat):
#     """
#     Convert quaternion to 3x3 rotation matrix.

#     Args:
#         quat: [w, x, y, z] quaternion (scalar-first convention)

#     Returns:
#         R: 3x3 rotation matrix
#     """
#     w, x, y, z = quat

#     R = np.array([
#         [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
#         [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
#         [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
#     ])
#     return R

# def get_camera_to_ee_transform():
#     """
#     Get the transformation from camera frame to end-effector (gripper) frame.
#     Based on camera mounting configuration in XML.

#     Returns:
#         R_cam_to_ee: 3x3 rotation matrix
#         p_cam_in_ee: 3x1 position vector of camera in EE frame
#     """
#     # Parse camera position from config
#     pos_parts = CAM_POS.split()
#     p_cam_in_ee = np.array([float(p) for p in pos_parts])

#     # Parse camera quaternion from config (MuJoCo XML uses [x,y,z,w])
#     quat_parts = CAM_QUAT.split()
#     quat_xyzw = np.array([float(q) for q in quat_parts])
#     # Convert to [w,x,y,z] for our function
#     quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

#     R_cam_to_ee = quat_to_rotation_matrix(quat_wxyz)

#     return R_cam_to_ee, p_cam_in_ee

# def transform_velocity_cam_to_ee(v_cam, R_cam_to_ee, p_cam_in_ee):
#     """
#     Transform 6D velocity from camera frame to end-effector frame.

#     Uses the adjoint transformation for rigid body velocities.

#     Args:
#         v_cam: [vx, vy, vz, wx, wy, wz] velocity in camera frame
#         R_cam_to_ee: 3x3 rotation from camera to EE frame
#         p_cam_in_ee: 3x1 position of camera origin in EE frame

#     Returns:
#         v_ee: [vx, vy, vz, wx, wy, wz] velocity in EE frame
#     """
#     v_linear_cam = v_cam[:3]
#     omega_cam = v_cam[3:]

#     # Rotate velocities to EE frame
#     v_linear_ee = R_cam_to_ee @ v_linear_cam
#     omega_ee = R_cam_to_ee @ omega_cam

#     # Account for lever arm effect: v_ee = v_cam + omega x p
#     v_linear_ee = v_linear_ee + np.cross(omega_ee, p_cam_in_ee)

#     return np.concatenate([v_linear_ee, omega_ee])

# # ==========================================
# # 3. ARUCO DETECTION
# # ==========================================
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# aruco_params = aruco.DetectorParameters()
# aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# def detect_aruco_corners(image):
#     """
#     Detect ArUco marker and return 4 corner points in consistent order.

#     Corners are sorted by angle from centroid (counter-clockwise from right)
#     to ensure consistent ordering regardless of marker orientation.

#     Returns:
#         corners: (4, 2) array of corner positions or None if not detected
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     corners, ids, rejected = aruco_detector.detectMarkers(gray)

#     if ids is None or len(ids) == 0:
#         return None

#     # Get corners of first detected marker
#     marker_corners = corners[0][0]  # Shape: (4, 2)

#     # Sort corners consistently by angle from centroid
#     centroid = np.mean(marker_corners, axis=0)
#     angles = np.arctan2(marker_corners[:, 1] - centroid[1],
#                         marker_corners[:, 0] - centroid[0])
#     sorted_indices = np.argsort(angles)

#     return marker_corners[sorted_indices]

# def estimate_marker_pose(corners, marker_size=MARKER_SIZE):
#     """
#     Estimate marker pose using PnP solver.

#     Args:
#         corners: (4, 2) array of image corner positions (sorted by angle)
#         marker_size: Physical size of marker in meters

#     Returns:
#         R_marker_to_cam: (3, 3) rotation matrix from marker to camera frame
#         t_marker_in_cam: (3,) translation vector
#         success: bool indicating if pose estimation succeeded
#     """
#     half = marker_size / 2

#     # 3D object points in marker frame (Z=0 plane)
#     # Order must match corner ordering from detect_aruco_corners (sorted by angle)
#     # Angles: -3pi/4 (top-left), -pi/4 (top-right), +pi/4 (bottom-right), +3pi/4 (bottom-left)
#     object_points = np.array([
#         [-half, -half, 0],  # top-left
#         [ half, -half, 0],  # top-right
#         [ half,  half, 0],  # bottom-right
#         [-half,  half, 0],  # bottom-left
#     ], dtype=np.float32)

#     # Image points
#     image_points = corners.astype(np.float32)

#     # Camera intrinsic matrix
#     camera_matrix = K.astype(np.float32)
#     dist_coeffs = np.zeros(4, dtype=np.float32)

#     # Solve PnP - use IPPE_SQUARE for planar square markers
#     success, rvec, tvec = cv2.solvePnP(
#         object_points, image_points, camera_matrix, dist_coeffs,
#         flags=cv2.SOLVEPNP_IPPE_SQUARE
#     )

#     if not success:
#         return None, None, False

#     # Convert rotation vector to matrix
#     R_marker_to_cam, _ = cv2.Rodrigues(rvec)
#     t_marker_in_cam = tvec.flatten()

#     return R_marker_to_cam, t_marker_in_cam, True

# def draw_aruco_detection(image, corners, desired_corners=None):
#     """Draw detected corners and desired corners on image."""
#     img_display = image.copy()

#     if corners is not None:
#         # Draw detected corners in green
#         for i, corner in enumerate(corners):
#             cv2.circle(img_display, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
#             cv2.putText(img_display, str(i), (int(corner[0])+10, int(corner[1])),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         # Draw polygon connecting corners
#         pts = corners.astype(np.int32).reshape((-1, 1, 2))
#         cv2.polylines(img_display, [pts], True, (0, 255, 0), 2)

#     if desired_corners is not None:
#         # Draw desired corners in red
#         for i, corner in enumerate(desired_corners):
#             cv2.circle(img_display, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), 2)
#             cv2.putText(img_display, str(i), (int(corner[0])-15, int(corner[1])),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#         pts = desired_corners.astype(np.int32).reshape((-1, 1, 2))
#         cv2.polylines(img_display, [pts], True, (0, 0, 255), 2)

#         # Draw error lines from current to desired
#         if corners is not None:
#             for curr, des in zip(corners, desired_corners):
#                 cv2.line(img_display,
#                         (int(curr[0]), int(curr[1])),
#                         (int(des[0]), int(des[1])),
#                         (255, 255, 0), 1)

#     return img_display

# # ==========================================
# # 4. IMAGE JACOBIAN (INTERACTION MATRIX)
# # ==========================================
# def compute_image_jacobian(points, depths):
#     """
#     Compute the image Jacobian (interaction matrix) for point features.

#     Following Chaumette & Hutchinson formulation with proper focal length scaling.
#     The interaction matrix relates image feature velocity to camera velocity:
#         s_dot = L * v_c

#     Args:
#         points: (n, 2) array of image points (u, v) in pixels
#         depths: (n,) array of depth values Z for each point

#     Returns:
#         L: (2n, 6) interaction matrix
#     """
#     L = []
#     for i, (u, v) in enumerate(points):
#         Z = depths[i]
#         if Z <= 0:
#             Z = 0.1  # Prevent division by zero

#         # Normalized image coordinates
#         x = (u - cx) / fx
#         y = (v - cy) / fy

#         # Interaction matrix for one point (2x6)
#         # Standard Chaumette formulation with focal length scaling
#         Lp = np.array([
#             [-fx/Z,    0,    fx*x/Z,  fx*x*y,      -fx*(1+x*x),  fx*y],
#             [   0,  -fy/Z,   fy*y/Z,  fy*(1+y*y),  -fy*x*y,     -fy*x]
#         ])
#         L.append(Lp)

#     return np.vstack(L)  # (2n, 6)

# # ==========================================
# # 4b. ORIENTATION FEATURES (for perpendicular approach)
# # ==========================================
# def compute_orientation_error(R_marker_to_cam):
#     """
#     Compute orientation error to align camera Z-axis perpendicular to marker plane.

#     The marker's normal in marker frame is [0, 0, 1].
#     We want the camera's optical axis [0, 0, 1] to align with the negative
#     of this normal in camera frame (camera looks toward marker).

#     Args:
#         R_marker_to_cam: (3, 3) rotation matrix from marker to camera frame

#     Returns:
#         e_orientation: (3,) orientation error in axis-angle form
#         theta: scalar angle error in radians (for monitoring)
#     """
#     # Marker normal in marker frame
#     n_marker = np.array([0, 0, 1])

#     # Transform to camera frame
#     n_marker_in_cam = R_marker_to_cam @ n_marker

#     # Desired: camera Z-axis points toward marker (opposite of marker normal)
#     n_target = -n_marker_in_cam  # direction camera should point
#     n_camera = np.array([0, 0, 1])  # camera's current forward direction

#     # Compute rotation axis (cross product)
#     axis = np.cross(n_camera, n_target)
#     axis_norm = np.linalg.norm(axis)

#     # Compute angle (dot product)
#     cos_theta = np.clip(np.dot(n_camera, n_target), -1.0, 1.0)
#     theta = np.arccos(cos_theta)

#     # Convert to axis-angle error vector
#     if axis_norm > 1e-6 and theta > 1e-6:
#         u = axis / axis_norm
#         e_orientation = theta * u
#     else:
#         e_orientation = np.zeros(3)
#         theta = 0.0

#     return e_orientation, theta

# def skew_symmetric(v):
#     """Create skew-symmetric matrix from 3-vector."""
#     return np.array([
#         [0, -v[2], v[1]],
#         [v[2], 0, -v[0]],
#         [-v[1], v[0], 0]
#     ])

# def compute_orientation_jacobian(theta, u):
#     """
#     Compute interaction matrix for theta-u orientation features.

#     Based on Chaumette's formulation. For small angles, L_omega ≈ I.

#     Args:
#         theta: rotation angle
#         u: unit rotation axis (3-vector)

#     Returns:
#         L_omega: (3, 3) matrix relating orientation feature rate to angular velocity
#     """
#     if theta < 1e-6:
#         # For small angles, L_omega ≈ I
#         L_omega = np.eye(3)
#     else:
#         ux = skew_symmetric(u)
#         # sinc(x) = sin(x)/x
#         sinc_theta = np.sin(theta) / theta
#         sinc_half = np.sin(theta/2) / (theta/2)

#         L_omega = np.eye(3) - (theta/2) * ux + (1 - sinc_theta / (sinc_half**2)) * (ux @ ux)

#     return L_omega

# def compute_combined_jacobian(points, depths, e_orientation):
#     """
#     Compute combined interaction matrix for position and orientation features.

#     Args:
#         points: (n, 2) array of image points
#         depths: (n,) array of depth values
#         e_orientation: (3,) orientation error vector

#     Returns:
#         L_combined: (2n+3, 6) combined interaction matrix
#     """
#     # Position Jacobian (existing function)
#     L_position = compute_image_jacobian(points, depths)  # (8, 6)

#     # Orientation Jacobian
#     theta = np.linalg.norm(e_orientation)
#     if theta > 1e-6:
#         u = e_orientation / theta
#     else:
#         u = np.array([0, 0, 1])

#     L_omega = compute_orientation_jacobian(theta, u)  # (3, 3)

#     # Full orientation Jacobian: [0_3x3 | L_omega]
#     L_orientation = np.zeros((3, 6))
#     L_orientation[:, 3:] = L_omega  # Angular velocity part only

#     # Stack them
#     L_combined = np.vstack([L_position, L_orientation])  # (11, 6)

#     return L_combined

# # ==========================================
# # 5. ROBOT JACOBIAN
# # ==========================================
# def get_robot_jacobian(model, data, body_name):
#     """
#     Get the geometric Jacobian at a body frame.

#     Args:
#         model: MuJoCo model
#         data: MuJoCo data
#         body_name: Name of the body

#     Returns:
#         J: (6, nv) Jacobian matrix [Jp; Jr] (position and rotation)
#     """
#     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
#     if body_id == -1:
#         raise ValueError(f"Body '{body_name}' not found")

#     jacp = np.zeros((3, model.nv))
#     jacr = np.zeros((3, model.nv))
#     mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

#     return np.vstack([jacp, jacr])  # (6, nv)

# # ==========================================
# # 6. DEPTH RENDERING
# # ==========================================
# def get_depth_at_points(depth_buffer, points, model):
#     """
#     Get metric depth Z for each feature point from MuJoCo depth buffer.

#     MuJoCo's depth buffer uses OpenGL conventions where the depth is
#     stored as normalized values between near and far planes.

#     Args:
#         depth_buffer: MuJoCo rendered depth buffer
#         points: (n, 2) array of image points (u, v)
#         model: MuJoCo model (for near/far plane info)

#     Returns:
#         depths: (n,) array of metric depth values
#     """
#     # Get depth range from model
#     extent = model.stat.extent
#     znear = model.vis.map.znear * extent
#     zfar = model.vis.map.zfar * extent

#     depths = []
#     h, w = depth_buffer.shape

#     for (u, v) in points:
#         u_int = int(np.clip(u, 0, w - 1))
#         v_int = int(np.clip(v, 0, h - 1))

#         z_norm = depth_buffer[v_int, u_int]

#         # Convert normalized depth to metric depth
#         # Standard OpenGL depth linearization formula
#         if z_norm < 1.0 and z_norm > 0.0:
#             z_metric = znear * zfar / (zfar - z_norm * (zfar - znear))
#         else:
#             z_metric = zfar  # At infinity or invalid

#         # Clamp to reasonable range
#         z_metric = np.clip(z_metric, 0.05, 5.0)
#         depths.append(z_metric)

#     return np.array(depths)

# def get_marker_depth_from_buffer(depth_buffer, corners, model):
#     """
#     Get marker center depth directly from depth buffer.

#     Samples depth at marker center with a small window for robustness.
#     More reliable than PnP Z-estimate for distance measurement.

#     Args:
#         depth_buffer: MuJoCo rendered depth buffer
#         corners: ArUco marker corners from detection
#         model: MuJoCo model (for near/far plane info)

#     Returns:
#         z_metric: Metric depth to marker center (meters), or None if invalid
#     """
#     if corners is None or len(corners) == 0:
#         return None

#     # Compute marker center from corners
#     # detect_aruco_corners returns shape (4, 2) directly
#     center = np.mean(corners, axis=0)  # Shape (2,)
#     cx, cy = int(center[0]), int(center[1])

#     # Sample depth at center with small window (median for robustness)
#     h, w = depth_buffer.shape
#     window = 3
#     depth_samples = []

#     for dy in range(-window, window + 1):
#         for dx in range(-window, window + 1):
#             y, x = cy + dy, cx + dx
#             if 0 <= y < h and 0 <= x < w:
#                 depth_samples.append(depth_buffer[y, x])

#     if not depth_samples:
#         return None

#     z_norm = np.median(depth_samples)

#     # Skip invalid depth values
#     if z_norm >= 1.0 or z_norm <= 0.0:
#         return None

#     # Convert normalized depth to metric depth (OpenGL linearization)
#     extent = model.stat.extent
#     znear = model.vis.map.znear * extent
#     zfar = model.vis.map.zfar * extent
#     z_metric = znear * zfar / (zfar - z_norm * (zfar - znear))

#     # Clamp to reasonable range
#     z_metric = np.clip(z_metric, 0.05, 2.0)

#     return z_metric

# # ==========================================
# # 7. IBVS CONTROL LAW
# # ==========================================
# def ibvs_control(current_features, desired_features, depths, model, data,
#                  lambda_gain=IBVS_LAMBDA, lambda_orient=IBVS_LAMBDA_ORIENT,
#                  R_marker_to_cam=None, enable_orientation=ENABLE_ORIENTATION_CONTROL,
#                  enable_joint_weighting=ENABLE_JOINT_WEIGHTING):
#     """
#     Compute joint velocities using IBVS control law with optional orientation control.

#     Classic IBVS: v_c = -lambda * L^+ * (s - s*)
#     Extended: includes orientation features to align gripper perpendicular to marker.

#     Args:
#         current_features: (4, 2) current corner positions in pixels
#         desired_features: (4, 2) desired corner positions in pixels
#         depths: (4,) depth values for each corner
#         model, data: MuJoCo model and data
#         lambda_gain: proportional gain for position
#         lambda_orient: proportional gain for orientation
#         R_marker_to_cam: (3, 3) rotation matrix from marker to camera (required if enable_orientation)
#         enable_orientation: whether to include orientation control
#         enable_joint_weighting: whether to use adaptive joint weights for singularity avoidance

#     Returns:
#         q_dot: (5,) joint velocities for arm joints
#         error_position: (8,) position feature error vector
#         v_camera: (6,) computed camera velocity
#         orientation_error_deg: scalar orientation error in degrees (for display)
#     """
#     # 1. Compute position feature error (flatten to 8x1)
#     error_position = (current_features - desired_features).flatten()

#     # 2. Compute orientation error if enabled
#     orientation_error_deg = 0.0
#     if enable_orientation and R_marker_to_cam is not None:
#         e_orientation, theta = compute_orientation_error(R_marker_to_cam)
#         orientation_error_deg = np.degrees(theta)

#         # Combined interaction matrix (11x6)
#         L = compute_combined_jacobian(current_features, depths, e_orientation)

#         # Combined weighted error vector
#         error_combined = np.concatenate([
#             lambda_gain * error_position,
#             lambda_orient * e_orientation
#         ])
#     else:
#         # Position-only control (original behavior)
#         L = compute_image_jacobian(current_features, depths)
#         error_combined = lambda_gain * error_position

#     # 3. Compute pseudo-inverse with damping for numerical stability
#     # Using damped least squares (Levenberg-Marquardt)
#     damping = 0.01
#     L_pinv = L.T @ np.linalg.inv(L @ L.T + damping * np.eye(L.shape[0]))

#     # 4. IBVS control law: camera velocity in camera frame
#     # Note: gains already applied to error components
#     v_camera = -L_pinv @ error_combined

#     # 5. Transform camera velocity to end-effector frame
#     R_cam_to_ee, p_cam_in_ee = get_camera_to_ee_transform()
#     v_ee = transform_velocity_cam_to_ee(v_camera, R_cam_to_ee, p_cam_in_ee)

#     # 6. Get robot Jacobian at gripper body
#     J_robot = get_robot_jacobian(model, data, CAM_MOUNT_BODY)

#     # 7. Only use first 5 joints (exclude gripper actuator)
#     J_arm = J_robot[:, :5]

#     # 8. Compute joint velocities using damped pseudo-inverse
#     # With optional joint weighting for singularity avoidance

#     # Adaptive damping based on manipulability - increases near singularities
#     manipulability = compute_manipulability(model, data)
#     base_damping = 0.01
#     max_damping = 0.3
#     if manipulability < 0.1:
#         robot_damping = base_damping + (max_damping - base_damping) * (1 - manipulability / 0.1)
#     else:
#         robot_damping = base_damping

#     if enable_joint_weighting:
#         # Compute adaptive weights based on joint configuration and error
#         weights = compute_joint_weights(data, np.linalg.norm(error_position))
#         W = np.diag(weights)
#         # Weighted damped least squares pseudo-inverse
#         J_pinv = W @ J_arm.T @ np.linalg.inv(J_arm @ W @ J_arm.T + robot_damping * np.eye(6))
#     else:
#         # Standard damped pseudo-inverse (original behavior)
#         J_pinv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + robot_damping * np.eye(6))

#     q_dot = J_pinv @ v_ee

#     return q_dot, error_position, v_camera, orientation_error_deg

# def clamp_joint_velocities(q_dot, max_vel=MAX_JOINT_VEL):
#     """Clamp joint velocities to safe limits."""
#     return np.clip(q_dot, -max_vel, max_vel)

# def compute_centered_desired_features(width=CAM_WIDTH, height=CAM_HEIGHT,
#                                        marker_size_px=DESIRED_MARKER_SIZE,
#                                        offset_y=DESIRED_OFFSET_Y):
#     """
#     Compute desired feature locations for marker target position.

#     The marker target is offset from center to keep gripper away from marker.
#     Offset_y > 0 moves target DOWN in image (marker appears lower, gripper stays above).

#     Args:
#         width, height: Image dimensions
#         marker_size_px: Desired marker size in pixels (smaller = further away)
#         offset_y: Vertical offset in pixels (positive = down in image)

#     Returns:
#         desired_corners: (4, 2) array of corner positions
#     """
#     cx_img = width / 2
#     cy_img = height / 2 + offset_y  # Offset target downward
#     half = marker_size_px / 2

#     # Corners of a SQUARE, ordered by angle from centroid (ascending)
#     # In image coords (Y down): arctan2(dy, dx)
#     desired_corners = np.array([
#         [cx_img - half, cy_img - half],  # Top-left (angle -3π/4)
#         [cx_img + half, cy_img - half],  # Top-right (angle -π/4)
#         [cx_img + half, cy_img + half],  # Bottom-right (angle +π/4)
#         [cx_img - half, cy_img + half],  # Bottom-left (angle +3π/4)
#     ])

#     return desired_corners

# # ==========================================
# # 8. STATE MACHINE ENUMS
# # ==========================================
# class SystemState(IntEnum):
#     """Main system state machine for grasp pipeline."""
#     INIT = 0
#     SEARCH = 1
#     IBVS_APPROACH = 2
#     GRASPABILITY_CHECK = 3
#     GRASP_EXECUTE = 4
#     COMPLETE = 5
#     FAILED = 6

# class SearchPhase(IntEnum):
#     """Search sub-states."""
#     IDLE = 0
#     COARSE_SWEEP = 1      # Joint 1 + Joint 4 sweep
#     MOVING_TO_NEXT = 2    # Transitioning to next search position

# class GraspState(IntEnum):
#     """Grasp execution sub-states."""
#     ORIENT_ALIGN = 0      # Align Joint 5 with cube edges
#     VERTICAL_DESCENT = 1  # Lower gripper to grasp height
#     GRIPPER_CLOSE = 2     # Close gripper around cube
#     LIFT_VERIFY = 3       # Lift and verify grasp success

# # ==========================================
# # 9. SEARCH STRATEGY FUNCTIONS
# # ==========================================
# def search_for_marker(data, model, search_state, search_pan_idx, search_pitch_idx, dwell_counter):
#     """
#     Simple, safe search strategy: rotate base only while keeping arm elevated.

#     The arm maintains a safe "observation pose" looking down at the workspace,
#     and only rotates the base joint (shoulder_pan) to scan the area.

#     This approach:
#     - Never lowers the gripper toward the ground
#     - Uses simple base rotation only
#     - Keeps arm in a stable, elevated configuration

#     Args:
#         data: MuJoCo data
#         model: MuJoCo model
#         search_state: Current SearchPhase
#         search_pan_idx: Current pan position index
#         search_pitch_idx: Unused (kept for API compatibility)
#         dwell_counter: Frames spent at current position

#     Returns:
#         (new_search_state, new_pan_idx, new_pitch_idx, new_dwell_counter)
#     """
#     # Only rotate base - small range (arm pose set by global SEARCH_*_ANGLE parameters)
#     pan_positions = np.arange(-0.4, 0.5, 0.2)  # -0.4 to 0.4 rad in 0.2 rad steps

#     if search_state == SearchPhase.IDLE:
#         # Start search - immediately set safe observation pose
#         set_search_pose(data)
#         return SearchPhase.COARSE_SWEEP, 0, 0, 0

#     elif search_state == SearchPhase.COARSE_SWEEP:
#         # Set safe observation pose (always)
#         pan_angle = pan_positions[min(search_pan_idx, len(pan_positions)-1)]

#         data.ctrl[0] = pan_angle          # Only this changes during search
#         data.ctrl[1] = SEARCH_LIFT_ANGLE  # Keep arm elevated (global param)
#         data.ctrl[2] = SEARCH_ELBOW_ANGLE # Elbow bent (global param)
#         data.ctrl[3] = SEARCH_WRIST_ANGLE # Looking down at workspace (global param)
#         data.ctrl[4] = 0.0                # Wrist roll neutral

#         # Dwell at current position
#         if dwell_counter < SEARCH_DWELL_FRAMES:
#             return SearchPhase.COARSE_SWEEP, search_pan_idx, 0, dwell_counter + 1
#         else:
#             # Move to next pan position
#             new_pan_idx = search_pan_idx + 1
#             if new_pan_idx >= len(pan_positions):
#                 # Completed sweep, restart from beginning
#                 new_pan_idx = 0
#                 print("Search: Base rotation sweep complete, restarting...")

#             return SearchPhase.COARSE_SWEEP, new_pan_idx, 0, 0

#     return search_state, search_pan_idx, search_pitch_idx, dwell_counter

# def get_search_state_name(state):
#     """Get human-readable name for search state."""
#     names = {
#         SearchPhase.IDLE: "IDLE",
#         SearchPhase.COARSE_SWEEP: "COARSE_SWEEP",
#         SearchPhase.MOVING_TO_NEXT: "MOVING"
#     }
#     return names.get(state, "UNKNOWN")

# def set_search_pose(data):
#     """Set joints to elevated search observation pose immediately."""
#     data.ctrl[0] = 0.0                # Center pan
#     data.ctrl[1] = SEARCH_LIFT_ANGLE  # Elevated shoulder
#     data.ctrl[2] = SEARCH_ELBOW_ANGLE # Bent elbow
#     data.ctrl[3] = SEARCH_WRIST_ANGLE # Camera looks down
#     data.ctrl[4] = 0.0                # Neutral wrist roll

# # ==========================================
# # 10. GRASPABILITY CHECK FUNCTIONS
# # ==========================================
# def compute_manipulability(model, data, body_name="gripper"):
#     """
#     Compute Yoshikawa manipulability index: sqrt(det(J @ J.T))

#     Low manipulability (<0.01) indicates proximity to singularity.

#     Args:
#         model: MuJoCo model
#         data: MuJoCo data
#         body_name: Name of the end-effector body

#     Returns:
#         manipulability: float (higher = better manipulability)
#     """
#     J = get_robot_jacobian(model, data, body_name)
#     J_arm = J[:, :5]  # Only arm joints

#     # Yoshikawa manipulability measure
#     JJT = J_arm @ J_arm.T
#     det_JJT = np.linalg.det(JJT)

#     if det_JJT > 0:
#         return np.sqrt(det_JJT)
#     else:
#         return 0.0

# def estimate_marker_3d_position(model, data, t_marker_in_cam):
#     """
#     Transform marker position from camera frame to world frame.

#     Args:
#         model: MuJoCo model
#         data: MuJoCo data
#         t_marker_in_cam: (3,) translation vector from camera to marker

#     Returns:
#         marker_pos_world: (3,) position in world frame
#     """
#     # Get camera pose in world frame
#     cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
#     if cam_id == -1:
#         return None

#     # Camera position and orientation in world frame
#     cam_pos = data.cam_xpos[cam_id]
#     cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

#     # Transform marker position to world frame
#     marker_pos_world = cam_pos + cam_mat @ t_marker_in_cam

#     return marker_pos_world

# def check_graspability(model, data, corners, depth_buffer, R_marker_to_cam, t_marker_in_cam,
#                        consecutive_detections, ibvs_error):
#     """
#     Check if the cube is graspable based on multiple criteria.

#     Args:
#         model: MuJoCo model
#         data: MuJoCo data
#         corners: Detected ArUco corners
#         depth_buffer: Depth image
#         R_marker_to_cam: Rotation matrix from marker to camera
#         t_marker_in_cam: Translation vector to marker in camera frame
#         consecutive_detections: Number of consecutive frames marker was detected
#         ibvs_error: Current IBVS position error norm

#     Returns:
#         (is_graspable: bool, reasons: list[str], metrics: dict)
#     """
#     reasons = []
#     metrics = {}
#     is_graspable = True

#     # Check 1: Marker visibility stability
#     metrics['consecutive_detections'] = consecutive_detections
#     if consecutive_detections < MARKER_STABILITY_FRAMES:
#         reasons.append(f"Marker unstable ({consecutive_detections}/{MARKER_STABILITY_FRAMES} frames)")
#         is_graspable = False

#     # Check 2: Distance from base
#     if t_marker_in_cam is not None:
#         marker_pos = estimate_marker_3d_position(model, data, t_marker_in_cam)
#         if marker_pos is not None:
#             distance_from_base = np.linalg.norm(marker_pos[:2])  # XY distance
#             metrics['distance_from_base'] = distance_from_base
#             if distance_from_base > MAX_GRASP_DISTANCE:
#                 reasons.append(f"Out of reach ({distance_from_base:.2f}m > {MAX_GRASP_DISTANCE}m)")
#                 is_graspable = False

#             # Check 3: Marker Z-height (ground clearance)
#             marker_z = marker_pos[2]
#             metrics['marker_z'] = marker_z
#             min_safe_z = GRIPPER_FINGER_LENGTH * 0.5
#             if marker_z < min_safe_z:
#                 reasons.append(f"Too close to ground (z={marker_z:.3f}m)")
#                 is_graspable = False

#     # Check 4: Manipulability
#     manipulability = compute_manipulability(model, data)
#     metrics['manipulability'] = manipulability
#     if manipulability < MIN_MANIPULABILITY:
#         reasons.append(f"Low manipulability ({manipulability:.4f} < {MIN_MANIPULABILITY})")
#         is_graspable = False
#     elif manipulability < MIN_MANIPULABILITY_WARNING:
#         reasons.append(f"Warning: Near singularity ({manipulability:.4f})")

#     # Check 5: IBVS convergence
#     metrics['ibvs_error'] = ibvs_error
#     if ibvs_error > CONVERGENCE_THRESHOLD * 2:
#         reasons.append(f"IBVS not converged (error={ibvs_error:.1f}px)")
#         is_graspable = False

#     if is_graspable and len(reasons) == 0:
#         reasons.append("All checks passed")

#     return is_graspable, reasons, metrics

# # ==========================================
# # 11. GRASP EXECUTION FUNCTIONS
# # ==========================================
# def align_gripper_orientation(data, model, R_marker_to_cam, current_joint5):
#     """
#     Compute Joint 5 (wrist_roll) angle to align gripper fingers with cube edges.

#     Extracts marker yaw angle and computes required wrist rotation to align
#     gripper fingers parallel to cube edges for optimal grasp.

#     Args:
#         data: MuJoCo data
#         model: MuJoCo model
#         R_marker_to_cam: (3,3) rotation matrix from marker to camera frame
#         current_joint5: Current wrist_roll joint angle

#     Returns:
#         target_joint5: Target wrist_roll angle
#         angle_error_deg: Remaining alignment error in degrees
#     """
#     if R_marker_to_cam is None:
#         return current_joint5, 0.0

#     # Extract marker yaw angle (rotation around Z in camera frame)
#     # The marker's X-axis in camera frame gives us the orientation
#     marker_x_in_cam = R_marker_to_cam[:, 0]  # First column = marker X-axis

#     # Project to camera XY plane and compute angle
#     marker_yaw = np.arctan2(marker_x_in_cam[1], marker_x_in_cam[0])

#     # We want gripper fingers (which open along gripper X-axis) to align with cube edges
#     # Cube edges are at marker_yaw and marker_yaw + 90°
#     # Choose the closer alignment
#     options = [marker_yaw, marker_yaw + np.pi/2, marker_yaw - np.pi/2, marker_yaw + np.pi]

#     # Find which orientation requires minimum rotation from current
#     best_target = current_joint5
#     min_rotation = float('inf')

#     for opt in options:
#         # Normalize to [-pi, pi]
#         opt_normalized = np.arctan2(np.sin(opt), np.cos(opt))
#         rotation_needed = abs(opt_normalized - current_joint5)
#         if rotation_needed < min_rotation:
#             min_rotation = rotation_needed
#             best_target = opt_normalized

#     # Clamp to joint limits
#     ctrl_range = model.actuator_ctrlrange[4]  # wrist_roll is actuator 4
#     best_target = np.clip(best_target, ctrl_range[0], ctrl_range[1])

#     angle_error_deg = np.degrees(abs(best_target - current_joint5))

#     return best_target, angle_error_deg

# def vertical_descent_control(data, model, target_z, descent_speed=DESCENT_SPEED):
#     """
#     Control vertical descent using Jacobian-based Cartesian velocity control.

#     Maintains X-Y position while lowering Z by computing required joint velocities
#     for a pure vertical (negative Z) end-effector velocity.

#     Args:
#         data: MuJoCo data
#         model: MuJoCo model
#         target_z: Target gripper Z height
#         descent_speed: Descent speed in m/step

#     Returns:
#         (reached_target: bool, current_z: float)
#     """
#     # Get current gripper position
#     gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
#     if gripper_id == -1:
#         return True, 0.0

#     current_pos = data.xpos[gripper_id].copy()
#     current_z = current_pos[2]

#     # Check if target reached (5mm tolerance)
#     if current_z <= target_z + 0.005:
#         return True, current_z

#     # Desired end-effector velocity: pure vertical descent
#     # v_ee = [vx, vy, vz, wx, wy, wz]
#     v_desired = np.array([0.0, 0.0, -descent_speed, 0.0, 0.0, 0.0])

#     # Get robot Jacobian
#     J_robot = get_robot_jacobian(model, data, "gripper")
#     J_arm = J_robot[:, :5]  # Only arm joints

#     # Compute joint velocities using damped pseudo-inverse
#     damping = 0.05  # Higher damping for stability during descent
#     J_pinv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + damping * np.eye(6))

#     q_dot = J_pinv @ v_desired

#     # Scale to position increment
#     dt = 0.1  # Time step
#     q_current = data.qpos[:5].copy()
#     q_new = q_current + q_dot * dt

#     # Clamp to joint limits
#     for i in range(5):
#         ctrl_range = model.actuator_ctrlrange[i]
#         q_new[i] = np.clip(q_new[i], ctrl_range[0], ctrl_range[1])

#     # Apply to controls
#     data.ctrl[:5] = q_new

#     return False, current_z

# def control_gripper(data, target_position, current_position, close_speed=GRIPPER_CLOSE_SPEED):
#     """
#     Gradual gripper control with position tracking.

#     Args:
#         data: MuJoCo data
#         target_position: Target gripper joint position (radians)
#         current_position: Current gripper joint position
#         close_speed: Speed of gripper motion (rad/step)

#     Returns:
#         (new_position: float, reached_target: bool)
#     """
#     gripper_idx = 5  # Gripper is actuator index 5

#     if target_position > current_position:
#         # Closing
#         new_pos = min(current_position + close_speed, target_position)
#     else:
#         # Opening
#         new_pos = max(current_position - close_speed, target_position)

#     data.ctrl[gripper_idx] = new_pos
#     reached_target = abs(new_pos - target_position) < 0.01

#     return new_pos, reached_target

# def verify_grasp_success(model, data, initial_marker_z, lift_height=LIFT_HEIGHT):
#     """
#     Verify grasp by checking if marker moved up with gripper.

#     Args:
#         model: MuJoCo model
#         data: MuJoCo data
#         initial_marker_z: Marker Z position before lifting
#         lift_height: Expected lift height

#     Returns:
#         grasp_successful: bool
#     """
#     # Get current cube position (if it exists in scene)
#     cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
#     if cube_id == -1:
#         # No cube body, assume success if we got this far
#         return True

#     current_cube_z = data.xpos[cube_id][2]

#     # Check if cube lifted with gripper
#     z_change = current_cube_z - initial_marker_z

#     # Success if cube moved up at least half the expected lift height
#     return z_change > lift_height * 0.5

# def get_system_state_name(state):
#     """Get human-readable name for system state."""
#     names = {
#         SystemState.INIT: "INIT",
#         SystemState.SEARCH: "SEARCH",
#         SystemState.IBVS_APPROACH: "IBVS_APPROACH",
#         SystemState.GRASPABILITY_CHECK: "GRASPABILITY_CHECK",
#         SystemState.GRASP_EXECUTE: "GRASP_EXECUTE",
#         SystemState.COMPLETE: "COMPLETE",
#         SystemState.FAILED: "FAILED"
#     }
#     return names.get(state, "UNKNOWN")

# def get_grasp_state_name(state):
#     """Get human-readable name for grasp state."""
#     names = {
#         GraspState.ORIENT_ALIGN: "ORIENT_ALIGN",
#         GraspState.VERTICAL_DESCENT: "VERTICAL_DESCENT",
#         GraspState.GRIPPER_CLOSE: "GRIPPER_CLOSE",
#         GraspState.LIFT_VERIFY: "LIFT_VERIFY"
#     }
#     return names.get(state, "UNKNOWN")

# def get_state_color(state):
#     """Get display color for system state."""
#     colors = {
#         SystemState.INIT: (255, 255, 255),        # White
#         SystemState.SEARCH: (255, 255, 0),         # Yellow
#         SystemState.IBVS_APPROACH: (255, 165, 0),  # Orange
#         SystemState.GRASPABILITY_CHECK: (0, 255, 255),  # Cyan
#         SystemState.GRASP_EXECUTE: (255, 0, 255),  # Magenta
#         SystemState.COMPLETE: (0, 255, 0),         # Green
#         SystemState.FAILED: (0, 0, 255)            # Red
#     }
#     return colors.get(state, (255, 255, 255))

# # ==========================================
# # 12. JOINT WEIGHTING FUNCTIONS
# # ==========================================
# def compute_joint_weights(data, error_norm):
#     """
#     Compute adaptive joint weights for singularity avoidance.

#     Weights are based on:
#     1. Joint 4 (wrist_flex) proximity to singularity (0° or ±180°)
#     2. Error magnitude (reduce base weight for fine convergence)

#     Args:
#         data: MuJoCo data
#         error_norm: Current IBVS error norm in pixels

#     Returns:
#         weights: (5,) array of joint weights
#     """
#     q = data.qpos[:5]

#     # Base weights: [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll]
#     # Elbow has highest weight (primary workhorse)
#     # Wrist_roll excluded from position control (handled separately for orientation)
#     weights = np.array([1.0, 1.2, 1.5, 1.0, 0.0])

#     # Joint 4 (wrist_flex) singularity check
#     # Singularity occurs near 0° or ±180°
#     j4_angle_deg = np.degrees(q[3])
#     near_zero = abs(j4_angle_deg) < WRIST_SINGULARITY_ZONE
#     near_180 = abs(abs(j4_angle_deg) - 180) < WRIST_SINGULARITY_ZONE

#     if near_zero or near_180:
#         # Near singularity: reduce wrist_flex contribution, redistribute to shoulder/elbow
#         weights[3] = 0.2  # Reduce wrist_flex
#         weights[1] *= 1.3  # Increase shoulder_lift
#         weights[2] *= 1.2  # Increase elbow
#         # print(f"Warning: Wrist near singularity ({j4_angle_deg:.1f}°)")

#     # Joint 2 (elbow_flex) singularity check - near fully extended
#     j2_angle_deg = np.degrees(q[2])
#     if abs(j2_angle_deg) < 15.0:
#         # Near elbow singularity: reduce elbow contribution, redistribute
#         weights[2] = 0.3  # Reduce elbow
#         weights[1] *= 1.4  # Increase shoulder_lift
#         weights[3] *= 1.2  # Increase wrist_flex

#     # Reduce base rotation (joint 0) during fine convergence
#     if error_norm < BASE_WEIGHT_FINE_THRESHOLD:
#         weights[0] = 0.5

#     return weights

# # ==========================================
# # 13. XML PATCHING UTILITIES
# # ==========================================
# def patch_robot_xml(input_path, output_path):
#     """Injects a camera definition into the gripper body of the robot XML."""
#     print(f"Patching {input_path} to add camera on '{CAM_MOUNT_BODY}' body...")
#     try:
#         tree = ET.parse(input_path)
#         root = tree.getroot()

#         mount_body = None
#         for body in root.iter('body'):
#             if body.get('name') == CAM_MOUNT_BODY:
#                 mount_body = body
#                 break

#         if mount_body is None:
#             print(f"Error: Could not find '{CAM_MOUNT_BODY}' body in {input_path}")
#             return False

#         # Remove existing camera if present
#         for cam in mount_body.findall('camera'):
#             if cam.get('name') == CAM_NAME:
#                 mount_body.remove(cam)
#                 print(f"Removed existing camera '{CAM_NAME}' for re-patching.")

#         # Create new camera element
#         cam_elem = ET.Element('camera')
#         cam_elem.set('name', CAM_NAME)
#         cam_elem.set('pos', CAM_POS)
#         cam_elem.set('quat', CAM_QUAT)
#         cam_elem.set('fovy', str(CAM_FOVY))
#         cam_elem.set('mode', 'fixed')

#         mount_body.insert(0, cam_elem)
#         tree.write(output_path)
#         print(f"Created patched robot file: {output_path}")
#         return True

#     except Exception as e:
#         print(f"Failed to patch robot XML: {e}")
#         return False


# def patch_scene_xml(scene_path, original_robot, new_robot, output_path):
#     """Updates the scene file to point to the new robot XML file."""
#     try:
#         with open(scene_path, 'r') as f:
#             content = f.read()

#         original_basename = os.path.basename(original_robot)
#         new_basename = os.path.basename(new_robot)

#         if original_basename in content:
#             new_content = content.replace(original_basename, new_basename)
#             with open(output_path, 'w') as f:
#                 f.write(new_content)
#             print(f"Created patched scene file: {output_path}")
#             return True
#         else:
#             print(f"Warning: Could not find reference to '{original_basename}' in '{scene_path}'.")
#             return False

#     except Exception as e:
#         print(f"Failed to patch scene XML: {e}")
#         return False


# # ==========================================
# # 9. MAIN SIMULATION LOOP
# # ==========================================
# def run_simulation():
#     Path(IMG_DIR).mkdir(exist_ok=True)

#     # Prepare XML files
#     if not os.path.exists(ROBOT_XML):
#         print(f"Error: {ROBOT_XML} not found!")
#         return

#     xml_ready = patch_robot_xml(ROBOT_XML, TEMP_ROBOT_XML)
#     if not xml_ready:
#         return

#     scene_ready = False
#     if os.path.exists(SCENE_XML):
#         scene_ready = patch_scene_xml(SCENE_XML, ROBOT_XML, TEMP_ROBOT_XML, TEMP_SCENE_XML)

#     model_file = TEMP_SCENE_XML if scene_ready else TEMP_ROBOT_XML
#     print(f"Loading model from: {model_file}")

#     # Load Model
#     try:
#         model = mujoco.MjModel.from_xml_path(model_file)
#         data = mujoco.MjData(model)
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Failed to load MuJoCo model: {e}")
#         return

#     # Setup Camera
#     try:
#         cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
#         if cam_id == -1:
#             raise ValueError(f"Camera '{CAM_NAME}' not found")
#         print(f"Camera '{CAM_NAME}' found with ID: {cam_id}")
#     except Exception as e:
#         print(f"Error: {e}")
#         return

#     # Initialize renderers
#     rgb_renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
#     depth_renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
#     depth_renderer.enable_depth_rendering()

#     # Set initial joint positions
#     # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
#     initial_qpos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#     # Apply to both qpos and ctrl, then step simulation to stabilize
#     for i, q in enumerate(initial_qpos):
#         if i < len(data.qpos):
#             data.qpos[i] = q
#         if i < len(data.ctrl):
#             data.ctrl[i] = q

#     # Step simulation a few times to let robot settle
#     mujoco.mj_forward(model, data)
#     for _ in range(100):
#         mujoco.mj_step(model, data)
#         # Keep applying control to maintain position
#         for i, q in enumerate(initial_qpos):
#             if i < len(data.ctrl):
#                 data.ctrl[i] = q

#     print(f"Initial joint positions set: {[f'{q:.2f}' for q in data.qpos[:6]]}")

#     # Initialize desired features (centered marker)
#     desired_features = compute_centered_desired_features()
#     use_auto_desired = True  # Start with auto-computed desired features

#     print("\n" + "="*60)
#     print("IBVS Visual Servoing with Grasp Pipeline")
#     print("="*60)
#     print(f"Scene: {'Cube with ArUco' if USE_CUBE_SCENE else 'Flat ArUco marker'}")
#     print(f"Position gain: {IBVS_LAMBDA}")
#     print(f"Orientation gain: {IBVS_LAMBDA_ORIENT}")
#     print(f"Orientation control: {'ENABLED' if ENABLE_ORIENTATION_CONTROL else 'DISABLED'}")
#     print(f"Joint weighting: {'ENABLED' if ENABLE_JOINT_WEIGHTING else 'DISABLED'}")
#     print(f"Search: {'ENABLED' if ENABLE_SEARCH else 'DISABLED'}")
#     print(f"Graspability check: {'ENABLED' if ENABLE_GRASPABILITY_CHECK else 'DISABLED'}")
#     print(f"Grasp execution: {'ENABLED' if ENABLE_GRASP_EXECUTION else 'DISABLED'}")
#     print(f"Max joint velocity: {MAX_JOINT_VEL} rad/s")
#     print(f"Convergence threshold: {CONVERGENCE_THRESHOLD} px")
#     print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
#     print("\nControls:")
#     print("  'G' - Go/Start pipeline (auto-search if marker not visible)")
#     print("  'D' - Set current marker position as Desired")
#     print("  'A' - Use Auto-centered desired features")
#     print("  'X' - Clear/reset desired features")
#     print("  'S' - Toggle search ON/OFF (manual override)")
#     print("  'C' - Run graspability check (manual)")
#     print("  'E' - Execute grasp (manual, skips auto-check)")
#     print("  'R' - Reset to initial state")
#     print("  ESC - Quit")
#     print("="*60 + "\n")

#     frame_count = 0
#     start_time = time.time()
#     control_step = 0
#     error_history = []

#     # State machine variables
#     system_state = SystemState.INIT
#     search_phase = SearchPhase.IDLE
#     grasp_state = GraspState.ORIENT_ALIGN
#     search_pan_idx = 0
#     search_pitch_idx = 0
#     search_dwell_counter = 0
#     marker_lost_frames = 0
#     consecutive_detections = 0
#     gripper_position = GRIPPER_OPEN_POSITION
#     initial_cube_z = 0.0  # For grasp verification
#     descent_target_z = CUBE_SIZE / 2 + GRASP_HEIGHT_OFFSET  # Default, updated by depth
#     last_error_norm = 0.0
#     t_marker_in_cam = None
#     R_marker_to_cam_global = None

#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         while viewer.is_running() and time.time() - start_time < 1200:
#             # Physics step
#             mujoco.mj_step(model, data)

#             # Control at reduced rate (every 5 frames)
#             if frame_count % 5 == 0:
#                 # Render RGB image
#                 rgb_renderer.update_scene(data, camera=CAM_NAME)
#                 img_rgb = rgb_renderer.render()
#                 img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#                 # Render depth
#                 depth_renderer.update_scene(data, camera=CAM_NAME)
#                 depth_buffer = depth_renderer.render()

#                 # Detect ArUco marker
#                 corners = detect_aruco_corners(img_bgr)

#                 # Track marker detection for stability
#                 if corners is not None:
#                     consecutive_detections += 1
#                     marker_lost_frames = 0
#                     # Estimate marker pose
#                     R_marker_to_cam_global, t_marker_in_cam, pose_success = estimate_marker_pose(corners)
#                     if not pose_success:
#                         R_marker_to_cam_global = None
#                         t_marker_in_cam = None
#                 else:
#                     consecutive_detections = 0
#                     marker_lost_frames += 1

#                 # Draw detection
#                 display_img = draw_aruco_detection(img_bgr, corners, desired_features)

#                 # ========================================
#                 # STATE MACHINE LOGIC
#                 # ========================================

#                 # --- INIT STATE ---
#                 if system_state == SystemState.INIT:
#                     # Check if marker is visible
#                     if corners is not None:
#                         system_state = SystemState.IBVS_APPROACH
#                         print("Marker visible - Starting IBVS approach")
#                     elif ENABLE_SEARCH and marker_lost_frames > MARKER_LOST_THRESHOLD:
#                         system_state = SystemState.SEARCH
#                         search_phase = SearchPhase.IDLE
#                         set_search_pose(data)  # Immediately set elevated pose
#                         print("Marker not visible - Auto-starting search")

#                 # --- SEARCH STATE ---
#                 elif system_state == SystemState.SEARCH:
#                     if corners is not None:
#                         # Marker found during search
#                         system_state = SystemState.IBVS_APPROACH
#                         search_phase = SearchPhase.IDLE
#                         print("Marker FOUND during search - Starting IBVS approach")
#                     else:
#                         # Continue search
#                         search_phase, search_pan_idx, search_pitch_idx, search_dwell_counter = \
#                             search_for_marker(data, model, search_phase, search_pan_idx,
#                                             search_pitch_idx, search_dwell_counter)

#                 # --- IBVS APPROACH STATE ---
#                 elif system_state == SystemState.IBVS_APPROACH:
#                     if corners is None:
#                         # Lost marker during approach
#                         marker_lost_frames += 1
#                         if marker_lost_frames > MARKER_LOST_THRESHOLD and ENABLE_SEARCH:
#                             system_state = SystemState.SEARCH
#                             search_phase = SearchPhase.IDLE
#                             set_search_pose(data)  # Immediately set elevated pose
#                             print("Marker LOST - Returning to search")
#                     else:
#                         # Run IBVS control
#                         depths = get_depth_at_points(depth_buffer, corners, model)

#                         q_dot, error, v_cam, orient_error_deg = ibvs_control(
#                             corners, desired_features, depths,
#                             model, data,
#                             lambda_gain=IBVS_LAMBDA,
#                             lambda_orient=IBVS_LAMBDA_ORIENT,
#                             R_marker_to_cam=R_marker_to_cam_global,
#                             enable_orientation=ENABLE_ORIENTATION_CONTROL,
#                             enable_joint_weighting=ENABLE_JOINT_WEIGHTING
#                         )

#                         q_dot = clamp_joint_velocities(q_dot)
#                         q_current = data.qpos[:5].copy()
#                         q_new = q_current + q_dot * IBVS_DT

#                         for i in range(5):
#                             ctrl_range = model.actuator_ctrlrange[i]
#                             q_new[i] = np.clip(q_new[i], ctrl_range[0], ctrl_range[1])

#                         data.ctrl[:5] = q_new

#                         last_error_norm = np.linalg.norm(error)
#                         error_history.append(last_error_norm)

#                         # Get marker depth for approach distance check
#                         marker_depth = get_marker_depth_from_buffer(depth_buffer, corners, model)

#                         # Check for convergence to transition to graspability check
#                         # Requires both: pixel error converged AND depth within pre-grasp distance
#                         depth_ok = marker_depth is not None and marker_depth < PRE_GRASP_DISTANCE
#                         if last_error_norm < CONVERGENCE_THRESHOLD and depth_ok and ENABLE_GRASPABILITY_CHECK:
#                             system_state = SystemState.GRASPABILITY_CHECK
#                             print(f"IBVS converged (error={last_error_norm:.1f}px, depth={marker_depth:.3f}m) - Checking graspability")

#                         control_step += 1

#                 # --- GRASPABILITY CHECK STATE ---
#                 elif system_state == SystemState.GRASPABILITY_CHECK:
#                     if corners is not None:
#                         is_graspable, reasons, metrics = check_graspability(
#                             model, data, corners, depth_buffer,
#                             R_marker_to_cam_global, t_marker_in_cam,
#                             consecutive_detections, last_error_norm
#                         )

#                         print(f"Graspability check: {'PASS' if is_graspable else 'FAIL'}")
#                         for reason in reasons:
#                             print(f"  - {reason}")

#                         if is_graspable and ENABLE_GRASP_EXECUTION:
#                             system_state = SystemState.GRASP_EXECUTE
#                             grasp_state = GraspState.ORIENT_ALIGN
#                             # Store initial cube Z for verification
#                             cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
#                             if cube_id != -1:
#                                 initial_cube_z = data.xpos[cube_id][2]
#                             print("Graspability PASSED - Starting grasp execution")
#                         else:
#                             # Stay in IBVS approach to maintain position
#                             system_state = SystemState.IBVS_APPROACH
#                     else:
#                         system_state = SystemState.IBVS_APPROACH

#                 # --- GRASP EXECUTE STATE ---
#                 elif system_state == SystemState.GRASP_EXECUTE:
#                     if grasp_state == GraspState.ORIENT_ALIGN:
#                         # Align gripper orientation with cube
#                         current_j5 = data.qpos[4]
#                         target_j5, angle_err = align_gripper_orientation(
#                             data, model, R_marker_to_cam_global, current_j5)
#                         data.ctrl[4] = target_j5

#                         if angle_err < ORIENTATION_ALIGN_THRESHOLD:
#                             grasp_state = GraspState.VERTICAL_DESCENT
#                             # Calculate descent target using depth
#                             marker_depth = get_marker_depth_from_buffer(depth_buffer, corners, model)
#                             gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
#                             if marker_depth is not None and gripper_id != -1:
#                                 gripper_z = data.xpos[gripper_id][2]
#                                 # Cube top is approximately at gripper_z - marker_depth
#                                 # Target is cube center (half cube height below top)
#                                 descent_target_z = gripper_z - marker_depth - CUBE_SIZE / 2 + GRASP_HEIGHT_OFFSET
#                                 print(f"Orientation aligned - Descent target: {descent_target_z:.3f}m (depth={marker_depth:.3f}m)")
#                             else:
#                                 # Fallback to fixed target if depth unavailable
#                                 descent_target_z = CUBE_SIZE / 2 + GRASP_HEIGHT_OFFSET
#                                 print(f"Orientation aligned - Using fixed descent target: {descent_target_z:.3f}m")

#                     elif grasp_state == GraspState.VERTICAL_DESCENT:
#                         # Lower gripper to grasp height (using depth-calculated target)
#                         reached, current_z = vertical_descent_control(data, model, descent_target_z)

#                         if reached:
#                             grasp_state = GraspState.GRIPPER_CLOSE
#                             print(f"Descent complete (z={current_z:.3f}m) - Closing gripper")

#                     elif grasp_state == GraspState.GRIPPER_CLOSE:
#                         # Close gripper
#                         gripper_position, closed = control_gripper(
#                             data, GRIPPER_CLOSE_POSITION, gripper_position)

#                         if closed:
#                             grasp_state = GraspState.LIFT_VERIFY
#                             print("Gripper closed - Lifting to verify grasp")

#                     elif grasp_state == GraspState.LIFT_VERIFY:
#                         # Lift and verify
#                         # Simple lift by reducing shoulder_lift
#                         q_current = data.qpos[:5].copy()
#                         q_current[1] -= 0.01  # Lift arm
#                         for i in range(5):
#                             ctrl_range = model.actuator_ctrlrange[i]
#                             q_current[i] = np.clip(q_current[i], ctrl_range[0], ctrl_range[1])
#                         data.ctrl[:5] = q_current

#                         # Check if lifted enough
#                         gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
#                         if gripper_id != -1:
#                             current_gripper_z = data.xpos[gripper_id][2]
#                             if current_gripper_z > initial_cube_z + LIFT_HEIGHT:
#                                 # Verify grasp success
#                                 success = verify_grasp_success(model, data, initial_cube_z)
#                                 if success:
#                                     system_state = SystemState.COMPLETE
#                                     print("="*40)
#                                     print("   GRASP SUCCESSFUL!")
#                                     print("="*40)
#                                 else:
#                                     system_state = SystemState.FAILED
#                                     print("GRASP FAILED - Cube did not move with gripper")

#                 # --- COMPLETE STATE ---
#                 elif system_state == SystemState.COMPLETE:
#                     # Hold position and gripper
#                     gripper_position, _ = control_gripper(data, GRIPPER_CLOSE_POSITION, gripper_position)

#                 # --- FAILED STATE ---
#                 elif system_state == SystemState.FAILED:
#                     # Open gripper and wait for user input
#                     gripper_position, _ = control_gripper(data, GRIPPER_OPEN_POSITION, gripper_position)

#                 # ========================================
#                 # DISPLAY STATUS
#                 # ========================================
#                 state_name = get_system_state_name(system_state)
#                 state_color = get_state_color(system_state)
#                 cv2.putText(display_img, f"State: {state_name}", (10, 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

#                 # Show error if in IBVS state
#                 if system_state == SystemState.IBVS_APPROACH and error_history:
#                     cv2.putText(display_img, f"Error: {error_history[-1]:.1f} px", (10, 60),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#                 # Show grasp sub-state if grasping
#                 if system_state == SystemState.GRASP_EXECUTE:
#                     grasp_name = get_grasp_state_name(grasp_state)
#                     cv2.putText(display_img, f"Grasp: {grasp_name}", (10, 60),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

#                 # Show search info if searching
#                 if system_state == SystemState.SEARCH:
#                     search_name = get_search_state_name(search_phase)
#                     cv2.putText(display_img, f"Search: {search_name} [{search_pan_idx},{search_pitch_idx}]",
#                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#                 # Show manipulability
#                 manip = compute_manipulability(model, data)
#                 cv2.putText(display_img, f"Manip: {manip:.4f}", (CAM_WIDTH - 150, 60),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#                 # Show marker depth if available
#                 if corners is not None:
#                     display_depth = get_marker_depth_from_buffer(depth_buffer, corners, model)
#                     if display_depth is not None:
#                         depth_color = (0, 255, 0) if display_depth < PRE_GRASP_DISTANCE else (0, 165, 255)
#                         cv2.putText(display_img, f"Depth: {display_depth:.3f}m", (CAM_WIDTH - 150, 90),
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_color, 1)

#                 # Show mode indicator
#                 mode_text = "Auto-centered" if use_auto_desired else "Manual"
#                 cv2.putText(display_img, f"Desired: {mode_text}", (CAM_WIDTH - 180, 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#                 # Show gripper position
#                 cv2.putText(display_img, f"Gripper: {gripper_position:.2f} rad", (10, CAM_HEIGHT - 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

#                 # Display image
#                 cv2.imshow("IBVS Camera", display_img)

#                 # ========================================
#                 # KEYBOARD INPUT
#                 # ========================================
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == 27:  # ESC
#                     print("\nESC pressed - exiting")
#                     break
#                 elif key == ord('d') or key == ord('D'):
#                     if corners is not None:
#                         desired_features = corners.copy()
#                         use_auto_desired = False
#                         error_history = []
#                         print(f"Desired features set to current marker position")
#                 elif key == ord('a') or key == ord('A'):
#                     desired_features = compute_centered_desired_features()
#                     use_auto_desired = True
#                     error_history = []
#                     print("Using auto-centered desired features")
#                 elif key == ord('x') or key == ord('X'):
#                     desired_features = None
#                     use_auto_desired = False
#                     error_history = []
#                     print("Desired features cleared")
#                 elif key == ord('g') or key == ord('G'):
#                     # Start/restart pipeline
#                     if system_state == SystemState.INIT or system_state == SystemState.FAILED:
#                         if corners is not None:
#                             system_state = SystemState.IBVS_APPROACH
#                             print("Starting IBVS approach")
#                         elif ENABLE_SEARCH:
#                             system_state = SystemState.SEARCH
#                             search_phase = SearchPhase.IDLE
#                             print("Starting search for marker")
#                         else:
#                             print("No marker visible and search disabled")
#                     elif system_state == SystemState.COMPLETE:
#                         print("Grasp already complete. Press 'R' to reset.")
#                     else:
#                         print(f"Pipeline running in state: {get_system_state_name(system_state)}")
#                 elif key == ord('s') or key == ord('S'):
#                     # Toggle search manually
#                     if system_state != SystemState.SEARCH:
#                         system_state = SystemState.SEARCH
#                         search_phase = SearchPhase.IDLE
#                         print("Manual search started")
#                     else:
#                         system_state = SystemState.INIT
#                         print("Search stopped")
#                 elif key == ord('c') or key == ord('C'):
#                     # Manual graspability check
#                     if corners is not None:
#                         is_graspable, reasons, metrics = check_graspability(
#                             model, data, corners, depth_buffer,
#                             R_marker_to_cam_global, t_marker_in_cam,
#                             consecutive_detections, last_error_norm if error_history else 999
#                         )
#                         print(f"\nManual graspability check: {'PASS' if is_graspable else 'FAIL'}")
#                         for reason in reasons:
#                             print(f"  - {reason}")
#                         for key_m, val in metrics.items():
#                             print(f"  {key_m}: {val}")
#                     else:
#                         print("No marker detected for graspability check")
#                 elif key == ord('e') or key == ord('E'):
#                     # Manual grasp execute (skip checks)
#                     if corners is not None:
#                         system_state = SystemState.GRASP_EXECUTE
#                         grasp_state = GraspState.ORIENT_ALIGN
#                         cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
#                         if cube_id != -1:
#                             initial_cube_z = data.xpos[cube_id][2]
#                         print("Manual grasp execution started (checks skipped)")
#                     else:
#                         print("No marker detected for grasp")
#                 elif key == ord('r') or key == ord('R'):
#                     # Reset to initial state
#                     system_state = SystemState.INIT
#                     search_phase = SearchPhase.IDLE
#                     grasp_state = GraspState.ORIENT_ALIGN
#                     search_pan_idx = 0
#                     search_pitch_idx = 0
#                     search_dwell_counter = 0
#                     marker_lost_frames = 0
#                     consecutive_detections = 0
#                     gripper_position = GRIPPER_OPEN_POSITION
#                     error_history = []
#                     data.ctrl[5] = GRIPPER_OPEN_POSITION  # Open gripper
#                     print("State machine RESET to INIT")

#             frame_count += 1

#             # Update viewer
#             with viewer.lock():
#                 viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

#             viewer.sync()

#     cv2.destroyAllWindows()

#     # Print summary
#     print("\n" + "="*60)
#     print("Session Summary")
#     print("="*60)
#     print(f"Total frames: {frame_count}")
#     print(f"Control steps: {control_step}")
#     print(f"Duration: {time.time() - start_time:.1f} seconds")
#     print(f"Final state: {get_system_state_name(system_state)}")
#     if error_history:
#         print(f"Final error: {error_history[-1]:.1f} px")
#         print(f"Min error: {min(error_history):.1f} px")
#     if system_state == SystemState.COMPLETE:
#         print("Result: GRASP SUCCESSFUL")
#     elif system_state == SystemState.FAILED:
#         print("Result: GRASP FAILED")


# if __name__ == "__main__":
#     run_simulation()
