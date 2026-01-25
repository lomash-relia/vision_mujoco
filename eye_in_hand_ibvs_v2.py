"""
eye_in_hand_ibvs_v2.py - Layered IBVS System

SO101 robot with eye-in-hand camera and ArUco cube scene.

Layers implemented:
- Layer 1: Robot + Camera + Scene
- Layer 2: ArUco detection
- Layer 3: Search strategy (bird's eye view scanning)
- Layer 4: IBVS control law (position-based visual servoing)

Future layers:
- Layer 5: Grasp execution
"""


import mujoco
import mujoco.viewer
import numpy as np
import cv2
import cv2.aruco as aruco
import os
import time
import json
import glob
import xml.etree.ElementTree as ET

# ==========================================
# CONFIGURATION
# ==========================================
ROBOT_XML = r"model\so101_new_calib.xml"
SCENE_XML = r"model\scene_ibvs_cube.xml"
TEMP_ROBOT_XML = r"model\so101_with_cam.xml"
TEMP_SCENE_XML = r"model\scene_ibvs_with_cam.xml"

# Camera Parameters (tuned for IBVS approach)
CAM_NAME = "eye_in_hand_cam"

CAM_MOUNT_BODY = "gripper"
CAM_POS = "0.02 -0.12 0.03"
CAM_QUAT = "0.05 0 -0.26 0.964"
CAM_FOVY = "70"
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Render every N frames for performance
RENDER_SKIP = 5

# Physics steps per frame (compensate for small timestep=0.0002 for stable grasping)
PHYSICS_STEPS_PER_FRAME = 10

# ArUco Detection
ARUCO_DICT = aruco.DICT_4X4_50

# ==========================================
# SEARCH CONFIGURATION
# ==========================================
# Bird's eye view pose (elevated, camera ~60° down)
SEARCH_POSE = {
    'shoulder_lift': -0.8,   # Arm raised (elevated)
    'elbow_flex': 0.3,       # Elbow less bent - arm extends outward
    'wrist_flex': 1.2,       # Wrist tilted more toward ground (was 0.5)
    'wrist_roll': 0.0,
    'gripper': 2          # Gripper open
}

# Search pattern parameters
PAN_RANGE = (-1.0, 1.0)      # ±57° base rotation
PAN_STEP = 0.3               # ~17° increments (7 viewpoints)
ELBOW_VARIATIONS = [0.0, -0.3, 0.3]  # Mid/near/far (try center first)

# Detection requirements
REQUIRED_CONSECUTIVE = 2     # Detections needed to confirm
SETTLE_TIME = 0.2          # Seconds to wait after motion


# ==========================================
# IBVS CONFIGURATION
# ==========================================
# Control parameters
IBVS_LAMBDA = 0.05            # Proportional gain (conservative)
IBVS_DT = 0.02                 # Control timestep (seconds)
MAX_JOINT_VEL = 0.5            # Maximum joint velocity (rad/s)
CONVERGENCE_THRESHOLD = 8.0    # Pixel error threshold for convergence

# Physical marker size (from scene_ibvs_cube.xml: 0.018 half-extent = 0.036m full)
PHYSICAL_MARKER_SIZE = 0.036   # 36mm ArUco marker

# IBVS config directory (contains teaching tool outputs)
IBVS_CONFIG_DIR = r"ibvs_configs"

# Set to True to use actual taught corners from config, False for idealized square
USE_TAUGHT_CORNERS = True


def load_ibvs_config():
    """
    Load the most recent IBVS config from the config directory.
    Returns config dict or None if not found.
    """
    config_pattern = os.path.join(IBVS_CONFIG_DIR, "desired_features_*.json")
    config_files = sorted(glob.glob(config_pattern), reverse=True)  # Most recent first

    if not config_files:
        print(f"Warning: No config files found in {IBVS_CONFIG_DIR}")
        return None

    config_path = config_files[0]  # Use most recent
    print(f"Loading IBVS config: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


# Load config at module import
_IBVS_CONFIG = load_ibvs_config()

# Extract parameters from config (with fallbacks)
if _IBVS_CONFIG:
    DESIRED_MARKER_SIZE = _IBVS_CONFIG['desired_features']['marker_size_px']
    DESIRED_OFFSET_X = _IBVS_CONFIG['desired_features']['offset_x']
    DESIRED_OFFSET_Y = _IBVS_CONFIG['desired_features']['offset_y']
    DESIRED_CORNERS_TAUGHT = np.array(_IBVS_CONFIG['desired_features']['corners'])
else:
    # Fallback defaults
    DESIRED_MARKER_SIZE = 45.0
    DESIRED_OFFSET_X = 0.0
    DESIRED_OFFSET_Y = 100.0
    DESIRED_CORNERS_TAUGHT = None
    USE_TAUGHT_CORNERS = False

# Safety limits
MIN_GRIPPER_HEIGHT = 0.1     # Minimum Z height (meters) - stop IBVS below this


# ==========================================
# CAMERA INTRINSICS
# ==========================================
def compute_camera_intrinsics(fovy, width, height):
    """Compute camera intrinsic parameters from FOV and resolution."""
    fovy_rad = np.radians(float(fovy))
    fy = height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # Square pixels assumed
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

# Global camera intrinsics (computed once)
FX, FY, CX, CY = compute_camera_intrinsics(CAM_FOVY, CAM_WIDTH, CAM_HEIGHT)


# ==========================================
# FRAME TRANSFORMATIONS
# ==========================================
def quat_to_rotation_matrix(quat):
    """
    Convert quaternion to 3x3 rotation matrix.
    Args:
        quat: [w, x, y, z] quaternion (scalar-first convention)
    Returns:
        R: 3x3 rotation matrix
    """
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def get_camera_to_ee_transform():
    """
    Get transformation from camera frame to end-effector (gripper) frame.
    Based on camera mounting configuration in XML.

    Returns:
        R_cam_to_ee: 3x3 rotation matrix
        p_cam_in_ee: 3x1 position vector of camera in EE frame
    """
    # Parse camera position from config
    pos_parts = CAM_POS.split()
    p_cam_in_ee = np.array([float(p) for p in pos_parts])

    # Parse camera quaternion (CAM_QUAT string is stored as "x y z w")
    quat_parts = CAM_QUAT.split()
    quat = np.array([float(q) for q in quat_parts])
    # Convert from [x, y, z, w] to MuJoCo's [w, x, y, z] format
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

    R_cam_to_ee = quat_to_rotation_matrix(quat_wxyz)
    return R_cam_to_ee, p_cam_in_ee


def transform_velocity_cam_to_ee(v_cam, R_cam_to_ee, p_cam_in_ee):
    """
    Transform 6D velocity from camera frame to end-effector frame.
    Uses the adjoint transformation for rigid body velocities.

    Args:
        v_cam: [vx, vy, vz, wx, wy, wz] velocity in camera frame
        R_cam_to_ee: 3x3 rotation from camera to EE frame
        p_cam_in_ee: 3x1 position of camera origin in EE frame

    Returns:
        v_ee: [vx, vy, vz, wx, wy, wz] velocity in EE frame
    """
    v_linear_cam = v_cam[:3]
    omega_cam = v_cam[3:]

    # Rotate velocities to EE frame
    v_linear_ee = R_cam_to_ee @ v_linear_cam
    omega_ee = R_cam_to_ee @ omega_cam

    # Account for lever arm effect: v_ee = v_cam - omega x p = v_cam + p x omega
    v_linear_ee = v_linear_ee + np.cross(p_cam_in_ee, omega_ee)

    return np.concatenate([v_linear_ee, omega_ee])


# ==========================================
# IMAGE JACOBIAN (INTERACTION MATRIX)
# ==========================================
def compute_image_jacobian(points, depths):
    """
    Compute the image Jacobian (interaction matrix) for point features.

    Following Chaumette & Hutchinson formulation with proper focal length scaling.
    The interaction matrix relates image feature velocity to camera velocity:
        s_dot = L * v_c

    Args:
        points: (n, 2) array of image points (u, v) in pixels
        depths: (n,) array of depth values Z for each point

    Returns:
        L: (2n, 6) interaction matrix
    """
    L = []
    for i, (u, v) in enumerate(points):
        Z = depths[i]
        if Z <= 0:
            Z = 0.1  # Prevent division by zero

        # Normalized image coordinates
        x = (u - CX) / FX
        y = (v - CY) / FY

        # Interaction matrix for one point (2x6)
        # Pixel-based Chaumette formulation: ALL columns scaled by FX/FY
        # for dimensional consistency (error in pixels → velocity in m/s, rad/s)
        Lp = np.array([
            [-FX/Z,    0,    FX*x/Z,  FX*x*y,      -FX*(1+x*x),  FX*y],
            [   0,  -FY/Z,   FY*y/Z,  FY*(1+y*y),  -FY*x*y,     -FY*x]
        ])
        L.append(Lp)

    return np.vstack(L)  # (2n, 6)


# ==========================================
# DEPTH RENDERING
# ==========================================
def get_depth_at_points(depth_buffer, points, model):
    """
    Get metric depth Z for each feature point from MuJoCo depth buffer.

    MuJoCo's depth buffer uses OpenGL conventions where the depth is
    stored as normalized values between near and far planes.

    Args:
        depth_buffer: MuJoCo rendered depth buffer
        points: (n, 2) array of image points (u, v)
        model: MuJoCo model (for near/far plane info)

    Returns:
        depths: (n,) array of metric depth values
    """
    # Get depth range from model
    extent = model.stat.extent
    znear = model.vis.map.znear * extent
    zfar = model.vis.map.zfar * extent

    depths = []
    h, w = depth_buffer.shape

    for (u, v) in points:
        u_int = int(np.clip(u, 0, w - 1))
        v_int = int(np.clip(v, 0, h - 1))

        z_norm = depth_buffer[v_int, u_int]

        # Convert normalized depth to metric depth
        # Standard OpenGL depth linearization formula
        if z_norm < 1.0 and z_norm > 0.0:
            z_metric = znear * zfar / (zfar - z_norm * (zfar - znear))
        else:
            z_metric = zfar  # At infinity or invalid

        # Clamp to reasonable range
        # z_metric = np.clip(z_metric, 0.05, 5.0)
        depths.append(z_metric)

    return np.array(depths)


def estimate_depth_from_marker(corners, physical_size=PHYSICAL_MARKER_SIZE):
    """
    Estimate depth from marker size using pinhole camera model.

    More robust than depth buffer for IBVS since marker size is known.
    Formula: Z = (physical_size * fx) / pixel_size

    Args:
        corners: (4, 2) array of marker corner positions in pixels
        physical_size: Real-world marker size in meters

    Returns:
        depths: (4,) array with estimated depth for each corner (same value)
    """
    # Calculate marker size in pixels (average of 4 side lengths)
    side1 = np.linalg.norm(corners[1] - corners[0])
    side2 = np.linalg.norm(corners[2] - corners[1])
    side3 = np.linalg.norm(corners[3] - corners[2])
    side4 = np.linalg.norm(corners[0] - corners[3])
    marker_size_px = (side1 + side2 + side3 + side4) / 4

    # Estimate depth using pinhole camera model
    Z = (physical_size * FX) / marker_size_px

    # Return same depth for all 4 corners (planar marker assumption)
    return np.full(4, Z)


# ==========================================
# ROBOT JACOBIAN
# ==========================================
def get_robot_jacobian(model, data, body_name):
    """
    Get the geometric Jacobian at a body frame.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body

    Returns:
        J: (6, nv) Jacobian matrix [Jp; Jr] (position and rotation)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    return np.vstack([jacp, jacr])  # (6, nv)


# ==========================================
# DESIRED FEATURES
# ==========================================
def compute_desired_features(width=CAM_WIDTH, height=CAM_HEIGHT,
                             marker_size_px=DESIRED_MARKER_SIZE,
                             offset_x=DESIRED_OFFSET_X,
                             offset_y=DESIRED_OFFSET_Y):
    """
    Compute desired feature locations for marker target position.

    Args:
        width, height: Image dimensions
        marker_size_px: Desired marker size in pixels (smaller = further away)
        offset_x: Horizontal offset in pixels (positive = right in image)
        offset_y: Vertical offset in pixels (positive = down in image)

    Returns:
        desired_corners: (4, 2) array of corner positions
    """
    # Use actual taught corners if enabled and available (preserves marker orientation)
    if USE_TAUGHT_CORNERS and DESIRED_CORNERS_TAUGHT is not None:
        return DESIRED_CORNERS_TAUGHT.copy()

    # Otherwise generate idealized square corners
    cx_img = width / 2 + offset_x
    cy_img = height / 2 + offset_y
    half = marker_size_px / 2

    # Corners of a SQUARE, ordered by angle from centroid (ascending)
    # In image coords (Y down): arctan2(dy, dx)
    desired_corners = np.array([
        [cx_img - half, cy_img - half],  # Top-left (angle -3π/4)
        [cx_img + half, cy_img - half],  # Top-right (angle -π/4)
        [cx_img + half, cy_img + half],  # Bottom-right (angle +π/4)
        [cx_img - half, cy_img + half],  # Bottom-left (angle +3π/4)
    ])

    return desired_corners


# ==========================================
# IBVS CONTROL LAW
# ==========================================
def ibvs_control(current_features, desired_features, depths, model, data,
                 lambda_gain=IBVS_LAMBDA):
    """
    Compute joint velocities using IBVS control law.

    Classic IBVS: v_c = -lambda * L^+ * (s - s*)

    Args:
        current_features: (4, 2) current corner positions in pixels
        desired_features: (4, 2) desired corner positions in pixels
        depths: (4,) depth values for each corner
        model, data: MuJoCo model and data
        lambda_gain: proportional gain

    Returns:
        q_dot: (5,) joint velocities for arm joints
        error_norm: scalar pixel error norm
        v_camera: (6,) computed camera velocity
    """
    # 1. Compute position feature error (flatten to 8x1)
    error = (current_features - desired_features).flatten()
    error_norm = np.linalg.norm(error)
    print(error_norm)

    # 2. Compute image Jacobian (8x6)
    L = compute_image_jacobian(current_features, depths)

    # 3. Compute pseudo-inverse with damping for numerical stability
    damping = 0.01  # Reduced from 0.05
    L_pinv = L.T @ np.linalg.inv(L @ L.T + damping * np.eye(L.shape[0]))

    # 4. IBVS control law: camera velocity in camera frame
    # Use constant gain for stable convergence
    v_camera = -lambda_gain * L_pinv @ error

    # 4b. Flip Z for MuJoCo's camera convention (looks along -Z, not +Z)
    v_camera[2] = -v_camera[2]

    # 5. Transform camera velocity to end-effector frame
    R_cam_to_ee, p_cam_in_ee = get_camera_to_ee_transform()
    v_ee = transform_velocity_cam_to_ee(v_camera, R_cam_to_ee, p_cam_in_ee)

    # 6. Get robot Jacobian at gripper body (in WORLD frame)
    J_robot = get_robot_jacobian(model, data, CAM_MOUNT_BODY)

    # 7. Only use first 5 joints (exclude gripper actuator)
    J_arm = J_robot[:, :5]

    # 8. Transform v_ee from gripper frame to WORLD frame
    # MuJoCo's mj_jacBody returns Jacobian in world frame, so v_ee must also be in world frame
    R_gripper_to_world = data.body(CAM_MOUNT_BODY).xmat.reshape(3, 3)
    v_linear_world = R_gripper_to_world @ v_ee[:3]
    v_angular_world = R_gripper_to_world @ v_ee[3:]
    v_world = np.concatenate([v_linear_world, v_angular_world])

    # 9. Compute joint velocities using damped pseudo-inverse
    robot_damping = 0.02  # Reduced from 0.1
    J_pinv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + robot_damping * np.eye(6))
    q_dot = J_pinv @ v_world

    return q_dot, error_norm, v_camera


def clamp_joint_velocities(q_dot, max_vel=MAX_JOINT_VEL):
    """Clamp joint velocities to safe limits."""
    return np.clip(q_dot, -max_vel, max_vel)


# ==========================================
# VISUALIZATION HELPERS
# ==========================================
def draw_ibvs_overlay(image, current_corners, desired_corners, error_norm, state):
    """Draw IBVS visualization overlay on image."""
    img_display = image.copy()

    # Draw detected corners in green
    if current_corners is not None:
        for i, corner in enumerate(current_corners):
            cv2.circle(img_display, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
            cv2.putText(img_display, str(i), (int(corner[0])+10, int(corner[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        pts = current_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], True, (0, 255, 0), 2)

    # Draw desired corners in red
    if desired_corners is not None:
        for i, corner in enumerate(desired_corners):
            cv2.circle(img_display, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), 2)
        pts = desired_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], True, (0, 0, 255), 2)

        # Draw error lines from current to desired (yellow)
        if current_corners is not None:
            for curr, des in zip(current_corners, desired_corners):
                cv2.line(img_display,
                        (int(curr[0]), int(curr[1])),
                        (int(des[0]), int(des[1])),
                        (0, 255, 255), 1)

    # Draw state and error info
    state_colors = {
        "SEARCH": (0, 255, 255),      # Yellow
        "IBVS_APPROACH": (0, 165, 255), # Orange
        "CONVERGED": (0, 255, 0),      # Green
        "LOST": (0, 0, 255),           # Red
        "MANUAL": (255, 0, 255)        # Magenta
    }
    color = state_colors.get(state, (255, 255, 255))
    cv2.putText(img_display, f"State: {state}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if error_norm is not None:
        cv2.putText(img_display, f"Error: {error_norm:.1f} px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Key hints at bottom (focus this window for keys to work)
    h = image.shape[0]
    cv2.putText(img_display, "O=start R=reset ESC=quit (click window for keys)",
               (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return img_display


# ==========================================
# XML PATCHING
# ==========================================
def patch_robot_xml(input_path, output_path):
    """Inject camera into gripper body of robot XML."""
    print(f"Patching {input_path} to add camera on '{CAM_MOUNT_BODY}'...")
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()

        # Find mount body
        mount_body = None
        for body in root.iter('body'):
            if body.get('name') == CAM_MOUNT_BODY:
                mount_body = body
                break

        if mount_body is None:
            print(f"Error: Body '{CAM_MOUNT_BODY}' not found")
            return False

        # Remove existing camera if present
        for cam in mount_body.findall('camera'):
            if cam.get('name') == CAM_NAME:
                mount_body.remove(cam)

        # Add camera element
        cam_elem = ET.Element('camera')
        cam_elem.set('name', CAM_NAME)
        cam_elem.set('pos', CAM_POS)
        cam_elem.set('quat', CAM_QUAT)
        cam_elem.set('fovy', CAM_FOVY)
        cam_elem.set('mode', 'fixed')
        mount_body.insert(0, cam_elem)

        tree.write(output_path)
        print(f"Created: {output_path}")
        return True

    except Exception as e:
        print(f"Failed to patch robot XML: {e}")
        return False


def patch_scene_xml(scene_path, original_robot, new_robot, output_path):
    """Update scene to use patched robot XML."""
    try:
        with open(scene_path, 'r') as f:
            content = f.read()

        original_basename = os.path.basename(original_robot)
        new_basename = os.path.basename(new_robot)

        if original_basename in content:
            new_content = content.replace(original_basename, new_basename)
            with open(output_path, 'w') as f:
                f.write(new_content)
            print(f"Created: {output_path}")
            return True
        else:
            print(f"Warning: '{original_basename}' not found in scene")
            return False

    except Exception as e:
        print(f"Failed to patch scene XML: {e}")
        return False


# ==========================================
# ARUCO DETECTION
# ==========================================
aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)


def detect_aruco_corners(image):
    """
    Detect ArUco marker and return 4 corner points in consistent order.
    Corners sorted by angle from centroid (counter-clockwise from right).

    Returns:
        corners: (4, 2) array of corner positions or None if not detected
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # Get corners of first detected marker
    marker_corners = corners[0][0]  # Shape: (4, 2)

    # Sort corners consistently by angle from centroid
    centroid = np.mean(marker_corners, axis=0)
    angles = np.arctan2(marker_corners[:, 1] - centroid[1],
                        marker_corners[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)

    return marker_corners[sorted_indices]


# ==========================================
# SEARCH STRATEGY
# ==========================================
def move_to_search_pose(data, pan_angle, elbow_offset=0.0):
    """
    Set joints to bird's eye view pose with given pan angle.

    Args:
        data: MuJoCo data object
        pan_angle: Base rotation angle in radians
        elbow_offset: Offset to elbow_flex for depth variation
    """
    data.ctrl[0] = pan_angle                                    # shoulder_pan
    data.ctrl[1] = SEARCH_POSE['shoulder_lift']                 # shoulder_lift
    data.ctrl[2] = SEARCH_POSE['elbow_flex'] + elbow_offset     # elbow_flex
    data.ctrl[3] = SEARCH_POSE['wrist_flex']                    # wrist_flex
    data.ctrl[4] = SEARCH_POSE['wrist_roll']                    # wrist_roll
    data.ctrl[5] = SEARCH_POSE['gripper']                       # gripper


def execute_search(model, data, renderer, viewer):
    """
    Execute radial search pattern with depth variation.

    Scans workspace using base rotation + elbow variation to find ArUco marker.
    Requires REQUIRED_CONSECUTIVE consecutive detections to confirm.

    Returns:
        (found, corners): Tuple of (bool, corner array or None)
    """
    consecutive = 0
    steps_per_settle = int(SETTLE_TIME / model.opt.timestep)

    # Generate pan angles (center-out pattern)
    pan_angles = np.arange(PAN_RANGE[0], PAN_RANGE[1] + PAN_STEP, PAN_STEP)

    print(f"\n=== Starting ArUco Search ===")
    print(f"Pan angles: {len(pan_angles)} positions")
    print(f"Elbow variations: {ELBOW_VARIATIONS}")
    print(f"Required detections: {REQUIRED_CONSECUTIVE}")

    for elbow_idx, elbow_offset in enumerate(ELBOW_VARIATIONS):
        depth_name = ["mid", "near", "far"][elbow_idx]
        print(f"\n--- Depth level: {depth_name} (elbow offset: {elbow_offset}) ---")

        for pan_idx, pan in enumerate(pan_angles):
            # Move to search pose
            move_to_search_pose(data, pan, elbow_offset)

            # Simulate to let arm settle WITH continuous camera feed
            render_interval = max(1, steps_per_settle // 50)  # ~50 frames during settle
            for step in range(steps_per_settle):
                if not viewer.is_running():
                    print("Viewer closed")
                    return False, None

                for _ in range(PHYSICS_STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
                viewer.sync()

                # Render camera at interval for smooth but fast video
                if step % render_interval == 0:
                    renderer.update_scene(data, camera=CAM_NAME)
                    img = renderer.render()
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(img_bgr, f"Moving... pan={pan:.2f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Eye-in-Hand Camera", img_bgr)
                    if cv2.waitKey(1) == 27:
                        print("Search aborted by user")
                        return False, None

            # Final capture and detect after settling
            renderer.update_scene(data, camera=CAM_NAME)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            corners = detect_aruco_corners(img_bgr)

            # Draw current view
            if corners is not None:
                consecutive += 1
                # Draw detection
                for corner in corners:
                    cv2.circle(img_bgr, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
                pts = corners.astype(int).reshape((-1, 1, 2))
                cv2.polylines(img_bgr, [pts], True, (0, 255, 0), 2)
                cv2.putText(img_bgr, f"DETECTED {consecutive}/{REQUIRED_CONSECUTIVE}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"  Pan {pan_idx+1}/{len(pan_angles)}: DETECTED ({consecutive}/{REQUIRED_CONSECUTIVE})")

                if consecutive >= REQUIRED_CONSECUTIVE:
                    cv2.imshow("Eye-in-Hand Camera", img_bgr)
                    cv2.waitKey(100)
                    print(f"\n*** MARKER FOUND at pan={pan:.2f}, elbow_offset={elbow_offset} ***")
                    return True, corners
            else:
                consecutive = 0  # Reset on miss
                cv2.putText(img_bgr, f"Searching... pan={pan:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"  Pan {pan_idx+1}/{len(pan_angles)}: no marker")

            cv2.imshow("Eye-in-Hand Camera", img_bgr)
            if cv2.waitKey(1) == 27:  # ESC to abort
                print("Search aborted by user")
                return False, None

    print("\n*** MARKER NOT FOUND in workspace ***")
    return False, None


# ==========================================
# MAIN SIMULATION
# ==========================================
def main():
    # Patch XML files
    if not os.path.exists(ROBOT_XML):
        print(f"Error: {ROBOT_XML} not found")
        return

    if not patch_robot_xml(ROBOT_XML, TEMP_ROBOT_XML):
        return

    scene_ready = False
    if os.path.exists(SCENE_XML):
        scene_ready = patch_scene_xml(SCENE_XML, ROBOT_XML, TEMP_ROBOT_XML, TEMP_SCENE_XML)

    model_file = TEMP_SCENE_XML if scene_ready else TEMP_ROBOT_XML
    print(f"Loading: {model_file}")

    # Load model
    try:
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Find camera
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    if cam_id == -1:
        print(f"Error: Camera '{CAM_NAME}' not found")
        return
    print(f"Camera '{CAM_NAME}' found (id={cam_id})")

    # Create RGB renderer
    rgb_renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    # Compute desired features once
    desired_features = compute_desired_features()

    print("\n=== IBVS Visual Servoing System ===")
    print(f"Camera intrinsics: fx={FX:.1f}, fy={FY:.1f}, cx={CX:.1f}, cy={CY:.1f}")
    print(f"IBVS gain: {IBVS_LAMBDA}, convergence threshold: {CONVERGENCE_THRESHOLD} px")
    print(f"Desired marker size: {DESIRED_MARKER_SIZE} px")
    print("\nKeys (click camera window first):")
    print("  O = Start automatic pipeline (search + IBVS)")
    print("  R = Reset to manual mode")
    print("  ESC = Quit")

    # State machine - start in MANUAL mode
    state = "MANUAL"
    frame_count = 0
    control_step = 0
    start_time = time.time()
    error_norm = None
    prev_error_norm = float('inf')  # Track previous error to detect divergence
    marker_lost_frames = 0
    MARKER_LOST_THRESHOLD = 30  # Frames without detection before returning to search

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n=== MANUAL MODE ===")
        print("Robot is idle. Press O to start automatic pipeline.")

        # Main control loop
        while viewer.is_running():
            # Physics steps (multiple for small timestep stability)
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            # Control at reduced rate
            if frame_count % RENDER_SKIP == 0:
                # Render RGB image
                rgb_renderer.update_scene(data, camera=CAM_NAME)
                img = rgb_renderer.render()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Detect ArUco marker
                corners = detect_aruco_corners(img_bgr)

                # State machine logic
                if state == "IBVS_APPROACH":
                    if corners is not None:
                        marker_lost_frames = 0

                        # Safety: Check gripper height before control
                        gripper_pos = data.body("gripper").xpos
                        if gripper_pos[2] < MIN_GRIPPER_HEIGHT:
                            print(f"\n*** HEIGHT LIMIT REACHED (Z={gripper_pos[2]:.3f}m) ***")
                            print("Stopping IBVS to prevent collision")
                            state = "CONVERGED"
                            continue

                        # Get depth at corner points
                        depths = estimate_depth_from_marker(corners)

                        # Run IBVS control
                        q_dot, error_norm, v_cam = ibvs_control(
                            corners, desired_features, depths, model, data
                        )

                        # Restart IBVS if error starts increasing (iterative approach)
                        if error_norm > prev_error_norm + 1.0:
                            print(f"\n*** Error increasing ({prev_error_norm:.1f} → {error_norm:.1f})")
                            print("*** Restarting IBVS from current pose ***")
                            desired_features = corners.copy()  # Current features become new desired (keep 4x2 shape)
                            prev_error_norm = float('inf')  # Reset error tracking
                            continue  # Stay in IBVS_APPROACH, skip control this frame
                        prev_error_norm = error_norm

                        q_dot = clamp_joint_velocities(q_dot)

                        # Apply control
                        q_current = data.qpos[:5].copy()
                        q_new = q_current + q_dot * IBVS_DT

                        # Clamp to joint limits
                        for i in range(5):
                            ctrl_range = model.actuator_ctrlrange[i]
                            q_new[i] = np.clip(q_new[i], ctrl_range[0], ctrl_range[1])

                        data.ctrl[:5] = q_new
                        control_step += 1

                        # Check convergence
                        if error_norm < CONVERGENCE_THRESHOLD:
                            state = "CONVERGED"
                            print(f"\n*** IBVS CONVERGED at step {control_step} ***")
                            print(f"Final error: {error_norm:.2f} px")
                            print("Holding position...")
                    else:
                        # Lost marker during approach
                        marker_lost_frames += 1
                        if marker_lost_frames > MARKER_LOST_THRESHOLD:
                            state = "LOST"
                            print("\nMarker lost - returning to search mode")

                elif state == "CONVERGED":
                    # Hold position - just keep detecting for visualization
                    if corners is not None:
                        depths = estimate_depth_from_marker(corners)
                        _, error_norm, _ = ibvs_control(
                            corners, desired_features, depths, model, data
                        )
                        # Don't apply control, just monitor error
                    else:
                        marker_lost_frames += 1
                        if marker_lost_frames > MARKER_LOST_THRESHOLD:
                            state = "LOST"
                            print("\nMarker lost after convergence")

                elif state == "LOST":
                    # Check if marker becomes visible again
                    if corners is not None:
                        marker_lost_frames = 0
                        prev_error_norm = float('inf')  # Reset for fresh start
                        state = "IBVS_APPROACH"
                        print("\nMarker re-acquired - resuming IBVS approach")

                elif state == "MANUAL":
                    # Manual mode - no automatic control applied
                    # User controls robot via MuJoCo viewer
                    # Just update error display if marker visible
                    if corners is not None:
                        depths = estimate_depth_from_marker(corners)
                        _, error_norm, _ = ibvs_control(
                            corners, desired_features, depths, model, data
                        )

                # Draw visualization
                img_display = draw_ibvs_overlay(img_bgr, corners, desired_features, error_norm, state)

                # Show depth info if available
                if corners is not None:
                    depths = estimate_depth_from_marker(corners)
                    avg_depth = np.mean(depths)
                    
                    # print(f"Depths per corner: {[f'{d:.3f}' for d in depths]} → avg = {avg_depth:.3f} m")
                    cv2.putText(img_display, f"Depth: {avg_depth:.3f}m", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Eye-in-Hand Camera", img_display)

            # Key handling - check every frame (must focus OpenCV window to use keys)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                # Reset to manual mode
                state = "MANUAL"
                print("\n=== MANUAL MODE ===")
                print("Robot is idle. Press O to start automatic pipeline.")
            elif key == ord('o') or key == ord('O'):
                # Start automatic pipeline (search + IBVS)
                if state == "MANUAL":
                    print("\n=== Starting Automatic Pipeline ===")
                    marker_found, _ = execute_search(model, data, rgb_renderer, viewer)
                    if marker_found:
                        prev_error_norm = float('inf')  # Reset for fresh start
                        state = "IBVS_APPROACH"
                        print("Marker found - starting IBVS approach...")
                    else:
                        state = "LOST"
                        print("Marker not found - press O to retry or R for manual mode")

            frame_count += 1
            viewer.sync()

    # Cleanup
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"\n=== Session Summary ===")
    print(f"Total frames: {frame_count}")
    print(f"Control steps: {control_step}")
    print(f"Duration: {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    print(f"Final state: {state}")
    if error_norm is not None:
        print(f"Final error: {error_norm:.2f} px")


if __name__ == "__main__":
    main()
