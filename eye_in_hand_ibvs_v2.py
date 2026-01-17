"""
eye_in_hand_ibvs_v2.py - Layer 1: Foundation

SO101 robot with eye-in-hand camera and ArUco cube scene.
This is the clean foundation for building IBVS control in layers.

Layer 1: Robot + Camera + Scene (this file)
Layer 2: ArUco detection + pose estimation (future)
Layer 3: IBVS control law (future)
Layer 4: Grasp execution (future)
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import cv2.aruco as aruco
import os
import time
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

# ArUco Detection
ARUCO_DICT = aruco.DICT_4X4_50


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

    # Create renderer
    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    print("\n=== Simulation Running ===")
    print("- MuJoCo viewer: drag robot with mouse")
    print("- OpenCV window: eye-in-hand camera view")
    print("- Press ESC to quit")

    frame_count = 0
    start_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Physics step
            mujoco.mj_step(model, data)

            # Render camera periodically
            if frame_count % RENDER_SKIP == 0:
                renderer.update_scene(data, camera=CAM_NAME)
                img = renderer.render()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Detect ArUco marker
                corners = detect_aruco_corners(img_bgr)
                if corners is not None:
                    # Draw corners on image
                    for i, corner in enumerate(corners):
                        cv2.circle(img_bgr, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
                        cv2.putText(img_bgr, str(i), tuple(corner.astype(int) + [5, -5]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # Draw polygon connecting corners
                    pts = corners.astype(int).reshape((-1, 1, 2))
                    cv2.polylines(img_bgr, [pts], True, (0, 255, 0), 2)
                    cv2.putText(img_bgr, "DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(img_bgr, "NO MARKER", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Eye-in-Hand Camera", img_bgr)
                if cv2.waitKey(1) == 27:  # ESC
                    break

            frame_count += 1
            viewer.sync()

    # Cleanup
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"\nSession ended: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")


if __name__ == "__main__":
    main()
