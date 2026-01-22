"""
IBVS Teaching Tool - SIMPLIFIED VERSION

Uses MuJoCo's built-in joint controls (no redundant panel).
Just shows camera view and lets you save configurations.

MuJoCo Controls:
- Double-click body → Select and control
- Right-click → Context menu with perturbations
- Ctrl + Right-drag → Apply forces
- Scroll wheel → Zoom
- Right-drag → Rotate view

This Tool:
- Shows live camera feed in separate window
- Press 'S' to save current configuration
- Press 'C' to capture screenshot
- Press ESC to quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import cv2.aruco as aruco
import json
import os
from datetime import datetime
import xml.etree.ElementTree as ET

# ==========================================
# CONFIGURATION
# ==========================================
ROBOT_XML = r"D:\vision_mujoco\model\so101_new_calib.xml"
SCENE_XML = r"D:\vision_mujoco\model\scene_ibvs_cube.xml"
TEMP_ROBOT_XML = r"D:\vision_mujoco\model\so101_with_cam_teach.xml"

# Camera configuration
CAM_NAME = "eye_in_hand_cam"
CAM_MOUNT_BODY = "gripper"
CAM_POS = "0.01 -0.12 0.03"
CAM_QUAT = "0.05 0 -0.26 0.964"
CAM_FOVY = "70"
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ArUco detection
ARUCO_DICT = aruco.DICT_4X4_50

# Output directory
OUTPUT_DIR = "ibvs_configs"

# Physics steps per frame (increase for faster response)
PHYSICS_STEPS = 20  # Run 20 physics steps per frame for faster settling

# ==========================================
# CAMERA INTRINSICS
# ==========================================
def compute_camera_intrinsics(fovy, width, height):
    fovy_rad = np.radians(float(fovy))
    fy = height / (2 * np.tan(fovy_rad / 2))
    fx = fy
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

FX, FY, CX, CY = compute_camera_intrinsics(CAM_FOVY, CAM_WIDTH, CAM_HEIGHT)

# ==========================================
# XML PATCHING
# ==========================================
def patch_robot_xml(input_path, output_path):
    """Add camera to robot XML."""
    print(f"Patching robot XML: {input_path}")
    
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # Find gripper body
        gripper_body = None
        for body in root.iter('body'):
            if body.get('name') == CAM_MOUNT_BODY:
                gripper_body = body
                break
        
        if gripper_body is None:
            raise ValueError(f"Body '{CAM_MOUNT_BODY}' not found")
        
        # Remove existing camera
        for cam in list(gripper_body.iter('camera')):
            if cam.get('name') == CAM_NAME:
                gripper_body.remove(cam)
        
        # Add camera
        camera = ET.SubElement(gripper_body, 'camera')
        camera.set('name', CAM_NAME)
        camera.set('mode', 'fixed')
        camera.set('pos', CAM_POS)
        camera.set('quat', CAM_QUAT)
        camera.set('fovy', CAM_FOVY)
        
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"  ✓ Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def load_scene_with_robot(robot_xml_path, scene_xml_path):
    """Load scene with robot (tries multiple strategies)."""
    print("\nLoading MuJoCo model...")
    
    # Strategy 1: Scene XML
    try:
        if os.path.exists(scene_xml_path):
            scene_tree = ET.parse(scene_xml_path)
            scene_root = scene_tree.getroot()
            
            # Update robot include
            for include in scene_root.iter('include'):
                file_attr = include.get('file', '')
                if 'so101' in file_attr.lower():
                    include.set('file', os.path.basename(robot_xml_path))
            
            temp_scene = scene_xml_path.replace('.xml', '_with_cam_teach.xml')
            scene_tree.write(temp_scene, encoding='utf-8', xml_declaration=True)
            
            model = mujoco.MjModel.from_xml_path(temp_scene)
            print(f"  ✓ Loaded from: {temp_scene}")
            return model, temp_scene
    except Exception as e:
        print(f"  Strategy 1 failed: {e}")
    
    # Strategy 2: Robot only
    try:
        model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print(f"  ✓ Loaded robot only")
        return model, robot_xml_path
    except Exception as e:
        print(f"  Strategy 2 failed: {e}")
    
    # Strategy 3: Create combined
    try:
        combined_xml = f"""<mujoco model="teaching_scene">
    <compiler angle="radian" autolimits="true"/>
    <option timestep="0.002" integrator="RK4"/>
    <visual>
        <global offwidth="640" offheight="480"/>
    </visual>
    <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="1 1"/>
    </asset>
    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
        <body name="aruco_cube" pos="0.3 0 0.025">
            <freejoint/>
            <geom name="cube" type="box" size="0.025 0.025 0.025" rgba="1 1 1 1"/>
        </body>
        <include file="{os.path.basename(robot_xml_path)}"/>
    </worldbody>
</mujoco>"""
        combined_path = robot_xml_path.replace('.xml', '_combined.xml')
        with open(combined_path, 'w') as f:
            f.write(combined_xml)
        
        model = mujoco.MjModel.from_xml_path(combined_path)
        print(f"  ✓ Created combined scene")
        return model, combined_path
    except Exception as e:
        print(f"  Strategy 3 failed: {e}")
    
    raise RuntimeError("Could not load model")

# ==========================================
# ARUCO DETECTION
# ==========================================
def detect_aruco_marker(image):
    """Detect ArUco marker."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    corners_list, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        corners = corners_list[0].reshape(4, 2)
        center = np.mean(corners, axis=0)
        
        # Sort corners
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        corners = corners[sorted_idx]
        
        return corners, center, ids[0][0]
    
    return None, None, None

# ==========================================
# VISUALIZATION
# ==========================================
def draw_camera_overlay(image, corners, center, marker_id, gripper_pos):
    """Draw overlay on camera image."""
    img_display = image.copy()
    h, w = image.shape[:2]
    
    # Grid
    for i in range(0, w, 40):
        cv2.line(img_display, (i, 0), (i, h), (60, 60, 60), 1)
    for i in range(0, h, 40):
        cv2.line(img_display, (0, i), (w, i), (60, 60, 60), 1)
    
    # Image center
    cv2.line(img_display, (int(CX)-15, int(CY)), (int(CX)+15, int(CY)), (100, 100, 100), 1)
    cv2.line(img_display, (int(CX), int(CY)-15), (int(CX), int(CY)+15), (100, 100, 100), 1)
    
    if corners is not None and center is not None:
        # Marker corners
        for i, corner in enumerate(corners):
            cv2.circle(img_display, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
            cv2.putText(img_display, str(i), (int(corner[0])+8, int(corner[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], True, (0, 255, 0), 2)
        
        # Center
        cv2.circle(img_display, (int(center[0]), int(center[1])), 6, (0, 255, 255), 2)
        cv2.circle(img_display, (int(center[0]), int(center[1])), 2, (0, 255, 255), -1)
        
        # Offset line
        cv2.line(img_display, (int(CX), int(CY)), 
                (int(center[0]), int(center[1])), (255, 0, 255), 2)
        
        # Measurements
        offset_x = center[0] - CX
        offset_y = center[1] - CY
        
        edge_lengths = [np.linalg.norm(corners[i] - corners[(i+1)%4]) for i in range(4)]
        avg_size = np.mean(edge_lengths)
        
        # Info panel
        info = [
            f"Marker ID: {marker_id}",
            f"Center: ({center[0]:.0f}, {center[1]:.0f}) px",
            f"Offset: ({offset_x:+.0f}, {offset_y:+.0f}) px",
            f"Size: {avg_size:.0f} px",
            f"Height: {gripper_pos[2]:.3f} m",
            "",
            "Press 'S' to SAVE"
        ]
        
        # Semi-transparent background
        overlay = img_display.copy()
        cv2.rectangle(overlay, (5, 5), (280, 5 + len(info) * 22 + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, img_display, 0.25, 0, img_display)
        
        # Draw text
        for i, text in enumerate(info):
            color = (0, 255, 0) if i < len(info) - 1 else (0, 255, 255)
            cv2.putText(img_display, text, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        # No marker
        cv2.putText(img_display, "NO MARKER", (int(w/2 - 80), int(h/2)),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(img_display, "Adjust robot to see marker", 
                   (int(w/2 - 140), int(h/2 + 35)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    
    # Instructions at bottom
    cv2.rectangle(img_display, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(img_display, "Use MuJoCo viewer to control robot", 
               (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(img_display, "S=Save | C=Screenshot | ESC=Quit",
               (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_display

# ==========================================
# SAVE CONFIGURATION
# ==========================================
def save_configuration(corners, center, marker_id, gripper_pos, joint_positions):
    """Save desired features configuration."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    offset_x = center[0] - CX
    offset_y = center[1] - CY
    
    edge_lengths = [np.linalg.norm(corners[i] - corners[(i+1)%4]) for i in range(4)]
    avg_size = np.mean(edge_lengths)
    
    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'marker_id': int(marker_id),
        'camera': {
            'width': CAM_WIDTH,
            'height': CAM_HEIGHT,
            'fx': FX,
            'fy': FY,
            'cx': CX,
            'cy': CY,
            'fovy': float(CAM_FOVY)
        },
        'desired_features': {
            'corners': corners.tolist(),
            'center': center.tolist(),
            'offset_x': float(offset_x),
            'offset_y': float(offset_y),
            'marker_size_px': float(avg_size)
        },
        'robot_state': {
            'joint_positions': joint_positions.tolist(),
            'joint_names': ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'],
            'gripper_position': gripper_pos.tolist(),
            'gripper_height': float(gripper_pos[2])
        }
    }
    
    # Generate code snippet
    code = f"""# IBVS Configuration - {config['timestamp']}

import numpy as np

# Constants for your IBVS controller
DESIRED_MARKER_SIZE = {avg_size:.1f}  # pixels
DESIRED_OFFSET_X = {offset_x:.1f}     # pixels (+ = right)
DESIRED_OFFSET_Y = {offset_y:.1f}     # pixels (+ = down)

# IMPORTANT: If offset_y is positive and large, gripper will move DOWN!
# For eye-in-hand camera, positive Y means marker appears BELOW center
# which requires gripper to descend. Check if this is safe!

def compute_desired_features(width={CAM_WIDTH}, height={CAM_HEIGHT}):
    \"\"\"Generate desired feature corners.\"\"\"
    cx_img = width / 2 + {offset_x:.1f}
    cy_img = height / 2 + {offset_y:.1f}
    half = {avg_size:.1f} / 2
    
    return np.array([
        [cx_img - half, cy_img - half],  # Top-left
        [cx_img + half, cy_img - half],  # Top-right
        [cx_img + half, cy_img + half],  # Bottom-right
        [cx_img - half, cy_img + half],  # Bottom-left
    ])

# Direct corner positions (measured)
DESIRED_CORNERS = np.array({corners.tolist()})

# Robot state at this configuration
JOINT_POSITIONS = {joint_positions.tolist()}
GRIPPER_HEIGHT = {gripper_pos[2]:.3f}  # meters
"""
    
    # Save JSON
    filename = f"desired_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(OUTPUT_DIR, filename)
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save Python code
    code_path = os.path.join(OUTPUT_DIR, filename.replace('.json', '.py'))
    with open(code_path, 'w') as f:
        f.write(code)
    
    print(f"\n{'='*65}")
    print(f"✓ CONFIGURATION SAVED")
    print(f"{'='*65}")
    print(f"  JSON: {json_path}")
    print(f"  Code: {code_path}")
    print(f"\n  Marker ID: {marker_id}")
    print(f"  Center: ({center[0]:.0f}, {center[1]:.0f}) px")
    print(f"  Offset: ({offset_x:+.0f}, {offset_y:+.0f}) px")
    print(f"  Size: {avg_size:.0f} px")
    print(f"  Height: {gripper_pos[2]:.3f} m")
    
    # Safety warning
    if offset_y > 50:
        print(f"\n  ⚠️  WARNING: Large positive offset_y = {offset_y:.0f}px")
        print(f"      This means marker is BELOW center")
        print(f"      IBVS will move gripper DOWN - check if safe!")
    elif gripper_pos[2] < 0.10:
        print(f"\n  ⚠️  WARNING: Low height = {gripper_pos[2]:.3f}m")
        print(f"      Risk of ground collision!")
    else:
        print(f"\n  ✓ Configuration looks safe")
    
    print(f"{'='*65}\n")

# ==========================================
# MAIN
# ==========================================
def main():
    print("\n" + "="*70)
    print("IBVS TEACHING TOOL - Simplified Version")
    print("="*70)
    print("\nThis tool:")
    print("  • Shows live camera view from gripper")
    print("  • Lets you save desired feature positions")
    print("  • Uses MuJoCo's built-in controls for robot manipulation")
    print("\nMuJoCo Controls:")
    print("  • Double-click any body to select and control it")
    print("  • Right-click for context menu (apply forces, etc.)")
    print("  • Ctrl+Right-drag to apply forces")
    print("  • Use MuJoCo UI sliders to adjust joints")
    print("\nThis Tool's Keys:")
    print("  • S = Save current configuration")
    print("  • C = Capture screenshot")
    print("  • ESC = Quit")
    print("="*70 + "\n")
    
    # Patch and load
    if not patch_robot_xml(ROBOT_XML, TEMP_ROBOT_XML):
        return
    
    try:
        model, _ = load_scene_with_robot(TEMP_ROBOT_XML, SCENE_XML)
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return
    
    data = mujoco.MjData(model)
    
    # Check camera
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        if cam_id == -1:
            raise ValueError(f"Camera not found")
        print(f"✓ Camera '{CAM_NAME}' ready (ID: {cam_id})")
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return
    
    # Initialize renderer (HEIGHT, WIDTH order!)
    rgb_renderer = mujoco.Renderer(model, CAM_HEIGHT, CAM_WIDTH)
    
    print("\n" + "="*70)
    print("READY! MuJoCo viewer and camera window are now active.")
    print("="*70 + "\n")
    
    # Main loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_count = 0
        
        while viewer.is_running():
            # Run multiple physics steps for faster response
            for _ in range(PHYSICS_STEPS):
                mujoco.mj_step(model, data)
            
            # Update camera view every frame
            if frame_count % 1 == 0:  # Update every frame
                try:
                    rgb_renderer.update_scene(data, camera=CAM_NAME)
                    img = rgb_renderer.render()
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    corners, center, marker_id = detect_aruco_marker(img_bgr)
                    
                    try:
                        gripper_pos = data.body("gripper").xpos
                    except:
                        gripper_pos = np.array([0, 0, 0])
                    
                    img_display = draw_camera_overlay(img_bgr, corners, center, 
                                                     marker_id, gripper_pos)
                    
                    cv2.imshow("Gripper Camera (Eye-in-Hand)", img_display)
                    
                except Exception as e:
                    print(f"Camera error: {e}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            elif key == ord('s') or key == ord('S'):
                if corners is not None:
                    try:
                        joint_positions = data.qpos[:5].copy()
                        gripper_pos = data.body("gripper").xpos
                        save_configuration(corners, center, marker_id, 
                                         gripper_pos, joint_positions)
                    except Exception as e:
                        print(f"✗ Save error: {e}")
                else:
                    print("✗ No marker detected - cannot save")
            
            elif key == ord('c') or key == ord('C'):
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, img_display)
                print(f"✓ Screenshot: {filepath}")
            
            frame_count += 1
            viewer.sync()
    
    cv2.destroyAllWindows()
    print("\n" + "="*70)
    print("Teaching session complete!")
    print(f"Saved configs: {os.path.abspath(OUTPUT_DIR)}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()