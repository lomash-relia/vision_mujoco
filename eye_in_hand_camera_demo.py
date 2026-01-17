import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
ROBOT_XML = r"model\so101_new_calib.xml"
SCENE_XML = r"model\scene.xml"
TEMP_ROBOT_XML = r"model\so101_with_cam.xml"
TEMP_SCENE_XML = r"model\scene_with_cam.xml"
IMG_DIR = "img"

# Performance settings
CAMERA_RENDER_SKIP = 5  # Render camera every N frames

# Camera Parameters
CAM_NAME = "eye_in_hand_cam"
CAM_MOUNT_BODY = "gripper"  # Mount on gripper
CAM_POS = "0.02 -0.09 0.02"  # More distance from gripper
CAM_QUAT = "0 0 -0.259 0.966"  
CAM_FOVY = "70"


# ==========================================
# 1. XML PATCHING UTILITIES
# ==========================================
def patch_robot_xml(input_path, output_path):
    """
    Injects a camera definition into the specified body of the robot XML.
    """
    print(f"Patching {input_path} to add camera on '{CAM_MOUNT_BODY}' body...")
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()

        # Find the mount body recursively
        mount_body = None
        for body in root.iter('body'):
            if body.get('name') == CAM_MOUNT_BODY:
                mount_body = body
                break

        if mount_body is None:
            print(f"Error: Could not find '{CAM_MOUNT_BODY}' body in {input_path}")
            return False

        # Remove existing camera if present (to allow re-patching with new params)
        for cam in mount_body.findall('camera'):
            if cam.get('name') == CAM_NAME:
                mount_body.remove(cam)
                print(f"Removed existing camera '{CAM_NAME}' for re-patching.")

        # Create new camera element
        cam_elem = ET.Element('camera')
        cam_elem.set('name', CAM_NAME)
        cam_elem.set('pos', CAM_POS)
        cam_elem.set('quat', CAM_QUAT)
        cam_elem.set('fovy', CAM_FOVY)
        cam_elem.set('mode', 'fixed')

        # Insert camera as the first child of the mount body
        mount_body.insert(0, cam_elem)

        tree.write(output_path)
        print(f"Created patched robot file: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to patch robot XML: {e}")
        return False


def patch_scene_xml(scene_path, original_robot, new_robot, output_path):
    """
    Updates the scene file to point to the new robot XML file.
    """
    try:
        # Simple string replacement is safer for includes than parsing
        with open(scene_path, 'r') as f:
            content = f.read()
        
        # Try different possible filenames in scene.xml
        original_basename = os.path.basename(original_robot)
        new_basename = os.path.basename(new_robot)
        
        if original_basename in content:
            new_content = content.replace(original_basename, new_basename)
            with open(output_path, 'w') as f:
                f.write(new_content)
            print(f"Created patched scene file: {output_path}")
            return True
        else:
            print(f"Warning: Could not find reference to '{original_basename}' in '{scene_path}'.")
            print("Loading patched robot XML directly instead of scene.")
            return False
            
    except Exception as e:
        print(f"Failed to patch scene XML: {e}")
        return False


# ==========================================
# 2. MAIN SIMULATION LOOP
# ==========================================
def run_simulation():
    # Setup directories
    Path(IMG_DIR).mkdir(exist_ok=True)
    
    # 1. Prepare XML files
    if not os.path.exists(ROBOT_XML):
        print(f"Error: {ROBOT_XML} not found!")
        return

    xml_ready = patch_robot_xml(ROBOT_XML, TEMP_ROBOT_XML)
    if not xml_ready:
        return

    # Try to patch scene, otherwise load robot directly
    scene_ready = False
    if os.path.exists(SCENE_XML):
        scene_ready = patch_scene_xml(SCENE_XML, ROBOT_XML, TEMP_ROBOT_XML, TEMP_SCENE_XML)
    
    model_file = TEMP_SCENE_XML if scene_ready else TEMP_ROBOT_XML
    print(f"Loading model from: {model_file}")

    # 2. Load Model
    try:
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # 3. Setup Camera
    try:
        # Find camera ID
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
        if cam_id == -1:
            raise ValueError(f"Camera '{CAM_NAME}' not found")
        print(f"Camera '{CAM_NAME}' found with ID: {cam_id}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Initialize renderer for eye-in-hand camera
    renderer = mujoco.Renderer(model, height=480, width=640)

    print("\nSimulation Running...")
    print(f" - Eye-in-Hand Camera: Separate OpenCV window")
    print(f" - Interact: Drag robot in MuJoCo viewer with mouse")
    print(f" - Saving: Every 10th frame to '{IMG_DIR}/'")
    print(f" - Camera renders every {CAMERA_RENDER_SKIP} frames")
    print(f" - Press ESC in camera window or close viewer to quit")
    print(f" - Duration: 1200 seconds (20 minutes)")

    frame_count = 0
    start = time.time()
    img_bgr = None  # Cache last rendered image
    
    # Use context manager for proper viewer interaction
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and time.time() - start < 1200:
            # NO step_start timing - let it run as fast as possible

            # Physics step (always runs at full speed)
            mujoco.mj_step(model, data)

            # Render camera only every N frames for performance
            if frame_count % CAMERA_RENDER_SKIP == 0:
                renderer.update_scene(data, camera=CAM_NAME)
                img = renderer.render()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Display and check for key press only when rendering
                cv2.imshow("Eye-in-Hand Camera", img_bgr)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    print("\nESC pressed in camera window")
                    break

            # Save every 10th frame
            if frame_count % 10 == 0 and img_bgr is not None:
                fname = os.path.join(IMG_DIR, f"frame_{frame_count:06d}.png")
                cv2.imwrite(fname, img_bgr)
                if frame_count % 100 == 0:
                    fps = frame_count / (time.time() - start)
                    print(f"Time: {time.time()-start:.1f}s | Frame: {frame_count} | FPS: {fps:.1f}")
            
            frame_count += 1

            # Example viewer option modification
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

            # Pick up changes to physics state, apply perturbations, update GUI
            viewer.sync()

            # REMOVED: time.sleep() - Let simulation run at full speed!

    # Cleanup
    cv2.destroyAllWindows()
    print(f"\nSession ended.")
    print(f"Total frames: {frame_count}")
    print(f"Images saved: {frame_count // 10}")
    print(f"Duration: {time.time() - start:.1f} seconds")
    print(f"Average FPS: {frame_count / (time.time() - start):.1f}")


if __name__ == "__main__":
    run_simulation()
