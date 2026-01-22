"""
IBVS Configuration Viewer - Review and Compare Saved Poses

This tool allows you to:
1. Load saved IBVS configurations
2. Visualize the desired features
3. Restore robot to saved poses
4. Compare multiple configurations side-by-side
5. Generate code snippets for your controller

Usage:
    python ibvs_config_viewer.py [config_file.json]

Or run without arguments to see list of available configs.
"""

import json
import os
import numpy as np
import cv2
from datetime import datetime
import argparse

OUTPUT_DIR = "ibvs_configs"

def load_config(filepath):
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def visualize_desired_features(config, image_size=(640, 480)):
    """
    Create visualization of desired features.
    
    Args:
        config: Configuration dictionary
        image_size: (width, height) of image
    
    Returns:
        BGR image showing desired features
    """
    width, height = image_size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw grid
    for i in range(0, width, 40):
        cv2.line(img, (i, 0), (i, height), (40, 40, 40), 1)
    for i in range(0, height, 40):
        cv2.line(img, (0, i), (width, i), (40, 40, 40), 1)
    
    # Get camera params
    cx = config['camera']['cx']
    cy = config['camera']['cy']
    
    # Draw image center
    cv2.line(img, (int(cx)-20, int(cy)), (int(cx)+20, int(cy)), (100, 100, 100), 2)
    cv2.line(img, (int(cx), int(cy)-20), (int(cx), int(cy)+20), (100, 100, 100), 2)
    cv2.putText(img, "Image Center", (int(cx)+10, int(cy)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Draw desired marker
    corners = np.array(config['desired_features']['corners'])
    center = np.array(config['desired_features']['center'])
    
    # Draw corners
    for i, corner in enumerate(corners):
        cv2.circle(img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), -1)
        cv2.putText(img, f"C{i}", (int(corner[0])+12, int(corner[1])+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw boundary
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 0, 255), 3)
    
    # Draw center
    cv2.circle(img, (int(center[0]), int(center[1])), 10, (0, 255, 255), 2)
    cv2.circle(img, (int(center[0]), int(center[1])), 3, (0, 255, 255), -1)
    
    # Draw offset line
    cv2.line(img, (int(cx), int(cy)), (int(center[0]), int(center[1])),
            (255, 0, 255), 2)
    
    # Add measurements
    offset_x = config['desired_features']['offset_x']
    offset_y = config['desired_features']['offset_y']
    marker_size = config['desired_features']['marker_size_px']
    gripper_height = config['robot_state']['gripper_height']
    
    # Info panel
    y = 30
    info_texts = [
        f"Marker Size: {marker_size:.1f} px",
        f"Center Offset: ({offset_x:+.1f}, {offset_y:+.1f}) px",
        f"Gripper Height: {gripper_height:.3f} m",
        "",
        f"Configuration from:",
        f"{config['timestamp']}"
    ]
    
    for text in info_texts:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        y += 25
    
    # Safety indicator
    if gripper_height < 0.10:
        cv2.putText(img, "⚠ LOW HEIGHT - UNSAFE!", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif gripper_height < 0.15:
        cv2.putText(img, "⚠ Moderate height", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    else:
        cv2.putText(img, "✓ Safe height", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    return img

def print_config_summary(config, filepath):
    """Print configuration summary to console."""
    print("\n" + "="*70)
    print(f"Configuration: {os.path.basename(filepath)}")
    print("="*70)
    print(f"Created: {config['timestamp']}")
    print(f"Marker ID: {config['marker_id']}")
    
    print("\n📷 Camera Parameters:")
    cam = config['camera']
    print(f"  Resolution: {cam['width']}x{cam['height']}")
    print(f"  Focal length: fx={cam['fx']:.1f}, fy={cam['fy']:.1f}")
    print(f"  Principal point: cx={cam['cx']:.1f}, cy={cam['cy']:.1f}")
    print(f"  FOV: {cam['fovy']}°")
    
    print("\n🎯 Desired Features:")
    feat = config['desired_features']
    print(f"  Marker center: ({feat['center'][0]:.1f}, {feat['center'][1]:.1f}) px")
    print(f"  Offset from image center: ({feat['offset_x']:+.1f}, {feat['offset_y']:+.1f}) px")
    print(f"  Marker size: {feat['marker_size_px']:.1f} px")
    
    print("\n🤖 Robot State:")
    robot = config['robot_state']
    print(f"  Gripper position: ({robot['gripper_position'][0]:.3f}, "
          f"{robot['gripper_position'][1]:.3f}, {robot['gripper_position'][2]:.3f}) m")
    print(f"  Gripper height: {robot['gripper_height']:.3f} m")
    print("\n  Joint angles (rad):")
    for name, val in zip(robot['joint_names'], robot['joint_positions']):
        print(f"    {name:15s}: {val:+.4f}")
    
    print("="*70 + "\n")

def compare_configs(config_files):
    """Compare multiple configurations side-by-side."""
    if len(config_files) < 2:
        print("Need at least 2 configurations to compare")
        return
    
    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)
    
    configs = [load_config(f) for f in config_files]
    
    # Create comparison table
    headers = ["Property"] + [f"Config {i+1}" for i in range(len(configs))]
    
    print(f"\n{'Property':<30}" + "".join(f"{'Config ' + str(i+1):>15}" for i in range(len(configs))))
    print("-" * (30 + 15 * len(configs)))
    
    # Compare key metrics
    metrics = [
        ("Timestamp", lambda c: c['timestamp'][:16]),
        ("Marker Size (px)", lambda c: f"{c['desired_features']['marker_size_px']:.1f}"),
        ("Offset X (px)", lambda c: f"{c['desired_features']['offset_x']:+.1f}"),
        ("Offset Y (px)", lambda c: f"{c['desired_features']['offset_y']:+.1f}"),
        ("Gripper Height (m)", lambda c: f"{c['robot_state']['gripper_height']:.3f}"),
        ("Shoulder Pan", lambda c: f"{c['robot_state']['joint_positions'][0]:+.3f}"),
        ("Shoulder Lift", lambda c: f"{c['robot_state']['joint_positions'][1]:+.3f}"),
        ("Elbow Flex", lambda c: f"{c['robot_state']['joint_positions'][2]:+.3f}"),
    ]
    
    for name, getter in metrics:
        row = f"{name:<30}"
        for config in configs:
            try:
                value = getter(config)
                row += f"{value:>15}"
            except:
                row += f"{'N/A':>15}"
        print(row)
    
    print("\n" + "="*70)
    
    # Show visualizations
    print("\nPress any key to cycle through visualizations (ESC to exit)...")
    
    for i, (config, filepath) in enumerate(zip(configs, config_files)):
        img = visualize_desired_features(config)
        
        # Add title
        title = f"Config {i+1}: {os.path.basename(filepath)}"
        cv2.putText(img, title, (10, img.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Configuration Comparison", img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def list_available_configs():
    """List all available configuration files."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"No configurations found (directory '{OUTPUT_DIR}' does not exist)")
        return []
    
    json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    
    if not json_files:
        print(f"No configurations found in '{OUTPUT_DIR}/'")
        return []
    
    print("\n" + "="*70)
    print("AVAILABLE CONFIGURATIONS")
    print("="*70)
    
    configs = []
    for i, filename in enumerate(sorted(json_files), 1):
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            config = load_config(filepath)
            configs.append((filepath, config))
            
            timestamp = config['timestamp']
            marker_size = config['desired_features']['marker_size_px']
            offset_x = config['desired_features']['offset_x']
            offset_y = config['desired_features']['offset_y']
            height = config['robot_state']['gripper_height']
            
            print(f"\n{i}. {filename}")
            print(f"   Created: {timestamp}")
            print(f"   Marker: {marker_size:.0f}px @ offset({offset_x:+.0f}, {offset_y:+.0f})px")
            print(f"   Height: {height:.3f}m")
        except Exception as e:
            print(f"\n{i}. {filename} - Error loading: {e}")
    
    print("\n" + "="*70 + "\n")
    
    return configs

def generate_code_for_config(config, output_file=None):
    """Generate Python code snippet from configuration."""
    code = config['code_snippets']['python_constants']
    code += "\n\n"
    code += config['code_snippets']['python_function']
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"Code snippet saved to: {output_file}")
    
    return code

def main():
    """Main viewer interface."""
    parser = argparse.ArgumentParser(
        description='IBVS Configuration Viewer - Review saved teaching poses'
    )
    parser.add_argument('config', nargs='*', help='Configuration file(s) to view')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List all available configurations')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare multiple configurations side-by-side')
    parser.add_argument('--export-code', '-e', metavar='OUTPUT',
                       help='Export Python code to file')
    
    args = parser.parse_args()
    
    # List mode
    if args.list or not args.config:
        configs = list_available_configs()
        if not configs:
            return
        
        print("To view a configuration: python ibvs_config_viewer.py <config_file>")
        print("To compare configurations: python ibvs_config_viewer.py --compare <config1> <config2> ...")
        return
    
    # Load config files
    config_files = []
    for cfg in args.config:
        if os.path.exists(cfg):
            config_files.append(cfg)
        else:
            # Try in OUTPUT_DIR
            alt_path = os.path.join(OUTPUT_DIR, cfg)
            if os.path.exists(alt_path):
                config_files.append(alt_path)
            else:
                print(f"⚠ Configuration not found: {cfg}")
    
    if not config_files:
        print("No valid configuration files specified")
        return
    
    # Compare mode
    if args.compare and len(config_files) > 1:
        compare_configs(config_files)
        return
    
    # Single config view
    config_file = config_files[0]
    config = load_config(config_file)
    
    # Print summary
    print_config_summary(config, config_file)
    
    # Show visualization
    img = visualize_desired_features(config)
    
    window_name = "IBVS Configuration Viewer"
    cv2.imshow(window_name, img)
    
    print("Visualization window open. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Export code if requested
    if args.export_code:
        generate_code_for_config(config, args.export_code)
    else:
        print("\nGenerated Python code:")
        print("-" * 70)
        print(config['code_snippets']['python_function'])
        print("-" * 70)
        print("\nTo export code: python ibvs_config_viewer.py <config> --export-code output.py")

if __name__ == "__main__":
    main()
