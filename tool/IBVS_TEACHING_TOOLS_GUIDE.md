# IBVS Teaching Tools - Usage Guide

## Overview

These tools help you systematically determine the optimal desired features for your IBVS controller by **teaching through demonstration** rather than guessing pixel coordinates.

## Tools Provided

1. **`ibvs_teaching_tool.py`** - Interactive robot positioning with live camera feedback
2. **`ibvs_config_viewer.py`** - Review, compare, and export saved configurations
3. This guide

## Workflow

### Step 1: Run the Teaching Tool

```bash
python ibvs_teaching_tool.py
```

**What happens:**
- Opens two windows:
  - **"Eye-in-Hand Camera"** - Shows what the gripper camera sees
  - **"Joint Control Panel"** - Sliders to control each joint

### Step 2: Position the Robot

Use the sliders to manually position the robot arm:

1. **Start with the base joints** (Shoulder Pan, Shoulder Lift)
   - Get the gripper roughly above the marker
   
2. **Adjust Elbow** to set the approach distance
   - Closer = larger marker in image
   - Further = smaller marker in image
   
3. **Adjust Wrist joints** for fine positioning
   - Wrist Flex: pitch angle
   - Wrist Roll: rotation

4. **Watch the camera window** - you'll see:
   - Green detection when marker is found
   - Yellow lines showing offset from image center
   - Real-time measurements

### Step 3: Find Good Poses

Look for poses where:

✅ **Marker is clearly visible** (60-100 pixels size is good)
✅ **Gripper height is safe** (> 0.12m recommended)
✅ **Marker is centered or slightly offset** (not at image edge)
✅ **Approach angle allows gripper jaws to fit**

**Teaching Strategy:**
Try multiple poses with different characteristics:

- **Pose A: High and centered** (safest)
  - Gripper: 0.15-0.20m height
  - Marker: centered in image
  - Good for: Conservative approach

- **Pose B: Lower with offset** (more aggressive)
  - Gripper: 0.12-0.15m height  
  - Marker: offset to side
  - Good for: Angled approach to avoid jaw collision

- **Pose C: Directly above** (top-down)
  - Gripper: 0.15m+ height
  - Marker: dead center, larger size
  - Good for: Pick-and-place tasks

### Step 4: Save Configurations

When you find a good pose:

1. Press **'S'** key in the camera window
2. Configuration saved to `ibvs_configs/` folder
3. Creates two files:
   - `.json` - Full configuration data
   - `.py` - Python code snippet

**Console output shows:**
```
=============================================================
CONFIGURATION SAVED: ibvs_configs/desired_features_20260121_143022.json
=============================================================
Marker ID: 0
Center: (380.5, 320.2) px
Offset from image center: (+60.5, +80.2) px
Marker size: 75.3 px
Gripper height: 0.145 m
Joint positions: [0.234, -0.678, 0.445, 1.123, 1.570]
=============================================================
```

### Step 5: Review and Compare

List all saved configurations:
```bash
python ibvs_config_viewer.py --list
```

View a specific configuration:
```bash
python ibvs_config_viewer.py ibvs_configs/desired_features_20260121_143022.json
```

Compare multiple configurations:
```bash
python ibvs_config_viewer.py --compare config1.json config2.json config3.json
```

**Comparison shows:**
- Side-by-side metrics table
- Visual representations of each pose
- Safety indicators (height warnings)

### Step 6: Choose Best Configuration

Select based on:
1. **Safety** - Higher gripper height is safer
2. **Marker visibility** - Size should be 60-100px
3. **Approach angle** - Should allow gripper to fit around object
4. **Convergence basin** - Centered poses have larger convergence zones

### Step 7: Integrate Into Your Code

**Option A: Copy Python code directly**

The viewer exports ready-to-use code:
```bash
python ibvs_config_viewer.py config.json --export-code my_desired_features.py
```

Then in your IBVS code:
```python
from my_desired_features import compute_desired_features

# In your control loop:
desired_features = compute_desired_features()
```

**Option B: Use JSON configuration**

```python
import json

# Load configuration
with open('ibvs_configs/desired_features_20260121_143022.json', 'r') as f:
    config = json.load(f)

# Extract desired features
desired_corners = np.array(config['desired_features']['corners'])
marker_size = config['desired_features']['marker_size_px']
offset_x = config['desired_features']['offset_x']
offset_y = config['desired_features']['offset_y']

# Or use the compute function from generated code
DESIRED_MARKER_SIZE = marker_size
DESIRED_OFFSET_X = offset_x
DESIRED_OFFSET_Y = offset_y
```

**Option C: Update existing code constants**

Replace these lines in `eye_in_hand_ibvs_v2.py`:

```python
# OLD (guessed values)
DESIRED_MARKER_SIZE = 80
DESIRED_OFFSET_Y = 80
DESIRED_OFFSET_X = 60

# NEW (from teaching tool)
DESIRED_MARKER_SIZE = 75.3  # From saved config
DESIRED_OFFSET_Y = 80.2     # From saved config
DESIRED_OFFSET_X = 60.5     # From saved config
```

## Pro Tips

### 1. Create Multiple Safe Poses

Save 3-5 different configurations:
- Different heights
- Different approach angles
- Different offsets

This gives you options if one doesn't converge well.

### 2. Test Convergence Basin

For each saved pose, note how much deviation is acceptable:
- Try starting IBVS from slightly different robot positions
- Poses with marker centered have larger convergence basins
- Offset poses may converge faster but from smaller initial region

### 3. Height Safety

**Always verify gripper height in saved configs:**
- < 0.10m: ⚠️ DANGEROUS - Do not use
- 0.10-0.12m: ⚠️ Risky - Use with caution
- 0.12-0.15m: ✓ Safe for most cases
- > 0.15m: ✓✓ Very safe, may be too high for grasping

### 4. Marker Size Guidelines

- **50-70px**: Far approach, good for large convergence basin
- **70-90px**: Balanced, recommended for most tasks
- **90-120px**: Close approach, may risk collision
- **> 120px**: Too close, avoid

### 5. Keyboard Shortcuts

While using the teaching tool:

| Key | Action |
|-----|--------|
| S | Save current configuration |
| R | Reset to default pose |
| C | Capture screenshot |
| G | Toggle gripper open/close |
| V | Toggle measurement overlay |
| ESC | Exit tool |

## Troubleshooting

### "No marker detected"
- Adjust joint angles to point camera at marker
- Check lighting conditions
- Ensure marker is within camera FOV
- Try increasing marker size or getting closer

### "Sliders not responding"
- Click on "Joint Control Panel" window first
- Some systems require the window to have focus

### "Robot crashes in IBVS"
This means your desired features require downward motion:
- Check `offset_y` value - positive means marker appears LOWER
- With eye-in-hand, lower in image = gripper moves DOWN
- Solution: Use pose with negative or small positive offset_y

### "Saved config not loading"
- Verify JSON file is valid (not corrupted)
- Check file permissions
- Ensure `ibvs_configs/` directory exists

## Example Workflow Session

```bash
# 1. Start teaching tool
python ibvs_teaching_tool.py

# [User positions robot, finds good pose, presses 'S']
# [User tries different angles, saves 3 more poses]

# 2. List saved configurations
python ibvs_config_viewer.py --list

Output:
=======================================================================
AVAILABLE CONFIGURATIONS
=======================================================================

1. desired_features_20260121_143022.json
   Created: 2026-01-21_14-30-22
   Marker: 75px @ offset(+60, +80)px
   Height: 0.145m

2. desired_features_20260121_143156.json
   Created: 2026-01-21_14-31-56
   Marker: 82px @ offset(+45, +65)px
   Height: 0.132m

3. desired_features_20260121_143301.json
   Created: 2026-01-21_14-33-01
   Marker: 68px @ offset(+20, -10)px
   Height: 0.168m
=======================================================================

# 3. Compare configurations
python ibvs_config_viewer.py --compare \
    ibvs_configs/desired_features_20260121_143022.json \
    ibvs_configs/desired_features_20260121_143301.json

# 4. Choose config #3 (safest height, good visibility)
# Export code
python ibvs_config_viewer.py \
    ibvs_configs/desired_features_20260121_143301.json \
    --export-code my_ibvs_config.py

# 5. Update your IBVS controller
# Copy values from my_ibvs_config.py into eye_in_hand_ibvs_v2.py

# 6. Test IBVS with new configuration
python eye_in_hand_ibvs_v2.py
```

## Understanding the Output

### Configuration JSON Structure

```json
{
  "timestamp": "2026-01-21_14-30-22",
  "marker_id": 0,
  
  "camera": {
    "width": 640,
    "height": 480,
    "fx": 338.5,
    "fy": 338.5,
    "cx": 320.0,
    "cy": 240.0,
    "fovy": 70
  },
  
  "desired_features": {
    "corners": [[...], [...], [...], [...]],  // 4 corner positions
    "center": [380.5, 320.2],
    "offset_x": 60.5,   // Positive = marker right of center
    "offset_y": 80.2,   // Positive = marker below center
    "marker_size_px": 75.3
  },
  
  "robot_state": {
    "joint_positions": [0.234, -0.678, 0.445, 1.123, 1.570],
    "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    "gripper_position": [0.123, -0.045, 0.145],
    "gripper_height": 0.145
  },
  
  "code_snippets": {
    "python_constants": "...",
    "python_function": "..."
  }
}
```

### Key Metrics Explained

**offset_x, offset_y:**
- How far marker center is from image center (pixels)
- Positive X = marker appears to the RIGHT
- Positive Y = marker appears BELOW center (DOWN)
- For eye-in-hand: Positive Y requires gripper to move DOWN (⚠️)

**marker_size_px:**
- Average length of marker edges in pixels
- Larger = closer to marker
- Use 60-90px for safe, visible approach

**gripper_height:**
- Absolute Z-height of gripper in world frame (meters)
- CRITICAL for collision avoidance
- Always check this value before using a configuration

## Next Steps

After finding good desired features:

1. **Fix the Z-constraint issue** in your IBVS code:
   ```python
   # In ibvs_control() function, after computing v_camera:
   v_camera[2] = 0.0  # Prevent downward motion
   ```

2. **Use staged control** (your Stage 1 is good, expand it):
   - Stage 1: Align X (pan) only
   - Stage 2: Align distance (elbow) with fixed height
   - Stage 3: Fine alignment with wrist only

3. **Add height maintenance**:
   ```python
   # In control loop:
   current_height = data.body("gripper").xpos[2]
   if current_height < SAFE_HEIGHT:
       # Override: force shoulder_lift upward
       q_dot[1] = max(q_dot[1], 0.1)
   ```

4. **Test incrementally**:
   - Start with large error threshold (50px)
   - Verify it approaches without crashing
   - Gradually tighten threshold (20px, then 10px, then 5px)

## Summary

The teaching tool eliminates guesswork by:
✅ Showing you exactly what the camera sees
✅ Letting you position the robot manually
✅ Saving the exact pixel coordinates of good poses
✅ Generating ready-to-use code

**Key advantage:** You're setting desired features based on **actual achievable poses**, not arbitrary pixel coordinates that might require impossible robot motions.

Good luck with your IBVS system! 🤖
