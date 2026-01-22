# IBVS Teaching Tools for SO-101

## Quick Start

```bash
# 1. Position robot and save desired features
python ibvs_teaching_tool.py

# 2. Review saved configurations
python ibvs_config_viewer.py --list

# 3. Compare multiple poses
python ibvs_config_viewer.py --compare config1.json config2.json

# 4. Export code for your IBVS controller
python ibvs_config_viewer.py config.json --export-code output.py
```

## Files Included

| File | Purpose |
|------|---------|
| `ibvs_teaching_tool.py` | Interactive robot positioning tool with sliders and live camera |
| `ibvs_config_viewer.py` | Review, compare, and export saved configurations |
| `IBVS_TEACHING_TOOLS_GUIDE.md` | Complete usage guide with examples |

## Why Use These Tools?

**Before:** Trial and error with arbitrary pixel values
```python
DESIRED_OFFSET_Y = 80  # Guess - might crash robot! ❌
```

**After:** Measured values from actual safe poses
```python
DESIRED_OFFSET_Y = 15.3  # From teaching - verified safe ✅
```

## The Problem You Had

Your current code crashes because:

1. ❌ `DESIRED_OFFSET_Y = 80` means marker should appear 80px BELOW center
2. ❌ With eye-in-hand camera, this requires gripper to MOVE DOWN
3. ❌ No Z-axis constraint prevents downward motion
4. ❌ Result: gripper crashes into ground

## How Teaching Tools Fix This

1. ✅ Position robot manually to a SAFE pose
2. ✅ See exactly what marker looks like at that pose
3. ✅ Save those pixel coordinates as desired features
4. ✅ IBVS will converge to a pose you KNOW is safe

## Example Session Output

```
=============================================================
CONFIGURATION SAVED: ibvs_configs/desired_features_20260121.json
=============================================================
Marker ID: 0
Center: (340.2, 235.8) px
Offset from image center: (+20.2, -4.2) px  ← Safe: slightly UP!
Marker size: 72.5 px
Gripper height: 0.152 m  ← Safe height!
=============================================================
```

Now use these values instead of guesses!

## Next Steps

1. **Run the teaching tool** - Position robot safely
2. **Save multiple poses** - Try different heights/angles
3. **Compare configurations** - Choose the best one
4. **Update your IBVS code** - Use measured values
5. **Add Z-constraint** - Prevent downward motion
6. **Test safely** - Start with loose thresholds

See `IBVS_TEACHING_TOOLS_GUIDE.md` for detailed instructions.

## Key Insight

**Your desired features define the TARGET POSE.**

If you set features that require the gripper to be at ground level, IBVS will dutifully try to reach that pose. The teaching tool ensures you only set features corresponding to REACHABLE, SAFE poses.

---

**Bottom line:** Stop guessing pixel coordinates. Teach the robot where you want it to be. 🎯
