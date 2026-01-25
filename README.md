# Eye-in-Hand IBVS for SO-101 Robot Arm (MuJoCo Simulation)

**Layered Image-Based Visual Servoing (IBVS) system** using an eye-in-hand camera to detect and servo toward an ArUco marker cube in a MuJoCo simulation of the SO-101 open-source robot arm.

This project demonstrates a complete pipeline for autonomous visual servoing:
- ArUco marker detection
- Bird's-eye-view radial search strategy
- Classic IBVS control law (4-point corner features)
- Depth estimation from marker size (robust alternative to depth buffer)
- Safety checks and divergence recovery

### Key Features

- **Robot & Scene**: SO-101 5-DoF arm (open-source design) + eye-in-hand camera mounted on gripper + ArUco cube target
- **Camera**: Simulated 640×480 RGB-D camera with realistic intrinsics (fovy=70°)
- **Detection**: OpenCV ArUco (DICT_4X4_50) with consistent corner ordering
- **Search Strategy**: Multi-depth panoramic scan (shoulder lift + base pan + elbow variation)
- **IBVS Control**: 
  - Pixel-based interaction matrix (Chaumette formulation with focal length scaling)
  - Damped pseudo-inverse for numerical stability
  - Lever-arm compensation for camera offset
  - MuJoCo camera convention handling (-Z forward)
  - Automatic restart on detected divergence (error increasing)
- **Depth Handling**: Marker-size-based depth estimation (more robust than raw depth buffer for planar target)
- **Safety**: Minimum gripper height limit, joint velocity clamping, joint limit clipping
- **Visualization**: Real-time eye-in-hand camera feed with overlay (current vs desired corners, state, error, depth)

### Current Status

The system successfully:
- Locates the marker via search pattern
- Approaches without crashing into the ground/table
- Converges to ~30–40 px total error after 1–2 divergence resets

Future work includes:
- Full grasp execution layer
- Tighter convergence (<10 px) via integral action & better tuning
- Hybrid PBVS/IBVS fallback
- Real robot porting

### Demo & Usage

1. Install dependencies: `pip install mujoco opencv-python numpy`
2. Place robot/scene XML files in `./model/`
3. Add desired feature JSON in `./ibvs_configs/` using `tool/ibvs_teaching_tool.py`
4. Run: `python eye_in_hand_ibvs_v2.py`
5. In viewer: Press **O** to start search + servoing

Keys (focus OpenCV window):
- **O** → Start automatic pipeline
- **R** → Reset to manual mode without changing pose
- **ESC** → Quit

### Why this project?

Built as an educational/experimental platform to explore classical visual servoing challenges in simulation before hardware deployment.  
Handles many real-world IBVS pitfalls: sign conventions, depth noise, divergence, lever-arm effects, and MuJoCo-specific camera behavior.
