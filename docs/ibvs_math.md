# IBVS Mathematical Formulation

## Image-Based Visual Servoing for Eye-in-Hand Configuration

This document describes the mathematical formulation used in the IBVS controller for the SO101 robot arm with an eye-in-hand camera setup.

---

## 1. Feature Representation

We use the four corner points of an ArUco marker as visual features:

$$\mathbf{s} = [u_1, v_1, u_2, v_2, u_3, v_3, u_4, v_4]^T \in \mathbb{R}^8$$

where $(u_i, v_i)$ are pixel coordinates of corner $i$.

---

## 2. Image Jacobian (Interaction Matrix)

For a single point feature at pixel coordinates $(u, v)$ with depth $Z$, the interaction matrix relates feature velocity to camera velocity:

$$\dot{\mathbf{s}} = \mathbf{L} \cdot \mathbf{v}_c$$

### 2.1 Normalized Image Coordinates

First, convert from pixel to normalized coordinates:

$$x = \frac{u - c_x}{f_x}, \quad y = \frac{v - c_y}{f_y}$$

where:
- $(c_x, c_y)$ = principal point (image center)
- $(f_x, f_y)$ = focal lengths in pixels

### 2.2 Interaction Matrix for One Point

The 2×6 interaction matrix for one point is:

$$\mathbf{L}_p = \begin{bmatrix}
-\frac{f_x}{Z} & 0 & \frac{f_x \cdot x}{Z} & f_x \cdot xy & -f_x(1+x^2) & f_x \cdot y \\
0 & -\frac{f_y}{Z} & \frac{f_y \cdot y}{Z} & f_y(1+y^2) & -f_y \cdot xy & -f_y \cdot x
\end{bmatrix}$$

### 2.3 Full Interaction Matrix

For 4 corner points, stack the individual matrices:

$$\mathbf{L} = \begin{bmatrix} \mathbf{L}_{p1} \\ \mathbf{L}_{p2} \\ \mathbf{L}_{p3} \\ \mathbf{L}_{p4} \end{bmatrix} \in \mathbb{R}^{8 \times 6}$$

---

## 3. IBVS Control Law

The classic IBVS control law computes camera velocity to minimize feature error:

$$\mathbf{v}_c = -\lambda \cdot \mathbf{L}^+ \cdot \mathbf{e}$$

where:
- $\lambda = 0.05$ (proportional gain)
- $\mathbf{e} = \mathbf{s} - \mathbf{s}^* \in \mathbb{R}^8$ (feature error in pixels)
- $\mathbf{L}^+$ = pseudo-inverse of interaction matrix

### 3.1 Damped Pseudo-Inverse

For numerical stability, we use the damped least-squares pseudo-inverse:

$$\mathbf{L}^+ = \mathbf{L}^T (\mathbf{L}\mathbf{L}^T + \mu^2 \mathbf{I})^{-1}$$

where $\mu = 0.01$ is the damping factor.

---

## 4. Frame Transformations

The camera velocity must be transformed through the kinematic chain to compute joint velocities.

### 4.1 Camera to End-Effector Frame

$$\mathbf{v}_{ee} = \mathbf{R}_{c \to ee} \cdot \mathbf{v}_c^{lin} + \mathbf{p}_c \times \boldsymbol{\omega}_c$$
$$\boldsymbol{\omega}_{ee} = \mathbf{R}_{c \to ee} \cdot \boldsymbol{\omega}_c$$

where:
- $\mathbf{R}_{c \to ee}$ = rotation matrix from camera to end-effector frame
- $\mathbf{p}_c$ = position of camera in end-effector frame
- $\mathbf{p}_c \times \boldsymbol{\omega}_c$ = lever arm compensation

### 4.2 End-Effector to World Frame

$$\mathbf{v}_{world} = \mathbf{R}_{ee \to w} \cdot \mathbf{v}_{ee}$$
$$\boldsymbol{\omega}_{world} = \mathbf{R}_{ee \to w} \cdot \boldsymbol{\omega}_{ee}$$

### 4.3 Joint Velocities via Robot Jacobian

$$\dot{\mathbf{q}} = \mathbf{J}^+ \cdot \begin{bmatrix} \mathbf{v}_{world} \\ \boldsymbol{\omega}_{world} \end{bmatrix}$$

where $\mathbf{J}^+$ is the damped pseudo-inverse of the robot geometric Jacobian:

$$\mathbf{J}^+ = \mathbf{J}^T (\mathbf{J}\mathbf{J}^T + \mu_r^2 \mathbf{I})^{-1}$$

with $\mu_r = 0.02$.

---

## 5. Depth Estimation from Marker Size

Instead of using a depth sensor, we estimate depth from the known physical marker size using the pinhole camera model:

$$Z = \frac{S_{physical} \cdot f_x}{S_{pixels}}$$

where:
- $S_{physical} = 0.036$ m (known marker size: 36mm)
- $S_{pixels}$ = average side length of marker in pixels
- $f_x$ = focal length in pixels

This approach is robust because it only depends on the marker's apparent size, not on depth buffer accuracy.

---

## 6. MuJoCo Camera Convention Correction

MuJoCo uses the OpenGL camera convention where the camera looks along the **negative Z-axis**. Standard IBVS formulations assume the camera looks along **positive Z**.

**Correction applied:**

$$v_c^z \leftarrow -v_c^z$$

This is applied after computing the camera velocity and before the frame transformations.

---

## 7. Iterative IBVS (Novel Contribution)

When the feature error starts increasing (indicating divergence or local minimum), we implement an iterative restart strategy:

1. **Detect divergence:** $\|\mathbf{e}_t\| > \|\mathbf{e}_{t-1}\| + \epsilon$
2. **Capture current features:** $\mathbf{s}^* \leftarrow \mathbf{s}_{current}$
3. **Reset error tracking**
4. **Continue IBVS from new baseline**

This allows incremental progress toward the target, even when a single IBVS trajectory would fail.

---

## 8. System Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| IBVS gain | $\lambda$ | 0.05 | Proportional control gain |
| Image Jacobian damping | $\mu$ | 0.01 | Pseudo-inverse regularization |
| Robot Jacobian damping | $\mu_r$ | 0.02 | Joint velocity regularization |
| Marker size | $S_{physical}$ | 0.036 m | ArUco marker dimension |
| Focal length | $f_x, f_y$ | 342.8 px | Camera intrinsic parameter |
| Principal point | $(c_x, c_y)$ | (320, 240) | Image center |
| Convergence threshold | - | 8.0 px | Feature error for convergence |

---

## References

1. F. Chaumette and S. Hutchinson, "Visual servo control. I. Basic approaches," IEEE Robotics & Automation Magazine, vol. 13, no. 4, pp. 82-90, 2006.

2. F. Chaumette and S. Hutchinson, "Visual servo control. II. Advanced approaches," IEEE Robotics & Automation Magazine, vol. 14, no. 1, pp. 109-118, 2007.
