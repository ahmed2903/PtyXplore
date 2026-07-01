# Projection-Based $\mathbf{k}_{in}$ Refinement

## Overview

This document describes the projection algorithm implemented in `ptyxplore/kvectors.py`
(`projection_kin_refinement`) and the associated ROI-level wrapper
(`compute_kins_projection` in `roi_ops.py`).

The goal is to recover **incident wave-vectors** $\mathbf{k}_{in}$ from a set of measured
diffraction streaks.  Each streak corresponds to a single scattering vector $\mathbf{G}_s$;
different streaks need not share the same $\mathbf{G}$.  The algorithm enforces the elastic
scattering condition $|\mathbf{k}_{in}| = k$ exactly by projecting onto the **Ewald circle**
for each streak.

---

## Optimisation Problem

Given $S$ measured diffraction streaks, each consisting of $N_s$ detector pixels with known
scattered wave-vectors $\{\mathbf{k}_{out,j}\}_{j=1}^{N_s}$, find for each streak $s$:

- a scattering vector $\mathbf{G}_s \in \mathbb{R}^3$, and
- incident wave-vectors $\{\mathbf{k}_{in,j}\}_{j=1}^{N_s}$,

such that the following constraints are satisfied simultaneously:

$$\underbrace{\mathbf{k}_{in,j} = \mathbf{k}_{out,j} - \mathbf{G}_s}_{\text{Bragg (C1)}} \qquad
\underbrace{|\mathbf{k}_{in,j}| = k}_{\text{elastic (C2)}} \qquad
\underbrace{\mathbf{k}_{in,j} \in \mathcal{P}}_{\text{pupil support (C6)}}$$

where $k = 2\pi/\lambda$ and $\mathcal{P}$ is the rectangular pupil aperture.

Because C1 fixes $\mathbf{k}_{in,j} = \mathbf{k}_{out,j} - \mathbf{G}_s$, the problem reduces to
finding one $\mathbf{G}_s$ per streak.  In general, no $\mathbf{G}_s$ satisfies C1 and C2
exactly for every pixel simultaneously (the measurements are noisy and the detector is
discrete), so the problem is cast as a feasibility problem in the sense of alternating
projections: find $\mathbf{G}_s$ such that the implied $\mathbf{k}_{in,j} = \mathbf{k}_{out,j} - \mathbf{G}_s$
lie as close as possible to the **Ewald circle** — the intersection of the elastic sphere
$|\mathbf{k}_{in}| = k$ and the Bragg plane $\mathbf{G}_s \cdot \mathbf{k}_{in} = -|\mathbf{G}_s|^2/2$.

Formally, for each streak $s$ the algorithm solves:

$$\min_{\mathbf{G}_s}\; \frac{1}{N_s}\sum_{j=1}^{N_s} d\!\left(\mathbf{k}_{out,j} - \mathbf{G}_s,\; \mathcal{E}_s\right)^2$$

where $d(\mathbf{x}, \mathcal{E}_s)$ is the distance from $\mathbf{x}$ to the Ewald circle

$$\mathcal{E}_s = \left\{\mathbf{k}_{in} : |\mathbf{k}_{in}| = k \;\text{ and }\; \mathbf{G}_s\cdot\mathbf{k}_{in} = -\frac{|\mathbf{G}_s|^2}{2}\right\},$$

and $\mathbf{G}_s$ is updated after each projection step as the mean residual
$\mathbf{G}_s \leftarrow \tfrac{1}{N_s}\sum_j (\mathbf{k}_{out,j} - \mathbf{k}_{in,j}^{\mathrm{proj}})$.

---

## Physical Setup

In convergent-beam diffraction (CBD), the sample is illuminated by a convergent beam whose
angular spread is defined by the objective-lens pupil aperture.  Each incident direction
$\hat{\mathbf{k}}_{in}$ that satisfies the Bragg condition for a reciprocal-lattice vector
$\mathbf{G}$ produces a scattered beam

$$\mathbf{k}_{out} = \mathbf{k}_{in} + \mathbf{G}.$$

Because the pupil spans a range of incident directions, the corresponding scattered beams
trace out a **streak** on the detector — a rigid translation of the pupil aperture by
$\mathbf{G}$ in reciprocal space.

### Variables

| Symbol | Meaning |
|--------|---------|
| $\lambda$ | X-ray wavelength |
| $k = 2\pi/\lambda$ | magnitude of all wave-vectors (elastic scattering) |
| $\mathbf{k}_{out,j}$ | scattered wave-vector for detector pixel $j$ |
| $\mathbf{k}_{in,j}$ | incident wave-vector producing pixel $j$ (to be recovered) |
| $\mathbf{G}_s$ | scattering vector for streak $s$ |
| $\mathcal{P}$ | pupil aperture — a rectangular box in $(k_x, k_y)$ defined by $(k_x^{\min}, k_x^{\max}, k_y^{\min}, k_y^{\max})$ |

---

## Constraint Sets

For each streak $s$ with pixel set $\{j\}$, the following constraints must hold simultaneously.

### C1 — Bragg condition (definition of $\mathbf{G}_s$)

$$\mathbf{G}_s = \mathbf{k}_{out,j} - \mathbf{k}_{in,j} \quad \forall\, j \in s.$$

All pixels in the streak share the **same** $\mathbf{G}_s$.

### C2 — Elastic scattering (Ewald sphere)

$$|\mathbf{k}_{in,j}| = |\mathbf{k}_{out,j}| = k = \frac{2\pi}{\lambda}.$$

### C3 — 2theta constraint

The angle between $\mathbf{k}_{in}$ and $\mathbf{k}_{out}$ equals the scattering angle
$2\theta_s$:

$$\hat{\mathbf{k}}_{in,j} \cdot \hat{\mathbf{k}}_{out,j} = \cos(2\theta_s).$$

Using the law of cosines, this is equivalent to a constraint on $|\mathbf{G}_s|$:

$$|\mathbf{G}_s| = 2k\sin\!\left(\frac{\theta_s}{2}\right).$$

### C4 — Bragg plane (C1 + C2 combined)

From C1, $\mathbf{k}_{in,j} = \mathbf{k}_{out,j} - \mathbf{G}_s$.  Substituting into C2, $|\mathbf{k}_{in,j}| = k$:

$$|\mathbf{k}_{out,j} - \mathbf{G}_s|^2 = k^2.$$

Expanding and using $|\mathbf{k}_{out,j}| = k$:

$$k^2 - 2\,\mathbf{G}_s\cdot\mathbf{k}_{out,j} + |\mathbf{G}_s|^2 = k^2
\implies \mathbf{G}_s\cdot\mathbf{k}_{out,j} = \frac{|\mathbf{G}_s|^2}{2}.$$

Applying C1 once more to convert to a constraint on $\mathbf{k}_{in,j}$:

$$\mathbf{G}_s\cdot\mathbf{k}_{in,j}
= \mathbf{G}_s\cdot(\mathbf{k}_{out,j} - \mathbf{G}_s)
= \frac{|\mathbf{G}_s|^2}{2} - |\mathbf{G}_s|^2
= -\frac{|\mathbf{G}_s|^2}{2}.$$

Every $\mathbf{k}_{in,j}$ satisfying both C1 and C2 must lie in the **Bragg plane**:

$$\mathcal{H}_s = \left\{\mathbf{k}_{in} : \mathbf{G}_s \cdot \mathbf{k}_{in} = -\frac{|\mathbf{G}_s|^2}{2}\right\}.$$

### C5 — Ewald circle

The intersection of the Bragg plane $\mathcal{H}_s$ with the elastic sphere $|\mathbf{k}_{in}| = k$
is a **circle** — the Ewald circle for streak $s$:

$$\text{centre:}\quad \mathbf{C}_s = -\frac{\mathbf{G}_s}{2}, \qquad \text{radius:}\quad r_s = \sqrt{k^2 - \tfrac{|\mathbf{G}_s|^2}{4}}.$$

Every physically valid $\mathbf{k}_{in,j}$ must lie on this circle.

### C6 — Pupil support

Every implied $\mathbf{k}_{in,j}$ must lie inside the rectangular pupil aperture:

$$k_x^{\min} \le k_{in,j}^x \le k_x^{\max}, \qquad k_y^{\min} \le k_{in,j}^y \le k_y^{\max}.$$

In G-space this translates to a feasible interval for $G_s^x$ (and identically for $y$):

$$\max_j k_{out,j}^x - k_x^{\max} \;\le\; G_s^x \;\le\; \min_j k_{out,j}^x - k_x^{\min}.$$

### C7 — Coplanarity (multi-streak coupling)

All scattering vectors $\{\mathbf{G}_s\}$ from the same crystal arise from the same set of
lattice planes and therefore lie in a common **scattering plane** through the origin.
Denoting the unit normal to this plane as $\hat{\mathbf{n}}$:

$$\hat{\mathbf{n}} \cdot \mathbf{G}_s = 0 \quad \forall\, s.$$

This is a **coupling constraint** — it links all streaks simultaneously and cannot be
enforced per-streak. It requires at least 3 streaks to be non-trivial (any 2 vectors
trivially lie in a common plane through the origin).

---

## Ewald Circle Projection (`_ewald_circle_projection`)

Given a current estimate $\mathbf{G}_s$ and the raw difference $\mathbf{k}_{in,j} = \mathbf{k}_{out,j} - \mathbf{G}_s$,
the projection onto the Ewald circle is a two-step procedure applied **per pixel**:

**Step 1 — Project onto the Bragg plane (enforce C4).**

The excess of $\mathbf{k}_{in,j}$ above the Bragg plane along the $\mathbf{G}_s$ direction:

$$\epsilon_j = \frac{\mathbf{G}_s \cdot \mathbf{k}_{in,j} + |\mathbf{G}_s|^2/2}{|\mathbf{G}_s|^2}.$$

The projection onto $\mathcal{H}_s$:

$$\tilde{\mathbf{k}}_{in,j} = \mathbf{k}_{in,j} - \epsilon_j\,\mathbf{G}_s.$$

This removes the component of $\mathbf{k}_{in,j}$ that violates the Bragg plane, leaving the
transverse (in-plane) part unchanged.

**Step 2 — Scale to the elastic sphere (enforce C2).**

$$\mathbf{k}_{in,j}^{\mathrm{proj}} = k\;\frac{\tilde{\mathbf{k}}_{in,j}}{|\tilde{\mathbf{k}}_{in,j}|}.$$

**Guarantees after projection:**

$$\left|\mathbf{k}_{in,j}^{\mathrm{proj}}\right| = k \quad\checkmark \qquad
\mathbf{G}_s \cdot \mathbf{k}_{in,j}^{\mathrm{proj}} = -\frac{|\mathbf{G}_s|^2}{2} \quad\checkmark$$

The projection is degenerate when $|\mathbf{G}_s| \approx 0$; in that case
$\mathbf{k}_{in,j}^{\mathrm{proj}} = \mathbf{k}_{out,j} - \mathbf{G}_s$ is returned directly.

---

## Algorithm

### Initialisation

A good starting $\mathbf{G}_s$ is essential because the Ewald circle projection is defined
relative to $\mathbf{G}_s$.  The recommended workflow is to run `compute_kins` or
`compute_kins_w_geometry` first (which stores `roi.g_init` / `roi.g_list`), then call
`compute_kins_projection`.  This starting G is automatically forwarded.

If no prior G is available (first call without prior optimisation), $G_z$ is bootstrapped
from the elastic-z estimate with $G_{xy} = 0$:

$$G_z^{(0)} = \operatorname{median}_j\!\left(k_{out,j}^z - \sqrt{k^2 - |k_{out,j}^{xy}|^2}\right).$$

### Inner loop — alternating projections per streak (`_project_single_streak`)

Repeat `n_inner` times:

```
Step 1  (Ewald circle projection → kins)
    for each pixel j:
        kin_j = kout_j - G
        ε_j   = (G · kin_j + |G|²/2) / |G|²
        kin_j_plane = kin_j - ε_j * G          # project onto Bragg plane
        kin_j_proj  = k * kin_j_plane / |kin_j_plane|   # scale to sphere

Step 2  (G update from mean residual)
    G ← mean_j( kout_j − kin_j_proj )

Step 3  (Pupil-support clamp → refine G_xy, optional)
    G_x ← clip( G_x,  max_j(kout_x) − kx_max,  min_j(kout_x) − kx_min )
    G_y ← clip( G_y,  max_j(kout_y) − ky_max,  min_j(kout_y) − ky_min )
```

After the loop, a **final Ewald projection** is applied unconditionally so that the returned
$\mathbf{k}_{in}$ vectors satisfy $|\mathbf{k}_{in}| = k$ exactly on exit.  $\mathbf{G}_s$ is
then recomputed as the mean residual of the final projected kins.

### Outer loop — convergence tracking (`projection_kin_refinement`)

The outer loop re-runs the inner loop for all streaks.  Each outer iteration starts from the
G refined by the previous one (G is carried forward, not reset):

$$\delta G = \max_s \|\mathbf{G}_s^{(t)} - \mathbf{G}_s^{(t-1)}\|.$$

Iteration stops early if $\delta G < 10^{-12}$.

### Full algorithm schematic

```
Input:  kouts_list       — list of (N_s, 3) arrays, one per streak
        pupil_kbounds    — (kx_min, kx_max, ky_min, ky_max)
        wavelength
        G_init_list      — (optional) list of (3,) initial G vectors from prior optimisation
        enforce_coplanar — (optional) enforce C7 across all streaks

Initialise G_list from G_init_list (or elastic-z bootstrap if absent)

for iteration = 1 … n_iterations:

    for each streak s:                                 ← per-streak, independent
        G ← G_list[s]                                 ← carry forward
        for inner = 1 … n_inner:
            kins ← ewald_circle_projection(kouts_s, G, k)   # |kin| = k  (C2 + C4)
            G    ← mean(kouts_s − kins)                     # G update
            [G_xy ← pupil_clamp(G_xy)]                      # C6, optional
        kins ← ewald_circle_projection(kouts_s, G, k)       # final projection
        G    ← mean(kouts_s − kins)                         # consistent G

    if enforce_coplanar and n_streaks ≥ 3:             ← cross-streak coupling
        n_hat ← smallest right singular vector of [G_1 | … | G_S]ᵀ
        for each streak s:
            G_s ← G_s − (n_hat · G_s) n_hat           # project onto scattering plane (C7)

    compute δG = max_s ‖G_s(t) − G_s(t-1)‖
    if δG < 1e-12: break

if enforce_coplanar:                                   ← final Ewald sync
    for each streak s:
        kins_s ← ewald_circle_projection(kouts_s, G_s, k)
        G_s    ← mean(kouts_s − kins_s)

Output: G_list, kins_list, width_ratios, history  (history includes plane_normal per iteration)
```

---

## Properties of the Ewald Circle Update

### Why |kin| = k is exact on exit

The final Ewald projection scales each $\tilde{\mathbf{k}}_{in,j}$ to the sphere radius $k$,
so $|\mathbf{k}_{in,j}| = k$ holds exactly for every pixel on every exit from
`_project_single_streak`.

### Why the update G = mean(kouts − kins_proj) is natural

After projection, each pixel $j$ implies a slightly different residual
$\mathbf{G}_j = \mathbf{k}_{out,j} - \mathbf{k}_{in,j}^{\mathrm{proj}}$ because
the projection is per-pixel.  Taking the mean is the least-squares estimate of a single
$\mathbf{G}$ consistent with all projected kins:

$$\mathbf{G}^{\text{new}} = \operatorname{argmin}_\mathbf{G} \sum_j \|\mathbf{k}_{out,j} - \mathbf{k}_{in,j}^{\mathrm{proj}} - \mathbf{G}\|^2.$$

### Convergence behaviour

The Ewald projection is a contraction near the Ewald circle.  The outer loop converges
rapidly (typically $\delta G \ll 10^{-12}$ within 2–5 iterations for well-conditioned data)
because each inner loop is already nearly closed-form.  The outer loop's main purpose is
to expose per-iteration diagnostics and to allow re-entry after an external geometry update.

### delta_q preservation

Because $\mathbf{G}$ is updated as a **mean** over all pixels — not per-pixel — the
incremental difference $\Delta\mathbf{q} = \mathbf{k}_{in,j_1} - \mathbf{k}_{in,j_2}$
is preserved to leading order.  The Ewald projection introduces a small second-order
correction to $\Delta\mathbf{q}$ that is proportional to $(\Delta\mathbf{k}_{out})^2 / k$
and is negligible for narrow streaks (small pupil angle).

---

## Coplanar Projection (`_coplanar_projection`)

Given $S \ge 3$ scattering vectors $\{\mathbf{G}_s\}$, find the plane through the origin
that best contains them all and project each $\mathbf{G}_s$ onto it.

**Best-fit plane normal via SVD.**  Stack the G vectors into a matrix:

$$\mathbf{M} = \begin{pmatrix} \mathbf{G}_1^\top \\ \vdots \\ \mathbf{G}_S^\top \end{pmatrix} \in \mathbb{R}^{S \times 3}.$$

The SVD $\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ gives right singular vectors
$\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ ordered by decreasing singular value.  The smallest
singular value corresponds to the direction that minimises the sum of squared projections of all
$\mathbf{G}_s$ — i.e. the direction they are most orthogonal to.  The plane normal is:

$$\hat{\mathbf{n}} = \mathbf{v}_3 \quad (\text{smallest right singular vector}).$$

**Projection onto the plane.**  For each streak:

$$\mathbf{G}_s^{\mathrm{proj}} = \mathbf{G}_s - (\hat{\mathbf{n}} \cdot \mathbf{G}_s)\,\hat{\mathbf{n}}.$$

This removes the out-of-plane component while leaving the in-plane component unchanged.  When
C7 is exactly satisfied, $\hat{\mathbf{n}} \cdot \mathbf{G}_s = 0$ for all $s$ and the
projection is a no-op.

**Placement in the outer loop.**  The coplanar projection is applied *after* all per-streak
Ewald steps in each outer iteration.  Because it changes $\mathbf{G}_s$, the final iteration
is followed by one additional Ewald projection to re-sync $\mathbf{k}_{in}$ with $|\mathbf{k}_{in}| = k$.

---

## Pupil-Support Clamp (optional, `_pupil_clamp_G`)

When `clamp_to_pupil=True`, after each G update the transverse components are clipped to
the feasible interval derived from C6:

$$G_s^x \leftarrow \operatorname{clip}\!\left(G_s^x,\;\max_j k_{out,j}^x - k_x^{\max},\;\min_j k_{out,j}^x - k_x^{\min}\right),$$

and identically for $y$.  This is only reliable when the streak exactly fills the pupil
(width ratio $\approx 1$).  When the streak is wider than the pupil the feasible interval
is empty, the clip anchors one edge of the kin distribution to the pupil boundary while
the other overshoots, producing apparent stretching.  Leave `clamp_to_pupil=False`
(the default) to preserve $\Delta q$.

---

## Width Ratio Diagnostic

The width ratio for streak $s$ is computed on exit:

$$\text{width\_ratio}_s^x = \frac{\max_j k_{out,j}^x - \min_j k_{out,j}^x}{k_x^{\max} - k_x^{\min}}, \qquad
\text{width\_ratio}_s^y = \frac{\max_j k_{out,j}^y - \min_j k_{out,j}^y}{k_y^{\max} - k_y^{\min}}.$$

A ratio $\approx 1$ means the streak exactly fills the pupil (as expected from C6).  A
systematic deviation across all streaks indicates that the sample-to-detector distance $d$
is miscalibrated, since the transverse $k$-scale is $k \propto 1/d$.

---

## Outputs and Diagnostics

| Output | Description |
|--------|-------------|
| `G_list` | List of refined $\mathbf{G}_s$ vectors, one per streak |
| `kins_list` | List of $(N_s, 3)$ arrays with $|\mathbf{k}_{in}| = k$ exactly, one per streak |
| `width_ratios` | $(n_\text{streaks}, 2)$ — streak width / pupil width in $x$ and $y$ |
| `history` | Per-iteration dicts: `iteration`, `G_list`, `width_ratios`, `dG` |

---

## Initialisation and Recommended Workflow

The Ewald circle projection requires a starting $\mathbf{G}_s$ that is already in roughly
the right direction (in particular $G_{xy}$ must be non-trivial, as it is not estimated
internally from the streak extent by default).

**Recommended sequence:**

```
1. compute_kins(roi, pupil_roi, est_ttheta)         # sets roi.g_init via calc_qvec
   or
   compute_kins_w_geometry(roi, pupil_roi)           # also refines detector distance

2. compute_kins_projection(roi, pupil_roi,
       n_iterations=..., n_inner=...)
   # roi.g_init / roi.g_list is forwarded automatically as G_init_list
```

Step 1 provides a geometrically consistent $\mathbf{G}$ (correct direction and approximate
magnitude).  Step 2 then refines it so that $|\mathbf{k}_{in}| = k$ is enforced exactly,
without distorting $\Delta q$ along the streak.

---

## Iterating Refinement (`iterate_kin_refinement`)

For a single-streak ROI, a complementary algorithm alternates between a **geometry step**
and an **angle step**, accepting the projection result as initialisation.

### Geometry step — `optimise_kin_w_geometry`

Jointly minimises over detector distance $d$ and Euler rotation angles $(\alpha, \beta, \gamma)$:

$$\mathcal{L}_{\text{geo}} = \frac{w_\text{elastic}\,r_\text{elastic} + w_\text{pupil}\,r_\text{pupil}}{w_\text{elastic} + w_\text{pupil}},$$

where

$$r_\text{elastic} = \sum_j \bigl||\mathbf{k}_{in,j}^{\text{calc}}| - k\bigr|, \qquad
r_\text{pupil} = \sum_j \bigl(\max(0,\, k_{in,j}^x - k_x^{\max}) + \max(0,\, k_x^{\min} - k_{in,j}^x) + \text{(y terms)}\bigr).$$

### Angle step — `optimise_kin`

Minimises over Euler angles $(\alpha, \beta, \gamma)$ the 2theta residual:

$$\mathcal{L}_\text{angle} = \sum_j \left|\frac{\hat{\mathbf{k}}_{in,j} \cdot \hat{\mathbf{k}}_{out,j} - \cos(2\theta)}{\cos(2\theta)}\right|.$$

### Alternating-refinement cycle

```
for iteration = 1 … n_iterations:
    (kin, kout, kin_pupil, params) ← optimise_kin_w_geometry(G, d, …)
    d ← params[0]
    G ← R(params[1:]) @ G

    optionally: θ ← estimate_ttheta(kout, kin_pupil)

    (kin, angles) ← optimise_kin(G, θ, kout, …)
    G ← R(angles) @ G
```
