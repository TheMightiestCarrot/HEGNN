# HEGNN - velocity

* **Question.** Are higher-degree ( $l>1$ ) steerable representations actually unnecessary for E(3)/O(3)-equivariant GNNs (like EGNN) that only use Cartesian vectors?
* **Answer.** No. On many **symmetric graphs** (e.g., $k$-fold rotations and regular polyhedra), any O(3)-equivariant network restricted to fixed degree $l=1$ (or certain other degrees) will **collapse to the zero function**.
* **Fix.** **HEGNN**: an EGNN-style architecture that **keeps the scalarization trick** (messages are invariants) **but uses higher-degree steerable features** internally. It stays lightweight because cross-degree interactions are formed via inner products (invariants); only the feature update may use a compact CG tensor product.&#x20;

---

# Preliminaries (you’ll implement these)

## Geometric graph and group action

* A geometric graph with $N$ nodes:

  $$
  \mathcal{G}=(H,X;A),\quad H=\{h_i\in\mathbb{R}^{d_h}\}_{i=1}^N,\quad
  X=\{\mathbf{x}_i\in\mathbb{R}^3\}_{i=1}^N,\quad A\in\{0,1\}^{N\times N}.
  $$
* The relevant symmetry group is $O(3)=SO(3)\rtimes C_i$ (rotations $t\in SO(3)$ and inversion $m\in C_i=\{e,i\}$).
* **Equivariance.** A function $f:\mathcal{X}\to\mathcal{Y}$ is equivariant to $\mathcal{G}$ if

  $$
  \forall g\in \mathcal{G},\quad \rho_\mathcal{Y}(g)\,f(\bar{g}(x))=f(x),
  $$

  where $\rho$ is the representation on outputs and $\bar g$ the action on inputs. We work with $O(3)$.

## Irreps & steerable features

* For each **degree** $l\in\{0,1,2,\dots\}$, the real Wigner-D irrep $D^{(l)}(t)\in\mathbb{R}^{(2l+1)\times(2l+1)}$ acts on a $(2l+1)$-dimensional feature $\mathbf{v}^{(l)}$.
* Extend to $O(3)$ with inversion:

  $$
  \rho^{(l)}(tm)=\eta_l(m)\,D^{(l)}(t),\qquad \eta_l(e)=1,\ \eta_l(i)=(-1)^l.
  $$
* **Spherical harmonics.** For a unit vector $\hat{\mathbf{r}}=\mathbf{r}/\|\mathbf{r}\|$,

  $$
  Y^{(l)}(\hat{\mathbf{r}})\in\mathbb{R}^{2l+1},\qquad
  Y^{(l)}(-\hat{\mathbf{r}})=(-1)^l Y^{(l)}(\hat{\mathbf{r}}).
  $$

  A **modulated** steerable function uses a radial envelope $\varphi:\mathbb{R}_+\to\mathbb{R}$:

  $$
  f^{(l)}(\mathbf{r})=\varphi(\|\mathbf{r}\|)\,Y^{(l)}(\hat{\mathbf{r}}).
  $$
* **Invariant inner product between degree-$l$ features** (you’ll use this constantly):

  $$
  \langle \mathbf{a}^{(l)},\mathbf{b}^{(l)}\rangle \;=\; \sum_{m=-l}^l a^{(l)}_m\,b^{(l)}_m\ \in\ \mathbb{R}.
  $$

  This is O(3)-invariant.&#x20;

---

# Why $l=1$ (or some fixed degrees) fails on symmetric graphs

Let $\mathcal{S}\subset O(3)$ be a **finite symmetry group** of a graph (e.g., dihedral $D_k$, tetrahedral $T$, octahedral $O$, icosahedral $I$). For an O(3)-equivariant output of degree $l$, the **group-averaged representation**

$$
\bar\rho^{(l)}(\mathcal{S}) \;\triangleq\; \frac{1}{|\mathcal{S}|}\sum_{g\in\mathcal{S}}\rho^{(l)}(g)
$$

forces

$$
\bigl(I_{2l+1}-\bar\rho^{(l)}(\mathcal{S})\bigr)\, f^{(l)}(\mathcal{G})=0.
$$

If $I_{2l+1}-\bar\rho^{(l)}(\mathcal{S})$ is **nonsingular**, then $f^{(l)}(\mathcal{G})\equiv 0$. For finite groups, if $\bar\rho^{(l)}(\mathcal{S})=0$ then again $f^{(l)}\equiv 0$. This happens for many $(\mathcal{S},l)$ pairs; notably:

$$
\begin{array}{lll}
\text{2\(k\)-fold} & \mathcal{S}=C_i,D_{2k} & l\ \text{odd} \\
\text{(2\(k\)+1)-fold} & \mathcal{S}=D_{2k+1} & l<2k+1\ \text{and}\ l\ \text{odd} \\
\text{Tetrahedron} & \mathcal{S}=T & l\in\{1,2,5\} \\
\text{Cube/Octahedron} & \mathcal{S}=C_i,O & l=2\ \text{or}\ l\ \text{odd} \\
\text{Dodecahedron/Icosahedron} & \mathcal{S}=C_i,I & l\in\{2,4,8,14\}\ \text{or}\ l\ \text{odd}
\end{array}
$$

Hence a network restricted to $l=1$ (EGNN-style) **cannot distinguish** such symmetric structures.&#x20;

---

# HEGNN: the architecture (what to implement)

**Core idea.** Keep EGNN’s **invariant message passing** (scalarization) but maintain **steerable features** at multiple degrees $l=0,\dots,L$. Messages are scalar invariants formed by inner products **within each degree**; this is cheap and cross-degree interactions happen through the invariant channel (and optionally a compact CG update).

We maintain at each node $i$:

* scalar node state $h_i\in\mathbb{R}^{d_h}$,
* position $\mathbf{x}_i\in\mathbb{R}^3$,
* steerable features $\mathbf{v}_i^{(l)}\in\mathbb{R}^{2l+1}$ for $l=0,\dots,L$ (you may use $l=0$ as extra scalars).

We denote neighbors by $\mathcal{N}(i)$ and $\mathbf{r}_{ij}=\mathbf{x}_i-\mathbf{x}_j$, $d_{ij}=\|\mathbf{r}_{ij}\|$, $\hat{\mathbf{r}}_{ij}=\mathbf{r}_{ij}/d_{ij}$.

## 1) Initialization of steerable features

For each degree $l=1,\dots,L$:

$$
\boxed{\;\mathbf{v}^{(l)}_{i,\text{init}}
=\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}
\phi^{(l)}_{v,\text{init}}\!\Bigl(m_{ij,\text{init}}\Bigr)\cdot Y^{(l)}\!\bigl(\hat{\mathbf{r}}_{ij}\bigr)\;}
\tag{5}
$$

with an invariant scalar

$$
m_{ij,\text{init}}=\phi_{m,\text{init}}\!\bigl(h_i,h_j,e_{ij},d_{ij}\bigr).
$$

$\phi^{(l)}_{v,\text{init}},\ \phi_{m,\text{init}}$ are MLPs producing scalars.
*Note.* Eq. (5) cannot create **axial/pseudovector** type-1 features (even under inversion). If you need them (e.g., torque), multiply by a CG tensor product with $\mathbf{r}_{ij}$ to set the correct parity.&#x20;

## 2) Cross-degree **invariant** messages (scalarization)

Per edge $(i,j)$, form invariants:

$$
d_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|,\qquad
z^{(l)}_{ij}=\left\langle \mathbf{v}_i^{(l)},\mathbf{v}_j^{(l)}\right\rangle,\quad l=0,\dots,L.
$$

Then the message (a scalar, or a small scalar vector) is

$$
\boxed{\;m_{ij}=\phi_m\!\Bigl(h_i,h_j,e_{ij},d_{ij},\{z^{(l)}_{ij}\}_{l=0}^L\Bigr). \;}
\tag{6}
$$

$\phi_m$ is an MLP. This generalizes EGNN’s scalarization from $l=1$ to arbitrary degrees.&#x20;

## 3) Residual aggregation & updates

Aggregate scalar messages to update $h_i,\mathbf{x}_i,\mathbf{v}_i^{(l)}$:

$$
\boxed{\;
\Delta h_i=\phi_h\!\Bigl(h_i,\ \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} m_{ij}\Bigr)}
\tag{7a}
$$

$$
\boxed{\;
\Delta \mathbf{x}_i=\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} \phi_x(m_{ij})\;(\mathbf{x}_i-\mathbf{x}_j)}
\tag{7b}
$$

$$
\boxed{\;
\Delta \mathbf{v}_i^{(l)}=\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} \phi_{\varphi}^{(l)}(m_{ij})\;\bigl(\mathbf{v}_i^{(l)}-\mathbf{v}_j^{(l)}\bigr),\quad l=0,\dots,L}
\tag{8}
$$

where $\phi_h,\phi_x,\phi_{\varphi}^{(l)}$ are MLPs outputting scalars (gates).
Then apply residuals:

$$
\boxed{\;h_i\leftarrow h_i+\Delta h_i,\quad \mathbf{x}_i\leftarrow \mathbf{x}_i+\Delta \mathbf{x}_i,\quad
\mathbf{v}_i^{(l)}\leftarrow \mathbf{v}_i^{(l)}+\Delta \mathbf{v}_i^{(l)}\;}
\tag{9}
$$

**Optional (compact cross-degree mixing).** Eq. (8) can be re-written as a single CG tensor product across the direct sum $\bigoplus_{l=0}^L \mathbf{v}_i^{(l)}$, with weights produced from $m_{ij}$. This is a small **FullyConnectedTensorProduct** over all degrees, and is the only CG-heavy step if you enable it. (You can implement Eq. (8) without CG as above; enabling the compact CG gives a tighter implementation of mixing.)&#x20;

## 4) Readouts

* **Node-level invariant**: any MLP on invariants (e.g., pooled $\sum_j m_{ij}$, norms $\|\mathbf{v}_i^{(l)}\|^2$, etc.).
* **Node-level equivariant**: linear maps from steerable features of desired degree.
* **Graph-level**: pool node invariants (or append a graph head after the last layer).&#x20;

---

# Why these messages are expressive (intuition you’ll rely on)

Using only inner products **within** each degree, you can encode **all pairwise angles** between neighbors. In particular, with degree $l$,

$$
\Big\langle \sum_{s\in\mathcal{N}(i)} Y^{(l)}(\hat{\mathbf{r}}_{is}),\ 
\sum_{t\in\mathcal{N}(j)} Y^{(l)}(\hat{\mathbf{r}}_{jt})
\Big\rangle
= \frac{4\pi}{2l+1}\sum_{s,t} P^{(l)}\!\bigl(\cos\theta_{is,jt}\bigr),
\tag{10}
$$

where $P^{(l)}$ is the Legendre polynomial and $\theta_{is,jt}=\arccos(\hat{\mathbf{r}}_{is}\!\cdot\!\hat{\mathbf{r}}_{jt})$. Across degrees $l=0,1,2,\dots$, the $\{P^{(l)}\}$ form orthogonal bases, so the inner-product set $\{z_{ij}^{(l)}\}_{l}$ bijects with the set of edge angles. In practice, $L\le 6$ already works very well.&#x20;

---

# One HEGNN layer (drop-in recipe)

**Inputs per node $i$:** $h_i,\mathbf{x}_i,\{\mathbf{v}_i^{(l)}\}_{l=0}^L$.
**Per edge** $(i,j)$ (with $A_{ij}=1$):

1. $d_{ij}\leftarrow \|\mathbf{x}_i-\mathbf{x}_j\|$, $\hat{\mathbf{r}}_{ij}\leftarrow (\mathbf{x}_i-\mathbf{x}_j)/d_{ij}$.

2. For every $l$: $z_{ij}^{(l)}\leftarrow \langle \mathbf{v}_i^{(l)},\mathbf{v}_j^{(l)}\rangle$.

3. $m_{ij}\leftarrow \phi_m\bigl(h_i,h_j,e_{ij},d_{ij},\{z_{ij}^{(l)}\}_{l=0}^L\bigr)$.

**Node aggregations:**

$$
\Delta h_i \leftarrow \phi_h\!\left(h_i,\frac{1}{|\mathcal{N}(i)|}\sum_{j} m_{ij}\right),\quad
\Delta \mathbf{x}_i \leftarrow \frac{1}{|\mathcal{N}(i)|}\sum_{j}\phi_x(m_{ij})\,(\mathbf{x}_i-\mathbf{x}_j),
$$

$$
\forall l:\ \Delta \mathbf{v}_i^{(l)} \leftarrow 
\frac{1}{|\mathcal{N}(i)|}\sum_{j}\phi^{(l)}_{\varphi}(m_{ij})\,\bigl(\mathbf{v}_i^{(l)}-\mathbf{v}_j^{(l)}\bigr)
\quad\text{(or the single compact CG version).}
$$

**Residual updates:** apply Eq. (9).
**Repeat** for several layers; read out as needed.&#x20;

---

# Practical notes (hyperparameters & implementation choices)

* **Degrees.** $L\in[2,6]$ is typically enough; the paper finds $L\le 6$ sufficient to outperform both EGNN and high-degree CG-heavy baselines.&#x20;
* **MLPs.** $\phi_{m,\cdot},\phi_{v,\cdot}^{(l)},\phi_h,\phi_x,\phi_{\varphi}^{(l)}$ are standard small MLPs (scalar outputs for gates; optionally multi-scalar heads).
* **Normalization.** The $1/|\mathcal{N}(i)|$ factors in Eqs. (5)–(8) are part of the method—keep them.
* **Parity / axial types.** If you need pseudo-vectors (type-1 but inversion-even), compose with $\mathbf{r}_{ij}$ via a CG product to flip parity (footnote under Eq. 5).&#x20;
* **Efficiency.** Messages are **invariant scalars**, so there’s **no CG** in message construction; you may keep Eq. (8) without CG entirely. If you enable the **compact** CG update, it’s one small tensor product per edge. Overall cost is close to EGNN and much lower than TFN/MACE-style layers.&#x20;
* **Training.** Standard objectives (e.g., MSE for dynamics/forces). The paper validates on symmetric-graph classification, N-body, MD17; HEGNN is both more accurate than EGNN and faster than heavy CG models.&#x20;

---

# Why it works in practice (very short)

* The degeneracy theorems prove loss of expressivity at fixed degrees on symmetric graphs; HEGNN avoids it by letting degrees $l>1$ participate while still **routing all interactions through invariants** (cheap and stable).
* With degrees up to 6, inner-product sets $\{z_{ij}^{(l)}\}$ recover fine angular information (Eq. 10), which is what EGNN (using only $l=1$) cannot do under symmetry.&#x20;


wich is a "base", this document "extends" the base with velocity head and multihead attention:
# Spec: HEGNN with Vector Heads (A) + Equivariant Neighbor Attention

This document specifies an **E(3)-equivariant** message-passing GNN for **next-state prediction** in n-body dynamics. It extends HEGNN with:

1. a **shared equivariant trunk** and **two vector heads** that directly predict both next **positions** and next **velocities** (Option **A**), and
2. **multi-head invariant attention** over neighbors to prevent detail loss from uniform averaging.

No code is included—only definitions and equations sufficient for implementation.

---

## 0) Notation & Objects

* Graph $G=(V,E)$. For node $i$: neighbors $\mathcal N(i)=\{j\mid (i,j)\in E\}$.
* **Node state** at a layer:

  * Scalar (type-0) feature $h_i \in \mathbb R^{d_h}$ (E(3)-invariant).
  * Position $\mathbf x_i\in\mathbb R^3$ (type-1, vector).
  * Velocity $\mathbf v_i\in\mathbb R^3$ (type-1, vector).
  * Steerable features $\tilde{\mathbf v}_i^{(l)} \in \mathbb R^{2l+1}$ for degrees $l=0,\dots,L$ (SO(3)/O(3) reps).
* **Edge attributes** $e_{ij}\in\mathbb R^{d_e}$ (assumed invariant).
* Relative vectors:

  $$
  \mathbf r_{ij} \coloneqq \mathbf x_i-\mathbf x_j,\quad
  \Delta \mathbf v_{ij} \coloneqq \mathbf v_i-\mathbf v_j,\quad
  d_{ij}\coloneqq \|\mathbf r_{ij}\|_2,\quad \hat{\mathbf r}_{ij}\coloneqq \mathbf r_{ij}/d_{ij}.
  $$
* Spherical harmonics $Y^{(l)}:\mathbb S^2\to\mathbb R^{2l+1}$ (normalized).

All **MLPs** in this spec output **invariant scalars**, i.e., functions of invariant inputs (e.g., $h_i,h_j,e_{ij}, d_{ij},$ inner products), thereby preserving equivariance when used as gates.

---

## 1) Initialization of Steerable Features (unchanged from HEGNN)

For $l=0,\dots,L$:

$$
\tilde{\mathbf v}^{(l)}_{i,\text{init}}
\;=\;
\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}
\phi_{\tilde v,\text{init}}^{(l)}\!\big(m^{\text{init}}_{ij}\big)\cdot
Y^{(l)}\!\big(\hat{\mathbf r}_{ij}\big),
\tag{1}
$$

with

$$
m^{\text{init}}_{ij}
=
\phi_{m,\text{init}}\!\Big(h_i,h_j,e_{ij},\,d_{ij}^2\Big).
\tag{2}
$$

Here $\phi_{m,\text{init}},\phi_{\tilde v,\text{init}}^{(l)}$ are invariant MLPs. (Optionally use an RBF/Bessel expansion of $d_{ij}$ within $\phi_{m,\text{init}}$ for better radial resolution.)

---

## 2) Cross-Degree Scalarization (unchanged core)

Per edge and degree:

$$
z^{(l)}_{ij}\;\coloneqq\;\big\langle \tilde{\mathbf v}^{(l)}_i,\; \tilde{\mathbf v}^{(l)}_j\big\rangle, 
\qquad
d_{ij}=\|\mathbf r_{ij}\|_2.
\tag{3}
$$

Invariant edge **message**:

$$
m_{ij} \;=\;
\phi_m\!\Big(h_i,h_j,e_{ij},\, d_{ij}^2,\; z^{(0)}_{ij},\dots,z^{(L)}_{ij}\Big)
\;\in\;\mathbb R^{d_m}.
\tag{4}
$$

(Optionally include richer radial features of $d_{ij}$.)

---

## 3) Vector Heads (Option **A**): Equivariant Mixers for $\Delta\mathbf x_i,\Delta\mathbf v_i$

Each head computes **edge-level vector contributions** by mixing **vector bases** with **invariant scalar gates** derived from $m_{ij}$. Use at least the bases $\mathbf r_{ij}$ and $\Delta\mathbf v_{ij}$; optionally add a steerable basis.

### 3.1 Edge-level gates (invariant scalars)

$$
\begin{aligned}
a_{xx}(m_{ij}),\; a_{xv}(m_{ij}) &\in \mathbb R, &&\text{(position head)}\\
a_{vv}(m_{ij}),\; a_{vx}(m_{ij}) &\in \mathbb R, &&\text{(velocity head)}\\
c_x^{(1)}(m_{ij}),\; c_v^{(1)}(m_{ij}) &\in \mathbb R, &&\text{(optional steerable contribution)}
\end{aligned}
\tag{5}
$$

where each coefficient is produced by a small invariant MLP with input $m_{ij}$.

### 3.2 Edge-level vector contributions

$$
\begin{aligned}
\mathbf u^{x}_{ij}
&=
a_{xx}(m_{ij})\,\mathbf r_{ij}
\;+\;
a_{xv}(m_{ij})\,\Delta\mathbf v_{ij}
\;+\;
\underbrace{c_x^{(1)}(m_{ij})\big(\tilde{\mathbf v}^{(1)}_i-\tilde{\mathbf v}^{(1)}_j\big)}_{\text{optional}},\\[4pt]
\mathbf u^{v}_{ij}
&=
a_{vv}(m_{ij})\,\Delta\mathbf v_{ij}
\;+\;
a_{vx}(m_{ij})\,\mathbf r_{ij}
\;+\;
\underbrace{c_v^{(1)}(m_{ij})\big(\tilde{\mathbf v}^{(1)}_i-\tilde{\mathbf v}^{(1)}_j\big)}_{\text{optional}}.
\end{aligned}
\tag{6}
$$

Each term is “scalar $\times$ vector,” hence transforms as a vector; using differences ensures translation invariance.

---

## 4) **Equivariant Neighbor Attention** (multi-head, invariant weights)

To avoid uniform averaging, compute **input-dependent, per-receiver** weights that are **invariant scalars**, then use them to combine edge vector contributions.

### 4.1 Multi-head logits and weights

For heads $p=1,\dots,H$:

$$
\begin{aligned}
s^{x,(p)}_{ij} &= \psi^{x,(p)}\!\big(m_{ij}\big)\in\mathbb R,
&
\alpha^{x,(p)}_{ij}
&=
\frac{\exp\big(s^{x,(p)}_{ij}/\tau_x\big)}
{\sum\limits_{k\in\mathcal N(i)}\exp\big(s^{x,(p)}_{ik}/\tau_x\big)},
\\[4pt]
s^{v,(p)}_{ij} &= \psi^{v,(p)}\!\big(m_{ij}\big)\in\mathbb R,
&
\alpha^{v,(p)}_{ij}
&=
\frac{\exp\big(s^{v,(p)}_{ij}/\tau_v\big)}
{\sum\limits_{k\in\mathcal N(i)}\exp\big(s^{v,(p)}_{ik}/\tau_v\big)}.
\end{aligned}
\tag{7}
$$

* $\psi^{x,(p)},\psi^{v,(p)}$ are invariant MLPs.
* $\tau_x,\tau_v>0$ are temperatures (fixed or learnable, clipped to avoid collapse).
* Since logits depend only on **invariants**, all $\alpha$ are invariant scalars.

### 4.2 Head combination and aggregation

Two reasonable variants:

* **(Recommended, economical)**: shared edge vectors across heads; heads differ only by weights:

  $$
  \begin{aligned}
  \mathbf u^{x,(p)}_{ij} &= \mathbf u^{x}_{ij},\qquad
  \mathbf u^{v,(p)}_{ij} = \mathbf u^{v}_{ij},\\[2pt]
  \Delta\mathbf x_i &= \sum_{p=1}^H \sum_{j\in\mathcal N(i)} \alpha^{x,(p)}_{ij}\, \mathbf u^{x,(p)}_{ij},\\
  \Delta\mathbf v_i &= \sum_{p=1}^H \sum_{j\in\mathcal N(i)} \alpha^{v,(p)}_{ij}\, \mathbf u^{v,(p)}_{ij}.
  \end{aligned}
  \tag{8a}
  $$

* **(More expressive, costlier)**: head-specific gates (i.e., $a^{(p)}_{xx},a^{(p)}_{xv},a^{(p)}_{vv},a^{(p)}_{vx}$ from $m_{ij}$) to produce $\mathbf u^{x,(p)}_{ij},\mathbf u^{v,(p)}_{ij}$. Equivariance still holds because gates are invariant.

Either way, final aggregation is a **sum of invariant-weighted vectors**, hence equivariant.

---

## 5) Steerable Feature Update (unchanged core)

For each degree $l$:

$$
\Delta\tilde{\mathbf v}^{(l)}_{i}
\;=\;
\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}
\phi^{(l)}_{\tilde v}(m_{ij})\cdot\big(\tilde{\mathbf v}^{(l)}_i-\tilde{\mathbf v}^{(l)}_j\big)
\quad\text{(optionally via a small CG tensor product with scalar weights).}
\tag{9}
$$

Then:

$$
\tilde{\mathbf v}^{(l)}_i \leftarrow \tilde{\mathbf v}^{(l)}_i + \Delta\tilde{\mathbf v}^{(l)}_{i}.
\tag{10}
$$

---

## 6) Node Scalar Feature Update (unchanged core)

Aggregate invariant messages (mean or attention-weighted mean over neighbors) and update:

$$
\Delta h_i \;=\; \phi_h\!\Big(h_i,\; \operatorname{Agg}_{j\in\mathcal N(i)} m_{ij}\Big),
\qquad
h_i \leftarrow h_i + \Delta h_i.
\tag{11}
$$

(Aggregation may reuse the attention weights $\alpha^{x,(p)}$ or $\alpha^{v,(p)}$ or use a separate invariant attention.)

---

## 7) State Updates and Outputs

Per layer:

$$
\mathbf x_i \leftarrow \mathbf x_i + \Delta \mathbf x_i,
\qquad
\mathbf v_i \leftarrow \mathbf v_i + \Delta \mathbf v_i.
\tag{12}
$$

After $T$ layers, the model outputs $(\mathbf x_i^{\text{out}},\mathbf v_i^{\text{out}})$ to be interpreted as $(\mathbf x_{t+1},\mathbf v_{t+1})$. For **rollouts**, feed predictions back as inputs to the next step.

(If training/evaluating with multiple $\Delta t$, append $\Delta t$ as an extra invariant scalar to $m_{ij}$; equivariance is preserved.)

---

## 8) Equivariance/Invariant Properties (proof sketch)

Let $R\in O(3)$ and $\mathbf t\in\mathbb R^3$. Transform inputs:

$$
\mathbf x'_i=R\mathbf x_i+\mathbf t,\quad
\mathbf v'_i=R\mathbf v_i,\quad
h'_i=h_i,\quad
e'_{ij}=e_{ij}.
\tag{13}
$$

Then:

* $\mathbf r'_{ij}=\mathbf x'_i-\mathbf x'_j=R(\mathbf x_i-\mathbf x_j)=R\mathbf r_{ij}$,
  $\Delta\mathbf v'_{ij}=R\Delta\mathbf v_{ij}$.
* $d'_{ij}=d_{ij}$ (invariant), $Y^{(l)}(\hat{\mathbf r}'_{ij})=D^{(l)}(R)Y^{(l)}(\hat{\mathbf r}_{ij})$.
* Initialization (1) is equivariant; scalarization (3) yields invariants; thus $m'_{ij}=m_{ij}$.
* Gates $a_{\cdot\cdot}(\cdot),c^{(1)}(\cdot)$, and attention weights $\alpha(\cdot)$ are invariant.
* Therefore $\mathbf u^{x}_{ij}$ and $\mathbf u^{v}_{ij}$ transform as $R$-vectors; sums in (8) also transform as $R$-vectors.
* Final updates (12) give $\Delta\mathbf x'_i=R\Delta\mathbf x_i$, $\Delta\mathbf v'_i=R\Delta\mathbf v_i$, hence
  $\mathbf x'_i{}^{\text{out}}=R\mathbf x_i^{\text{out}}+\mathbf t$, $\mathbf v'_i{}^{\text{out}}=R\mathbf v_i^{\text{out}}$.

Thus, $h_i,\tilde{\mathbf v}^{(0)}_i$ are invariant, $\mathbf x_i,\mathbf v_i,\tilde{\mathbf v}^{(1)}_i$ are O(3)-equivariant and translation-invariant; higher degrees follow standard parity (even $l$: inversion-invariant, odd $l$: inversion-equivariant) as in HEGNN.

---

## 9) Optional Enhancements (mathematically compatible)

* **Richer radial basis**: replace $d_{ij}^2$ in (4) with $\rho(d_{ij})\in\mathbb R^{d_r}$ containing RBF/Bessel features and explicit $d_{ij}^{-2}, d_{ij}^{-1}, \log d_{ij}$ (properly clipped), improving near-/far-field resolution.
* **Multi-radius neighborhoods**: partition edges by radius bands and apply separate attention/gates per band; sum band contributions (all weights invariant).
* **Head-specific gates**: make $a_{\cdot\cdot}$ depend on $(p)$ in (8a) for more capacity; equivariance unchanged.

---

## 10) Implementation Checklist (no code)

1. **Inputs** per node: $h_i,\mathbf x_i,\mathbf v_i$; per edge: $e_{ij}$.
2. **Initialize** $\tilde{\mathbf v}^{(l)}_i$ via (1)–(2).
3. **Per layer**:

   * Compute $\mathbf r_{ij}, \Delta\mathbf v_{ij}, d_{ij}$, $z^{(l)}_{ij}$ by (3); build $m_{ij}$ by (4).
   * Produce gates $a_{\cdot\cdot}, c^{(1)}$ by (5).
   * Build edge vectors $\mathbf u^{x}_{ij}, \mathbf u^{v}_{ij}$ by (6).
   * Compute attention logits and weights by (7).
   * Aggregate with (8a) (or head-specific variant) to get $\Delta\mathbf x_i,\Delta\mathbf v_i$.
   * Update steerable features by (9)–(10) and scalars by (11).
   * Update $\mathbf x_i,\mathbf v_i$ by (12).
4. **Output** $\{\mathbf x_i,\mathbf v_i\}_{i\in V}$ as next state.
5. **Equivariance test**: verify (13) numerically holds to tolerance.

---

This design keeps the symmetry guarantees of HEGNN, **adds explicit velocity prediction**, and replaces uniform neighbor averaging with **multi-head invariant attention**, preserving fine-grained interactions needed for accurate rollouts.
