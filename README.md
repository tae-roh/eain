# EAIN: Element-wise Action Importance Estimation for Entropy Weighting in High-Dimensional Action Spaces

## Overview
Reinforcement learning in high-dimensional action spaces often suffers from inefficient exploration, as all action dimensions are treated equally in entropy regularization. This indiscriminate treatment forces noisy or low-impact dimensions to contribute as much stochasticity as influential ones, leading to unnecessary randomness and unstable policy updates. This study introduces a method that estimates the relative importance of each action dimension and applies element-wise weighting specifically to the entropy term of the policy objective. An auxiliary network adaptively predicts dimension-wise importance, allowing exploration to be concentrated on reward-relevant dimensions while suppressing extraneous entropy from less important ones.

Experiments on the high-dimensional Humanoid-v5 benchmark demonstrate that the proposed method reduces variance in evaluation returns. With fixed α, the approach achieves up to a 36% reduction in mean standard deviation across 7 seeds and a 37% reduction over the last 500k training steps compared to the baseline SAC. With auto-tuned α, the method still yields notable improvements, reducing the overall standard deviation by 18% and the last-500k-steps deviation by 35%.

These findings highlight the effectiveness of dimension-wise entropy weighting for stabilizing policy learning in complex, high-dimensional action spaces, under both fixed and auto-tuned entropy-temperature settings.

## Approach
<p align="center">
<img width="700" height="1109" alt="image" src="https://github.com/user-attachments/assets/0d3b2543-6920-4a50-ac75-bfb8e0b541a5" />
</p>
<p align="center">
<i>Overview of SAC with EAIN</i>
</p>

### 1. Problem Formulation
Soft Actor-Critic (SAC) optimizes a stochastic policy $\pi_\theta(a \mid s)$ with entropy regularization:

$$
J(\pi_\theta) = 𝔼_{s\sim D, a\sim\pi_\theta(⋅ \mid s)} \ [Q_\psi(s,a) - \alpha \ \log \ \pi_\theta(a \mid s)],
$$

where
- $\pi_\theta$: policy with parameters $\theta$,
- $Q_\psi$: critic with parameters $\psi$,
- $\alpha$: entropy temperature coefficient.

The entropy term can be decomposed as:

$$
\log \ \pi_\theta(a \mid s) = \sum_{i=1}^{d} \log \ \pi_\theta(a_i \mid s).
$$

This uniform treatment forces noisy or low-impact dimensions to contribute equally, leading to unstable updates in high-dimensional action spaces.

---

### 2. Dimension-wise Entropy Weighting
To address this, a dimension-wise importance vector $w \in \mathbb{R}^d$ is introduced.
The modified policy objective becomes:

$$
J(\pi_\theta) = 𝔼_{s\sim D, a\sim\pi_\theta(⋅ \mid s)} \ [Q_\psi(s,a) - \alpha  \sum_{i=1}^d  w_i(s) \ \log \ \pi_\theta(a_i \mid s)].
$$

Here $w_i(s)$ adaptively scales the entropy contribution of each action dimension.

---

### 3. Element-wise Action Importance (EAI) Network
The importance weights are predicted by an auxiliary network:

$$
\mathbf{w} = f_\phi(s), \quad \mathbf{w} \in \mathbb{R}^d,
$$

where
- $f_\phi$: EAI network with parameters $\phi$,
- Input: state $s$,
- Output: importance weights for each action dimension.

---

### 4. Training Objectives
#### (a) Policy Loss
The actor is optimized using the entropy-weighted objective:

$$
L_\pi(\theta;\phi) = -𝔼_{s\sim D, a\sim\pi_\theta(⋅ \mid s)} \ [Q_\psi(s,a) - \alpha \sum_{i=1}^d w_i(s) \ \log \ \pi_\theta(a_i \mid s)], \quad w(s) = f_\phi(s).
$$

- The EAI network output $f_\phi(s)$ modulates entropy.
- Policy gradients update $\theta$, not $\phi$. ($f_\phi(s)$ is detached in this step.)

#### (b) Critic Loss
The critics are updated as in SAC, using the Bellman error:

$$
L_Q(\psi) = \big(Q_\psi(s,a) - (r + \gamma \hat{Q}(s\prime, a\prime))\big)^2.
$$

#### (c) EAI Loss
The EAI network is trained with a regression objective against a proxy importance signal 
$\hat{w}(s)$, often derived from the action-gradient of the Q-function:

$$
\hat{w_i}(s) \propto \left|{\frac{\partial Q_\psi(s,a)}{\partial a_i}}\right|,
$$

scaled across dimensions.

The EAI loss is then:

$$
L_{EAI}(\phi) = 𝔼_{s \sim D} \left| \left| f_\phi(s) - \hat{w}(s) \right| \right| ^2
$$

This allows the EAI network to generalize noisy, local gradient signals into a smoother, state-dependent importance estimate.

## Experiments
<img width="2267" height="1168" alt="image" src="https://github.com/user-attachments/assets/d67d12e7-de02-4b75-8a1d-8532aadb6bd2" />
<p align="center">
<i>Learning curves on Humanoid-v5 over 2M environment interactions (7 seeds; mean ±1 s.d.)</i>
</p>

### Results
- **Fixed α (a)**: SAC + EAIN lowers the cross-seed variability compared to SAC. Overall average s.d. drops **36.0%** (from 824.4 to 527.6), and the last-500k-steps average s.d. drops **36.8%** (from 678.3 to 428.4), while achieving comparable final return (5242.1 ± 137.0 vs. 5361.6 ± 295.9).
- **Auto-tuned α (b)**: SAC + EAIN again improves stability over auto-tuned SAC. Overall average s.d. decreases **17.9%** (from 525.6 to 431.4) and the last-500k-steps average s.d. decreases **35.3%** (from 519.1 to 335.7), with a slightly higher final return (5165.1 ± 263.8 vs. 4933.2 ± 457.6).  

### Setup
- **Environment:** Humanoid-v5 (Gymnasium / MuJoCo), continuous high-dimensional action space
- **Compared methods:**
  - **SAC (baseline)** — standard entropy bonus
  - **SAC + EAIN** — entropy term reweighted by state-dependent, per-dimension importance
    - **α auto-tuning with adaptive target entropy**: because weighting changes the scale of the entropy term (typically $$\sum_i w_i < d, \ d: action \ dim$$), the target entropy is slowly adapted to this scale to avoid alpha miscalibration.
- **Training steps:** 2M environment steps
- **Evaluation:** every 10k steps, 10 evaluation episodes (no exploration noise; mean action)
- **Seeds:** 7 (report mean ± std across seeds)

### Metrics
- **Return (mean ± std)** over seeds
- **Variance/stability:** (i) overall std of evaluation return, (ii) std over the **last 500k** steps

