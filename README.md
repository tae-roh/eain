# EAIN: Element-wise Action Importance Estimation for Entropy Weighting in High-Dimensional Action Spaces

## Overview
Reinforcement learning in high-dimensional action spaces often suffers from inefficient exploration, as all action dimensions are treated equally in entropy regularization. This indiscriminate treatment forces noisy or low-impact dimensions to contribute as much stochasticity as influential ones, leading to unnecessary randomness and unstable policy updates. This study introduces a method that estimates the relative importance of each action dimension and applies element-wise weighting specifically to the entropy term of the policy objective. An auxiliary network adaptively predicts dimension-wise importance, allowing exploration to be concentrated on reward-relevant dimensions while suppressing extraneous entropy from less important ones. Experiments on the high-dimensional Humanoid-v5 benchmark demonstrate that the proposed method reduces variance in evaluation returns, achieving up to a 36% reduction in mean standard deviation across 7 seeds and a 37% reduction in the last 500k training steps compared to the baseline. These findings highlight the effectiveness of dimension-wise entropy weighting for stabilizing policy learning in complex, high-dimensional action spaces.

## Approach
### 1. Problem Formulation
Soft Actor-Critic (SAC) optimizes a stochastic policy $\pi_\theta(a \mid s)$ with entropy regularization:

$$
J(\pi_\theta) = 𝔼_{(s,a)\sim D} \ [Q_\psi(s,a) - \alpha \ \log \ \pi_\theta(a \mid s)],
$$

where
- $\pi_\theta$: policy with parameters $\theta$,
- $Q_\psi$: critic with parameters $\psi$,
- $\alpha$: enropy temperature coefficient.

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
J(\pi_\theta) = 𝔼_{(s,a)\sim D} \ [Q_\psi(s,a) - \alpha  \sum_{i=1}^d  w_i(s) \ \log \ \pi_\theta(a_i \mid s)].
$$

Here $w_i(s)$ adaptively scales the entropy contribution of each action dimension.

---

### 3. Element-wise Action Importance (EAI) Network
The importance weights are predicted by an auxiliary network:

$$
w = f_\phi(s), \quad w \in \mathbb{R}^d,
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
L_\pi(\theta;\phi) = -𝔼_{(s,a)\sim D} \ [Q_\psi(s,a) - \alpha \sum_{i=1}^d w_i(s) \ \log \ \pi_\theta(a_i \mid s)], \quad w(s) = f_\phi(s).
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
<img width="1600" height="400" alt="sac_seed7" src="https://github.com/user-attachments/assets/76493d78-ba26-4a21-8106-fb29e0d908a5" />
<img width="1600" height="400" alt="eain_seed7" src="https://github.com/user-attachments/assets/708a5ab8-bef6-4fbc-ab3e-70a76cd4f348" />

*Figure 1.* Learning curves on Humanoid-v5. SAC with EAIN achieves a 36% reduction in the overall standard deviation of evaluation returns across seeds, and a 37% reduction in the standard deviation computed over the final 500k environment steps, relative to the SAC baseline.

### Setup
- **Environment:** Humanoid-v5 (Gymnasium / MuJoCo), continuous high-dimensional action space.
- **Compared methods:**
  - **SAC (baseline)** — standard entropy bonus.
  - **SAC + EAI** — entropy term reweighted by state-dependent, per-dimension importance.
- **Training steps:** 2M environment steps.
- **Evaluation:** every 10k steps, 10 evaluation episodes (no exploration noise; mean action).
- **Seeds:** 7 (report mean ± std across seeds).

### Metrics
- **Return (mean ± std)** over seeds.  
- **Variance/stability:** (i) overall std of evaluation return, (ii) std over the **last 500k** steps.

