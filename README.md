# EAIN: Element-wise Action Importance Estimation for Adaptive Exploration in High-Dimensional Action Spaces
**Keywords**: Reinforcement Learning, Maximum Entropy RL, Exploration, Reproducibility<br><br>

## 💡Motivation (Why I Built This)
The importance of each action dimension can vary depending on the current state. Excessive exploration in action dimensions that are less important at a given state can be inefficient and unnecessary.

<p align="center">
<img width="2553" height="845" alt="image" src="https://github.com/user-attachments/assets/5c87f0d9-bc55-4129-8b07-c7e7b0dc1a1b" />
</p>
<p align="center">
<i>Varying Joint Importance Across States</i>
</p>

Consider a 6-DoF robot arm with a gripper. At the start of a grasping task — when the gripper is distant from the object — excessive exploration of wrist and gripper joints can be unnecessary or even harmful, while base joints may matter more for reachability. As the dimensionality increases, this inefficiency becomes more pronounced.

These observations motivate an approach that allocates exploration according to the state-dependent importance of each action dimension, rather than applying uniform exploration across all dimensions.

<br>

## 🤖Overview
Exploration in high-dimensional reinforcement learning is fundamentally challenging, as the agent must navigate a complex action space. Although maximum entropy RL mitigates some of these difficulties by providing a principled mechanism for encouraging broad and consistent exploration, it can still cause misaligned or overly diffuse exploration behaviors in complex control tasks, sometimes leading to unstable policy updates and suboptimal learning dynamics (Zhang et al., [2025](https://arxiv.org/abs/2506.05615)). This study argues that such instability arises because standard entropy regularization treats all action dimensions identically, injecting unnecessary randomness into dimensions that do not contribute meaningfully to policy improvement.

To address this, this study introduces an auxiliary **E**lement-wise **A**ction **I**mportance **N**etwork (**EAIN**) that estimates the state-dependent importance of each action dimension. These importance values are used to apply dimension-wise weighting exclusively to the entropy term of the policy objective, allowing  adaptive exploration to focus on reward-relevant dimensions while suppressing extraneous entropy from less influential ones.

Experiments on the high-dimensional Humanoid-v5 benchmark demonstrate that this method significantly reduces variance in evaluation returns. In 2M-step experiments, with fixed α, it achieves up to a 36% reduction in average standard deviation across 7 seeds and a 37% reduction over the last 500k steps compared to baseline SAC. With auto-tuned α, the method similarly yields notable improvements, reducing overall standard deviation by 18% and last-500k standard deviation by 35%. A longer 5M-step experiment with 5 seeds under auto-tuned α shows consistent variance-reduction effects: the overall standard deviation across training decreases by 27.4%, and the last-1M-step standard deviation decreases by 50.1%.

These findings highlight the effectiveness of dimension-wise entropy weighting in stabilizing policy learning in complex, high-dimensional action spaces, ultimately improving the reproducibility of maximum entropy RL training outcomes.

<br>

## 🔎Approach
<p align="center">
<img width="2166" height="1142" alt="image" src="https://github.com/user-attachments/assets/90262577-5530-43da-b29b-a4aee56ddb03" />
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
To address this, a dimension-wise importance vector $\hat{w} \in \mathbb{R}^d$ is introduced.
The modified policy objective becomes:

$$
J(\pi_\theta) = 𝔼_{s\sim D, a\sim\pi_\theta(⋅ \mid s)} \ [Q_\psi(s,a) - \alpha  \sum_{i=1}^d  \hat{w}_i(s) \ \log \ \pi_\theta(a_i \mid s)].
$$

Here $\hat{w}_i(s)$ adaptively scales the entropy contribution of each action dimension.

---

### 3. Element-wise Action Importance (EAI) Network
The importance weights are predicted by an auxiliary network:

$$
\mathbf{\hat{w}} = f_\phi(s), \quad \mathbf{\hat{w}} \in \mathbb{R}^d,
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
L_\pi(\theta;\phi) = -𝔼_{s\sim D, a\sim\pi_\theta(⋅ \mid s)} \ [Q_\psi(s,a) - \alpha \sum_{i=1}^d \hat{w}_i(s) \ \log \ \pi_\theta(a_i \mid s)], \quad \hat{w}(s) = f_\phi(s).
$$

- The EAI network output $f_\phi(s)$ modulates entropy.
- Policy gradients update $\theta$, not $\phi$. ($f_\phi(s)$ is detached in this step.)

#### (b) Critic Loss
The critics are updated as in SAC, using the Bellman error:

$$
y(r, s', a') 
= r + \gamma\mathbb{E}_{a' \sim \pi(\cdot \mid s')}
\left[
Q_{\bar{\psi}}(s', a') - \alpha \log \pi(a' \mid s')
\right],
$$

$$
L_{Q}(\psi) =
\mathbb{E}_{(s,a,r,s')\sim D}
\left[
\left( Q_{\psi}(s, a) - y(r, s', a') \right)^2
\right]
$$

#### (c) EAI Loss
The EAI network is trained with a regression objective against a proxy importance signal 
$w(s)$, derived from the action-gradient of the Q-function:

$$
w_i(s) \propto \left|{\frac{\partial Q_\psi(s,a)}{\partial a_i}}\right|,
$$

scaled across dimensions.

The EAI loss is then:

$$
L_{EAI}(\phi) = 𝔼_{s \sim D} \left| \left| f_\phi(s) - w(s) \right| \right| ^2
$$

Through this loss, the EAI network learns a state-dependent approximation of a proxy signal to estimate the importance of each action dimension for entropy weighting.

<br>

## 📉Experiments

### Result 1 (2M; 7 seeds) 
<img width="2379" height="1135" alt="image" src="https://github.com/user-attachments/assets/dbd9f4f6-9a36-4055-9677-b77e10cda8f0" />
<p align="center">
<i>Learning Curves for Result 1: Learning curves on Humanoid-v5 over 2M environment interactions (7 seeds; mean ±1 s.d.)</i>
</p>

<br>

<div align="center">

| **Fixed α** | Final return (mean ± s.d.) | Last 500K steps s.d. | Overall avg s.d. |
|:------------------:|:---------------:|:-----------:|:------------------:|
| SAC (Baseline) | 5361.6 ± 295.9 | 678.3 | 824.4 |
| SAC + **EAIN** | 5242.1 ($${\color{red}\text{-2.2\\%}}$$) ± 137.0 ($${\color{green}\text{-53.7\\%}}$$) | 428.4 ($${\color{green}\text{-36.8\\%}}$$) | 527.6 ($${\color{green}\text{-36.0\\%}}$$) |

| **Auto-tuned α** | Final return (mean ± s.d.) | Last 500K steps s.d. | Overall avg s.d. |
|:----------------:|:-------------:|:---------:|:----------------:|
| SAC (Baseline) | 4933.2 ± 457.6 | 519.1 | 525.6 |
| SAC + **EAIN** | 5165.1 ($${\color{green}\text{+4.7\\%}}$$) ± 263.8 ($${\color{green}\text{-42.3\\%}}$$) | 335.7 ($${\color{green}\text{-35.3\\%}}$$) | 431.4 ($${\color{green}\text{-17.9\\%}}$$) |

</div>

<p align="center">
<i>Table for Result 1</i>
</p>

✅**Analysis**: Across both the Fixed α and Auto-tuned α settings, applying EAIN consistently reduces seed-to-seed variance, leading to improved reproducibility. The method not only lowers the overall evaluation standard deviation but also stabilizes performance in the later stages of training, demonstrating more reliable learning dynamics across runs.

---
  
### Result 2 (5M; 5 seeds; auto-tuned α)

<img width="2558" height="984" alt="image" src="https://github.com/user-attachments/assets/8ae32bbe-546e-4b77-9fb0-128c0e99cea3" />
<br>
<p align="center">
<i>Learning Curves for Result 2: Learning curves on Humanoid-v5 over 5M environment interactions (5 seeds; mean ±1 s.d.; auto-tuned α)</i>
</p>

<br>

<div align="center">

| **Auto-tuned α** | Final return (mean ± s.d.) | Last 1M steps s.d. | Overall avg s.d. |
|:------------------:|:---------------:|:-----------:|:------------------:|
| SAC (Baseline) | 5774.3 ± 467.2 | 508.2 | 503.0 |
| SAC + **EAIN** | 5410.4 ($${\color{red}\text{-6.3\\%}}$$) ± 128.0 ($${\color{green}\text{-72.6\\%}}$$) | 253.7 ($${\color{green}\text{-50.1\\%}}$$) | 365.2 ($${\color{green}\text{-27.4\\%}}$$) |

</div>

<p align="center">
<i>Table for Result 2</i>
</p>

✅**Analysis**: In the 5M-step experiment with auto-tuned α, EAIN once again reduces variance throughout training, with the effect becoming especially pronounced toward the later stages. As training progresses, the variance reduction becomes more apparent, and the learning curve exhibits a more stable and reliable convergence behavior compared to baseline SAC.

---

### Setup
- **Environment:** Humanoid-v5 (Gymnasium / MuJoCo), continuous high-dimensional action space
- **Compared Methods**
  - **SAC (baseline)** — standard entropy bonus
  - **SAC + EAIN** — entropy term reweighted by state-dependent, per-dimension importance
    - **α auto-tuning with adaptive target entropy**: because weighting changes the scale of the entropy term (typically $$\sum_i w_i < d, \ d: action \ dim$$), the target entropy is slowly adapted to this scale to avoid alpha miscalibration.
- **Training Steps:** 2/5M environment steps
- **Evaluation:** every 10K steps, 10 evaluation episodes (no exploration noise; mean action)

- **Metrics**
  - **Return (mean ± std)** over seeds
  - **Variance/stability:** (i) overall std of evaluation return, (ii) std over the **last 500K/1M** steps

<br>

## 📊Overall Summary: Stable Learning with Adaptive Exploration

Across both the 2M-step and 5M-step experiments, EAIN demonstrates a clear and consistent advantage in stabilizing policy learning without sacrificing performance. Despite applying selective suppression of entropy in less influential action dimensions, **mean returns remain comparable to those of standard SAC**, indicating that the method avoids harmful under-exploration.

Importantly, **early-stage convergence speed is preserved**, showing that adaptive entropy reduction does not slow down initial learning. As training progresses, the strengths of EAIN become increasingly apparent: **evaluation variance drops substantially**, and the learning curves exhibit **markedly more stable late-stage behavior**. This suggests that dimension-wise exploration control helps prevent the excessive, misaligned randomness that normally accumulates in high-dimensional action spaces.

Overall, these results show that EAIN effectively reduces variance while maintaining return levels and convergence speed, ultimately delivering a more stable and reproducible training process for maximum entropy RL in high-dimensional continuous-control tasks.

<br>

## 🤔What Can We Try Next?
- **Handling Early-Phase Uncertainty:** During early training, the Q-function is still inaccurate, making its action-gradient signal (∂Q/∂a) noisy and unreliable. Because EAIN uses this gradient to estimate per-dimension importance, early predictions can fluctuate and momentarily distort entropy weighting.
  
  - **Learning signal with RND bonus** — Applying Random Network Distillation (Burda et al., [2018](https://arxiv.org/abs/1810.12894)) adds an uncertainty bonus to the learning signal based on state visitation frequency. In early training, this encourages the per-dimension learning signal to take on a more uniform structure across action dimensions. [Branch](https://github.com/tae-roh/eain/tree/rnd)
  - **Pretrained EAIN as prior**
 
- **Boosting Importance-Based Weighting:** This approach not only restricts exploration in less important dimensions but also encourages greater exploration in those deemed more important.

  - **Bold EAIN** — Instead of scaling importance values to the 0–1 range, this strategy distributes weights so that the sum of per-dimension importances equals the action dimensionality, preserving relative importance while maintaining a fixed total exploration budget. [Branch](https://github.com/tae-roh/eain/tree/bold)

- **Designing More Stable Learning Signals**

