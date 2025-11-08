


| Component                                 | Best setting                               | Why                                                           |
| :---------------------------------------- | :----------------------------------------- | :------------------------------------------------------------ |
| **Regularizer**                           | ✅ Parseval Regularization                 | Maintains orthogonality + equal norms                         |
| **Regularization strength λ**             | **1e−3** (for 64-width networks)           | Balanced orthogonality & expressiveness                       |
| **Width scaling rule**                    | λ scaled as (64 / width)²                  | Keeps regularization consistent across widths                 |
| **Applied to layers**                     | All hidden layers, not the final layer     | Preserves output flexibility                                  |
| **Activation function**                   | **tanh**                                   | Benefits most from well-conditioned weights (less saturation) |
| **Diagonal layer**                        | ✅ Added after each hidden layer           | Restores expressiveness while keeping orthogonality           |
| **Learnable input scale**                 | ✅ Enabled                                 | Further relaxes Lipschitz constraint (better capacity)        |
| **Grouping (`parseval_num_groups`)**      | **1** (full orthogonality)                 | Highest stable rank and best performance                      |
| **Angle normalization (`parseval_norm`)** | False (use full norm+angle regularization) | Combines both effects for best results                        |
| **RPO coefficient (`rpo_alpha`)**         | 0.5 (for continuous actions)               | Balances robustness and PPO stability                         |
| **Weight init**                           | Orthogonal                                 | Works synergistically with Parseval constraint                |
| **Hidden width**                          | 64                                         | Default width; scaling rule keeps results stable at 128       |



| Metric name       | Symbol / def                          | Used in which env              | Interpretation                             |
| ----------------- | ------------------------------------- | ------------------------------ | ------------------------------------------ |
| Average Return    | (R_{\text{avg}}(t))                   | Both                           | Overall competence over time               |
| Zero-Shot Return  | (R_{\text{ZS}})                       | Both                           | Immediate robustness after a regime change |
| Adaptation Gain   | (\Delta_{\text{Adapt}})               | Both                           | Ability to learn/improve within new regime |
| Forgetting        | (F(T_i))                              | ContinualPendulumSequence only | How much past tasks degrade over time      |
| Stable Rank       | (|W|_F^2 / |W|_2^2)                   | Both                           | Capacity / diversity of representation     |
| Cosine Similarity | avg pairwise row cosine in each layer | Both                           | Neuron redundancy / entanglement           |

