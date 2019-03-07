# Understanding Off-Policy Gradient Algorithms 

## Sample Complexity of Algorithms 
To what extend can we re-use the samples from Experience Replay Buffer. What does it mean to be truly sample efficient? How much can we re-use old samples, without doing 
"online updates anymore".

## Distribution Shift
The samples in the buffer are effectively from a different/old behaviour policy, under which the stationary state distribution is highly different than the state distribution under the current policy. To what extend does distribution shift matter? We know, we never correct for the distribution shift - do we actually need to?

## Effect of Critic 
To what extend does the critic overfitting to the sampled trajectories matter? In off-policy gradient algorithms, often the critic is to be blamed, leading to the instability of these algorithms. Does learning an accurate until convergence critic help? Do we want the critic to overfit?

## Off-Policyness of the algorithms 
How much can we interpolate between completely going off-policy and staying on-policy. Does the "off-policyness" of the algorithm matter?

## Variance Issues
Do we actually need the importance sampling corrections in off-policy algorithms like ACER? Does it indeed matter? Is there a trade-off? 
