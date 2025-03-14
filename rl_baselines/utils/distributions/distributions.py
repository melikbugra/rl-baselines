import torch
from torch.distributions import Normal, Categorical, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


class MultiCategorical:
    """
    Wrap a list of independent Categorical distributions (one for each discrete action)
    and provide a unified interface.
    """

    def __init__(self, dists):
        self.dists = dists

    def sample(self):
        # Sample from each distribution and stack into one tensor.
        samples = [d.sample() for d in self.dists]
        # Assume each sample has shape (batch_size,); stack along last dim.
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions):
        # Expect actions to be a tensor of shape (..., num_discrete)
        # Compute the log_prob of each component and sum them.
        log_probs = [d.log_prob(actions[..., i]) for i, d in enumerate(self.dists)]
        return sum(log_probs)

    def entropy(self):
        entropies = [d.entropy() for d in self.dists]
        return sum(entropies)


class SquashedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def log_prob(self, value):
        gaussian_value = torch.atanh(torch.clamp(value, -1 + 1e-6, 1 - 1e-6))
        log_prob = super().log_prob(gaussian_value)

        # Apply the tanh correction
        log_prob -= torch.sum(torch.log(1 - value.pow(2) + 1e-6), dim=-1)
        return log_prob

    def rsample(self):
        action = super().rsample()
        return torch.tanh(action)


class TanhNormal(TransformedDistribution):
    """
    A distribution obtained by taking a Normal distribution,
    applying a tanh transform to squash to [-1, 1], and then an affine
    transformation to scale and shift the output into the environment's
    action range [low, high]. This distribution also overrides its entropy
    method using a Monte Carlo approximation.
    """

    def __init__(self, loc, scale, low, high, eps=1e-6):
        self.eps = eps
        self.low = low
        self.high = high
        base_dist = Normal(loc, scale)
        # First, TanhTransform squashes outputs to (-1,1); then AffineTransform maps (-1,1) to (low, high)
        affine = AffineTransform(loc=(low + high) / 2.0, scale=(high - low) / 2.0)
        transforms = [TanhTransform(cache_size=1), affine]
        super().__init__(base_dist, transforms)
        self.loc = loc
        self.scale = scale
        self.base_dist = base_dist  # store the base distribution

    def entropy(self, n_samples=10):
        """
        Approximate the entropy of the squashed distribution via Monte Carlo.
        Samples n_samples times and returns the average negative log probability.
        Note: This approximation is not differentiable and may add extra variance.
        """
        # samples shape: (n_samples, batch_size, event_dim)
        samples = self.rsample(torch.Size([n_samples]))
        # log_probs shape: (n_samples, batch_size)
        log_probs = self.log_prob(samples)
        # Entropy ≈ -E[log_prob]
        return -log_probs.mean(dim=0)
