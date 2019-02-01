import torch
import numpy as np


def sample(self, dist, n_samples):
    if len(dist.shape) == 1:  # 1D distribution
        cumsum = dist.cumsum(dim=0).detach().cpu().numpy()
        ind = np.searchsorted(cumsum, cumsum[-1] * np.random.rand(self.n_samples))
        return torch.from_numpy(ind).to(dist.device)

    else:  # multidimensional distribution
        shape = dist.shape
        ind = self.sample(dist.contiguous().view(-1))
        result = torch.zeros((n_samples, len(shape))).to(dist).long()
        for i, s in enumerate(reversed(shape)):
            result[:, -(i + 1)] = ind % s
            ind = (ind - result[:, -i]) / s
        return result
