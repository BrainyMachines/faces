import torch
from torch.utils.data.sampler import Sampler
import numpy as np


class TripletSampler(Sampler):
    """Samples elements from a given list of triplet indices.
    Arguments:
        triplets (array): a Nx3 array of triplet indices
    """

    def __init__(self, triplets, batch_size):
        self.triplets = triplets
        self.batch_size = batch_size

    def __iter__(self):
        L = []
        N = self.triplets.shape[0]
        for i in range(0, N, self.batch_size):
            L.extend(self.triplets[i:i+self.batch_size, 0].ravel().tolist())
            L.extend(self.triplets[i:i+self.batch_size, 1].ravel().tolist())
            L.extend(self.triplets[i:i+self.batch_size, 2].ravel().tolist())
        return iter(L)

    def __len__(self):
        return self.triplets.shape[0] * self.triplets.shape[1]
