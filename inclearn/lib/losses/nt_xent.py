__author__ = "https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py"

import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # TODO this might potentially lead to problems
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def get_correlated_samples_mask(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, p1):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        mask = self.get_correlated_samples_mask(self.batch_size)

        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )
        negative_samples = sim[mask].reshape(self.batch_size * 2, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # TODO According to documentation https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html,
        # if you inherit from LightningModule your class should have attribute device
        # In practice, such attribute is absent
        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size

        return loss