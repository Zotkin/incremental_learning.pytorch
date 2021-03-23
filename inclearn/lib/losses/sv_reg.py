from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class SV_regularization_loss(nn.Module):

    def __init__(self, args: Dict):
        super().__init__()
        self._args = args

    def forward(self, training_network, metrics):
        linear_matrix = training_network.classifier.weights
        mean_linear_tensors = []
        num_proxy_per_class = self._args['classifier_config']['proxy_per_class']
        num_classes = int(linear_matrix.shape[0] / num_proxy_per_class)
        for i in range(num_classes):
            from_ = i * num_proxy_per_class
            to = (i + 1) * num_proxy_per_class
            mean_linear_tensors.append(torch.mean(linear_matrix[from_:to], axis=0, keepdim=True))
        linear_matrix_means = torch.cat(mean_linear_tensors)
        u, s, v = torch.svd(torch.matmul(linear_matrix_means, linear_matrix_means.T))
        sv_type = self._args['sv_regularization_type']
        if sv_type == "ratio":
            sv = s[0] / (s[-1] + 0.00001)
        else:
            sign = -1 if sv_type == "entropy_positive" else 1
            sv = sign * torch.sum(F.softmax(torch.sqrt(s), dim=0) * F.log_softmax(torch.sqrt(s), dim=0))

        norm = torch.mean(torch.norm(linear_matrix_means, dim=1))
        loss = self._args['sv_regularization_strength'] * sv + self._args['sv_regularization_strength'] * norm
        # sv regularization ends
        metrics[f"sv_{sv_type}"] += self._args['sv_regularization_strength'] * sv.item()
        metrics['norm'] += self._args['sv_regularization_strength'] * norm.item()
        return loss
