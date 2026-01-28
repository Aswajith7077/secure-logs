# utils/loss.py
import torch
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.05):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    similarity = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)

    return F.cross_entropy(similarity, labels)


def matching_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def joint_loss(logits, labels, z1, z2, alpha=1.0):
    ttm = matching_loss(logits, labels)
    ttc = info_nce_loss(z1, z2)
    return ttm + alpha * ttc
