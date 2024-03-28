import numpy as np
import torch
from torch.nn import functional as F


def compute_unsupervised_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)

    loss = F.cross_entropy(predict, target, ignore_index=255, reduction="none")
    loss = weight * torch.sum(loss) / torch.sum(target != 255)

    return loss
