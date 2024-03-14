import torch
import torch.nn as nn
import metrics

class JaccardLoss(nn.Module):
    def __init__(self, class_weights=1.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.name = "Jaccard"

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        losses = 0
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            losses += 1 - metrics.fscore(ypr, ygt)
        return losses
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", None]
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.name = "Focal"

    def forward(self, input, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        else:
            focal_loss = focal_loss.sum()
        return focal_loss