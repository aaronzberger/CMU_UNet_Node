import torch
import torch.nn as nn
import torch.nn.functional as F

from discriminative import DiscriminativeLoss


class ClassificationLoss(nn.Module):
    def __init__(self, device, num_classes=1):
        super(ClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        self.discriminative_loss = DiscriminativeLoss(0.5, 3.0, 2)

    def focal_loss(self, x, y, mask=None):
        '''Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.9
        gamma = 0

        log_x = F.logsigmoid(x)
        x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        log_x_t = log_x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * log_x_t

        if mask is not None:
            loss = loss * mask

        return loss.mean()

    def cross_entropy(self, x, y, weight=None):
        x = torch.sigmoid(x)

        return F.binary_cross_entropy(input=x, target=y, weight=weight, reduction='mean')


    def forward(self, preds, targets, mask=None):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        loss = self.cross_entropy(preds, targets, weight=mask)
        
        return loss