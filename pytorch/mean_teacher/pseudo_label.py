import torch
import torch.nn as nn
import torch.nn.functional as F

class PL(nn.Module):
    def __init__(self, n_classes, threshold, soft=False):
        super().__init__()
        self.n_classes = n_classes
        self.th = threshold
        self.soft = soft

    def forward(self, x, y, model, mask):
        y_probs = y.softmax(1)
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0] # reduce_any
        lt_mask = 1 - gt_mask # logical not
        p_target = gt_mask[:,None] * self.n_classes * onehot_label
        if self.soft:
            p_target += lt_mask[:,None] * y_probs
        model.module.update_batch_stats(False)
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        model.module.update_batch_stats(True)
        return loss

    def __make_one_hot(self, y):
        return torch.eye(self.n_classes)[y].to(y.device)
