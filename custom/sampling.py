import torch
from detectron2.layers import nonzero_tuple
import numpy
import torch.nn.functional as F  
__all__ = ["subsample_labels"]

#def subsample_labels_by_group(labels: torch.Tensor, num_samples: int, positive_fraction:float, bg_label:int, gt_boxes):
#    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
#    negative = nonzero_tuple(labels == bg_label)[0]
#    pass

def subsample_labels(labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int, gt_boxes):
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    positive_boxes = gt_boxes[positive]
    negative_boxes = gt_boxes[negative]

    positive_n = (positive_boxes[:, 2] - positive_boxes[:, 0]) * (positive_boxes[:, 3] - positive_boxes[:, 1])
    sum_n = torch.sum(positive_n, dim = 0)
    prob = F.softmax(-1.0 * positive_n.float() / sum_n, dim = -1)

    perm1 = torch.multinomial(prob, num_samples = num_pos)

    # randomly select positive and negative examples
    #perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx
