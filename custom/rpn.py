from detectron2.modeling import RPN_HEAD_REGISTRY, PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN
from detectron2.modeling.matcher import Matcher
from detectron2.config import configurable
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from typing import Dict, List, Optional, Tuple, Union
from .proposal_utils import find_top_rpn_proposals
from .sampling import subsample_labels
from torch import nn
import torch

@RPN_HEAD_REGISTRY.register()
class MyStandardRPNHead(StandardRPNHead):
    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
            conv_dims (list[int]): a list of integers representing the output channels
                of N conv layers. Set it to -1 to use the same number of output channels
                as input channels.
        """
        super().__init__(in_channels = in_channels, num_anchors = num_anchors, box_dim = box_dim, conv_dims = conv_dims)

    def forward(self, features: List[torch.Tensor]):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))

        return pred_objectness_logits, pred_anchor_deltas

@PROPOSAL_GENERATOR_REGISTRY.register()
class MyRPN(RPN):
    @configurable
    def __init__(self, *, in_features: List[str], head: nn.Module, anchor_generator: nn.Module, anchor_matcher: Matcher, box2box_transform: Box2BoxTransform, batch_size_per_image: int, positive_fraction: float, pre_nms_topk: Tuple[float, float], post_nms_topk: Tuple[float, float], nms_thresh: float = 0.7, min_box_size: float = 0.0, anchor_boundary_thresh: float = -1.0, loss_weight: Union[float, Dict[str, float]] = 1.0, box_reg_loss_type: str = "smooth_l1", smooth_l1_beta: float = 0.0,):
        super().__init__(in_features = in_features, head = head, anchor_generator = anchor_generator, anchor_matcher = anchor_matcher, box2box_transform = box2box_transform, batch_size_per_image = batch_size_per_image, positive_fraction = positive_fraction, pre_nms_topk = pre_nms_topk, post_nms_topk = post_nms_topk, nms_thresh = nms_thresh, min_box_size = min_box_size, anchor_boundary_thresh = anchor_boundary_thresh, loss_weight = loss_weight, box_reg_loss_type = box_reg_loss_type, smooth_l1_beta = smooth_l1_beta,)

    def predict_proposals(self, anchors: List[Boxes], pred_objectness_logits: List[torch.Tensor], pred_anchor_deltas: List[torch.Tensor], image_sizes: List[Tuple[int, int]],):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.
        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i, sorted by their
            objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                    pred_proposals,
                    pred_objectness_logits,
                    image_sizes,
                    self.nms_thresh,
                    self.pre_nms_topk[self.training],
                    self.post_nms_topk[self.training],
                    self.min_box_size,
                    self.training,
                    )

    def _subsample_labels(self, label, gt_boxes):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.
        
        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0, gt_boxes)
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors( self, anchors: List[Boxes], gt_instances: List[Instances] ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.
        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances
        
        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
            
            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i, matched_gt_boxes_i)

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes
