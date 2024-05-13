from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from custom.box_head import build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.config import configurable
from custom.fast_rcnn import MyFastRCNNOutputLayers
from typing import Dict, List, Optional, Tuple, Union
import torch
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

@ROI_HEADS_REGISTRY.register()
class MyStandardROIHeads(StandardROIHeads):
    #@configurable
    def __init__(self, cfg, input_shape):
        super(MyStandardROIHeads, self).__init__(cfg, input_shape)

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        
        box_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
                )
        
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
                cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
                )
        box_predictor = MyFastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
                "box_in_features": in_features,
                "box_pooler": box_pooler,
                "box_head": box_head,
                "box_predictor": box_predictor,
                }

    #def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
    #    """
    #    Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
    #    the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
    #    Args:
    #        features (dict[str, Tensor]): mapping from feature map names to tensor.
    #        Same as in :meth:`ROIHeads.forward`.
    #        proposals (list[Instances]): the per-image object proposals with their matching ground truth.
    #        Each has fields "proposal_boxes", and "objectness_logits",
    #        "gt_classes", "gt_boxes".

    #    Returns:
    #        In training, a dict of losses.
    #        In inference, a list of `Instances`, the predicted instances.
    #    """
    #    features = [features[f] for f in self.box_in_features]
    #    box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
    #    box_features = self.box_head(box_features)
    #    predictions = self.box_predictor(box_features)
    #    del box_features
    #    
    #    if self.training:
    #        losses = self.box_predictor.losses(predictions, proposals)
    #        # proposals is modified in-place below, so losses must be computed first.
    #        if self.train_on_pred_boxes:
    #            with torch.no_grad():
    #                pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
    #                for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
    #                    proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
    #        return losses
    #    else:
    #        pred_instances, _ = self.box_predictor.inference(predictions, proposals)
    #        return pred_instances
