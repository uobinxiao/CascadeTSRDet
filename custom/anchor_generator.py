from detectron2.modeling import ANCHOR_GENERATOR_REGISTRY
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, BufferList
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, RotatedBoxes
import cv2
import torch
import math

def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device)
    
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    return shift_x, shift_y

@ANCHOR_GENERATOR_REGISTRY.register()
class MyAnchorGenerator(DefaultAnchorGenerator):

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        super().__init__(sizes = sizes, aspect_ratios = aspect_ratios, strides = strides, offset = offset)
    
    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
                #self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
                self.my_generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
                ]
        return BufferList(cell_anchors)
    
    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        # buffers() not supported by torchscript. use named_buffers() instead
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        
        return anchors

    def my_generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios= (0.5, 1, 2)):
        anchors = set()
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = size
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.add((x0, y0, x1, y1))

                h = size
                w = aspect_ratio * h
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.add((x0, y0, x1, y1))

        return torch.tensor(list(anchors))

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):

        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]
