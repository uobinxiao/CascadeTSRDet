from detectron2.engine import DefaultTrainer, DefaultPredictor
import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from imgaug.augmenters.arithmetic import Cutout
from imgaug.augmenters import Flipud, Fliplr
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from detectron2.evaluation import COCOEvaluator
import cv2
import numpy
import random
import copy
import torch
import os

def random_pairs(length):
    def pop_random(l):
        index = random.randrange(0, len(l))
        return l.pop(index)
    
    l = [ x for x in range(0, length)]
    pair_list = []
    while len(l) > 1:
        rand1 = pop_random(l)
        rand2 = pop_random(l)
        pair_list.append((rand1, rand2))

    return pair_list

class LayoutMapper:
    def __init__(self, if_augmentation, cfg):
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.if_augmentation = if_augmentation
        self.shortest_side_range = [80, 160, 320, 640, 672, 704, 736, 768, 800, 1000]
        self.resized_height = 1333
        self.resized_width = 800

    def get_aug_list(self):
        if self.if_augmentation:
            return [T.ResizeShortestEdge(self.shortest_side_range, self.resized_height, sample_style='choice'),]
        else:
            return [T.ResizeShortestEdge([self.resized_width, self.resized_width], self.resized_height, sample_style='choice')]
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        #image = cv2.imread(dataset_dict["file_name"])
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
        utils.check_image_size(dataset_dict, image)

        annotations = dataset_dict["annotations"]

        if self.num_classes == 7:
            table_list = []
            column_list = []
            row_list = []
            projected_row_list = []
            projected_row_bbox_list = []
            header_list = []
            header_bbox_list = []
            spanning_list = []
            for anno in annotations:
                category_id = anno["category_id"]
                if category_id == 0:
                    table_list.append(anno)
                elif category_id == 1:
                    column_list.append(anno)
                elif category_id == 2:
                    row_list.append(anno)
                elif category_id == 3:
                    spanning_list.append(anno)
                elif category_id == 4:
                    projected_row_list.append(anno)
                    projected_row_bbox_list.append(anno["bbox"])
                elif category_id == 5:
                    header_list.append(anno)
                    header_bbox_list.append(anno["bbox"])

            new_row_list = []
            for row in row_list:
                if row["bbox"] in projected_row_bbox_list:
                    continue
                elif row["bbox"] in header_bbox_list:
                    index = header_bbox_list.index(row["bbox"])
                    header_list[index]["category_id"] = 6
                else:
                    new_row_list.append(row)

            dataset_dict["annotations"] = table_list + column_list + spanning_list + new_row_list + projected_row_list + header_list 

        augmentations = T.AugmentationList(self.get_aug_list())
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = augmentations(aug_input)
        image = torch.tensor(aug_input.image.transpose(2, 0, 1).astype("float32"))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(aug_input.sem_seg.astype("long"))
        
        annos = [
                utils.transform_instance_annotations(annotation, [transforms], image.shape[1:])
                for annotation in dataset_dict.pop("annotations")
                ]
        dataset_dict["image"] = image
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:])
        
        return dataset_dict


class LayoutTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=LayoutMapper(if_augmentation = True, cfg = cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=LayoutMapper(if_augmentation = False, cfg = cfg))
