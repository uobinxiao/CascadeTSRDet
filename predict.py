# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import random
import multiprocessing as mp
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm
import torch
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from torch.utils.data import Dataset, DataLoader
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.data import (DatasetCatalog, MetadataCatalog, MapDataset, get_detection_dataset_dicts, build_detection_test_loader )
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data import transforms as T_
from results import gen_blank_res_df, print_res_df, print_update_ic19_res_df, print_tncr_res_df, gen_blank_res_df_tncr, print_icttd_res_df, gen_blank_res_df_gtc, print_gtc_res_df
from custom import MyStandardROIHeads, MyStandardRPNHead, MyGeneralizedRCNN, MyFastRCNNConvFCHead, MyAnchorGenerator, MyRPN
from layout_trainer import LayoutMapper
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from custom_coco_summarize import Summarize
from detectron2.evaluation import inference_on_dataset, COCOEvaluator

def seed_everything(seed = 250):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def register_test_dataset(meta, img_dir, test_dataset_name, test_instances_json):
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    
    image_root = img_dir
    register_coco_instances(name=test_dataset_name, metadata=meta, image_root=image_root, json_file=test_instances_json)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weight_path
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def build_test_loader(test_json_path, test_dataset_name):
    def collate_fn(batch):
        return batch

    test_aug_list = [T_.ResizeShortestEdge([800, 800], 1333)]
    test_dataset = get_detection_dataset_dicts([test_dataset_name])
    test_dataset = MapDataset(test_dataset, DatasetMapper(cfg, augmentations=test_aug_list, is_train=False, image_format="RGB"))
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    return test_dataloader

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        #default="/data/logs/SciTSR_Cascade_Mask_RCNN_Output_R50_padded2_final/config.yaml",
        #default="/data/logs/PubTables1M_Cascade_RCNN_7classes_baseline/config.yaml",
        #default="/data/logs/FinTabNet_Cascade_Mask_RCNN_Output_R50_padded2_6class_deform/config.yaml",
        #default="/data/logs/FinTabNet_Cascade_Mask_RCNN_Output_R50_padded2_7classes/config.yaml",
        default="/data/logs/SciTSR_Cascade_Mask_RCNN_Output_R50_padded2_spatial_attn_test/config.yaml",
        #default="/data/logs/PubTables1M_Cascade_RCNN_7classes_final/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--weight_path",
        default="/data/logs/diffusionnet_tncr_r101/model_0009999.pth",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def predict(model, test_dataloader, test_json_path):
    #results_df = gen_blank_res_df()
    #results_df = gen_blank_res_row_col_df()
    results_df = gen_blank_res_df_tncr()
    #results_df = gen_blank_res_df_gtc()
    img = ""
    outputs = ""
    results = []
    with torch.no_grad():
        model.eval()
        for idx, items in enumerate(tqdm(test_dataloader)):
            outputs = model(items)
            boxes = outputs[0]["instances"].get_fields()["pred_boxes"].tensor.detach().cpu().numpy()
            scores = outputs[0]["instances"].get_fields()["scores"].detach().cpu().numpy()
            classes = outputs[0]["instances"].get_fields()["pred_classes"].detach().cpu().numpy()
            for idx, bbox in enumerate(boxes):
                bbox = list(bbox)

                #category id starts from 0
                #class_map = {
                #        'table': 0,
                #        'table column': 1,
                #        'table row': 2,
                #        'table spanning cell': 3,
                #        'table projected row header': 4,
                #        'table column header': 5,
                #        'no object': 6}

                #for rows
                #if classes[idx] == 4 : # Only tables are evaluated
                #    result = {
                #            "image_id": items[0]["image_id"],
                #            "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                #            "category_id": classes[idx] + 1,
                #            "score": scores[idx]
                #            }
                #    results.append(result)

                result = {
                        "image_id": items[0]["image_id"],
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "category_id": classes[idx] + 1,
                        "score": scores[idx]
                        }
                results.append(result)
        
        if len(results) > 0:
            #Coco eval
            coco_gt = COCO(test_json_path)
            coco_dt = coco_gt.loadRes(results)

            annType = 'bbox'
            coco_eval = COCOeval(coco_gt, coco_dt, annType)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            #For result_df
            summary = Summarize(coco_eval.stats, coco_eval.params, coco_eval.eval)
            #Returns an array of size 20
            #summary_dets = summary.summarizeDets()
            #summary_dets = summary.summarizeDets()
            summary_dets = summary.summarizeDetsTNCR()
            #summary_dets = summary.summarizeDetsGTC()
            results_df.loc[len(results_df)] = summary_dets
        else:
            print("No results yet")
            results_df.loc[len(results_df)] = [0] * 16
    #print_res_df(results_df)
    #print_res_df(results_df)
    print_tncr_res_df(results_df)
    #print_gtc_res_df(results_df)
    #print_update_ic19_res_df(results_df, combined = False)

    return results_df


if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = {"ting_classes": ['table', 'table column', 'table row', 'table spanning cell', 'table projected row header', 'table column header']}

    img_dir = "/data/datasets/TSR_Detection/SciTSR/scitsr_detection_format_padded/SciTSR.c-Image_Structure_PASCAL_VOC_PADDING_2/images"
    test_json_path = "/data/datasets/TSR_Detection/SciTSR/scitsr_detection_format_padded/pad2_6classes_test.json"

    register_test_dataset(meta = meta, img_dir = img_dir, test_dataset_name = "test_ic19", test_instances_json = test_json_path)
    
    cfg = setup_cfg(args)
    model = build_model(cfg)


    weight_list = glob.glob("/data/logs/SciTSR_Cascade_Mask_RCNN_Output_R50_padded2_spatial_attn_test/model_0059999.pth")

    #torch.cuda.manual_seed(seed)


    weight_list = sorted(weight_list)
    result_list = []
    for weight_path in tqdm(weight_list):
        #DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        print(weight_path)
        DetectionCheckpointer(model).load(weight_path)
        model.to(device)
        
        test_loader = build_test_loader(test_json_path, test_dataset_name = "test_ic19")
        evaluator = COCOEvaluator("test_ic19")
        ss = inference_on_dataset(model, test_loader, evaluator)
        print(ss)
        continue

        results_df = predict(model, test_loader, test_json_path)
        result_list.append(results_df)

    for redf in result_list:
        print_update_ic19_res_df(redf, combined = False)
        #print_icttd_res_df(redf, combined = False)
