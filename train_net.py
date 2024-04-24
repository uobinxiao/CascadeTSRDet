#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from custom import MyStandardROIHeads, MyStandardRPNHead, MyGeneralizedRCNN, MyFastRCNNConvFCHead, MyAnchorGenerator, MyRPN, build_resnet_backbone_with_spatial_attention
import logging
from collections import OrderedDict
from detectron2 import model_zoo
import sys
import argparse
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets.coco import register_coco_instances
from layout_trainer import LayoutTrainer
import yaml

class Trainer(LayoutTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    
    return cfg

def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    cfg = setup(args)
    register_coco_instances(name=args.train_set_name, metadata = args.meta, image_root = args.train_image_path, json_file = args.train_json_path)
    register_coco_instances(name=args.val_set_name, metadata = args.meta, image_root = args.val_image_path, json_file = args.val_json_path)
    register_coco_instances(name=args.test_set_name, metadata = args.meta, image_root = args.test_image_path, json_file = args.test_json_path)

    #if args.eval_only:
    #    model = LayoutTrainer.build_model(cfg)
    #    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    #    res = LayoutTrainer.test(cfg, model)
    #    if cfg.TEST.AUG.ENABLED:
    #        res.update(LayoutTrainer.test_with_TTA(cfg, model))
    #    if comm.is_main_process():
    #        verify_results(cfg, res)
    #    return res

    trainer = LayoutTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    
    return trainer.train()

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    args = argparse.Namespace(**config)
    #publaynet_args = argparse.Namespace(**args.PubTables1M)
    #publaynet_args = argparse.Namespace(**args.FinTabNet)
    publaynet_args = argparse.Namespace(**args.SciTSR)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(publaynet_args,),)
