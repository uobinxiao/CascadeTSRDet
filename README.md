# Rethinking Detection Based Table Structure Recognition for Visually Rich Document Images

## Paper Link
https://arxiv.org/abs/2312.00699

## Requirements
This codebase is built on top of [Detectron2](https://github.com/facebookresearch/detectron2). Follow the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install Detectron2.

## Datasets and Pretrained Model

|Dataset | Weights|
|--------|--------|
|[PubTables1M](https://huggingface.co/datasets/bsmock/pubtables-1m) | [PubTables1M](https://drive.google.com/drive/folders/1BTB3aWw7R1xeztAp7NPrwpV75sejbxtb?usp=sharing)|
|[FinTabNet](https://huggingface.co/datasets/bsmock/FinTabNet.c)|[FinTabNet](https://drive.google.com/drive/folders/1lM8ydqVo9Ksje1-L2UDXCN62Vst4Mu2e?usp=sharing)|
|[SciTSR](https://huggingface.co/datasets/uobinxiao/SciTSR_Detection)|[SciTSR](https://drive.google.com/drive/folders/1IogkVxQ1IkOpvqtieYYoTir-NrXHsNdg?usp=sharing)|

## Configuration and Training

## Inference and Evaluation
Check the inference.py and test.sh for the inference. A sample inference command could be:
```
python inference.py --mode recognize --structure_config_path <path of config.yaml> --structure_model_path <path of weight> --structure_device cuda --image_dir <dir of table images> --out_dir <output dir> --html --visualize --csv --crop_padding 0
```

Check the teds.py for calculating the TEDS score.

## Citing

Please cite our work if you think it is helpful:
```
@article{xiao2023rethinking,
  title={Rethinking Detection Based Table Structure Recognition for Visually Rich Documents},
  author={Xiao, Bin and Simsek, Murat and Kantarci, Burak and Alkheir, Ala Abu},
  journal={arXiv preprint arXiv:2312.00699},
  year={2023}
}
```

## Acknowledgement
This project heavily relys on [Table-Transformer](https://github.com/microsoft/table-transformer), especially for the post-processing part. We thank the authors for sharing their implementations and related resources.
