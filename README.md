# dd3d-supplement

## fork from https://github.com/TRI-ML/dd3d.git

## DD3D: "Is Pseudo-Lidar needed for Monocular 3D Object detection?"

Official [PyTorch](https://pytorch.org/) implementation of _DD3D_: [**Is Pseudo-Lidar needed for Monocular 3D Object detection? (ICCV 2021)**](https://arxiv.org/abs/2108.06417),


#### KITTI


The dataset must be organized as follows:

```
<DATASET_ROOT>
    └── KITTI3D
        ├── mv3d_kitti_splits
        │   ├── test.txt
        │   ├── train.txt
        │   ├── trainval.txt
        │   └── val.txt
        ├── testing
        │   ├── calib
        |   │   ├── 000000.txt
        |   │   ├── 000001.txt
        |   │   └── ...
        │   └── image_2
        │       ├── 000000.png
        │       ├── 000001.png
        │       └── ...
        └── training
            ├── calib
            │   ├── 000000.txt
            │   ├── 000001.txt
            │   └── ...
            ├── image_2
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── label_2
                ├── 000000.txt
                ├── 000001.txt
                └── ..
```

## Install

```bash

pip install -r requirements.txt

```

### Pre-trained DD3D models

The DD3D models pre-trained on dense depth estimation using DDAD15M can be downloaded here:
| backbone | download |
| :---: | :---: |
| DLA34 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped.pth) |
| V2-99 | [model](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth) |
| DLA34 | (https://pan.baidu.com/s/1vpncowhJjhivGNrqMLSTYg 提取码: 4f6q) |


### Validating installation

To validate and visualize the dataloader (including [data augmentation](./configs/defaults/augmentation.yaml)), run the following:

```bash
./visualize_dataloader.py +experiments=dd3d_kitti_dla34 SOLVER.IMS_PER_BATCH=4
```

To validate the entire training loop (including [evaluation](./configs/evaluators) and [visualization](./configs/visualizers)), run the [overfit experiment](configs/experiments/dd3d_kitti_dla34_overfit.yaml) (trained on test set):

```bash
./train.py +experiments=dd3d_kitti_dla34
```

### Predict

```bash
./predict.py +experiments=dd3d_kitti_dla34
```

### Evaluation

One can run only evaluation using the pretrained models:

```bash
./train.py +experiments=dd3d_kitti_dla34 EVAL_ONLY=True MODEL.CKPT=<path-to-pretrained-model>
# use smaller batch size for single-gpu
./train.py +experiments=dd3d_kitti_dla34  EVAL_ONLY=True MODEL.CKPT=<path-to-pretrained-model> TEST.IMS_PER_BATCH=4
```



## License

The source code is released under the [MIT license](LICENSE.md). We note that some code in this repository is adapted from the following repositories:

- [detectron2](https://github.com/facebookresearch/detectron2)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)

## Reference

```
@inproceedings{park2021dd3d,
  author = {Dennis Park and Rares Ambrus and Vitor Guizilini and Jie Li and Adrien Gaidon},
  title = {Is Pseudo-Lidar needed for Monocular 3D Object detection?},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  primaryClass = {cs.CV},
  year = {2021},
}
```
