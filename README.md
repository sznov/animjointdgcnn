# AnimJointDGCNN – Learning Localization of Body and Finger Animation Skeleton Joints on Three-Dimensional Models of Human Bodies

Fork of PointMetaBase (https://github.com/linhaojia13/PointMetaBase).

Implementation of the network described in https://ieeexplore.ieee.org/document/10579426 (DOI: 10.1109/ZINC61849.2024.10579426).

Abstract: Contemporary approaches to solving various problems that require analyzing three-dimensional (3D) meshes and point clouds have adopted the use of deep learning algorithms that directly process 3D data such as point coordinates, normal vectors and vertex connectivity information. Our work proposes one such solution to the problem of positioning body and linger animation skeleton joints within 3D models of human bodies. Due to scarcity of annotated real human scans, we resort to generating synthetic samples while varying their shape and pose parameters. Similarly to the state-of-the-art approach, our method computes each joint location as a convex combination of input points. Given only a list of point coordinates and normal vector estimates as input, a dynamic graph convolutional neural network is used to predict the coefficients of the convex combinations. By comparing our method with the state-of-the-art, we show that it is possible to achieve significantly better results with a simpler architecture, especially for finger joints. Since our solution requires fewer precomputed features, it also allows for shorter processing times.

See https://github.com/sznov/joint-localization for additional preprocessing scripts.

Link to processed dataset used in paper: https://drive.google.com/file/d/1Mvf2nO74B0vJB5xJ--bq9ba5qZllpef8/view?usp=sharing

To achieve better results, some modifications were done in this implementation:
- LayerNorm is used instead of BatchNorm
- Normal vectors are not used as additional input features
- Instead of computing dynamic neighbourhoods for EdgeConv blocks, fixed neighbourhoods are used

## Example scripts

Training:
```
CUDA_VISIBLE_DEVICES=0 python examples/keypoint_detection/main.py \
 --cfg cfgs/kpdettest/dgcnn.yaml wandb.use_wandb=False
```

Resume training:
```
CUDA_VISIBLE_DEVICES=0 python examples/keypoint_detection/main.py \
 --cfg cfgs/kpdettest/dgcnn.yaml wandb.use_wandb=False \
 --mode resume \
 --pretrained_path "log/kpdettest/kpdettest-train-dgcnn-ngpus1-seed0-XXXXXXXX-XXXXXX-XXXXXXXXXXXXXXXXXXXXXX/checkpoint/kpdettest-train-dgcnn-ngpus1-seed0-XXXXXXXX-XXXXXX-XXXXXXXXXXXXXXXXXXXXXX_ckpt_latest.pth"
```

Validation:
```
CUDA_VISIBLE_DEVICES=0 python examples/keypoint_detection/main.py \
 --cfg cfgs/kpdettest/dgcnn.yaml mode=val wandb.use_wandb=False \
 --pretrained_path log/kpdettest/kpdettest-train-dgcnn-ngpus1-seed0-XXXXXXXX-XXXXXX-XXXXXXXXXXXXXXXXXXXXXX/checkpoint/kpdettest-train-dgcnn-ngpus1-seed0-XXXXXXXX-XXXXXX-XXXXXXXXXXXXXXXXXXXXXX_ckpt_best.pth
```

Test:
```
CUDA_VISIBLE_DEVICES=0 python examples/keypoint_detection/main.py \
 --cfg cfgs/kpdettest/dgcnn.yaml mode=test wandb.use_wandb=False \
 --pretrained_path log/kpdettest/kpdettest-train-dgcnn-ngpus1-seed0-XXXXXXXX-XXXXXX-XXXXXXXXXXXXXXXXXXXXXX/checkpoint/kpdettest-train-dgcnn-ngpus1-seed0-XXXXXXXX-XXXXXX-XXXXXXXXXXXXXXXXXXXXXX_ckpt_best.pth
```

## Citation

BibTeX citation:

```
@INPROCEEDINGS{10579426,
  author={Novaković, Stefan and Risojević, Vladimir},
  booktitle={2024 Zooming Innovation in Consumer Technologies Conference (ZINC)}, 
  title={Learning Localization of Body and Finger Animation Skeleton Joints on Three-Dimensional Models of Human Bodies}, 
  year={2024},
  volume={},
  number={},
  pages={19-24},
  keywords={Solid modeling;Technological innovation;Three-dimensional displays;Shape;Biological system modeling;Fingers;Animation;character rigging;animation skeletons;pose estimation;object recognition;neural nets;machine learning},
  doi={10.1109/ZINC61849.2024.10579426}}
```
