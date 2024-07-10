# AnimJointDGCNN – Learning Localization of Body and Finger Animation Skeleton Joints on Three-Dimensional Models of Human Bodies

Fork of PointMetaBase (https://github.com/linhaojia13/PointMetaBase).

Implementation of the network described in https://ieeexplore.ieee.org/document/10579426 (DOI: 10.1109/ZINC61849.2024.10579426).

To achieve better results, some modifications were done in this implementation:
- LayerNorm is used instead of BatchNorm
- Normal vectors are not used as additional input features
- Instead of computing dynamic neighbourhoods for EdgeConv blocks, fixed neighbourhoods are used

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