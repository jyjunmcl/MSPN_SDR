# (CVPR 2024) Masked Spatial Propagation Network for Sparsity-Adaptive Depth Refinement
### Jinyoung Jun, Jae-Han Lee, and Chang-Su Kim

Official pytorch implementation for **"Masked Spatial Propagation Network for Sparsity-Adaptive Depth Refinement"** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Jun_Masked_Spatial_Propagation_Network_for_Sparsity-Adaptive_Depth_Refinement_CVPR_2024_paper.pdf).

![intro](https://github.com/jyjunmcl/MSPN_SDR/assets/112459638/7a04e3e8-9dd8-4979-8479-fd8e9e8b78a7)

### Dependecies
Ubuntu 22.04, PyTorch 1.10.1, CUDA 11.3, Python 3.8, [Natten](https://shi-labs.com/natten/) 0.14.6

### Instructions
1. [Download](https://drive.google.com/drive/folders/17npEM9PfxydZz9AcSlr_mVrGFCbY_M2X?usp=drive_link) the pretrained models and monocualr depth estimation results.
2. Move trained models to ```test_models``` folder and move ``pretrained.zip`` to ``lib`` folder and unzip
3. For NYUv2, move ``nyudepthv2_SDR.zip`` file to ``dataset`` folder and unzip the file.
4. For KITTI, download the [KITTI DC](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) and [KITTI RAW](https://www.cvlibs.net/datasets/kitti/raw_data.php) datasets, and move ``data_depth_velodyne_NewCRFs.zip`` file to directory as follows:
```
├── kitti_depth
|   ├──data_depth_annotated
|   |  ├── train
|   |  ├── val
|   ├── data_depth_velodyne
|   |  ├── train
|   |  ├── val
|   ├── data_depth_velodyne_NewCRFs
|   |  ├── val
|   ├── data_depth_selection
|   |  ├── test_depth_completion_anonymous
|   |  |── test_depth_prediction_anonymous
|   |  ├── val_selection_cropped
|   ├── kitti_raw
|   |   ├── 2011_09_26
|   |   ├── 2011_09_28
|   |   ├── 2011_09_29
|   |   ├── 2011_09_30
|   |   ├── 2011_10_03
```

5. Run with
```
cd root
python test_NYU.py
python test_kittidc.py
```
6. Trained models on conventional depth completion will be released soon.

### Acknowledgement
Thanks for the authors of [CompletionFormer](https://github.com/youmi-zym/CompletionFormer), opening the source of their work.

### Reference
```
@inproceedings{jun2024masked,
  title={Masked Spatial Propagation Network for Sparsity-Adaptive Depth Refinement},
  author={Jun, Jinyoung and Lee, Jae-Han and Kim, Chang-Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19768--19778},
  year={2024}
}
```

