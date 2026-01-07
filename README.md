
# 3D UVC FlowGAN: 3D Virtual H&E Staining with spatially consistent flow loss

<p align="center">
### Original BIT 3D Tomographic Stack (LEFT) and 3D Virtually Stained H&E (RIGHT)
![BIT_epoch0200_downsample4x_crypts_Duodenum_IM0012_cropped_BIT_40X_frame29-55_crop2](https://github.com/user-attachments/assets/4ce83eaf-b8b2-43b0-8dec-cf12252ea2c8) ![vHE_epoch0200_downsample4x_crypts_Duodenum_IM0012_cropped_BIT_40X_frame29-55_crop2](https://github.com/user-attachments/assets/95739260-1845-47b8-ac83-529f14f7abb5)
</p>

Full Manuscript in Progress. 

This package was inspired by: 
Cycle-Consistent GAN for Unpaired Image-to-Image Translation` applied to Virtual H&E Staining of Back Illumination Interference Tomography Images of Raw Tissue
Original Paper: [paper][uvcgan2_paper].

Building off work from these Proceedings: https://opg.optica.org/abstract.cfm?URI=NTM-2025-NTh1C.3 , PDF available in main: ntm-2025-nth1c.3.pdf


`uvcgan2` builds upon the CycleGAN method for unpaired image-to-image transfer
and improves its performance by modifying the generator, discriminator, and the
training procedure.

This README file provides brief instructions about how to set up the `uvcgan2`
package and reproduce the paper results. To further facilitate the
reproducibility we share the pre-trained models
(c.f. section Pre-trained models)

## Applying UVCGANv2 to Your Dataset
In short, the procedure to adapt the `uvcgan2` to your problem is as follows:

1. Arrange your dataset to the format:

```bash
    MUSE-BIT/          # Name of the dataset
        trainA/
        testA/
    FFPE-HE/          # Name of the dataset
        trainB/
        testB/

```

2. Next, take an existing training script as a starting point.
   For instance, this one should work
```
scripts/.../pretrain.py
scripts/.../train.py
```

# Installation & Requirements

## Requirements

`uvcgan2` models were trained under the official `pytorch` container
`pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`. A similar training
environment can be constructed with `conda`
```
conda env create -f contrib/conda_env.yaml
```

The created conda environment can be activated with
```bash
conda activate uvcgan2
```

## Installation

To install the `uvcgan2` package one can simply run the following command
```
python3 setup.py develop --user
```
from the `uvcgan2` source tree.

## Environment Setup

By default, `uvcgan2` will try to read datasets from the `./data` directory
and will save trained models under the `./outdir` directory. If you would
like to change this default behavior, set the two environment variables
`UVCGAN2_DATA` and `UVCGAN2_OUTDIR` to the desired paths.

For instance, on UNIX-like system (Linux, MacOS) these variables can be
set with:

```bash
export UVCGAN2_DATA=PATH_WHERE_DATA_IS_SAVED
export UVCGAN2_OUTDIR=PATH_TO_SAVE_MODELS_TO
```

