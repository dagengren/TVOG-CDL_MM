This project contains: (1) the code of TVOG+CDL; (2) the datasets of I2OVOG and R2MVOG.

# Data

Domain video files

The **I2OVOG** dataset is the indoor-to-outdoor domain pair. It is developed by splitting the VidSTG dataset into the indoor part and the outdoor part. The manual division results are in these files.

The **R2MVOG** is the real-to-movie domain pair dataset. It is developed by combining the VidSTG ( real life dataset ) and the HC-STVG ( movie dataset ).

The VidSTG dataset can download from https://github.com/Guaranteer/VidSTG-Dataset.

The HC-STVG dataset download from https://github.com/tzhhhh123/HC-STVG.



# Training
```python
python -m torch.distributed.launch --nproc_per_node=8 --user_env main.py --load .../pretrained_resnet101_checkpoint.pth --ema --no_contrastive_align_loss 
```
