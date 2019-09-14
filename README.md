## Run training ESSIMTV + Aug + Dilation + Resnet +128
```
python train_unet_essimtv_aug_scSE.py --challenge singlecoil --data-path /media/toanhoi/88f64337-6c11-4924-a165-ca6fefb38002/home/toanhoi/KneeData --exp-dir checkpoint --netG unet_upsampling_dilation --batch-size 16 --num-chans 128

```
## Run evaluate ESSIMTV + Aug + Dilation + Resnet +128 without TTA
```
python run_unet_transpose.py --data-path ../../Knee_fastMRI --data-split val --checkpoint ./checkpoint/model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --mask-kspace --batch-size 16 --netG unet_upsampling_dilation 
```
```
 python evaluate.py --target-path ../../Knee_fastMRI/singlecoil_val --predictions-path /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --challenge singlecoil

```

## Run evaluate ESSIMTV + Aug + Dilation + Resnet +128 without TTA

```
python run_unet_transpose.py --data-path ../../Knee_fastMRI --data-split val --checkpoint ./checkpoint/model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --mask-kspace --batch-size 16 --netG unet_upsampling_dilation  --tta 1
```
```
 python evaluate.py --target-path ../../Knee_fastMRI/singlecoil_val --predictions-path /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --challenge singlecoil

```

## -----------------------------------Old --------------------------------


## Run ESSIMTV + AUG + Attention
```
python train_unet_essimtv_aug_scSE.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint --netG unet_upsampling_scSE --batch-size 16 --num-chans 128
```

## Run L1+SSIm+TV+newnet+res+aug
```
CUDA_VISIBLE_DEVICES=0 python train_unet_l1cssimtv_unet_transpose.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint --aug True --batch_size 16 --netG unet_transpose_res
```
## Run L1+ssim+tv+newunet+aug
```
CUDA_VISIBLE_DEVICES=0 python train_unet_l1cssimtv_unet_transpose.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint --aug True --batch_size 16
```

## Run L1+ssim+tv+newunet
```
python train_unet_l1cssimtv_unet_transpose.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint
```

## Run evaluation on new network
```
python run_unet_transpose.py --data-path ../../Knee_fastMRI --data-split val --checkpoint ./checkpoint/best_model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --mask-kspace --batch-size 16 --netG unet_transpose
```

```
python evaluate.py --target-path ../../Knee_fastMRI/singlecoil_val --predictions-path /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --challenge singlecoil
```
## Run test on new network

```
python run_unet_transpose.py --data-path ../../Knee_fastMRI/ --data-split test --checkpoint ./checkpoint/best_model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_test --batch-size 16 --netG unet_transpose
```

## Run with modified unet using up-sampling
```
python train_unet_l1cssimtv_unet_transpose.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint --netG unet_upsampling
```
## =============BASELINE=========

## Run train l1+ssim
```
python train_unet_l1cssim.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint
```
## RUn train l1+ssim+cGAN +resume G
```
python train_unet_l1cssim_dis.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint --resume --checkpoint ./checkpoint/best_model.pt
```
## RUn train l1+ssim+cGAN +noresume G
```
python train_unet_l1cssim_dis.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint
```

## Run test
```
python run_unet.py --data-path ../../Knee_fastMRI/ --data-split test --checkpoint ./checkpoint/best_model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_test
```
## Run val
```
python run_unet.py --data-path ../../Knee_fastMRI --data-split val --checkpoint ./checkpoint/best_model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --mask-kspace
```
```
python evaluate.py --target-path ../../Knee_fastMRI/singlecoil_val --predictions-path /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --challenge singlecoil
```

## ===================Val baseline=======================
### Baseline
MSE = 1.608e-10 +/- 3.858e-10 NMSE = 0.04453 +/- 0.05602 PSNR = 30.42 +/- 5.805 SSIM = 0.6755 +/- 0.2835 
### Baseline+ssim
MSE = 1.506e-10 +/- 3.54e-10 NMSE = 0.04342 +/- 0.05655 PSNR = 30.6 +/- 5.964 SSIM = 0.6901 +/- 0.2718
