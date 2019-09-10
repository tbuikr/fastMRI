python run_unet.py --data-path ../../Knee_fastMRI --data-split val --checkpoint ./checkpoint/best_model.pt --challenge singlecoil --out-dir /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --mask-kspace
python evaluate.py --target-path ../../Knee_fastMRI/singlecoil_val --predictions-path /media/toanhoi/Data/Knee_fastMRI/reconstructions_val --challenge singlecoil
python train_unet_l1cssim.py --challenge singlecoil --data-path ../../Knee_fastMRI/ --exp-dir checkpoint


#Baseline
MSE = 1.608e-10 +/- 3.858e-10 NMSE = 0.04453 +/- 0.05602 PSNR = 30.42 +/- 5.805 SSIM = 0.6755 +/- 0.2835 
#Baseline+ssim
MSE = 1.506e-10 +/- 3.54e-10 NMSE = 0.04342 +/- 0.05655 PSNR = 30.6 +/- 5.964 SSIM = 0.6901 +/- 0.2718
#Val baseline
