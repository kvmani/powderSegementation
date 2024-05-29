rem Window script to run Ml model Pix2PixTensorFlow 2.0 developed by amrutha
set INPUT_DIR="D:\mani\aiswarya\DATASET-UPDATED\VALIDATION"
set INPUT_DIR="C:\Users\vk0237\mldata\powderQuantifcation\validationData_fresh imagesPowderedData256X256"

rem CHECKPOINT_DIR="D:\mani\amrutha\mlOutputs\1KData-200Epochs\logs\1KData-200Epochs\L2-100-lr_1e-5_Ld_1e-5_20230529-075521"
rem CHECKPOINT_DIR="D:\mani\amrutha\mlOutputs\1KData-200Epochs\logs\1KData-200Epochs\GAN-100-lr_1e-6_Ld_1e-5_20230528-034911"
rem CHECKPOINT_DIR="D:\mani\amrutha\mlOutputs\grainBoundaryLearning2023\logs\fit\grainBoundaryLearning2023\l2_100_GB_Data_v1_20230419-084142"
set CHECKPOINT_DIR="D:\numerical-Img-Dr.Mani\SurrogateMeltPool_DL-master\Datasets\MeltPool_LR1e-4_GAN_1_L1_100_20230725-120135\MeltPool_LR1e-4_GAN_1_L1_100_20230725-120135"
set CHECKPOINT_DIR="D:\mani\amrutha\mlOutputs\MeltPoolSim\logs\MeltPoolSim\MeltPool_LR1e-4_GAN_1_L1_100_20230713-093018"
set CHECKPOINT_DIR="D:\mani\amrutha\mlOutputs\MeltPoolSim\logs\MeltPoolSim\MeltPool_LR1e-4_GAN_1_L1_100_20230725-120135"
set CHECKPOINT_DIR="C:\Users\vk0237\mloutputs\PowderWork\logs\PowderWork\FilledPowderData_LR1e-4_DLR_1e-3_GAN_1_L2_100.0_L1_0_20240123-123817"
rem set CHECKPOINT_DIR="D:\mani\amrutha\mlOutputs\30KData\logs\30KData\Lr_1e_4_20230516-100812"
set CODEDEVELOPMENTMODE=False
set MODE=inference

echo pix2pixTensorFlow2.0.py  --mode "inference" --input_dir %INPUT_DIR%    ^
      --checkpoint %CHECKPOINT_DIR%

python pix2pixTensorFlow2.0.py  --mode "inference" --input_dir %INPUT_DIR%    ^
      --checkpoint %CHECKPOINT_DIR%