rem Window script to run Ml model Pix2PixTensorFlow 2.0 developed by amrutha
set INPUT_DIR="C:\Users\kvman\PycharmProjects\powderSegementation\data\powder"
set CODEDEVELOPMENTMODE=False
set MODE=train
set IMDATASETTYPE=grainBoundary
set SUMMARY_FREQ=500
set SEED=10
set MAX_STEPS=2000000
set MAX_EPOCHS=50
set LOG_HIST_WEIGHTS_FREQUENCY=20
set PROGRESS_FREQ=100
set TRACE_FREQ=100
set DISPLAY_FREQ=250
set SAVE_FREQ=50
set LR=1e-4
set LRDESCRIMINATOR=1e-4
set IMTYPE=.png
set BETA1=0.99
set BATCH_SIZE=1
set L1_WEIGHT=0.0
set GAN_WEIGHT=1.0
set L2_WEIGHT=100.0
set OUTPUT_GROUP_DIR=PowderWork
set FILENAMEPREFIX=FilledPowderData_LR1e-4_DLR_1e-3_GAN_1_L2_100.0_L1_0
set COMMENT="Using Filled powder data"


echo  pix2pixTensorFlow2.0.py --input_dir %INPUT_DIR% --mode %MODE% --seed %SEED% ^
                        --codeDevelopmentMode %CODEDEVELOPMENTMODE% --imtype %IMTYPE% --max_steps %MAX_STEPS% --max_epochs %MAX_EPOCHS% ^
                               --summary_freq %SUMMARY_FREQ% --log_hist_weights_frequency %LOG_HIST_WEIGHTS_FREQUENCY% ^
                                --imDataSetType %IMDATASETTYPE% --progress_freq %PROGRESS_FREQ% --trace_freq %TRACE_FREQ% ^
                                --display_freq %DISPLAY_FREQ% --save_freq %SAVE_FREQ% ^
                                --lr %LR% --lrDescriminator %LRDESCRIMINATOR% ^
                              --l1_weight %L1_WEIGHT% --gan_weight %GAN_WEIGHT% --l2_weight %L2_WEIGHT% ^
                               --output_group_dir %OUTPUT_GROUP_DIR% --fileNamePrefix %FILENAMEPREFIX% ^
                               --comment %COMMENT% --beta1 %BETA1% --batch_size %BATCH_SIZE%

python pix2pixTensorFlow2.0.py --input_dir %INPUT_DIR% --mode %MODE% --seed %SEED% ^
                       --codeDevelopmentMode %CODEDEVELOPMENTMODE% --max_steps %MAX_STEPS% --max_epochs %MAX_EPOCHS% ^
                              --summary_freq %SUMMARY_FREQ% --log_hist_weights_frequency %LOG_HIST_WEIGHTS_FREQUENCY% ^
                               --imDataSetType %IMDATASETTYPE% --progress_freq %PROGRESS_FREQ% --trace_freq %TRACE_FREQ% ^
                               --display_freq %DISPLAY_FREQ% --save_freq %SAVE_FREQ% ^
                               --lr %LR% --lrDescriminator %LRDESCRIMINATOR% ^
                             --l1_weight %L1_WEIGHT% --gan_weight %GAN_WEIGHT% --l2_weight %L2_WEIGHT% ^
                              --output_group_dir %OUTPUT_GROUP_DIR% --fileNamePrefix %FILENAMEPREFIX% ^
                              --comment %COMMENT% --beta1 %BETA1% --batch_size %BATCH_SIZE%

