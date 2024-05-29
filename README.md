# Powder Segementation

A Deep learning code based on Pix2Pix archetecture for image translation for achieving superior powder particle segmentation.

This work is in continuation of our previous work on 
1. ["Kikuchi patterns refinements (or denoising)"](https://doi.org/10.1016/j.ultramic.2023.113703)
2. and [ grain segmentation from back scatter image ](https://doi.org/10.1016/j.matchar.2023.113396)

## Data:
we provide a subset of training data (20 images out of 2000 used in our paper) as part of this code base so that you can get started with training.
For full data set, please contact the developers.
The code requires the data folder to have three sub folders viz., train, test and val. 
All images are expected to be in .png format of 256X512 size, see the provided examples in data folder.
Currently, test and val data can be put same as there is no separate validation for the current task. 
However, code requires presence of both test and val folders for running. 

## Code development mode:
One can run the main training script pix2pixTensorFlow2.0.py in code development mode. 
For this, set the flag codeDevelopmentMode to True. Default is False.
In this mode a very small train ing run of few steps is run with frequent dumps suitable for quick debugging.

## Training:
The training can be performed by employing the utility script runPowderTraining.bat where in multiple jobs with different
hyperparameters (weights to different lossess for example) can be set. The script provides several useful parameters
such as set OUTPUT_GROUP_DIR, FILENAMEPREFIX, and COMMENT which help in giving descriptive message and names to the outputs generated 
in training for easier examination in tensorboard.

## Inference:
We provide another script file runPowderInference.bat for easy inference on unseen data. 

## Citing:
If you find the work shown here useful please cite our works :

## References

1. Krishna, K.V. Mani, Madhavan, R., Pantawane, Mangesh V., Banerjee, Rajarshi, & Dahotre, Narendra B. (2023). "Machine learning based de-noising of electron back scatter patterns of various crystallographic metallic materials fabricated using laser directed energy deposition". *Ultramicroscopy*, 247, 113703. [https://doi.org/10.1016/j.ultramic.2023.113703](https://doi.org/10.1016/j.ultramic.2023.113703)
2. Anantatamukala, A., Krishna, K.V. Mani, & Dahotre, Narendra B. (2023). "Generative adversarial networks assisted machine learning based automated quantification of grain size from scanning electron microscope back scatter images". *Materials Characterization*, 206, 113396. [https://doi.org/10.1016/j.matchar.2023.113396](https://doi.org/10.1016/j.matchar.2023.113396)
3. Krishna, K.V. Mani, Anantatamukala, A.,  & Dahotre, Narendra B. (2024). "Deep Learning Based Automated Quantification of Powders used in Additive Manufacturing". *Additive Manufacturing Letters*, Under review.

