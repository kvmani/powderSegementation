# Powder Segementation

A Deep learning code based on Pix2Pix archetecture for image translation for achieving superior powder particle segmentation.

This work is in continuation of our previous work on 
1. ["Kikuchi patterns refinements (or denoising)"](https://doi.org/10.1016/j.ultramic.2023.113703)
2. and [ grain segmentation from back scatter image ](https://doi.org/10.1016/j.matchar.2023.113396)

## Data:
we provide a subset of training data (20 images out of 2000 used in our paper) as part of this code base so that you can get started with training.
For full data set please contact the developers.
The code requires the data folder to have three sub folders viz., train, test and val. All images are expcted to be in .png format of 256X512 size see the provided examples.
Curetnly, test and val data can be put same as there is no separate validation for the current task. However, Code requires presence of both test and val folders for running. 

## Code development mode:
One can run the main trining script in pix2pixTensorFlow2.0.py in code development mode. For this set the falg codeDevelopmentMode to True. Default is False.
In this mode a very small train ing run of few steps is run with frequent dumps suitable for quick debugging.

## Training:
The training can be perfomed by employing the utility script runPowderTraining.bat where in multiple jobs with different
hyper parameters (weights to different lossess for example) can be set. The script provides several useful parameters
such as set OUTPUT_GROUP_DIR, FILENAMEPREFIX, and COMMENT which help in giving descriptive message and names to the outputs generated 
in training for easier examination in tensorboard.

## Inference:
We provide another script file runPowderInference.bat for easy inferene on unseen data. 

## Citing:
If you find the work shown here useful please cite our works :
@article{krishna2023machine,
  title={Machine learning based de-noising of electron back scatter patterns of various crystallographic metallic materials fabricated using laser directed energy deposition},
  author={Krishna, KV Mani and Madhavan, R and Pantawane, Mangesh V and Banerjee, Rajarshi and Dahotre, Narendra B},
  journal={Ultramicroscopy},
  volume={247},
  pages={113703},
  year={2023},
  publisher={North-Holland}
}

@article{anantatamukala2023generative,
  title={Generative adversarial networks assisted machine learning based automated quantification of grain size from scanning electron microscope back scatter images},
  author={Anantatamukala, A and Krishna, KV Mani and Dahotre, Narendra B},
  journal={Materials Characterization},
  volume={206},
  pages={113396},
  year={2023},
  publisher={Elsevier}
}
