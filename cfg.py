##### configuration file to store all global variables for usage across the entire project. it needs to be imported
## in all of the scripts whereever access to these variables is needed.
import platform, sys, os
import warnings

import numpy as np
degree = np.pi/180
### detecting the system. for each specific system appropriate set of folowing paths may be defined. this is to ensure that
### paths used in the rest of scripts are independent of the system on which code is running.

if "DESKTOP-DE0M9JB" in platform.node(): #### mani office computer
    rootDir = r"D:\CurrentProjects\ml_microStructureQuantification\imageData\grainBoundaryLearning"
    kikuchiDataRootFolder = r"D:\CurrentProjects\ml_microStructureQuantification\kikuchiProject\realKikuchiData\kikuchiTraining_June_2021\stainLessSteel_1x1binning"
elif "atulya" in platform.node(): ### atulya server of BARC
    rootDir = r"/home/lus03/kvmani/machineLearning/data/grainBoundaryLearning/data4.0/Pix2Pix_2/train"
    #kikuchiDataRootFolder = r"D:\CurrentProjects\ml_microStructureQuantification\kikuchiProject\realKikuchiData\kikuchiTraining_June_2021\stainLessSteel_1x1binning",
elif "ManiCodingLaptop" in platform.node(): ### mani coding laptop at home
    rootDir = r"datasets/facades/train"
    kikuchiDataRootDir= r'C:\Users\ManiKrishna\OneDrive - UNT System\UNT_work\kikuchiProject'
    kikuchiDataRootFolder = r"D:\mani\kikuchiProject\realKikuchiData\kikuchiTraining_June_2021\stainLessSteel_1x1binning"
    pyCrystallographyPath=r'C:\Users\ManiKrishna\PycharmProjects\pycrystallographyUSA\pycrystallography'
    outPath = r'D:\tmp\mlOutputs'
    mlOutputRootDir=r'D:\mani\mlOutputs'
elif "PCJMFBT13-CAAM" in platform.node(): ### CAAM 1st computer on which work started
    dataRootDir = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject'
    kikuchiDataRootFolder = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject'
    kikuchiPutputrootFolder = r'D:\mani\mlOutputs\kikuchiMlOutputs'
    outPath = r'D:\mani\mlOutputs\tmp'
    pyCrystallographyPath = r'C:\Users\vk0237\PycharmProjects\pycrystallography\pycrystallography'
    dataRootDir = r'D:\mani\mlData' #### for tensorflow inputs
    mlOutputRootDir = r'D:\mani\mlOutputs'
elif "PCFWDKXP3-CAAM" in platform.node(): ### CAAM GPU server bought in
    dataRootDir = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject'
    kikuchiDataRootFolder = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject'
    kikuchiPutputrootFolder = r'D:\mani\mlOutputs\kikuchiMlOutputs'
    grainBoundaryWorkCodeGeneratedDataPath = r"D:\mani\amrutha"
    outPath = r'D:\mani\mlOutputs\tmp'
    kikuchiOutPath = r'D:\tmp\mlOutputs'
    mlOutputRootDir = r"D:\mani\amrutha\mlOutputs"
    mlCodeDevptInputPath=r"D:\Amrutha\CodeDevelopmentData"
    Set_2CodeGeneratedDataPath = r'D:\Amrutha\ML Data\GrainBoundaryWork\codeGeneratedData'
    PriasOutputROOtDir=r'D:\Amrutha\ML Data\PraisImages\CodeGeneratedData'


elif "PCJMF9T13-CAAM" in platform.node(): ### CAAAM Amrutha's PC
    dataRootDir = r'D:\Amrutha\ML Data\GrainBoundaryWork'
    kikuchiDataRootFolder = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject'
    kikuchiPutputrootFolder = r'D:\mani\mlOutputs\kikuchiMlOutputs'
    grainBoundaryWorkCodeGeneratedDataPath = r"D:\Amrutha\ML Data\GrainBoundaryWork\codeGeneratedData"
    semDenoisingyWorkCodeGeneratedDataPath = r"D:\Amrutha\ML Data\SEM-denoising\codeGeneratedData"
    AlNiCoWorkCodeGeneratedDataPath=r"D:\Amrutha\ML Data\AlNiCo\codeGeneratedData"
    NoiseLevelsCodeGeneratedDataPath=r"D:\Amrutha\ML Data\NoiseLevelImages\codeGeneratedData"
    Set_1CodeGeneratedDataPath = r"D:\Amrutha\ML Data\Set_1images"
    Set_2CodeGeneratedDataPath = r"D:\Amrutha\ML Data\Set_2images\CodeGeneratedData"
    outPath = r'D:\mani\mlOutputs\tmp'
    kikuchiOutPath = r'D:\tmp\mlOutputs'
    mlOutputRootDir = r'D:\Amrutha\ML Data\MLOutPuts'
    mlOutputRootDir = r'D:\mani\mlOutputs'
    PriasOutputROOtDir=r'D:\Amrutha\ML Data\PraisImages\CodeGeneratedData'

elif "PC7YSV2Q3-CAAM" in platform.node(): #### 1.CAAAM GPU server bought in (Aishwarya)
    dataRootDir = r'C:\Users\am2195\Desktop\num2img\machineLearning\data\trialNpzData'
    mlOutputRootDir = r'C:\Users\am2195\Desktop\num2img\machineLearning\data\trialNpzData\MeltPoolSim'

elif "PC5L3L853-CAAM" in platform.node(): #### 2.CAAAM GPU server bought in (Aishwarya)
    dataRootDir = r'C:\Users\am2195\Desktop\num2img\Data\SS 316L'
    mlOutputRootDir = r'C:\Users\am2195\Desktop\num2img\Data\SS 316L\MeltPoolSim'
    mlCodeDevptInputPath = r'C:\Users\am2195\Desktop\num2img\Data\SS 316L'

elif "CENG-47JSG04" in platform.node(): #### 3.CAAAM GPU server bought in (Aishwarya) jan, 2024
    dataRootDir = r'D:\Aishwarya\temperatureProfiles\data\Cu_revised\splittedData'
    mlOutputRootDir = r'D:\mlOutputs'
    mlCodeDevptInputPath = r'D:\mlOutputs\devMLouput'

elif "CENG-FWDKXP3" in platform.node(): #### Mani GPU PC after windows upgrade to windows 11
    dataRootDir = r'C:\Users\am2195\Desktop\num2img\Data\SS 316L'
    mlOutputRootDir = r'C:\Users\vk0237\mloutputs'
    mlCodeDevptInputPath = r'C:\Users\vk0237\mloutputs\codeDepvt'


else:
    dataRootDir = r"data\powder"
    mlOutputRootDir = r'tmp'
    warnings.warn(f"Unknown host system !!!! {platform.node()}. Probably you are running on a new system/host. "
                     f"resorting to defaults!!!")


print(f"Running on the system : {platform.node()}")
print(f"The root directory is set to be {dataRootDir}")

if "grainBoundaryWorkCodeGeneratedDataPath" in locals():
    print(f"The code generated data for grain boundary work is set to be wriiten to folder \n {grainBoundaryWorkCodeGeneratedDataPath}")
else:
    print(f"The host {platform.node()} was not set with the variable 'grainBoundaryWorkCodeGeneratedDataPath'")

if "kikuchiDataRootDir" in locals():
    print(f"The root directory is set to be {kikuchiDataRootDir}")
else:
    print(f"The host {platform.node()} was not set with the variable 'kikuchiDataRootDir'")

augmentorOutputFolder = os.path.join(mlOutputRootDir,'augmentorOutput')
print("Done with the configuration")