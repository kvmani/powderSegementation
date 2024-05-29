#### all utilitiy functions to be used in the ml code are to be here !!!
import glob
import json
import os
import warnings
from os.path import getsize
import PIL
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects
import pydicom
from scipy.ndimage import label, generate_binary_structure
from PIL import Image
import struct
import logging
from itertools import product
from scipy.ndimage import zoom
from scipy.stats import zscore

RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2

settings = {"COLOR_UPPER_THRESHOLD": 100,
            "COLOR_LOWER_THRESHOLD": 10,
            "dilationOn": True,
            "dilationSize": (5, 5),
            "imageFormats": [".png", ],  # "*.jpg", "*.png", "*.tif", "*.gif", ".bmp"],
            "excludePatterns": ["/*skelton*", "/*segMask*"],
            "generateLabeledImages": True,
            "randomColrs": True,  ### for genrating unique grain color map relavent when generateLabeledImages is True
            "numberOfUniqueColors": 20,
            "generateSemanticSegImages": True,
            }


def serialize_kikuchi_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()


def generateUniqueGrainMap(image, thinBoundaries=False):
    """
    image : numpy array or PIL image from which the grain are identified and colored useful for quantifcation tasks
    """
    # if len(image.shape)>2:
    #     greenBoundaries = image[:,:,GREEN_CHANNEL] > settings["COLOR_UPPER_THRESHOLD"]
    #     redBoundaries = image[:,:,RED_CHANNEL] > settings["COLOR_UPPER_THRESHOLD"]
    #     totalBoundariees = np.ma.mask_or(greenBoundaries, redBoundaries)
    # else: ## gray scale image
    #     totalBoundariees = image>settings["COLOR_UPPER_THRESHOLD"] ### assuming the boundaries are white !!!

    convertBackToPILImage = False
    if isinstance(image, PIL.Image.Image):
        image = np.array(image.convert('L'))
        convertBackToPILImage = True
    # tmp = (image<settings["COLOR_LOWER_THRESHOLD"]).astype(np.uint8)
    boundaries = image < settings["COLOR_LOWER_THRESHOLD"]
    kernel = np.ones((3, 3), np.uint8)
    boundaries = cv2.dilate(boundaries.astype(np.uint8), kernel)
    grains, number_of_grains = ndimage.label(boundaries)
    tmp = np.zeros_like(grains)
    if settings["randomColrs"]:
        tmp = np.dstack((tmp, tmp, tmp))
        uniqueColors = np.random.randint(255, size=number_of_grains * 3).reshape((number_of_grains, 3)).tolist()
        for i, color in enumerate(uniqueColors):
            objectPixcels = grains == i
            tmp[objectPixcels] = color

    # if convertBackToPILImage:
    #     tmp = PIL.Image.fromarray(tmp)
    return tmp


def ebsdNeighborPixcelIndiccesForIJ(queryIndices, ebsdMapShape, dataMode="TSL"):
    """

    :param queryIndices: tuple of indices of pixcel (i,j) for which negbors are sought
    :param ebsdMapShape: tuple : ebsd map shappe : (nRows,nColumns)
    :param dataMode : optional one of TSL and HKL default is TSL
    :return: list of 6 immediate neighbors of point (i,j)
    """
    i, j = queryIndices
    nRows, nColumns = ebsdMapShape
    if "TSL" in dataMode:
        neighbors = [[i + 1, j], [i + 1, j + 1], [i, j - 1], [i, j + 1], [i - 1, j], [i - 1, j + 1]]
    else:
        raise NotImplementedError(
            "At the moment only TSL hexagonal grid is implemented. For hkl please edit bove line of the code!!")
    toBeDeleted = []
    for k, item in enumerate(neighbors):
        if item[0] < 0 or item[0] >= nRows or item[1] < 0 or item[
            1] >= nColumns:  ## checking if the neighbors are within in the map bounds
            toBeDeleted.append(k)
    for item in sorted(toBeDeleted, reverse=True):
        del neighbors[item]
    if len(neighbors) < 6:
        warnings.warn(f"Found less than only {len(neighbors)} number of neighbors for the point {queryIndices}"
                      f"ebsdDimensions are : {ebsdMapShape}:"
                      f" neigbors are : {neighbors}")
    return neighbors


def rgb2label(img):
    """
    converts input image or array which has 3 color channels into labelled image suitable for semanticn segmantion

    :return:
    """

    labeledImage = np.array(Image.fromarray(img).convert('1'))

    return labeledImage


def plotComparitiveImages(images, titles=[], mainTitle="", figLayoutChoice=-1,
                          shouldBlock=True, filePath="", cmap='gray', figSize=None):
    """
    utility method for plotting the images side by side for easy comparision:
    images : list of numpy arrays (2D arrays/3d arrays to be plotted as images)
    titles : list of caption for each image.
    mainTitle main title of the figure
    if shouldBlock is set to False, figure will be drwan in non blocking mode so that automation/animation can be achieved
    """
    if figSize is not None:
        fig = plt.figure(figsize=figSize)
    else:
        fig = plt.figure()

    fig = plt.figure()
    fig.suptitle(mainTitle)
    numOfImages = len(images)
    figLayout = [[(1, 1, 1)],
                 [(1, 2, 1), (1, 2, 2)],
                 [(1, 3, 1), (1, 3, 2), (1, 3, 3)],
                 [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)],
                 [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)],  ## for 5 images
                 [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)],  ## for 6 images
                 [(3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7), (3, 3, 8), (3, 3, 9)],
                 ## for 7 images
                 [(3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7), (3, 3, 8), (3, 3, 9)],
                 ## for 8 images
                 [(3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7), (3, 3, 8), (3, 3, 9)],
                 ## for 9 images
                 [(3, 3, 1), (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7), (3, 3, 9)],
                 ## special case for showing the kikuchi of neighbors

                 ]

    numImages = len(images)
    ax = []
    if len(titles) == 0:
        titles = ["image_" + str(i) for i in range(numOfImages)]
    for i, image in enumerate(images):
        if figLayoutChoice < 0:  ### automatic depending on number of images
            ax.append(fig.add_subplot(*figLayout[numImages - 1][i]))
        else:
            ax.append(
                fig.add_subplot(*figLayout[figLayoutChoice - 1][i]))  # figLayoutChoice = 10 special case for kikuchi
        ax[i].set_title(insert_newline_at_middle(titles[i]))
        ax[i].imshow(image, cmap=cmap)
        ax[i].axis('off')

    if len(filePath) > 0:
        ### this means that path is provided and we should save the file in that path.

        try:
            plt.gcf().set_size_inches(figSize)
            plt.savefig(filePath, dpi=100)
            # shouldBlock=False

            logging.debug(f"plotComparitiveImages: saved the plot to {filePath}")
            if shouldBlock:
                plt.show(block=shouldBlock)
            plt.close()
            return

        except:
            logging.error(f"plotComparitiveImages: unable to save the plot to {filePath}")

    plt.show(block=shouldBlock)


def segMask2RgbImage(inputImage):
    """
    inputImage : numpy Array or PIL image object
    """
    numberOfColors = 3
    colorMap = [
        [0, 0, 0],  ##  0 : background : black
        [0, 255, 0],  ## 1 : good boundaries green
        [255, 0, 0],  ## red : bad boundaries : red
    ]
    colorImage = np.dstack([inputImage] * 3).astype(inputImage.dtype)

    for i in range(numberOfColors):
        if i > 0:
            colorMask = np.where(inputImage == i)
            colorImage[colorMask, :0] = colorMap[i][0]
            colorImage[colorMask, :1] = colorMap[i][1]
            colorImage[colorMask, :2] = colorMap[i][2]
    return colorImage


def correctSegmentationMask(segMask, desiredBoundaryThickness=3, debugOn=False):
    """
    Useful for correcting the images for Semantic Segmentation.
    method for correcting the segMask image. Due to augmentation operations (such as rotation, distortion etc) image interpolations
    lead to some pixcels esp at boundaries of the object to have non ideal values. This function corrects them back. Note that this is written keeping
    boundary learning images in mind and may not work for other type of image contexts.
    segMask : numpyarray representing the segMask with each pixcel value equal to the class value
    """

    result = ndimage.maximum_filter(segMask, size=7)
    numberOfClasses = result.max() + 1
    tmp = np.zeros_like(result)
    tmpMasks = []
    finalMask = result > 500  ### we kno value will never be 500 hence it all false matrix to start with
    for i in range(
            numberOfClasses - 1):  ## we dont need to process the background i.e 0 values hene loop till numberOfClasses-1
        boundaryMask = result == i + 1
        finalMask = np.ma.mask_or(boundaryMask, finalMask)

    finalMask = ndimage.maximum_filter(skeletonize(finalMask), desiredBoundaryThickness)
    finalImage = np.where(finalMask, result, 0)

    if debugOn:
        plotComparitiveImages([segMask, result, finalImage], titles=["original", "maxFiltered", "finalImage"],
                              mainTitle="CorrectSegMask")
    return finalImage


def rgb2labelModified(image, colorThreshold=200, debugOn=False):
    """
    Useful for correcting the images genrated by Pix2Pix.
    method for correcting the boundary Images (target images) for pix2pix learning.
    Due to augmentation operations (such as rotation, distortion etc) image interpolations
    lead to some pixcels esp at boundaries of the object to hacve non ideal values. This function corrects them back. Note that this is written keeping
    boundary learning images in mind and may not work for other type of image contexts.
    boundaryImage : numpyarray representing the target with each pixcel value equal to color of the boundary
    """

    shape = image.shape
    if len(shape) == 3:  ### case of RGB image
        if shape[2] == 4:  ## alpha channel present
            logging.warning("Its ia 4 channel image I am expecting only 3 channels!!! REmoveing the last channel")
            tmp = np.zeros_like(image[:, :, 0:3])
        else:
            logging.info("Got the RGB Image!!! No issue")
            tmp = np.zeros_like(image)
    else:
        logging.info("Gray scale image!!!")
        tmp = np.zeros_like(image)

    greenBoundaries = image[:, :, GREEN_CHANNEL] > colorThreshold
    tmp[greenBoundaries, GREEN_CHANNEL] = 255
    redBoundaries = image[:, :, RED_CHANNEL] > colorThreshold
    tmp[redBoundaries, RED_CHANNEL] = 255

    labelImage = np.zeros_like(image[:, :, 0]).astype(np.uint8)
    labelImage[redBoundaries] = 1
    labelImage[greenBoundaries] = 2
    if debugOn:
        plotComparativeImages([image, greenBoundaries, redBoundaries, tmp, labelImage],
                              ['image', 'green', 'red', 'tmp', 'label image'])
    return labelImage


def correctBoundaryColorImages(boundaryImage, colorThreshold=200, skeltenizeOn=True, desiredBoundaryThickness=3,
                               debugOn=False):
    """
    Useful for correcting the images genrated by Pix2Pix.
    method for correcting the boundary Images (target images) for pix2pix learning.
    Due to augmentation operations (such as rotation, distortion etc) image interpolations
    lead to some pixcels esp at boundaries of the object to hacve non ideal values. This function corrects them back. Note that this is written keeping
    boundary learning images in mind and may not work for other type of image contexts.
    boundaryImage : numpyarray representing the target with each pixcel value equal to color of the boundary
    """

    if isinstance(boundaryImage, PIL.Image.Image):
        boundaryImage = np.array(boundaryImage)

    shape = boundaryImage.shape
    if len(shape) == 3:  ### case of RGB image
        if shape[2] == 4:  ## alpha channel present
            logging.warning("Its ia 4 channel image I am expecting only 3 channels!!! REmoveing the last channel")
            tmp = np.zeros_like(boundaryImage[:, :, 0:3])
        else:
            logging.info("Got the RGB Image!!! No issue")
            tmp = np.zeros_like(boundaryImage)
    else:
        logging.info("Gray scale image!!!")
        tmp = np.zeros_like(boundaryImage)

    greenBoundaries = boundaryImage[:, :, GREEN_CHANNEL] > colorThreshold
    tmp[greenBoundaries, GREEN_CHANNEL] = 255
    redBoundaries = boundaryImage[:, :, RED_CHANNEL] > colorThreshold
    tmp[redBoundaries, RED_CHANNEL] = 255

    tmp[:, :, 0] = ndimage.maximum_filter(tmp[:, :, 0], size=3)
    tmp[:, :, 1] = ndimage.maximum_filter(tmp[:, :, 1], size=3)
    tmp[:, :, 2] = ndimage.maximum_filter(tmp[:, :, 2], size=3)

    greenBoundaries = tmp[:, :, GREEN_CHANNEL] == 255
    redBoundaries = tmp[:, :, RED_CHANNEL] == 255
    totalBoundaries = np.logical_or(greenBoundaries, redBoundaries)
    if skeltenizeOn:
        struct2 = ndimage.generate_binary_structure(2, 2)
        totalBoundaries = ndimage.binary_dilation(skeletonize(totalBoundaries), struct2)

    mask3d = np.dstack([totalBoundaries] * 3)
    finalImage = np.where(mask3d, tmp, 0)
    mixedColors = np.sum(finalImage, axis=2) > 255
    finalImage[mixedColors, :] = [0, 255, 0]

    if debugOn:
        plotComparitiveImages([boundaryImage, tmp, finalImage], titles=["original", "tmp", "finalImage"],
                              mainTitle="correctBoundaryColorImages")
    return tmp


def makeMask2ndTypeDeepLab(mask):
    """
    Converts mask of DeepLab images into 0 and 1 type where 0 is background and 1 is grain (instead of grain boundary!!!)
    """
    mask2ndType = np.zeros_like(mask)
    # mask2ndType[mask>0]=255
    mask2ndType[mask == 0] = 1
    return mask2ndType


def boundaryDissimilarityIndex(im1, im2, threshold=50, debugOn=False):
    """
    method to compute how similar are the 2 images with grain boundaries.
    input : im1, im2 numpy arrays of MXN size (equivalent of grey scale images) which can be thresolded to to their boundaries
    """
    im1Binary = skeletonize(im1 < threshold)
    im2Binary = skeletonize(im2 < threshold)
    boundaryFractionIm1 = im1Binary.sum() / im1Binary.size
    boundaryFractionIm2 = im2Binary.sum() / im1Binary.size
    if boundaryFractionIm1 > 0.1 or boundaryFractionIm2 > 0.1:
        plotComparitiveImages(images=[im1, im2, im1Binary.astype(np.uint8), im2Binary.astype(np.uint8)],
                              mainTitle=f"Issue in the boundary images")
        logging.error("The boudaries seem to accupy more than 10% of the image See if the threshold criteria is "
                      "correct! Most likely the image need to be inverted or threshold need to be adjusted")

    meanError1 = __meanErrorIm1Im2(im1Binary, im2Binary)
    meanError2 = __meanErrorIm1Im2(im2Binary, im1Binary)
    maxError = max(meanError1, meanError2)
    logging.debug(f"The boundarySimilarityIndex = {maxError}")
    if debugOn:
        plotComparitiveImages(images=[im1, im2, im1Binary.astype(np.uint8), im2Binary.astype(np.uint8)],
                              mainTitle=f"The boundarySimilarityIndex = {maxError}")

    return maxError


def __meanErrorIm1Im2(im1Binary, im2Binary):
    refPoints = np.argwhere(im1Binary)
    compPoints = np.argwhere(im2Binary)
    totalError = 0.
    for refPoint in refPoints.tolist():
        # logging.debug(f"Ref point is :{refPoint}")
        distSquares = np.sum(np.square(compPoints - refPoint), axis=1)
        closestPointIndex = np.argmin(distSquares)
        # logging.debug(f"ref point : {refPoint}, closest point : {compPoints[closestPointIndex]}, dist = {np.sqrt(distSquares[closestPointIndex])}")
        totalError += np.sqrt(distSquares[closestPointIndex])  ##dist = np.sqrt(distSquares[closestPointIndex])
    meanError = totalError / min(refPoints.shape[0], compPoints.shape[0])
    return meanError


def insert_newline_at_middle(text, minLength=12):
    if len(text) < minLength:  ### string length is lower than the min threshold
        return text
    if "\n" in text:  ## already \n is there nothing need to be done
        return text
    # Find the approximate middle position
    middle_index = len(text) // 2

    # Look for a space or underscore near the middle position
    for i in range(middle_index - 5, middle_index + 5):
        if text[i] in [' ', '_']:
            return text[:i + 1] + '\n' + text[i + 1:]

    # If no suitable character is found, simply insert newline at the middle index
    return text[:middle_index] + '\n' + text[middle_index:]


def removeSmallIslands(im, islandSize=10, colorThreshold=10):
    """
    method to remove the small isoloated packets of regions (typically from a bouyndary image)
    """
    # colorThreshold = 100
    originalImage = im.copy()

    if len(im.shape) > 2:  ## case of RGB image
        imLocal = np.array(PIL.Image.fromarray(im).convert('L'))
    else:
        imLocal = im.copy()
    imLocal = ndimage.gaussian_filter(imLocal, sigma=1)
    mask = imLocal > colorThreshold
    # orignalMask = mask.copy()
    # mask = ndimage.binary_erosion(mask).astype(mask.dtype)
    mask = remove_small_objects(mask, islandSize, connectivity=4)
    # plotComparitiveImages([orignalMask.astype(np.uint8),mask.astype(np.uint8)],titles=["maskOri","mask"])
    finalImage = np.zeros_like(originalImage)
    if finalImage.shape[2] == 4:  ## alpha layer is present
        finalImage[:, :, 3] = originalImage[:, :, 3]
    for dim in range(im.shape[2]):
        finalImage[mask, dim] = originalImage[mask, dim]
    for dim in range(3):
        finalImage[:, :, dim] = remove_small_objects(finalImage[:, :, dim], islandSize * 2)

    plotComparitiveImages([originalImage, imLocal, mask, finalImage])
    return finalImage


def scale(x, out_range=None):
    """
    scale a givne numpy array (typically an image) to a range of values defined
    param : x numpy array
    out_range : tuple for ex: (0,1) between which the x is scaled default is None. In that case the array is scaled to limits of
    its data type i.e [0 255] for uint8 , etc.
    """

    dtype = x.dtype
    if out_range is None:

        if x.dtype == np.dtype(np.uint8):
            out_range = (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
        elif x.dtype == np.dtype(np.uint16):
            out_range = (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max)
        else:
            raise TypeError("The image data to be scaled is neither uint8 or unint16 type. If you want to scale!!!!")

    tmp = x.astype(np.double)
    domain = np.min(tmp), np.max(tmp)
    y = (tmp - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return y.astype(dtype)


def pixcelIdfromIJ(ij, mapSize):
    """
    Function to convert (i,j) notation into ID which can be used to directly acces the corresponding elment from image/ebsd data
    :param ij: tuple (i,j) for which liner index ID is sought
    :param mapSize tuple (nRows,nColumns) of the map to which i,j refer.
    :return: ID
    """
    i, j = ij
    nRows, nColumns = mapSize
    ID = i * nColumns + j  ## note that this formula is valid for TSL only. i expected nColimns to appear in the formula but
    ## the way these variables are defined it apears to be nRows !!!
    return ID


def indextoIJ(Id, mapSize):
    """
    it is inverse of the above method i.e. to convert a given linear index Id into corresponding (i,j) for a given map
    :param Id: Liner index of the pixcel in the h5 data file
    :param mapSize: tuple (nRows,nColumns) of the map to which i,j refer.
    :return: tuple (i,j)
    """
    nRows, nColumns = mapSize
    return (Id // nColumns, Id % nColumns)


def imageComparisionMse(imageA, imageB):
    """
	the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	:param imageA: np array
	:param imageB:  np array
	:return: err
	"""

    err = np.sum((scale(imageA.astype("float"), [0, 1.0]) - scale(imageB.astype("float"), [0, 1.0])) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def splitImage(im, mode='horizontal'):
    """
    used for extreacting iunput and target from the input image for Pix2Pix tasks
    :param im: image to be split (a numpy array)
    :param mode: default is horizontal
    :return: im1, im2 two splits of the images
    """
    h, w = im.shape[0], im.shape[1]
    w = w // 2
    shape = im.shape
    isGrayScale = len(shape) == 2
    if isGrayScale:
        im1 = im[:, :w]
        im2 = im[:, w:]
    else:
        im1 = im[:, :w, :]
        im2 = im[:, w:, :]

    if not im1.shape == im2.shape:
        warnings.warn(f"Split warning: the splitted images are not of equal size!! "
                      f"shape of input; out1 and out are ::{im.shape} ; {im1.shape}; {im2.shape}")

    return im1, im2


def imageSplit(img, imSize, imageName, dir_out=None):
    """

    :param img: array representation of  image
    :param imSize: Desired splitted image height(each split will be of size imSizeXimSize)
    :param imageName: Base name of the input image(used for generating output image name)
    :param dir_out: path of the folder where split images are saved
    :return: list of images
    """

    filename = imageName
    name, ext = os.path.splitext(filename)
    img = Image.fromarray(img)
    w, h = img.size
    grid = product(range(0, h - h % imSize, imSize), range(0, w - w % imSize, imSize))
    imgList = []
    for i, j in grid:
        box = (j, i, j + imSize, i + imSize)
        if dir_out is not None:
            out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
            img.crop(box).save(out)
            imgList.append(img.crop(box))

    return imgList


def stitchImages(im0, im2, mode='horizontal'):
    """
    stiching the two numpy array images and returning a combined image (for Pix2Pix etc).
    """
    if im0.shape == im2.shape:
        if len(im0.shape) == 2:
            XPixcels, YPixcels = im0.shape
            dtype = im0.dtype
            if "horizontal" in mode:
                combined = np.zeros((XPixcels, YPixcels * 2), dtype=dtype)
                combined[:, 0:YPixcels, ] = im0[:, :, ]
                combined[:, YPixcels:2 * YPixcels, ] = im2[:, :, ]
            elif "vertical" in mode:
                combined = np.zeros((XPixcels * 2, YPixcels, 3), dtype=dtype)
                combined[0:XPixcels, :, ] = im0[:, :, ]
                combined[XPixcels:2 * XPixcels, :, ] = im2[:, :, ]
            else:
                raise ValueError("Unknown stitching mode: Only horizontal and vertical are supported")


        else:

            XPixcels, YPixcels, channels = im0.shape
            dtype = im0.dtype
            if "horizontal" in mode:
                combined = np.zeros((XPixcels, YPixcels * 2, 3), dtype=dtype)
                combined[:, 0:YPixcels, :] = im0[:, :, :]
                combined[:, YPixcels:2 * YPixcels, :] = im2[:, :, :]
            elif "vertical" in mode:
                combined = np.zeros((XPixcels * 2, YPixcels, 3), dtype=dtype)
                combined[0:XPixcels, :, :] = im0[:, :, :]
                combined[XPixcels:2 * XPixcels, :, :] = im2[:, :, :]
            else:
                raise ValueError("Unknown stitching mode: Only horizontal and vertical are supported")

        return combined
    else:
        raise ValueError(f"The shapes of the images {im0.shape} {im2.shape} are not matcing !!")


def combineImagesForPixPix(imaageFolder, outputPrefix='combined'):
    """
    method for helping in the generation of combined images for the Pix2Pix model
    :param imaageFolder: path of the images folder
    :param outputPrefix: optional : default combined
    :return: None
    """
    imageList = glob.glob(os.path.join(imaageFolder, "*.png"))
    logging.info(f"Found {len(imageList)} number of images !!!!")
    for image in imageList:
        baseName = os.path.basename(image)[:-3]
        im = PIL.Image.open(image)
        palette = im.getpalette()
        im1 = np.array(PIL.Image.open(image))
        # Determine the total number of colours
        num_colours = int(len(palette) / 3)
        # Determine maximum value of the image data type
        max_val = float(np.iinfo(im1.dtype).max)
        # Create a colour map matrix
        map = np.array(palette).reshape(num_colours, 3)
        im1 = map[im1, :]

        # im1 = np.dstack([im1, im1, im1])  ### step was required as the image was just 2d array (of indexed image)

        im2 = np.array(PIL.Image.open(image[:-3] + "jpg"))
        combined = PIL.Image.fromarray(stitchImages(im1, im2))
        outName = os.path.join(imaageFolder, 'combined_' + baseName + 'jpg')
        combined.save(outName)
    logging.info("Completed the creation of the combined images")


def loadKikuchiFromUp2File(up2FilePath, ignoreKikuchiData=False):
    """
    Method to load the up2 file (of kikuchi patterns). Note that it is still not complete and may need to be debugged.
    works ok for the sqare grid but still not implemented for the hex grid.

    :param up2FilePath: full path of the up2 file
    :return: header info :a dict with various info wrt number of patterns in the data, their size etc.
             kikuchiData a nXm numpy array of uint16 data type where n,m are rows and columns of the ebsd data set.
    """
    # global headerInfo, nPatters, data
    headerData = []
    with open(up2FilePath, "rb") as u2file:

        headerInfo = {"FileFormat": {"nBytes": 4}, "patx": {"nBytes": 4}, "paty": {"nBytes": 4},
                      "Offset": {"nBytes": 4},
                      "areExtraPatternsPresent": {"nBytes": 1}, "nColumns": {"nBytes": 4},
                      "nRows": {"nBytes": 4}, "isHexGrid": {"nBytes": 1}, "xStep": {"nBytes": 8},
                      "yStep": {"nBytes": 8}}  ## 4 bytes is integer, 1 byte is boolean, 8 bytyes is double
        for key in headerInfo:
            # headerData.append(int.from_bytes(u2file.read(4),"little"))
            if 2 < headerInfo[key]["nBytes"] < 8:  ## 4 bytes integers
                headerInfo[key]["value"] = int.from_bytes(u2file.read(headerInfo[key]["nBytes"]), "little")
            elif headerInfo[key]["nBytes"] == 1:  ## boolean values
                headerInfo[key]["value"] = bool(int.from_bytes(u2file.read(headerInfo[key]["nBytes"]), "little"))
            else:  ## case of double float
                headerInfo[key]["value"] = struct.unpack('d', u2file.read(headerInfo[key]["nBytes"]))[0]

        nPatterns = __calculateNPatternsFromHeader(headerInfo)

    if not ignoreKikuchiData:

        data = np.fromfile(up2FilePath, count=nPatterns * headerInfo["patx"]["value"] * headerInfo["paty"]["value"],
                           dtype=np.uint16, offset=headerInfo["Offset"]["value"])
        # rawKikuchiData = data.tobytes()
        with open(up2FilePath, 'rb') as f:
            f.seek(headerInfo["Offset"]["value"])
            rawKikuchiData = f.read()
        kikuchiData = data.reshape((-1, headerInfo["patx"]["value"], headerInfo["paty"]["value"]))  ##

    else:
        kikuchiData = None
        rawKikuchiData = None

    return headerInfo, kikuchiData, rawKikuchiData


def __calculateNPatternsFromHeader(headerInfo):
    if (headerInfo["isHexGrid"]["value"] and headerInfo["areExtraPatternsPresent"]["value"]) or not \
            headerInfo["isHexGrid"]["value"]:
        nPatters = headerInfo["nRows"]["value"] * headerInfo["nColumns"]["value"]
    elif headerInfo["isHexGrid"]["value"] and not headerInfo["areExtraPatternsPresent"]["value"]:
        nPatters = headerInfo["nRows"]["value"] * headerInfo["nColumns"]["value"] - int(
            np.around(headerInfo["nRows"]["value"] / 2.0))
        warnings.warn("Note that extra patterns are not present for this case of Hex grid data set. "
                      "Hence  an error may come up in rehspaing data into rowsXcolumns format. "
                      "needs logic to insert blank patters at last columns of alternatie rows.")
    else:
        raise ValueError(
            "Unknown condition!! only square grid/hexagonal grid are supported. In case of hex grid there might be extra patters/not")
    print(f"Header = {headerInfo} ")
    return nPatters


def __calculatedUp2Filesize(headerInfo):
    bytesPerPixcel = 2  ## for up2 =2, for up1 = 1

    headerTotalBytes = 0
    for key in headerInfo:
        headerTotalBytes += headerInfo[key]["nBytes"]

    bytesPerPattern = headerInfo['patx']["nBytes"] * headerInfo['paty']["nBytes"] * bytesPerPixcel
    if headerInfo['areExtraPatternsPresent']["value"]:
        nPatterns = headerInfo['nColumns']['value'] * headerInfo['nRows']['value']
    else:
        nPatterns = headerInfo['nColumns']['value'] * headerInfo['nRows']['value'] - int(
            headerInfo['nRows']['value'] / 2)

    bytesKikuchiData = nPatterns * bytesPerPattern
    totalFileSize = bytesKikuchiData + headerTotalBytes

    return totalFileSize, bytesKikuchiData


def removeAlphaChannel(im):
    """

    :param im: numpy array or PIL image
    :return: same image data as numpy array the input but with alpha channel removed
    """
    if isinstance(im, PIL.Image.Image):
        im = np.asarray(im)
    shape = im.shape
    if len(shape) == 3 and shape[2] == 4:  ## case of RGBA image
        im = im[:, :, 0:3]
        assert len(im.shape) == 3, f"removeAlphaChannel : the shape of the array is : {im.shape}"
    return im


def exportToUp2(fileName, headerInfo, rawKikuchiData):
    bytesPerPixcel = 2  ## for up2 =2, for up1 = 1
    with open(fileName, 'wb') as f:
        headerTotalBytes = 0
        for key in headerInfo:
            headerTotalBytes += headerInfo[key]["nBytes"]
            # headerData.append(int.from_bytes(u2file.read(4),"little"))
            if 2 < headerInfo[key]["nBytes"] < 8:  ## 4 bytes integers
                f.write(struct.pack('@i', headerInfo[key]["value"]))
            elif headerInfo[key]["nBytes"] == 1:  ## boolean values
                f.write(struct.pack('?', headerInfo[key][
                    "value"]))  # bool(int.from_bytes(u2file.read(headerInfo[key]["nBytes"]), "little"))
            else:  ## case of double float
                f.write(struct.pack('d', headerInfo[key][
                    "value"]))  # = struct.unpack('d', f.read(headerInfo[key]["nBytes"]))[0]

        f.write(rawKikuchiData)
    bytesPerPattern = headerInfo['patx']["nBytes"] * headerInfo['paty']["nBytes"] * bytesPerPixcel
    if headerInfo['areExtraPatternsPresent']["value"]:
        nPatterns = headerInfo['nColumns']['value'] * headerInfo['nRows']['value']
    else:
        nPatterns = headerInfo['nColumns']['value'] * headerInfo['nRows']['value'] - int(
            headerInfo['nRows']['value'] / 2)

    bytesKikuchiData = nPatterns * bytesPerPattern
    totalFileSize = bytesKikuchiData + headerTotalBytes
    print(f"just wrote the biary file : {fileName=},{headerTotalBytes=} ; {totalFileSize =} "
          f"Bytes {nPatterns =}")


def generateSourceArraypix2pix(extracted_values, normalization_params):
    """
    Generate a 256X256X3 nparray which, when plotted, looks like an image with three rows,
    each for one parameter except for temperature.

    :parameters: extracted_values (dictionary with 'power', 'velocity', 'timestamp'),
                 normalization_params (dictionary containing normalizing values)
    :return: 256X256X3 nparray
    """
    final_image = np.zeros((256, 256))

    for i, (param, value) in enumerate(extracted_values.items()):
        normalized_value = (float(value) / normalization_params[param]) - 0.5

        top = i * (256 // len(extracted_values))
        bottom = (i + 1) * (256 // len(extracted_values))
        final_image[top:bottom, :] = normalized_value

    stackedArray = np.dstack([final_image] * 3)

    return stackedArray


def generateTargetArray(tempArray, normalization_params, imDatasetMathType):
    """
    to generate target array

    :parameters : tempArray - is a npy file with the temperature distribition values originallly from the data that is acquired by converting txt files to csv,png and npy files.
     and a dictionary containing normalizing values for
                    power,velocity,timestamp and temperature
    :return: 256X256X3 normalized nparray
    """
    # Scale the normalized temperature to -0.5 to 0.5 (min-max)
    if imDatasetMathType == "nonLogarithmic":
        tempArray_scaled = (tempArray - normalization_params['T_min']) / (
                normalization_params['T_max'] - normalization_params['T_min']) - 0.5
    elif imDatasetMathType == "logarithmic":
        tempArray_scaled = (np.log(tempArray) - np.log(normalization_params['T_min'])) / (
                np.log(normalization_params['T_max']) - np.log(normalization_params['T_min'])) - 0.5

    # Resize the temperature array to 256x256x3
    target_shape = (256, 256)
    resizedArray = np.array(Image.fromarray(tempArray_scaled).resize(target_shape))
    resizedArray = np.dstack([resizedArray, resizedArray, resizedArray])

    return resizedArray


def makeTrainExampleNum2Pix(extracted_values, tempArraySlice, normalization_params, imDatasetMathType):
    """
    Create a training example for Num2Pix.

    :parameters : extracted_values (dictionary with 'power', 'velocity', 'timestamp'),
                  tempArraySlice (a slice of the temperature array),
                  normalization_params (dictionary containing normalizing values)
    :return: A tuple of three 256X512X3 normalized numpy arrays: A, B, C
    """
    A = generateSourceArraypix2pix(extracted_values, normalization_params)
    B = generateTargetArray(tempArraySlice, normalization_params, imDatasetMathType)
    C = stitchImages(A, B)

    return A, B, C


def plotComparativeImages(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Image {chr(65 + i)}: {title}')
    plt.tight_layout()
    plt.show()


def precisionErrorCheck(a, b, tol=1e-9):
    """
    Compare two arrays/numbers with tolerance.

    Parameters:
        a (array-like): First input.
        b (array-like): Second input.
        tol (float, optional): Tolerance for equality. Default is 1e-9.

    Returns:
        bool: True if |a - b| < tol for all elements, else False.

    This function checks if absolute differences between inputs are within tolerance. It returns True if they are, else False.
    """
    return np.all(np.abs(a - b) < tol)


def parametersFromNparray(sourceArray, normalization_params):
    """
    Extract and reverse normalization from a stacked NumPy array (sourceArray).

    Parameters:
        sourceArray (np.ndarray): Stacked array containing normalized power, velocity, and timestamp data (256X256X3).
        normalization_params (dict): Dictionary with normalization values for power, velocity,timestamp AND temperature(which is not used in this method).

    Returns:
         (p, v, t) representing power, velocity, and timestamp data,
        respectively, after reversing the normalization.

    """
    # Split the stacked array into three equal parts
    p_size = len(sourceArray) // 3
    p_normalized = sourceArray[:p_size]
    v_normalized = sourceArray[p_size:2 * p_size]
    t_normalized = sourceArray[2 * p_size:]

    # Reverse the normalization using the provided parameters
    p = p_normalized * normalization_params['power']
    v = v_normalized * normalization_params['velocity']
    t = t_normalized * normalization_params['timestamp']

    return p, v, t


def denormalizedTargetArray(targetArray, normalization_params):
    denormalizedtempArray = targetArray * normalization_params["T_max"]

    return denormalizedtempArray


def crop_to_square(original_image, target_size=None):
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)
    # Get the minimum dimension (width or height)
    min_dimension = min(original_image.width, original_image.height)

    # Calculate the cropping box
    left = (original_image.width - min_dimension) // 2
    top = (original_image.height - min_dimension) // 2
    right = (original_image.width + min_dimension) // 2
    bottom = (original_image.height + min_dimension) // 2

    # Crop the image
    cropped_image = original_image.crop((left, top, right, bottom))

    if cropped_image.width != cropped_image.height:
        raise ValueError("Cropped image is not a square")

        # Optionally resize the image to target size
    if target_size:
        cropped_image = cropped_image.resize((target_size, target_size))

    return cropped_image

    # Example usage


if __name__ == "__main__":
    import logging

    logFileName = 'util.log'
    logging.basicConfig(filename=logFileName, level=logging.DEBUG,
                        format='%(asctime)-15s  %(levelname)-5s  %(message)-20s')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    from PIL import Image

    test_crop_to_square = True

    if test_crop_to_square:
        originalImage = Image.open(r"../data/img2Img/val/260_900_16.png")
        cropped_image = crop_to_square(originalImage)
        plotComparativeImages([originalImage, cropped_image], titles=['original', 'cropped'])
        exit(-300)

    testGenerateSourceAndTargetArrayPix2Pix = False
    if testGenerateSourceAndTargetArrayPix2Pix:
        p = 106.0
        v = 700.0
        t = 29.0
        tolerance = 1e-5
        normalization_params = {'power': 520.0, 'velocity': 1200.0, 'timestamp': 100.0, 'T_max': 5000}
        sourceArray = generateSourceArraypix2pix(p, v, t, normalization_params)
        # tempForNorm=5000.0
        tempArrayPath = r"C:\Users\am2195\Desktop\Ti64-5_cropped\melt_power_156_velocity_400\\temperature_02_cropped.npy"
        tempArray = np.load(tempArrayPath)
        tempArraySlice = tempArray[:, :, 5]
        targetArray = generateTargetArray(tempArraySlice, normalization_params)
        stitchedImages = makeTrainExampleNum2Pix(p, v, t, tempArraySlice, normalization_params)
        # plotComparitiveImages([sourceArray,targetArray,stitchedImages],["A","B","C"],cmap='jet')
        # plt.imshow(stitchedImages)
        # plt.show()
        systemParameters = parametersFromNparray(sourceArray, normalization_params)
        if precisionErrorCheck(systemParameters[0], p, tolerance) and precisionErrorCheck(systemParameters[1], v,
                                                                                          tolerance) and precisionErrorCheck(
                systemParameters[2], t, tolerance):
            print("The values in systemParameters are close to the original values.")
        else:
            print("The values in systemParameters are not close to the original values.")
        denormalizedTargetArray = denormalizedArrayTargetArray(targetArray, normalization_params)

        exit(-13)

    testGenerateUniqueGrainMap = False
    if testGenerateUniqueGrainMap:
        imgFolder = r"C:\Users\am2195\Desktop\num2img\machineLearning\data\dataForNpz\test"
        imgName = "156_900_3.png"
        img = PIL.Image.open(os.path.join(imgFolder, imgName))
        tmp = generateUniqueGrainMap(img)
        plotComparitiveImages([img, tmp], titles=["Input", "iswarya  Grain Color map"])
        exit(-20)

    testRgb2label = False
    if testRgb2label:
        colorImage = r"D:\Amrutha\ML Data\GrainBoundaryWork\INCONEL images\RawAndMarkedImages\Raw_Target_6.png"

        colorImage = np.asarray(PIL.Image.open(colorImage))
        labelImage = rgb2label(colorImage)
        plotComparitiveImages(images=[colorImage, labelImage], titles=["ColorImage", "LabeledImage"],
                              mainTitle="Compare")
        exit(-111)

    testUp2 = False
    if testUp2:
        up2FilePath = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject\expData\5x5\DB_Files\mr1712\Cole\Mapping\Ebsd\MHEA\Block 3\Area 2\map20220420152003503.up2'
        up2FilePath = r'F:\kiranData\Type_1_EBSD\DB_Files\vk0237\Cole\Mapping\Ebsd\NbCreep\Type_1_T_1500Deg_SR_1e-3_TD_LTDSection\Area 4\map20220809140436297.up2'
        patFile = r'C:\Users\vk0237\OneDrive - UNT System\UNT_work\kikuchiProject\expData\5x5\DB_Files\mr1712\Cole\Mapping\Ebsd\MHEA\Block 3\Area 2\map20220420152003503.up2'
        patFileBaseName = os.path.basename(patFile)
        header, data, rawKikuchiData = loadKikuchiFromUp2File(up2FilePath)
        #### below is the data of kiran of scan : D:\mani\kiranData\OIM Map 4.ang
        header["nRows"]["value"] = 592
        header["nColumns"]["value"] = 669
        header["xStep"]["value"] = 0.5000
        header["yStep"]["value"] = 0.433013
        header["patx"]["value"] = 480
        header["paty"]["value"] = 480
        nPatterns = __calculateNPatternsFromHeader(header)
        with open(patFile, 'rb') as pat:
            rawKikuchiData = pat.read()
        patFileSize = getsize(patFile)
        totalFileSize, bytesKikuchiData = __calculatedUp2Filesize(header)
        print(f"{totalFileSize=} {bytesKikuchiData=} {patFileSize=}, {patFileSize-bytesKikuchiData=}")

        with open('header.txt', 'w') as f:
            json.dump(header, f)
        print(f"dumped header to header.txt")
        tmpOutUp2 = os.path.join(r'../tmp', patFileBaseName[:-4] + '.up2')
        exportToUp2(tmpOutUp2, header, rawKikuchiData)
        # header2, data, rawKikuchiData = loadKikuchiFromUp2File(tmpOutUp2HeaderFile,ignoreKikuchiData=True)
        print(header)

        exit(-10)

    testStitchImages = False
    if testStitchImages:
        dir = r"D:\Amrutha\ML Data\GrainBoundaryWork\fullImagesForTesting\test"
        # combineImagesForPixPix(dir,)
        # imageList = glob.glob(os.path.join(dir, "*.png"))
        im1Name = r'D:\Amrutha\ML Data\GrainBoundaryWork\fullImagesForTesting\test\rex_twins_03_source_testing_01_small_1.png'
        im2Name = r'D:\Amrutha\ML Data\GrainBoundaryWork\fullImagesForTesting\test\rex_twins_03_target_testing_01_small_1.png'

        im1 = removeAlphaChannel(np.asarray(PIL.Image.open(im1Name)))
        im2 = removeAlphaChannel(np.asarray(PIL.Image.open(im2Name)))

        combinedImage = stitchImages(im1, im2)

        outName = os.path.join(dir, os.path.basename(im1Name).split(sep=".")[0] + "_" + os.path.basename(im2Name))
        Image.fromarray((combinedImage)).save(outName)

        plotComparitiveImages(images=[im1, im2, combinedImage])

        exit(-200)

    testPixcelIdfromIJandItsReverse = False
    if testPixcelIdfromIJandItsReverse:
        ij = (8, 11)
        ID = pixcelIdfromIJ(ij, (134, 119))
        print(ID)
        ijNew = indextoIJ(ID, (134, 119))
        print(f"Started with ij = {ij} and ended up with ij={ijNew}")

        #### case of cropped scan
        ij = (2, 0)
        ID = pixcelIdfromIJ(ij, (19, 70))
        print(f"The linear Id ={ID}")
        ijNew = indextoIJ(ID, (19, 70))
        print(f"Started with ij = {ij} and ended up with ij={ijNew}")
        exit(20)

    testEstebsdNeighborPixcelIndiccesForIJ = False
    if testEstebsdNeighborPixcelIndiccesForIJ:
        queryIndices = (3, 3)
        ebsdMapShape = (134, 119)
        neighbors = ebsdNeighborPixcelIndiccesForIJ(queryIndices=queryIndices, ebsdMapShape=ebsdMapShape)
        print(neighbors)
        queryIndices = (156, 119)
        ebsdMapShape = (134, 119)
        neighbors = ebsdNeighborPixcelIndiccesForIJ(queryIndices=queryIndices, ebsdMapShape=ebsdMapShape)
        print(neighbors)
        exit(10)

    testPlotComparitive = False
    if testPlotComparitive:
        segImagePath = r"D:\CurrentProjects\ml_microStructureQuantification\imageData\semanticSegData\semanticSegData\SegmentationClassRaw\val_augmetned_0.png"
        segImagePath = r"../data/testingData/img2.jpg"
        # outImagePath =
        segImage = np.asarray(PIL.Image.open(segImagePath))
        segImage2 = segImage + 100  ## just adding brigghtness for testing
        plotComparitiveImages(images=[segImage, segImage2], filePath=r'../tmp/comparitivePltot.png')
        exit(-100)
    testCorrectSegmentationMask = False
    if testCorrectSegmentationMask:
        segMaskPath = r"D:\CurrentProjects\ml_microStructureQuantification\imageData\semanticSegData\semanticSegData\SegmentationClassRaw\train_augmetned_200.png"
        segImage = np.array(PIL.Image.open(segMaskPath))
        correctedMask = correctSegmentationMask(segImage, desiredBoundaryThickness=7, debugOn=False)
        mask2ndType = makeMask2ndTypeDeepLab(correctedMask)
        plotComparitiveImages([segImage, correctedMask, mask2ndType],
                              mainTitle="SegMask and @nd Type SegMask for DeepLearning!!!")
    testCorrectBoundaryColorImages = False

    if testCorrectBoundaryColorImages:
        boundaryImagePath = r"../data/testingData/redFilledImage.png"
        boundaryImage = np.array(PIL.Image.open(boundaryImagePath))
        correctBoundaryColorImages(boundaryImage, colorThreshold=100, desiredBoundaryThickness=3, debugOn=True)

    testRgb2labelModified = True

    if testRgb2labelModified:
        imagePath = r"../data/testingData/redFilledImage.png"
        image = np.array(PIL.Image.open(imagePath))
        rgb2labelModified(image, colorThreshold=100, debugOn=True)

    testBoundarySimilarityIndex = False
    if testBoundarySimilarityIndex:
        im1Path = r"../data/programeInternalData/syntheticBoundaryImages/boundaryModel_1.jpg"
        for i in range(5):
            im2Path = r"../data/programeInternalData/syntheticBoundaryImages/boundaryModel_1" + f"_{i + 1}.jpg"
            logging.debug(f"im1: {os.path.basename(im1Path)}, im2: {os.path.basename(im2Path)}")
            im1 = np.array(PIL.Image.open(im1Path).convert('L'))
            im2 = np.array(PIL.Image.open(im2Path).convert('L'))
            boundaryDissimilarityIndex(im1, im2, debugOn=True)
    testRemoveSmallIlands = False
    if testRemoveSmallIlands:
        im1Path = r"../data/programeInternalData/rpv_micro_target_03.png"  ## for testing !!!!

        im1Path = r"D:\CurrentProjects\ml_microStructureQuantification\RPVMicro\rpv_micro_target_03.png"
        im1 = np.array(PIL.Image.open(im1Path))
        im1 = removeSmallIslands(im1, islandSize=35)
        outName = im1Path[:-4] + "_cleaned.png"
        im1 = PIL.Image.fromarray(im1)
        im1.save(outName)
    testLoadDicomImage = False
    if testLoadDicomImage:
        imPath = r"../data/testingData/RSC84 DICOM-DICONDE.dcm"
        ds = pydicom.dcmread(imPath)
        data = ds.pixel_array
        plt.imshow(data)
        plt.show()
        # print(dcmImage)
    testImageSplit = True
    if testImageSplit:
        path = r"../data/inputOutputData/augmentorTestData"
        dir_in = r"../data/inputOutputData/augmentorTe![](D:/Amrutha/Images/5000Ximages/Markedimages at 5000X/5KX_Raw_1_boundary_1.png)stData"
        dir_out = r"../tmp"

        imSize = 256
        imList = os.listdir(path)
        for imageName in imList:
            if (imageName.endswith(".png")):
                img = np.array(PIL.Image.open(os.path.join(dir_in, imageName)))
                imageSplit(img, imSize, imageName, dir_out)