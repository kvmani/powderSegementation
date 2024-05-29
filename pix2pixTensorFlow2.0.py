import argparse
import glob
import pathlib
import shutil
import warnings
import PIL
import h5py

import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import platform
import json
import os
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import logging
import io
import sys
import keras
import shutil
import pandas as pd

try:
    import cfg
    from utilities import util
except:
    print("Unable to find the cfg or utilities package!!! trying to now alter the system path !!")
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.dirname( r'../'))
    sys.path.insert(0, os.path.dirname(r'../utilities'))
    import cfg
    from utilities import util

timeStamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logFilename = f"tmp/pix2pixTensorFlow2.0_{timeStamp}.log"
saveImagesToDisk = True  ### set it for true if you want inference images are to be saved on disk
logging.basicConfig(filename=logFilename, level=logging.INFO,
                    format='%(asctime)-15s  %(levelname)-5s  %(message)-20s')

logger = tf.get_logger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)
# tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

logging.info(f"Testing the loggging to console")

# runInDebugMode=False
# if runInDebugMode:
warnings.warn("Running in debug mode dont forget to switch this off in production code!!!")
tf.config.run_functions_eagerly(True)
# tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()
tf.data.experimental.enable_debug_mode()

generateModelImage = False
runMode = "train"  ### one of ['train', 'test', 'inference']

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing training data")
parser.add_argument("--mode", required=True, choices=["train", "test", "inference"])
parser.add_argument("--output_dir", default="", help="where to put output files")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--codeDevelopmentMode", default="False", choices=["True", "False"],
                    help='if set true runs on toy data set so that several changes can be made to code without waiting for too much time for data load etc')  ## change to store_false
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing/inference")
parser.add_argument("--max_steps", type=int, default=2000000, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=500, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=500, help="update summaries every summary_freq steps")
parser.add_argument("--log_hist_weights_frequency", type=int, default=50,
                    help="log weights histograms evry log_hist_weights_frequency epochs")
parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=500,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=3000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=128, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--augment", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--lr", type=float, default=1e-5, help="initial learning rate for adam")
parser.add_argument("--lrDescriminator", type=float, default=1e-5, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.99, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=0.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=0.0, help="weight on GAN term for generator gradient")
parser.add_argument("--l2_weight", type=float, default=1.0,
                    help="weight on L2 term for generator gradient")
parser.add_argument("--imType", default=".npz", help="type of image '.png' or '.jpeg' or '.tiff' ,'.npz' ",
                    choices=[".png", ".npz", ".tif", ".tfrecord"])
parser.add_argument("--imDataSetType", default="grainBoundary",
                    help="type of data set ebsd euler maps or kikuchi maps ",
                    choices=["ebsd", "kikuchi", "grainBoundary", "numerical2Img"])

parser.add_argument("--imDataSubSetType", default="LPBF_TemperatureData",
                    help="type of sub data sets within a given dat aset ",
                    choices=["LPBF_TemperatureData", "LPBF_microstructure", ])
parser.add_argument("--imDatasetMathType", default="nonLogarithmic",
                    help="type of math implemented on data i.e., logarithmic or non-logarithmic",
                    choices=["logarithmic", "NonLogarithmic", ])
parser.add_argument("--cuda_device", type=int, default=0,
                    help="Device ID of the cuda core on which jo needs to be run defaults to :0 ")  # added by mani
parser.add_argument("--output_group_dir", default='grainBoundaryLearning2023',
                    help="dir name under which runs needs to be saved useful for grouping different experiments")
parser.add_argument("--fileNamePrefix", default='debugTestValLr1em5',
                    help="file name prefix ; useful for indicating the main parameters of the run in experiments")
parser.add_argument("--comment", default='grainboundarydata_1.1:  Debug run2 gen lr reduced to 1e-5, lr desc 1e-5 ',
                    help="Comment describning the run details")

# export options
a = parser.parse_args()
a.hostName = platform.node()
PATH = a.input_dir
if "True" in a.codeDevelopmentMode:

    a.output_dir = os.path.abspath(r'tmp')
    a.max_steps = 1000
    a.max_epochs = 10
    a.summary_freq = 10
    a.progress_freq = 5
    a.display_freq = 5
    a.save_freq = 50
    a.output_group_dir = 'codeDevptOutPut'
    a.log_hist_weights_frequency = 1
    logging.info(f"Running in code development mode outputdir:{a.output_dir} ")
    if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
        dataSetPath = r'..\data\num2Img'
    elif "grainBoundary" in a.imDataSetType:
        dataSetPath = r'data\powder'

    #dataSetPath = cfg.mlCodeDevptInputPath

    PATH = dataSetPath
    # exit(-1)

BUFFER_SIZE = 400
BATCH_SIZE = a.batch_size
IMG_WIDTH = 256
IMG_HEIGHT = 256

if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
    # TARGET_IMAGE_NORMALIZTION_VALUE = tf.constant(5000)  # temperature (K)
    # SOURCE_IMAGE_SEGMENTS = tf.constant(3)
    # SOURCE_DATA_BLOCK_START_INDEX = tf.constant([0, 85, 171])
    # SOURCE_DATA_BLOCK_SIZES = tf.constant([85, 170, 86])
    normalization_values = {
        "SS316L":
            {
                'power': tf.constant(250.0),
                'velocity': tf.constant(1.5),
                'timestamp': tf.constant(20.6),
                # 'material': tf.constant(4.0),
                'T_min': tf.constant(298.0),
                'T_max': tf.constant(3760.0),
            },
        "W":
            {
                'power': tf.constant(900.0),
                'velocity': tf.constant(0.5),
                'timestamp': tf.constant(77.0),
                # 'material': tf.constant(4.0),
                'T_min': tf.constant(298.0),
                'T_max': tf.constant(14891.0),
            },
        "Cu":
            {
                'power': tf.constant(500.0),
                'velocity': tf.constant(1.1),
                'timestamp': tf.constant(7.7),
                # 'material': tf.constant(4.0),
                'T_min': tf.constant(298.0),
                'T_max': tf.constant(3915.0),
            },
        "AlSiMg":
            {
                'power': tf.constant(250.0),
                'velocity': tf.constant(1.5),
                'timestamp': tf.constant(25.6),
                # 'material': tf.constant(4.0),
                'T_min': tf.constant(298.0),
                'T_max': tf.constant(2685.0),
            },
        "ALL":
            {
                'power': tf.constant(900.0),
                'velocity': tf.constant(1.5),
                'timestamp': tf.constant(77.0),
                'material': tf.constant(4.0),
                'T_min': tf.constant(298.0),
                'T_max': tf.constant(14891.0),
            }

    }
    SOURCE_IMAGE_NORMALIZATION_VALUES = normalization_values["SS316L"]
else:
    TARGET_IMAGE_NORMALIZTION_VALUE = 127.5
    SOURCE_IMAGE_SEGMENTS = 1
    SOURCE_DATA_BLOCK_SIZES = [256, ]
    SOURCE_IMAGE_NORMALIZTION_VALUES = [tf.constant(127.5)]


@tf.function()
def sourceArrayDenormalizedWithTF(stackedArray):
    n = len(SOURCE_IMAGE_NORMALIZATION_VALUES) - 2
    size = len(stackedArray) // n
    normalized_arrays = [stackedArray[i * size:(i + 1) * size] for i in range(n)]

    denormalized_arrays = []
    for i, array in enumerate(normalized_arrays):
        array_float32 = tf.cast(array, dtype=tf.float32)
        mean_normalized_value = tf.reduce_mean(array_float32)
        normalization_value = tf.cast(SOURCE_IMAGE_NORMALIZATION_VALUES[list(SOURCE_IMAGE_NORMALIZATION_VALUES.keys())[i]], dtype=tf.float32)
        denormalized_value = (mean_normalized_value + 0.5) * normalization_value
        denormalized_arrays.append(tf.fill((size, 256), denormalized_value))

    denormalizedArray = tf.concat(denormalized_arrays, axis=0)
    stackedDenormalizedArray = tf.stack([denormalizedArray] * 3, axis=-1)

    return stackedDenormalizedArray


@tf.function
def denormalizedTargetArray(targetArray):

    if "nonLogarithmic" in a.imDatasetMathType:
        denormalizedtempArray = (((targetArray + 0.5) * (
                    SOURCE_IMAGE_NORMALIZATION_VALUES['T_max'] - SOURCE_IMAGE_NORMALIZATION_VALUES['T_min'])) + SOURCE_IMAGE_NORMALIZATION_VALUES['T_min'])
    elif "logarithmic" in a.imDatasetMathType:

        # denormalizedtempArray = tf.exp((targetArray + 0.5) * (
        #             tf.math.log(SOURCE_IMAGE_NORMALIZATION_VALUES['T_max']) - tf.math.log(
        #         SOURCE_IMAGE_NORMALIZATION_VALUES['T_min'])) + tf.math.log(SOURCE_IMAGE_NORMALIZATION_VALUES['T_min']))
        targetArray = tf.cast(targetArray, dtype=tf.float64)
        denormalizedtempArray = tf.exp((targetArray + 0.5) * (
                tf.math.log(tf.cast(SOURCE_IMAGE_NORMALIZATION_VALUES['T_max'], dtype=tf.float64)) - tf.math.log(
            tf.cast(SOURCE_IMAGE_NORMALIZATION_VALUES['T_min'], dtype=tf.float64))) + tf.math.log(
            tf.cast(SOURCE_IMAGE_NORMALIZATION_VALUES['T_min'], dtype=tf.float64)))


    return denormalizedtempArray


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(images, mainTitle=""):
    """Return a 1X3 grid of the images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(12, 5))
    titles = ["input", "target", "output"]
    for i in range(3):
        # Start next subplot.
        plt.subplot(1, 3, i + 1, title=titles[i])
        # plt.title(mainTitle)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
            plt.colorbar()
    plt.tight_layout()
    figure.suptitle(mainTitle, fontsize=16)
    figure.subplots_adjust(top=0.88)
    return figure


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def create_circular_mask2d(h=256, w=256, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    return mask


def create_circular_mask(h=256, w=256, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    a = np.zeros((h, w, 3), dtype=np.float32)
    a[mask, :] = 1.
    return tf.convert_to_tensor(a)


class CircualrMaskLayer(tf.keras.layers.Layer):
    def __init__(self, mask):
        super(CircualrMaskLayer, self).__init__()
        self.mask = mask

    def call(self, inputs):
        return tf.expand_dims(inputs * self.mask, 0)


circualrMasklayer = CircualrMaskLayer(mask=create_circular_mask())


def load(image_file, mode='testOrTrain'):
    image = tf.io.read_file(image_file)
    # image = tf.image.decode_jpeg(image)
    # image = tf.image.decode_png(image, dtype=tf.dtypes.uint16)
    image = tf.image.decode_png(image)
    w = tf.shape(image)[1]
    h = tf.shape(image)[0]
    depth = tf.shape(image)[-1]
    with tf.control_dependencies([tf.Assert(tf.logical_or(tf.equal(depth, 3), tf.equal(depth, 1)), [depth])]):
        image = tf.cond(tf.equal(tf.shape(image)[-1], 3), lambda: image, lambda: tf.image.grayscale_to_rgb(image))

    if "testOrTrain" in mode:
        w = w // 2
        input_image = image[:, :w, :]
        real_image = image[:, w:, :]

    else:
        if w == 2 * h:
            w = w // 2  ### case of ground truth being stiched
            input_image = image[:, :w, :]
            real_image = image[:, w:, :]
            print("Case of inference but stitched with ground truth as the width is double the height")
        else:
            print("Case of inference image hence faking the ground truth!!!")
            input_image = image[:, :w, :]
            real_image = image[:, :w, :]  #### just faking target to be equal to input itself

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    ### below is for normal images with max 2555 (uint8 data type)
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    #
    ### below is for kikuchi images with max 2^**16 (uint16 data type)
    # input_image = (input_image / 32767.5) - 1
    # real_image = (real_image / 32767.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    # if tf.random.uniform(()) > 0.5:
    #     # random mirroring
    #     input_image = tf.image.flip_left_right(input_image)
    #     real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


# inp, re = load(os.path.join(PATH,'train/combined_cmp_b0001.jpg'))
## pick a raandom file from the directory for just plotting to ensure that everythign is fine

try:
    file = np.random.choice(glob.glob(os.path.join(PATH, 'train\*.png')))
    inp, re = load(file)
    print(inp, re)
    exit(-1)
except:
    warnings.warn(f"Unable to load example for viweing possibly {os.path.join(PATH, 'train/*.png')} does not exist")


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    # input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image, image_file


def _parse_function_testOrTrain(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    exampleData = tf.io.parse_single_example(example_proto, feature_description)
    imData = exampleData["image/data"]
    # print(imData, "type : ", type(imData), imData.shape)
    imData = exampleData["image/data"]
    height, width, numChannels = exampleData["image/height"], exampleData["image/width"], \
                                 exampleData["image/numChannels"]
    dataType = exampleData["image/dataType"]
    img_arr = np.fromstring(imData, dtype=dataType).reshape(height, width, numChannels)
    # img_arr = tf.reshape(imData, [height, width, numChannels])
    input_image, real_image = __read_image_from_memory(img_arr, mode='testOrTrain')
    return input_image, real_image


def load_image_inference(image_file):
    input_image, real_image = load(image_file, mode="inference")
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image, image_file


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image, image_file


def __loadNumpyData(path):
    mask = create_circular_mask().numpy().astype(bool)
    with np.load(path) as data:
        inputs, targets = data['inputs'].astype(np.float32), data['targets'].astype(np.float32)
        inputs, targets = (inputs / 32767.5) - 1, (targets / 32767.5) - 1,
        truncateData = False
        if truncateData:
            warnings.warn(f"currently truncating data to only 100 images for debugging !! remove this for real cases")
            inputs = inputs[:100, :, :, :]
            targets = targets[:100, :, :, :]

        inputs[:, ~mask] = 0
        targets[:, ~mask] = 0
        nExamples = inputs.shape[0]  ### number of examples
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        print(f"loaded the data from : {path}")
    return dataset, nExamples


def tf_load_npz(npz_file_tensor, mode='testOrTrain'):
    [input_image, real_image] = tf.py_function(
        func=load_npz,
        inp=[npz_file_tensor, mode],
        Tout=[tf.float32, tf.float32]
    )
    return input_image, real_image, npz_file_tensor


def load_npz_files_num2Img(pathPattern, mode='testOrTrain'):
    listOfFiles = glob.glob(pathPattern)
    #blankFileName = [" " for i in range(500)]
    fileNamearray = []
    data_inp, data_out = [], []
    for item in listOfFiles:
        blankFileName = [" " for i in range(500)]
        with np.load(item) as data:
            image = data['data']
            h, w, depth = image.shape
            if mode == "testOrTrain":
                w = w // 2
                input_image = image[:, :w, :]
                real_image = image[:, w:, :]
            else:
                if w == 2 * h:
                    w = w // 2
                    input_image = image[:, :w, :]
                    real_image = image[:, w:, :]
                    print("Case of inference but stitched with ground truth as the width is double the height")
                else:
                    print("Case of inference image hence faking the ground truth!!!")
                    input_image = image[:, :w, :]
                    real_image = image[:, :w, :]  # just faking target to be equal to input itself

            data_inp.append(input_image)
            data_out.append(real_image)
            for i in range(len(item)):
                blankFileName[i] = item[i]
            fileNamearray.append(blankFileName)

    return data_inp, data_out, np.array(fileNamearray)


def load_npz(npz_file, mode='testOrTrain'):
    # decoded_file = npz_file.decode("utf-8")
    decoded_file = tf.get_static_value(npz_file).decode()
    print(f"{decoded_file=}, {type(decoded_file)}")
    # .numpy()

    with np.load(decoded_file) as data:
        image = data['image']
        # if 'input_image' in data.keys() and 'real_image' in data.keys():
        #     input_image = data['input_image']
        #     real_image = data['real_image']

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    h, w, depth = image.shape

    if mode == "testOrTrain":
        w = w // 2
        input_image = image[:, :w, :]
        real_image = image[:, w:, :]
    else:
        if w == 2 * h:
            w = w // 2
            input_image = image[:, :w, :]
            real_image = image[:, w:, :]
            print("Case of inference but stitched with ground truth as the width is double the height")
        else:
            print("Case of inference image hence faking the ground truth!!!")
            input_image = image[:, :w, :]
            real_image = image[:, :w, :]  # just faking target to be equal to input itself

    return input_image, real_image


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.5, seed=seed, fill_mode='constant', fill_value=0),
            # layers.RandomZoom((0.1, 0.15), seed=seed),
            CircualrMaskLayer(mask=create_circular_mask())
            # layers.Multiply()([circularMask])

        ])
        self.augment_labels = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.5, seed=seed, fill_mode='constant', fill_value=0),
            # layers.RandomZoom((0.05, 0.25), seed=seed),
            CircualrMaskLayer(mask=create_circular_mask())
            # layers.Multiply()([circularMask])
            # layers.RandomZoom((0.1,0.15), seed=seed)
        ])

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


logging.info("Just started data loading!!! It can take a while for large data sets be patient!!!")

if "train" in a.mode:
    if "kikuchi" in a.imDataSetType:
        logging.info("Setting the data type to Kikuchi type and processing the .npz files !!!")
        root_path = r'D:\mani\mlOutputs\tmp'
        a.trainDataPath = root_path + r'\train\train.npz'
        a.valDataPath = root_path + r'\val\val.npz'
        a.testDataPath = root_path + r'\test\test.npz'
        train_dataset, a.numberOfTrainExamples = __loadNumpyData(path=a.trainDataPath)
        test_dataset, a.numberOfTestExamples = __loadNumpyData(path=a.testDataPath)
        val_dataset, a.numberOfValExamples = __loadNumpyData(path=a.valDataPath)

    if "numerical2Img" in a.imDataSetType:
        logging.debug("Processing the numerical 2 image data type from numpy files!!!")

        data_inp, data_out, listOfFiles = load_npz_files_num2Img(pathPattern=PATH + r'/train/*.npz')
        train_dataset = tf.data.Dataset.from_tensor_slices((data_inp, data_out, listOfFiles))
        logging.info(f"loaded train data set : {len(listOfFiles)} number of examples")

        data_test, data_out_test, listOfFiles_test = load_npz_files_num2Img(pathPattern=PATH + r'/test/*.npz')
        test_dataset = tf.data.Dataset.from_tensor_slices((data_test, data_out_test, listOfFiles_test))
        logging.info(f"loaded test data set : {len(listOfFiles_test)} number of examples")

        data_val, data_out_val, listOfFiles_val = load_npz_files_num2Img(pathPattern=PATH + r'/val/*.npz')
        val_dataset = tf.data.Dataset.from_tensor_slices((data_val, data_out_val, listOfFiles_val))
        logging.info(f"loaded val data set : {len(listOfFiles_val)} number of examples")

        # Get the number of files in each dataset
        a.numberOfTrainExamples = len(glob.glob(PATH + r'/train/*.npz'))
        a.numberOfTestExamples = len(glob.glob(PATH + r'/test/*.npz'))
        a.numberOfValExamples = len(glob.glob(PATH + r'/val/*.npz'))


        logging.info("Loaded the data successfully in case of npz files of Numerical2Img data type")
        logging.info(
            f"input Dir : {PATH} Detected number of Images -->train : {a.numberOfTrainExamples}  test : {a.numberOfTestExamples}"
            f" val: {a.numberOfValExamples}")

    else:
        logging.info("Setting the data type to regular images type and processing the .png files !!!")
        train_dataset = tf.data.Dataset.list_files(PATH + r'/train/*.png')
        train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = tf.data.Dataset.list_files(PATH + '/test/*.png', shuffle=False)
        test_dataset = test_dataset.map(load_image_test)
        val_dataset = tf.data.Dataset.list_files(PATH + '/val/*.png', shuffle=False)
        val_dataset = val_dataset.map(load_image_inference)  ####

        numberOfTrainImages = len(glob.glob(PATH + r'/train/*.png'))
        numberOfTestImages = len(glob.glob(PATH + r'/test/*.png'))
        numberOfValImages = len(glob.glob(PATH + r'/val/*.png'))

        logging.info(
            f"input Dir : {PATH} Detected number of Images -->train : {numberOfTrainImages}  test : {numberOfTestImages}"
            f" val:{numberOfValImages}")



    if "kikuchi" in a.imDataSetType:
        train_batches = (
            train_dataset
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .repeat()
                .map(Augment())
                # .map(applyCircualrMask())
                .prefetch(buffer_size=tf.data.AUTOTUNE))
    else:
        train_batches = (
            train_dataset
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .repeat()
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_dataset.batch(BATCH_SIZE)

    if "true" in a.codeDevelopmentMode.lower():
        for example_input, example_target, imageFile in train_dataset.take(1):
            image_grid([example_input[:,:,0],example_target[:,:,0],example_input[:,:,0]],mainTitle="Train data")
            plt.show()
        for example_input, example_target, imageFile in test_dataset.take(1):
            image_grid([example_input[:, :, 0], example_target[:, :, 0], example_input[:, :, 0]],
                       mainTitle="Test data")
            plt.show()
        for example_input, example_target, imageFile in val_dataset.take(1):
            image_grid([example_input[:, :, 0], example_target[:, :, 0], example_input[:, :, 0]],
                       mainTitle="val data")
            plt.show()

            # image_name = os.path.splitext(file_name)[0] + '.png'
            # save_path = os.path.join(r"../tmp",image_name)
            # plt.imshow(selected_image[:,:,0])
            # plt.title(file_name)
            # plt.savefig(save_path)
            # plt.show()



# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)
# val_dataset = val_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3


def display(display_list, mainTitle=""):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        # plt.imshow(display_list[i].numpy())
        plt.axis('off')
    plt.suptitle(mainTitle, fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()


# if a.codeDevelopmentMode:
#     for dataset, datasetName in zip([train_dataset,test_dataset,val_dataset],
#                                     ['train','test','val']):
#         for images, masks in dataset.take(3):
#             sample_image, sample_mask = images, masks
#             display([tf.squeeze(sample_image), tf.squeeze(sample_mask)],mainTitle=datasetName)
#
#     exit(-1)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# if "train" in a.mode:
#     down_model = downsample(3, 4)
#     down_result = down_model(tf.expand_dims(inp, 0))
#     print(down_result.shape)
#     up_model = upsample(3, 4)
#     up_result = up_model(down_result)
#     print(up_result.shape)


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()

# if "train" in a.mode:
#     gen_output = generator(inp[tf.newaxis, ...], training=False)
#     plt.imshow(gen_output[0, ...])
LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    ###aishwarya
    target_float = tf.cast(target, tf.float32)
    gen_output_float = tf.cast(gen_output, tf.float32)
    diff = target_float - gen_output_float
    ###aishwarya

    # # mean absolute error
    # diff = target - gen_output

    l1_loss = tf.reduce_mean(tf.abs(diff))
    l2_loss = tf.reduce_mean((diff) * (diff))

    total_gen_loss = a.gan_weight * gan_loss + (a.l1_weight * l1_loss) \
                     + (a.l2_weight * l2_loss)

    return total_gen_loss, gan_loss, l1_loss, l2_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
# plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# exit(-100)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(a.lr, beta_1=a.beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(a.lrDescriminator, beta_1=a.beta1)
if "train" in a.mode:
    a.output_dir = cfg.mlOutputRootDir
    log_dir = os.path.join(a.output_dir, a.output_group_dir, 'logs')
    outDir = os.path.join(log_dir, a.output_group_dir, f"{a.fileNamePrefix}_" + timeStamp)
    checkpoint_dir = outDir
    logging.info(f"Out put is writtten to : {outDir}")

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

if generateModelImage:
    modelImgFile = outDir + 'generator.png'
    tf.keras.utils.plot_model(generator, to_file=modelImgFile, show_shapes=True, dpi=64)

    modelImgFile = outDir + 'descrimaniator.png'
    tf.keras.utils.plot_model(discriminator, to_file=modelImgFile, show_shapes=True, dpi=64)
    logging.info(f"Generated the models images at : {modelImgFile} an dnow exiting with code -100 !!")
    exit(-100)


def generate_inference_images(model, inference_input, plotOn=True):
    prediction = model(inference_input, training=True)
    if plotOn:
        plt.figure(figsize=(15, 15))
        display_list = [inference_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            if "kikuchi" in a.imDataSetType:
                plt.imshow(display_list[i] * 0.5 + 0.5)
            else:
                plt.imshow(display_list[i] * 12)
            plt.axis('off')
        plt.show()
    return prediction


def getAccuracy(gen_output, target):
    # print(f"{gen_output=}; {target=}")
    gen, target = np.squeeze(gen_output.numpy()), np.squeeze(target.numpy())
    mse = mean_squared_error(gen, target)
    ssim_output = ssim(gen, target, data_range=target.max() - target.min(), channel_axis=2)
    return mse, ssim_output


def generate_images(model, test_input, tar, epoch=0, currentStep=0, outDir=None,
                    imageFile="", plotId=0, plotOn=False, saveIndividualPredictions=False):
    prediction = model(tf.expand_dims(test_input, 0), training=True)

    if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
        ### write your code
        gen_output = denormalizedTargetArray(tf.squeeze(prediction))[:,:,0]
        input_image = sourceArrayDenormalizedWithTF(tf.squeeze(test_input))[:,:,0]
        target = denormalizedTargetArray(tf.squeeze(tar))[:,:,0]

    else:
        gen_output = tf.cast(tf.squeeze((prediction + 1.0) * 127.5), tf.uint8)
        input_image = tf.cast(tf.squeeze((test_input + 1.0) * 127.5), tf.uint8)
        target = tf.cast(tf.squeeze((tar + 1) * 127.5), tf.uint8)

    display_list = [input_image, target, gen_output]
    mse, ssim_output = getAccuracy(prediction, tar)
    imageBaseName = os.path.basename(imageFile.strip(" "))
    title = f'{imageBaseName}     Epoch : {epoch :4}; Step :{currentStep :9}\n' + \
            f"L2 Losss : {mse:10.4f};  SSIM :{ssim_output :8.4f}"
    print(title)
    fig = image_grid(display_list, mainTitle=title)
    if plotOn:
        plt.show()
    if outDir is not None:
        if len(imageFile) > 0:
            figName =imageBaseName[:-4]

        else:
            figName = f"{plotId}"
        fig.savefig(os.path.join(outDir, figName+".png"))
        # print(f"Just saved the figure {os.path.join(outDir,figName)}")
    plt.close(fig)
    if saveIndividualPredictions:
        img_source, img_prediction = Image.fromarray(input_image.numpy()), Image.fromarray(gen_output.numpy())
        imgSrcName, imgPredName = os.path.join(outDir,figName+"_source.png"),os.path.join(outDir,figName+"_prediction.png")
        img_source.save(imgSrcName)
        img_prediction.save(imgPredName)
        print(f"saving : {imgSrcName}")
    return prediction


if "train" in a.mode:
    EPOCHS = a.max_epochs
    a.output_dir = outDir
    a.checkPointDir = checkpoint_dir
    summary_writer = tf.summary.create_file_writer(outDir)


def log_hist_weights(model, writer, epoch):
    print("Now logging the histogram")
    with writer.as_default():
        for tf_var in model.trainable_weights:
            tf.summary.histogram(tf_var.name, tf_var.numpy(), step=epoch)


@tf.function
def writeImagesToTensorbaord(input_image, target, epoch, currentStep, imageFile):

    gen_output = generator(input_image, training=True)
    mse, ssim_output = getAccuracy(gen_output, target)

    if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
        ### write your code
        gen_output = denormalizedTargetArray(tf.squeeze(gen_output))[:,:,0]
        input_image = sourceArrayDenormalizedWithTF(tf.squeeze(input_image))[:, :, 0]
        target = denormalizedTargetArray(tf.squeeze(target))[:,:,0]

    else:

        gen_output = tf.cast(tf.squeeze((gen_output + 1.0) * 127.5), tf.uint8)
        input_image = tf.cast(tf.squeeze((input_image + 1.0) * 127.5), tf.uint8)
        target = tf.cast(tf.squeeze((target + 1) * 127.5), tf.uint8)

    fileName = os.path.basename(imageFile).strip(" ")
    title = f"{fileName} \n Epoch : {epoch :4}; Step :{currentStep :9}\n" + \
            f"L2 Losss : {mse:10.4f};  SSIM :{ssim_output :8.4f}"

    figure = image_grid([input_image, target, gen_output], mainTitle=title)
    with summary_writer.as_default():
        # tf.summary.image('images', [input_image, target, gen_output],
        #                  step=currentStep)
        tf.summary.image("images_combined", plot_to_image(figure), step=currentStep)


@tf.function
def train_step(input_image, target, epoch, currentStep, imageFile):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss, gen_l2_loss = generator_loss(disc_generated_output, gen_output,
                                                                                target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=currentStep)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=currentStep)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=currentStep)
        tf.summary.scalar('gen_l2_loss', gen_l2_loss, step=currentStep)
        tf.summary.scalar('disc_loss', disc_loss, step=currentStep)


def fit(train_ds, epochs, test_ds, plotImages=True, callback=None):
    currentStep = 0
    trainingStart = time.time()
    # tf.profiler.experimental.start(a.output_dir)
    shutil.copy(logFilename, os.path.join(outDir, os.path.basename(logFilename)))

    for epoch in range(epochs):
        start = time.time()
        # display.clear_output(wait=True)
        if plotImages:
            for example_input, example_target, imageFile in test_ds.take(-1):
                imageFile = imageFile.numpy().decode('ascii')
                # example_input = tf.expand_dims(example_input,0)
                # example_target = tf.expand_dims(example_target,0)
                generate_images(generator, example_input, example_target, imageFile=imageFile)
        print("Epoch: ", epoch)
        if epoch % a.log_hist_weights_frequency == 0:
            log_hist_weights(generator, summary_writer, epoch)

        # Train
        trainingStartLocal = time.time()
        for n, (input_image, target, imageFile) in train_ds.enumerate():
            if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
                #imageFile = imageFile.numpy()
                imageFile = "".join([i.decode('ascii') for i in imageFile.numpy().tolist()])

                # Converts the tensor to a numpy array and extracts the first element (the byte string).aishwarya
            else:
                imageFile = imageFile.numpy().decode('ascii')

            input_image = tf.expand_dims(input_image, 0)
            target = tf.expand_dims(target, 0)
            currentStep += 1
            print('.', end='')
            if ((n.numpy() + 1) % 100) == 0:
                print(f"{currentStep} : {checkpoint_dir}")
            currentStepTensor = tf.cast(tf.convert_to_tensor(currentStep), tf.int64)
            train_step(input_image, target, epoch, currentStepTensor, imageFile)
            if currentStep % a.display_freq == 0:
                timePerStep = (time.time() - trainingStartLocal) / a.display_freq
                trainingStartLocal = time.time()
                print(f"Now writing the image to tensorboard, out dir is {a.output_dir}")
                writeImagesToTensorbaord(input_image, target, epoch, currentStepTensor, imageFile)
                with summary_writer.as_default():
                    tf.summary.scalar('timePerStep', tf.convert_to_tensor(timePerStep), step=currentStep)

        print()

        # saving (checkpoint) the model every 20 epochs

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Loading the latest check point from {checkpoint_dir}")
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        mseList, ssimList , summary_list = [], [], []
        testOutputdir = os.path.join(outDir, f'test_{str(epoch)}')
        if not os.path.exists(testOutputdir):
            os.mkdir(testOutputdir)
        count = 0
        for inp, tar, test_imageFile in test_dataset.take(-1):
            if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
                test_imageFile_local = "".join([i.decode('ascii') for i in test_imageFile.numpy().tolist()])

            else:
                test_imageFile_local = test_imageFile.numpy().decode('ascii')


            prediction = generate_images(generator, inp, tar, epoch=epoch, currentStep=currentStep,
                                         outDir=testOutputdir, imageFile=test_imageFile_local, plotId=count)
            count += 1
            mse, ssim_output = getAccuracy(prediction, tar)
            mseList.append(mse)
            ssimList.append(ssim_output)
            tmp_dict = {"time_stamp":datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                        "dataType":"test", "epoch":epoch,"step":currentStep,
                        "mse": mse, "ssim": ssim_output,
                        "input":os.path.basename(test_imageFile_local)

                        }
            summary_list.append(tmp_dict)



        meanMse = tf.convert_to_tensor(np.array(mseList).mean(), dtype=tf.float32)
        meanSsim = tf.convert_to_tensor(np.array(ssimList).mean(), dtype=tf.float32)

        timePerStep = (time.time() - trainingStart) / currentStep
        with summary_writer.as_default():
            tf.summary.scalar('meanMse', meanMse, step=currentStep)
            tf.summary.scalar('meanSsim', meanSsim, step=currentStep)
            tf.summary.scalar('timePerStep', tf.convert_to_tensor(timePerStep), step=currentStep)

        logging.info('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                  time.time() - start))
        print(f"Copying the log file : {logFilename}-->{os.path.join(outDir, os.path.basename(logFilename))}")
        dataFrameFile = os.path.join(outDir, 'summary.csv')
        df = pd.DataFrame(summary_list)
        if epoch==0: ### first epoch
            mainDf = df.copy(deep=True)
        else:
            mainDf = pd.concat([mainDf, df], ignore_index=True)

        mainDf.to_csv(dataFrameFile,index=False)
        shutil.copy(logFilename, os.path.join(outDir, os.path.basename(logFilename)))
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

        if currentStep > a.max_steps:
            print(f"Max steps {a.max_steps} criterion satisfied {currentStep=}")
            break

    # tf.profiler.experimental.stop()
    checkpoint.save(file_prefix=checkpoint_prefix)


if "train" in a.mode:
    a.input_dir = PATH
    with open(os.path.join(a.output_dir, "run_parameters.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    tensorBoardText = vars(a)


def kikuchi_inference():
    global checkpoint_dir, inp
    enableModifiedResolution = False
    outImageDataList = []
    if enableModifiedResolution:
        outputResolution = (240, 240)
        warnings.warn(f"Note that kikuchis are being written at {outputResolution} resolution")
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\10000ImagesData\l1_100_20220912-091110"
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\manualTrainingData\l1_100_20220909-083521"
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\errorWeightVariations\20220710-134950"
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\KikuchiBinningDataForML2_26Data\l1_100_20220915-093206"
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\KikuchiDataForML2_26Data_CorrectNormalizzation\l1_100_20221012-101251"
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\KikuchiDataForML2_26Data_CorrectNormalizzation10Kimages\l1_100_20221012-231937"
    checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\KikuchiDataForML2_26Data_CorrectNormalizzation10Kimages\l2_100_20221014-164145"
    ##checkpoint_dir = r"D:\mani\mlOutputs\kikuchiMlOutputs\logs\fit\errorWeightVariations\20220708-231306"
    print(f"Loading the latest check point from {checkpoint_dir}")
    # hdfFilePath = os.path.join(cfg.kikuchiDataRootFolder, r'expData\HEA_Kikuchi_27-5-22.h5')
    # hdfFilePath = os.path.join(cfg.kikuchiDataRootFolder, r'expData\B4C_ML_TestingData.h5')
    hdfFilePath = os.path.join(cfg.kikuchiDataRootFolder, r'expData\KikuchiBinningDataForML2_26-5-22.h5')
    # ebsdDataPathInh5 = "16X16Good"
    # ebsdDataPathInh5 = "4X4_lowExpHighGain"
    ebsdDataPathsInh5 = ['4X4_lowExpHighGain',
                         '8X8defocusGodExp',
                         '4X4defocusGoodExp',
                         '8X8_good'
                         ]
    # ebsdDataPathsInh5 = ['map20220802083822460',
    #                      ]
    fileCounter = 0
    while True:
        outHdfFileBaseName = os.path.basename(hdfFilePath)[:-3] + 'Processed_' + str(fileCounter)
        if not os.path.isfile(os.path.join(cfg.kikuchiPutputrootFolder, outHdfFileBaseName + '.h5')):
            break
        else:
            fileCounter += 1
    outHdfName = os.path.join(cfg.kikuchiPutputrootFolder, outHdfFileBaseName + '.h5')
    shutil.copy(hdfFilePath, outHdfName)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    groundTruthDataAvailable = True
    if groundTruthDataAvailable:
        groundTruthData = "4X4Ref"
    with h5py.File(outHdfName, 'r+') as hdfData:
        hdfData.create_dataset("comment", (3, 1), 'S10000', [f'Inferred from ckpt file at :{checkpoint_dir}',
                                                             f'SourceFile:{hdfFilePath}',
                                                             f'date run on :{datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")}'])

        for ebsdDataPathInh5 in ebsdDataPathsInh5:
            if saveImagesToDisk:
                hdfBaseName = os.path.basename(outHdfName)[:-3]
                rootDir = os.path.join(os.path.dirname(outHdfName), hdfBaseName)
                outImDir = os.path.join(rootDir, ebsdDataPathInh5, "inferredImages")
                pathlib.Path(outImDir).mkdir(parents=True, exist_ok=True)

            hdfData[ebsdDataPathInh5 + 'MlProcessed'] = hdfData[ebsdDataPathInh5]

            del hdfData[ebsdDataPathInh5]
            ebsdDataPathInh5 = ebsdDataPathInh5 + 'MlProcessed'
            key = f'["{ebsdDataPathInh5}"]["EBSD"]["Data"]["Pattern"]'
            inputPatterns = hdfData[ebsdDataPathInh5]["EBSD"]["Data"]["Pattern"][()]
            nPatterns = inputPatterns.shape[0]
            dataType = inputPatterns.dtype
            patternShape = inputPatterns[0].shape
            mask = create_circular_mask2d(patternShape[0], patternShape[1])
            for i in tqdm(range(nPatterns), f"inference in progress for {ebsdDataPathInh5}"):
                # if i>50:
                #     break;
                originalPattern = inputPatterns[i, :, :].copy()
                inpArray = inputPatterns[i, :, :].copy().astype(np.float32)
                inpArray = util.scale(inpArray, [-1.0, 1.0])
                inpArray[~mask] = 0
                originalshape = inpArray.shape
                resizedImage = np.array(PIL.Image.fromarray(inpArray).resize((256, 256)))
                inp = tf.convert_to_tensor(resizedImage)
                inp = tf.stack([inp, inp, inp])
                inp = tf.transpose(inp, [1, 2, 0])  ### to make number of chanels as the last elment of the shape
                inp = tf.expand_dims(inp, axis=0)
                prediction = generate_inference_images(generator, inp, plotOn=False)
                outImage = prediction[0].numpy()[:, :, 0]  ## getting only the first channel of 3 channel image
                rawOutput = outImage.copy()
                outImage = util.scale(rawOutput, [0, np.iinfo(dataType).max]).astype(dataType)
                # hdfData[ebsdDataPathInh5]["EBSD"]["Data"]["Pattern"][i] = outImageData
                outImageDataList.append(outImage)

                plotOn = False
                if plotOn:
                    util.plotComparitiveImages([originalPattern, inpArray, rawOutput, outImage],
                                               ['original data', 'recived by model', 'rawOutput', 'outut of model'],
                                               f" {i}", shouldBlock=False)
                    plt.axis('scaled')
                    plt.show(block=False)
                    plt.pause(2)  # 3 seconds, I use 1 usually
                    plt.close("all")
            if not enableModifiedResolution:
                outputResolution = originalshape
            else:
                del hdfData[ebsdDataPathInh5]["EBSD"]["Data"]["Pattern"]
                hdfData.create_dataset(f"{ebsdDataPathInh5}/EBSD/Data/Pattern",
                                       data=np.zeros((nPatterns, outputResolution[0], outputResolution[1]),
                                                     dtype=np.uint16))

                hdfData[ebsdDataPathInh5]["EBSD"]["Header"]["Pattern Height"][...] = outputResolution[0]
                hdfData[ebsdDataPathInh5]["EBSD"]["Header"]["Pattern Width"][...] = outputResolution[1]

            mapSize = (int(hdfData[ebsdDataPathInh5]["EBSD"]["Header"]["nRows"][()]), \
                       int(hdfData[ebsdDataPathInh5]["EBSD"]["Header"]["nColumns"][()]))

            for i in tqdm(range(nPatterns), "writing to hdf in progress"):
                outImage = outImageDataList[i]
                outImageData = np.array(PIL.Image.fromarray(util.scale(outImage)).resize(outputResolution))
                if saveImagesToDisk:
                    px, py = util.indextoIJ(i, mapSize)
                    outFileName = os.path.join(outImDir, f'{ebsdDataPathInh5}_{i}_{px}_{py}.png')
                    original = hdfData[ebsdDataPathInh5]["EBSD"]["Data"]["Pattern"][i]
                    groundTruthIm = hdfData[groundTruthData]["EBSD"]["Data"]["Pattern"][i][()]
                    util.plotComparitiveImages(images=[original, outImageData, groundTruthIm],
                                               titles=["Input", "Inference", "ground truth"],
                                               mainTitle=f"{i} : ({px},{py})", filePath=outFileName)
                hdfData[ebsdDataPathInh5]["EBSD"]["Data"]["Pattern"][i] = outImageData
            logging.info(f"Completed writing the data in File : {outHdfName}")


if "train" in a.mode:
    with summary_writer.as_default():
        tf.summary.text("run_params", pretty_json(tensorBoardText),
                        step=0)
    fit(train_dataset, EPOCHS, test_dataset, plotImages=False)


elif "inference" in a.mode:  #### now for the inference:
    #### the inference part of the code which was used for kikuchi pattrern porocessing is now being
    #### delegated to another dedciated function (which is not tested) on 07/03/2023. IN case you
    ### you need to recover this part of the code get the git code before this date.!!!!
    if "kikuchi" in a.imDataSetType:
        kikuchi_inference()
    else:
        logging.info(f"Running inferece for normal images.")
        # checkpoint_dir =
        checkpoint_dir = a.checkpoint
        # input_image, real_image = load(inputImagePath, mode='inference')
        inferenceDataPath = a.input_dir
        infer_dataset = tf.data.Dataset.list_files(inferenceDataPath + '/*.png', shuffle=False)
        infer_dataset = infer_dataset.map(load_image_inference)  ####
        infer_bathes = infer_dataset.batch(BATCH_SIZE)
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        modelName = pathlib.PurePath(checkpoint_dir)
        if len(a.output_dir)>0:
            inferOutDir=a.output_dir
        else:
            logging.info(f"Did not get the inference out path specified. hence resorting to default folder inside input dir :{inferenceDataPath}")
            inferOutDir = os.path.join(inferenceDataPath, 'predictions' + modelName.name)
        if not os.path.exists(inferOutDir):
            os.mkdir(inferOutDir)
        count = 0
        for inp, tar, imageFile in infer_dataset.take(-1):
            if "numerical2Img" in a.imDataSetType and "LPBF_TemperatureData" in a.imDataSubSetType:
                test_imageFile_local = "".join([i.decode('ascii') for i in imageFile.numpy().tolist()])

            else:
                test_imageFile_local = imageFile.numpy().decode('utf-8')

            #imageFile = imageFile.numpy().decode('ascii')
            prediction = generate_images(generator, inp, tar, epoch=1, currentStep=0,
                                         outDir=inferOutDir, imageFile=test_imageFile_local, plotId=count,
                                         saveIndividualPredictions=True)

            count += 1
        logging.info(f"Completed inferene. Writing the image in dir {inferOutDir}")







else:
    raise ValueError(f"only allowed modes are train, test or inference: but was given {a.mode}")