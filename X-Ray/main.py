import os
import matplotlib.pyplot as plt
import cv2
import glob
import img as img
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps
import tflearn
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.python.keras.layers.core import Dropout
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

image_list = []

###################################################################
######################### Preproseccing ############################
## readFileImages Function
def readFileImages(strFolderName):
    print(strFolderName)

    st = os.path.join(strFolderName, "*.jpg")

    for filename in glob.glob(st):
        image_list.append(filename)

## Display Function
def display(img_path):
    dpi = 80
    img_data = plt.imread(img_path)

    height, width = img_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(img_data, cmap='gray')
    plt.show()

## Crop Function
def cropImage(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # threshold
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

    # apply open morphology
    # kernel = np.ones((5,5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # get bounding box coordinates from largest external contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big_contour)

    # crop image contour image
    result = img.copy()
    result = result[y:y + h, x:x + w]

    # write result to disk
    ##cv2.imwrite("E:/sonar_thresh.jpg", thresh)
    ##cv2.imwrite("E:/sonar_morph.jpg", morph)
    cv2.imwrite(img_path,result)

    # display results
    ##cv2.imshow("THRESH", thresh)
    ##cv2.imshow("MORPH", morph)
    #cv2.imshow("CROPPED", result)  ##########to show output
    #cv2.waitKey(0)
    ##cv2.destroyAllWindows()

# Add border Function
def add_border(input_image, output_image, border):
    img = Image.open(input_image)

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(img, border=border)
    else:
        raise RuntimeError('Border is not an integer or tuple!')
    bimg.save(output_image)


# Histogram Graph Function
def HistogramGraph(img_path):
    img = cv2.imread(img_path, 0)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

###################################################################
###################### Classification ############################



IMG_SIZE = 224
LR = 0.001
MODEL_NAME = 'shoulder-implants-cnn'
TRAIN_DIR='/content/sample_data/codeai'
TEST_DIR='/content/sample_data/test'
def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'eld':
        return np.array([1,0,0,0])
    elif word_label == 'puy':
        return np.array([0,1,0,0])
    elif word_label == 'ier':
        return np.array([0,0,1,0])
    elif word_label == 'mer ':
        return np.array([0,0,0,1])


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data= []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data),create_label(img)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    train_data =np.load('train_data.npy',allow_pickle=True)
    #train_data = create_train_data()
else: # If dataset is not created:
    train_data = create_train_data()

if (os.path.exists('test_data.npy')):
    test_data =np.load('test_data.npy')
else:
    test_data = create_test_data()


train = train_data
test = test_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input,244, 3 , activation='relu')
pool1 = max_pool_2d(conv1, 2)

conv2 = conv_2d(pool1, 32, 3, activation='relu')
tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input,244, 3 , activation='relu')
pool1 = max_pool_2d(conv1, 2,2)

conv2 = conv_2d(pool1, 32, (3,3), activation='relu')
conv2 = conv_2d(pool1, 32, (3,3), activation='relu')
pool2 = max_pool_2d(conv2, 2,2)

conv3 = conv_2d(pool2, 64, (3,3), activation='relu')
conv3 = conv_2d(pool2, 64, (3,3), activation='relu')
pool3 = max_pool_2d(conv3, 2)

conv4 = conv_2d(pool3, 128, (3,3), activation='relu')
conv4 = conv_2d(pool3, 128, (3,3), activation='relu')
pool4 = max_pool_2d(conv4, 2)

conv5 = conv_2d(pool4, 256,3, activation='relu')
pool5 = max_pool_2d(conv5, 2)
fully_layer= fully_connected(pool4 , 256 , activation='relu' )
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 4, activation='softmax')
cnn_layers = regression(cnn_layers, optimizer='name',metric='accuracy', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
print(X_train.shape)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')








###################### Main ###############################
readFileImages('C:/Users/Yara/PycharmProjects/aixray/data')
for i in range (len(image_list)):
    im = Image.open(image_list[i])
    ##im.show()
    #display(image_list[i])
    cropImage(image_list[i])
    #display(image_list[i])
    add_border(image_list[i], output_image=image_list[i],border=10)
    #display(image_list[i])
    HistogramGraph(image_list[i])