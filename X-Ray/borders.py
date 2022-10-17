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

image_list = []


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

readFileImages('C:/Users/Yara/PycharmProjects/aixray/data')
for i in range(len(image_list)):
        im = Image.open(image_list[i])
        ##im.show()
        # display(image_list[i])
        cropImage(image_list[i])
        # display(image_list[i])
        add_border(image_list[i], output_image=image_list[i], border=10)
        # display(image_list[i])
        HistogramGraph(image_list[i])