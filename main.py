import os

import requests

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Getting input image

filename = '142995.jpg'

# Loading the image

from tensorflow.keras.preprocessing import image

# setting the image size to 224 because most of the Ai algo are trained on 224 size

img = image.load_img(filename, target_size=(224, 224))
plt.imshow(img)

# Loading the deep learning model
# deep learning architecture is already pre-trained

mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
resize_img = image.img_to_array(img)  # convert to array with 3 channels
final_image = np.expand_dims(resize_img, axis=0)  # Expanding the array at position 0
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image)

# preprocessing is done

predictions = mobile.predict(final_image)
from tensorflow.keras.applications import imagenet_utils

results = imagenet_utils.decode_predictions(predictions)  # Decodes the prediction of an ImageNet model
# print(results)
for result in results[0]:
    print( result[1],"Accuracy is", result[2] )
    exit()


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
