import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model


# creating a list of all the foods, in the argument I put the path to the folder that has all folders for food
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            list_.append(name)
    return list_


# loading the model I trained and finetuned
my_model = load_model('model_trained.h5', compile=False)
food_list = create_foodlist("food-101/images")

pred_value = []


# function to help in predicting classes of new images loaded from my computer(for now)
def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        # Returns the indices of the maximum values along an axis,
        # In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        print(index)
        food_list.sort()
        pred_value = food_list[index]
        print(pred_value)
    if show:
        plt.imshow(img[0])
        plt.axis('off')
        plt.title(pred_value)
        plt.show()
    print(fetch_calories(pred_value))


# Function to get the calories, fetched via web scraping per 100 grams

def fetch_calories(prediction):
    url = 'https://www.google.com/search?&q=calories in ' + prediction
    # print(prediction)
    req = requests.get(url).text
    scrap = BeautifulSoup(req, 'html.parser')
    # print(scrap)
    calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    # print(calories)
    return calories


# add the images you want to predict into a list (these are in the WD)
images = ['15074.jpg']

print("IMAGE UPLOADED")
predict_class(my_model, images, True)
