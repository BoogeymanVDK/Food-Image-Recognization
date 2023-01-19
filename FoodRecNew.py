# import os
# import shutil
# import stat
# import seaborn as sns
# import collections
# import h5py
# import numpy as np
# import tensorflow as tf
# import matplotlib.image as img
# import random
# import cv2
# import PIL
# import matplotlib.pyplot as plt
# import matplotlib.image as img
# from os import listdir
# from os.path import isfile, join
# from collections import defaultdict
# from ipywidgets import interact, interactive, fixed
# import ipywidgets as widgets
# from sklearn.model_selection import train_test_split
# from skimage.io import imread
# from keras.utils.np_utils import to_categorical
# from keras.applications.inception_v3 import preprocess_input
# from keras.models import load_model
# from shutil import copy
# from shutil import copytree, rmtree
# import tensorflow.keras.backend
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras import regularizers
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.regularizers import l2
# from tensorflow import keras
# from tensorflow.keras import models
#
# class_N = {}
# N_class = {}
# with open('Foods_DS/meta/classes.txt', 'r') as txt:
#     classes = [i.strip() for i in txt.readlines()]
#     class_N = dict(zip(classes, range(len(classes))))
#     N_class = dict(zip(range(len(classes)), classes))
#     class_N = {i: j for j, i in N_class.items()}
# class_N_sorted = collections.OrderedDict(sorted(class_N.items()))
# print(class_N)
#
#
# # Method to generate directory-file map.
# def gen_dir_file_map(path):
#     dir_files = defaultdict(list)
#     with open(path, 'r') as txt:
#         files = [i.strip() for i in txt.readlines()]
#         for f in files:
#             dir_name, id = f.split('/')
#             dir_files[dir_name].append(id + '.jpg')
#     return dir_files
#
#
# # Method to recursively copy a directory.
# def copytree(source, target, symlinks=False, ignore=None):
#     if not os.path.exists(target):
#         os.makedirs(target)
#         shutil.copystat(source, target)
#     data = os.listdir(source)
#     if ignore:
#         exclude = ignore(source, data)
#         data = [x for x in data if x not in exclude]
#     for item in data:
#         src = os.path.join(source, item)
#         dest = os.path.join(target, item)
#         if symlinks and os.path.islink(src):
#             if os.path.lexists(dest):
#                 os.remove(dest)
#             os.symlink(os.readlink(src), dest)
#             try:
#                 st = os.lstat(src)
#                 mode = stat.S_IMODE(st.st_mode)
#                 os.lchmod(dest, mode)
#             except:
#                 pass
#         elif os.path.isdir(src):
#             copytree(src, dest, symlinks, ignore)
#         else:
#             shutil.copy2(src, dest)
#
#
# # Train files to ignore.
# def ignore_train(d, filenames):
#     subdir = d.split('/')[-1]
#     train_dir_files = gen_dir_file_map('Foods_DS/meta/train.txt')
#     to_ignore = train_dir_files[subdir]
#     return to_ignore
#
#
# # Test files to ignore.
# def ignore_test(d, filenames):
#     subdir = d.split('/')[-1]
#     test_dir_files = gen_dir_file_map('Foods_DS/meta/test.txt')
#     to_ignore = test_dir_files[subdir]
#     return to_ignore
#
#
# # Method to load and resize images.
# def load_images(path_to_imgs):
#     resize_count = 0
#
#     invalid_count = 0
#     all_imgs = []
#     all_classes = []
#
#     for i, subdir in enumerate(listdir(path_to_imgs)):
#         imgs = listdir(join(path_to_imgs, subdir))
#         classN = class_N[subdir]
#         for img_name in imgs:
#             img_arr = cv2.imread(join(path_to_imgs, subdir, img_name))
#             img_arr_rs = img_arr
#             img_arr_rs = cv2.resize(img_arr, (200, 200), interpolation=cv2.INTER_AREA)
#             resize_count += 1
#             im_rgb = cv2.cvtColor(img_arr_rs, cv2.COLOR_BGR2RGB)
#             all_imgs.append(im_rgb)
#             all_classes.append(classN)
#
#     return np.array(all_imgs), np.array(all_classes)
#
#
# # Method to generate train-test files.
# def gen_train_test_split(path_to_imgs='Foods_DS/images', target_path='Foods_DS'):
#     copytree(path_to_imgs, target_path + '/train', ignore=ignore_test)
#     copytree(path_to_imgs, target_path + '/test', ignore=ignore_train)
#
#
# # Method to load train-test files.
# def load_train_test_data(path_to_train_imgs, path_to_test_imgs):
#     X_train, y_train = load_images(path_to_train_imgs)
#     X_test, y_test = load_images(path_to_test_imgs)
#     return X_train, y_train, X_test, y_test
#
#
# # Generate train-test files.
# if not os.path.isdir('Foods_DS/test') and not os.path.isdir('Foods_DS/train'):
#     gen_train_test_split()
#     len_train = len(os.listdir('Food_DS/train'))
#     len_test = len(os.listdir('Food_DS/test'))
#     print(len_train, len_test)
# else:
#     print('train and test folders already exists.')
#     len_train = len(os.listdir('Foods_DS/train'))
#     len_test = len(os.listdir('Foods_DS/test'))
#     print(len_train, len_test)
#
# # List of all the food classes.
# foods_sorted = sorted(os.listdir('Foods_DS/images'))
# foods_sorted
#
# # Display an image.
# testImg = imread('Food_DS/test/Apple/IMG_20220208_201524.jpg')
# print(testImg.shape)
# plt.imshow(testImg)
#
# X_train, y_train, X_test, y_test = load_train_test_data('FOOD_DS/train', 'Food_DS/test')
#
# tensorflow.keras.backend.clear_session()
#
# n_classes = 101
# batch_size = 16
# width, height = 200, 200
# train_data = 'FOOD_DS/train'
# test_data = 'FOOD_DS/test'
# train_samples = 75750
# test_samples = 25250
#
# train_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test_data_gen = ImageDataGenerator(rescale=1. / 255)
#
# train_gen = train_data_gen.flow_from_directory(train_data, target_size=(height, width), batch_size=batch_size,
#                                                class_mode='categorical')
#
# test_gen = test_data_gen.flow_from_directory(test_data, target_size=(height, width), batch_size=batch_size,
#                                              class_mode='categorical')
#
# inception = InceptionV3(weights='imagenet', include_top=False)
# layer = inception.output
# layer = GlobalAveragePooling2D()(layer)
# layer = Dense(128, activation='relu')(layer)
# layer = Dropout(0.2)(layer)
#
# predictions = Dense(n_classes, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(layer)
#
# model = Model(inputs=inception.input, outputs=predictions)
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# checkpointer = ModelCheckpoint(filepath='best_model_101class.hdf5', save_best_only=True)
# csv_logger = CSVLogger('history_101class.log')
#
# history_101class = model.fit(train_gen,
#                              steps_per_epoch=train_samples // batch_size,
#                              validation_data=test_gen,
#                              validation_steps=test_samples // batch_size,
#                              epochs=30,
#                              callbacks=[csv_logger, checkpointer])
#
# model.save('tarainedmodel.hdf5')
#
# # Plot training-accuracy & validation-accuracy.
# _ = plt.style.library['seaborn-darkgrid']
# _ = plt.title('FOOD_CLASSIFICATION-Inceptionv3')
# _ = plt.plot(history_101class.history['accuracy'], marker='o', linestyle='dashed')
# _ = plt.plot(history_101class.history['val_accuracy'], marker='x', linestyle='dashed')
# _ = plt.ylabel('Accuracy')
# _ = plt.xlabel('Epoch')
# _ = plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
# plt.show()
#
