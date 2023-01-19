# import numpy as np
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import keras
#
# # from keras.applications.
#
# train_dir = Path('Foods_DS')
# train_filepath = list(train_dir.glob(r'**/*.*'))
#
# test_dir = Path('Foods_DS')
# test_filepath = list(test_dir.glob(r'**/*.*'))
#
# val_dir = Path('Foods_DS')
# val_filepath = list(test_dir.glob(r'**/*.*'))
#
#
# def image_processing(filepath):
#     labelss0 = [str(filepath[i]).split("/")[0]
#                 for i in range(len(filepath))]
#
#     filepath = pd.Series(filepath, name='Filepath').astype(str)
#     labelss0 = pd.Series(labelss0, name='Label')
#
#     # Concatenating the filepath and labels
#     df = pd.concat([filepath, labelss0], axis=1)
#
#     df = df.sample(frac=1).reset_index(drop=True)
#
#     return df
#
#
# train_df = image_processing(train_filepath)
# test_df = image_processing(test_filepath)
# val_df = image_processing(val_filepath)
#
# print("Training set \n")
# print(f'Number of pictures:{train_df.shape[0]}\n')
# print(f'Number of diff labels: {len(train_df.Label.unique())}\n')
# print(f'Labels: {train_df.Label.unique()}')
#
# train_df.head(5)
#
# # df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()
# # fig, axes = plt.subplot(nrows=6, ncol=6, figsize=(8, 7)),
# # subplot_kw = {'xticks': [], 'yticks': []}
#
# # Create a DataFrame with one Label of each category
# df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()
#
# train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
# )
#
# test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
# )
#
# train_image = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=0,
#     rotation_range=30,
#     zoom_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     sheer_range=0.15,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# val_image = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=True,
#     seed=0,
#     rotation_range=30,
#     zoom_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     sheer_range=0.15,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# test_image = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=(224, 224),
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     shuffle=False
# )
# # x1 = tf.cast(train_image, tf.float32)
# # # y = tf.cast(x1, tf.float32)
# # y = tf.keras.applications.mobilenet.preprocess_input(x1)
# # core = tf.keras.applications.MobileNet()
# # y = core(y)
# # model = tf.keras.Model(inputs=[train_image], output=[y])
# #
# #
# # # pretrain_model = tf.keras.applications.MobileNetV2()
# # # pretrain_model.trainable = False
# # #
# # # x = tf.keras.layers.Dense(128, activation='relu')
# # # #(pretrain_model.decode_predictions)
# # #
# # # inputs = pretrain_model.preprocess_input(x1)
# # # # inputs = pretrain_model.preprocess_input()
# # #
# # # x = tf.keras.layers.Dense(128, activation='relu')
# # # #(pretrain_model.output())
# # # x = tf.keras.layers.Dense(128, activation='relu')
# #
# # outputs = tf.keras.layers.Dense(36, activation='softmax')
# # #
# # # model = tf.keras.Model(input=[train_image], output=outputs)
#
# pretrained_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
# pretrained_model.trainable = False
#
# inputs = pretrained_model.input
#
# x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
#
# outputs = tf.keras.layers.Dense(36, activation='softmax')(x)
# # outputs = keras.layers.Flatten()(x)
#
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# # out = keras.layers.Flatten()(x)
#
# from tensorflow.python.keras.engine import data_adapter
# adapter = data_adapter.KerasSequenceAdapter(x, y=None)
# print(adapter)
#
# history = model.fit(
#     train_image,
#     val_image,
#     validation_data=val_image,
#     batch_size=32,
#     epochs=5,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=2,
#             restore_best_weights=True
#         )
#     ]
# )
#
# predict = model.predict(test_image)
# predict = np.argmax(predict, axis=1)
# labels = train_image.class_indices
# labels = dict((v, k) for k, v in labels.items())
# pred1 = [labels[k] for k in predict]
# pred1
#
#
# def output(location):
#     img0 = load_img(location, target_size=(224, 224, 3))
#     img1 = img_to_array(img0)
#     img2 = img1 / 225
#     img3 = np.expand_dims(img2, [0])
#     answer = model.predict(img3)
#     y_class = answer.argmax(axis=-1)
#     y = " ".join(str(x) for x in y_class)
#     y = int(y)
#     res = labels[y]
#     return res
#
#
#
# img = output('images')