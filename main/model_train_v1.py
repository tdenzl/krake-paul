import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
import glob


class ModelV1:

    def __init__(self):
        self.load_timestamp = str(datetime.now())

    @classmethod
    def train(cls):
        df_ingestion = cls._read_parquet('./data/gold/model_ingestion_v1/*')
        labels = tf.keras.utils.to_categorical(df_ingestion["outcome"])
        df_features = df_ingestion.drop(columns=["outcome"])







    @classmethod
    def _read_parquet(cls, base_path):
        files = glob.glob(base_path)
        df_list = []
        for file in files:
            df_tmp = pd.read_parquet(file)
            df_list.append(df_tmp)
        if len(df_list) == 0:
            print(base_path, " was None")
            return None
        df = pd.concat(df_list)
        print("finished reading ", base_path, " with ", len(df.index), " records")
        return df

    @classmethod
    def _write_parquet(cls, df, file_path):
        df = df.drop_duplicates()
        df["modify_timestamp"] = str(datetime.now())
        df.to_parquet(file_path, index=False)
        print("finished writing ", file_path, " with ", len(df.index), " records")


    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


    @classmethod
    def get_model():
        """
        Returns a compiled convolutional neural network model. Assume that the
        `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
        The output layer should have `NUM_CATEGORIES` units, one for each category.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Flatten())

        #model.add(tf.keras.layers.InputLayer(input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))

        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


        #model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        #model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        #model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

        return model