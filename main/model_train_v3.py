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
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

N_CLASS = 2
N_FEATURES = 0
TEST_SIZE = 0.3
EPOCHS = 100

class ModelV3:

    def __init__(self):
        self.load_timestamp = str(datetime.now())

    @classmethod
    def train(cls):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        df_ingestion = cls._read_parquet('./data/gold/model_ingestion/*')
        label_column = "corner_over_10_5"
        labels = df_ingestion[label_column]
        labels = tf.keras.utils.to_categorical(labels)

        drop_cols = ["game_id", "corner_over_11_5", "corner_over_10_5", "corner_over_9_5", "corner_over_8_5", "outcome","total_corners"]
        df_features = df_ingestion.drop(columns=drop_cols)
        print(df_features.columns)

        for column in df_features.columns:
            df_features[column] = df_features[column].astype('float32')
        N_FEATURES = len(df_features.columns)
        #df_features = np.asarray(df_features).astype('float32')
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(df_features)

        x_train, x_test, y_train, y_test = train_test_split(
            df_features, labels, test_size=TEST_SIZE
        )

        print("N_CLASS=", N_CLASS)
        print("N_FEATURES=", N_FEATURES)
        print("TEST_SIZE=", TEST_SIZE)
        print("EPOCHS=", EPOCHS)

        # Get a compiled neural network
        model = cls._get_model(normalizer)

        print(len(x_train))
        print(len(y_train))

        # Fit model on training data
        model.fit(x_train, y_train, epochs=EPOCHS)

        # Evaluate neural network performance
        model.evaluate(x_test, y_test, verbose='auto')

        # Save model to file
        model.save("./models/krake_paul_v3")

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


    @classmethod
    def _get_model(cls, normalizer):
        """
        Returns a compiled convolutional neural network model. Assume that the
        `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
        The output layer should have `NUM_CATEGORIES` units, one for each category.
        """

        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(N_CLASS, activation="softmax")
        ])

        model.compile(optimizer='adam',
                      loss="binary_crossentropy",
                      metrics=['accuracy'])

        return model

    @classmethod
    def predict(cls):
        model = tf.keras.models.load_model("./models/krake_paul_v3")

        df_ingestion = cls._read_parquet('./data/gold/model_ingestion/*')

        drop_cols = ["game_id", "corner_over_11_5", "corner_over_10_5", "corner_over_9_5", "corner_over_8_5", "outcome","total_corners"]
        df_features = df_ingestion.drop(columns=drop_cols)

        for column in df_features.columns:
            df_features[column] = df_features[column].astype('float32')

        # Evaluate neural network performance
        predictions = model.predict(df_features)
        prediction_list = []
        for prediction in predictions:
            prediction_list.append({"prediction_over":prediction[0], "prediction_under":prediction[1]})
        df_predictions = pd.DataFrame(prediction_list)
        df_predictions["game_id"] = df_ingestion["game_id"]

        df_match = cls._read_parquet('./data/silver/team_elo/*')[["game_id","season_start","matchday","home_elo","away_elo","home_team","away_team","home_goals","away_goals"]]

        df_match_stats = cls._read_parquet('./data/silver/team_stats/*').groupby(['game_id']).agg(total_corners=('corners', np.sum)).reset_index()


        df_predictions = df_predictions.merge(df_match, on=["game_id"], how="inner")
        df_predictions = df_predictions.merge(df_match_stats, on=["game_id"], how="inner")


        df_predictions['odd_over'] = 1.06/ df_predictions["prediction_over"]
        df_predictions['odd_under'] = 1.06 / df_predictions["prediction_under"]

        cls._write_parquet(df_predictions, './data/gold/predictions/model_predictions_v3.parquet')