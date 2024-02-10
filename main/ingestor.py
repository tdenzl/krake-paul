import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import linear_model
from .elo_calculator import EloCalculator

pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
import glob
import Levenshtein
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class Ingestor:

    def __init__(self):
        self.load_timestamp = str(datetime.now())

    @classmethod
    def create_ingestion_data(cls):
        feature_columns = json.load(open('./config/mapping_features_v3.json', 'r'))

        df_match_info = cls._read_parquet('./data/silver/match_info/*')[feature_columns.get("df_match_info")]

        df_coach_elo = cls._read_parquet('./data/silver/coach_elo/*')[feature_columns.get("df_coach_elo")].rename(columns={"home_elo": "home_coach_elo", "away_elo": "away_coach_elo"})

        df_referee_profiles = cls._read_parquet('./data/silver/referee_profiles/*')[feature_columns.get("df_referee_profiles")]

        df_match_stats = cls._read_parquet('./data/silver/team_stats/*').groupby(['game_id']).agg(total_corners=('corners', np.sum)).reset_index()


        df_team_profiles_lin_reg = cls._read_parquet('./data/silver/team_profiles_lin_reg/*').drop(columns=["team_name"])
        df_team_profiles_lin_reg_home = df_team_profiles_lin_reg[df_team_profiles_lin_reg["indicator"]=="home"].drop(columns=["indicator"])
        df_team_profiles_lin_reg_away = df_team_profiles_lin_reg[df_team_profiles_lin_reg["indicator"]=="away"].drop(columns=["indicator"])

        for column in df_team_profiles_lin_reg_home.columns:
            if column in ["game_id","indicator"]: continue
            df_team_profiles_lin_reg_home = df_team_profiles_lin_reg_home.rename(columns={column:column+"_home"})
            df_team_profiles_lin_reg_away = df_team_profiles_lin_reg_away.rename(columns={column:column+"_away"})

        df_player_elo = cls._read_parquet('./data/silver/player_elo/*')[feature_columns.get("df_player_elo")]
        df_player_elo = df_player_elo.groupby(['game_id', 'indicator']).agg(
            player_elo_mean=('old_player_elo', np.mean),
            player_elo_stdev=('old_player_elo', np.std)).reset_index()
        df_player_elo_home = df_player_elo[df_player_elo["indicator"] == "home"].drop(columns=["indicator"])
        df_player_elo_away = df_player_elo[df_player_elo["indicator"] == "away"].drop(columns=["indicator"])
        for column in df_player_elo_home.columns:
            if column in ["game_id"]: continue
            df_player_elo_home = df_player_elo_home.rename(columns={column: column+"_home"})
            df_player_elo_away = df_player_elo_away.rename(columns={column: column+"_away"})


        df_relationships = cls._read_parquet('./data/silver/relationships/*')
        df_team_elo = cls._read_parquet('./data/silver/team_elo/*')[feature_columns.get("df_team_elo")]

        df_team_elo["goal_diff"] = df_team_elo["home_goals"] - df_team_elo["away_goals"]
        df_team_elo['outcome'] = np.where(df_team_elo['goal_diff'] < 0, 2,
                                            np.where(df_team_elo['goal_diff'] > 0, 0, 1))



        df_team_elo = df_team_elo.drop(columns=["home_goals","away_goals","goal_diff"])

        df_feature = df_match_info.merge(df_coach_elo, on = ["game_id"], how = "inner")
        df_feature = df_feature.merge(df_referee_profiles, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_team_profiles_lin_reg_home, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_team_profiles_lin_reg_away, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_relationships, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_team_elo, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_player_elo_home, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_player_elo_away, on=["game_id"], how="inner")
        df_feature = df_feature.merge(df_match_stats, on=["game_id"], how="inner")

        df_feature['corner_over_11_5'] = np.where(df_feature['total_corners'] > 11.5, 1, 0)
        df_feature['corner_over_10_5'] = np.where(df_feature['total_corners'] > 10.5, 1, 0)
        df_feature['corner_over_9_5'] = np.where(df_feature['total_corners'] > 9.5, 1, 0)
        df_feature['corner_over_8_5'] = np.where(df_feature['total_corners'] > 8.5, 1, 0)

        for regression_type in feature_columns.get("reg_cols"):
            for indicator in ["home", "away"]:
                if indicator == "home": df_feature["exp_"+regression_type+"_"+indicator] = (df_feature[regression_type+"_coefficient_"+indicator] * (df_feature["home_elo"] - df_feature["away_elo"])) + df_feature[regression_type+"_intercept_"+indicator]
                if indicator == "away": df_feature["exp_"+regression_type+"_"+indicator] = (df_feature[regression_type+"_coefficient_"+indicator] * (df_feature["away_elo"] - df_feature["home_elo"])) + df_feature[regression_type+"_intercept_"+indicator]
                df_feature = df_feature.drop(columns=[regression_type+"_coefficient_"+indicator, regression_type+"_intercept_"+indicator])

        df_feature["kick_off_seconds"] = df_feature["kick_off_time"].str.split(":").str[0].astype(float) + df_feature["kick_off_time"].str.split(":").str[0].astype(float) * 60
        df_feature = df_feature.drop(columns=feature_columns.get("drop"))
        for column in df_feature.columns:
            df_feature = df_feature[df_feature[column].notna()]

        #df_feature["outcome"] = tf.keras.utils.to_categorical(df_feature["outcome"])

        df_feature["weekday"] = tf.keras.utils.to_categorical(df_feature["weekday"].factorize()[0])
        df_feature["kick_off_date_home"] = pd.to_numeric(df_feature["kick_off_date_home"])
        df_feature = df_feature[df_feature["match_day"]>=5]
        cls._write_parquet(df_feature, './data/gold/model_ingestion/model_ingestion.parquet')


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
        df.to_parquet(file_path, index=False)
        print("finished writing ", file_path, " with ", len(df.index), " records")
