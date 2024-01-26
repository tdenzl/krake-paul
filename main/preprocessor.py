import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Pool
import hashlib
from .job_bookmark import JobBookmark
from copy import deepcopy
from tqdm import tqdm

import os
import glob


class Preprocessor:

    def __init__(self):
        self.load_timestamp = str(datetime.now())

    @classmethod
    def preprocess_table(cls, table_name):
        if table_name == "coaches": cls._preprocess_coaches()
        if table_name == "match_info": cls._preprocess_match_info()
        if table_name == "team_stats": cls._preprocess_team_stats()
        if table_name == "player_stats": cls._preprocess_player_stats()
        if table_name == "player_ratings": cls._preprocess_player_ratings()

    @classmethod
    def _read_parquet(cls, base_path):
        files = glob.glob(base_path)
        df_list = []
        for file in files:
            df_tmp = pd.read_parquet(file)
            df_list.append(df_tmp)
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
    def _preprocess_coaches(cls):
        df_coaches = cls._read_parquet('./data/bronze/coaches/*')
        df_coaches["coach_name"] = df_coaches["coach_name"].str.split(pat="/").str[0].str.strip()
        df_coaches["indicator"] = df_coaches["indicator"].str.strip()

        dtype_dict = {'matchday': 'Int32'}
        df_coaches = df_coaches.astype(dtype_dict)
        cls._write_parquet(df_coaches, './data/silver/coaches/coaches.parquet')

    @classmethod
    def _preprocess_match_info(cls):
        df_match_info = cls._read_parquet('./data/bronze/match_info/*')
        df_match_info["kick_off_time"] = df_match_info["kick_off_time"].str.replace(',', '').str.replace('.', '-')
        df_match_info["kick_off_date"] = pd.to_datetime(df_match_info["kick_off_time"],format='%d-%m-%Y %H:%M', errors='coerce')
        df_match_info["kick_off_time"] = df_match_info["kick_off_date"].dt.strftime('%H:%M')
        df_match_info["referee"] = df_match_info["referee"].str.split(pat="/").str[0].str.strip()

        cls._write_parquet(df_match_info, './data/silver/match_info/match_info.parquet')

    @classmethod
    def _preprocess_team_stats(cls):
        df_team_stats = cls._read_parquet('./data/bronze/team_stats/*')
        df_team_stats = df_team_stats.rename(columns={'dribble_reatio': 'dribble_ratio'})
        ratio_columns = ["air_tackle_ratio", "dribble_ratio", "pass_ratio", "possession", "tackle_ratio", "cross_ratio"]
        for column in ratio_columns:
            df_team_stats[column] = df_team_stats[column].str.replace('%', '').astype(float) / 100

        df_team_stats["position"] = df_team_stats["position"].str.strip().str.replace('. Platz', '')
        df_team_stats["distance"] = df_team_stats["position"].str.replace('km', '').str.strip().str.replace(',', '.')
        df_team_stats["team_name"] = df_team_stats["team_name"].str.strip()

        int_columns = ['match_day', 'ht_goals', 'position', 'goals', 'shots_on_goal', 'total_passes', 'crosses',
                       'dribblings', 'tackles', 'air_tackles', 'fouls', 'got_fouled', 'offside', 'corners']
        for column in int_columns:
            df_team_stats[column] = np.floor(pd.to_numeric(df_team_stats[column], errors='coerce')).astype('Int64')

        float_columns = ['distance']
        for column in float_columns:
            df_team_stats[column] = np.floor(pd.to_numeric(df_team_stats[column], errors='coerce')).astype('float')

        cls._write_parquet(df_team_stats, './data/silver/team_stats/team_stats.parquet')

    @classmethod
    def _preprocess_player_stats(cls):
        df_player_stats = cls._read_parquet('./data/bronze/player_stats/*')
        df_player_stats["cum_yellow_cards"] = df_player_stats["card_description"]\
            .str.replace('Gelbe Karte', '') \
            .str.replace('(', '') \
            .str.replace(')', '') \
            .str.replace('.', '').str.strip()

        minute_indicator_columns = ["card_time","sub_in","sub_out"]
        for column in minute_indicator_columns:
            df_player_stats[column] = df_player_stats[column].str.split(pat="'").str[0].str.strip().astype('float')/90
        df_player_stats["player_name"] = df_player_stats["player_name"].str.split(pat="/").str[0].str.strip()

        int_columns = ['goals', 'matchday', 'red_card','yellow_card', 'yellow_red_card', 'cum_yellow_cards']
        for column in int_columns:
            df_player_stats[column] = np.floor(pd.to_numeric(df_player_stats[column], errors='coerce')).astype('Int64')

        cls._write_parquet(df_player_stats, './data/silver/player_stats/player_stats.parquet')

    @classmethod
    def _preprocess_player_ratings(cls):
        df_player_ratings = cls._read_parquet('./data/bronze/player_ratings/*')

        month_dict = {"Jan.":1,"Feb.":2,"March":3,"April":4,"May":5,"June":6,"June":7,"Aug.":8,"Sept.":9,"Oct.":10,"Nov.":11,"Dec.":12}

        df_player_ratings["birth_date"] = df_player_ratings["birth_date"].str.replace(', ', '-')
        for month_string, month_num in month_dict.items():
            df_player_ratings["birth_date"] = df_player_ratings["birth_date"].str.replace(month_string, str(month_num)+'-')
        df_player_ratings["birth_date"] = pd.to_datetime(df_player_ratings["birth_date"], format='%m-%d-%Y', errors='coerce')

        df_player_ratings["height"] = df_player_ratings["height"] \
            .str.replace(' cm', '').str.strip()

        df_player_ratings["weight"] = df_player_ratings["weight"] \
            .str.replace(' kg', '').str.strip()

        df_player_ratings["name"] = df_player_ratings["name"].str.strip()
        df_player_ratings["nationality_id"] = df_player_ratings["nationality_id"].str.split(pat="=").str[1].str.strip()

        df_player_ratings["preferred_foot"] = df_player_ratings["preferred_foot"].str.replace('Left', '0').str.replace('Right', '1')
        df_player_ratings["preferred_foot"] = df_player_ratings["preferred_foot"]
        pos_dict = {"x":{"LF":-0.5,"LW":-1,"LM":-1,"LWB":-1,"LB":-0.7,"RF":0.5,"RW":1,"RM":1,"RWB":1,"RB":0.7,"ST":0,"CF":0,"CAM":0,"CM":0,"CDM":0,"CB":0,"GK":0},
                    "y":{"LF":3,"LW":2.5,"LM":2,"LWB":1.5,"LB":1,"RF":3,"RW":2.5,"RM":2,"RWB":1.5,"RB":1,"ST":3,"CF":2.7,"CAM":2.5,"CM":2,"CDM":1.5,"CB":1,"GK":0}}

        position_weights = [0.53,0.27,0.13,0.07]
        position_weights_cum = {1:0.53,2:0.8,3:0.93,4:1.00}

        df_player_ratings["position_x"] = 0
        df_player_ratings["position_y"] = 0

        position_cols = ["preferred_position_1","preferred_position_2","preferred_position_3","preferred_position_4"]
        df_player_ratings["position_count"] = len(position_weights) - df_player_ratings[position_cols].isna().sum(1)

        for w, weight in enumerate(position_weights):
            df_player_ratings["position_x"] = df_player_ratings["position_x"] + df_player_ratings["preferred_position_" + str(w+1)].apply(lambda x: pos_dict.get("x").get(x)).fillna(0.00) * weight
            df_player_ratings["position_y"] = df_player_ratings["position_y"] + df_player_ratings["preferred_position_" + str(w+1)].apply(lambda x: pos_dict.get("y").get(x)).fillna(0.00) * weight

        df_player_ratings["position_x"] = df_player_ratings["position_x"] / df_player_ratings["position_count"].apply(lambda x: position_weights_cum.get(x))
        df_player_ratings["position_y"] = df_player_ratings["position_y"] / df_player_ratings["position_count"].apply(lambda x: position_weights_cum.get(x))

        df_player_ratings["position_x"] = df_player_ratings["position_x"].round(2)
        df_player_ratings["position_y"] = df_player_ratings["position_y"].round(2)

        int_cols = ['fifa', 'player_id', 'height',
       'weight', 'preferred_foot', 'age',
       'weak_foot', 'skill_moves', 'ball_control', 'dribbling', 'slide_tackle',
       'stand_tackle', 'aggression', 'reactions', 'att_position',
       'interceptions', 'vision', 'composure', 'crossing', 'short_pass',
       'long_pass', 'acceleration', 'stamina', 'strength', 'balance',
       'sprint_speed', 'agility', 'jumping', 'heading', 'shot_power',
       'finishing', 'long_shots', 'curve', 'fk_acc', 'penalties', 'volleys',
       'gk_positioning', 'gk_diving', 'gk_handling', 'gk_kicking',
       'gk_reflexes', 'nationality_id']

        for column in int_cols:
            df_player_ratings[column] = np.floor(pd.to_numeric(df_player_ratings[column], errors='coerce')).astype('Int64')


        df_player_ratings = df_player_ratings.drop(columns = ["preferred_position_1","preferred_position_2","preferred_position_3","preferred_position_4"])
        cls._write_parquet(df_player_ratings, './data/silver/player_ratings/player_ratings.parquet')
