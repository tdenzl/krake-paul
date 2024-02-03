import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import linear_model
import hashlib
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
import glob
import Levenshtein


class BetMaker:

    def __init__(self):
        self.load_timestamp = str(datetime.now())

    @classmethod
    def evaluate_bets(cls):
        #cls._load_fut_data()
        cls._match_bets()


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
    def _read_csv(cls, base_path):
        files = glob.glob(base_path, recursive=True)
        df_list = []
        for file in files:
            try:
                df_tmp = pd.read_csv(file, header=0, on_bad_lines='skip', encoding_errors='replace')
                df_list.append(df_tmp)
            except pd.errors.EmptyDataError:
                print("EmptyDataError for ", file)
            except Exception as e:
                print("Error for ", file, " : ", e)
        if len(df_list) == 0:
            print(base_path, " was None")
            return None
        df = pd.concat(df_list, axis=0, ignore_index=True)
        print("finished reading ", base_path, " with ", len(df.index), " records")
        return df

    @classmethod
    def _load_fut_data(cls):
        df_fut_data = cls._read_csv('./data/bronze/fut_data/**/*.csv')
        df_fut_data["Season"] = df_fut_data["Season"].str.split("/")[0]
        df_fut_data["BbAH"] = df_fut_data["BbAH"].str.extract('(\d+)', expand=False).astype(float)
        df_fut_data["BbAHh"] = df_fut_data["BbAHh"].str.extract('(\d+)', expand=False).astype(float)
        df_fut_data["PSCH"] = df_fut_data["PSCH"].str.extract('(\d+)', expand=False).astype(float)
        df_fut_data["B365CH"] = df_fut_data["B365CH"].str.extract('(\d+)', expand=False).astype(float)
        df_fut_data["BWCA"] = df_fut_data["BWCA"].str.extract('(\d+)', expand=False).astype(float)
        cols = df_fut_data.columns
        for column in cols:
            df_fut_data = df_fut_data.rename(columns={column: column.lower().replace(" ","_").replace(":","")})
        cls._write_parquet(df_fut_data, './data/silver/fut_data/fut_data.parquet')
        return

    @classmethod
    def _match_bets(cls):
        df_predictions = cls._read_parquet('./data/gold/predictions/model_predictions_v3.parquet')
        df_match_info = cls._read_parquet('./data/silver/match_info/*')[["game_id","kick_off_date"]]


        df_match_info['kick_off_date'] = df_match_info['kick_off_date'].dt.date

        df_predictions = df_predictions.merge(df_match_info, on=["game_id"], how="inner")

        df_fut_data = cls._read_parquet('./data/silver/fut_data/*')

        df_fut_data["kick_off_date"] = pd.to_datetime(df_fut_data["date"], format='mixed')
        df_fut_data["kick_off_date"] = df_fut_data["kick_off_date"].dt.date
        df_fut_data["game_id_string"] = df_fut_data["league"].astype(str) + df_fut_data["kick_off_date"].astype(str) + df_fut_data["home"].astype(str) + df_fut_data["away"].astype(str)
        print(df_fut_data["game_id_string"])

        #df_fut_data["game_id_fut"] = str(hashlib.md5(df_fut_data["game_id_string"]).hexdigest())
        df_fut_data['game_id_fut'] = [hashlib.md5(val.encode('utf-8')).hexdigest() for val in df_fut_data['game_id_string']]
        df_fut_data = df_fut_data.drop_duplicates('game_id_fut')
        #print(df_fut_data.groupby(["league"]).count().reset_index())

        print(df_fut_data["game_id_fut"])

        return

    @classmethod
    def _fuzzy_merge(cls, df_matches_kicker, df_matches_fut, threshold=85):
        mapping_list = []
        df_matches_kicker = df_matches_kicker[["game_id","kick_off_date","home_team","away_team"]]
        matches_kicker = df_matches_kicker.to_dict('records')


        df_matches_fut = df_matches_fut.to_list()

        for kicker_player in tqdm(players_kicker):
            kicker_name = kicker_player.get("player_name")
            appearances = kicker_player.get("count")
            if kicker_name is None: continue
            max_score = -1
            max_name = ''
            for fifa_name in players_fifa:
                if fifa_name is None: continue
                score = Levenshtein.ratio(kicker_name, fifa_name) * 100
                if (score > threshold) & (score > max_score):
                    # print(score)
                    max_name = fifa_name
                    max_score = score
                    if score == 100: break
            if max_name == '':
                mapping_list.append({"kicker_name": kicker_name, "fifa_name": None, "score": None, "appearances":appearances})
            else:
                mapping_list.append({"kicker_name": kicker_name, "fifa_name": max_name, "score": max_score, "appearances":appearances})

        for kicker_player in tqdm(mapping_list):
            if kicker_player.get("fifa_name") is not None: continue
            for fifa_name in players_fifa:
                kicker_split = kicker_player.get("kicker_name").split(" ")
                last_name = kicker_split[len(kicker_split)-1]
                if last_name == fifa_name:
                    kicker_player["fifa_name"] = last_name

        return pd.DataFrame(mapping_list).sort_values(by=["appearances"], ascending = False)
