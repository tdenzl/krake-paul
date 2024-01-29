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
        if table_name == "team_elo": cls._calculate_elos()
        if table_name == "team_lin_regs": cls._calculate_team_linear_regressions()
        if table_name == "player_mapping": cls._player_mapping_kicker_fifa()
        if table_name == "team_fifa_rating": cls._team_fifa_rating()


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
    def _preprocess_coaches(cls):
        df_coaches = cls._read_parquet('./data/bronze/coaches/*')
        df_coaches["coach_name"] = df_coaches["coach_name"].str.split(pat="/").str[0].str.strip()
        df_coaches["indicator"] = df_coaches["indicator"].str.strip()

        dtype_dict = {'matchday': 'Int32'}
        df_coaches = df_coaches.astype(dtype_dict)
        df_coaches["season_start"] = df_coaches["season"].str.split(pat="-").str[0].str.strip()

        cls._write_parquet(df_coaches, './data/silver/coaches/coaches.parquet')

    @classmethod
    def _preprocess_match_info(cls):
        df_match_info = cls._read_parquet('./data/bronze/match_info/*')
        df_match_info["kick_off_time"] = df_match_info["kick_off_time"].str.replace(',', '').str.replace('.', '-')
        df_match_info["kick_off_date"] = pd.to_datetime(df_match_info["kick_off_time"], format='%d-%m-%Y %H:%M',
                                                        errors='coerce')
        df_match_info["kick_off_time"] = df_match_info["kick_off_date"].dt.strftime('%H:%M')
        df_match_info["referee"] = df_match_info["referee"].str.split(pat="/").str[0].str.strip()
        df_match_info["season_start"] = df_match_info["season"].str.split(pat="-").str[0].str.strip()

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

        df_team_stats["season_start"] = df_team_stats["season"].str.split(pat="-").str[0].str.strip()

        df_elo = cls._calculate_elos(df_team_stats)

        cls._write_parquet(df_elo, './data/silver/team_elo/team_elo.parquet')
        cls._write_parquet(df_team_stats, './data/silver/team_stats/team_stats.parquet')

    @classmethod
    def _preprocess_player_stats(cls):
        df_player_stats = cls._read_parquet('./data/bronze/player_stats/*')
        df_player_stats["cum_yellow_cards"] = df_player_stats["card_description"] \
            .str.replace('Gelbe Karte', '') \
            .str.replace('(', '') \
            .str.replace(')', '') \
            .str.replace('.', '').str.strip()

        minute_indicator_columns = ["card_time", "sub_in", "sub_out"]
        for column in minute_indicator_columns:
            df_player_stats[column] = df_player_stats[column].str.split(pat="'").str[0].str.strip().astype('float') / 90

        df_player_stats = cls._clean_kicker_name_string(df_player_stats)
        df_player_stats["season_start"] = df_player_stats["season"].str.split(pat="-").str[0].str.strip()

        int_columns = ['goals', 'matchday', 'red_card', 'yellow_card', 'yellow_red_card', 'cum_yellow_cards']
        for column in int_columns:
            df_player_stats[column] = np.floor(pd.to_numeric(df_player_stats[column], errors='coerce')).astype('Int64')

        cls._write_parquet(df_player_stats, './data/silver/player_stats/player_stats.parquet')

    @classmethod
    def _preprocess_player_ratings(cls):
        df_player_ratings = cls._read_parquet('./data/bronze/player_ratings/*')
        month_dict = {"Jan.": 1, "Feb.": 2, "March": 3, "April": 4, "May": 5, "June": 6, "June": 7, "Aug.": 8,
                      "Sept.": 9, "Oct.": 10, "Nov.": 11, "Dec.": 12}

        df_player_ratings["birth_date"] = df_player_ratings["birth_date"].str.replace(', ', '-')
        for month_string, month_num in month_dict.items():
            df_player_ratings["birth_date"] = df_player_ratings["birth_date"].str.replace(month_string,
                                                                                          str(month_num) + '-')
        df_player_ratings["birth_date"] = pd.to_datetime(df_player_ratings["birth_date"], format='%m-%d-%Y',
                                                         errors='coerce')

        df_player_ratings["height"] = df_player_ratings["height"] \
            .str.replace(' cm', '').str.strip()

        df_player_ratings["weight"] = df_player_ratings["weight"] \
            .str.replace(' kg', '').str.strip()

        df_player_ratings = cls._clean_fifa_name_string(df_player_ratings)
        df_player_ratings["nationality_id"] = df_player_ratings["nationality_id"].str.split(pat="=").str[1].str.strip()

        df_player_ratings["preferred_foot"] = df_player_ratings["preferred_foot"].str.replace('Left', '0').str.replace(
            'Right', '1')
        df_player_ratings["preferred_foot"] = df_player_ratings["preferred_foot"]
        pos_dict = {"x": {"LF": -0.5, "LW": -1, "LM": -1, "LWB": -1, "LB": -0.7, "RF": 0.5, "RW": 1, "RM": 1, "RWB": 1,
                          "RB": 0.7, "ST": 0, "CF": 0, "CAM": 0, "CM": 0, "CDM": 0, "CB": 0, "GK": 0},
                    "y": {"LF": 3, "LW": 2.5, "LM": 2, "LWB": 1.5, "LB": 1, "RF": 3, "RW": 2.5, "RM": 2, "RWB": 1.5,
                          "RB": 1, "ST": 3, "CF": 2.7, "CAM": 2.5, "CM": 2, "CDM": 1.5, "CB": 1, "GK": 0}}

        position_weights = [0.53, 0.27, 0.13, 0.07]
        position_weights_cum = {1: 0.53, 2: 0.8, 3: 0.93, 4: 1.00}

        df_player_ratings["position_x"] = 0
        df_player_ratings["position_y"] = 0

        position_cols = ["preferred_position_1", "preferred_position_2", "preferred_position_3", "preferred_position_4"]
        df_player_ratings["position_count"] = len(position_weights) - df_player_ratings[position_cols].isna().sum(1)

        for w, weight in enumerate(position_weights):
            df_player_ratings["position_x"] = df_player_ratings["position_x"] + df_player_ratings[
                "preferred_position_" + str(w + 1)].apply(lambda x: pos_dict.get("x").get(x)).fillna(0.00) * weight
            df_player_ratings["position_y"] = df_player_ratings["position_y"] + df_player_ratings[
                "preferred_position_" + str(w + 1)].apply(lambda x: pos_dict.get("y").get(x)).fillna(0.00) * weight

        df_player_ratings["position_x"] = df_player_ratings["position_x"] / df_player_ratings["position_count"].apply(
            lambda x: position_weights_cum.get(x))
        df_player_ratings["position_y"] = df_player_ratings["position_y"] / df_player_ratings["position_count"].apply(
            lambda x: position_weights_cum.get(x))

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
            df_player_ratings[column] = np.floor(pd.to_numeric(df_player_ratings[column], errors='coerce')).astype(
                'Int64')

        df_player_ratings = df_player_ratings.drop(
            columns=["preferred_position_1", "preferred_position_2", "preferred_position_3", "preferred_position_4"])
        cls._write_parquet(df_player_ratings, './data/silver/player_ratings/player_ratings.parquet')

    @classmethod
    def _calculate_elos(cls):
        df_team_stats = cls._read_parquet('./data/silver/team_stats/*')
        goal_data = df_team_stats.groupby(["season_start", "match_day", "game_id", "indicator", "team_name"])[
            'goals'].sum().reset_index()
        goal_data_home = goal_data.loc[goal_data['indicator'] == "home"].rename(
            columns={"team_name": "home_team_name", "goals": "home_goals"}).drop(columns=["indicator"])
        goal_data_away = goal_data.loc[goal_data['indicator'] == "away"].rename(
            columns={"team_name": "away_team_name", "goals": "away_goals"}).drop(columns=["indicator"])
        goal_data = goal_data_home.merge(goal_data_away, on=["season_start", "match_day", "game_id"],
                                         how="inner").sort_values(by=["season_start", "match_day"], ascending=True)

        elo_dict = {}
        df_elo = []
        for idx, row in goal_data.iterrows():
            game_id = row["game_id"]
            year = int(row["season_start"])
            matchday = int(row["match_day"])
            home_team = row["home_team_name"]
            away_team = row["away_team_name"]
            home_goals = row["home_goals"]
            away_goals = row["away_goals"]
            if elo_dict.get(year) is None: elo_dict[year] = dict()
            if elo_dict.get(year).get(matchday) is None: elo_dict[year][matchday] = dict()

            old_home_elo = cls._get_latest_elo(elo_dict, year, matchday, home_team)
            old_away_elo = cls._get_latest_elo(elo_dict, year, matchday, away_team)
            new_home_elo, new_away_elo = EloCalculator.calculcate_new_elos(old_home_elo, old_away_elo, home_goals,
                                                                           away_goals)
            elo_dict[year][matchday][home_team] = new_home_elo
            elo_dict[year][matchday][away_team] = new_away_elo
            df_elo.append({"game_id": game_id, "season_start": year, "matchday": matchday, "home_team": home_team,
                           "home_elo": old_home_elo, "new_home_elo": new_home_elo, "home_goals": home_goals,
                           "away_goals": away_goals, "away_elo": old_away_elo, "new_away_elo": new_away_elo,
                           "away_team": away_team})

        return pd.DataFrame(df_elo)

    @classmethod
    def _get_latest_elo(cls, elo_dict, year, matchday, team_name):
        # find last matchday
        previous_elo = 1300
        try:
            previous_elo = elo_dict[year][matchday - 1][team_name]
        except KeyError:
            # previous matchday not found try last season last matchday
            for y in range(int(year), 12, -1):
                for m in range(int(matchday), 0, -1):
                    try:
                        return elo_dict[y][m][team_name]
                    except KeyError:
                        continue
                matchday = 50
        return previous_elo

    @classmethod
    def _calculate_team_linear_regressions(cls):
        df_team_stats = cls._read_parquet('./data/silver/team_stats/*')
        df_match_info = cls._read_parquet('./data/silver/match_info/*')[["game_id", "kick_off_date"]]
        df_elo_diff = cls._read_parquet('./data/silver/team_elo/*')

        df_home_elo_diff = df_elo_diff
        df_home_elo_diff["elo_diff"] = df_home_elo_diff["home_elo"] - df_home_elo_diff["away_elo"]
        df_home_elo_diff = df_home_elo_diff.rename(columns={'home_team': 'team_name'})
        df_home_elo_diff = df_home_elo_diff[["game_id", "elo_diff", "team_name"]]

        df_away_elo_diff = df_elo_diff
        df_away_elo_diff["elo_diff"] = df_away_elo_diff["away_elo"] - df_away_elo_diff["home_elo"]
        df_away_elo_diff = df_away_elo_diff.rename(columns={'away_team': 'team_name'})
        df_away_elo_diff = df_away_elo_diff[["game_id", "elo_diff", "team_name"]]

        df_elo_diff = pd.concat([df_home_elo_diff, df_away_elo_diff])

        df_team_stats = df_team_stats.merge(df_match_info, on=["game_id"], how="inner")
        df_team_stats = df_team_stats.merge(df_elo_diff, on=["game_id", "team_name"], how="left")

        lin_reg_cols = ["shots_on_goal", "distance", "total_passes", "pass_ratio", "crosses", "cross_ratio",
                        "dribblings", "dribble_ratio", "possession", "tackles", "tackle_ratio", "air_tackles",
                        "air_tackle_ratio", "fouls", "got_fouled", "offside", "corners"]
        reg_df_list = []

        df_teams = df_team_stats.groupby('team_name')
        df_teams_stat_list = [df_teams.get_group(x) for x in df_teams.groups]
        for df_team_stat in tqdm(df_teams_stat_list):
            df_team_stat = df_team_stat[df_team_stat["kick_off_date"].notnull()]
            kick_off_dates = df_team_stat['kick_off_date'].drop_duplicates().sort_values(ascending=True).tolist()
            for kick_off_date in kick_off_dates:
                df_team_stat_pit = df_team_stat[df_team_stat["kick_off_date"] < kick_off_date]
                df_team_stat_entry = df_team_stat[["game_id", "team_name", "kick_off_date"]][
                    df_team_stat["kick_off_date"] == kick_off_date]
                for lin_reg_col in lin_reg_cols:
                    df_team_stat_pit_col = df_team_stat_pit[df_team_stat_pit[lin_reg_col].notnull()]
                    if len(df_team_stat_pit_col.index) <= 5:
                        # print("skip")
                        df_team_stat_entry[lin_reg_col + "_intercept"] = None
                        df_team_stat_entry[lin_reg_col + "_coefficient"] = None
                        continue

                    first_matchday = df_team_stat_pit_col["kick_off_date"].min()
                    last_matchday = df_team_stat_pit_col["kick_off_date"].max()
                    df_team_stat_pit_col['total_days'] = (
                            last_matchday - first_matchday + pd.to_timedelta(1, unit='D')).days
                    df_team_stat_pit_col['day_since_first_kickoff'] = ((df_team_stat_pit_col[
                                                                            "kick_off_date"] - first_matchday + pd.to_timedelta(
                        1, unit='D')).dt.days.values)
                    df_team_stat_pit_col['norm_weight'] = (
                            df_team_stat_pit_col['day_since_first_kickoff'] / df_team_stat_pit_col['total_days'])

                    # df_team_stat_pit_col["log_weight"] = np.log(df_team_stat_pit_col['norm_weight'])
                    q = 2
                    df_team_stat_pit_col["q_weight"] = (1 // 1.03 + (df_team_stat_pit_col['norm_weight'] ** q))
                    # X = (df_team_stat_pit_col["kick_off_date"] - first_matchday).dt.days.values.reshape(-1, 1)

                    X = df_team_stat_pit_col["elo_diff"].values.reshape((-1, 1))
                    y = df_team_stat_pit_col[lin_reg_col]
                    weights = df_team_stat_pit_col["q_weight"]
                    reg = linear_model.LinearRegression().fit(X, y, weights)
                    # print("intercept=", reg.intercept_, " coefficient=", reg.coef_)
                    df_team_stat_entry[lin_reg_col + "_intercept"] = reg.intercept_
                    df_team_stat_entry[lin_reg_col + "_coefficient"] = reg.coef_[0]
                    df_team_stat_entry[lin_reg_col + "_intercept"] = df_team_stat_entry[
                        lin_reg_col + "_intercept"].round(6)
                    df_team_stat_entry[lin_reg_col + "_coefficient"] = df_team_stat_entry[
                        lin_reg_col + "_coefficient"].round(6)
                reg_df_list.append(df_team_stat_entry)
        df_lin_reg = pd.concat(reg_df_list)
        cls._write_parquet(df_lin_reg, './data/silver/team_profiles_lin_reg/team_profiles_lin_reg.parquet')

    @classmethod
    def _player_mapping_kicker_fifa(cls):
        df_player_mapping = cls._read_parquet('./data/silver/player_mapping/*')

        df_players_kicker = cls._read_parquet('./data/silver/player_stats/*')[["player_name"]]
        df_players_fifa = cls._read_parquet('./data/silver/player_ratings/*')[["name"]]

        # filter out player mappes
        if df_player_mapping is not None:
            # @TODO: Identify what values are in TableB and not in TableA
            print("anti join")

        df_players_kicker["player_name"] = df_players_kicker["player_name"]
        df_players_fifa["name"] = df_players_fifa["name"]
        df_fuzzy_matched = cls._fuzzy_merge(df_players_kicker, df_players_fifa)
        cls._write_parquet(df_fuzzy_matched, './data/silver/player_mapping/player_mapping.parquet')

    @classmethod
    def _fuzzy_merge(cls, df_players_kicker, df_players_fifa, threshold=85):
        mapping_list = []
        players_kicker = df_players_kicker.groupby(["player_name"]).size().reset_index(name="count")
        players_kicker = players_kicker.to_dict('records')
        players_fifa = df_players_fifa["name"].drop_duplicates().to_list()
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

    @classmethod
    def _team_fifa_rating(cls):
        df_player_mapping = cls._read_parquet('./data/silver/player_mapping/*')
        df_players_kicker = cls._read_parquet('./data/silver/player_stats/*').rename(columns={'player_name': 'kicker_name','season_start':'fifa'})
        df_players_fifa = cls._read_parquet('./data/silver/player_ratings/*').rename(columns={'name': 'fifa_name'})

        df_players_kicker["fifa"] = df_players_kicker["fifa"].astype(str)
        df_players_fifa["fifa"] = df_players_fifa["fifa"].astype(str)

        df_players_kicker = df_players_kicker[['game_id', 'indicator','kicker_name','fifa']]
        df_players_fifa = df_players_fifa[['fifa_name', 'fifa','age']]


        df_team_rating = df_players_kicker.merge(df_player_mapping, on=["kicker_name"], how="inner")
        df_team_rating = df_team_rating.merge(df_players_fifa, on=["fifa_name","fifa"], how="inner")

        df_team_rating = df_team_rating.drop(columns=["kicker_name","fifa_name"])

        df_team_rating = df_team_rating[['game_id','indicator','age']]
        print(len(df_team_rating.index))
        df_teams = df_team_rating.groupby(['game_id','indicator'])
        df_line_up_list = [df_teams.get_group(x) for x in df_teams.groups]
        for df_line_up in df_line_up_list:
            print(df_line_up)
            break

    @classmethod
    def _clean_kicker_name_string(cls, df):
        df["player_name"] = df["player_name"].str.split(pat="/").str[0].str.strip()
        for i in range(1,10):
            df["player_name"] = df["player_name"].str.replace(str(i), '')
        df["player_name"] = df["player_name"].str.replace('-', ' ').str.lower().str.strip()
        return df

    @classmethod
    def _clean_fifa_name_string(cls, df):
        replace_dict = {"ü": "ue", "ä": "ae", "ö": "oe", "ç": "c", "á": "a", "à": "a", "è": "e", "é": "e", "ć": "c",
                        "č": "c", "š": "s", "ð": "d", "í": "i", "ï": "i", "ø": "oe","ğ":"g","ş":"s","ñ":"n","ł":"l","ń":"ń","ó":"o","ã":"a","ă":"a","aæ":"ae"}
        df["name"] = df["name"].str.strip().str.lower().str.replace('-', ' ')
        for special_char, replace_char in replace_dict.items():
            df["name"] = df["name"].str.replace(special_char, replace_char)
        return df
