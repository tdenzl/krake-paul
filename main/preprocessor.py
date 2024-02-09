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
        if table_name == "team_elo": cls._calculate_team_elos()
        if table_name == "team_lin_regs": cls._calculate_team_linear_regressions()
        if table_name == "player_mapping": cls._player_mapping_kicker_fifa()
        if table_name == "referee_profiles": cls._referee_profiles()
        if table_name == "coach_elo": cls._coach_elo()
        if table_name == "player_elo": cls._player_elo()
        if table_name == "relationships": cls._relationships()


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
    def _calculate_team_elos(cls):
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
        df_home_elo_diff["elo_gain"] = df_home_elo_diff["new_home_elo"] - df_home_elo_diff["home_elo"]
        df_home_elo_diff = df_home_elo_diff.rename(columns={'home_team': 'team_name'})
        df_home_elo_diff = df_home_elo_diff[["game_id", "elo_diff", "elo_gain", "team_name"]]

        df_away_elo_diff = df_elo_diff
        df_away_elo_diff["elo_diff"] = df_away_elo_diff["away_elo"] - df_away_elo_diff["home_elo"]
        df_away_elo_diff["elo_gain"] = df_away_elo_diff["new_away_elo"] - df_away_elo_diff["away_elo"]
        df_away_elo_diff = df_away_elo_diff.rename(columns={'away_team': 'team_name'})
        df_away_elo_diff = df_away_elo_diff[["game_id", "elo_diff", "elo_gain", "team_name"]]

        df_elo_diff = pd.concat([df_home_elo_diff, df_away_elo_diff])

        df_team_stats = df_team_stats.merge(df_match_info, on=["game_id"], how="inner")
        df_team_stats = df_team_stats.merge(df_elo_diff, on=["game_id", "team_name"], how="left")

        lin_reg_cols = ["shots_on_goal", "distance", "total_passes", "pass_ratio", "crosses", "cross_ratio",
                        "dribblings", "dribble_ratio", "possession", "tackles", "tackle_ratio", "air_tackles",
                        "air_tackle_ratio", "fouls", "got_fouled", "offside", "corners", "elo_gain", "goals"]
        reg_df_list = []

        df_teams = df_team_stats.groupby('team_name')
        df_teams_stat_list = [df_teams.get_group(x) for x in df_teams.groups]
        for df_team_stat in tqdm(df_teams_stat_list):
            df_team_stat = df_team_stat[df_team_stat["kick_off_date"].notnull()]
            kick_off_dates = df_team_stat['kick_off_date'].drop_duplicates().sort_values(ascending=True).tolist()
            for kick_off_date in kick_off_dates:
                df_team_stat_pit = df_team_stat[df_team_stat["kick_off_date"] < kick_off_date]
                df_team_stat_entry = df_team_stat[["game_id", "team_name", "indicator", "kick_off_date"]][
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
                    weights = df_team_stat_pit_col["norm_weight"]
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

        df_players_kicker = df_players_kicker[['game_id', 'indicator','kicker_name','fifa','league_code']]
        df_players_fifa = df_players_fifa.drop_duplicates(['fifa_name', 'fifa'])


        df_team_rating = df_players_kicker.merge(df_player_mapping, on=["kicker_name"], how="inner")
        df_team_rating = df_team_rating.merge(df_players_fifa, on=["fifa_name","fifa"], how="inner")

        df_team_rating = df_team_rating.drop(columns=["kicker_name"])
        print(len(df_team_rating.index))

        compile_cols_dict={"ball_skills":["ball_control","dribbling"],
                           "defense":["slide_tackle", "stand_tackle"],
                            "mental":["aggression", "reactions", "att_position", "interceptions", "vision", "composure"],
                            "passing":["crossing", "short_pass", "long_pass"],
                            "physical":["acceleration", "stamina", "strength", "balance", "sprint_speed", "agility", "jumping"],
                            "shooting":["heading", "shot_power", "finishing", "long_shots", "curve", "fk_acc", "penalties", "volleys"],
                            "goal_keeper":["gk_positioning", "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes"]}

        df_team_rating["gk"] = 1
        #df_team_rating["def"] = 1
        #df_team_rating["mid"] = 1
        #df_team_rating["att"] = 1
        df_team_rating.loc[df_team_rating["position_x"] != 0.00, "gk"] = 0
        df_team_rating.loc[df_team_rating["position_y"] != 0.00, "gk"] = 0
        df_team_rating.loc[df_team_rating["gk"] != 1, "goal_keeper"] = None

        for category, column_list in compile_cols_dict.items():
            df_team_rating[category] = 0
            for column in column_list:
                df_team_rating[category] = df_team_rating[category] + df_team_rating[column]
            df_team_rating[category] = df_team_rating[category]/ len(column_list)
            df_team_rating = df_team_rating.drop(columns=column_list)

        df_team_rating_agg = df_team_rating.groupby(['game_id','indicator','league_code']).agg(
            players_captured=('fifa_name', np.count_nonzero),
            age_mean=('age', np.mean),
            age_stdev=('age', np.std),
            height_mean=('height', np.mean),
            height_stdev=('height', np.std),
            weight_mean=('weight', np.mean),
            weight_stdev=('weight', np.std),
            weak_foot__mean=('weak_foot', np.mean),
            skill_mean=('skill_moves', np.mean),
            position_x_mean=('position_x', np.mean),
            position_y_mean=('position_y', np.mean),
            ball_skills_mean=('ball_skills', np.mean),
            ball_skills_stdev=('ball_skills', np.std),
            defense_mean=('defense', np.mean),
            defense_stdev=('defense', np.std),
            mental_mean=('mental', np.mean),
            mental_stdev=('mental', np.std),
            passing_mean=('passing', np.mean),
            passing_stdev=('passing', np.std),
            physical_mean=('physical', np.mean),
            physical_stdev=('physical', np.std),
            shooting_mean=('shooting', np.mean),
            shooting_stdev=('shooting', np.std),
            goal_keeper_mean=('goal_keeper', np.mean),
            gk_captured=('gk', np.count_nonzero),
        ).reset_index()

        #print(df_team_rating_agg)
        cls._write_parquet(df_team_rating_agg, './data/silver/team_ratings/team_ratings.parquet')

    @classmethod
    def _referee_profiles(cls):
        df_match_info = cls._read_parquet('./data/silver/match_info/*')[["game_id","referee","kick_off_date"]]

        df_players_stats = cls._read_parquet('./data/silver/player_stats/*').groupby(['game_id','indicator']).agg(
            yellow_cards=('yellow_card', np.sum),
            yellow_red_cards=('yellow_red_card', np.sum),
            red_cards=('red_card', np.sum)).reset_index()
        df_players_stats_home = df_players_stats.loc[df_players_stats["indicator"] == "home"].rename(columns={'yellow_cards': 'yellow_cards_home','yellow_red_cards': 'yellow_red_cards_home','red_cards': 'red_cards_home'}).drop(columns=["indicator"])
        df_players_stats_away = df_players_stats.loc[df_players_stats["indicator"] == "away"].rename(columns={'yellow_cards': 'yellow_cards_away','yellow_red_cards': 'yellow_red_cards_away','red_cards': 'red_cards_away'}).drop(columns=["indicator"])

        df_card_stats = df_players_stats_home.merge(df_players_stats_away, on=["game_id"], how="inner")

        df_team_stats = cls._read_parquet('./data/silver/team_stats/*')[["game_id","indicator","goals"]]
        df_elo_diff = cls._read_parquet('./data/silver/team_elo/*')
        df_elo_diff["elo_diff"] = df_elo_diff["home_elo"] - df_elo_diff["away_elo"]
        df_elo_diff = df_elo_diff[["game_id","elo_diff"]]
        df_team_stats = df_team_stats.merge(df_elo_diff, on=["game_id"], how="left")

        df_team_stats.loc[df_team_stats["indicator"] == "away", 'goals'] = -1 * df_team_stats["goals"]
        df_team_stats_agg = df_team_stats.groupby('game_id').agg(
            goal_diff=('goals', np.sum),
            elo_diff=('elo_diff', np.sum)).reset_index()

        df_team_stats_agg['hxa'] = np.where(df_team_stats_agg['goal_diff'] < 0, -1,
                        np.where(df_team_stats_agg['goal_diff'] == 0, 0, 1))

        df_referees = df_match_info.merge(df_card_stats, on=["game_id"], how="left")
        df_referees = df_referees.merge(df_team_stats_agg, on=["game_id"], how="left")

        df_referees["cards_home"] = df_referees["yellow_cards_home"] + df_referees["yellow_red_cards_home"]*1.7 + df_referees["red_cards_home"]*2
        df_referees["cards_away"] = df_referees["yellow_cards_away"] + df_referees["yellow_red_cards_away"]*1.7 + df_referees["red_cards_away"]*2
        df_referees["cards_diff"] = df_referees["cards_home"] - df_referees["cards_away"]

        reg_df_list = []
        lin_reg_cols = ["cards_home","cards_away","cards_diff","goal_diff","hxa"]
        df_referees = df_referees.groupby('referee')
        df_referees_stat_list = [df_referees.get_group(x) for x in df_referees.groups]
        for df_referee_stat in tqdm(df_referees_stat_list):
            df_referee_stat = df_referee_stat[df_referee_stat["kick_off_date"].notnull()]
            kick_off_dates = df_referee_stat['kick_off_date'].drop_duplicates().sort_values(ascending=True).tolist()
            for kick_off_date in kick_off_dates:
                df_referee_stat_pit = df_referee_stat[df_referee_stat["kick_off_date"] < kick_off_date]
                df_referee_stat_entry = df_referee_stat[["game_id", "referee", "kick_off_date"]][
                    df_referee_stat["kick_off_date"] == kick_off_date]
                for lin_reg_col in lin_reg_cols:
                    df_referee_stat_pit_col = df_referee_stat_pit[df_referee_stat_pit[lin_reg_col].notnull()]
                    if len(df_referee_stat_pit.index) <= 5:
                        # print("skip")
                        df_referee_stat_entry[lin_reg_col + "_intercept"] = None
                        df_referee_stat_entry[lin_reg_col + "_coefficient"] = None
                        continue
                    first_matchday = df_referee_stat_pit_col["kick_off_date"].min()
                    last_matchday = df_referee_stat_pit_col["kick_off_date"].max()
                    df_referee_stat_pit_col['total_days'] = (
                            last_matchday - first_matchday + pd.to_timedelta(1, unit='D')).days
                    df_referee_stat_pit_col['day_since_first_kickoff'] = ((df_referee_stat_pit_col[
                                                                            "kick_off_date"] - first_matchday + pd.to_timedelta(
                        1, unit='D')).dt.days.values)
                    df_referee_stat_pit_col['norm_weight'] = (df_referee_stat_pit_col['day_since_first_kickoff'] / df_referee_stat_pit_col['total_days'])

                    """
                    # df_team_stat_pit_col["log_weight"] = np.log(df_team_stat_pit_col['norm_weight'])
                    q = 2
                    df_referee_stat_pit_col["q_weight"] = (1 // 1.03 + (df_referee_stat_pit_col['norm_weight'] ** q))
                    # X = (df_team_stat_pit_col["kick_off_date"] - first_matchday).dt.days.values.reshape(-1, 1)
                    """

                    X = df_referee_stat_pit_col["elo_diff"].values.reshape((-1, 1))
                    y = df_referee_stat_pit_col[lin_reg_col]
                    weights = df_referee_stat_pit_col["norm_weight"]
                    reg = linear_model.LinearRegression().fit(X, y, weights)
                    # print("intercept=", reg.intercept_, " coefficient=", reg.coef_)
                    df_referee_stat_entry[lin_reg_col + "_intercept"] = reg.intercept_
                    df_referee_stat_entry[lin_reg_col + "_coefficient"] = reg.coef_[0]
                    df_referee_stat_entry[lin_reg_col + "_intercept"] = df_referee_stat_entry[
                        lin_reg_col + "_intercept"].round(6)
                    df_referee_stat_entry[lin_reg_col + "_coefficient"] = df_referee_stat_entry[
                        lin_reg_col + "_coefficient"].round(6)
                reg_df_list.append(df_referee_stat_entry)
        df_lin_reg = pd.concat(reg_df_list)

        # print(df_team_rating_agg)
        cls._write_parquet(df_lin_reg, './data/silver/referee_profiles/referee_profiles.parquet')

    @classmethod
    def _player_elo(cls):
        df_match_info = cls._read_parquet('./data/silver/match_info/*')[["game_id", "kick_off_date"]]
        df_player_stats = cls._read_parquet('./data/silver/player_stats/*')[["game_id", "player_name", "indicator"]]
        df_team_stats = cls._read_parquet('./data/silver/team_stats/*')
        df_team_stats = df_team_stats.merge(df_match_info, on=["game_id"], how="inner")

        goal_data = df_team_stats.groupby(["game_id", "kick_off_date", "indicator"])[
            'goals'].sum().reset_index()
        goal_data_home = goal_data.loc[goal_data['indicator'] == "home"].rename(
            columns={"goals": "home_goals"}).drop(columns=["indicator"])
        goal_data_away = goal_data.loc[goal_data['indicator'] == "away"].rename(
            columns={"goals": "away_goals"}).drop(
            columns=["indicator", "kick_off_date"])
        goal_data = goal_data_home.merge(goal_data_away, on=["game_id"], how="inner").sort_values(by=["kick_off_date"],
                                                                                                  ascending=True)
        goal_data = df_player_stats.merge(goal_data, on=["game_id"], how="inner")

        df_player_elo = cls._read_parquet('./data/silver/player_elo/*')[["game_id", "player_name", "kick_off_date", "new_player_elo","opponnent_elo","old_player_elo"]]

        new_data = goal_data.merge(df_player_elo[["game_id", "player_name"]], on=["game_id","player_name"], how = 'outer', indicator = True)

        goal_data = new_data[~(new_data._merge == 'both')]

        player_elo_dict = dict()
        for idx, row in tqdm(df_player_elo.iterrows()):
            player_name = row["player_name"]
            kick_off_date = row["kick_off_date"]
            new_player_elo = row["new_player_elo"]
            opponnent_elo = row["opponnent_elo"]
            old_player_elo = row["old_player_elo"]
            if player_elo_dict.get(player_name) is None: player_elo_dict[player_name] = dict()
            if player_elo_dict.get(player_name).get(kick_off_date) is None: player_elo_dict[player_name][kick_off_date] = dict()
            player_elo_dict[player_name][kick_off_date]["player_elo"] = new_player_elo
            player_elo_dict[player_name][kick_off_date]["opponnent_elo"] = opponnent_elo
            player_elo_dict[player_name][kick_off_date]["old_player_elo"] = old_player_elo

        game_elo_dict = {}
        elo_list = []
        for idx, row in tqdm(goal_data.iterrows()):
            game_id = row["game_id"]
            indicator = row["indicator"]
            kick_off_date = row["kick_off_date"]
            player_name = row["player_name"]
            home_goals = row["home_goals"]
            away_goals = row["away_goals"]

            # check if already calculated
            new_player_elo = None
            try:
                new_player_elo = player_elo_dict[player_name][kick_off_date]["player_elo"]
                opponnent_elo = player_elo_dict[player_name][kick_off_date]["opponnent_elo"]
                player_elo = player_elo_dict[player_name][kick_off_date]["old_player_elo"]
                elo_list.append({"game_id": game_id, "kick_off_date": kick_off_date, "player_name": player_name,
                                 "old_player_elo": player_elo, "new_player_elo": new_player_elo,
                                 "opponnent_elo": opponnent_elo, "home_goals": home_goals,
                                 "away_goals": away_goals, "indicator": indicator})
            except KeyError:
                pass
            if new_player_elo is None:
                continue
            df_player = goal_data.loc[goal_data['player_name'] == player_name]
            df_player['p_kick_off_date'] = df_player["kick_off_date"].shift(1)
            last_game_date = df_player.loc[df_player['game_id'] == game_id]["p_kick_off_date"].iloc[0]
            if player_elo_dict.get(player_name) is None: player_elo_dict[player_name] = dict()
            if player_elo_dict.get(player_name).get(last_game_date) is None:
                player_elo_dict[player_name][last_game_date] = dict()
                player_elo_dict[player_name][last_game_date]["player_elo"] = 1300
                #print("could not find ", player_name, " last_game_date=" , last_game_date)
            player_elo = player_elo_dict.get(player_name).get(last_game_date).get("player_elo")

            if indicator == "home": opponent_indicator = "away"
            else: opponent_indicator = "home"
            try:
                opponnent_elo = game_elo_dict[game_id][opponent_indicator]
            except KeyError:
                try:
                    opponnent_elo = player_elo_dict[player_name][last_game_date]["opponnent_elo"]
                except KeyError:
                    opponnent_elo = None

            if opponnent_elo is None:
                opponnent_elo = 0
                opponent_players = goal_data.loc[goal_data["game_id"] == game_id]
                opponent_players = opponent_players.loc[goal_data["indicator"] == opponent_indicator]["player_name"].tolist()
                for opponent_player in opponent_players:
                    df_opponent_player = goal_data.loc[goal_data['player_name'] == opponent_player]
                    df_opponent_player['p_kick_off_date'] = df_opponent_player["kick_off_date"].shift(1)
                    last_game_date =  df_opponent_player.loc[df_opponent_player['game_id'] == game_id]["p_kick_off_date"].iloc[0]
                    if player_elo_dict.get(opponent_player) is None: player_elo_dict[opponent_player] = dict()
                    if player_elo_dict.get(opponent_player).get(last_game_date) is None:
                        player_elo_dict[opponent_player][last_game_date] = dict()
                        player_elo_dict[opponent_player][last_game_date]["player_elo"] = 1300
                    opponnent_elo += player_elo_dict.get(opponent_player).get(last_game_date).get("player_elo")
                opponnent_elo = opponnent_elo/len(opponent_players)

            new_player_elo, new_opponnent_elo = EloCalculator.calculcate_new_elos(player_elo, opponnent_elo, home_goals, away_goals)
            player_elo_dict[player_name][kick_off_date]["player_elo"] = new_player_elo
            elo_list.append({"game_id": game_id, "kick_off_date": kick_off_date, "player_name": player_name,
                             "old_player_elo": player_elo, "new_player_elo": new_player_elo,"opponnent_elo":opponnent_elo,"home_goals": home_goals,
                             "away_goals": away_goals, "indicator":indicator})
        df_elo = pd.DataFrame(elo_list)
        df_elo = pd.concat(df_elo, df_player_elo)
        cls._write_parquet(df_elo, './data/silver/player_elo/player_elo.parquet')

    @classmethod
    def _coach_elo(cls):
        df_match_info = cls._read_parquet('./data/silver/match_info/*')[["game_id", "kick_off_date"]]
        df_coaches = cls._read_parquet('./data/silver/coaches/*')[["game_id","coach_name","indicator"]]
        df_coach_stats = df_coaches.merge(df_match_info, on=["game_id"],how="inner")

        df_team_stats = cls._read_parquet('./data/silver/team_stats/*')
        df_coach_stats = df_coach_stats.merge(df_team_stats, on=["game_id","indicator"], how="inner")
        goal_data = df_coach_stats.groupby(["game_id", "kick_off_date", "indicator", "coach_name"])['goals'].sum().reset_index()
        goal_data_home = goal_data.loc[goal_data['indicator'] == "home"].rename(
            columns={"coach_name": "home_coach_name", "goals": "home_goals"}).drop(columns=["indicator"])
        goal_data_away = goal_data.loc[goal_data['indicator'] == "away"].rename(
            columns={"coach_name": "away_coach_name", "goals": "away_goals"}).drop(columns=["indicator","kick_off_date"])
        goal_data = goal_data_home.merge(goal_data_away, on=["game_id"], how="inner").sort_values(by=["kick_off_date"], ascending=True)

        elo_dict = {}
        elo_list = []
        for idx, row in tqdm(goal_data.iterrows()):
            game_id = row["game_id"]
            kick_off_date = row["kick_off_date"]
            home_coach = row["home_coach_name"]
            away_coach = row["away_coach_name"]
            home_goals = row["home_goals"]
            away_goals = row["away_goals"]

            df_coach_stats_home = df_coach_stats.loc[df_coach_stats['coach_name'] == home_coach]
            df_coach_stats_home['p_kick_off_date'] = df_coach_stats_home["kick_off_date"].shift(1)
            latest_kick_off_date_home_coach = df_coach_stats_home.loc[df_coach_stats_home['game_id'] == game_id]["p_kick_off_date"].iloc[0]

            df_coach_stats_away = df_coach_stats.loc[df_coach_stats['coach_name'] == away_coach]
            df_coach_stats_away['p_kick_off_date'] = df_coach_stats_away["kick_off_date"].shift(1)
            latest_kick_off_date_away_coach = df_coach_stats_away.loc[df_coach_stats_away['game_id'] == game_id]["p_kick_off_date"].iloc[0]


            if elo_dict.get(home_coach) is None: elo_dict[home_coach] = dict()
            if elo_dict.get(away_coach) is None: elo_dict[away_coach] = dict()
            if elo_dict.get(home_coach).get(latest_kick_off_date_home_coach) is None: elo_dict[home_coach][latest_kick_off_date_home_coach] = 1300
            if elo_dict.get(away_coach).get(latest_kick_off_date_away_coach) is None: elo_dict[away_coach][latest_kick_off_date_away_coach] = 1300
            old_home_elo = elo_dict[home_coach][latest_kick_off_date_home_coach]
            old_away_elo = elo_dict[away_coach][latest_kick_off_date_away_coach]
            new_home_elo, new_away_elo = EloCalculator.calculcate_new_elos(old_home_elo, old_away_elo, home_goals,
                                                                           away_goals)
            elo_dict[home_coach][kick_off_date] = new_home_elo
            elo_dict[away_coach][kick_off_date] = new_away_elo
            elo_list.append({"game_id": game_id, "kick_off_date": kick_off_date,"home_coach": home_coach,
                           "home_elo": old_home_elo, "new_home_elo": new_home_elo, "home_goals": home_goals,
                           "away_goals": away_goals, "away_elo": old_away_elo, "new_away_elo": new_away_elo,
                           "away_coach": away_coach})

        df_elo = pd.DataFrame(elo_list)
        cls._write_parquet(df_elo, './data/silver/coach_elo/coach_elo.parquet')

    @classmethod
    def _relationships(cls):
        df_match_info = cls._read_parquet('./data/silver/match_info/*')[["game_id","kick_off_date"]]
        df_coaches = cls._read_parquet('./data/silver/coaches/*')[["game_id", "coach_name","indicator"]].rename(
            columns={"coach_name": "name"})
        df_player_stats = cls._read_parquet('./data/silver/player_stats/*')[["game_id", "player_name","indicator"]].rename(
            columns={"player_name": "name"})

        df_relationships = pd.concat([df_coaches, df_player_stats])
        df_relationships = df_match_info.merge(df_relationships, on=["game_id"],how="inner").sort_values(by=["kick_off_date"], ascending=True)
        relationship_dict = {}
        game_relationship_dict = {}
        i = 0
        for idx, row in tqdm(df_relationships.iterrows()):
            game_id = row["game_id"]
            player = row["name"]
            indicator = row["indicator"]
            if game_relationship_dict.get(game_id) is None:
                game_relationship_dict[game_id] = {"home":0,"away":0}

            if relationship_dict.get(player) is None: relationship_dict[player] = dict()
            co_players = df_relationships[df_relationships["game_id"]==game_id]["name"].tolist()
            for co_player in co_players:
                if player == co_player: continue
                if relationship_dict.get(player).get(co_player) is None: relationship_dict[player][co_player] = 0
                relationship_dict[player][co_player] += 1
                game_relationship_dict[game_id][indicator] += relationship_dict[player][co_player]

        team_relationship_list = []
        for game_id, indicator_dict in game_relationship_dict.items():
            relationships_home = indicator_dict.get("home")
            relationships_away = indicator_dict.get("away")
            relationships_diff = indicator_dict.get("home") - indicator_dict.get("away")
            team_relationship_list.append({"game_id": game_id, "relationships_home": relationships_home, "relationships_away": relationships_away, "relationships_diff": relationships_diff})

        df_relationships = pd.DataFrame(team_relationship_list)
        cls._write_parquet(df_relationships, './data/silver/relationships/relationships.parquet')

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
