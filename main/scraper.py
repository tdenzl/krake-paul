import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join

class Scraper:

    def __init__(self, league, start_season, end_season, matchdays):
        self.base_url = "https://www.kicker.de"
        self.league = league
        self.start_season = start_season
        self.end_season = end_season
        self.matchdays = matchdays
        self.mappings =  json.load(open('./config/mapping_columns.json', 'r'))
        # data dicts
        self.match_info = []
        self.team_stats = []
        self.player_stats = []
        self.player_attributes = []

    def scrape(self):
        for season in tqdm(range(self.start_season, self.end_season, 1)):
            for matchday in tqdm(range(1, self.matchdays + 1, 1)):
                self._get_match_data(season - 1, season, matchday)

    def _get_match_data(self, start_year, end_year, match_day):
        season = str(start_year) + '-' + str(end_year)
        link_dict = self._get_links_matchday(season, match_day)

        # MATCH INFO
        for link in link_dict.get("match_info"):
            self._get_match_info(link, season, match_day)
        print("finished getting match info")

        # TEAM STATS
        for link in link_dict.get("team_stats"):
            self._get_teams_stats(link, season, match_day)
        print("finished getting match stats")

        # PLAYER STATS
        for link in link_dict.get("player_stats"):
            self._get_player_stats(link, season, match_day)
        print("finished getting player_stats and coaches")

    def _get_links_matchday(self, season, match_day):
        URL = self.base_url + "/" + self.league + "/spieltag/20" + str(season) + '/' + str(match_day)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.findAll('a', {'class': 'kick__v100-scoreBoard kick__v100-scoreBoard--standard'})
        link_dict = {"team_stats":[],"player_stats":[],"match_info":[]}
        for elem in results:
            link = elem.get('href')
            link_dict["team_stats"].append(self.base_url + link.replace("analyse", "spieldaten"))
            link_dict["player_stats"].append(self.base_url + link.replace("analyse", "schema"))
            link_dict["match_info"].append(self.base_url + link.replace("analyse", "spielinfo"))

        return link_dict

    def _get_match_info(self, URL, season, match_day):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        kick_off = soup.findAll('span', {'class': 'kick__weekday_box'})
        referee = soup.findAll('a')
        referee = referee[0].get("href").split("/")[1]+ "/" + referee[0].get('href').split("/")[2]

        match_info_dict = {"season": season, "match_day": match_day, "kick_off_time":kick_off[0].text, "referee": referee}

        self.match_info.append(match_info_dict)

    def _get_teams_stats(self, URL, season, match_day):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')

        teams = soup.findAll('div', {'class': 'kick__v100-gameCell__team__name'})
        stat_titles = soup.findAll('div', {'class': 'kick__stats-bar__title'})
        stats_home = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent1'})
        stats_away = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent2'})
        score = soup.findAll('div', {'class': 'kick__v100-scoreBoard__scoreHolder__score'})
        positions = soup.findAll('div', {'class': 'kick__v100-gameCell__team__info'})
        stats_home_dict = {"season": season, "match_day": match_day, "indicator": "home", "team_name": teams[0].text, "ht_goals":score[2].text, "position": positions[0].text}
        stats_away_dict = {"season": season, "match_day": match_day, "indicator": "away", "team_name": teams[1].text, "ht_goals":score[3].text, "position": positions[1].text}

        for stat_titles, stats_home, stats_away in zip(stat_titles, stats_home, stats_away):
            column_name = self.mappings.get("team_stats").get(stat_titles.text)
            if column_name is None: print(stat_titles.text)
            stats_home_dict[column_name] = stats_home.text
            stats_away_dict[column_name] = stats_away.text

        self.team_stats.append(stats_home_dict)
        self.team_stats.append(stats_away_dict)

    def _get_player_stats(self, URL, season, match_day):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')

        scorer_dict = dict()
        scorers = soup.findAll('a', {'class': 'kick__goals__player'})
        for player in scorers:
            player_link = player.get('href').split("/")[1] + "/" + player.get('href').split("/")[2]
            goal = 1
            for span in player.findAll('span', {'class': 'kick__goals__player-subtxt'}):
                if span.text == "(Eigentor)": goal = -1
            if scorer_dict.get(player_link) is None: scorer_dict[player_link] = goal
            else: scorer_dict[player_link] += goal

        substitution_dict = {"in":dict(), "out":dict()}
        card_dict = {}

        section_items = soup.findAll('section', {'class': 'kick__section-item'})
        substitution_section = None
        cards_section = None
        for section_item in section_items:
            for element in section_item.findAll('h4', {'class': 'kick__card-headline kick__text-center'}):
                if element.text == "Wechsel": substitution_section = section_item
            for element in section_item.findAll('h4', {'class': 'kick__card-headline kick__text-center'}):
                if element.text == "Karten": cards_section = section_item


        substitution_players = substitution_section.findAll('a', {'class': 'kick__substitutions__player'})
        substitution_times = substitution_section.findAll('div', {'class': 'kick__substitutions__time'})
        cards = cards_section.findAll('div', {'class': 'kick__substitutions__cell'})

        for p, player in enumerate(substitution_players):
            player_link = player.get('href').split("/")[1] + "/" + player.get('href').split("/")[2]
            if p % 2 == 0:
                substitution_dict["in"][player_link] = substitution_times[p].text
            if p % 2 == 1:
                substitution_dict["out"][player_link] = substitution_times[p-1].text

        for event in cards:
            player_link = None
            card_time = None
            card_description = None
            yellow_card = 0
            yellow_red_card = 0
            red_card = 0
            for pl in event.findAll('a', {'class': 'kick__substitutions__player'}):
                player_link = pl.get('href').split("/")[1] + "/" + player.get('href').split("/")[2]
            for ct in event.findAll('div', {'class': 'kick__substitutions__time'}):
                card_time = ct.text
            for cd in event.findAll('span', {'class': 'kick__substitutions__player-subtxt'}):
                card_description = cd.text
            for e in event.findAll('span', {'class': 'kick__ticker-icon kick__ticker-icon-color--yellow kick__icon-Gelb'}):
                yellow_card = 1
            for e in event.findAll('span', {'class': 'kick__ticker-icon kick__ticker-icon-color--red kick__icon-Gelb'}):
                yellow_red_card = 1
            for e in event.findAll('span', {'class': 'kick__ticker-icon kick__ticker-icon-color--red kick__icon-Rot'}):
                red_card = 1
            card_dict[player_link] = {"card_time":card_time,"card_description":card_description,"yellow_card":yellow_card,"yellow_red_card":yellow_red_card,"red_card":red_card}

        line_ups = {"home": soup.findAll('div', {'class': 'kick__lineup__team kick__lineup__team--left'}),\
                    "away": soup.findAll('div', {'class': 'kick__lineup__team kick__lineup__team--right'})}

        for indicator, line_up in line_ups.items():
            for element in line_up:
                players = element.findAll('a')
                for player in players:
                    name = player.get("href").split("/")[1] + "/" + player.get('href').split("/")[2]

                    if "/spieler" in player.get("href"):
                        #line_up_dict[name] = {"season": season, "match_day": match_day, "indicator": indicator, "goals": scorer_dict.get(name), "sub_in": substitution_dict.get("in").get(name), "sub_out": substitution_dict.get("out").get(name)}
                        card_entry = card_dict.get(name)
                        if card_entry is None: card_entry = {}
                        self.schema_data.append([season, match_day, indicator, name, scorer_dict.get(name), substitution_dict.get("in").get(name), substitution_dict.get("out").get(name), card_entry.get("card_time"), card_entry.get("card_description"), card_entry.get("yellow_card"), card_entry.get("yellow_red_card"), card_entry.get("red_card")])
                    if "/trainer" in player.get("href"):
                        self.coach_data.append([season, match_day, indicator, name])

    def store_data(self, league_code):
        df_team_stats = pd.DataFrame(self.team_stats)
        df_team_stats["league_code"] = league_code
        df_team_stats.to_csv("./data/team_stats/" + str(self.league) + "_" + str(self.start_season) + "_" + str(self.end_season) + ".csv", index=False)
        print(df_team_stats)

        df_match_info = pd.DataFrame(self.match_info)
        df_match_info["league_code"] = league_code
        df_match_info.to_csv("./data/match_info/" + str(self.league) + "_" + str(self.start_season) + "_" + str(self.end_season) + ".csv", index=False)
        print(df_match_info)
        
        df_player_performance = pd.DataFrame(self.schema_data, columns=['season',
                                                                'matchday',
                                                                'indicator',
                                                                'name'
                                                                'goals',
                                                                'sub_in',
                                                                'sub_out',
                                                                'card_time',
                                                                'card_time',
                                                                'card_description',
                                                                'yellow_card',
                                                                'yellow_red_card',
                                                                'red_card'])
        df_player_performance["league_code"] = league_code
        df_player_performance.to_csv("./data/player_stats/" + str(self.league) + "_" + str(self.start_season) + "_" + str(
            self.end_season) + ".csv", index=False)

        df_coaches = pd.DataFrame(self.coach_data, columns=['season',
                                                             'matchday',
                                                             'indicator',
                                                             'name'])
        df_coaches["league_code"] = league_code
        df_coaches.to_csv(
            "./data/coaches/" + str(self.league) + "_" + str(self.start_season) + "_" + str(
                self.end_season) + ".csv", index=False)

    def set_start_season(self, start_season):
        self.start_season = start_season

    def set_end_season(self, end_season):
        self.end_season = end_season

    def set_matchdays(self, matchdays):
        self.matchdays = matchdays