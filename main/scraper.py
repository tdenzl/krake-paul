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
        # data dicts
        self.match_data = []
        self.line_up_data = []

    def scrape(self):
        for season in tqdm(range(self.start_season, self.end_season, 1)):
            for matchday in tqdm(range(1, self.matchdays + 1, 1)):
                self._get_match_data(season - 1, season, matchday)

    def _get_match_data(self, start_year, end_year, match_day):
        season = str(start_year) + '-' + str(end_year)
        link_dict = self._get_links_matchday(season, match_day)
        """
        for link in link_dict.get("match_stats"):
            stats_home_list, stats_away_list = self._get_match_stats(link, season, match_day)
            self.match_data.append(stats_home_list)
            self.match_data.append(stats_away_list)
        print("finished getting match stats")
        """
        for link in link_dict.get("schema"):
            self._get_schema(link, season, match_day)
        print("finished getting schema")

    def _get_links_matchday(self, season, match_day):
        URL = self.base_url + "/" + self.league + "/spieltag/20" + str(season) + '/' + str(match_day)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.findAll('a', {'class': 'kick__v100-scoreBoard kick__v100-scoreBoard--standard'})
        link_dict = dict()
        link_dict["match_stats"] = []
        link_dict["schema"] = []
        for elem in results:
            link = elem.get('href')
            link_dict["match_stats"].append(self.base_url + link.replace("analyse", "spieldaten"))
            link_dict["schema"].append(self.base_url + link.replace("analyse", "schema"))

        return link_dict

    def _get_match_stats(self, URL, season, match_day):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')

        teams = soup.findAll('div', {'class': 'kick__v100-gameCell__team__name'})
        stats_home = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent1'})
        stats_away = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent2'})
        score = soup.findAll('div', {'class': 'kick__v100-scoreBoard__scoreHolder__score'})

        stats_home_list = [season, match_day, "home", teams[0].text]
        stats_away_list = [season, match_day, "away", teams[1].text]

        for s, goal in enumerate(score):
            stats_home_list.append(score[s].text)
            stats_away_list.append(score[s].text)

        for i in range(1, 13, 1):
            stats_home_list.append(stats_home[i].text)
            stats_away_list.append(stats_away[i].text)
        return stats_home_list, stats_away_list

    def _get_line_ups(self, URL, season, match_day):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')

        home_line_up = soup.findAll('div', {'class': 'kick__lineup-field__field-half kick__lineup-field__field-half--top'})
        away_line_up = soup.findAll('div', {'class': 'kick__lineup-field__field-half kick__lineup-field__field-half--bottom'})
        home_team_link = None
        away_team_link = None
        for element in home_line_up:
            if element.find("a") is not None:
                home_team_link = element.find("a").get("href").split("/")[1]
        for element in away_line_up:
            if element.find("a") is not None:
                away_team_link = element.find("a").get("href").split("/")[1]

        player_infos = []

        substitutions = {"in": dict(), "out": dict()}
        player_subs = soup.findAll('div', {'class': 'kick__ticker__row'})
        for element in player_subs:
            for sub_minute in element.findAll('div', {'class': 'kick__ticker__cell-info'}):
                minute = sub_minute.text
            sub_players = element.findAll('a')
            player_in = sub_players[0].get('href').split("/")[1] + "/" + sub_players[0].get('href').split("/")[2]
            player_out = sub_players[1].get('href').split("/")[1] + "/" + sub_players[1].get('href').split("/")[2]
            substitutions["in"][player_in] = minute
            substitutions["out"][player_out] = minute

        for element in home_line_up:
            players = element.findAll('a', {'class': 'kick__lineup-player-card'})
            for player in players:
                player_link = player.get('href').split("/")[1] + "/" + player.get('href').split("/")[2]
                player_infos.append([season, match_day, "home", home_team_link, player_link, substitutions.get("in").get(player_link), substitutions.get("out").get(player_link)])

        for element in away_line_up:
            players = element.findAll('a', {'class': 'kick__lineup-player-card'})
            for player in players:
                player_link = player.get('href').split("/")[1] + "/" + player.get('href').split("/")[2]
                player_infos.append([season, match_day, "away", away_team_link, player_link, substitutions.get("in").get(player_link), substitutions.get("out").get(player_link)])

        print(player_infos)

    def _get_schema(self, URL, season, match_day):
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
            if p % 2 == 0: substitution_dict["in"][player_link] = substitution_times[int(p/2)].text
            if p % 2 == 1: substitution_dict["out"][player_link] = substitution_times[int(p/2)].text

        for event in cards:
            player_link = None
            card_time = None
            card_description = None
            yellow_card = 0
            yellow_red_card = 0
            red_card = 0
            for e in event.findAll('a', {'class': 'kick__substitutions__player'}):
                player_link = e.get('href').split("/")[1] + "/" + player.get('href').split("/")[2]
            for e in event.findAll('div', {'class': 'kick__substitutions__time'}):
                card_time =  e.text
            for e in event.findAll('span', {'class': 'kick__substitutions__player-subtxt'}):
                card_description =  e.text
            for e in event.findAll('span', {'class': 'kick__ticker-icon kick__ticker-icon-color--yellow kick__icon-Gelb'}):
                yellow_card += 1
            for e in event.findAll('span', {'class': 'kick__ticker-icon kick__ticker-icon-color--red kick__icon-Gelb'}):
                yellow_red_card += 1
            for e in event.findAll('span', {'class': 'kick__ticker-icon kick__ticker-icon-color--red kick__icon-Rot'}):
                red_card += 1
            card_dict



        print(substitution_dict)

        line_up_dict = {}
        coaches = []
        line_ups = {"home": soup.findAll('div', {'class': 'kick__lineup__team kick__lineup__team--left'}),\
                    "away": soup.findAll('div', {'class': 'kick__lineup__team kick__lineup__team--right'})}

        for indicator, line_up in line_ups.items():
            for element in line_up:
                players = element.findAll('a')
                for player in players:
                    name = player.get("href").split("/")[1] + "/" + player.get('href').split("/")[2]
                    if "/spieler" in player.get("href"):
                        line_up_dict[name] = {"season": season, "match_day": match_day, "indicator": indicator, "goals": scorer_dict.get(name)}
                    if "/trainer" in player.get("href"):
                        coaches.append([season, match_day, "home", name])

        print(scorer_dict)
        print(line_up_dict)
        print(coaches)

    def store_data(self, league_code):
        df_match_stats = pd.DataFrame(self.match_data, columns=['season',
                                              'matchday',
                                              'indicator',
                                              'team_name',
                                              'h_goals',
                                              'a_goals',
                                              'h_ht_goals',
                                              'a_ht_goals',
                                              'shots_on_goal',
                                              'distance',
                                              'total_passes',
                                              'success_passes',
                                              'failed_passes',
                                              'pass_ratio',
                                              'possesion',
                                              'tackle_ratio',
                                              'fouls',
                                              'got_fouled',
                                              'offside',
                                              'corners',])
        df_match_stats["league_code"] = league_code
        df_match_stats.to_csv("./data/match_stats/"+ str(self.league) + "_" + str(self.start_season) + "_" + str(self.end_season) + ".csv", index=False)

    def set_start_season(self, start_season):
        self.start_season = start_season

    def set_end_season(self, end_season):
        self.end_season = end_season

    def set_matchdays(self, matchdays):
        self.matchdays = matchdays