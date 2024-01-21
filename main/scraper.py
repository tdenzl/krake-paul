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
        self.match_list = []

    def scrape(self):
        for season in tqdm(range(self.start_season, self.end_season, 1)):
            for matchday in tqdm(range(1, self.matchdays, 1)):
                self._get_match_data(season - 1, season, matchday)

    def _get_match_data(self, start_year, end_year, match_day):
        season = str(start_year) + '-' + str(end_year)
        links = self._get_links_matchday(season, match_day)
        for link in links:
            game_stats = self._get_game_stats(link, season, match_day)
            self.match_list.append(game_stats)

    def _get_links_matchday(self, season, match_day):
        URL = self.base_url + "/" + self.league + "/spieltag/20" + str(season) + '/' + str(match_day)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.findAll('a', {'class': 'kick__v100-scoreBoard kick__v100-scoreBoard--standard'})
        links = []
        for elem in results:
            link = elem.get('href')
            link_spieldaten = link.replace("analyse", "spieldaten")
            URL = 'https://www.kicker.de/' + link_spieldaten
            links.append(URL)
        return links

    def _get_game_stats(self, URL, season, match_day):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        stats = []
        stats.append(season)
        stats.append(match_day)
        teams = soup.findAll('div', {'class': 'kick__v100-gameCell__team__name'})
        stats.append(teams[0].text)
        stats.append(teams[1].text)
        score = soup.findAll('div', {'class': 'kick__v100-scoreBoard__scoreHolder__score'})
        for s, goal in enumerate(score):
            stats.append(score[s].text)
        stats_home = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent1'})
        stats_away = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent2'})
        for i in range(1, 13, 1):
            stats.append(stats_home[i].text)
            stats.append(stats_away[i].text)
        return stats



    def set_start_season(self, start_season):
        self.start_season = start_season

    def set_end_season(self, end_season):
        self.end_season = end_season

    def set_matchdays(self, matchdays):
        self.matchdays = matchdays