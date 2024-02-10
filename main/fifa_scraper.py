import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import hashlib
from .job_bookmark import JobBookmark
from copy import deepcopy
from tqdm import tqdm

import os
import glob

class FifaScraper:

    def __init__(self, year):
        self.year = year
        self.last_page = json.load(open('./config/mapping_fifa.json', 'r')).get(str(year)).get("last_page")
        self.fifa_ratings_columns = json.load(open('./config/mapping_fifa_ratings.json', 'r'))
        self.base_url = "https://example.com/players/fifa" + str(year)
        # !!!  url changed due to legal implications !!!
        self.profile_links = set()
        # data dicts
        self.player_ratings = []
        print("fifa scraper initialized for year ", self.year)

    def scrape(self):
        if JobBookmark.get_data_scraped("fifa_scraper").get(self.year): return

        #create page index
        for p in tqdm(range(1, self.last_page + 1, 1)):
            page_url = self.base_url + "/?page=" + str(p)
            self._get_initial_profile_links(page_url)

        last_index = JobBookmark.get_data_scraped("fifa_scraper").get(self.year)

        for profile_link in tqdm(self.profile_links):
            if last_index is not None and p+1<last_index: continue
            self._get_player_stats(profile_link)
        self._store_data()
        self.player_ratings = []



    def _get_initial_profile_links(self, page_url):
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        players = soup.findAll('a', {'class': 'link-player'})

        for player in players:
            self.profile_links.add(player.get("href"))

    def _get_player_stats(self, player_url):
        if "fifa" + str(self.year) not in player_url: player_url = player_url + "fifa" + str(self.year)
        player_page = requests.get("https://example.com" + player_url)
        # !!!  url changed due to legal implications !!!
        soup = BeautifulSoup(player_page.content, 'html.parser')
        player_id = player_url.split("/")[2]
        player_dict = {"fifa": self.year,"player_id":player_id, "preferred_position_1":None, "preferred_position_2":None, "preferred_position_3":None,"preferred_position_4":None, "team_link": None,"team_name":None, "national_team_link": None, "national_team_name":None}
        for column in self.fifa_ratings_columns.values():
            player_dict[column] = None

        pclasses = soup.findAll('p', {'class': ''})
        preferred_positions = set()

        for pclass in pclasses:
            for v in pclass.findAll('span', {'class': 'float-right'}):
                attribute = self.fifa_ratings_columns.get(pclass.text.replace(v.text, "").strip())
                if attribute is not None:
                    value = v.text
                    for metric_value in v.findAll('span', {'class': 'data-units data-units-metric'}):
                        value = metric_value.text
                    for position in v.findAll('a', {'class': 'link-position'}):
                        value = position.text
                        if attribute == "preferred_position_1": preferred_positions.add(value)
                    for stars in v.findAll('i', {'class': 'fas fa-star fa-lg'}):
                        value = len(v.findAll('i', {'class': 'fas fa-star fa-lg'}))
                    player_dict[attribute] = value

        for v, value in enumerate(preferred_positions):
            player_dict["preferred_position_"+str(v+1)] = value
            if v>=3: break

        team_info = soup.findAll('a', {'class': 'link-team'})

        for n, tinfo in enumerate(team_info):
            if n==1:
                player_dict["team_link"] = tinfo.get("href")
                player_dict["team_name"] = tinfo.text
            if n==3:
                player_dict["national_team_link"] = tinfo.get("href")
                player_dict["national_team_name"] = tinfo.text

        nationality_info = soup.findAll('h2', {'class': 'd-flex align-items-center'})
        for ninfo in nationality_info:
            for n in ninfo.findAll('a'):
                player_dict["nationality_id"] = n.get("href")
            player_dict["nationality"] = ninfo.text

        for name_info in soup.findAll('div', {'class': 'align-self-center pl-3'}):
            for n in name_info.findAll('h1'):
                name = n.text
                for suffix in n.findAll('span'):
                    name = name.replace(" "+suffix.text, "")
                player_dict["name"] = name
        self.player_ratings.append(player_dict)


    def _store_data(self):
        df = pd.DataFrame(self.player_ratings)
        df = df.drop_duplicates()
        df.to_parquet("./data/bronze/player_ratings/players_fifa" + str(self.year) + ".parquet", index=False)
        print("finished storing player stats for season ", str(self.year))

        JobBookmark.update_bookmark("fifa_scraper", self.year, True)

    @classmethod
    def delete_bronze_data(self):
        files = glob.glob('./data/bronze/player_ratings/*')
        for f in files:
            os.remove(f)