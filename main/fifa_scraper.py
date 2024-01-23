import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import hashlib

from tqdm import tqdm

import os
import glob

class FifaScraper:

    def __init__(self, year):
        self.year = year
        self.last_page = json.load(open('./config/mapping_fifa.json', 'r')).get(str(year)).get("last_page")
        self.fifa_ratings_columns = json.load(open('./config/mapping_fifa_ratings.json', 'r'))
        self.base_url = "https://fifaindex.com/players/fifa" + str(year)
        self.profile_links = set()
        # data dicts
        self.player_data = []
        print("fifa scraper initialized for year ", self.year)

    def scrape(self):
        #create page index
        for p in tqdm(range(1, self.last_page + 1, 1)):
            page_url = self.base_url + "/?page=" + str(p)
            self._get_initial_profile_links(page_url)
            break

        for profile_link in tqdm(self.profile_links):
            self._get_player_stats(profile_link)
            break

        print(self.player_data)

    def _get_initial_profile_links(self, page_url):
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        players = soup.findAll('a', {'class': 'link-player'})

        for player in players:
            self.profile_links.add(player.get("href"))

    def _get_player_stats(self, player_url):
        player_page = requests.get("https://fifaindex.com" + player_url + "fifa" + str(self.year))
        soup = BeautifulSoup(player_page.content, 'html.parser')
        player_id = player_url.split("/")[2]
        player_dict = {"fifa": self.year,"player_id":player_id, "preferred_position_1":None, "preferred_position_2":None, "preferred_position_3":None}

        #mb_5s = soup.findAll('div', {'class': 'card mb-5'})
        pclasses = soup.findAll('p', {'class': ''})
        preferred_positions = set()

        for pclass in pclasses:
            for v in pclass.findAll('span', {'class': 'float-right'}):
                attribute = self.fifa_ratings_columns.get(pclass.text.replace(" "+v.text, ""))
                if attribute is not None:
                    value = v.text
                    for metric_value in v.findAll('span', {'class': 'data-units data-units-metric'}):
                        value = metric_value.text
                    for position in v.findAll('a', {'class': 'link-position'}):
                        value = position.text
                        if attribute == "preferred_position_1": preferred_positions.add(value)
                    player_dict[attribute] = value


        for v, value in enumerate(preferred_positions):
            player_dict["preferred_position_"+str(v+1)] = value
            if v>=3: break


        """
        for mb_5 in mb_5s:
            if "Height" in mb_5:
                print(mb_5)

        team_info = mb_5s[2]
        #print(mb_5s[2])
        player_dict["team_link"] = team_info.findAll('a', {'class': 'link-team'})[1].get("href")
        player_dict["team_name"] = team_info.findAll('a', {'class': 'link-team'})[1].text

        #print(mb_5s[3])
        """

        self.player_data.append(player_dict)


    def _store_data(self):
        df = pd.DataFrame(self.player_data)
        df = df.drop_duplicates()
        df.to_parquet("./data/bronze/player_ratings/" + str(self.year) + ".parquet", index=False)
        print("finished storing player stats for season ", str(self.year))


    @classmethod
    def delete_bronze_data(self):
        files = glob.glob('./data/bronze/player_ratings/*')
        for f in files:
            os.remove(f)