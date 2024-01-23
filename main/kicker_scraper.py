import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import hashlib
from .job_bookmark import JobBookmark

from tqdm import tqdm

import os
import glob

class KickerScraper:

    def __init__(self, league, start_season, end_season, matchdays=None):
        self.base_url = "https://www.kicker.de"
        self.league = league
        self.start_season = start_season
        self.end_season = end_season
        self.mapping_columns = json.load(open('./config/mapping_columns.json', 'r'))
        self.mapping_leagues = json.load(open('./config/mapping_leagues.json', 'r'))
        if matchdays is None: matchdays = self.mapping_leagues.get(league).get("matchdays")
        self.matchdays = matchdays
        # data dicts
        self.match_info = []
        self.team_stats = []
        self.player_stats = []
        self.coach_data = []
        self.player_attributes = []

        print("scraper initialized league=", self.league, " start_season=", self.start_season, " end_season=", self.end_season, " matchdays=", self.matchdays)

    def scrape(self):
        for season in tqdm(range(self.start_season, self.end_season, 1)):
            start_matchday = 1
            for matchday in tqdm(range(1, self.matchdays + 1, 1)):
                matchdays_parsed = []

                try:matchdays_parsed = JobBookmark.get_data_scraped("kicker_scraper")[self.league][season].keys()
                except KeyError: pass

                if matchday in matchdays_parsed:
                    continue
                if not self._check_team_stats_available(season, matchday): continue
                self._get_data(season, season + 1, matchday)
                if matchday%9==0:
                    self._save(season, start_matchday, matchday)
                    start_matchday = matchday
            self._save(season, start_matchday, matchday)

    def _get_current_season_matchday(self):
        #@TODO: method that returns the current season and matchday to avoid parsing the last day over and over again
        return

    def _check_team_stats_available(self, season, matchday):
        #check for first match of matchday if it has team stats
        URL = self.base_url + "/" + self.league + "/spieltag/20" + str(season) + '/' + str(matchday)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.findAll('a', {'class': 'kick__v100-scoreBoard kick__v100-scoreBoard--standard'})
        first_stats_link = self.base_url + results[0].get('href').replace("analyse", "spieldaten").replace("spielbericht", "spieldaten")

        page = requests.get(first_stats_link)
        soup = BeautifulSoup(page.content, 'html.parser')
        stat_titles = soup.findAll('div', {'class': 'kick__stats-bar__title'})
        if stat_titles is None or len(stat_titles)==0:
            print("data not yet available for season ", season, " matchday ", matchday)
            return False
        return True

    def _get_data(self, start_year, end_year, match_day):
        season = str(start_year) + '-' + str(end_year)
        game_dict = self._get_links_matchday(season, match_day)
        for game_id, link_dict in game_dict.items():
            self._get_match_info(link_dict.get("match_info"), season, match_day, game_id)
            self._get_team_stats(link_dict.get("team_stats"), season, match_day, game_id)
            self._get_player_stats(link_dict.get("player_stats"), season, match_day, game_id)


    def _get_links_matchday(self, season, match_day):
        URL = self.base_url + "/" + self.league + "/spieltag/20" + str(season) + '/' + str(match_day)
        #print(URL)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.findAll('a', {'class': 'kick__v100-scoreBoard kick__v100-scoreBoard--standard'})
        link_dict = dict()
        for game in results:
            link = game.get('href')
            game_id_string = (str(self.league)+str(season)+str(match_day)+str(link)).encode('utf-8')
            game_id = str(hashlib.md5(game_id_string).hexdigest())
            link_dict[game_id] = dict()
            link_dict[game_id]["team_stats"] = self.base_url + link.replace("analyse", "spieldaten").replace("spielbericht", "spieldaten")
            link_dict[game_id]["player_stats"] = self.base_url + link.replace("analyse", "schema").replace("spielbericht", "schema")
            link_dict[game_id]["match_info"] = self.base_url + link.replace("analyse", "spielinfo").replace("spielbericht", "spielinfo")

        return link_dict

    def _get_match_info(self, URL, season, match_day, game_id):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        kick_off = None
        referee = None
        weekday = None
        for info in soup.findAll('div', {'class': 'kick__gameinfo-block'}):
            found = False
            for tag in info.findAll('span', {'class': 'kick__weekday_box'}):
                weekday = tag.text
                found = True
            if found:
                for p in info.findAll('p'):
                    kick_off = p.text.split(weekday)[1].split('\\')[0].strip()

        for info in soup.findAll('strong', {'class': 'kick__gameinfo__person'}):
            for ref in info.findAll('a'): referee = ref.get("href").split("/")[1]+ "/" + ref.get("href").split("/")[2]

        match_info_dict = {"game_id": game_id, "season": season, "match_day": match_day, "weekday":weekday, "kick_off_time":kick_off, "referee": referee}
        self.match_info.append(match_info_dict)

    def _get_team_stats(self, URL, season, match_day, game_id):
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')

        teams = soup.findAll('div', {'class': 'kick__v100-gameCell__team__name'})
        stat_titles = soup.findAll('div', {'class': 'kick__stats-bar__title'})
        stats_home = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent1'})
        stats_away = soup.findAll('div', {'class': 'kick__stats-bar__value kick__stats-bar__value--opponent2'})
        score = soup.findAll('div', {'class': 'kick__v100-scoreBoard__scoreHolder__score'})
        positions = soup.findAll('div', {'class': 'kick__v100-gameCell__team__info'})
        stats_home_dict = {"game_id": game_id, "season": season, "match_day": match_day, "indicator": "home", "team_name": teams[0].text, "ht_goals":score[2].text, "position": positions[0].text}
        stats_away_dict = {"game_id": game_id, "season": season, "match_day": match_day, "indicator": "away", "team_name": teams[1].text, "ht_goals":score[3].text, "position": positions[1].text}

        for stat_titles, stats_home, stats_away in zip(stat_titles, stats_home, stats_away):
            column_name = self.mapping_columns.get("team_stats").get(stat_titles.text)
            if column_name is None: print("column not found in mapping: ", stat_titles.text)
            stats_home_dict[column_name] = stats_home.text
            stats_away_dict[column_name] = stats_away.text
        self.team_stats.append(stats_home_dict)
        self.team_stats.append(stats_away_dict)

    def _get_player_stats(self, URL, season, matchday, game_id):
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

        substitution_section = None
        cards_section = None
        section_items = soup.findAll('section', {'class': 'kick__section-item'})
        for section_item in section_items:
            for element in section_item.findAll('h4', {'class': 'kick__card-headline kick__text-center'}):
                if element.text == "Wechsel": substitution_section = section_item
            for element in section_item.findAll('h4', {'class': 'kick__card-headline kick__text-center'}):
                if element.text == "Karten": cards_section = section_item

        if substitution_section is None: substitution_players = None
        else: substitution_players = substitution_section.findAll('a', {'class': 'kick__substitutions__player'})
        if substitution_section is None: substitution_times = None
        else: substitution_times = substitution_section.findAll('div', {'class': 'kick__substitutions__time'})

        if cards_section is None: cards = None
        else: cards = cards_section.findAll('div', {'class': 'kick__substitutions__cell'})

        if substitution_players is None: substitution_players = []
        if substitution_times is None: substitution_times = []
        if cards is None: cards = []

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
                        self.player_stats.append({"game_id": game_id, "season": season, "matchday":matchday, "indicator": indicator, "player_name": name, "goals": scorer_dict.get(name), "sub_in":substitution_dict.get("in").get(name), "sub_out":substitution_dict.get("out").get(name), "card_time":card_entry.get("card_time"), "card_description":card_entry.get("card_description"), "yellow_card":card_entry.get("yellow_card"), "yellow_red_card":card_entry.get("yellow_red_card"), "red_card":card_entry.get("red_card")})
                    if "/trainer" in player.get("href"):
                        self.coach_data.append({"game_id": game_id, "season": season, "matchday":matchday, "indicator": indicator, "coach_name": name})

    def _save(self, season, start_matchday, end_matchday):
        self._store_data(self.match_info, "match_info", season, start_matchday, end_matchday)
        self._store_data(self.team_stats, "team_stats", season, start_matchday, end_matchday)
        self._store_data(self.player_stats, "player_stats", season, start_matchday, end_matchday)
        self._store_data(self.coach_data, "coaches", season, start_matchday, end_matchday)
        self._reset_data_lists()

    def _store_data(self, data_list, table_name, season, start_matchday, end_matchday):
        df = pd.DataFrame(data_list)
        df["league_code"] = self.mapping_leagues.get(self.league).get("code")
        df = df.drop_duplicates()
        df.to_parquet("./data/bronze/"+table_name+"/" + str(self.league) + "_" + str(season) + "_" + str(season+1) + "_" + str(start_matchday) + "_" + str(end_matchday) + ".parquet", index=False)
        print("finished storing ",table_name," for season ", str(season), " matchday ", start_matchday," until ", end_matchday," of ", self.league)

        jb_entry = JobBookmark.get_data_scraped("kicker_scraper")
        if jb_entry.get(self.league) is None: jb_entry[self.league] = dict()
        if jb_entry.get(self.league).get(season) is None: jb_entry[self.league][season] = dict()
        for d in range(start_matchday,end_matchday+1,1):
            jb_entry[self.league][season][d] = True
        JobBookmark.update_bookmark("kicker_scraper", jb_entry)

    def _reset_data_lists(self):
        self.match_info = []
        self.team_stats = []
        self.player_stats = []
        self.coach_data = []
        self.player_attributes = []

    @classmethod
    def delete_bronze_data(self):
        tables = ["coaches","match_info","player_stats","team_stats"]
        for table in tables:
            files = glob.glob('./data/bronze/'+table+"/*")
            for f in files:
                os.remove(f)