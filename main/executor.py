from .scraper import Scraper
import json
class Executor:

    def execute(self):
        league_dict = json.load(open('./config/mapping_leagues.json', 'r'))
        Scraper.delete_bronze_data()
        for league, code in league_dict.items():
            scraper = Scraper(league, 13, 24)
            scraper.scrape()
            print("finished ", league, " scraping")