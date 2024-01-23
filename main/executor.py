from .kicker_scraper import KickerScraper
from .fifa_scraper import FifaScraper
from .job_bookmark import JobBookmark

import json
class Executor:

    def execute(self):
        #self._scrape_kicker_data()
        self._scrape_fifa_rating_data()

    def _scrape_kicker_data(self, load_type="latest"):
        if load_type == "full":
            KickerScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("kicker_scrape")
        for league in json.load(open('./config/mapping_leagues.json', 'r')).keys():
            kicker_scraper = KickerScraper(league, 13, 24)
            kicker_scraper.scrape()
            print("finished ", league, " scraping")

    def _scrape_fifa_rating_data(self, load_type="latest"):
        if load_type == "full":
            FifaScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("fifa_scrape")
        for year in json.load(open('./config/mapping_fifa.json', 'r')).keys():
            scraper = FifaScraper(year)
            scraper.scrape()
            print("finished year ", year, " scraping")