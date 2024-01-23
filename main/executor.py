from .kicker_scraper import KickerScraper
from .fifa_scraper import FifaScraper
from .job_bookmark import JobBookmark

import json
class Executor:

    def execute(self):
        #self._scrape_kicker_data()
        self._scrape_fifa_rating_data()

    def _scrape_kicker_data(self, load_type="latest"):
        league_dict = json.load(open('./config/mapping_leagues.json', 'r'))
        if load_type == "full":
            KickerScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("kicker_scraper")
        for league, code in league_dict.items():
            kicker_scraper = KickerScraper(league, 13, 24)
            kicker_scraper.scrape()
            print("finished ", league, " scraping")

    def _scrape_fifa_rating_data(self):
        FifaScraper.delete_bronze_data()
        scraper = FifaScraper(24)
        scraper.scrape()