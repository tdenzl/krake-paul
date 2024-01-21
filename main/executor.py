from .scraper import Scraper

class Executor:

    def execute(self):
        scraper = Scraper("bundesliga", 21, 22, 1)
        scraper.scrape()
        scraper.store_data("GER1")
