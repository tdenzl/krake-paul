from .scraper import Scraper

class Executor:

    def execute(self):
        league_dict = {"bundesliga":"GER1", "premier-league":"ENG1", "la-liga":"ESP1", "serie-a":"ITA1", "ligue-1":"FRA1"}
        for league, code in league_dict.items():
            scraper = Scraper(league, 21, 22, 1)
            scraper.scrape()
            scraper.store_data(code)
