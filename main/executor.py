from .kicker_scraper import KickerScraper
from .fifa_scraper import FifaScraper
from .job_bookmark import JobBookmark
from .preprocessor import Preprocessor

from .ingestor import Ingestor
from .model_train_v1 import ModelV1
from .model_train_v2 import ModelV2
from .model_train_v3 import ModelV3

from .betmaker import BetMaker


from multiprocessing import Pool
import os
import json
class Executor:

    def execute(self):
        self.start_year = 13
        self.end_year = 24
        #self._scrape_kicker_data_multiprocessing()
        #self._scrape_fifa_rating_data()
        #self._preprocess()
        #self._ingestion()
        #self._train()
        self._predict()
        #self._evaluate()


    def _scrape_kicker_data(self, load_type="latest"):
        if load_type == "full":
            KickerScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("kicker_scraper")
        for league in json.load(open('./config/mapping_leagues.json', 'r')).keys():
            kicker_scraper = KickerScraper(league, self.start_year, self.end_year)
            kicker_scraper.scrape()
            print("finished ", league, " scraping")

    def _scrape_fifa_rating_data(self, load_type="latest"):
        if load_type == "full":
            FifaScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("fifa_scraper")
        for year in json.load(open('./config/mapping_fifa.json', 'r')).keys():
            scraper = FifaScraper(year)
            scraper.scrape()
            print("finished year ", year, " scraping")

    def _scrape_kicker_data_multiprocessing(self, load_type="latest"):
        if load_type == "full":
            FifaScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("fifa_scraper")
        job_list = []
        for league in json.load(open('./config/mapping_leagues.json', 'r')).keys():
            job_list.append({"league":league, "start_season":13, "end_season":24})

        print("cpu_count=", os.cpu_count())
        if os.cpu_count() == 1:
            for job_dict in job_list:
                self._execute_kicker_scraping_sub_job(job_dict)
        else:
            pool = Pool(os.cpu_count() - 1)  # Create a multiprocessing Pool os.cpu_count()//2
            pool.map(self._execute_kicker_scraping_sub_job, job_list)  # process data_inputs iterable with pool

    def _execute_kicker_scraping_sub_job(self, job_entry):
        kicker_scraper = KickerScraper(job_entry.get("league"), job_entry.get("start_season"), job_entry.get("end_season"))
        kicker_scraper.scrape()
        print("finished ", job_entry.get("league"), " scraping")

    def _preprocess(self):
        print("starting preprocessing")
        #Preprocessor.preprocess_table("coaches")
        #Preprocessor.preprocess_table("match_info")
        #Preprocessor.preprocess_table("team_stats")
        #Preprocessor.preprocess_table("player_stats")
        #Preprocessor.preprocess_table("player_ratings")
        #Preprocessor.preprocess_table("team_elo")
        #Preprocessor.preprocess_table("team_lin_regs")
        #Preprocessor.preprocess_table("player_mapping")
        #Preprocessor.preprocess_table("team_fifa_rating")
        #Preprocessor.preprocess_table("referee_profiles")
        #Preprocessor.preprocess_table("player_elo")
        #Preprocessor.preprocess_table("relationships")


    def _ingestion(self):
        Ingestor.create_ingestion_data()


    def _train(self):
        #ModelV1.train()
        #ModelV2.train()
        ModelV3.train()

    def _predict(self):
        #ModelV1.train()
        #ModelV2.train()
        ModelV3.predict()

    def _evaluate(self):
        BetMaker.evaluate_bets()
