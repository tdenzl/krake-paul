from .kicker_scraper import KickerScraper
from .fifa_scraper import FifaScraper
from .job_bookmark import JobBookmark
from multiprocessing import Pool
import time
import os
import json
class Executor:

    def execute(self):
        #self._scrape_kicker_data()
        self._scrape_kicker_data_multiprocessing()
        #self._scrape_fifa_rating_data()

    def _scrape_kicker_data(self, load_type="latest"):
        if load_type == "full":
            KickerScraper.delete_bronze_data()
            JobBookmark.delete_bookmark("kicker_scraper")
        for league in json.load(open('./config/mapping_leagues.json', 'r')).keys():
            kicker_scraper = KickerScraper(league, 13, 24)
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