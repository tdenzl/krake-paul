from datetime import date, timedelta, datetime
import json
import os
from os.path import exists

class JobBookmark():

    @classmethod
    def _initialize_bookmark(self, job_name):
        bookmark = {}
        metadata = {}
        metadata["job_name"] = job_name
        metadata["job_runs"] = 0

        bookmark['metadata'] = metadata
        bookmark['job_history'] = []
        bookmark["data_scraped"] = dict()
        return bookmark

    @classmethod
    def update_bookmark(self, job_name, dict):
        job_bookmark, bookmark_path = self.get_bookmark(job_name)
        job_bookmark["data_scraped"] = dict
        job_bookmark['latest_update'] = str(datetime.now())
        self._write_bookmark(job_bookmark, bookmark_path)

    @classmethod
    def get_bookmark(self, job_name):
        bookmark_path = "./job_bookmark/" + job_name + '.json'
        if not exists(bookmark_path):
            job_bookmark = self._initialize_bookmark(job_name)
        else:
            job_bookmark = json.load(open(bookmark_path, 'r'))
        return job_bookmark, bookmark_path

    @classmethod
    def _write_bookmark(self, job_bookmark, bookmark_path):
        with open(bookmark_path, 'w') as fp:
            json.dump(job_bookmark, fp)

    @classmethod
    def delete_bookmark(self, job_name):
        os.remove("./job_bookmark/" + job_name + '.json')

    @classmethod
    def get_latest_update(self, job_name):
        return self.get_bookmark(job_name).get("latest_update")

    @classmethod
    def get_data_scraped(self, job_name):
        return self.get_bookmark(job_name)[0].get("data_scraped")