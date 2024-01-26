import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Pool
import hashlib
from .job_bookmark import JobBookmark
from copy import deepcopy
from tqdm import tqdm

import os
import glob


class EloCalculator:

    def __init__(self):
        self.load_timestamp = str(datetime.now())

    @classmethod
    def calculcate_new_elos(cls, h_team_elo, a_team_elo, h_goals, a_goals, k=20):
        # goal diff
        d_diff = abs(h_goals-a_goals)
        if d_diff <= 1: g = 1
        elif d_diff == 2: g = 1.5
        else: g = (11 + abs(h_goals-a_goals))/8

        #outcome
        w = 0.5
        if h_goals > a_goals:
            w = 1
        if h_goals < a_goals:
            w = 0

        #expected outcome
        w_e = 1/(10 ** (-(h_team_elo-a_team_elo)/400) + 1)

        #elo change
        p = k * g * (w-w_e)
        h_team_new_elo = h_team_elo + p
        a_team_new_elo = a_team_elo - p
        return h_team_new_elo, a_team_new_elo