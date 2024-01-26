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
        #calculate g
        d_diff = abs(h_goals-a_goals)
        if d_diff <= 1: g = 1
        elif d_diff == 2: g = 1.5
        else: g = (11 + abs(h_goals-a_goals))/8
        print("g=",g)

        w_home = 0.5
        w_away = 0.5
        #outcome
        w = 0.5
        if h_goals > a_goals:
            w_home = 1; w_away = 0
        if h_goals < a_goals:
            w_home = 0; w_away = 1

        #expected outcome
        w_e_home = 1/(10 ** (-(h_team_elo - a_team_elo)/400) + 1)
        w_e_away = 1 / (10 ** (-(a_team_elo - h_team_elo) / 400) + 1)
        print("w_e_home=", w_e_home)
        print("w_e_away=", w_e_away)

        #elo change
        p_home = k * g * (w-w_e_home)
        p_away = k * g * (w-w_e_away)
        print("p_home=", p_home)
        print("p_away=", p_away)

        h_team_new_elo = h_team_elo + p_home
        a_team_new_elo = a_team_elo + p_away

        return h_team_new_elo, a_team_new_elo