import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import hashlib
from .job_bookmark import JobBookmark
from copy import deepcopy
from tqdm import tqdm

import os
import glob

class PlayerMapper:

    def __init__(self):
        kicker_path = ""

