"""NFL data scraping and API integration module."""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from typing import Optional
import time
import logging
import nfl_data_py as nfl
from pathlib import Path



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _import_data():
    file_path = DATA_DIR/ "seasonal_20_25.parquet"
    raw_seasonal = nfl.import_seasonal_data(range(2014, 2025))
    raw_seasonal.to_parquet(file_path)

    file_path = DATA_DIR/ "seasonal_roster_20_25.parquet"
    raw_roster = nfl.import_seasonal_rosters(range(2014, 2025))
    for col in ['jersey_number', 'draft_number', 'draft_club']:    #older data has mixed-type cols
        raw_roster[col] = raw_roster[col].astype(str)    
    raw_roster.to_parquet(file_path)

    file_path = DATA_DIR/ "ids.parquet"
    raw_id = nfl.import_ids()
    raw_id.to_parquet(file_path)

    file_path = DATA_DIR/ "weekly_20_25"
    raw_weekly = nfl.import_weekly_data(range(2014, 2025))
    raw_weekly.to_parquet(file_path)

    file_path = DATA_DIR/ "pbp_20_25.parquet"
    raw_pbp = nfl.import_pbp_data(range(2014, 2025))
    raw_pbp.to_parquet(file_path)

    file_path = DATA_DIR/ "snap_counts_20_25.parquet"
    raw_snap_counts = nfl.import_snap_counts(range(2014,2025))
    raw_snap_counts.to_parquet(file_path)

    


if __name__ == "__main__":
    _import_data()

