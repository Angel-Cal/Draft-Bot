import pandas as pd
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.build_features import build_features


RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

def load_data():
    seasonal = pd.read_parquet(RAW_DIR /'seasonal_20_25.parquet')
    # pbp = pd.read_parquet('../data/raw/pbp_20_25.parquet')
    roster = pd.read_parquet(RAW_DIR /'seasonal_roster_20_25.parquet')
    # id = pd.read_parquet('../data/raw/ids.parquet')

    return seasonal, roster

def clean_data(roster, season):
    ROSTER_KEEP_COLS = [
        "player_id",
        "season",
        "team",
        "position",
        "age",
        "height",
        "weight",
        "years_exp",
        "college",
        "rookie_year",
        "draft_number",
        "draft_club",
    ]
    SEASON_KEEP_COLS = [
        # Keys
        "player_id", "season", "season_type", "games",

        # Passing
        "completions", "attempts", "passing_yards", "passing_tds",
        "interceptions", "sacks", "passing_first_downs", "passing_epa",

        # Rushing
        "carries", "rushing_yards", "rushing_tds",
        "rushing_first_downs", "rushing_epa",

        # Receiving
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_first_downs", "receiving_epa",

        # Efficiency
        "pacr", "racr", "dakota", "yptmpa",

        # Opportunity / role
        "target_share", "air_yards_share", "wopr_x", "dom", "w8dom",

        # Fantasy outputs
        "fantasy_points", "fantasy_points_ppr", "ppr_sh"
    ]

    filtered_roster = roster[roster["position"].isin(["QB", "RB", "WR", "TE"])] # Only Keep relevant positions
    season_filtered = season[SEASON_KEEP_COLS]
    end_roster = (
        filtered_roster[ROSTER_KEEP_COLS + ["week"]]
        .sort_values(["season", "week"])
        .groupby(["player_id", "season"], as_index=False)
        .last()
        .drop(columns=["week"], errors="ignore")
        )
    assert end_roster.groupby(["player_id", "season"]).size().max() == 1


    merged_df = season_filtered.merge(end_roster, on=["player_id", "season"], how = "inner")
    merged_df = merged_df[merged_df['games'] >= 4]  # Filter out low usage players (noise)
    merged_df['draft_number'] = merged_df['draft_number'].replace(['nan', 'None'], '300').astype(float)


    return merged_df



def save_data(df):
    filepath = PROCESSED_DIR / 'processed_data.parquet'
    df.to_parquet(filepath)

if __name__ == "__main__":
    seasonal, roster = load_data()
    df = clean_data(roster, seasonal)
    df = build_features(df)
    save_data(df)
    print(f"Saved {len(df)} rows to {PROCESSED_DIR}")

    





