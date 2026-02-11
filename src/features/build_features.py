import pandas as pd
from pathlib import Path
import numpy as np

DELTA_SHIFT_COLS = [
        "games",
        "attempts",
        "carries",
        "targets",
        "completions_pg",
        "attempts_pg",
        "passing_yards_pg",
        "passing_tds_pg",
        "interceptions_pg",
        "sacks_pg",
        "passing_first_downs_pg",    
        "carries_pg",
        "rushing_yards_pg",
        "rushing_tds_pg",
        "rushing_first_downs_pg",
        "targets_pg",
        "receptions_pg",
        "receiving_yards_pg",
        "receiving_tds_pg",
        "receiving_first_downs_pg",
        "target_share",
        "air_yards_share",
        "wopr_x",
        "dom",
        "w8dom" 
    ]    
def add_eras(df):
    conditions = [
        df['season'] <= 2016,
        df['season'].between(2017, 2019),
        df['season'] >= 2020
    ]

    choices = [
        'pre_rpo',
        'early_rpo',
        'modern'
    ]

    df['era'] = np.select(conditions, choices)
    return df

def add_per_game(df):
    PER_GAME_COLS = [
    # Passing volume
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "passing_first_downs",

    # Rushing volume
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",

    # Receiving volume
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_first_downs",
]

    for col in PER_GAME_COLS:
        df[f"{col}_pg"] = df[col]/df["games"]
    df.loc[df["games"] ==0, [f"{c}_pg" for c in PER_GAME_COLS]]= 0
    return df

def add_lag(df):
    LAG_COLS = [
    # total volume features
    "games",
    "attempts",
    "carries",
    "targets",


    # Per-game volume features
    "completions_pg",
    "attempts_pg",
    "passing_yards_pg",
    "passing_tds_pg",
    "interceptions_pg",
    "sacks_pg",
    "passing_first_downs_pg",

    "carries_pg",
    "rushing_yards_pg",
    "rushing_tds_pg",
    "rushing_first_downs_pg",

    "targets_pg",
    "receptions_pg",
    "receiving_yards_pg",
    "receiving_tds_pg",
    "receiving_first_downs_pg",

    # EPA & efficiency (RAW, NOT per-game)
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "pacr",
    "racr",
    "dakota",
    "yptmpa",

    # Opportunity / role shares
    "target_share",
    "air_yards_share",
    "wopr_x",
    "dom",
    "w8dom",

    # Prior fantasy output (RAW totals)
    "fantasy_points",
    "fantasy_points_ppr",
    "ppr_sh",
]
    
    df["target_ppr"] = df["fantasy_points_ppr"]
    delta_df = df.copy()
    delta_df[DELTA_SHIFT_COLS] = (df.groupby("player_id")[DELTA_SHIFT_COLS].shift(2))
    df[LAG_COLS] = (df.groupby("player_id")[LAG_COLS].shift(1))
    df = df.dropna(subset=LAG_COLS)
    return df, delta_df

def compute_deltas(df, delta_df):
    for col in DELTA_SHIFT_COLS:
        df[f"{col}_delta"] = df[col] - delta_df[col]
    delta_cols = [f"{col}_delta" for col in DELTA_SHIFT_COLS]
    df[delta_cols] = df[DELTA_SHIFT_COLS].fillna(0)
    return df
 
    
    
def remove_cols(df):
    DROP_COLS = [
        # Passing outcomes
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks",
        "passing_first_downs",

        # Rushing outcomes
        "rushing_yards",
        "rushing_tds",
        "rushing_first_downs",

        # Receiving outcomes
        "receiving_yards",
        "receiving_tds",
        "receiving_first_downs",
        "receptions",

        # General outcome totals
        "completions",
        
        # Unecessary categoricals
        "team",
        "college",
        "draft_club",
        "season_type",
        "player_id"
    ]
    df = df.drop(columns = DROP_COLS)
    return df

def convert_categoricals(df):
    df['position'] = df['position'].astype('category')
    df['era'] = df['era'].astype('category')
    return df


def build_features(df):
    df = df.sort_values(['player_id', 'season'])
    df = add_per_game(df)
    df = add_eras(df)
    df = convert_categoricals(df)
    df, delta_df = add_lag(df)
    df = compute_deltas(df, delta_df)
    df = remove_cols(df)
    return df
