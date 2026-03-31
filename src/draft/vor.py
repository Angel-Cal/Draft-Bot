import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.pipeline import save_data, save_data_as_csv

RMSE = {"QB": 78.68, "RB": 62.10, "WR": 54.87, "TE": 39.25}
REPLACEMENT_RANK = {"QB": 12, "RB": 29, "WR": 41, "TE": 12}

def load_predictions():
    filepath = Path(__file__).parent.parent.parent / "data" / "processed"
    qb_predictions = pd.read_parquet(filepath / "qb_predictions.parquet")
    qb_predictions['position'] = 'QB'
    rb_predictions = pd.read_parquet(filepath / "rb_predictions.parquet")
    rb_predictions['position'] = 'RB'
    wr_predictions = pd.read_parquet(filepath / "wr_predictions.parquet")
    wr_predictions['position'] = 'WR'
    te_predictions = pd.read_parquet(filepath / "te_predictions.parquet")
    te_predictions['position'] = 'TE'
    return qb_predictions, rb_predictions, wr_predictions, te_predictions

def compute_vor(df, baseline_pos):
    df = df.sort_values(by="projected_ppr", ascending=False).reset_index(drop=True)
    df['vor'] = (df["projected_ppr"] - df["projected_ppr"]
                             .iloc[baseline_pos])
                             
    return df

def compute_ceiling_floor(df, rmse):
    mask = (df["games"] < 13) & (df["games_delta"] < -5)
    scaled = df["projected_ppr"] * (17/df["games"])
    capped = np.minimum(scaled, df["projected_ppr"] * 2)
    rmse_ceiling = df["projected_ppr"] + rmse

    df["ceiling"] = np.where(mask, np.maximum(capped, rmse_ceiling), rmse_ceiling)
    df["floor"] = (df["projected_ppr"] - rmse).clip(lower=0)
    df["availability_flag"] = mask

    return df

def compute_value_gap(df):
    df = df.sort_values(by="vor", ascending = False).reset_index(drop=True)
    df["vor_rank"] = df["vor"].rank(ascending=False, method="min")
    df["value_gap"] = df["underdog"] - df["vor_rank"]
    return df

def compute_tiers(df):
    df['tier'] = 8 - pd.qcut(df['vor'], q=8, labels=False)   
    return df

if __name__ == "__main__":


    qb_predictions, rb_predictions, wr_predictions, te_predictions = load_predictions()
    qb_predictions = compute_vor(qb_predictions, REPLACEMENT_RANK["QB"])
    rb_predictions = compute_vor(rb_predictions, REPLACEMENT_RANK["RB"])
    wr_predictions = compute_vor(wr_predictions, REPLACEMENT_RANK["WR"])
    te_predictions = compute_vor(te_predictions, REPLACEMENT_RANK["TE"])

    qb_predictions = compute_ceiling_floor(qb_predictions, RMSE["QB"])
    rb_predictions = compute_ceiling_floor(rb_predictions, RMSE["RB"])
    wr_predictions = compute_ceiling_floor(wr_predictions, RMSE["WR"])
    te_predictions = compute_ceiling_floor(te_predictions, RMSE["TE"])

    predictions_df = pd.concat([qb_predictions, rb_predictions, wr_predictions, te_predictions])
    predictions_df = compute_value_gap(predictions_df)
    predictions_df = compute_tiers(predictions_df)
    save_data(predictions_df, "predictions_table")
    save_data_as_csv(predictions_df, "predictions_table")
    






    


