import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import sklearn.metrics as lk
from pathlib import Path
import matplotlib.pyplot as plt



class Model:
    def __init__(self, train_range = (range(2015, 2023)), val_season = 2023, test_season = 2024):
        self.train_range = train_range
        self.val_season = val_season
        self.test_season = test_season

    def temporal_splits(self, df):
        x_train = df[df["season"].isin(self.train_range)]
        y_train = x_train["target_ppr"]
        x_train = x_train.drop(columns = ["target_ppr", "season"])

        x_val = df[df["season"] == self.val_season]
        y_val = x_val["target_ppr"]
        x_val = x_val.drop(columns =["target_ppr", "season"])

        x_test = df[df["season"] == self.test_season]
        y_test = x_test["target_ppr"]
        x_test = x_test.drop(columns = ["target_ppr", "season"])

        return x_train, x_val, x_test, y_train, y_val, y_test
     
        
    def rolling_splits(self, df, min_train_seasons=6, offset=1):
        FIRST_SEASON = 2015
        LAST_SEASON = 2024
        df_list =[]
        first_test = FIRST_SEASON + min_train_seasons
        for test_season in range(first_test, LAST_SEASON + 1):
            val_season = test_season - offset
            train_range = range(FIRST_SEASON, val_season)

            x_train = df[df["season"].isin(train_range)]
            y_train = x_train["target_ppr"]
            x_train = x_train.drop(columns = ["target_ppr", "season"])

            x_val = df[df["season"] == val_season]
            y_val = x_val["target_ppr"]
            x_val = x_val.drop(columns =["target_ppr", "season"])

            x_test = df[df["season"] == test_season]
            y_test = x_test["target_ppr"]
            x_test = x_test.drop(columns = ["target_ppr", "season"])
            
            df_list.append((x_train, x_val, x_test, y_train, y_val, y_test)) 
        return df_list
    
    def walk_forward(self, df_list):
        metrics_list = []
        for fold in df_list:
           x_train, x_val, x_test, y_train, y_val, y_test = fold
           trained_model= self.train_model(x_train, x_val, y_train, y_val)
           mae, r2, rmse = self.evaluate_model(trained_model, x_test, y_test)
           metrics_list.append((mae, r2, rmse))
        return metrics_list
            



    def train_model(self, x_train, x_val, y_train, y_val):
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            min_child_weight=1e-3,
            reg_alpha=.1,
            reg_lambda=1.0,
            subsample=.8,
            subsample_freq=1,
            colsample_bytree=.8,
            random_state=42,
            verbosity=-1,
            objective="regression",
            n_jobs=-1
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        return model

    def evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        mae = lk.mean_absolute_error(y_test, y_pred)
        r2 = lk.r2_score(y_test, y_pred)
        rmse = lk.root_mean_squared_error(y_test, y_pred)
        return mae, r2, rmse

    def evaluate_per_position(self, model, x_test, y_test, positions):
        y_pred = model.predict(x_test)
        for pos in positions.unique():
            mask = positions == pos
            mae = lk.mean_absolute_error(y_test[mask], y_pred[mask])
            r2 = lk.r2_score(y_test[mask], y_pred[mask])
            rmse = lk.root_mean_squared_error(y_test[mask], y_pred[mask])
            print("Metric for position: ", pos, "\nMAE: ", mae, "\nr2:", r2, "\nRMSE: ", rmse, "\n")

    def ablate(self, x_train, x_test, x_val, exclude_patterns = []):
        drop_cols = [c for c in x_train.columns if any(p in c for p in exclude_patterns)]
        x_train = x_train.drop(columns = drop_cols)
        x_test = x_test.drop(columns = drop_cols)
        x_val = x_val.drop(columns = drop_cols)
        return x_train, x_test, x_val

if __name__ == "__main__":
    filepath = Path(__file__).parent.parent.parent / "data" / "processed"/ 'processed_data.parquet'

    df = pd.read_parquet(filepath)
    model = Model()
    x_train, x_val, x_test, y_train, y_val, y_test = model.temporal_splits(df)
    x_train_org = x_train
    x_val_org = x_val
    x_test_org = x_test
    positions = x_test["position"]
    trained_model = model.train_model(x_train, x_val, y_train, y_val)
    mae, r2, rmse = model.evaluate_model(trained_model, x_test, y_test)
    print("Baseline:\nMAE: ", mae, "R2: ", r2, "RMSE: ", rmse, "\n")

    df_list = model.rolling_splits(df)
    metrics_list = model.walk_forward(df_list)
    counter =1
    for metrics in metrics_list:
        mae, r2, rmse = metrics
        print("Fold ", counter, "metrics\nMAE: ", mae, "R2: ", r2, "RMSE: ", rmse )
        counter+=1

    # # Ablate Deltas
    # x_train, x_test, x_val = model.ablate(x_train_org, x_test_org, x_val_org, exclude_patterns=['_delta'])
    # trained_model = model.train_model(x_train, x_val, y_train, y_val)
    # mae, r2, rmse = model.evaluate_model(trained_model, x_test, y_test)
    # print("Delta Ablation:\nMAE: ", mae, "R2: ", r2, "RMSE: ", rmse, "\n")

    # # Ablate surge
    # x_train, x_test,x_val = model.ablate(x_train_org, x_test_org, x_val_org,exclude_patterns=['late_'])
    # trained_model = model.train_model(x_train, x_val, y_train, y_val)
    # mae, r2, rmse = model.evaluate_model(trained_model, x_test, y_test)
    # print("Surge Ablation:\nMAE: ", mae, "R2: ", r2, "RMSE: ", rmse, "\n")

    # # Ablate both
    # x_train, x_test, x_val = model.ablate(x_train_org, x_test_org, x_val_org, exclude_patterns=['late_', '_delta'])
    # trained_model = model.train_model(x_train, x_val, y_train, y_val)
    # mae, r2, rmse = model.evaluate_model(trained_model, x_test, y_test)
    # print("Delta + Surge Ablation:\nMAE: ", mae, "R2: ", r2, "RMSE: ", rmse, "\n")

  

    #model.evaluate_per_position(trained_model, x_test, y_test, positions)

   

    # lgb.plot_importance(trained_model, max_num_features=15, importance_type="gain")
    # plt.show()
