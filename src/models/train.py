import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
import sklearn.metrics as lk
from pathlib import Path
import matplotlib.pyplot as plt



class Model:
    def __init__(self, mode = "simple", train_range = (range(2015, 2023)), val_season = 2023, test_season = 2024):
        self.mode = mode
        self.train_range = train_range
        self.val_season = val_season
        self.test_season = test_season

    def temporal_splits(self, df):
        if self.mode == "simple":
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
        elif self.mode == "rolling":
            raise NotImplementedError("Rolling mode to be implemented in v2")

        else:
            raise ValueError(f"unknown mode: {self.mode}")

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




if __name__ == "__main__":
    filepath = Path(__file__).parent.parent.parent / "data" / "processed"/ 'processed_data.parquet'

    df = pd.read_parquet(filepath)
    model = Model()
    x_train, x_val, x_test, y_train, y_val, y_test = model.temporal_splits(df)
    positions = x_test["position"]
    trained_model = model.train_model(x_train, x_val, y_train, y_val)
    model.evaluate_per_position(trained_model, x_test, y_test, positions)

    # mae, r2, rmse = model.evaluate_model(trained_model, x_test, y_test)
    # print("MAE: ", mae, "R2: ", r2, "RMSE: ", rmse, "\n")
    # lgb.plot_importance(trained_model, max_num_features=15, importance_type="gain")
    # plt.show()
