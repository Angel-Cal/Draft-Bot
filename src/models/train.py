"""Model training module for player projections."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any
import joblib
import logging

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate player projection models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the model trainer.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir or Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.models: dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.feature_columns: list[str] = []

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "fantasy_points",
        exclude_cols: Optional[list[str]] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Args:
            df: DataFrame with features
            target_col: Target column name
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        exclude = exclude_cols or []
        exclude.extend([target_col, "player", "player_clean", "team", "season"])

        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        # Remove excluded columns
        feature_cols = [c for c in numeric_df.columns if c not in exclude]
        self.feature_columns = feature_cols

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle any remaining NaN values
        X = X.fillna(0)

        return X, y

    def train_test_split_by_season(
        self,
        df: pd.DataFrame,
        test_season: int,
        target_col: str = "fantasy_points"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data by season to avoid data leakage.

        Args:
            df: DataFrame with features and target
            test_season: Season to use for testing
            target_col: Target column name

        Returns:
            X_train, X_test, y_train, y_test
        """
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]

        X_train, y_train = self.prepare_features(train_df, target_col)
        X_test, y_test = self.prepare_features(test_df, target_col)

        return X_train, X_test, y_train, y_test

    def get_models(self) -> dict[str, Any]:
        """
        Get dictionary of models to train.

        Returns:
            Dictionary of model name -> model instance
        """
        return {
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.1),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> dict[str, dict[str, float]]:
        """
        Train multiple models and evaluate performance.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target

        Returns:
            Dictionary of model results
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        models = self.get_models()

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Train
            model.fit(X_train_scaled, y_train)
            self.models[name] = model

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }

            logger.info(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

        # Find best model by MAE
        best_name = min(results, key=lambda x: results[x]["mae"])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        logger.info(f"Best model: {best_name}")

        return results

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> dict[str, dict[str, float]]:
        """
        Perform cross-validation on all models.

        Args:
            X: Features
            y: Target
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        X_scaled = self.scaler.fit_transform(X)

        results = {}
        models = self.get_models()

        for name, model in models.items():
            logger.info(f"Cross-validating {name}...")

            # Use TimeSeriesSplit for proper temporal validation
            tscv = TimeSeriesSplit(n_splits=cv)

            scores = cross_val_score(
                model, X_scaled, y,
                cv=tscv,
                scoring="neg_mean_absolute_error"
            )

            results[name] = {
                "mean_mae": -scores.mean(),
                "std_mae": scores.std()
            }

            logger.info(f"{name}: MAE={-scores.mean():.2f} (+/- {scores.std():.2f})")

        return results

    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model_name: Name of model (uses best model if None)

        Returns:
            DataFrame with feature importances
        """
        model_name = model_name or self.best_model_name
        model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found")

        # Get importance based on model type
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            raise ValueError(f"Model {model_name} doesn't support feature importance")

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self, model_name: Optional[str] = None, filename: Optional[str] = None):
        """
        Save a trained model to disk.

        Args:
            model_name: Name of model to save (uses best if None)
            filename: Output filename
        """
        model_name = model_name or self.best_model_name
        model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found")

        filename = filename or f"{model_name}_model.joblib"
        filepath = self.models_dir / filename

        # Save model and scaler together
        model_data = {
            "model": model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "model_name": model_name
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to {filepath}")

    def load_model(self, filename: str) -> dict[str, Any]:
        """
        Load a trained model from disk.

        Args:
            filename: Model filename

        Returns:
            Dictionary with model, scaler, and metadata
        """
        filepath = self.models_dir / filename
        model_data = joblib.load(filepath)

        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.best_model = model_data["model"]
        self.best_model_name = model_data["model_name"]
        self.models[self.best_model_name] = self.best_model

        logger.info(f"Loaded {model_data['model_name']} from {filepath}")
        return model_data


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    sample_data = pd.DataFrame({
        "season": [2022] * 50 + [2023] * 50,
        "fantasy_points": np.random.normal(150, 50, 100),
        "rushing_yds": np.random.normal(500, 200, 100),
        "receiving_yds": np.random.normal(400, 150, 100),
        "targets": np.random.normal(60, 20, 100),
        "age": np.random.randint(22, 35, 100),
    })

    trainer = ModelTrainer()

    # Split by season
    X_train, X_test, y_train, y_test = trainer.train_test_split_by_season(
        sample_data, test_season=2023
    )

    # Train and evaluate
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    # Show feature importance
    print("\nFeature Importance:")
    print(trainer.get_feature_importance())

    # Save best model
    trainer.save_model()
