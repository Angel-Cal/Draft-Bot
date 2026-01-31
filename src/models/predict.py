"""Prediction module for generating player projections."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerPredictor:
    """Generate player projections using trained models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the predictor.

        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir or Path(__file__).parent.parent.parent / "models"
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_columns: list[str] = []
        self.model_name: Optional[str] = None

    def load_model(self, filename: str = "gradient_boosting_model.joblib"):
        """
        Load a trained model.

        Args:
            filename: Model filename
        """
        filepath = self.models_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.model_name = model_data["model_name"]

        logger.info(f"Loaded {self.model_name} model")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for players.

        Args:
            df: DataFrame with player features

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with 0.")

        # Prepare features
        X = df.reindex(columns=self.feature_columns, fill_value=0)
        X = X.fillna(0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_with_intervals(
        self,
        df: pd.DataFrame,
        confidence: float = 0.9
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals.

        Note: This uses a simple approach based on training residuals.
        For production, consider using quantile regression or
        conformal prediction.

        Args:
            df: DataFrame with player features
            confidence: Confidence level for intervals

        Returns:
            DataFrame with predictions and intervals
        """
        predictions = self.predict(df)

        # Simple interval estimation (placeholder)
        # In production, use proper uncertainty quantification
        std_estimate = np.std(predictions) * 0.3  # Rough estimate

        z_score = 1.645 if confidence == 0.9 else 1.96  # 90% or 95%

        results = pd.DataFrame({
            "prediction": predictions,
            "lower_bound": predictions - z_score * std_estimate,
            "upper_bound": predictions + z_score * std_estimate
        })

        return results

    def generate_projections(
        self,
        df: pd.DataFrame,
        player_col: str = "player",
        position_col: str = "position"
    ) -> pd.DataFrame:
        """
        Generate full projections table for all players.

        Args:
            df: DataFrame with player data
            player_col: Player name column
            position_col: Position column

        Returns:
            DataFrame with player projections sorted by projected points
        """
        predictions = self.predict(df)

        projections = pd.DataFrame({
            "player": df[player_col] if player_col in df.columns else range(len(df)),
            "position": df[position_col] if position_col in df.columns else "Unknown",
            "projected_points": predictions
        })

        # Sort by projected points
        projections = projections.sort_values("projected_points", ascending=False)
        projections["rank"] = range(1, len(projections) + 1)

        # Add position rank
        projections["position_rank"] = (
            projections.groupby("position")["projected_points"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        return projections.reset_index(drop=True)

    def get_top_players(
        self,
        projections: pd.DataFrame,
        position: Optional[str] = None,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N players, optionally filtered by position.

        Args:
            projections: Projections DataFrame
            position: Filter by position (None for all)
            n: Number of players to return

        Returns:
            Top N players
        """
        df = projections.copy()

        if position:
            df = df[df["position"] == position]

        return df.head(n)


if __name__ == "__main__":
    # Example usage
    predictor = PlayerPredictor()

    try:
        predictor.load_model()

        # Create sample data
        sample_players = pd.DataFrame({
            "player": ["Patrick Mahomes", "Josh Allen", "Lamar Jackson"],
            "position": ["QB", "QB", "QB"],
            "passing_yds": [5000, 4500, 3500],
            "passing_td": [40, 35, 25],
            "rushing_yds": [200, 500, 1000],
            "age": [28, 27, 26],
        })

        projections = predictor.generate_projections(sample_players)
        print(projections)

    except FileNotFoundError:
        print("No trained model found. Train a model first using ModelTrainer.")
