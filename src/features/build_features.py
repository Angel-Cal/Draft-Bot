"""Feature engineering for fantasy football player projections."""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build features for machine learning models."""

    def __init__(self):
        """Initialize the feature builder."""
        pass

    def add_rolling_averages(
        self,
        df: pd.DataFrame,
        columns: list[str],
        windows: list[int] = [3, 5],
        group_by: str = "player"
    ) -> pd.DataFrame:
        """
        Add rolling average features for specified columns.

        Args:
            df: DataFrame with player data (must be sorted by date/week)
            columns: Columns to calculate rolling averages for
            windows: Window sizes for rolling averages
            group_by: Column to group by (usually player)

        Returns:
            DataFrame with rolling average features added
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            for window in windows:
                feature_name = f"{col}_rolling_{window}"
                df[feature_name] = (
                    df.groupby(group_by)[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

        return df

    def add_year_over_year_change(
        self,
        df: pd.DataFrame,
        columns: list[str],
        player_col: str = "player",
        season_col: str = "season"
    ) -> pd.DataFrame:
        """
        Add year-over-year change features.

        Args:
            df: DataFrame with multi-season data
            columns: Columns to calculate YoY change for
            player_col: Player identifier column
            season_col: Season column

        Returns:
            DataFrame with YoY features added
        """
        df = df.copy().sort_values([player_col, season_col])

        for col in columns:
            if col not in df.columns:
                continue

            feature_name = f"{col}_yoy_change"
            df[feature_name] = df.groupby(player_col)[col].diff()

            # Also add percentage change
            pct_feature = f"{col}_yoy_pct_change"
            df[pct_feature] = df.groupby(player_col)[col].pct_change()

        return df

    def add_age_features(
        self,
        df: pd.DataFrame,
        birth_year_col: Optional[str] = None,
        age_col: Optional[str] = None,
        season_col: str = "season"
    ) -> pd.DataFrame:
        """
        Add age-related features including age curves.

        Args:
            df: DataFrame with player data
            birth_year_col: Column with birth year
            age_col: Column with age (if already present)
            season_col: Season column

        Returns:
            DataFrame with age features
        """
        df = df.copy()

        # Calculate age if birth year is provided
        if birth_year_col and birth_year_col in df.columns:
            df["age"] = df[season_col] - df[birth_year_col]
        elif age_col and age_col in df.columns:
            df["age"] = df[age_col]

        if "age" in df.columns:
            # Age squared for non-linear aging effects
            df["age_squared"] = df["age"] ** 2

            # Age buckets
            df["age_bucket"] = pd.cut(
                df["age"],
                bins=[0, 24, 27, 30, 33, 100],
                labels=["young", "prime_early", "prime_late", "declining", "old"]
            )

            # Peak age indicators by position (approximate)
            peak_ages = {"QB": 29, "RB": 25, "WR": 27, "TE": 28}
            if "position" in df.columns:
                df["years_from_peak"] = df.apply(
                    lambda x: x["age"] - peak_ages.get(x["position"], 27)
                    if pd.notna(x.get("position")) else 0,
                    axis=1
                )

        return df

    def add_position_encoding(
        self,
        df: pd.DataFrame,
        position_col: str = "position"
    ) -> pd.DataFrame:
        """
        One-hot encode position column.

        Args:
            df: DataFrame with position column
            position_col: Name of position column

        Returns:
            DataFrame with position one-hot encoded
        """
        df = df.copy()

        if position_col in df.columns:
            position_dummies = pd.get_dummies(df[position_col], prefix="pos")
            df = pd.concat([df, position_dummies], axis=1)

        return df

    def add_target_share_features(
        self,
        df: pd.DataFrame,
        targets_col: str = "targets",
        team_col: str = "team"
    ) -> pd.DataFrame:
        """
        Calculate target share and related receiving features.

        Args:
            df: DataFrame with receiving data
            targets_col: Column with target counts
            team_col: Team column for grouping

        Returns:
            DataFrame with target share features
        """
        df = df.copy()

        if targets_col in df.columns and team_col in df.columns:
            # Calculate team total targets
            team_targets = df.groupby([team_col, "season"])[targets_col].transform("sum")
            df["target_share"] = df[targets_col] / team_targets

            # Air yards share if available
            if "air_yards" in df.columns:
                team_air_yards = df.groupby([team_col, "season"])["air_yards"].transform("sum")
                df["air_yards_share"] = df["air_yards"] / team_air_yards

        return df

    def add_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add efficiency-based features.

        Args:
            df: DataFrame with player statistics

        Returns:
            DataFrame with efficiency features
        """
        df = df.copy()

        # Rushing efficiency
        if "rushing_yds" in df.columns and "rushing_att" in df.columns:
            df["yards_per_carry"] = df["rushing_yds"] / df["rushing_att"].replace(0, np.nan)

        # Receiving efficiency
        if "receiving_yds" in df.columns and "receptions" in df.columns:
            df["yards_per_reception"] = df["receiving_yds"] / df["receptions"].replace(0, np.nan)

        if "receptions" in df.columns and "targets" in df.columns:
            df["catch_rate"] = df["receptions"] / df["targets"].replace(0, np.nan)

        # Passing efficiency
        if "passing_yds" in df.columns and "passing_att" in df.columns:
            df["yards_per_attempt"] = df["passing_yds"] / df["passing_att"].replace(0, np.nan)

        if "passing_td" in df.columns and "passing_att" in df.columns:
            df["td_rate"] = df["passing_td"] / df["passing_att"].replace(0, np.nan)

        # Fill NaN efficiency metrics with 0
        efficiency_cols = [
            "yards_per_carry", "yards_per_reception", "catch_rate",
            "yards_per_attempt", "td_rate"
        ]
        for col in efficiency_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features for the dataset.

        Args:
            df: Raw/processed DataFrame

        Returns:
            DataFrame with all features built
        """
        logger.info("Building features...")

        # Add efficiency metrics
        df = self.add_efficiency_metrics(df)

        # Add position encoding
        df = self.add_position_encoding(df)

        # Add age features if age data available
        df = self.add_age_features(df)

        # Add target share features
        df = self.add_target_share_features(df)

        # Add year-over-year changes for key stats
        key_stats = ["fantasy_points", "rushing_yds", "receiving_yds", "passing_yds"]
        existing_stats = [s for s in key_stats if s in df.columns]
        if existing_stats:
            df = self.add_year_over_year_change(df, existing_stats)

        logger.info(f"Built features. Final shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Example usage with sample data
    sample_data = pd.DataFrame({
        "player": ["Player A", "Player A", "Player B", "Player B"],
        "season": [2022, 2023, 2022, 2023],
        "position": ["RB", "RB", "WR", "WR"],
        "rushing_yds": [1000, 1200, 100, 50],
        "rushing_att": [200, 240, 10, 5],
        "receiving_yds": [300, 400, 1100, 1300],
        "receptions": [40, 50, 80, 95],
        "targets": [50, 65, 120, 130],
        "team": ["KC", "KC", "BUF", "BUF"],
    })

    builder = FeatureBuilder()
    featured_data = builder.build_all_features(sample_data)
    print(featured_data.columns.tolist())
