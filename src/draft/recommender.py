"""Draft recommendation engine."""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

from .vbd import ValueBasedDrafter
from .simulator import DraftState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A draft pick recommendation."""
    player: str
    position: str
    projected_points: float
    vor: float
    rank: int
    confidence: float
    reasoning: str


class DraftRecommender:
    """
    Recommend optimal draft picks based on multiple factors.

    Combines VBD, roster needs, and positional scarcity.
    """

    # Default roster requirements
    DEFAULT_ROSTER = {
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,  # RB/WR/TE
        "K": 1,
        "DST": 1,
        "BENCH": 6
    }

    # Position eligibility for FLEX
    FLEX_ELIGIBLE = ["RB", "WR", "TE"]

    def __init__(
        self,
        roster_requirements: Optional[dict[str, int]] = None,
        vbd: Optional[ValueBasedDrafter] = None
    ):
        """
        Initialize recommender.

        Args:
            roster_requirements: Required roster slots by position
            vbd: ValueBasedDrafter instance
        """
        self.roster_requirements = roster_requirements or self.DEFAULT_ROSTER.copy()
        self.vbd = vbd or ValueBasedDrafter()

    def calculate_roster_needs(
        self,
        roster: dict[str, list[str]]
    ) -> dict[str, int]:
        """
        Calculate remaining roster needs.

        Args:
            roster: Current roster {position: [players]}

        Returns:
            Needs by position
        """
        needs = {}

        for pos, required in self.roster_requirements.items():
            if pos in ["FLEX", "BENCH"]:
                continue

            current = len(roster.get(pos, []))
            needs[pos] = max(0, required - current)

        # Calculate FLEX needs
        flex_players = sum(
            len(roster.get(pos, [])) - self.roster_requirements.get(pos, 0)
            for pos in self.FLEX_ELIGIBLE
            if len(roster.get(pos, [])) > self.roster_requirements.get(pos, 0)
        )
        needs["FLEX"] = max(0, self.roster_requirements.get("FLEX", 0) - max(0, flex_players))

        return needs

    def get_recommendations(
        self,
        available_players: pd.DataFrame,
        roster: dict[str, list[str]],
        current_pick: int,
        total_picks: int,
        n_recommendations: int = 5
    ) -> list[Recommendation]:
        """
        Get top pick recommendations.

        Args:
            available_players: DataFrame of available players
            roster: Current roster
            current_pick: Current overall pick number
            total_picks: Total picks in draft
            n_recommendations: Number of recommendations to return

        Returns:
            List of Recommendation objects
        """
        # Ensure VBD is calculated
        if "vor" not in available_players.columns:
            available_players = self.vbd.calculate_vbd(available_players)

        needs = self.calculate_roster_needs(roster)
        picks_remaining = total_picks - current_pick

        recommendations = []

        for _, player in available_players.nlargest(20, "vor").iterrows():
            pos = player["position"]

            # Calculate base score from VOR
            base_score = player["vor"]

            # Adjust for roster needs
            need_bonus = 0
            if needs.get(pos, 0) > 0:
                need_bonus = 10  # Bonus for filling a need

            # Adjust for positional scarcity (late picks should prioritize scarce positions)
            scarcity_bonus = 0
            if picks_remaining < 50:  # Later in draft
                if pos in ["TE", "QB"] and needs.get(pos, 0) > 0:
                    scarcity_bonus = 5

            # Calculate final score
            total_score = base_score + need_bonus + scarcity_bonus

            # Generate reasoning
            reasons = []
            if player["vbd_rank"] <= 5:
                reasons.append("Elite VBD value")
            elif player["vbd_rank"] <= 15:
                reasons.append("Strong VBD value")
            if needs.get(pos, 0) > 0:
                reasons.append(f"Fills {pos} need")
            if pos in ["RB", "WR"] and player["vbd_rank"] <= 20:
                reasons.append("FLEX eligible")

            reasoning = "; ".join(reasons) if reasons else "Best available"

            # Calculate confidence (normalize score to 0-1)
            max_vor = available_players["vor"].max()
            confidence = min(1.0, max(0.3, total_score / max_vor)) if max_vor > 0 else 0.5

            recommendations.append(Recommendation(
                player=player["player"],
                position=pos,
                projected_points=player["projected_points"],
                vor=player["vor"],
                rank=int(player["vbd_rank"]),
                confidence=confidence,
                reasoning=reasoning
            ))

        # Sort by adjusted score and return top N
        recommendations.sort(key=lambda x: x.vor, reverse=True)
        return recommendations[:n_recommendations]

    def format_recommendations(
        self,
        recommendations: list[Recommendation]
    ) -> str:
        """
        Format recommendations for display.

        Args:
            recommendations: List of recommendations

        Returns:
            Formatted string
        """
        lines = ["=" * 50, "DRAFT RECOMMENDATIONS", "=" * 50]

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"\n{i}. {rec.player} ({rec.position})")
            lines.append(f"   Projected: {rec.projected_points:.1f} pts | VOR: {rec.vor:.1f}")
            lines.append(f"   VBD Rank: #{rec.rank} | Confidence: {rec.confidence:.0%}")
            lines.append(f"   Reason: {rec.reasoning}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def auto_pick(
        self,
        state: DraftState,
        team_id: int,
        current_pick: int,
        total_picks: int
    ) -> str:
        """
        Automatically select best pick for a team.

        Args:
            state: Current draft state
            team_id: Team making pick
            current_pick: Current pick number
            total_picks: Total picks in draft

        Returns:
            Name of recommended player
        """
        roster = state.get_roster(team_id)
        available = state.available_players

        # Ensure VBD is calculated
        if "vor" not in available.columns:
            available = self.vbd.calculate_vbd(available)
            state.available_players = available

        recommendations = self.get_recommendations(
            available,
            roster,
            current_pick,
            total_picks,
            n_recommendations=1
        )

        if recommendations:
            return recommendations[0].player
        else:
            # Fallback to best available
            return available.nlargest(1, "projected_points")["player"].iloc[0]


if __name__ == "__main__":
    # Example usage
    sample_players = pd.DataFrame({
        "player": ["CMC", "Jefferson", "Kelce", "Mahomes", "Henry", "Hill", "Andrews"],
        "position": ["RB", "WR", "TE", "QB", "RB", "WR", "TE"],
        "projected_points": [350, 290, 240, 380, 300, 280, 200]
    })

    recommender = DraftRecommender()

    # Calculate VBD first
    sample_players = recommender.vbd.calculate_vbd(sample_players)

    # Get recommendations
    roster = {"QB": [], "RB": [], "WR": [], "TE": [], "K": [], "DST": []}
    recommendations = recommender.get_recommendations(
        sample_players,
        roster,
        current_pick=1,
        total_picks=180,
        n_recommendations=5
    )

    print(recommender.format_recommendations(recommendations))
