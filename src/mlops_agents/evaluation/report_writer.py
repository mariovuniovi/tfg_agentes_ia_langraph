"""Evaluation Report LLM node — evidence-aligned audit, not a decision maker.

The promotion decision is already made by evaluate_promotion() before this
node runs. This module produces a structured narrative that triangulates
planner reasoning, training results, and empirical metrics.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    summary: str
    champion_model: str
    why_champion_won: str
    planner_alignment: str
    deviations_from_planner_expectations: list[str] = Field(default_factory=list)
    evidence_consistency_warnings: list[str] = Field(default_factory=list)
    risks_and_warnings: list[str] = Field(default_factory=list)
    promotion_decision_explanation: str
    human_review_notes: list[str] = Field(default_factory=list)
