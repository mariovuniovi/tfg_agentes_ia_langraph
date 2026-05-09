"""Distribute total_trials across candidates per the TrialBudget policy."""
from __future__ import annotations
from mlops_agents.contracts.training import TrainingPlanCandidate, TrialBudget


def allocate_trials(candidates: list[TrainingPlanCandidate], budget: TrialBudget) -> dict[str, int]:
    n = len(candidates)
    if n == 0:
        return {}

    if budget.allocation_strategy == "priority_weighted":
        weights = [n + 1 - c.priority for c in candidates]
        total_weight = sum(weights)
        base = [int(round(budget.total_trials * w / total_weight)) for w in weights]
    else:
        base = [budget.total_trials // n] * n

    final = []
    for cand, b in zip(candidates, base):
        final.append(cand.requested_trials if cand.requested_trials is not None else b)

    final = [max(budget.min_trials_per_candidate, min(budget.max_trials_per_candidate, x)) for x in final]

    if sum(final) > budget.total_trials:
        slack = sum(final) - budget.total_trials
        reducible = [(i, x - budget.min_trials_per_candidate) for i, x in enumerate(final)
                     if x > budget.min_trials_per_candidate]
        if reducible:
            total_reducible = sum(r for _, r in reducible)
            for i, r in reducible:
                cut = int(round(slack * r / total_reducible))
                final[i] = max(budget.min_trials_per_candidate, final[i] - cut)

    return {c.model_key: t for c, t in zip(candidates, final)}
