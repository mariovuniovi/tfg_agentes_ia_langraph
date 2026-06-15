"""Tests for trial budget allocation."""
from mlops_agents.contracts.training import TrainingPlanCandidate, TrialBudget
from mlops_agents.training.trial_budget import allocate_trials


def _candidates(priorities, requested=None):
    return [
        TrainingPlanCandidate(priority=p, model_key=f"m{i}",
                              requested_trials=(requested[i] if requested else None))
        for i, p in enumerate(priorities)
    ]


def test_priority_weighted_3_candidates_60_trials():
    budget = TrialBudget(total_trials=60, allocation_strategy="priority_weighted",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2, 3]), budget)
    assert alloc["m0"] == 30
    assert alloc["m1"] == 20
    assert alloc["m2"] == 10


def test_equal_2_candidates_60_trials():
    budget = TrialBudget(total_trials=60, allocation_strategy="equal",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2]), budget)
    assert alloc["m0"] == 30
    assert alloc["m1"] == 30


def test_min_clamp_applied():
    budget = TrialBudget(total_trials=10, allocation_strategy="priority_weighted",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2, 3, 4]), budget)
    assert min(alloc.values()) >= 5


def test_max_clamp_applied():
    budget = TrialBudget(total_trials=200, allocation_strategy="equal",
                         min_trials_per_candidate=5, max_trials_per_candidate=20)
    alloc = allocate_trials(_candidates([1, 2]), budget)
    assert max(alloc.values()) <= 20


def test_requested_trials_advisory():
    budget = TrialBudget(total_trials=60, allocation_strategy="priority_weighted",
                         min_trials_per_candidate=5, max_trials_per_candidate=30)
    alloc = allocate_trials(_candidates([1, 2, 3], requested=[15, 15, 15]), budget)
    assert alloc["m0"] == 15
    assert alloc["m1"] == 15
    assert alloc["m2"] == 15
