"""MLRule loader and match_rules() for the static knowledge base."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field, model_validator
from mlops_agents.config.settings import settings


class MLRule(BaseModel):
    rule_id: str
    applies_when: dict[str, Any]
    prefer: list[str] = Field(default_factory=list)
    avoid_or_deprioritize: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    recommend: dict[str, Any] = Field(default_factory=dict)
    reason: str
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_applies_when_fields(self) -> "MLRule":
        from mlops_agents.contracts.profile import DatasetProfile
        valid = set(DatasetProfile.model_fields.keys())
        for field in self.applies_when:
            if field not in valid:
                raise ValueError(
                    f"Rule {self.rule_id}: applies_when references unknown profile field "
                    f"'{field}'. Valid fields: {sorted(valid)}"
                )
        return self

    @model_validator(mode="after")
    def validate_model_keys(self) -> "MLRule":
        from mlops_agents.models.loader import load_registry
        registry = load_registry()
        rule_pt = self.applies_when.get("problem_type")
        if isinstance(rule_pt, str):
            allowed = {rule_pt}
        elif isinstance(rule_pt, list):
            allowed = set(rule_pt)
        else:
            allowed = None
        for k in self.prefer + self.avoid_or_deprioritize:
            if k not in registry:
                raise ValueError(f"Rule {self.rule_id}: unknown model_key '{k}'")
            if allowed is not None and registry[k].problem_type not in allowed:
                raise ValueError(
                    f"Rule {self.rule_id}: model_key '{k}' "
                    f"(problem_type={registry[k].problem_type!r}) does not match "
                    f"applies_when.problem_type {sorted(allowed)}"
                )
        return self


@lru_cache(maxsize=1)
def load_rules(path: Path | None = None) -> list[MLRule]:
    """Load and validate ml_rules.yaml; cached after first call."""
    p = path or settings.ml_rules_path
    raw = yaml.safe_load(Path(p).read_text()) or []
    return [MLRule(**entry) for entry in raw]


def match_rules(profile: dict[str, Any]) -> list[MLRule]:
    """All rules whose applies_when conditions are fully satisfied by the profile."""
    matched = []
    for rule in load_rules():
        ok = True
        for field, expected in rule.applies_when.items():
            actual = profile.get(field)
            if actual is None:
                ok = False
                break
            if isinstance(expected, list):
                if actual not in expected:
                    ok = False
                    break
            else:
                if actual != expected:
                    ok = False
                    break
        if ok:
            matched.append(rule)
    return matched
