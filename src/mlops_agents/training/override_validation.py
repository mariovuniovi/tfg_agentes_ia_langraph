"""Validate search_space_override entries against the registry's approved space."""

from __future__ import annotations

from copy import deepcopy

from mlops_agents.contracts.training import SearchParamOverride
from mlops_agents.models.loader import SearchParamSpec, SearchSpaceSpec, get_model


def validate_override(model_key: str, overrides: dict[str, SearchParamOverride]) -> None:
    """Raise ValueError if any override is out of range, disjoint, or unknown."""
    spec = get_model(model_key).search_space
    for param_name, override in overrides.items():
        if param_name not in spec.params:
            raise ValueError(
                f"{model_key}: override references unknown parameter {param_name!r}. "
                f"Registry params: {sorted(spec.params)}"
            )
        registry_param = spec.params[param_name]
        if registry_param.type == "categorical":
            if override.choices is None:
                raise ValueError(
                    f"{model_key}.{param_name}: registry param is categorical; "
                    f"override must use {{choices}}, not {{low,high}}."
                )
            for c in override.choices:
                if c not in (registry_param.choices or []):
                    raise ValueError(
                        f"{model_key}.{param_name}: override choice {c!r} not in registry "
                        f"choices {registry_param.choices!r}"
                    )
        else:  # int / float
            lo, hi = registry_param.low, registry_param.high
            if override.choices is not None:
                for c in override.choices:
                    if not (lo <= c <= hi):
                        raise ValueError(
                            f"{model_key}.{param_name}: override choice {c} out of registry "
                            f"range [{lo}, {hi}]"
                        )
            else:
                if not (lo <= override.low <= override.high <= hi):
                    raise ValueError(
                        f"{model_key}.{param_name}: override range [{override.low}, {override.high}] "
                        f"is wider than or disjoint from registry range [{lo}, {hi}]"
                    )


def narrow_search_space(
    model_key: str,
    overrides: dict[str, SearchParamOverride],
) -> SearchSpaceSpec:
    """Return a copy of the registry's SearchSpaceSpec with override-narrowed params."""
    validate_override(model_key, overrides)
    base = deepcopy(get_model(model_key).search_space)
    for name, ovr in overrides.items():
        registry_param = base.params[name]
        if ovr.choices is not None:
            base.params[name] = SearchParamSpec(type="categorical", choices=list(ovr.choices))
        else:
            base.params[name] = SearchParamSpec(
                type=registry_param.type,
                low=ovr.low,
                high=ovr.high,
                step=registry_param.step,
                log=registry_param.log,
            )
    return base
