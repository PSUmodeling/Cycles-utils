from __future__ import annotations
import types
from dataclasses import fields
from pathlib import Path
from typing import Union

def _format_block(label: str, block) -> str:
    lines = [f'## {label.replace("_", " ").upper()} ##']
    for f in fields(block):
        lines.append('%-27s\t%s' % (f.name.upper(), getattr(block, f.name)))
    lines.append('')

    return '\n'.join(lines)


def write_file(fn: Path, config) -> None:
    """Write any config dataclass to a fixed-width file."""
    content = '\n'.join(
        _format_block(f.name, getattr(config, f.name))
        for f in fields(config)
        if getattr(config, f.name) is not None
    )
    fn.write_text(content)


def resolve_dict_values(user_dict: dict, simulation: dict[str, Any] | None) -> dict:
    return {key: func(simulation) if callable(func) else func for key, func in user_dict.items()}


def extract(dc_class, resolved: dict) -> dict:
    return {f.name: resolved[f.name] for f in fields(dc_class) if f.name in resolved}


def parse_value(raw: str, name: str, hint: type) -> int | float | str:
    """Cast a raw string token to the field's annotated type."""
    if raw.split()[0].lower() == name.lower():
        if hint is int:   return int(raw.split()[1])
        if hint is float: return float(raw.split()[1])
        return raw.split()[1]
    else:
        raise ValueError(f"Expected field name '{name}' not found in line: {raw}")


def unwrap_optional(t) -> type:
    """Strip | None from a type hint, returning the underlying type."""
    origin = getattr(t, '__origin__', None)
    if origin is Union or origin is types.UnionType:
        return next(arg for arg in t.__args__ if arg is not type(None))
    return t
