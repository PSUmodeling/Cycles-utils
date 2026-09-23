from __future__ import annotations
import types
from dataclasses import fields
from pathlib import Path
from typing import Union, Any

def _fmt_float(value: float, digits: int=2) -> str:
    if value == -999.0:
        return '-999'
    for d in range(1, digits):
        if value == round(value, d):
            return f'{value:.{d}f}'
    else:
        return f'{value:.{digits}f}'

# Convenience lambdas for metadata
FMT_1F = lambda v: _fmt_float(v, 1)
FMT_2F = lambda v: _fmt_float(v, 2)
FMT_3F = lambda v: _fmt_float(v, 3)
FMT_4F = lambda v: _fmt_float(v, 4)

def _format_field(object, attribute, widths: tuple[int, int], value=None) -> str:
    fmt = attribute.metadata.get('fmt')
    description = attribute.metadata.get('description', '')

    if value is None:
        value = getattr(object, attribute.name)

    formatted = fmt(value) if fmt is not None else str(value)

    return f'{attribute.name.upper():<{widths[0]}}{formatted:<{widths[1]}}# {description}' if description else f'{attribute.name.upper():<{widths[0]}}{formatted}'


def _format_block(label: str, object, *, widths: tuple[int, int]=(28, 8), center: bool=False, all_caps: bool=True) -> str:
    label = label.replace("_", " ").upper() if all_caps else label.replace("_", " ").capitalize()

    lines = [f' ' * (28 if center else 0) + f'## {label} ##']
    for f in fields(object):
        lines.append(_format_field(object, f, widths))
    lines.append('')

    return '\n'.join(lines)


def _write_file(fn: Path, config) -> None:
    content = '\n'.join(
        _format_block(f.name, getattr(config, f.name))
        for f in fields(config)
        if getattr(config, f.name) is not None
    )
    fn.write_text(content)


def _resolve_dict_values(user_dict: dict, simulation: dict[str, Any] | None) -> dict:
    return {key: func(simulation) if callable(func) else func for key, func in user_dict.items()}


def _extract(dc_class, resolved: dict) -> dict:
    return {f.name: resolved[f.name] for f in fields(dc_class) if f.name in resolved}


def _parse_value(raw: str, name: str, hint: type) -> int | float | str:
    hint = _unwrap_optional(hint)
    if raw.split()[0].lower() == name.lower():
        if hint is int: return int(raw.split()[1])
        if hint is float: return float(raw.split()[1])
        return raw.split()[1]
    else:
        raise ValueError(f"Expected field name '{name}' not found in line: {raw}")


def _unwrap_optional(t) -> type:
    origin = getattr(t, '__origin__', None)
    if origin is Union or origin is types.UnionType or isinstance(t, types.UnionType):
        return next(arg for arg in t.__args__ if arg is not type(None))
    return t
