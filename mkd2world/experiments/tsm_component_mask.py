from __future__ import annotations

import re

LEVELS = ("type", "semantic", "coord", "block", "full")

_ATOM_TYPES = "text|link|button|input|icon|image|toggle|select"
_ATOM = re.compile(
    r"^(?P<indent>\s*)-\s+(?P<state>\[[ xXv>]\]\s+)?"
    r"\[(?P<content>.*)\]\((?P<type>" + _ATOM_TYPES + r")\)"
    r"(?P<box><\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*>)?\s*$"
)
_ENV = "[Environment: Web]"


def mask_tsm(markdown: str, level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown TSM mask level {level!r}; expected one of {LEVELS}")
    if level == "full" or not markdown:
        return markdown
    lines = markdown.splitlines()
    if level == "block":
        out = []
        for line in lines:
            m = _ATOM.match(line)
            if m and m.group("state"):
                line = line[: m.start("state")] + line[m.end("state") :]
            out.append(line)
        return "\n".join(out)

    out = [_ENV, ""] if any(line.strip() == _ENV for line in lines) else []
    for line in lines:
        m = _ATOM.match(line)
        if not m:
            continue
        kind = m.group("type")
        if level == "type":
            out.append(f"- ({kind})")
        elif level == "semantic":
            out.append(f"- [{m.group('content')}]({kind})")
        else:
            out.append(f"- [{m.group('content')}]({kind}){m.group('box') or ''}")
    return "\n".join(out)
