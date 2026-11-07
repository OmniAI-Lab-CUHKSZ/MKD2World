
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class GUINode:
    node_id: str
    tag: str
    text: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    attributes: Dict = field(default_factory=dict)
    children: List['GUINode'] = field(default_factory=list)
    is_structural_block: bool = False
    checked: bool = False
    expanded: bool = False


def normalize_bbox(bbox: Tuple[float, float, float, float], w: int, h: int) -> str:
    """Render an xyxy bbox as a 0..999 relative-grid center box."""
    if not bbox:
        return "<0,0,0,0>"
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    gx = round(cx / w * 999)
    gy = round(cy / h * 999)
    gw = round(bw / w * 999)
    gh = round(bh / h * 999)
    return f"<{gx},{gy},{gw},{gh}>"
