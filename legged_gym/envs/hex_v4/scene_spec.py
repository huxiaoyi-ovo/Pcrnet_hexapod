from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SceneSpec:
    """Classic tile scene spec (lightweight).

    This is a minimal scene spec for classic tiles, keeping only
    fields that HexGround uses for spawn/goal logic.
    """

    scene_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    static_obstacles: List[Any] = field(default_factory=list)
    layout_seed: Optional[int] = None

    def to_meta(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "params": dict(self.params),
            "layout_seed": self.layout_seed,
        }
