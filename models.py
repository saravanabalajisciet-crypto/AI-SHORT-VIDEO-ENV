from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Dict, Any, Optional
from enum import Enum


# ── Action ─────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    cut_scene           = "cut_scene"           # {"scene_id": str}
    add_subtitles       = "add_subtitles"        # {}
    reorder_scenes      = "reorder_scenes"       # {"order": [str, ...]}
    add_music           = "add_music"            # {"genre": str}  optional
    boost_hook          = "boost_hook"           # {}
    trim_duration       = "trim_duration"        # {"target_seconds": float}
    enhance_pacing      = "enhance_pacing"       # {}
    improve_transition  = "improve_transition"   # {}  — raise avg transition_quality
    smooth_cut          = "smooth_cut"           # {}  — raise avg cut_smoothness
    sync_audio          = "sync_audio"           # {}  — raise avg audio_sync_score


class Action(BaseModel):
    # Accepts: action_type, action, or type — all map to the same field
    action_type: Optional[ActionType] = Field(None, description="Action to perform")
    action: Optional[str] = Field(None, description="Alias for action_type")
    type: Optional[str] = Field(None, description="Alias for action_type")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"action_type": "boost_hook", "parameters": {}},
                {"action": "cut_scene", "parameters": {"scene_id": "scene_1"}},
                {"type": "add_music", "parameters": {"genre": "pop"}},
            ]
        }
    }

    @model_validator(mode="after")
    def resolve_action_type(self):
        if self.action_type is None:
            raw = self.action or self.type
            if raw is None:
                raise ValueError(
                    "action_type is required. Send one of: "
                    '{"action_type": "boost_hook"}, {"action": "boost_hook"}, or {"type": "boost_hook"}'
                )
            try:
                self.action_type = ActionType(raw)
            except ValueError:
                valid = [e.value for e in ActionType]
                raise ValueError(f"Invalid action '{raw}'. Valid actions: {valid}")
        return self


# ── Scene ──────────────────────────────────────────────────────────────────────

class Scene(BaseModel):
    id: str
    duration: float                     # seconds
    engagement_score: float             # 0.0 – 1.0
    scene_type: Literal["hook", "highlight", "content", "filler", "transition", "cta"]
    has_hook: bool = False
    hook_strength: float = 0.0          # 0.0 – 1.0
    transition_quality: float = 0.5     # 0.0 – 1.0  (hook→high, filler→low)
    cut_smoothness: float = 0.5         # 0.0 – 1.0
    audio_sync_score: float = 0.5       # 0.0 – 1.0


# ── Observation ────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    scenes: List[Scene]
    total_duration: float
    subtitles_present: bool
    music_added: bool
    platform: Literal["reels", "shorts", "tiktok"]
    current_engagement_score: float
    hook_first: bool
    platform_compliant: bool
    # ── advanced video intelligence ───────────────────────────────────────────
    retention_curve: List[float]
    avg_retention: float
    watch_time: float
    hook_strength: float
    pacing_score: float
    # ── production quality metrics ────────────────────────────────────────────
    avg_transition_quality: float       # mean of scene.transition_quality
    avg_cut_smoothness: float           # mean of scene.cut_smoothness
    avg_audio_sync_score: float         # mean of scene.audio_sync_score


# ── State ──────────────────────────────────────────────────────────────────────

class State(BaseModel):
    episode_id: str
    step_count: int
    max_steps: int
    done: bool
    observation: Observation
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── API response shapes ────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    state: State
    reward: float
    done: bool
    info: Dict[str, Any]


class GraderResponse(BaseModel):
    score: float
    breakdown: Dict[str, float]
    passed: bool


class AIFeedback(BaseModel):
    overall: str
    tips: List[str]
    score: float


class TaskDefinition(BaseModel):
    id: str
    level: Literal["easy", "medium", "hard"]
    description: str
    expected_behavior: str
    evaluation_criteria: str
    target_score: float


class BaselineResult(BaseModel):
    task_id: str
    score: float
    passed: bool
    steps: int
    final_engagement: float
    final_retention: float
    final_duration: float
    subtitles: bool
    hook_first: bool
    hook_strength: float
    pacing_score: float
    avg_transition_quality: float
    avg_cut_smoothness: float
    avg_audio_sync_score: float
    platform_compliant: bool
    feedback: AIFeedback
