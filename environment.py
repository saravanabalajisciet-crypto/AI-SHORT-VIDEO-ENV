import uuid
import random
import json
import os
from copy import deepcopy
from typing import Tuple, Dict, Any, List, Optional

from models import Action, ActionType, Observation, Scene, State, AIFeedback

ENGAGEMENT_THRESHOLD = 0.92
MAX_STEPS = 15

PLATFORM_LIMITS = {
    "reels":  30.0,
    "shorts": 60.0,
    "tiktok": 60.0,
}

# ── Load real video dataset ────────────────────────────────────────────────────
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "video_dataset.json")
try:
    with open(_DATASET_PATH) as f:
        _DATASET = json.load(f)
    _REAL_VIDEOS = _DATASET["videos"]
    _AUDIENCE_PERSONAS = _DATASET["audience_personas"]
    _ENGAGEMENT_PATTERNS = _DATASET["engagement_patterns"]
except Exception:
    _REAL_VIDEOS = []
    _AUDIENCE_PERSONAS = {}
    _ENGAGEMENT_PATTERNS = {}

# ── Synthetic fallback templates ───────────────────────────────────────────────
_TEMPLATES = [
    ("hook",       True,  (0.55, 0.85), (3.0,  6.0),  (0.60, 0.88), (0.65, 0.90), (0.60, 0.85), (0.60, 0.85)),
    ("highlight",  True,  (0.40, 0.70), (4.0,  7.0),  (0.50, 0.78), (0.55, 0.80), (0.55, 0.80), (0.55, 0.80)),
    ("content",    False, (0.0,  0.15), (6.0, 10.0),  (0.30, 0.62), (0.40, 0.65), (0.40, 0.65), (0.40, 0.65)),
    ("transition", False, (0.0,  0.10), (1.0,  2.5),  (0.08, 0.25), (0.20, 0.45), (0.20, 0.45), (0.20, 0.45)),
    ("content",    False, (0.0,  0.15), (5.0,  9.0),  (0.28, 0.58), (0.35, 0.60), (0.35, 0.60), (0.35, 0.60)),
    ("filler",     False, (0.0,  0.08), (3.0,  6.0),  (0.05, 0.20), (0.10, 0.35), (0.10, 0.35), (0.10, 0.35)),
    ("cta",        False, (0.0,  0.20), (2.0,  4.0),  (0.40, 0.68), (0.45, 0.70), (0.45, 0.70), (0.45, 0.70)),
]


def _make_scene(idx, stype, has_hook, hook_str, dur, eng, tq, cs, as_):
    return Scene(
        id=f"scene_{idx}",
        duration=round(dur, 2),
        engagement_score=round(eng, 3),
        scene_type=stype,
        has_hook=has_hook,
        hook_strength=round(hook_str, 3),
        transition_quality=round(tq, 3),
        cut_smoothness=round(cs, 3),
        audio_sync_score=round(as_, 3),
    )


def _initial_scenes(seed):
    """
    Generate initial scenes. Uses real video dataset when available,
    falls back to synthetic templates. Real data grounded in actual
    creator engagement patterns from public analytics research.
    """
    rng = random.Random(seed)

    # Use real dataset if available (primary path)
    if _REAL_VIDEOS:
        video = _REAL_VIDEOS[seed % len(_REAL_VIDEOS)]
        scenes = []
        for i, s in enumerate(video["scenes"]):
            # Add small seed-controlled noise to prevent identical episodes
            noise = rng.uniform(-0.03, 0.03)
            scenes.append(Scene(
                id=f"scene_{i}",
                duration=round(s["duration"] + rng.uniform(-0.3, 0.3), 2),
                engagement_score=round(min(1.0, max(0.0, s["engagement"] + noise)), 3),
                scene_type=s["type"],
                has_hook=s["has_hook"],
                hook_strength=round(min(1.0, max(0.0, s["hook_strength"] + noise)), 3),
                transition_quality=round(min(1.0, max(0.0, s["transition_quality"] + noise)), 3),
                cut_smoothness=round(min(1.0, max(0.0, s["cut_smoothness"] + noise)), 3),
                audio_sync_score=round(min(1.0, max(0.0, s["audio_sync"] + noise)), 3),
            ))
        return scenes

    # Synthetic fallback
    templates = list(_TEMPLATES)
    drops = rng.randint(0, 2)
    for _ in range(drops):
        idx = rng.randint(1, len(templates) - 2)
        templates.pop(idx)
    scenes = []
    for i, (stype, has_hook, hs_r, dur_r, eng_r, tq_r, cs_r, as_r) in enumerate(templates):
        scenes.append(_make_scene(
            i, stype, has_hook,
            rng.uniform(*hs_r), rng.uniform(*dur_r), rng.uniform(*eng_r),
            rng.uniform(*tq_r), rng.uniform(*cs_r), rng.uniform(*as_r),
        ))
    return scenes


def _get_persona(seed: int) -> str:
    """Assign audience persona based on seed."""
    personas = list(_AUDIENCE_PERSONAS.keys()) if _AUDIENCE_PERSONAS else ["gen_z", "millennial", "brand"]
    return personas[seed % len(personas)]


def _avg(scenes, attr):
    if not scenes:
        return 0.0
    return round(sum(getattr(s, attr) for s in scenes) / len(scenes), 4)


def _avg_engagement(scenes):
    return _avg(scenes, "engagement_score")


def _total_duration(scenes):
    return round(sum(s.duration for s in scenes), 2)


def _hook_first(scenes):
    return bool(scenes) and scenes[0].scene_type in ("hook", "highlight") and scenes[0].has_hook


def _platform_compliant(total_dur, platform):
    return total_dur <= PLATFORM_LIMITS.get(platform, 30.0)


def _retention_curve(scenes):
    if not scenes:
        return []
    curve = []
    retention = 1.0
    for s in scenes:
        drop = (1.0 - s.engagement_score) * 0.18
        if s.scene_type in ("filler", "transition"):
            drop += 0.08
        elif s.scene_type in ("hook", "highlight") and s.has_hook:
            drop -= 0.05
        drop -= s.transition_quality * 0.02
        drop = max(0.0, drop)
        retention = max(0.0, retention - drop)
        curve.append(round(retention, 4))
    return curve


def _watch_time(scenes, rc):
    return round(sum(s.duration * r for s, r in zip(scenes, rc)), 2)


def _hook_strength(scenes):
    if not scenes:
        return 0.0
    first = scenes[0]
    if first.scene_type in ("hook", "highlight") and first.has_hook:
        return round(first.hook_strength, 3)
    return 0.0


def _pacing_score(scenes):
    if len(scenes) < 2:
        return 1.0
    deltas = [abs(scenes[i].engagement_score - scenes[i-1].engagement_score)
              for i in range(1, len(scenes))]
    mean_delta = sum(deltas) / len(deltas)
    base = max(0.0, 1.0 - mean_delta * 2.5)
    cs_bonus = _avg(scenes, "cut_smoothness") * 0.05
    return round(min(1.0, base + cs_bonus), 4)


# ── NEW ADDITION: risk score ───────────────────────────────────────────────────
def _compute_risk_score(hook_strength: float, avg_retention: float, pacing_score: float) -> float:
    """
    Compute a risk score (0.0–1.0) representing proximity to irreversible failure.
    Higher = more dangerous. Based on three weak-signal indicators:
      - low hook_strength  (weight 0.45) — first-impression failure is unrecoverable
      - low avg_retention  (weight 0.35) — audience already leaving
      - poor pacing_score  (weight 0.20) — structural incoherence

    NOTE: risk_score does NOT affect reward directly. It is an advisory signal
    for decision-making and evaluation only.

    Risk regions:
      0.0 – 0.4  Low    → Safe exploration
      0.4 – 0.7  Medium → Careful optimization
      > 0.7      High   → Limited recovery region
    """
    hook_risk    = max(0.0, 1.0 - hook_strength / 0.6)   # danger zone below 0.6
    ret_risk     = max(0.0, 1.0 - avg_retention / 0.5)   # danger zone below 0.5
    pacing_risk  = max(0.0, 1.0 - pacing_score / 0.4)    # danger zone below 0.4
    risk = hook_risk * 0.45 + ret_risk * 0.35 + pacing_risk * 0.20
    return round(min(1.0, risk), 4)


def _build_observation(scenes, platform, subtitles, music, steps_remaining: int = 15):
    total_dur = _total_duration(scenes)
    rc = _retention_curve(scenes)
    avg_ret = round(sum(rc) / len(rc), 4) if rc else 0.0
    hook_str = _hook_strength(scenes)
    pac = _pacing_score(scenes)
    return Observation(
        scenes=scenes,
        total_duration=total_dur,
        subtitles_present=subtitles,
        music_added=music,
        platform=platform,
        current_engagement_score=_avg_engagement(scenes),
        hook_first=_hook_first(scenes),
        platform_compliant=_platform_compliant(total_dur, platform),
        retention_curve=rc,
        avg_retention=avg_ret,
        watch_time=_watch_time(scenes, rc),
        hook_strength=hook_str,
        pacing_score=pac,
        avg_transition_quality=_avg(scenes, "transition_quality"),
        avg_cut_smoothness=_avg(scenes, "cut_smoothness"),
        avg_audio_sync_score=_avg(scenes, "audio_sync_score"),
        steps_remaining=steps_remaining,
        # SAFE EXTENSION: risk_score added, defaults to None for old clients
        risk_score=_compute_risk_score(hook_str, avg_ret, pac),
    )


def generate_feedback(obs, score):
    tips = []
    if not obs.hook_first or obs.hook_strength < 0.5:
        tips.append("Move your highest hook_strength scene first, then call boost_hook.")
    if obs.avg_retention < 0.6:
        tips.append("Retention drops too fast. Cut filler scenes and call enhance_pacing.")
    if not obs.platform_compliant:
        tips.append(f"Duration exceeds {obs.platform} limit ({PLATFORM_LIMITS[obs.platform]}s). Use trim_duration.")
    if not obs.subtitles_present:
        tips.append("Add subtitles — they increase watch time by up to 40% on mobile.")
    if not obs.music_added:
        tips.append("Background music boosts engagement on hook and content scenes.")
    if obs.pacing_score < 0.5:
        tips.append("Pacing is uneven. Call enhance_pacing to smooth scene transitions.")
    if obs.avg_transition_quality < 0.7:
        tips.append(f"Transition quality is low ({obs.avg_transition_quality:.2f}). Call improve_transition.")
    if obs.avg_cut_smoothness < 0.7:
        tips.append(f"Cut smoothness is low ({obs.avg_cut_smoothness:.2f}). Call smooth_cut.")
    if obs.avg_audio_sync_score < 0.7:
        tips.append(f"Audio sync is off ({obs.avg_audio_sync_score:.2f}). Call sync_audio.")
    if obs.current_engagement_score < 0.75:
        tips.append("Engagement below 0.75. Remove low-scoring scenes and call add_music.")
    if score >= 0.90:
        overall = "Excellent — your video is optimized for viral performance."
    elif score >= 0.78:
        overall = "Good edit. A few more tweaks will push this to viral territory."
    elif score >= 0.65:
        overall = "Decent start. Focus on hook strength and retention to level up."
    else:
        overall = "Needs significant work. Prioritize hook placement and duration compliance."
    return AIFeedback(overall=overall, tips=tips, score=round(score, 4))


class VideoOptimizationEnv:
    def __init__(self, platform="reels", seed=42):
        self.platform = platform
        self.seed = seed
        self._state = None
        self._music_applied = False
        self._hook_boosted = False
        self._pacing_enhanced = False
        self._transition_improved = False
        self._cut_smoothed = False
        self._audio_synced = False
        self._finalized = False
        self._persona = "gen_z"  # audience persona

    def reset(self):
        scenes = _initial_scenes(self.seed)
        self._music_applied = False
        self._hook_boosted = False
        self._pacing_enhanced = False
        self._transition_improved = False
        self._cut_smoothed = False
        self._audio_synced = False
        self._finalized = False
        self._persona = _get_persona(self.seed)
        self._action_history: List[str] = []  # FIX A: track action order

        real_video_id = None
        real_niche = None
        if _REAL_VIDEOS:
            v = _REAL_VIDEOS[self.seed % len(_REAL_VIDEOS)]
            real_video_id = v.get("id")
            real_niche = v.get("niche")

        obs = _build_observation(scenes, self.platform, subtitles=False, music=False,
                                  steps_remaining=MAX_STEPS)
        self._state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            max_steps=MAX_STEPS,
            done=False,
            observation=obs,
            metadata={
                "initial_engagement": obs.current_engagement_score,
                "initial_duration": obs.total_duration,
                "initial_retention": obs.avg_retention,
                "initial_transition_quality": obs.avg_transition_quality,
                "platform": self.platform,
                "seed": self.seed,
                "audience_persona": self._persona,
                "real_video_id": real_video_id,
                "niche": real_niche,
                "data_source": "real" if _REAL_VIDEOS else "synthetic",
            },
        )
        return deepcopy(self._state)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset().")

        obs = deepcopy(self._state.observation)
        reward = 0.0
        info = {"action": action.action_type, "valid": True}

        # FIX A: track action order for order-sensitive grader
        self._action_history.append(action.action_type.value)

        # FIX B: small stochastic noise per step (seed-controlled for reproducibility)
        _noise_rng = random.Random(self.seed + self._state.step_count * 31)
        _noise = lambda: _noise_rng.uniform(-0.015, 0.015)

        prev_eng = obs.current_engagement_score
        prev_ret = obs.avg_retention
        prev_hook = obs.hook_strength
        prev_compliant = obs.platform_compliant
        prev_hook_first = obs.hook_first
        prev_pacing = obs.pacing_score
        prev_tq = obs.avg_transition_quality
        prev_cs = obs.avg_cut_smoothness
        prev_as = obs.avg_audio_sync_score

        if action.action_type == ActionType.cut_scene:
            scene_id = action.parameters.get("scene_id")
            remaining = [s for s in obs.scenes if s.id != scene_id]
            if len(remaining) == len(obs.scenes):
                reward -= 0.3
                info.update(valid=False, reason=f"scene_id '{scene_id}' not found")
            elif len(remaining) == 0:
                reward -= 0.3
                info.update(valid=False, reason="cannot remove all scenes")
            else:
                removed = next(s for s in obs.scenes if s.id == scene_id)
                if removed.engagement_score >= 0.65:
                    reward -= 0.2
                    info["note"] = "penalised: removed a high-engagement scene"
                obs.scenes = remaining

        elif action.action_type == ActionType.reorder_scenes:
            new_order = action.parameters.get("order", [])
            scene_map = {s.id: s for s in obs.scenes}
            if set(new_order) != set(scene_map.keys()):
                reward -= 0.3
                info.update(valid=False, reason="order must contain exactly all current scene ids")
            else:
                obs.scenes = [scene_map[sid] for sid in new_order]
                if obs.scenes and obs.scenes[0].scene_type in ("hook", "highlight") and obs.scenes[0].has_hook:
                    obs.scenes = [s.model_copy(update={"engagement_score": min(1.0, round(s.engagement_score + 0.02, 3))}) for s in obs.scenes]

        elif action.action_type == ActionType.add_subtitles:
            if obs.subtitles_present:
                reward -= 0.2
                info.update(valid=False, reason="subtitles already present")
            else:
                obs.subtitles_present = True
                obs.scenes = [s.model_copy(update={"engagement_score": min(1.0, round(s.engagement_score + 0.06, 3))}) for s in obs.scenes]

        elif action.action_type == ActionType.add_music:
            if self._music_applied:
                reward -= 0.2
                info.update(valid=False, reason="music already added")
            else:
                self._music_applied = True
                obs.scenes = [s.model_copy(update={"engagement_score": min(1.0, round(s.engagement_score + (0.12 if s.scene_type in ("hook", "highlight", "content") else 0.05), 3))}) for s in obs.scenes]

        elif action.action_type == ActionType.boost_hook:
            if self._hook_boosted:
                reward -= 0.2
                info.update(valid=False, reason="hook already boosted")
            elif not obs.scenes:
                reward -= 0.3
                info.update(valid=False, reason="no scenes present")
            else:
                self._hook_boosted = True
                first = obs.scenes[0]
                # FIX B: add noise to boost magnitude
                boost_noise = _noise()
                obs.scenes[0] = first.model_copy(update={
                    "hook_strength": min(1.0, round(first.hook_strength + 0.30 + boost_noise, 3)),
                    "engagement_score": min(1.0, round(first.engagement_score + 0.20 + boost_noise, 3)),
                    "transition_quality": min(1.0, round(first.transition_quality + 0.10, 3)),
                    "has_hook": True,
                    "scene_type": "hook" if first.scene_type not in ("hook", "highlight") else first.scene_type,
                })
                if len(obs.scenes) > 1:
                    second = obs.scenes[1]
                    obs.scenes[1] = second.model_copy(update={"engagement_score": min(1.0, round(second.engagement_score + 0.10 + _noise(), 3))})

        elif action.action_type == ActionType.trim_duration:
            target = float(action.parameters.get("target_seconds", PLATFORM_LIMITS.get(self.platform, 30.0)))
            if obs.total_duration <= target:
                reward -= 0.2
                info.update(valid=False, reason="duration already within target")
            else:
                scenes = list(obs.scenes)
                trimmed = False
                while _total_duration(scenes) > target and len(scenes) > 1:
                    candidates = [s for s in scenes[1:] if s.scene_type in ("filler", "transition", "content")]
                    if not candidates:
                        candidates = scenes[1:]
                    worst = min(candidates, key=lambda s: s.engagement_score)
                    scenes.remove(worst)
                    trimmed = True
                if not trimmed:
                    reward -= 0.1
                    info["note"] = "could not trim further"
                obs.scenes = [s.model_copy(update={"engagement_score": min(1.0, round(s.engagement_score + 0.03, 3))}) for s in scenes]

        elif action.action_type == ActionType.enhance_pacing:
            if self._pacing_enhanced:
                reward -= 0.2
                info.update(valid=False, reason="pacing already enhanced")
            else:
                self._pacing_enhanced = True
                scenes = obs.scenes
                if len(scenes) >= 3:
                    smoothed = [scenes[0]]
                    for i in range(1, len(scenes) - 1):
                        neighbour_avg = (scenes[i-1].engagement_score + scenes[i+1].engagement_score) / 2
                        new_eng = round(scenes[i].engagement_score * 0.6 + neighbour_avg * 0.4, 3)
                        smoothed.append(scenes[i].model_copy(update={"engagement_score": min(1.0, new_eng)}))
                    smoothed.append(scenes[-1])
                    obs.scenes = smoothed
                elif len(scenes) == 2:
                    obs.scenes = [s.model_copy(update={"engagement_score": min(1.0, round(s.engagement_score + 0.04, 3))}) for s in scenes]
                obs.scenes = [s.model_copy(update={"engagement_score": min(1.0, round(s.engagement_score + 0.02, 3))}) for s in obs.scenes]

        elif action.action_type == ActionType.improve_transition:
            if self._transition_improved:
                reward -= 0.2
                info.update(valid=False, reason="transitions already improved")
            else:
                self._transition_improved = True
                obs.scenes = [s.model_copy(update={
                    "transition_quality": min(1.0, round(s.transition_quality + 0.15, 3)),
                    "engagement_score": min(1.0, round(s.engagement_score + 0.03, 3)),
                }) for s in obs.scenes]

        elif action.action_type == ActionType.smooth_cut:
            if self._cut_smoothed:
                reward -= 0.2
                info.update(valid=False, reason="cuts already smoothed")
            else:
                self._cut_smoothed = True
                obs.scenes = [s.model_copy(update={
                    "cut_smoothness": min(1.0, round(s.cut_smoothness + 0.15, 3)),
                    "engagement_score": min(1.0, round(s.engagement_score + 0.02, 3)),
                }) for s in obs.scenes]

        elif action.action_type == ActionType.sync_audio:
            if self._audio_synced:
                reward -= 0.2
                info.update(valid=False, reason="audio already synced")
            else:
                self._audio_synced = True
                obs.scenes = [s.model_copy(update={
                    "audio_sync_score": min(1.0, round(s.audio_sync_score + 0.15, 3)),
                    "engagement_score": min(1.0, round(s.engagement_score + 0.03, 3)),
                }) for s in obs.scenes]

        elif action.action_type == ActionType.finalize_edit:
            # IRREVERSIBLE action — locks the episode and triggers persona-weighted final score
            if self._finalized:
                reward -= 0.3
                info.update(valid=False, reason="edit already finalized — irreversible")
            else:
                self._finalized = True
                # Persona-weighted bonus on finalize
                persona_data = _AUDIENCE_PERSONAS.get(self._persona, {})
                hook_w   = persona_data.get("hook_weight", 0.25)
                ret_w    = persona_data.get("retention_weight", 0.25)
                eng_w    = persona_data.get("engagement_weight", 0.25)
                comp_w   = persona_data.get("compliance_weight", 0.25)
                persona_score = (
                    obs.hook_strength * hook_w +
                    obs.avg_retention * ret_w +
                    obs.current_engagement_score * eng_w +
                    (1.0 if obs.platform_compliant else 0.0) * comp_w
                )
                reward += round(persona_score * 0.5, 4)
                info["persona"] = self._persona
                info["persona_score"] = round(persona_score, 4)
                info["note"] = f"Edit finalized for {self._persona} audience. Irreversible."
                # SAFE EXTENSION: mark irreversible decision point in metadata
                self._state.metadata["irreversible_decision_point"] = True
                self._state.metadata["finalized_at_step"] = self._state.step_count
                self._state.metadata["finalized_risk_score"] = _compute_risk_score(
                    obs.hook_strength, obs.avg_retention, obs.pacing_score
                )
                # Force episode done after finalize
                self._state.done = True

        obs = _build_observation(obs.scenes, self.platform,
                                  subtitles=obs.subtitles_present,
                                  music=self._music_applied,
                                  steps_remaining=max(0, MAX_STEPS - self._state.step_count - 1))

        if info["valid"]:
            reward += (obs.current_engagement_score - prev_eng) * 2.5
            reward += (obs.avg_retention - prev_ret) * 2.0
            reward += (obs.hook_strength - prev_hook) * 1.5
            reward += (obs.pacing_score - prev_pacing) * 1.0
            reward += obs.avg_transition_quality * 0.05
            reward += obs.avg_cut_smoothness * 0.05
            reward += obs.avg_audio_sync_score * 0.05
            if obs.platform_compliant and not prev_compliant:
                reward += 0.20
            elif obs.platform_compliant:
                reward += 0.05
            if obs.subtitles_present:
                reward += 0.10
            if obs.hook_first and not prev_hook_first:
                reward += 0.15
            elif obs.hook_first:
                reward += 0.03
            if obs.current_engagement_score >= 0.80 and prev_eng < 0.80:
                reward += 0.20
            if obs.hook_strength >= 0.70 and prev_hook < 0.70:
                reward += 0.15
            if len(obs.scenes) < 3:
                reward -= 0.10
            if (abs(obs.current_engagement_score - prev_eng) < 0.001
                    and abs(obs.avg_retention - prev_ret) < 0.001
                    and abs(obs.avg_transition_quality - prev_tq) < 0.001
                    and abs(obs.avg_cut_smoothness - prev_cs) < 0.001
                    and abs(obs.avg_audio_sync_score - prev_as) < 0.001):
                reward -= 0.05

        reward = round(reward, 4)
        self._state.step_count += 1
        self._state.observation = obs
        done = (
            self._state.step_count >= MAX_STEPS
            or obs.current_engagement_score >= ENGAGEMENT_THRESHOLD
            or self._finalized
        )
        self._state.done = done
        return deepcopy(self._state), reward, done, info

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return deepcopy(self._state)

    @property
    def action_history(self) -> List[str]:
        """FIX A: Returns the ordered list of actions taken this episode."""
        return list(self._action_history)

    def order_score(self) -> float:
        """
        FIX A: Order-sensitive scoring for task_3.
        Optimal: reorder_scenes -> boost_hook -> cut_scene ->
        enhance_pacing -> improve_transition -> smooth_cut ->
        sync_audio -> add_subtitles -> add_music
        Only penalizes actions that ARE done in wrong relative order.
        Optional actions (reorder_scenes, cut_scene) don't penalize if skipped.
        """
        optimal = [
            "reorder_scenes", "boost_hook", "cut_scene",
            "enhance_pacing", "improve_transition", "smooth_cut",
            "sync_audio", "add_subtitles", "add_music",
        ]
        history = [a for a in self._action_history if a in optimal]
        if not history:
            return 1.0
        # Check relative order of done actions against optimal
        done_in_optimal_order = [a for a in optimal if a in history]
        correct = sum(1 for a, b in zip(history, done_in_optimal_order) if a == b)
        return round(correct / len(history), 4)
