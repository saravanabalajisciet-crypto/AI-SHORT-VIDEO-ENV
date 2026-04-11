"""
strategy_engine.py — Intelligent strategy layer for AI Video Optimizer Env.

Provides optimal action sequence suggestions based on current observation,
risk_score, and audience persona. Completely standalone — does NOT modify
any existing endpoint, reward, or environment logic.

Usage:
    from strategy_engine import suggest_strategy
    plan = suggest_strategy(obs_dict, persona="gen_z", steps_used=3)
"""

from typing import Dict, Any, List, Optional


# ── Optimal action ordering (mirrors order_score in environment.py) ────────────
_OPTIMAL_ORDER = [
    "reorder_scenes",
    "boost_hook",
    "cut_scene",
    "trim_duration",
    "enhance_pacing",
    "improve_transition",
    "smooth_cut",
    "sync_audio",
    "add_subtitles",
    "add_music",
]

# ── Persona-specific priorities ────────────────────────────────────────────────
_PERSONA_PRIORITIES: Dict[str, Dict[str, float]] = {
    "gen_z":      {"hook": 1.5, "engagement": 1.3, "retention": 1.0, "compliance": 0.8},
    "millennial": {"hook": 1.0, "engagement": 1.1, "retention": 1.5, "compliance": 1.0},
    "brand":      {"hook": 0.9, "engagement": 1.0, "retention": 1.0, "compliance": 1.8},
}


def _risk_label(risk: float) -> str:
    if risk < 0.4:
        return "low"
    if risk < 0.7:
        return "medium"
    return "high"


def suggest_strategy(
    obs: Dict[str, Any],
    persona: str = "gen_z",
    steps_used: int = 0,
    max_steps: int = 15,
) -> Dict[str, Any]:
    """
    Suggest an optimal action plan given the current observation.

    Args:
        obs:        observation dict (from /state or /step response)
        persona:    audience persona (gen_z | millennial | brand)
        steps_used: steps already taken this episode
        max_steps:  episode step budget

    Returns:
        {
          "recommended_sequence": [...],   # ordered list of remaining actions
          "immediate_action": str,         # single best next action
          "immediate_params": dict,        # parameters for immediate_action
          "risk_label": str,               # low | medium | high
          "risk_score": float,
          "point_of_no_recovery": bool,    # True if risk > 0.7 and steps > 8
          "strategy_mode": str,            # aggressive | balanced | conservative
          "reasoning": str,
          "steps_remaining": int,
          "persona_focus": str,
        }
    """
    steps_remaining = max_steps - steps_used
    risk = float(obs.get("risk_score") or 0.0)
    hook = float(obs.get("hook_strength", 0.0))
    retention = float(obs.get("avg_retention", 0.0))
    engagement = float(obs.get("current_engagement_score", 0.0))
    pacing = float(obs.get("pacing_score", 0.0))
    tq = float(obs.get("avg_transition_quality", 0.0))
    cs = float(obs.get("avg_cut_smoothness", 0.0))
    asy = float(obs.get("avg_audio_sync_score", 0.0))
    compliant = bool(obs.get("platform_compliant", True))
    hook_first = bool(obs.get("hook_first", False))
    subtitles = bool(obs.get("subtitles_present", False))
    music = bool(obs.get("music_added", False))
    duration = float(obs.get("total_duration", 0.0))
    scenes = obs.get("scenes", [])

    prio = _PERSONA_PRIORITIES.get(persona, _PERSONA_PRIORITIES["gen_z"])
    ponr = risk > 0.7 and steps_used > 8

    # ── Determine strategy mode ────────────────────────────────────────────────
    if risk < 0.4:
        mode = "aggressive"   # explore freely, maximize score
    elif risk < 0.7:
        mode = "balanced"     # careful sequencing
    else:
        mode = "conservative" # only safe, high-impact actions

    # ── Build recommended sequence ─────────────────────────────────────────────
    sequence: List[Dict[str, Any]] = []

    # 1. Reorder if hook not first
    if not hook_first:
        hook_scenes = [s for s in scenes
                       if isinstance(s, dict)
                       and s.get("scene_type") in ("hook", "highlight")
                       and s.get("has_hook")]
        if hook_scenes:
            best = max(hook_scenes, key=lambda s: float(s.get("hook_strength", 0.0)))
            order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
            sequence.append({"action": "reorder_scenes", "params": {"order": order},
                              "reason": "Hook-first placement is mandatory for viral reach."})

    # 2. Boost hook if weak
    if hook < 0.75:
        sequence.append({"action": "boost_hook", "params": {},
                          "reason": f"hook_strength={hook:.2f} < 0.75. +0.30 lift available."})

    # 3. Cut low-engagement filler
    if mode != "conservative":
        low = sorted(
            [s for s in scenes
             if isinstance(s, dict)
             and float(s.get("engagement_score", 1.0)) < 0.30
             and s.get("scene_type") in ("filler", "transition")],
            key=lambda s: float(s.get("engagement_score", 0.0)),
        )
        if low and len(scenes) > 3:
            sequence.append({"action": "cut_scene",
                              "params": {"scene_id": low[0]["id"]},
                              "reason": f"Scene {low[0]['id']} engagement={low[0].get('engagement_score',0):.2f}. Drag on retention."})

    # 4. Trim if non-compliant
    if not compliant:
        sequence.append({"action": "trim_duration", "params": {"target_seconds": 30.0},
                          "reason": "Duration exceeds platform limit. -0.20 compliance penalty."})

    # 5. Pacing
    if retention < 0.6 or pacing < 0.5:
        sequence.append({"action": "enhance_pacing", "params": {},
                          "reason": f"avg_retention={retention:.2f}, pacing={pacing:.2f}. Smooth transitions."})

    # 6. Production quality (only if budget allows or mode is not conservative)
    if tq < 0.72:
        sequence.append({"action": "improve_transition", "params": {},
                          "reason": f"transition_quality={tq:.2f} < 0.72."})
    if cs < 0.72:
        sequence.append({"action": "smooth_cut", "params": {},
                          "reason": f"cut_smoothness={cs:.2f} < 0.72."})
    if asy < 0.72:
        sequence.append({"action": "sync_audio", "params": {},
                          "reason": f"audio_sync={asy:.2f} < 0.72."})

    # 7. Subtitles + music last
    if not subtitles:
        sequence.append({"action": "add_subtitles", "params": {},
                          "reason": "+0.10 grader score. Always add."})
    if not music:
        sequence.append({"action": "add_music", "params": {},
                          "reason": "+0.12 on hook/content scenes."})

    # Trim to remaining budget
    sequence = sequence[:steps_remaining]

    # ── Immediate action ───────────────────────────────────────────────────────
    immediate = sequence[0] if sequence else None

    # ── Persona focus message ──────────────────────────────────────────────────
    persona_focus = {
        "gen_z":      "Prioritize hook_strength and engagement — Gen-Z scrolls fast.",
        "millennial": "Prioritize retention and watch_time — Millennials watch to completion.",
        "brand":      "Prioritize platform_compliance and reach — Brand safety is critical.",
    }.get(persona, "Balanced optimization.")

    # ── Reasoning summary ─────────────────────────────────────────────────────
    if ponr:
        reasoning = (
            f"POINT OF NO RECOVERY: risk={risk:.2f} at step {steps_used}. "
            f"Only {steps_remaining} steps left. Focus on highest-impact actions only."
        )
    elif mode == "aggressive":
        reasoning = f"Low risk ({risk:.2f}). Explore freely — full sequence recommended."
    elif mode == "balanced":
        reasoning = f"Medium risk ({risk:.2f}). Follow optimal order carefully."
    else:
        reasoning = f"High risk ({risk:.2f}). Conservative actions only — avoid cuts."

    return {
        "recommended_sequence": sequence,
        "immediate_action": immediate["action"] if immediate else None,
        "immediate_params": immediate["params"] if immediate else {},
        "risk_label": _risk_label(risk),
        "risk_score": round(risk, 4),
        "point_of_no_recovery": ponr,
        "strategy_mode": mode,
        "reasoning": reasoning,
        "steps_remaining": steps_remaining,
        "persona_focus": persona_focus,
    }


def explain_score(
    obs: Dict[str, Any],
    score: float,
    raw_score: float,
    breakdown: Dict[str, Any],
    action_history: List[str],
    cap_applied: bool = False,
    cap_reason: Optional[str] = None,
) -> str:
    """
    Generate a human-readable explanation of why the grader gave this score.
    Used by the /grader endpoint's explanation field.
    """
    parts = []

    # Overall verdict
    if score >= 0.90:
        parts.append("Elite performance — video is fully optimized for viral reach.")
    elif score >= 0.875:
        parts.append("Strong result — passes the hard task threshold.")
    elif score >= 0.78:
        parts.append("Good edit — passes medium task threshold but has room to improve.")
    elif score >= 0.65:
        parts.append("Acceptable — passes easy task threshold. Significant gains still available.")
    else:
        parts.append("Below passing threshold. Core issues remain unresolved.")

    # Top contributors
    top = sorted(
        [(k, v) for k, v in breakdown.items()
         if k not in ("raw_score", "rubric_score", "final_score",
                      "efficiency_bonus", "order_penalty", "order_score")
         and isinstance(v, (int, float))],
        key=lambda x: x[1], reverse=True,
    )[:3]
    if top:
        contrib = ", ".join(f"{k}={v:.3f}" for k, v in top)
        parts.append(f"Top score contributors: {contrib}.")

    # Efficiency
    eff = breakdown.get("efficiency_bonus", 0.0)
    if eff > 0:
        parts.append(f"Efficiency bonus +{eff:.2f} earned (solved in few steps).")
    elif eff < 0:
        parts.append(f"Efficiency penalty {eff:.2f} applied (too many steps).")

    # Order
    order_pen = breakdown.get("order_penalty", 0.0)
    if order_pen < 0:
        parts.append(f"Action order penalty {order_pen:.2f} — actions were not in optimal sequence.")

    # Soft cap
    if cap_applied and cap_reason:
        label = {
            "low_hook_strength": "hook was not boosted early enough",
            "low_retention": "retention dropped critically low",
            "platform_non_compliant": "video was not platform-compliant",
        }.get(cap_reason, cap_reason)
        parts.append(f"Score cap applied ({label}) — rubric_score limited by decision pressure.")

    # Missing actions
    missing = []
    if not obs.get("subtitles_present"):
        missing.append("add_subtitles")
    if not obs.get("music_added"):
        missing.append("add_music")
    if obs.get("hook_strength", 0) < 0.5:
        missing.append("boost_hook")
    if missing:
        parts.append(f"Actions not taken that would improve score: {', '.join(missing)}.")

    return " ".join(parts)
