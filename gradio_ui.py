"""
gradio_ui.py — Live interactive demo for AI Video Optimizer Env.
Mounts as a sub-application on the FastAPI server at /ui
"""

import gradio as gr
import requests
import json

BASE = "http://localhost:7860"


def _post(path, **kwargs):
    try:
        r = requests.post(f"{BASE}{path}", timeout=15, **kwargs)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _get(path):
    try:
        r = requests.get(f"{BASE}{path}", timeout=15)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def reset_env(platform, seed):
    state = _post("/reset", params={"platform": platform, "seed": int(seed)})
    if not state:
        return "❌ Reset failed", "", "", ""
    obs = state.get("observation", {})
    meta = state.get("metadata", {})
    scenes = obs.get("scenes", [])

    status = f"""✅ Episode started
Platform: {obs.get('platform','?')}  |  Seed: {seed}  |  Persona: {meta.get('audience_persona','?')}
Steps remaining: {obs.get('steps_remaining', 15)}  |  Risk score: {obs.get('risk_score', 0):.3f}"""

    obs_text = f"""Engagement:  {obs.get('current_engagement_score', 0):.3f}
Retention:   {obs.get('avg_retention', 0):.3f}
Hook:        {obs.get('hook_strength', 0):.3f}
Pacing:      {obs.get('pacing_score', 0):.3f}
Duration:    {obs.get('total_duration', 0):.1f}s
Compliant:   {obs.get('platform_compliant', False)}
Subtitles:   {obs.get('subtitles_present', False)}
Music:       {obs.get('music_added', False)}
Risk score:  {obs.get('risk_score', 0):.3f}"""

    scenes_text = "\n".join([
        f"  {s['id']:<10} type={s['scene_type']:<12} eng={s['engagement_score']:.2f}  hook={s['hook_strength']:.2f}"
        for s in scenes
    ])

    hint = _get("/hint")
    hint_text = f"💡 {hint.get('best_action','?')}: {hint.get('reason','')}" if hint else ""

    return status, obs_text, scenes_text, hint_text


def take_action(action_type, scene_id):
    params = {}
    if action_type == "cut_scene" and scene_id.strip():
        params = {"scene_id": scene_id.strip()}
    elif action_type == "reorder_scenes":
        # auto-reorder: put hook-first
        state = _get("/state")
        obs = state.get("observation", {})
        scenes = obs.get("scenes", [])
        hooks = [s for s in scenes if s.get("scene_type") in ("hook","highlight") and s.get("has_hook")]
        if hooks:
            best = max(hooks, key=lambda s: s.get("hook_strength", 0))
            order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
            params = {"order": order}

    resp = _post("/step", json={"action_type": action_type, "parameters": params})
    if not resp:
        return "❌ Step failed", "", "", ""

    state = resp.get("state", {})
    reward = resp.get("reward", 0.0)
    done = resp.get("done", False)
    info = resp.get("info", {})
    obs = state.get("observation", {})
    scenes = obs.get("scenes", [])

    valid = info.get("valid", True)
    icon = "✅" if valid else "⚠️"
    reason = info.get("reason", "")

    status = f"""{icon} Action: {action_type}  |  Reward: {reward:+.4f}  |  Done: {done}
Step: {state.get('step_count','?')}  |  Steps left: {obs.get('steps_remaining','?')}
{('Reason: ' + reason) if reason else ''}"""

    obs_text = f"""Engagement:  {obs.get('current_engagement_score', 0):.3f}
Retention:   {obs.get('avg_retention', 0):.3f}
Hook:        {obs.get('hook_strength', 0):.3f}
Pacing:      {obs.get('pacing_score', 0):.3f}
Duration:    {obs.get('total_duration', 0):.1f}s
Compliant:   {obs.get('platform_compliant', False)}
Subtitles:   {obs.get('subtitles_present', False)}
Music:       {obs.get('music_added', False)}
Risk score:  {obs.get('risk_score', 0):.3f}"""

    scenes_text = "\n".join([
        f"  {s['id']:<10} type={s['scene_type']:<12} eng={s['engagement_score']:.2f}  hook={s['hook_strength']:.2f}"
        for s in scenes
    ])

    hint = _get("/hint")
    hint_text = f"💡 {hint.get('best_action','?')}: {hint.get('reason','')}" if hint else ""

    return status, obs_text, scenes_text, hint_text


def grade_env():
    g = _get("/grader")
    if not g:
        return "❌ Grader failed"

    score = g.get("score", 0)
    passed = g.get("passed", False)
    raw = g.get("raw_score", 0)
    rubric = g.get("rubric_score", 0)
    explanation = g.get("explanation", "")
    meta = g.get("grader_metadata", {})
    bd = g.get("breakdown", {})

    result = f"""{'🏆 PASSED' if passed else '❌ FAILED'}  Score: {score:.4f}
Raw score:    {raw:.4f}
Rubric score: {rubric:.4f}
Risk score:   {meta.get('risk_score', 0):.4f}
Cap applied:  {meta.get('score_cap_applied', False)}  ({meta.get('cap_reason') or 'none'})

📊 Breakdown:
  Engagement:   {bd.get('engagement', 0):.4f}
  Retention:    {bd.get('retention', 0):.4f}
  Compliance:   {bd.get('platform_compliance', 0):.4f}
  Subtitles:    {bd.get('subtitles', 0):.4f}
  Hook:         {bd.get('hook_strength', 0):.4f}
  Pacing:       {bd.get('pacing', 0):.4f}
  Efficiency:   {bd.get('efficiency_bonus', 0):+.4f}
  Order:        {bd.get('order_penalty', 0):+.4f}

💬 {explanation}"""

    return result


def get_strategy():
    s = _get("/strategy")
    if not s:
        return "❌ Strategy failed"

    seq = s.get("recommended_sequence", [])
    seq_text = "\n".join([
        f"  {i+1}. {a['action']} — {a['reason']}"
        for i, a in enumerate(seq[:6])
    ])

    return f"""🧠 Strategy: {s.get('strategy_mode','?').upper()}
Risk: {s.get('risk_score', 0):.3f} ({s.get('risk_label','?')})
Point of no recovery: {s.get('point_of_no_recovery', False)}
Steps remaining: {s.get('steps_remaining','?')}

🎯 Immediate action: {s.get('immediate_action','?')}

📋 Recommended sequence:
{seq_text}

👥 {s.get('persona_focus','')}
{s.get('reasoning','')}"""


def build_ui():
    with gr.Blocks(title="AI Video Optimizer — RL Environment") as demo:

        gr.Markdown("""
# 🎬 AI Video Optimizer — Decision-Constrained RL Environment

An OpenEnv environment where agents make **irreversible editorial decisions under uncertainty**.
`cut_scene` is permanent. `finalize_edit` ends the episode. `risk_score` tracks proximity to failure.

**API:** `https://saravanabalajisara-ai-video-optimizer-env.hf.space` | **Docs:** `/docs`
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Reset Episode")
                platform = gr.Dropdown(
                    ["reels", "shorts", "tiktok"], value="reels", label="Platform"
                )
                seed = gr.Number(value=42, label="Seed", precision=0)
                reset_btn = gr.Button("🔄 Reset", variant="primary")
                status_box = gr.Textbox(label="Status", lines=4, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Observation")
                obs_box = gr.Textbox(label="Current State", lines=10, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Scenes")
                scenes_box = gr.Textbox(label="Scene List", lines=8, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 💡 Hint")
                hint_box = gr.Textbox(label="Best Next Action", lines=3, interactive=False)

        gr.Markdown("### ▶️ Take Action")
        with gr.Row():
            action_dd = gr.Dropdown(
                choices=[
                    "boost_hook", "enhance_pacing", "improve_transition",
                    "smooth_cut", "sync_audio", "add_subtitles", "add_music",
                    "cut_scene", "reorder_scenes", "trim_duration", "finalize_edit"
                ],
                value="boost_hook",
                label="Action",
                scale=2,
            )
            scene_id_box = gr.Textbox(
                label="scene_id (for cut_scene only)", placeholder="e.g. scene_3", scale=1
            )
            step_btn = gr.Button("▶ Step", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🏆 Grade")
                grade_btn = gr.Button("📊 Grade Current State", variant="secondary")
                grade_box = gr.Textbox(label="Grader Result", lines=16, interactive=False)

            with gr.Column():
                gr.Markdown("### 🧠 Strategy")
                strategy_btn = gr.Button("🧠 Get Strategy Plan", variant="secondary")
                strategy_box = gr.Textbox(label="Strategy Engine", lines=16, interactive=False)

        # Wire up
        reset_btn.click(
            reset_env,
            inputs=[platform, seed],
            outputs=[status_box, obs_box, scenes_box, hint_box]
        )
        step_btn.click(
            take_action,
            inputs=[action_dd, scene_id_box],
            outputs=[status_box, obs_box, scenes_box, hint_box]
        )
        grade_btn.click(grade_env, outputs=[grade_box])
        strategy_btn.click(get_strategy, outputs=[strategy_box])

    return demo
