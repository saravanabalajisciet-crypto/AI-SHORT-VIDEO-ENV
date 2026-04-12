"""
gradio_ui.py — Live interactive demo for AI Video Optimizer Env.
Mounts as a sub-application on the FastAPI server at /ui

Pre-loaded with a BMW drift video scenario so the panel sees
a real editing session the moment they open the page.
"""

import gradio as gr
import requests

BASE = "http://localhost:7860"

# ── BMW Drift scenario — pre-loaded on page open ───────────────────────────────
# Seed 21 = tiktok platform, high-energy niche, hook-first challenge
BMW_SCENARIO = {
    "platform": "tiktok",
    "seed": 21,
    "label": "🚗 BMW Drift — TikTok Short",
    "description": (
        "Raw footage: BMW M3 drift sequence. 5 scenes, 58s total — over TikTok's 60s limit.\n"
        "Hook scene is buried at position 3. Filler transition at scene_1 dragging retention.\n"
        "Agent must reorder → boost → cut filler → trim → enhance → finalize before 15 steps."
    ),
}

SCENARIOS = {
    "🚗 BMW Drift — TikTok Short (seed 21)":       {"platform": "tiktok",  "seed": 21},
    "🎣 Fishing Highlight — Reels (seed 42)":       {"platform": "reels",   "seed": 42},
    "📱 Tech Review — YouTube Shorts (seed 7)":     {"platform": "shorts",  "seed": 7},
    "🔥 Retention Crisis — Reels (seed 13)":        {"platform": "reels",   "seed": 13},
    "💪 Fitness Hook Challenge — TikTok (seed 99)": {"platform": "tiktok",  "seed": 99},
}


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


def _format_obs(obs, meta=None):
    risk = obs.get("risk_score", 0) or 0
    risk_icon = "🟢" if risk < 0.4 else "🟡" if risk < 0.7 else "🔴"
    compliant = obs.get("platform_compliant", False)
    persona = (meta or {}).get("audience_persona", "?")
    return f"""📊  CURRENT STATE
{'─'*35}
Engagement:   {obs.get('current_engagement_score', 0):.3f}
Retention:    {obs.get('avg_retention', 0):.3f}
Hook:         {obs.get('hook_strength', 0):.3f}
Pacing:       {obs.get('pacing_score', 0):.3f}
Duration:     {obs.get('total_duration', 0):.1f}s  {'✅' if compliant else '❌ OVER LIMIT'}
Subtitles:    {'✅' if obs.get('subtitles_present') else '❌'}
Music:        {'✅' if obs.get('music_added') else '❌'}
Hook-first:   {'✅' if obs.get('hook_first') else '❌'}
{risk_icon} Risk score:  {risk:.3f}
Steps left:   {obs.get('steps_remaining', 15)}
Persona:      {persona}"""


def _format_scenes(scenes):
    if not scenes:
        return "No scenes"
    lines = ["🎬  SCENE TIMELINE", "─" * 50]
    for i, s in enumerate(scenes):
        eng = s.get("engagement_score", 0)
        hook = s.get("hook_strength", 0)
        stype = s.get("scene_type", "?")
        sid = s.get("id", "?")
        has_hook = s.get("has_hook", False)
        bar = "█" * int(eng * 10) + "░" * (10 - int(eng * 10))
        flag = " ← 🎯 HOOK" if (has_hook and hook > 0.3) else ""
        warn = " ⚠️ LOW" if eng < 0.30 else ""
        lines.append(f"  [{i}] {sid:<10} {stype:<12} [{bar}] {eng:.2f}{flag}{warn}")
    return "\n".join(lines)


def load_scenario(scenario_name):
    cfg = SCENARIOS.get(scenario_name, {"platform": "tiktok", "seed": 21})
    return reset_env(cfg["platform"], cfg["seed"])


def reset_env(platform, seed):
    state = _post("/reset", params={"platform": platform, "seed": int(seed)})
    if not state:
        return "❌ Reset failed — is the server running?", "", "", "", ""

    obs = state.get("observation", {})
    meta = state.get("metadata", {})
    scenes = obs.get("scenes", [])
    risk = obs.get("risk_score", 0) or 0
    risk_label = "🟢 LOW" if risk < 0.4 else "🟡 MEDIUM" if risk < 0.7 else "🔴 HIGH"

    status = f"""✅  EPISODE STARTED
{'─'*35}
Platform:  {obs.get('platform','?').upper()}
Seed:      {seed}
Persona:   {meta.get('audience_persona','?')}
Scenes:    {len(scenes)}
Duration:  {obs.get('total_duration',0):.1f}s
Risk:      {risk_label} ({risk:.3f})
Steps:     {obs.get('steps_remaining',15)} remaining"""

    hint = _get("/hint")
    hint_text = ""
    if hint and hint.get("best_action"):
        hint_text = f"💡 Best action: {hint['best_action']}\n   {hint.get('reason','')}"

    return status, _format_obs(obs, meta), _format_scenes(scenes), hint_text, ""


def take_action(action_type, scene_id):
    params = {}
    if action_type == "cut_scene":
        if not scene_id.strip():
            # auto-pick lowest engagement filler
            state = _get("/state")
            obs = state.get("observation", {})
            scenes = obs.get("scenes", [])
            fillers = sorted(
                [s for s in scenes if s.get("scene_type") in ("filler","transition")
                 and s.get("engagement_score", 1) < 0.35],
                key=lambda s: s.get("engagement_score", 1)
            )
            if fillers:
                params = {"scene_id": fillers[0]["id"]}
                scene_id = fillers[0]["id"]
            else:
                return "⚠️ No low-engagement filler found to cut. Specify scene_id manually.", "", "", "", ""
        else:
            params = {"scene_id": scene_id.strip()}

    elif action_type == "reorder_scenes":
        state = _get("/state")
        obs = state.get("observation", {})
        scenes = obs.get("scenes", [])
        hooks = [s for s in scenes if s.get("scene_type") in ("hook","highlight") and s.get("has_hook")]
        if hooks:
            best = max(hooks, key=lambda s: s.get("hook_strength", 0))
            order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
            params = {"order": order}
        else:
            return "⚠️ No hook scene found to reorder to front.", "", "", "", ""

    elif action_type == "trim_duration":
        state = _get("/state")
        obs = state.get("observation", {})
        platform = obs.get("platform", "reels")
        limits = {"reels": 30.0, "shorts": 60.0, "tiktok": 60.0}
        params = {"target_seconds": limits.get(platform, 30.0)}

    resp = _post("/step", json={"action_type": action_type, "parameters": params})
    if not resp:
        return "❌ Step failed", "", "", "", ""

    state = resp.get("state", {})
    reward = resp.get("reward", 0.0)
    done = resp.get("done", False)
    info = resp.get("info", {})
    obs = state.get("observation", {})
    meta = state.get("metadata", {})
    scenes = obs.get("scenes", [])

    valid = info.get("valid", True)
    icon = "✅" if valid else "⚠️"
    reason = info.get("reason", "")
    reward_icon = "📈" if reward > 0 else "📉"

    status = f"""{icon}  ACTION: {action_type.upper()}
{'─'*35}
{reward_icon} Reward:    {reward:+.4f}
Step:      {state.get('step_count','?')} / {state.get('max_steps',15)}
Steps left:{obs.get('steps_remaining','?')}
Done:      {'🏁 YES' if done else 'No'}
{('⚠️  ' + reason) if reason else ''}
{('ℹ️  ' + info.get('note','')) if info.get('note') else ''}"""

    if done and info.get("persona"):
        status += f"\n\n🎭 Finalized for {info['persona']} — persona score: {info.get('persona_score',0):.3f}"

    hint = _get("/hint")
    hint_text = ""
    if hint and hint.get("best_action") and not done:
        hint_text = f"💡 Best action: {hint['best_action']}\n   {hint.get('reason','')}"
    elif done:
        hint_text = "🏁 Episode complete — click Grade to see final score"

    traj_note = ""
    traj = _get("/trajectory")
    if traj and traj.get("episode_steps", 0) > 0:
        dqs = traj.get("decision_quality_summary", {})
        traj_note = f"📈 Trajectory: {traj['episode_steps']} steps | excellent={dqs.get('excellent',0)} good={dqs.get('good',0)} poor={dqs.get('poor',0)}"

    return status, _format_obs(obs, meta), _format_scenes(scenes), hint_text, traj_note


def grade_env():
    g = _get("/grader")
    if not g:
        return "❌ Grader failed — reset and take at least one step first"

    score = g.get("score", 0)
    passed = g.get("passed", False)
    raw = g.get("raw_score", 0)
    rubric = g.get("rubric_score", 0)
    explanation = g.get("explanation", "")
    meta = g.get("grader_metadata", {})
    bd = g.get("breakdown", {})

    bar_len = int(score * 20)
    score_bar = "█" * bar_len + "░" * (20 - bar_len)

    cap_info = ""
    if meta.get("score_cap_applied"):
        cap_info = f"\n⚠️  Score cap: {meta.get('cap_reason','?')} — early decision penalty applied"

    irrev = ""
    if meta.get("irreversible_decision_point"):
        irrev = f"\n🔒 Finalized at step {meta.get('finalized_at_step','?')} | risk at commit: {meta.get('finalized_risk_score',0):.3f}"

    return f"""{'🏆 PASSED' if passed else '❌ FAILED'}
{'─'*40}
Score:        [{score_bar}] {score:.4f}
Raw score:    {raw:.4f}  (pure observation quality)
Rubric score: {rubric:.4f}  (RL training signal){cap_info}{irrev}

📊  BREAKDOWN
  Engagement:    {bd.get('engagement',0):.4f}  (weight 0.35)
  Retention:     {bd.get('retention',0):.4f}  (weight 0.15)
  Compliance:    {bd.get('platform_compliance',0):.4f}  (weight 0.20)
  Subtitles:     {bd.get('subtitles',0):.4f}  (weight 0.10)
  Hook:          {bd.get('hook_strength',0):.4f}  (weight 0.10)
  Pacing:        {bd.get('pacing',0):.4f}  (weight 0.05)
  Transition:    {bd.get('transition_quality',0):.4f}
  Cut smooth:    {bd.get('cut_smoothness',0):.4f}
  Audio sync:    {bd.get('audio_sync',0):.4f}
  Efficiency:    {bd.get('efficiency_bonus',0):+.4f}
  Order penalty: {bd.get('order_penalty',0):+.4f}

💬  {explanation}"""


def get_strategy():
    s = _get("/strategy")
    if not s:
        return "❌ Strategy failed — reset first"

    seq = s.get("recommended_sequence", [])
    seq_text = "\n".join([
        f"  {i+1}. {a['action']:<22} {a['reason'][:60]}"
        for i, a in enumerate(seq[:8])
    ]) or "  (no actions needed — well optimized)"

    risk = s.get("risk_score", 0)
    risk_icon = "🟢" if risk < 0.4 else "🟡" if risk < 0.7 else "🔴"
    ponr = s.get("point_of_no_recovery", False)

    return f"""🧠  STRATEGY ENGINE
{'─'*40}
Mode:     {s.get('strategy_mode','?').upper()}
{risk_icon} Risk:     {risk:.3f} ({s.get('risk_label','?')})
PONR:     {'🚨 YES — limited recovery' if ponr else 'No'}
Steps:    {s.get('steps_remaining','?')} remaining

🎯  IMMEDIATE ACTION: {s.get('immediate_action','?') or 'none needed'}

📋  RECOMMENDED SEQUENCE:
{seq_text}

👥  {s.get('persona_focus','')}

💭  {s.get('reasoning','')}"""


def build_ui():
    with gr.Blocks(title="🎬 AI Video Optimizer — RL Environment Demo") as demo:

        gr.Markdown("""
# 🎬 AI Video Optimizer — Decision-Constrained RL Environment

> Agents make **irreversible editorial decisions** under uncertainty.
> `cut_scene` is permanent. `finalize_edit` ends the episode. `risk_score` tracks proximity to failure.

| 🚀 [Live API](https://saravanabalajisara-ai-video-optimizer-env.hf.space) | 📖 [Docs](https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs) | 💻 [GitHub](https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV) |
|---|---|---|
        """)

        # ── Scenario selector ──────────────────────────────────────────────────
        with gr.Row():
            scenario_dd = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value="🚗 BMW Drift — TikTok Short (seed 21)",
                label="📽️ Select Scenario",
                scale=3,
            )
            load_btn = gr.Button("🎬 Load Scenario", variant="primary", scale=1)

        gr.Markdown("""
> **🚗 BMW Drift scenario:** Raw footage — 5 scenes, 58s, hook buried at position 3.
> Agent must reorder → boost → cut filler → trim → enhance → finalize in ≤ 15 steps.
        """)

        # ── Manual reset ───────────────────────────────────────────────────────
        with gr.Accordion("⚙️ Manual Reset (custom platform + seed)", open=False):
            with gr.Row():
                platform_dd = gr.Dropdown(["reels", "shorts", "tiktok"], value="tiktok", label="Platform", scale=1)
                seed_num = gr.Number(value=21, label="Seed", precision=0, scale=1)
                reset_btn = gr.Button("🔄 Reset", scale=1)

        # ── State display ──────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                status_box = gr.Textbox(label="📋 Episode Status", lines=9, interactive=False)
            with gr.Column(scale=1):
                obs_box = gr.Textbox(label="📊 Observation", lines=9, interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                scenes_box = gr.Textbox(label="🎬 Scene Timeline", lines=9, interactive=False)
            with gr.Column(scale=1):
                hint_box = gr.Textbox(label="💡 Hint", lines=4, interactive=False)
                traj_box = gr.Textbox(label="📈 Trajectory", lines=3, interactive=False)

        # ── Action panel ───────────────────────────────────────────────────────
        gr.Markdown("### ▶️ Take Action")
        with gr.Row():
            action_dd = gr.Dropdown(
                choices=[
                    "reorder_scenes", "boost_hook", "cut_scene", "trim_duration",
                    "enhance_pacing", "improve_transition", "smooth_cut", "sync_audio",
                    "add_subtitles", "add_music", "finalize_edit"
                ],
                value="reorder_scenes",
                label="Action (optimal order shown)",
                scale=2,
            )
            scene_id_box = gr.Textbox(
                label="scene_id (cut_scene only — leave blank to auto-pick worst filler)",
                placeholder="e.g. scene_1",
                scale=2,
            )
            step_btn = gr.Button("▶ Execute", variant="primary", scale=1)

        gr.Markdown("""
> **Optimal sequence:** `reorder_scenes` → `boost_hook` → `cut_scene` → `trim_duration` →
> `enhance_pacing` → `improve_transition` → `smooth_cut` → `sync_audio` → `add_subtitles` → `add_music`
> → `finalize_edit`
        """)

        # ── Grade + Strategy ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                grade_btn = gr.Button("📊 Grade Current State", variant="secondary")
                grade_box = gr.Textbox(label="🏆 Grader Result", lines=22, interactive=False)
            with gr.Column():
                strategy_btn = gr.Button("🧠 Get Strategy Plan", variant="secondary")
                strategy_box = gr.Textbox(label="🧠 Strategy Engine", lines=22, interactive=False)

        # ── Wire up ────────────────────────────────────────────────────────────
        load_btn.click(
            load_scenario,
            inputs=[scenario_dd],
            outputs=[status_box, obs_box, scenes_box, hint_box, traj_box]
        )
        reset_btn.click(
            reset_env,
            inputs=[platform_dd, seed_num],
            outputs=[status_box, obs_box, scenes_box, hint_box, traj_box]
        )
        step_btn.click(
            take_action,
            inputs=[action_dd, scene_id_box],
            outputs=[status_box, obs_box, scenes_box, hint_box, traj_box]
        )
        grade_btn.click(grade_env, outputs=[grade_box])
        strategy_btn.click(get_strategy, outputs=[strategy_box])

        # ── Auto-load BMW drift on page open ───────────────────────────────────
        demo.load(
            lambda: load_scenario("🚗 BMW Drift — TikTok Short (seed 21)"),
            outputs=[status_box, obs_box, scenes_box, hint_box, traj_box]
        )

    return demo
