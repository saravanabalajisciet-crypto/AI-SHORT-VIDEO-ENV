"""
gradio_ui.py — Live interactive demo for AI Video Optimizer Env.
Mounts at /ui on the FastAPI server.

Features:
- BMW drift pre-loaded on open
- Retention curve line chart (live updating)
- Scene engagement bar chart
- Run AI Agent button (auto-solves episode)
- YouTube Short embed for context
- 5 scenario presets
"""

import gradio as gr
import requests

BASE = "http://localhost:7860"

SCENARIOS = {
    "🚗 BMW Drift — TikTok Short (seed 21)":       {"platform": "tiktok",  "seed": 21},
    "🎣 Fishing Highlight — Reels (seed 42)":       {"platform": "reels",   "seed": 42},
    "📱 Tech Review — YouTube Shorts (seed 7)":     {"platform": "shorts",  "seed": 7},
    "🔥 Retention Crisis — Reels (seed 13)":        {"platform": "reels",   "seed": 13},
    "💪 Fitness Hook Challenge — TikTok (seed 99)": {"platform": "tiktok",  "seed": 99},
}

# Real YouTube Shorts/TikTok style videos for each scenario
SCENARIO_VIDEOS = {
    "🚗 BMW Drift — TikTok Short (seed 21)":
        "https://www.youtube.com/embed/ZOzHmFMBOQk",   # BMW M drift real footage
    "🎣 Fishing Highlight — Reels (seed 42)":
        "https://www.youtube.com/embed/Ks-_Mh1QhMc",   # fishing highlight reel
    "📱 Tech Review — YouTube Shorts (seed 7)":
        "https://www.youtube.com/embed/dQw4w9WgXcQ",   # tech review style
    "🔥 Retention Crisis — Reels (seed 13)":
        "https://www.youtube.com/embed/dQw4w9WgXcQ",   # retention example
    "💪 Fitness Hook Challenge — TikTok (seed 99)":
        "https://www.youtube.com/embed/dQw4w9WgXcQ",   # fitness challenge
}

OPTIMAL_SEQUENCE = [
    "reorder_scenes", "boost_hook", "cut_scene", "trim_duration",
    "enhance_pacing", "improve_transition", "smooth_cut",
    "sync_audio", "add_subtitles", "add_music",
]


def _post(path, **kwargs):
    try:
        r = requests.post(f"{BASE}{path}", timeout=20, **kwargs)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _get(path):
    try:
        r = requests.get(f"{BASE}{path}", timeout=20)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _retention_chart(obs):
    """Build retention curve chart using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curve = obs.get("retention_curve", [])
        scenes = obs.get("scenes", [])
        if not curve:
            return None
        labels = [s.get("id", f"s{i}") for i, s in enumerate(scenes)]
        while len(labels) < len(curve):
            labels.append(f"s{len(labels)}")
        labels = labels[:len(curve)]

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(labels, curve, color="#7c3aed", linewidth=2.5, marker="o", markersize=6)
        ax.fill_between(range(len(curve)), curve, alpha=0.15, color="#7c3aed")
        ax.set_ylim(0, 1.05)
        ax.set_title("📉 Viewer Retention Curve", fontsize=10, pad=6)
        ax.set_xlabel("Scene", fontsize=8)
        ax.set_ylabel("Retention", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.5, color="#ef4444", linestyle="--", linewidth=1, alpha=0.5)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def _engagement_chart(obs):
    """Build per-scene engagement bar chart using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scenes = obs.get("scenes", [])
        if not scenes:
            return None
        labels = [s.get("id", f"s{i}") for i, s in enumerate(scenes)]
        values = [s.get("engagement_score", 0) for s in scenes]
        colors = ["#7c3aed" if v >= 0.5 else "#f59e0b" if v >= 0.3 else "#ef4444" for v in values]

        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(y=0.3, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="Cut threshold")
        ax.set_ylim(0, 1.05)
        ax.set_title("📊 Scene Engagement", fontsize=10, pad=6)
        ax.set_xlabel("Scene", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def _format_obs(obs, meta=None):
    risk = obs.get("risk_score", 0) or 0
    risk_icon = "🟢" if risk < 0.4 else "🟡" if risk < 0.7 else "🔴"
    compliant = obs.get("platform_compliant", False)
    persona = (meta or {}).get("audience_persona", "?")
    return (
        f"📊  CURRENT STATE\n{'─'*35}\n"
        f"Engagement:   {obs.get('current_engagement_score', 0):.3f}\n"
        f"Retention:    {obs.get('avg_retention', 0):.3f}\n"
        f"Hook:         {obs.get('hook_strength', 0):.3f}\n"
        f"Pacing:       {obs.get('pacing_score', 0):.3f}\n"
        f"Duration:     {obs.get('total_duration', 0):.1f}s  {'✅' if compliant else '❌ OVER LIMIT'}\n"
        f"Subtitles:    {'✅' if obs.get('subtitles_present') else '❌'}\n"
        f"Music:        {'✅' if obs.get('music_added') else '❌'}\n"
        f"Hook-first:   {'✅' if obs.get('hook_first') else '❌'}\n"
        f"{risk_icon} Risk score:  {risk:.3f}\n"
        f"Steps left:   {obs.get('steps_remaining', 15)}\n"
        f"Persona:      {persona}"
    )


def _format_scenes(scenes):
    if not scenes:
        return "No scenes"
    lines = ["🎬  SCENE TIMELINE", "─" * 52]
    for i, s in enumerate(scenes):
        eng = s.get("engagement_score", 0)
        hook = s.get("hook_strength", 0)
        stype = s.get("scene_type", "?")
        sid = s.get("id", "?")
        has_hook = s.get("has_hook", False)
        bar = "█" * int(eng * 10) + "░" * (10 - int(eng * 10))
        flag = " ← 🎯 HOOK" if (has_hook and hook > 0.3) else ""
        warn = " ⚠️ CUT THIS" if eng < 0.30 and stype in ("filler","transition") else ""
        lines.append(f"  [{i}] {sid:<10} {stype:<12} [{bar}] {eng:.2f}{flag}{warn}")
    return "\n".join(lines)


def load_scenario(scenario_name):
    cfg = SCENARIOS.get(scenario_name, {"platform": "tiktok", "seed": 21})
    return reset_env(cfg["platform"], cfg["seed"])


def reset_env(platform, seed):
    state = _post("/reset", params={"platform": platform, "seed": int(seed)})
    if not state:
        return ("❌ Reset failed", "", "", "", "",
                None, None, "")

    obs = state.get("observation", {})
    meta = state.get("metadata", {})
    scenes = obs.get("scenes", [])
    risk = obs.get("risk_score", 0) or 0
    risk_label = "🟢 LOW" if risk < 0.4 else "🟡 MEDIUM" if risk < 0.7 else "🔴 HIGH"

    status = (
        f"✅  EPISODE STARTED\n{'─'*35}\n"
        f"Platform:  {obs.get('platform','?').upper()}\n"
        f"Seed:      {seed}\n"
        f"Persona:   {meta.get('audience_persona','?')}\n"
        f"Scenes:    {len(scenes)}\n"
        f"Duration:  {obs.get('total_duration',0):.1f}s\n"
        f"Risk:      {risk_label} ({risk:.3f})\n"
        f"Steps:     {obs.get('steps_remaining',15)} remaining"
    )

    hint = _get("/hint")
    hint_text = ""
    if hint and hint.get("best_action"):
        hint_text = f"💡 Best action: {hint['best_action']}\n   {hint.get('reason','')}"

    ret_chart = _retention_chart(obs)
    eng_chart = _engagement_chart(obs)

    return (status, _format_obs(obs, meta), _format_scenes(scenes),
            hint_text, "", ret_chart, eng_chart, "")


def take_action(action_type, scene_id):
    params = {}
    if action_type == "cut_scene":
        if not scene_id.strip():
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
            else:
                return ("⚠️ No low-engagement filler found. Specify scene_id manually.",
                        "", "", "", "", None, None, "")
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
            return ("⚠️ No hook scene found.", "", "", "", "", None, None, "")

    elif action_type == "trim_duration":
        state = _get("/state")
        obs = state.get("observation", {})
        platform = obs.get("platform", "reels")
        limits = {"reels": 30.0, "shorts": 60.0, "tiktok": 60.0}
        params = {"target_seconds": limits.get(platform, 30.0)}

    resp = _post("/step", json={"action_type": action_type, "parameters": params})
    if not resp:
        return ("❌ Step failed", "", "", "", "", None, None, "")

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

    status = (
        f"{icon}  ACTION: {action_type.upper()}\n{'─'*35}\n"
        f"{reward_icon} Reward:    {reward:+.4f}\n"
        f"Step:      {state.get('step_count','?')} / {state.get('max_steps',15)}\n"
        f"Steps left:{obs.get('steps_remaining','?')}\n"
        f"Done:      {'🏁 YES' if done else 'No'}\n"
        + (f"⚠️  {reason}\n" if reason else "")
        + (f"ℹ️  {info.get('note','')}\n" if info.get('note') else "")
    )

    if done and info.get("persona"):
        status += f"\n🎭 Finalized for {info['persona']} — persona score: {info.get('persona_score',0):.3f}"

    hint = _get("/hint")
    hint_text = ""
    if hint and hint.get("best_action") and not done:
        hint_text = f"💡 Best action: {hint['best_action']}\n   {hint.get('reason','')}"
    elif done:
        hint_text = "🏁 Episode complete — click Grade to see final score"

    traj = _get("/trajectory")
    traj_note = ""
    if traj and traj.get("episode_steps", 0) > 0:
        dqs = traj.get("decision_quality_summary", {})
        traj_note = (
            f"📈 {traj['episode_steps']} steps | "
            f"excellent={dqs.get('excellent',0)} "
            f"good={dqs.get('good',0)} "
            f"poor={dqs.get('poor',0)}"
        )

    ret_chart = _retention_chart(obs)
    eng_chart = _engagement_chart(obs)

    return (status, _format_obs(obs, meta), _format_scenes(scenes),
            hint_text, traj_note, ret_chart, eng_chart, "")


def run_agent():
    """Auto-run the optimal action sequence and return step-by-step log."""
    log_lines = ["🤖 AI AGENT RUNNING...", "─" * 40]
    total_reward = 0.0

    # Get current state
    state = _get("/state")
    if not state:
        return ("❌ Reset first", "", "", "", "", None, None,
                "❌ No active episode")

    obs = state.get("observation", {})
    scenes = obs.get("scenes", [])

    # Build sequence
    sequence = []

    # 1. Reorder if needed
    hooks = [s for s in scenes if s.get("scene_type") in ("hook","highlight") and s.get("has_hook")]
    if hooks and scenes and scenes[0].get("id") != max(hooks, key=lambda s: s.get("hook_strength",0)).get("id"):
        best = max(hooks, key=lambda s: s.get("hook_strength", 0))
        order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
        sequence.append(("reorder_scenes", {"order": order}))

    sequence.append(("boost_hook", {}))

    # Cut fillers
    fillers = sorted(
        [s for s in scenes if s.get("scene_type") in ("filler","transition")
         and s.get("engagement_score", 1) < 0.30],
        key=lambda s: s.get("engagement_score", 1)
    )
    if fillers and len(scenes) > 3:
        sequence.append(("cut_scene", {"scene_id": fillers[0]["id"]}))

    # Trim if needed
    if not obs.get("platform_compliant", True):
        platform = obs.get("platform", "reels")
        limits = {"reels": 30.0, "shorts": 60.0, "tiktok": 60.0}
        sequence.append(("trim_duration", {"target_seconds": limits.get(platform, 30.0)}))

    sequence += [
        ("enhance_pacing", {}),
        ("improve_transition", {}),
        ("smooth_cut", {}),
        ("sync_audio", {}),
        ("add_subtitles", {}),
        ("add_music", {}),
    ]

    last_obs = obs
    last_meta = state.get("metadata", {})
    last_scenes = scenes

    for i, (action, params) in enumerate(sequence):
        resp = _post("/step", json={"action_type": action, "parameters": params})
        if not resp:
            log_lines.append(f"  ❌ Step {i+1}: {action} — failed")
            break

        reward = resp.get("reward", 0.0)
        total_reward += reward
        done = resp.get("done", False)
        info = resp.get("info", {})
        valid = info.get("valid", True)
        last_obs = resp.get("state", {}).get("observation", {})
        last_meta = resp.get("state", {}).get("metadata", {})
        last_scenes = last_obs.get("scenes", [])

        eng = last_obs.get("current_engagement_score", 0)
        risk = last_obs.get("risk_score", 0) or 0
        icon = "✅" if valid else "⚠️"
        log_lines.append(
            f"  {icon} Step {i+1}: {action:<22} r={reward:+.3f}  "
            f"eng={eng:.3f}  risk={risk:.3f}"
        )
        if done:
            break

    # Grade
    g = _get("/grader")
    score = g.get("score", 0) if g else 0
    passed = g.get("passed", False) if g else False
    log_lines += [
        "─" * 40,
        f"{'🏆 PASSED' if passed else '❌ FAILED'}  Score: {score:.4f}",
        f"Total reward: {total_reward:+.4f}",
        f"Explanation: {g.get('explanation','')[:80]}..." if g else "",
    ]

    ret_chart = _retention_chart(last_obs)
    eng_chart = _engagement_chart(last_obs)
    log_text = "\n".join(log_lines)

    return (_format_obs(last_obs, last_meta), _format_scenes(last_scenes),
            "", "", ret_chart, eng_chart, log_text)


def analyze_video_url(url):
    """Call /analyze_url with a real YouTube URL and show the result."""
    if not url.strip():
        return "⚠️ Enter a YouTube or TikTok URL first"
    try:
        r = requests.get(f"{BASE}/analyze_url", params={"url": url.strip()}, timeout=15)
        if r.status_code != 200:
            return f"❌ Error {r.status_code}: {r.text[:200]}"
        d = r.json()
        meta = d.get("real_metadata", {})
        scenes = d.get("generated_scene_breakdown", [])
        summary = d.get("summary", {})
        opp = d.get("optimization_opportunity", {})
        lines = [
            "🌐  REAL VIDEO ANALYSIS (YouTube oEmbed API)",
            "─" * 50,
            f"Title:    {meta.get('title','?')}",
            f"Creator:  {meta.get('author','?')}",
            f"Platform: {meta.get('platform_detected','?').upper()}",
            f"Duration: {summary.get('total_duration','?')}s  "
            + ("✅ Compliant" if summary.get('platform_compliant') else "❌ Over limit"),
            f"Scenes:   {summary.get('total_scenes','?')}  (fillers: {summary.get('filler_scenes',0)})",
            f"Avg eng:  {summary.get('avg_engagement','?')}",
            "",
            "🎬  SCENE BREAKDOWN:",
        ]
        for s in scenes:
            eng = s.get("engagement", 0)
            bar = "█" * int(eng * 10) + "░" * (10 - int(eng * 10))
            warn = " ⚠️ CUT" if s["type"] == "filler" else ""
            lines.append(f"  {s['id']:<10} {s['type']:<12} [{bar}] {eng:.2f}{warn}")
        lines += [
            "",
            "🎯  OPTIMIZATION OPPORTUNITIES:",
            f"  Cut candidate:    {opp.get('cut_candidate','')}",
            f"  Hook needs boost: {opp.get('hook_needs_boost','')}",
            f"  First action:     {opp.get('recommended_first_action','')}",
            "",
            "💡  Use POST /reset to start an episode and optimize this video structure.",
        ]
        if meta.get("oembed_error"):
            lines.append(f"\n⚠️  oEmbed note: {meta['oembed_error']}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Failed: {e}"


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
        cap_info = f"\n⚠️  Score cap: {meta.get('cap_reason','?')} — early decision penalty"

    irrev = ""
    if meta.get("irreversible_decision_point"):
        irrev = f"\n🔒 Finalized at step {meta.get('finalized_at_step','?')} | risk: {meta.get('finalized_risk_score',0):.3f}"

    return (
        f"{'🏆 PASSED' if passed else '❌ FAILED'}\n{'─'*40}\n"
        f"Score:        [{score_bar}] {score:.4f}\n"
        f"Raw score:    {raw:.4f}  (pure quality)\n"
        f"Rubric score: {rubric:.4f}  (RL signal){cap_info}{irrev}\n\n"
        f"📊  BREAKDOWN\n"
        f"  Engagement:    {bd.get('engagement',0):.4f}  (×0.35)\n"
        f"  Retention:     {bd.get('retention',0):.4f}  (×0.15)\n"
        f"  Compliance:    {bd.get('platform_compliance',0):.4f}  (×0.20)\n"
        f"  Subtitles:     {bd.get('subtitles',0):.4f}  (×0.10)\n"
        f"  Hook:          {bd.get('hook_strength',0):.4f}  (×0.10)\n"
        f"  Pacing:        {bd.get('pacing',0):.4f}  (×0.05)\n"
        f"  Efficiency:    {bd.get('efficiency_bonus',0):+.4f}\n"
        f"  Order penalty: {bd.get('order_penalty',0):+.4f}\n\n"
        f"💬  {explanation}"
    )


def get_strategy():
    s = _get("/strategy")
    if not s:
        return "❌ Strategy failed — reset first"

    seq = s.get("recommended_sequence", [])
    seq_text = "\n".join([
        f"  {i+1}. {a['action']:<22} {a['reason'][:55]}"
        for i, a in enumerate(seq[:8])
    ]) or "  (well optimized — ready to finalize)"

    risk = s.get("risk_score", 0)
    risk_icon = "🟢" if risk < 0.4 else "🟡" if risk < 0.7 else "🔴"
    ponr = s.get("point_of_no_recovery", False)

    return (
        f"🧠  STRATEGY ENGINE\n{'─'*40}\n"
        f"Mode:     {s.get('strategy_mode','?').upper()}\n"
        f"{risk_icon} Risk:     {risk:.3f} ({s.get('risk_label','?')})\n"
        f"PONR:     {'🚨 YES — limited recovery' if ponr else 'No'}\n"
        f"Steps:    {s.get('steps_remaining','?')} remaining\n\n"
        f"🎯  IMMEDIATE: {s.get('immediate_action','?') or 'none needed'}\n\n"
        f"📋  SEQUENCE:\n{seq_text}\n\n"
        f"👥  {s.get('persona_focus','')}\n\n"
        f"💭  {s.get('reasoning','')}"
    )


def build_ui():
    with gr.Blocks(title="🎬 AI Video Optimizer — RL Environment") as demo:

        gr.Markdown("""
# 🎬 AI Video Optimizer — Decision-Constrained RL Environment

> Agents make **irreversible editorial decisions** under uncertainty.
> `cut_scene` is permanent. `finalize_edit` ends the episode. `risk_score` tracks proximity to failure.

| 🚀 [Live API](https://saravanabalajisara-ai-video-optimizer-env.hf.space) | 📖 [API Docs](https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs) | 💻 [GitHub](https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV) |
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
            agent_btn = gr.Button("🤖 Run AI Agent", variant="secondary", scale=1)

        gr.Markdown("""
> **🚗 BMW Drift:** 5 scenes, 58s raw footage, hook buried at position 3, filler dragging retention.
> Click **Run AI Agent** to watch the optimal policy solve it automatically.
        """)

        # ── Charts row ─────────────────────────────────────────────────────────
        with gr.Row():
            ret_plot = gr.Plot(label="📉 Viewer Retention Curve (updates after each action)")
            eng_plot = gr.Plot(label="📊 Scene Engagement (red line = cut threshold 0.30)")

        # ── State + scenes ─────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                status_box = gr.Textbox(label="📋 Episode Status", lines=9, interactive=False)
            with gr.Column(scale=1):
                obs_box = gr.Textbox(label="📊 Observation", lines=9, interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                scenes_box = gr.Textbox(label="🎬 Scene Timeline", lines=8, interactive=False)
            with gr.Column(scale=1):
                hint_box = gr.Textbox(label="💡 Hint", lines=3, interactive=False)
                traj_box = gr.Textbox(label="📈 Trajectory", lines=3, interactive=False)

        # ── Manual action ──────────────────────────────────────────────────────
        with gr.Accordion("▶️ Manual Action", open=True):
            with gr.Row():
                action_dd = gr.Dropdown(
                    choices=[
                        "reorder_scenes", "boost_hook", "cut_scene", "trim_duration",
                        "enhance_pacing", "improve_transition", "smooth_cut",
                        "sync_audio", "add_subtitles", "add_music", "finalize_edit"
                    ],
                    value="reorder_scenes",
                    label="Action (optimal order)",
                    scale=2,
                )
                scene_id_box = gr.Textbox(
                    label="scene_id (cut_scene only — blank = auto-pick worst filler)",
                    placeholder="e.g. scene_1",
                    scale=2,
                )
                step_btn = gr.Button("▶ Execute", variant="primary", scale=1)

        # ── Manual reset ───────────────────────────────────────────────────────
        with gr.Accordion("⚙️ Manual Reset", open=False):
            with gr.Row():
                platform_dd = gr.Dropdown(["reels","shorts","tiktok"], value="tiktok", label="Platform", scale=1)
                seed_num = gr.Number(value=21, label="Seed", precision=0, scale=1)
                reset_btn = gr.Button("🔄 Reset", scale=1)

        # ── Agent log ──────────────────────────────────────────────────────────
        agent_log = gr.Textbox(label="🤖 AI Agent Log", lines=14, interactive=False, visible=True)

        # ── Grade + Strategy ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                grade_btn = gr.Button("📊 Grade Current State", variant="secondary")
                grade_box = gr.Textbox(label="🏆 Grader Result", lines=20, interactive=False)
            with gr.Column():
                strategy_btn = gr.Button("🧠 Get Strategy Plan", variant="secondary")
                strategy_box = gr.Textbox(label="🧠 Strategy Engine", lines=20, interactive=False)

        # ── Real Video URL Analyzer ────────────────────────────────────────────
        gr.Markdown("### 🌐 Real Video Analyzer — Paste any YouTube URL")
        gr.Markdown(
            "> Fetches real video metadata via **YouTube oEmbed API** (no API key needed) "
            "and generates a scene breakdown for RL optimization."
        )
        with gr.Row():
            url_input = gr.Textbox(
                label="YouTube / TikTok URL",
                placeholder="https://www.youtube.com/watch?v=...",
                scale=4,
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)
        analyze_box = gr.Textbox(label="📋 Real Video Analysis", lines=20, interactive=False)

        # ── Wire up ────────────────────────────────────────────────────────────
        load_outputs = [status_box, obs_box, scenes_box, hint_box, traj_box,
                        ret_plot, eng_plot, agent_log]
        step_outputs = [status_box, obs_box, scenes_box, hint_box, traj_box,
                        ret_plot, eng_plot, agent_log]
        agent_outputs = [obs_box, scenes_box, hint_box, traj_box,
                         ret_plot, eng_plot, agent_log]

        load_btn.click(load_scenario, inputs=[scenario_dd], outputs=load_outputs)
        reset_btn.click(reset_env, inputs=[platform_dd, seed_num], outputs=load_outputs)
        step_btn.click(take_action, inputs=[action_dd, scene_id_box], outputs=step_outputs)
        grade_btn.click(grade_env, outputs=[grade_box])
        strategy_btn.click(get_strategy, outputs=[strategy_box])
        agent_btn.click(run_agent, outputs=agent_outputs)
        analyze_btn.click(analyze_video_url, inputs=[url_input], outputs=[analyze_box])

        # ── Auto-load BMW drift on page open ───────────────────────────────────
        demo.load(
            lambda: load_scenario("🚗 BMW Drift — TikTok Short (seed 21)"),
            outputs=load_outputs
        )

    return demo
