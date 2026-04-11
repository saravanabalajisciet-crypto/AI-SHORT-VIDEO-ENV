"""
client.py — VideoOptimizationEnv client library.

Provides a clean Python interface to the AI Short-Form Video Optimization
environment, mirroring the OpenEnv standard interface.

Quick Start:
    from client import VideoOptimizationEnv, VideoAction

    # From Docker image (auto-starts container)
    with VideoOptimizationEnv.from_docker_image("video-env:latest") as env:
        result = env.reset(platform="reels", seed=42)
        print(result.observation["current_engagement_score"])

        result = env.step(VideoAction(action_type="boost_hook"))
        print(result.reward)

    # From running server
    env = VideoOptimizationEnv(base_url="http://localhost:7860")
    result = env.reset(platform="reels", seed=42)
"""

from __future__ import annotations

import subprocess
import time
import socket
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VideoAction:
    """Action to apply to the environment."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action_type": self.action_type, "parameters": self.parameters}


@dataclass
class StepResult:
    """Result returned by reset() and step()."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    state: Dict[str, Any]

    @property
    def engagement(self) -> float:
        return self.observation.get("current_engagement_score", 0.0)

    @property
    def retention(self) -> float:
        return self.observation.get("avg_retention", 0.0)

    @property
    def hook_strength(self) -> float:
        return self.observation.get("hook_strength", 0.0)

    @property
    def steps_remaining(self) -> int:
        return self.observation.get("steps_remaining", 0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class VideoOptimizationEnv:
    """
    Python client for the AI Short-Form Video Optimization environment.

    Supports two connection modes:
    1. from_docker_image() — auto-starts a Docker container
    2. Direct instantiation — connects to a running server

    Example:
        # Auto-start Docker
        env = VideoOptimizationEnv.from_docker_image("video-env:latest")

        # Connect to HF Space
        env = VideoOptimizationEnv("https://saravanabalajisara-ai-video-optimizer-env.hf.space")

        # Connect to local server
        env = VideoOptimizationEnv("http://localhost:7860")
    """

    DEFAULT_HF_URL = "https://saravanabalajisara-ai-video-optimizer-env.hf.space"

    def __init__(self, base_url: str = DEFAULT_HF_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._container_id: Optional[str] = None
        self._owns_container = False

    # ── factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_docker_image(
        cls,
        image: str = "video-env:latest",
        port: int = 7860,
        timeout: int = 60,
    ) -> "VideoOptimizationEnv":
        """
        Start a Docker container and return a connected client.

        Args:
            image:   Docker image name (default: video-env:latest)
            port:    Host port to bind (default: 7860)
            timeout: Seconds to wait for server ready (default: 60)

        Example:
            env = VideoOptimizationEnv.from_docker_image("video-env:latest")
        """
        # Find a free port if default is taken
        host_port = port
        with socket.socket() as s:
            if s.connect_ex(("localhost", port)) == 0:
                host_port = port + 1

        result = subprocess.run(
            ["docker", "run", "-d", "-p", f"{host_port}:7860", image],
            capture_output=True, text=True, check=True,
        )
        container_id = result.stdout.strip()

        env = cls(base_url=f"http://localhost:{host_port}", timeout=timeout)
        env._container_id = container_id
        env._owns_container = True

        # Wait for server to be ready
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{env.base_url}/health", timeout=3)
                if r.status_code == 200:
                    return env
            except Exception:
                pass
            time.sleep(1)

        env.close()
        raise TimeoutError(f"Server did not start within {timeout}s")

    @classmethod
    def from_hf_space(cls, timeout: int = 30) -> "VideoOptimizationEnv":
        """Connect to the live HF Space deployment."""
        return cls(base_url=cls.DEFAULT_HF_URL, timeout=timeout)

    # ── core API ───────────────────────────────────────────────────────────────

    def reset(
        self,
        platform: str = "reels",
        seed: int = 42,
    ) -> StepResult:
        """
        Start a new episode.

        Args:
            platform: reels | shorts | tiktok
            seed:     RNG seed for reproducibility

        Returns:
            StepResult with initial observation
        """
        r = requests.post(
            f"{self.base_url}/reset",
            params={"platform": platform, "seed": seed},
            timeout=self.timeout,
        )
        r.raise_for_status()
        state = r.json()
        return StepResult(
            observation=state.get("observation", {}),
            reward=0.0,
            done=state.get("done", False),
            info={"reset": True, "seed": seed, "platform": platform},
            state=state,
        )

    def step(self, action: VideoAction) -> StepResult:
        """
        Apply one action to the environment.

        Args:
            action: VideoAction with action_type and optional parameters

        Returns:
            StepResult with new observation, reward, done, info
        """
        r = requests.post(
            f"{self.base_url}/step",
            json=action.to_dict(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        resp = r.json()
        state = resp.get("state", {})
        return StepResult(
            observation=state.get("observation", {}),
            reward=resp.get("reward", 0.0),
            done=resp.get("done", False),
            info=resp.get("info", {}),
            state=state,
        )

    def state(self) -> StepResult:
        """Read current state without advancing the episode."""
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        s = r.json()
        return StepResult(
            observation=s.get("observation", {}),
            reward=0.0,
            done=s.get("done", False),
            info={},
            state=s,
        )

    def grade(self) -> Dict[str, Any]:
        """Score the current state. Returns score, breakdown, passed."""
        r = requests.get(f"{self.base_url}/grader", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def hint(self) -> Dict[str, Any]:
        """Get the best next action suggestion."""
        r = requests.get(f"{self.base_url}/hint", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def feedback(self) -> Dict[str, Any]:
        """Get AI coaching tips for the current state."""
        r = requests.get(f"{self.base_url}/feedback", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def tasks(self) -> List[Dict[str, Any]]:
        """Return all task definitions."""
        r = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def persona(self) -> Dict[str, Any]:
        """Return current episode audience persona and scoring weights."""
        r = requests.get(f"{self.base_url}/persona", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def trajectory(self) -> Dict[str, Any]:
        """Return full action trajectory of current episode."""
        r = requests.get(f"{self.base_url}/trajectory", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def leaderboard(self) -> Dict[str, Any]:
        """Return top scores across all graded episodes."""
        r = requests.get(f"{self.base_url}/leaderboard", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        """Check if the server is healthy."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "VideoOptimizationEnv":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Stop the Docker container if we started it."""
        if self._owns_container and self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True, timeout=10,
                )
                subprocess.run(
                    ["docker", "rm", self._container_id],
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass
            self._container_id = None

    def __repr__(self) -> str:
        return f"VideoOptimizationEnv(base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# CLI — python client.py
# ---------------------------------------------------------------------------

def _run_cli(host: str, platform: str, seed: int) -> None:
    env = VideoOptimizationEnv(base_url=host)
    W = 68

    print(f"\n{'='*W}")
    print(f"  AI Short-Form Video Optimization  v6.0")
    print(f"  {host}  platform={platform}  seed={seed}")
    print(f"{'='*W}")

    result = env.reset(platform=platform, seed=seed)
    obs = result.observation
    print(f"  Initial  eng={obs.get('current_engagement_score',0):.3f}  "
          f"ret={obs.get('avg_retention',0):.3f}  "
          f"dur={obs.get('total_duration',0):.1f}s  "
          f"scenes={len(obs.get('scenes',[]))}  "
          f"persona={result.state.get('metadata',{}).get('audience_persona','?')}")
    print(f"{'─'*W}")

    actions = [
        VideoAction("boost_hook"),
        VideoAction("enhance_pacing"),
        VideoAction("improve_transition"),
        VideoAction("smooth_cut"),
        VideoAction("sync_audio"),
        VideoAction("add_subtitles"),
        VideoAction("add_music"),
    ]

    for action in actions:
        if result.done:
            break
        result = env.step(action)
        o = result.observation
        tag = "OK" if result.info.get("valid", True) else "!!"
        print(f"  {tag} | {action.action_type:<20s} | "
              f"r={result.reward:+.3f} | "
              f"eng={o.get('current_engagement_score',0):.3f} | "
              f"ret={o.get('avg_retention',0):.3f} | "
              f"steps_left={o.get('steps_remaining',0)}")

    grader = env.grade()
    print(f"\n{'─'*W}")
    print(f"  SCORE: {grader['score']:.4f}  {'PASSED' if grader['passed'] else 'FAILED'}")
    for k, v in grader.get("breakdown", {}).items():
        if k != "final_score":
            print(f"    {k:<25s}: {float(v):.4f}")

    fb = env.feedback()
    print(f"\n  {fb.get('overall','')}")
    for tip in fb.get("tips", [])[:3]:
        print(f"  - {tip}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Optimizer client")
    parser.add_argument("--host",     default="http://localhost:7860")
    parser.add_argument("--platform", default="reels", choices=["reels", "shorts", "tiktok"])
    parser.add_argument("--seed",     default=42, type=int)
    args = parser.parse_args()
    _run_cli(args.host, args.platform, args.seed)
