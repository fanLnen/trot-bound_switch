"""Gym registration for the package-owned transition tasks."""

import gymnasium as gym


TASK_ID = "SA-16DOF-FuXian-Trot-Bound-Transition-v0"
PLAY_TASK_ID = "SA-16DOF-FuXian-Trot-Bound-Transition-Play-v0"


def _register(task_id: str, env_cfg: str) -> None:
    if task_id in gym.registry:
        del gym.registry[task_id]
    gym.register(
        id=task_id,
        entry_point="trot_bound_switch.tasks.transition_env_cfg:TransitionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                "trot_bound_switch.tasks.agents.rsl_rl_ppo_cfg:TransitionPPORunnerCfg"
            ),
        },
    )


_register(TASK_ID, "trot_bound_switch.tasks.transition_env_cfg:TransitionEnvCfg")
_register(PLAY_TASK_ID, "trot_bound_switch.tasks.transition_env_cfg:TransitionPlayEnvCfg")

__all__ = ["PLAY_TASK_ID", "TASK_ID"]
