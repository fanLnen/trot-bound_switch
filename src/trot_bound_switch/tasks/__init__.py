"""Gym registration for the package-owned transition tasks."""

import gymnasium as gym


TASK_ID = "SA-16DOF-FuXian-Trot-Bound-Transition-v0"
PLAY_TASK_ID = "SA-16DOF-FuXian-Trot-Bound-Transition-Play-v0"
BOUND_EXPERT_TASK_ID = "SA-16DOF-FuXian-Bound-Expert-v0"
TROT_EXPERT_TASK_ID = "SA-16DOF-FuXian-Trot-Expert-v0"
BOUND_COLLECT_TASK_ID = "SA-16DOF-FuXian-Bound-Collect-v0"
TROT_COLLECT_TASK_ID = "SA-16DOF-FuXian-Trot-Collect-v0"


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


def _register_expert(task_id: str, env_cfg: str, runner_cfg: str) -> None:
    if task_id in gym.registry:
        del gym.registry[task_id]
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register_expert(
    BOUND_EXPERT_TASK_ID,
    "trot_bound_switch.tasks.expert_env_cfg:BoundExpertEnvCfg",
    "trot_bound_switch.tasks.agents.rsl_rl_ppo_cfg:BoundExpertPPORunnerCfg",
)
_register_expert(
    TROT_EXPERT_TASK_ID,
    "trot_bound_switch.tasks.expert_env_cfg:TrotExpertEnvCfg",
    "trot_bound_switch.tasks.agents.rsl_rl_ppo_cfg:TrotExpertPPORunnerCfg",
)
_register_expert(
    BOUND_COLLECT_TASK_ID,
    "trot_bound_switch.tasks.expert_env_cfg:BoundCollectEnvCfg",
    "trot_bound_switch.tasks.agents.rsl_rl_ppo_cfg:BoundExpertPPORunnerCfg",
)
_register_expert(
    TROT_COLLECT_TASK_ID,
    "trot_bound_switch.tasks.expert_env_cfg:TrotCollectEnvCfg",
    "trot_bound_switch.tasks.agents.rsl_rl_ppo_cfg:TrotExpertPPORunnerCfg",
)

__all__ = [
    "BOUND_COLLECT_TASK_ID",
    "BOUND_EXPERT_TASK_ID",
    "PLAY_TASK_ID",
    "TASK_ID",
    "TROT_COLLECT_TASK_ID",
    "TROT_EXPERT_TASK_ID",
]
