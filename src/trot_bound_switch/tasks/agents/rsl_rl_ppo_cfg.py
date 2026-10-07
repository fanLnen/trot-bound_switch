"""RSL-RL PPO configuration used by the successful formal run."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mdp.expert_symmetry import mirror_symmetry_data_augmentation


@configclass
class ExpertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared PPO configuration for the original 16D joint-action experts."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1351
    save_interval = 50
    experiment_name = "SA_16dof_FuXian_Expert"
    run_name = ""
    resume = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mirror_symmetry_data_augmentation,
        ),
    )


@configclass
class BoundExpertPPORunnerCfg(ExpertPPORunnerCfg):
    experiment_name = "SA_16dof_FuXian_Bound_Expert"
    max_iterations = 1351


@configclass
class TrotExpertPPORunnerCfg(ExpertPPORunnerCfg):
    experiment_name = "SA_16dof_FuXian_Trot_Expert"
    # The public code-only recipe trains the final turning task directly from random initialization.
    max_iterations = 2101


@configclass
class TransitionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 1351
    save_interval = 100
    experiment_name = "SA_16dof_FuXian_Trot_Bound_Transition"
    run_name = ""
    resume = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    )
