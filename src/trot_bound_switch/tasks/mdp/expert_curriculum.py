import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def liangan_velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_levels: list[tuple[float, float]] = [
        (0.1, 1.0),
        (0.1, 1.5),
        (0.1, 2.0),
        (0.1, 2.5),
        (0.1, 3.0),
        (0.1, 3.5),
        (0.1, 4.0),
        (0.1, 4.5),
        (0.1, 5.0),
        (0.1, 6.0),
        (0.1, 7.0),
    ],
    error_threshold_ratio: float = 0.15,
    timeout_threshold: float = 0.96,
    min_iterations: int = 200,
    stats_window_iterations: int = 100,
    high_speed_ratio: float = 0.8,
    high_error_threshold_ratio: float = 0.20,
    min_high_samples: int = 2048,
    high_sample_prob: float = 0.30,
    print_interval_iterations: int = 50,
    num_steps_per_env: int = 24,
):
    """Velocity curriculum with recent-window statistics.

    Promotion condition:
    1. Current level must last at least min_iterations PPO iterations.
    2. Full-range avg_error uses all envs' command metric in recent stats_window_iterations.
    3. High-speed avg_high_error uses all envs where cmd_vx > high_speed_ratio * current_max_speed.
    4. Timeout rate still uses only reset env_ids.
    5. Reset env_ids are resampled with biased high-speed command distribution.
    """

    if len(env_ids) == 0:
        return

    # ------------------------------------------------------------
    # Compute approximate PPO iteration
    # ------------------------------------------------------------
    common_step_counter = getattr(env, "common_step_counter", 0)
    current_iter = int(common_step_counter // max(num_steps_per_env, 1))

    # ------------------------------------------------------------
    # Initialize curriculum states once
    # ------------------------------------------------------------
    if not hasattr(env, "_liangan_vel_level"):
        env._liangan_vel_level = 0
        env._liangan_last_print_iter = -1
        env._liangan_level_start_iter = current_iter

        # total reset episodes since current level start, only for display
        env._liangan_vel_episode_count_total = 0

        # sliding window over recent PPO iterations
        env._liangan_vel_iter_window = []

        command_term = env.command_manager.get_term(command_name)
        command_term.cfg.ranges.lin_vel_x = speed_levels[0]

        print(
            f"[Liangan Velocity Curriculum] "
            f"Start level 0/{len(speed_levels) - 1}: "
            f"lin_vel_x={speed_levels[0]}, "
            f"level_start_iter={env._liangan_level_start_iter}, "
            f"stats_window_iterations={stats_window_iterations}"
        )

    level = env._liangan_vel_level
    current_range = speed_levels[level]

    range_low = min(current_range[0], current_range[1])
    range_high = max(current_range[0], current_range[1])
    current_max_speed = max(abs(current_range[0]), abs(current_range[1]))

    # 保留你之前的误差阈值尺度写法。
    # 例如 level 0: error_limit = 0.15 * 1.5 = 0.225
    current_mean_speed = max(abs(current_range[0]), 1.5)

    iterations_at_level = current_iter - env._liangan_level_start_iter
    iterations_ok = iterations_at_level >= min_iterations

    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.lin_vel_x = current_range

    command = env.command_manager.get_command(command_name)
    env_ids_device = env_ids.to(device=command.device, dtype=torch.long)

    # ------------------------------------------------------------
    # Use current command BEFORE resampling for statistics
    # ------------------------------------------------------------
    command_for_stats = command.clone()

    # ------------------------------------------------------------
    # Full-range tracking error: use all envs' command metric
    # This should be much closer to Metrics/base_velocity/error_vel_xy.
    # ------------------------------------------------------------
    metric_error = None
    if hasattr(command_term, "metrics") and "error_vel_xy" in command_term.metrics:
        metric_error = command_term.metrics["error_vel_xy"]

    if torch.is_tensor(metric_error):
        error_xy_all = metric_error

        if error_xy_all.ndim > 1:
            error_xy_all = torch.norm(error_xy_all, dim=-1)

        # If metric is not per-env, fall back to direct per-env computation.
        if error_xy_all.ndim == 0 or error_xy_all.numel() != command_for_stats.shape[0]:
            metric_error = None

    if not torch.is_tensor(metric_error) or error_xy_all.ndim == 0 or error_xy_all.numel() != command_for_stats.shape[0]:
        asset: Articulation = env.scene[asset_cfg.name]
        base_vel_xy = asset.data.root_lin_vel_b[:, :2]
        error_xy_all = torch.norm(command_for_stats[:, :2] - base_vel_xy, dim=1)

    error_xy_all = error_xy_all.reshape(-1)

    # All-env full-range error for current iteration snapshot
    current_error_count = int(error_xy_all.numel())
    current_error_sum = torch.sum(error_xy_all).item()
    current_mean_error_all = current_error_sum / max(current_error_count, 1)

    # ------------------------------------------------------------
    # High-speed tracking error: all envs with cmd_vx > 0.8 * max_speed
    # ------------------------------------------------------------
    high_speed_threshold = high_speed_ratio * current_max_speed
    high_mask_all = command_for_stats[:, 0] > high_speed_threshold

    current_high_count = 0
    current_high_error = float("inf")
    current_high_error_sum = 0.0

    if torch.any(high_mask_all):
        high_errors = error_xy_all[high_mask_all]
        current_high_count = int(high_errors.numel())
        current_high_error_sum = torch.sum(high_errors).item()
        current_high_error = current_high_error_sum / max(current_high_count, 1)

    # ------------------------------------------------------------
    # Timeout rate: still only reset env_ids
    # ------------------------------------------------------------
    num_reset_eps = int(len(env_ids_device))

    if hasattr(env, "reset_time_outs"):
        timeout_rate = torch.mean(env.reset_time_outs[env_ids].float()).item()

    elif hasattr(env, "termination_manager") and hasattr(env.termination_manager, "time_outs"):
        timeout_rate = torch.mean(env.termination_manager.time_outs[env_ids].float()).item()

    else:
        timeout_rate = torch.mean(
            (env.episode_length_buf[env_ids] >= env.max_episode_length - 1).float()
        ).item()

    timeout_count = num_reset_eps
    timeout_sum = timeout_rate * timeout_count

    env._liangan_vel_episode_count_total += num_reset_eps

    # ------------------------------------------------------------
    # Add current statistics into iteration sliding window
    # Important:
    # - Full-range error and high-speed error are all-env snapshots.
    # - They should be recorded only once per PPO iteration.
    # - Timeout is reset-env based and can be accumulated if multiple resets occur in one iteration.
    # ------------------------------------------------------------
    window = env._liangan_vel_iter_window

    if len(window) > 0 and window[-1]["iter"] == current_iter:
        bucket = window[-1]

        # Do NOT add all-env error twice in the same PPO iteration.
        # Only accumulate timeout from additional reset calls.
        bucket["timeout_count"] += timeout_count
        bucket["timeout_sum"] += timeout_sum

        # For display only
        bucket["reset_eps"] += num_reset_eps

    else:
        window.append(
            {
                "iter": current_iter,

                # all-env full-range error snapshot
                "err_count": current_error_count,
                "err_sum": current_error_sum,

                # all-env high-speed error snapshot
                "high_count": current_high_count,
                "high_err_sum": current_high_error_sum,

                # reset-env timeout statistics
                "timeout_count": timeout_count,
                "timeout_sum": timeout_sum,

                # display only
                "reset_eps": num_reset_eps,
            }
        )

    # Keep only recent stats_window_iterations PPO iterations.
    oldest_iter_to_keep = current_iter - stats_window_iterations + 1
    while len(window) > 0 and window[0]["iter"] < oldest_iter_to_keep:
        window.pop(0)

    # ------------------------------------------------------------
    # Compute recent-window statistics
    # ------------------------------------------------------------
    window_error_count = sum(item["err_count"] for item in window)
    window_error_sum = sum(item["err_sum"] for item in window)

    window_high_count = sum(item["high_count"] for item in window)
    window_high_error_sum = sum(item["high_err_sum"] for item in window)

    window_timeout_count = sum(item["timeout_count"] for item in window)
    window_timeout_sum = sum(item["timeout_sum"] for item in window)

    window_reset_eps = sum(item["reset_eps"] for item in window)
    window_iter_count = len(window)

    if window_error_count > 0:
        avg_error = window_error_sum / window_error_count
    else:
        avg_error = float("inf")

    if window_timeout_count > 0:
        avg_timeout = window_timeout_sum / window_timeout_count
    else:
        avg_timeout = 0.0

    if window_high_count > 0:
        avg_high_error = window_high_error_sum / window_high_count
    else:
        avg_high_error = float("inf")

    error_limit = error_threshold_ratio * current_mean_speed
    high_error_limit = high_error_threshold_ratio * current_max_speed

    error_ok = avg_error < error_limit
    timeout_ok = avg_timeout >= timeout_threshold
    high_count_ok = window_high_count >= min_high_samples
    high_error_ok = avg_high_error < high_error_limit

    # ------------------------------------------------------------
    # Print current level and promotion condition comparison
    # ------------------------------------------------------------
    if (
        current_iter > 0
        and current_iter % print_interval_iterations == 0
        and current_iter != env._liangan_last_print_iter
    ):
        print(
            f"[Liangan Velocity Curriculum] Iter {current_iter}: "
            f"level={level}/{len(speed_levels) - 1}, "
            f"lin_vel_x={current_range}, "
            f"iters={iterations_at_level}/{min_iterations}, "
            f"iter_ok={iterations_ok}, "
            f"window_iters={window_iter_count}/{stats_window_iterations}, "
            f"window_error_envs={window_error_count}, "
            f"window_reset_eps={window_reset_eps}, "
            f"total_reset_eps_at_level={env._liangan_vel_episode_count_total}, "
            f"avg_error_recent_all_env={avg_error:.3f} < error_limit={error_limit:.3f} -> {error_ok}, "
            f"avg_timeout_recent_reset_envs={avg_timeout:.3f} >= timeout_limit={timeout_threshold:.3f} -> {timeout_ok}, "
            f"high_threshold={high_speed_threshold:.3f}, "
            f"high_count_recent_all_env={window_high_count}/{min_high_samples} -> {high_count_ok}, "
            f"avg_high_error_recent_all_env={avg_high_error:.3f} < high_limit={high_error_limit:.3f} -> {high_error_ok}, "
            f"current_mean_error_all_env={current_mean_error_all:.3f}, "
            f"current_high_envs={current_high_count}, "
            f"current_high_error_all_env={current_high_error:.3f}, "
            f"current_reset_eps={num_reset_eps}, "
            f"current_timeout_rate={timeout_rate:.3f}, "
            f"high_sample_prob={high_sample_prob:.2f}"
        )
        env._liangan_last_print_iter = current_iter

    # ------------------------------------------------------------
    # Iteration-based cooldown + promotion check
    # ------------------------------------------------------------
    promoted = False

    if (
        iterations_ok
        and error_ok
        and timeout_ok
        and high_count_ok
        and high_error_ok
        and level < len(speed_levels) - 1
    ):
        env._liangan_vel_level += 1
        promoted = True

        next_range = speed_levels[env._liangan_vel_level]
        command_term.cfg.ranges.lin_vel_x = next_range

        print(
            f"[Liangan Velocity Curriculum] Promote to level "
            f"{env._liangan_vel_level}/{len(speed_levels) - 1}: "
            f"lin_vel_x={next_range}, "
            f"avg_error_recent_all_env={avg_error:.3f} < error_limit={error_limit:.3f}, "
            f"avg_timeout_recent_reset_envs={avg_timeout:.3f} >= timeout_limit={timeout_threshold:.3f}, "
            f"avg_high_error_recent_all_env={avg_high_error:.3f} < high_limit={high_error_limit:.3f}, "
            f"high_count_recent_all_env={window_high_count}, "
            f"window_iters={window_iter_count}, "
            f"iterations_at_level={iterations_at_level}/{min_iterations}"
        )

        # Reset level timer and recent-window statistics for the new level.
        env._liangan_level_start_iter = current_iter
        env._liangan_vel_episode_count_total = 0
        env._liangan_vel_iter_window = []

    # ------------------------------------------------------------
    # Biased high-speed sampling for reset envs only
    # Do this AFTER statistics and possible promotion.
    # This avoids using freshly resampled commands to evaluate the just-finished episode.
    # ------------------------------------------------------------
    active_level = env._liangan_vel_level
    active_range = speed_levels[active_level]

    active_low = min(active_range[0], active_range[1])
    active_high = max(active_range[0], active_range[1])
    active_max_speed = max(abs(active_range[0]), abs(active_range[1]))

    high_sample_low = None
    high_resample_count = 0

    if active_high > active_low and high_sample_prob > 0.0 and len(env_ids_device) > 0:
        if active_level > 0:
            prev_range = speed_levels[active_level - 1]
            prev_max_speed = max(abs(prev_range[0]), abs(prev_range[1]))
            high_sample_low = max(active_low, min(prev_max_speed, active_high))
        else:
            high_sample_low = max(active_low, high_speed_ratio * active_max_speed)

        if high_sample_low >= active_high:
            high_sample_low = max(active_low, high_speed_ratio * active_max_speed)

        n = int(len(env_ids_device))

        # 70% full-range + 30% high-added range by default.
        samples = active_low + (active_high - active_low) * torch.rand(
            n, device=command.device
        )

        use_high = torch.rand(n, device=command.device) < high_sample_prob

        if torch.any(use_high) and high_sample_low < active_high:
            high_n = int(use_high.sum().item())
            samples[use_high] = high_sample_low + (active_high - high_sample_low) * torch.rand(
                high_n, device=command.device
            )
            high_resample_count = high_n

        command[env_ids_device, 0] = samples

    # Optional low-frequency print for sampling after promotion.
    if promoted:
        print(
            f"[Liangan Velocity Curriculum] After promotion command sampling: "
            f"active_level={active_level}, "
            f"active_range={active_range}, "
            f"reset_eps={num_reset_eps}, "
            f"high_sample_low={high_sample_low}, "
            f"high_sample_prob={high_sample_prob:.2f}, "
            f"high_resample_count={high_resample_count}"
        )

    # ------------------------------------------------------------
    # Stay message, but do not clear the sliding window.
    # The window already keeps only recent stats_window_iterations.
    # ------------------------------------------------------------
    if (
        not promoted
        and iterations_at_level > 4 * min_iterations
        and current_iter % print_interval_iterations == 0
    ):
        print(
            f"[Liangan Velocity Curriculum] Stay at level "
            f"{level}/{len(speed_levels) - 1}: "
            f"lin_vel_x={current_range}, "
            f"avg_error_recent_all_env={avg_error:.3f} < error_limit={error_limit:.3f} -> {error_ok}, "
            f"avg_timeout_recent_reset_envs={avg_timeout:.3f} >= timeout_limit={timeout_threshold:.3f} -> {timeout_ok}, "
            f"avg_high_error_recent_all_env={avg_high_error:.3f} < high_limit={high_error_limit:.3f} -> {high_error_ok}, "
            f"high_count_recent_all_env={window_high_count}/{min_high_samples} -> {high_count_ok}, "
            f"window_iters={window_iter_count}/{stats_window_iterations}, "
            f"iterations_at_level={iterations_at_level}"
        )