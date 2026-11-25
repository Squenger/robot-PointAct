import os
import sys
import time
import collections
import dataclasses
import datetime as dt
import json
import logging
import pathlib
from pathlib import Path

import imageio
import numpy as np
import tqdm
import tyro

from pointact.utils.torch_utils import set_seed
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient

from pointact.robot_envs.libero_utils.environments import LiberoEnv
from pointact.robot_envs.libero_utils.robot_joints import get_robot_joint_boxes
from pointact.robot_envs.libero_utils.perturbations import PerturbationManager
from pointact.robot_envs.libero_utils.adaptive_horizon import (
    AdaptiveHorizonMonitor,
    BacktrackController,
)


@dataclasses.dataclass
class ClientArgs:
    seed: int = 7
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    image_size: int = 256

    repo_id: str | None = None
    post_process_action: bool = True
    replan_steps: int = 8
    pred_rot_type: str = "rot6d"  # euler, rot6d
    use_depth: bool = False
    delta_action: bool = False
    remove_arm: bool = False

    save_dir: str = "results/libero_adaptive"
    save_video: bool = False
    verbose: bool = False

    host: str = "localhost"
    port: int = 5555

    # Perturbation parameters
    perturb_type: str = "none"  # none, object_shift, action_noise, combined
    shift_mag: float = 0.05
    action_noise_std: float = 0.30
    action_noise_duration: int = 6

    # Adaptive horizon parameters
    adaptive_mode: str = "fixed"  # fixed, adaptive
    depth_pixel_thresh: float = 0.025
    depth_ratio_thresh: float = 0.010
    depth_spike_factor: float = 2.5
    cos_sim_thresh: float = -0.10
    cooldown_steps: int = 3

    # Backtrack controller parameters (empty grasp detection + controlled recovery)
    enable_backtrack: bool = False
    empty_grasp_width: float = 0.008
    close_confirm_steps: int = 8
    retract_steps: int = 10
    max_backtracks: int = 2

    # Optional comma-separated subset of task IDs (e.g. "0,1,2,3")
    task_ids: str = ""


def setup_logging(filename=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def main(args: ClientArgs) -> None:
    assert args.pred_rot_type in ["rot6d", "euler"]

    policy_client = PolicyClient(args.host, args.port)

    is_server_running = False
    while not is_server_running:
        is_server_running = policy_client.ping()
    print(f"Server is running on host {args.host} port {args.port}")

    video_out_dir = None
    log_filename = None
    summary_filename = None
    if args.save_dir:
        bt_tag = (
            f"_bt_w{args.empty_grasp_width:g}_c{args.close_confirm_steps}_r{args.retract_steps}_m{args.max_backtracks}"
            if args.enable_backtrack
            else ""
        )
        tag = f"{args.task_suite_name}_{args.perturb_type}_{args.adaptive_mode}_k{args.replan_steps}{bt_tag}"
        date_base = Path(
            args.save_dir,
            f"{tag}-{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        )
        date_base.parent.mkdir(parents=True, exist_ok=True)
        log_filename = f"{date_base}.log"
        summary_filename = f"{date_base}_summary.json"
        print(f"Save logs in {log_filename}")
        if args.save_video:
            video_out_dir = f"{date_base}+videos"
            pathlib.Path(video_out_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(log_filename)
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    set_seed(args.seed)

    libero_env = LiberoEnv(
        task_suite_name=args.task_suite_name,
        seed=args.seed,
        use_depth=args.use_depth,
        use_point_cloud=args.use_depth,
        delta_action=args.delta_action,
        num_steps_wait=args.num_steps_wait,
        image_resolution=args.image_size,
    )

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520 + 50
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    perturber = PerturbationManager(
        perturb_type=args.perturb_type,
        shift_mag=args.shift_mag,
        action_noise_std=args.action_noise_std,
        action_noise_duration=args.action_noise_duration,
    )
    monitor = AdaptiveHorizonMonitor(
        mode=args.adaptive_mode,
        delta_action=args.delta_action,
        depth_pixel_thresh=args.depth_pixel_thresh,
        depth_ratio_thresh=args.depth_ratio_thresh,
        depth_spike_factor=args.depth_spike_factor,
        cos_sim_thresh=args.cos_sim_thresh,
        cooldown_steps=args.cooldown_steps,
    )
    backtracker = BacktrackController(
        enabled=args.enable_backtrack,
        empty_grasp_width=args.empty_grasp_width,
        close_confirm_steps=args.close_confirm_steps,
        retract_steps=args.retract_steps,
        max_backtracks=args.max_backtracks,
    )

    if args.task_ids.strip():
        selected_task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    else:
        selected_task_ids = list(range(libero_env.num_tasks_in_suite))

    total_episodes = 0
    total_successes = 0
    total_server_calls = 0
    total_early_replans = 0
    total_steps = 0
    total_backtracks = 0
    episodes_with_backtrack = 0
    failed_episodes_without_backtrack = 0
    failed_episodes_with_backtrack = 0
    succeeded_episodes_with_backtrack = 0
    replan_reasons = collections.Counter()
    per_task_stats = {}
    per_episode_stats = []

    t_start_wall = time.time()

    for task_id in tqdm.tqdm(selected_task_ids, desc="Tasks"):
        libero_env.set_task(task_id)
        initial_states = libero_env.initial_states
        task_description = libero_env.task_description

        task_episodes = 0
        task_successes = 0
        task_calls = 0
        task_backtracks = 0
        task_failed_no_bt = 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), leave=False):
            obs = libero_env.reset(initial_state=initial_states[episode_idx])
            perturber.reset(episode_seed=args.seed * 10000 + task_id * 100 + episode_idx)
            monitor.reset()
            backtracker.reset()

            action_plan = collections.deque()
            replay_images = []
            prev_obs = None
            last_action = None

            ep_server_calls = 0
            ep_early_replans = 0
            ep_backtracks = 0
            done = False

            if args.verbose:
                logging.info(f"Starting task {task_id} episode {episode_idx + 1}...")

            for t in range(max_steps):
                replay_images.append(obs["observation.images.front_image"])

                # Adaptive horizon check (only when not executing a backtrack sequence)
                if not backtracker.is_active:
                    should_replan, reason = monitor.should_replan(obs, prev_obs, last_action)
                    if should_replan and len(action_plan) > 0:
                        action_plan.clear()
                        ep_early_replans += 1
                        replan_reasons[reason] += 1
                        if args.verbose:
                            logging.info(f"  [t={t}] Early replan triggered ({reason})")

                using_backtrack = backtracker.is_active
                if using_backtrack:
                    action = backtracker.next_action()
                else:
                    rot = obs["observation.state"][3:7]
                    if args.pred_rot_type == "euler":
                        rot = convert_rotation(
                            rot, "quat", "euler", quat_order_src="xyzw", euler_order_dst="xyz"
                        )
                    elif args.pred_rot_type == "rot6d":
                        rot = convert_rotation(
                            rot, "quat", "rot6d", quat_order_src="xyzw"
                        )
                    state = np.concatenate(
                        [obs["observation.state"][:3], rot, obs["observation.state"][7:]]
                    )

                    if not action_plan:
                        batch = {
                            "observation.images.front_image": [obs["observation.images.front_image"]],
                            "observation.images.wrist_image": [obs["observation.images.wrist_image"]],
                            "observation.state": [state],
                            "task": [obs["task"]],
                            "repo_id": [args.repo_id],
                        }

                        if args.use_depth:
                            batch.update({
                                "observation.points.front": [obs["observation.points.front"]],
                                "observation.points.wrist": [obs["observation.points.wrist"]],
                            })
                            if args.remove_arm:
                                batch["observation.robot_joints_bbox"] = [
                                    get_robot_joint_boxes(
                                        obs["robot0_joints_pose_size"], return_o3d=False
                                    )
                                ]

                        points_workspace = libero_env.get_points_workspace(obs)

                        ov_out = policy_client.get_action(
                            batch,
                            options={
                                "pred_rot_type": args.pred_rot_type,
                                "points_workspace": points_workspace,
                                "remove_arm": args.remove_arm,
                            },
                        )
                        ep_server_calls += 1
                        action_chunk = ov_out.action[0].copy()

                        if args.post_process_action:
                            action_chunk[..., -1] = 2 * (1 - action_chunk[..., -1]) - 1

                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    raw_action = action_plan.popleft()
                    action_rot = convert_rotation(
                        raw_action[3:7], "quat", "axisangle", quat_order_src="xyzw"
                    )
                    action = np.concatenate(
                        [raw_action[:3], action_rot, raw_action[7:]]
                    ).astype(np.float32)
                    backtracker.record_action(action)

                # Apply physical or action perturbation before stepping simulator
                action_to_exec, perturb_event = perturber.maybe_perturb(
                    libero_env, t, obs, action
                )
                if perturb_event is not None and args.verbose:
                    logging.info(f"  [t={t}] Perturbation injected: {perturb_event}")

                prev_obs = obs
                last_action = None if using_backtrack else action_to_exec
                obs, reward, done, info = libero_env.step(action_to_exec.tolist())

                # Empty grasp check -> trigger controlled backtrack
                if not using_backtrack and backtracker.check_and_trigger(obs, action_to_exec):
                    action_plan.clear()
                    monitor.reset()
                    prev_obs = None
                    last_action = None
                    ep_backtracks += 1
                    total_backtracks += 1
                    replan_reasons["backtrack_empty_grasp"] += 1
                    if args.verbose:
                        logging.info(
                            f"  [t={t}] Empty grasp detected -> BACKTRACK triggered (#{backtracker.backtrack_count})"
                        )

                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            ep_steps = t + 1
            task_episodes += 1
            total_episodes += 1
            task_calls += ep_server_calls
            task_backtracks += ep_backtracks
            total_server_calls += ep_server_calls
            total_early_replans += ep_early_replans
            total_steps += ep_steps

            if ep_backtracks > 0:
                episodes_with_backtrack += 1
                if done:
                    succeeded_episodes_with_backtrack += 1
                else:
                    failed_episodes_with_backtrack += 1
            elif not done:
                failed_episodes_without_backtrack += 1
                task_failed_no_bt += 1

            per_episode_stats.append(
                {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "success": bool(done),
                    "steps": ep_steps,
                    "server_calls": ep_server_calls,
                    "early_replans": ep_early_replans,
                    "backtracks": ep_backtracks,
                }
            )

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if video_out_dir is not None:
                imageio.mimwrite(
                    pathlib.Path(video_out_dir)
                    / f"{task_segment}_rollout{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            if args.verbose:
                logging.info(
                    f"Success: {done} | server_calls: {ep_server_calls} | "
                    f"early_replans: {ep_early_replans} | backtracks: {ep_backtracks} | steps: {ep_steps}"
                )

        if libero_env.env is not None:
            libero_env.env.close()

        task_sr = float(task_successes) / float(max(1, task_episodes))
        task_avg_calls = float(task_calls) / float(max(1, task_episodes))
        task_avg_bt = float(task_backtracks) / float(max(1, task_episodes))
        per_task_stats[task_id] = {
            "description": task_description,
            "success_rate": task_sr,
            "avg_server_calls": task_avg_calls,
            "avg_backtracks": task_avg_bt,
            "failed_without_backtrack": task_failed_no_bt,
        }
        logging.info(
            f"Task {task_id} SR: {task_sr:.3f} | Avg calls/ep: {task_avg_calls:.1f} | "
            f"Avg bt/ep: {task_avg_bt:.2f} | Fail(bt=0): {task_failed_no_bt}/{task_episodes} | "
            f"Running Total SR: {float(total_successes) / float(total_episodes):.3f}"
        )

    wall_time = time.time() - t_start_wall
    total_failures = total_episodes - total_successes
    summary = {
        "args": dataclasses.asdict(args),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_failures": total_failures,
        "total_success_rate": float(total_successes) / float(max(1, total_episodes)),
        "avg_server_calls_per_episode": float(total_server_calls) / float(max(1, total_episodes)),
        "avg_early_replans_per_episode": float(total_early_replans) / float(max(1, total_episodes)),
        "avg_steps_per_episode": float(total_steps) / float(max(1, total_episodes)),
        "effective_horizon": float(total_steps) / float(max(1, total_server_calls)),
        "total_backtracks": total_backtracks,
        "avg_backtracks_per_episode": float(total_backtracks) / float(max(1, total_episodes)),
        "episodes_with_backtrack": episodes_with_backtrack,
        "episodes_with_backtrack_rate": float(episodes_with_backtrack) / float(max(1, total_episodes)),
        "failed_episodes_without_backtrack": failed_episodes_without_backtrack,
        "failed_episodes_with_backtrack": failed_episodes_with_backtrack,
        "succeeded_episodes_with_backtrack": succeeded_episodes_with_backtrack,
        "failed_without_backtrack_rate": float(failed_episodes_without_backtrack) / float(max(1, total_episodes)),
        "failed_without_backtrack_share_of_failures": (
            float(failed_episodes_without_backtrack) / float(total_failures)
            if total_failures > 0
            else 0.0
        ),
        "backtrack_recovery_rate": (
            float(succeeded_episodes_with_backtrack) / float(episodes_with_backtrack)
            if episodes_with_backtrack > 0
            else 0.0
        ),
        "replan_reasons": dict(replan_reasons),
        "wall_time_sec": wall_time,
        "per_task": per_task_stats,
        "per_episode": per_episode_stats,
    }

    logging.info("\n=== FINAL SUMMARY ===")
    logging.info(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("args", "per_episode")},
            indent=2,
        )
    )

    if summary_filename is not None:
        with open(summary_filename, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
