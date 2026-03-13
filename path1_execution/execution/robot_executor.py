"""Robot executor: run trajectory in LIBERO sim and collect rollout data."""
import logging
import os
import time
import numpy as np
import h5py
from path1_execution.config import Path1Config, Trajectory

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger(__name__)


def run_trial(trajectory: Trajectory, env, config: Path1Config, initial_state=None):
    """Execute trajectory as a trial rollout. Returns (final_obs, sim_success, total_reward).

    If initial_state is provided, the demo state is re-applied after env.reset()
    so that object positions match the scene context used for planning.
    """
    obs = env.reset()
    if initial_state is not None:
        from sketch_anything.tools.run_pipeline import set_env_state
        set_env_state(env, initial_state)
        obs, _, _, _ = env.step(np.zeros(7))
    sim_success = False
    total_reward = 0.0

    current_eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    current_eef_quat = obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))
    current_pose = np.concatenate([current_eef_pos, current_eef_quat])  # (7,) pos+quat

    num_steps = 0
    for i, action in enumerate(trajectory.actions):
        target_pose = np.asarray(action, dtype=np.float64)[:7]

        # LIBERO OSC controller: action=1.0 ≈ 0.01m of EEF movement per step.
        # Trajectory stores absolute positions in meters, so scale the positional
        # delta by 1/0.01 = 100, then clip to the [-1, 1] controller action range.
        CONTROLLER_SCALE = 100.0
        delta_pos_m = target_pose[:3] - current_pose[:3]
        delta_pos = np.clip(delta_pos_m * CONTROLLER_SCALE, -1.0, 1.0)
        delta_ori = target_pose[3:6]
        gripper = target_pose[6]

        env_action = np.concatenate([delta_pos, delta_ori, [gripper]])
        obs, reward, done, info = env.step(env_action)
        total_reward += float(reward)
        num_steps = i + 1

        if done and reward > 0:
            sim_success = True
            logger.info("Trial succeeded at step %d (reward=%.3f)", i, reward)

        current_eef_pos = obs.get("robot0_eef_pos", current_pose[:3])
        current_eef_quat = obs.get("robot0_eef_quat", current_pose[3:7])
        current_pose = np.concatenate([current_eef_pos, current_eef_quat])

        if done:
            break

    logger.info(
        "Trial finished: sim_success=%s, total_reward=%.3f, steps=%d",
        sim_success,
        total_reward,
        num_steps,
    )
    return obs, sim_success, total_reward


def execute_trajectory(trajectory: Trajectory, env, config: Path1Config, record_video: bool = True, initial_state=None):
    """Execute trajectory for real (post-verification). Returns rollout data for HDF5 saving.

    If record_video=True (default), captures agentview frames and saves an MP4 to
    {config.output_dir}/rollout_video.mp4. Requires opencv-python.
    """
    from sketch_anything.libero_utils.env import get_camera_image

    obs = env.reset()
    if initial_state is not None:
        from sketch_anything.tools.run_pipeline import set_env_state
        set_env_state(env, initial_state)
        obs, _, _, _ = env.step(np.zeros(7))
    observations = []
    actions_taken = []
    rewards = []
    dones = []
    video_frames = []  # agentview RGB frames (H, W, 3) uint8

    current_eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    current_eef_quat = obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))
    current_pose = np.concatenate([current_eef_pos, current_eef_quat])
    observations.append({k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()})

    # Capture initial frame
    if record_video and _HAS_CV2:
        try:
            frame = get_camera_image(env, "agentview")
            video_frames.append(frame)
        except Exception:
            pass

    for action in trajectory.actions:
        target_pose = np.asarray(action, dtype=np.float64)[:7]
        CONTROLLER_SCALE = 100.0
        delta_pos_m = target_pose[:3] - current_pose[:3]
        delta_pos = np.clip(delta_pos_m * CONTROLLER_SCALE, -1.0, 1.0)
        delta_ori = target_pose[3:6]
        gripper = target_pose[6]
        env_action = np.concatenate([delta_pos, delta_ori, [gripper]])

        obs, reward, done, info = env.step(env_action)
        actions_taken.append(env_action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        observations.append({k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()})

        if record_video and _HAS_CV2:
            try:
                frame = get_camera_image(env, "agentview")
                video_frames.append(frame)
            except Exception:
                pass

        current_eef_pos = obs.get("robot0_eef_pos", current_pose[:3])
        current_eef_quat = obs.get("robot0_eef_quat", current_pose[3:7])
        current_pose = np.concatenate([current_eef_pos, current_eef_quat])

        if done:
            break

    # Save video if we collected frames
    if video_frames and _HAS_CV2:
        _save_video(video_frames, config.output_dir, fps=config.video_fps)

    return observations, actions_taken, rewards, dones


def _save_video(frames: list, output_dir: str, fps: int = 20) -> str:
    """Write a list of RGB (H, W, 3) uint8 frames to an MP4 file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "rollout_video.mp4")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for frame in frames:
        # get_camera_image returns RGB; cv2 expects BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    logger.info("Saved rollout video (%d frames, %dfps): %s", len(frames), fps, path)
    return path


def save_rollout_hdf5(observations, actions, rewards, dones, output_dir: str) -> str:
    """Save rollout to HDF5 in LIBERO-compatible format. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    path = os.path.join(output_dir, f"demo_{timestamp}.hdf5")

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")

        if actions:
            demo.create_dataset("actions", data=np.array(actions, dtype=np.float32))
        if rewards:
            demo.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))
        if dones:
            demo.create_dataset("dones", data=np.array(dones, dtype=bool))

        # Save observations (only numpy arrays)
        obs_group = demo.create_group("obs")
        if observations:
            for key in observations[0]:
                vals = [ob[key] for ob in observations if isinstance(ob.get(key), np.ndarray)]
                if vals:
                    try:
                        obs_group.create_dataset(key, data=np.array(vals, dtype=np.float32))
                    except Exception:
                        pass

        data.attrs["num_demos"] = 1
        demo.attrs["num_samples"] = len(actions)

    logger.info("Saved rollout HDF5: %s", path)
    return path
