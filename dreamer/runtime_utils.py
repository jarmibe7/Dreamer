"""
runtime_utils.py

Shared runtime helpers used by training entrypoints.
"""
from pathlib import Path
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class RenderObservationEnv:
    """
    Wrapper that uses env.render() frames as model observations.
    Useful for classic control envs with low-dimensional state observations.
    """
    def __init__(self, env, image_size=64, grayscale=False):
        self.env = env
        self.image_size = image_size
        self.grayscale = grayscale
        self.action_space = env.action_space
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, image_size, image_size),
            dtype=np.float32,
        )

    def _process_frame(self, frame):
        if frame is None:
            raise RuntimeError("render frame is None. Ensure the env uses render_mode='rgb_array'.")

        image = torch.from_numpy(frame).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        if self.grayscale:
            image = image.mean(dim=1, keepdim=True)

        return image.squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        frame = self.env.render()
        model_obs = self._process_frame(frame)
        info = dict(info)
        info['state_obs'] = obs
        return model_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        model_obs = self._process_frame(frame)
        info = dict(info)
        info['state_obs'] = obs
        return model_obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_envs(config):
    env_cfg = config['env']
    env_name = env_cfg['name']
    train_env = gym.make(env_name, render_mode='rgb_array')
    train_env = RenderObservationEnv(
        train_env,
        image_size=env_cfg['image_size'],
        grayscale=env_cfg['grayscale'],
    )

    # default to rgb_array so evaluation can be recorded
    eval_render_mode = env_cfg.get('eval_render_mode', 'rgb_array')
    eval_env = gym.make(env_name, render_mode=eval_render_mode)
    eval_env = RenderObservationEnv(
        eval_env,
        image_size=env_cfg['image_size'],
        grayscale=env_cfg['grayscale'],
    )

    return train_env, eval_env


def get_control_size(action_space):
    if hasattr(action_space, 'n'):
        return int(action_space.n)
    return int(action_space.shape[0])


def build_model(config, device, control_size):
    model_cfg = config['model']
    env_cfg = config['env']

    channels = 1 if env_cfg['grayscale'] else 3

    from dreamer.model.rssm import RSSM

    model = RSSM(
        enc_latent_size=model_cfg['enc_latent_size'],
        stochastic_size=model_cfg['stochastic_size'],
        deterministic_size=model_cfg['deterministic_size'],
        control_size=control_size,
        past_length=model_cfg['past_length'],
        pred_length=model_cfg['pred_length'],
        conv_params=model_cfg['conv_params'],
        device=device,
        output_uncertainty=model_cfg.get('output_uncertainty', False),
        img_channel_count=channels,
        feature_latent_size=model_cfg.get('feature_latent_size'),
        feature_size=model_cfg.get('feature_size'),
        reward_size=model_cfg.get('reward_size'),
    ).to(device)

    return model


def maybe_load_weights(model, checkpoint_cfg, device):
    load_path = checkpoint_cfg.get('load_weights_path')
    if not load_path:
        return

    load_path = Path(load_path)
    if not load_path.is_file():
        raise FileNotFoundError(f"Could not find model weights at: {load_path}")

    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {load_path}")


def make_run_dir(config):
    env_name = config['env']['name']
    run_root = config['checkpoint'].get('run_root', 'runs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = Path(run_root) / env_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def coerce_training_config(train_cfg):
    """Cast common training config values to expected types.

    This helper mirrors what the original training script used so both
    entrypoints share the same validation logic.
    """
    _coerce_map = {
        'learning_rate': float,
        'batch_size': int,
        'buffer_capacity': int,
        'epochs': int,
        'updates_per_epoch': int,
        'start_training_after': int,
        'grad_clip': float,
    }
    for k, caster in _coerce_map.items():
        if k in train_cfg and train_cfg[k] is not None:
            try:
                train_cfg[k] = caster(train_cfg[k])
            except (TypeError, ValueError):
                raise ValueError(f"training.{k} must be a number (could not cast value: {train_cfg[k]!r})")

    return train_cfg
