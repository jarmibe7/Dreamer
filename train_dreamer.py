"""
train_dreamer.py

YAML-driven Dreamer training entrypoint.
"""
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch

from dreamer.dreamer import Dreamer
from dreamer.runtime_utils import (
    set_seed,
    make_envs,
    get_control_size,
    build_model,
    maybe_load_weights,
    make_run_dir,
    coerce_training_config,
)

def exploration_policy(obs, info):
    # Directed exploration for car env
    steering = np.random.uniform(-0.3, 0.3)
    gas = np.random.uniform(0.5, 1.0)
    brake = 0.0
    return np.array([steering, gas, brake], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train Dreamer from YAML config')
    parser.add_argument('--config', type=str, default='config/train_dreamer.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = config['training'].get('seed', 0)
    set_seed(seed)

    configured_device = config['training'].get('device', 'cpu')
    if configured_device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available. Falling back to CPU.')
        configured_device = 'cpu'
    device = torch.device(configured_device)

    train_env, eval_env = make_envs(config)
    control_size = get_control_size(train_env.action_space)
    run_dir = make_run_dir(config)

    train_cfg = config.get('training', {})
    coerce_training_config(train_cfg)

    # Save the config to the run directory for reproducibility
    with open(run_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    model = build_model(config, device, control_size)
    maybe_load_weights(model, config['checkpoint'], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    dreamer = Dreamer(
        env=train_env,
        model=model,
        optimizer=optimizer,
        loss_params=config['loss'],
        device=device,
        buffer_capacity=config['training']['buffer_capacity'],
        batch_size=config['training']['batch_size'],
        context_length=config['model']['past_length'],
        pred_length=config['model']['pred_length'],
        grad_clip=config['training']['grad_clip'],
        eval_env=eval_env,
        run_dir=run_dir,
        render_every_epochs=config['evaluation']['render_every_epochs'],
        eval_max_steps=config['evaluation'].get('max_episode_steps'),
        imagination_horizon=config['dreamer']['imagination_horizon'],
        discount=config['dreamer'].get('discount', 0.99),
        lambda_=config['dreamer'].get('lambda', 0.95),
        actor_hidden_size=config['dreamer'].get('actor_hidden_size', 256),
        value_hidden_size=config['dreamer'].get('value_hidden_size', 256),
        actor_lr=config['dreamer'].get('actor_lr', 1e-4),
        value_lr=config['dreamer'].get('value_lr', 1e-4),
        entropy_scale=config['dreamer'].get('entropy_scale', 1e-3),
        actor_grad_clip=config['dreamer'].get('actor_grad_clip', 100.0),
        value_grad_clip=config['dreamer'].get('value_grad_clip', 100.0),
    )

    train_episodes = int(config['training']['epochs'])
    train_max_steps = config['training'].get('max_episode_steps')
    if train_max_steps is None:
        print(f"Planned real training timesteps: unknown (training.max_episode_steps is unset for {train_episodes} episodes)")
    else:
        train_max_steps = int(train_max_steps)
        total_real_timesteps = train_episodes * train_max_steps
        print(
            f"Planned real training timesteps: {total_real_timesteps} "
            f"({train_episodes} episodes x {train_max_steps} max steps)"
        )

    try:
        dreamer.train_online(
            num_episodes=config['training']['epochs'],
            exp_fn=exploration_policy,
            max_steps=config['training'].get('max_episode_steps'),
            updates_per_step=config['training']['updates_per_epoch'],
            start_training_after=config['training']['start_training_after'],
            checkpoint_every_epochs=config['checkpoint']['save_every_epochs'],
            final_weights_name=config['checkpoint'].get('final_weights_name', 'dreamer_final.pt'),
            plot_losses_every_epochs=config['evaluation']['render_every_epochs'],
            exp_noise=config['training'].get('exploration_noise', 0.0),
        )
    finally:
        dreamer.close()


if __name__ == '__main__':
    main()
