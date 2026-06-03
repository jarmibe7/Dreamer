"""
model_utils.py

Utilities for RSSM training moved from utils.py. Kept separate so
model-related helpers live with the model package.
"""

from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont


def _mean_loss_dict(loss_dicts):
    if not loss_dicts:
        return {}

    totals = defaultdict(float)
    counts = defaultdict(int)

    for loss_dict in loss_dicts:
        for key, value in loss_dict.items():
            totals[key] += float(value)
            counts[key] += 1

    return {key: totals[key] / counts[key] for key in totals}


class RSSMEvaluator:
    """
    Simple evaluation and plotting helper for online RSSM training.
    """
    def __init__(self, eval_env=None, render_every_epochs=0, max_steps=None, run_dir=None):
        self.eval_env = eval_env
        self.render_every_epochs = render_every_epochs
        self.max_steps = max_steps
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.plot_dir = self.run_dir / 'plots' if self.run_dir is not None else None
        self.video_dir = self.run_dir / 'videos' if self.run_dir is not None else None

        if self.plot_dir is not None:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_episode(self, policy_fn=None):
        if self.eval_env is None:
            return None

        obs, info = self.eval_env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            if policy_fn is None:
                action = self.eval_env.action_space.sample()
            else:
                action = policy_fn(obs, info)

            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += float(reward)
            steps += 1

            if terminated or truncated or (self.max_steps is not None and steps >= self.max_steps):
                break

        return {
            'return': total_reward,
            'steps': steps,
        }

    def maybe_render(self, epoch, policy_fn=None):
        if self.eval_env is None or self.render_every_epochs <= 0:
            return None

        if epoch % self.render_every_epochs != 0:
            return None

        metrics = self.evaluate_episode(policy_fn=policy_fn)
        if metrics is not None:
            print(f"  Eval(render) | Return {metrics['return']:.2f} | Steps {metrics['steps']}")
        return metrics

    def plot_losses(self, history, epoch=None):
        if self.plot_dir is None:
            return None

        loss_history = [entry for entry in history if entry.get('loss_return') is not None]
        if not loss_history:
            return None

        episodes = []
        episode_returns = []
        reconstruction = []
        image_reconstruction = []
        reward_reconstruction = []
        actor_loss = []
        value_loss = []
        kld = []
        total = []
        imagined_reward_means = []

        for entry in loss_history:
            loss_return = entry['loss_return']
            image_recon_value = loss_return.get('Image Reconstruction Loss', 0.0)
            reward_recon_value = loss_return.get('Reward Reconstruction Loss', 0.0)
            recon_value = image_recon_value + loss_return.get('Feature Reconstruction Loss', 0.0) + reward_recon_value
            kld_value = loss_return.get('KLD', 0.0)
            episodes.append(entry['episode'])
            reconstruction.append(recon_value)
            image_reconstruction.append(image_recon_value)
            reward_reconstruction.append(reward_recon_value)
            actor_loss.append(loss_return.get('Actor Loss', 0.0))
            value_loss.append(loss_return.get('Value Loss', 0.0))
            kld.append(kld_value)
            total.append(recon_value + kld_value)
            episode_returns.append(entry.get('return', 0.0))
            imagined_reward_means.append(loss_return.get('Imagined Reward Mean', 0.0))

        # Use a non-interactive backend so saving works over SSH/headless
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            "Can't save over SSH"
            import matplotlib.pyplot as plt

        figure, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        axis = axes[0]
        axis_kld = axis.twinx()
        recon_axis = axes[1]
        reward_axis = recon_axis.twinx()
        policy_axis = axes[2]
        value_axis = policy_axis.twinx()
        episode_axis = axes[3]

        recon_line = axis.plot(episodes, reconstruction, label='Reconstruction', color='tab:blue')[0]
        total_line = axis.plot(episodes, total, label='Total', color='tab:green', linestyle='--', alpha=0.8)[0]
        kld_line = axis_kld.plot(episodes, kld, label='KLD', color='tab:orange')[0]

        image_recon_line = recon_axis.plot(episodes, image_reconstruction, label='Image Reconstruction', color='tab:blue')[0]
        reward_recon_line = reward_axis.plot(episodes, reward_reconstruction, label='Reward Reconstruction', color='tab:red')[0]

        axis.set_ylabel('Reconstruction / Total', color='tab:green')
        axis_kld.set_ylabel('KLD', color='tab:orange')
        axis.tick_params(axis='y', labelcolor='tab:green')
        axis_kld.tick_params(axis='y', labelcolor='tab:orange')
        axis.set_title('RSSM Training Loss')
        axis.grid(True, alpha=0.3)
        axis.legend([recon_line, total_line, kld_line], ['Reconstruction', 'Total', 'KLD'], loc='upper right')
        recon_axis.set_ylabel('Image Reconstruction Loss', color='tab:blue')
        reward_axis.set_ylabel('Reward Reconstruction Loss', color='tab:red')
        recon_axis.tick_params(axis='y', labelcolor='tab:blue')
        reward_axis.tick_params(axis='y', labelcolor='tab:red')
        recon_axis.set_title('Image and Reward Reconstruction')
        recon_axis.grid(True, alpha=0.3)
        recon_axis.legend([image_recon_line, reward_recon_line], ['Image Reconstruction', 'Reward Reconstruction'], loc='upper right')
        policy_line = policy_axis.plot(episodes, actor_loss, label='Actor Loss', color='tab:purple')[0]
        value_line = value_axis.plot(episodes, value_loss, label='Value Loss', color='tab:brown')[0]
        policy_axis.set_ylabel('Actor Loss', color='tab:purple')
        value_axis.set_ylabel('Value Loss', color='tab:brown')
        policy_axis.tick_params(axis='y', labelcolor='tab:purple')
        value_axis.tick_params(axis='y', labelcolor='tab:brown')
        policy_axis.set_title('Actor and Value Network Losses')
        policy_axis.grid(True, alpha=0.3)
        policy_axis.legend([policy_line, value_line], ['Actor Loss', 'Value Loss'], loc='upper right')
        
        imagined_reward_axis = episode_axis.twinx()
        episode_line = episode_axis.plot(episodes, episode_returns, label='Episode Return', color='tab:purple')[0]
        imagined_line = imagined_reward_axis.plot(episodes, imagined_reward_means, label='Imagined Reward Mean', color='tab:cyan')[0]
        episode_axis.set_xlabel('Epoch')
        episode_axis.set_ylabel('Cumulative Return', color='tab:purple')
        imagined_reward_axis.set_ylabel('Imagined Reward Mean', color='tab:cyan')
        episode_axis.tick_params(axis='y', labelcolor='tab:purple')
        imagined_reward_axis.tick_params(axis='y', labelcolor='tab:cyan')
        episode_axis.set_title('Cumulative Return and Imagined Rewards per Training Epoch')
        episode_axis.grid(True, alpha=0.3)
        episode_axis.legend([episode_line, imagined_line], ['Episode Return', 'Imagined Reward Mean'], loc='upper left')
        figure.tight_layout()

        if epoch is None:
            plot_name = 'losses.png'
        else:
            plot_name = f'losses_epoch_{epoch:04d}.png'

        plot_path = self.plot_dir / plot_name
        figure.savefig(plot_path)
        plt.close(figure)
        print(f'  Saved loss plot: {plot_path}')
        return plot_path

    def save_episode_video(self, model, policy_fn=None, pred_length=None, filename=None):
        if self.eval_env is None or self.video_dir is None:
            return None

        frames = []
        actions = []
        obs, info = self.eval_env.reset()
        done = False

        while not done:
            if policy_fn is None:
                action = self.eval_env.action_space.sample()
            else:
                action = policy_fn(obs, info)

            actions.append(action)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = bool(terminated or truncated)

            frame = self.eval_env.render()
            if frame is None and isinstance(obs, (np.ndarray,)) and obs.ndim == 3:
                frame = obs
            if frame is None:
                continue
            frames.append(np.asarray(frame).astype(np.uint8))

        if len(frames) == 0:
            print('  Eval video: no frames collected; skipping video.')
            return None

        def _format_action_for_model(a, action_space):
            if hasattr(action_space, 'n'):
                arr = np.zeros(action_space.n, dtype=np.float32)
                arr[int(a)] = 1.0
                return arr
            return np.asarray(a, dtype=np.float32)

        if filename is None:
            filename = f'episode_epoch.mp4'
        video_path = self.video_dir / filename

        writer = imageio.get_writer(str(video_path), fps=30, codec='libx264', macro_block_size=1)

        device = getattr(model, 'device', None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = torch.device('cpu')

        num_steps = len(frames)
        for t in range(num_steps):
            gt_images = []
            for k in range((pred_length or model.pred_length) + 1):
                idx = t + k
                if idx < num_steps:
                    gt = frames[idx]
                else:
                    gt = np.zeros_like(frames[0])
                gt_images.append(gt)

            curr_frame = frames[t]
            x0 = torch.from_numpy(curr_frame).float() / 255.0
            x0 = x0.permute(2, 0, 1).unsqueeze(0).to(device)

            control_space = self.eval_env.action_space
            u_seq_list = []
            for k in range((pred_length or model.pred_length)):
                idx = t + k
                if idx < len(actions):
                    a = actions[idx]
                else:
                    if hasattr(control_space, 'n'):
                        a = 0
                    else:
                        a = np.zeros(control_space.shape, dtype=np.float32)
                uvec = _format_action_for_model(a, control_space)
                u_seq_list.append(torch.from_numpy(uvec).float())

            if len(u_seq_list) == 0:
                raise RuntimeError(f'No control seq to predict in eval video {t}')
            else:
                u_seq = torch.stack(u_seq_list, dim=0)
                try:
                    pred_traj = model.sample_traj(x0, u_seq)
                except Exception:
                    pred_traj = None

                if pred_traj is None:
                    raise RuntimeError(f'Eval video prediction failed at step {t}')
                else:
                    pred_np = (pred_traj.cpu().numpy() * 255.0).astype(np.uint8)
                    pred_images = []
                    for k in range((pred_length or model.pred_length) + 1):
                        if k < pred_np.shape[0]:
                            pred_images.append(pred_np[k])
                        else:
                            pred_images.append(np.zeros_like(pred_np[0]))

            row_gt = np.concatenate(gt_images, axis=1)
            row_pred = np.concatenate(pred_images, axis=1)

            if row_gt.shape != row_pred.shape:
                row_pred = row_pred[:row_gt.shape[0], :row_gt.shape[1], :]

            composite = np.concatenate([row_gt, row_pred], axis=0)

            try:
                H, W = frames[0].shape[0], frames[0].shape[1]
            except Exception:
                H = row_gt.shape[0] // 2
                W = row_gt.shape[1] // ((pred_length or model.pred_length) + 1)

            pil = Image.fromarray(composite)
            draw = ImageDraw.Draw(pil)
            font_size = max(18, H // 12)
            font = None
            for font_name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except Exception:
                    continue
            if font is None:
                for font_path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        continue
            if font is None:
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

            label_margin = max(6, font_size // 3)
            left_w = max(80, font_size * 6)
            box_h = font_size + 6
            draw.rectangle([(0, 0), (left_w, box_h)], fill=(0, 0, 0))
            draw.text((label_margin, 1), 'GT', fill=(255, 255, 255), font=font)
            draw.rectangle([(0, H), (left_w, H + box_h)], fill=(0, 0, 0))
            draw.text((label_margin, H + 1), 'PRED', fill=(255, 255, 255), font=font)

            cols = (pred_length or model.pred_length) + 1
            W = max(1, row_gt.shape[1] // cols)
            for k in range(cols):
                x_center = int(k * W + W // 2)
                label = f't+{k}' if k > 0 else 't'
                try:
                    if font is not None and hasattr(font, 'getsize'):
                        text_w, text_h = font.getsize(label)
                    else:
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                except Exception:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                tx = max(2, x_center - text_w // 2)
                draw.rectangle([(tx - 2, 0), (tx + text_w + 2, box_h)], fill=(0, 0, 0))
                draw.text((tx, 1), label, fill=(255, 255, 255), font=font)

            composite = np.asarray(pil)
            writer.append_data(composite)

        writer.close()
        print(f'  Saved eval video: {video_path}')
        return video_path


class TrajectoryReplayBuffer:
    """
    Simple episodic replay buffer for online RSSM training.
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.episodes = []
        self.current_episode = None

    def start_episode(self, obs, feature=None):
        self.current_episode = {
            'obs': [obs],
            'feature': [feature] if feature is not None else None,
            'action': [],
            'reward': [],
        }

    def add_step(self, action, reward, next_obs, next_feature=None, done=False):
        if self.current_episode is None:
            raise RuntimeError('start_episode must be called before add_step')

        self.current_episode['action'].append(action)
        self.current_episode['reward'].append(reward)
        self.current_episode['obs'].append(next_obs)
        if self.current_episode['feature'] is not None:
            self.current_episode['feature'].append(next_feature)

        if done:
            self.finish_episode()

    def finish_episode(self):
        if self.current_episode is None:
            return

        if len(self.current_episode['action']) > 0:
            self.episodes.append(self.current_episode)
            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

        self.current_episode = None

    def __len__(self):
        return len(self.episodes)

    def sample_batch(self, batch_size, context_length, pred_length, device):
        if not self.episodes:
            raise RuntimeError('replay buffer is empty')

        min_obs_len = context_length + pred_length
        eligible_episodes = [episode for episode in self.episodes if len(episode['obs']) >= min_obs_len]
        if not eligible_episodes:
            raise RuntimeError('no episodes in replay buffer are long enough for the requested context/prediction window')

        batch = []
        for _ in range(batch_size):
            episode = random.choice(eligible_episodes)
            obs = episode['obs']
            actions = episode['action']
            rewards = episode['reward']
            features = episode['feature']

            max_start = len(obs) - context_length - pred_length
            if max_start < 0:
                raise RuntimeError('not enough trajectory length for the requested context/prediction window')

            start = random.randint(0, max_start)

            x = np.asarray(obs[start:start + context_length])
            x_next = np.asarray(obs[start + context_length:start + context_length + pred_length])
            u = np.asarray(actions[start + context_length - 1:start + context_length - 1 + pred_length])
            reward = np.asarray(rewards[start + context_length - 1])
            reward_next = np.asarray(rewards[start + context_length - 1:start + context_length - 1 + pred_length])

            if reward.ndim == 0:
                reward = reward.reshape(1)
            if reward_next.ndim == 1:
                reward_next = reward_next[:, None]

            sample = {
                'x': torch.as_tensor(x, device=device).float(),
                'x_next': torch.as_tensor(x_next, device=device).float(),
                'u': torch.as_tensor(u, device=device).float(),
                'reward': torch.as_tensor(reward, device=device).float(),
                'reward_next': torch.as_tensor(reward_next, device=device).float(),
            }

            if features is not None:
                feature = np.asarray(features[start:start + context_length])
                feature_next = np.asarray(features[start + context_length:start + context_length + pred_length])
                sample['feature'] = torch.as_tensor(feature, device=device).float()
                sample['feature_next'] = torch.as_tensor(feature_next, device=device).float()

            batch.append(sample)

        keys = batch[0].keys()
        return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys}
