"""
trainer.py

Online RSSM trainer (no RL).

Author: Jared Berry
"""
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from dreamer.model.loss import RSSMLoss
from dreamer.model_utils import TrajectoryReplayBuffer, RSSMEvaluator, _mean_loss_dict


class RSSMTrainer:
    """
    Modular online trainer for RSSM world model learning.
    """
    def __init__(self, env, model, optimizer, loss_fn=None, loss_params=None,
                 device=None, buffer_capacity=1000, batch_size=16,
                 context_length=None, pred_length=None, grad_clip=100.0,
                 feature_fn=None, action_fn=None, eval_env=None, run_dir=None,
                 render_every_epochs=0, eval_max_steps=None):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.device = device if device is not None else next(model.parameters()).device
        self.batch_size = batch_size
        self.context_length = context_length if context_length is not None else model.past_length
        self.pred_length = pred_length if pred_length is not None else model.pred_length
        self.grad_clip = grad_clip
        self.feature_fn = feature_fn
        self.action_fn = action_fn
        self.buffer = TrajectoryReplayBuffer(capacity=buffer_capacity)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.checkpoint_dir = self.run_dir / 'checkpoints' if self.run_dir is not None else None
        self.evaluator = RSSMEvaluator(
            eval_env=eval_env,
            render_every_epochs=render_every_epochs,
            max_steps=eval_max_steps,
            run_dir=run_dir,
        )

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            if loss_params is None:
                loss_params = {
                    'recon_mult': 1.0,
                    'beta': 1.0,
                    'kld_anneal_mode': 'const',
                    'image_loss': 'mse',
                }
            self.loss_fn = RSSMLoss(num_epochs=1, loss_params=loss_params)

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _format_action(self, action):
        if hasattr(self.env.action_space, 'n'):
            one_hot = np.zeros(self.env.action_space.n, dtype=np.float32)
            one_hot[int(action)] = 1.0
            return one_hot
        return self._to_numpy(action).astype(np.float32)

    def _get_feature(self, obs, info=None):
        if self.feature_fn is None:
            return None
        if info is None:
            return self.feature_fn(obs)
        return self.feature_fn(obs, info)

    def collect_episode(self, policy_fn=None, max_steps=None):
        obs, info = self.env.reset()
        feature = self._get_feature(obs, info)
        self.buffer.start_episode(self._to_numpy(obs), self._to_numpy(feature) if feature is not None else None)

        episode_return = 0.0
        steps = 0

        while True:
            if policy_fn is None:
                if self.action_fn is None:
                    action = self.env.action_space.sample()
                else:
                    action = self.action_fn(obs, info)
            else:
                action = policy_fn(obs, info)

            action_tensor = self._format_action(action)
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated
            next_feature = self._get_feature(next_obs, next_info)

            self.buffer.add_step(
                action_tensor,
                np.asarray(reward, dtype=np.float32),
                self._to_numpy(next_obs),
                self._to_numpy(next_feature) if next_feature is not None else None,
                done=done,
            )

            episode_return += float(reward)
            steps += 1
            obs = next_obs
            info = next_info

            if done or (max_steps is not None and steps >= max_steps):
                if self.buffer.current_episode is not None:
                    self.buffer.finish_episode()
                break

        return episode_return, steps

    def train_step(self, epoch=0):
        batch = self.buffer.sample_batch(
            self.batch_size,
            self.context_length,
            self.pred_length,
            self.device,
        )

        outputs = self.model(
            batch['x'],
            batch['x_next'],
            batch['u'],
            feature=batch.get('feature'),
            feature_next=batch.get('feature_next'),
        )

        training_data = dict(batch)
        training_data.update(outputs)

        loss, loss_return = self.loss_fn(training_data, epoch)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.detach().cpu().item(), loss_return

    def save_weights(self, filename):
        if self.run_dir is None:
            return None

        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
        return path

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        if self.evaluator is not None and self.evaluator.eval_env is not None:
            self.evaluator.eval_env.close()

    def train_online(self, num_episodes, policy_fn=None, exp_fn=None, eval_policy_fn=None, max_steps=None,
                     updates_per_step=1, start_training_after=1, checkpoint_every_epochs=0,
                     final_weights_name='rssm_final.pt', plot_losses_every_epochs=None,
                     epoch_offset=0, exp_noise=0.0):
        history = []
        progress_bar = tqdm(range(1, num_episodes + 1), desc='training', leave=True)

        if plot_losses_every_epochs is None:
            plot_losses_every_epochs = self.evaluator.render_every_epochs

        for episode in progress_bar:
            epoch = epoch_offset + episode

            if exp_fn is not None and (episode < start_training_after or np.random.rand() < exp_noise):
                exp_policy_fn = exp_fn
            else:
                exp_policy_fn = None
            episode_return, steps = self.collect_episode(
                policy_fn=exp_policy_fn,
                max_steps=max_steps, 
            )
            epoch_history = {
                'episode': episode,
                'return': episode_return,
                'steps': steps,
            }

            # Weight update after warm-up
            if len(self.buffer.episodes) >= start_training_after:
                episode_loss_values = []
                episode_loss_returns = []
                for _ in range(updates_per_step):
                    try:
                        loss_value, loss_return = self.train_step(epoch=epoch - 1)
                        episode_loss_values.append(loss_value)
                        episode_loss_returns.append(loss_return)
                    except RuntimeError:
                        break

                if episode_loss_values:
                    epoch_history['loss'] = float(np.mean(episode_loss_values))
                    epoch_history['loss_return'] = _mean_loss_dict(episode_loss_returns)

            # Eval
            eval_metrics = self.evaluator.maybe_render(epoch, policy_fn=eval_policy_fn)
            if eval_metrics is not None:
                epoch_history['eval_return'] = eval_metrics['return']
                epoch_history['eval_steps'] = eval_metrics['steps']
                # Also save an evaluation video showing ground-truth vs predictions
                try:
                    video_name = f'episode_epoch_{epoch:04d}.mp4'
                    self.evaluator.save_episode_video(self.model, policy_fn=eval_policy_fn, pred_length=self.pred_length, filename=video_name)
                except Exception as e:
                    print(f'  Eval video generation failed: {e}')

            if checkpoint_every_epochs > 0 and epoch % checkpoint_every_epochs == 0:
                checkpoint_name = f'rssm_epoch_{epoch}.pt'
                checkpoint_path = self.save_weights(checkpoint_name)
                if checkpoint_path is not None:
                    print(f'  Saved checkpoint: {checkpoint_path}')

            if plot_losses_every_epochs > 0 and epoch % plot_losses_every_epochs == 0:
                self.evaluator.plot_losses(history + [epoch_history], epoch=epoch)

            postfix = {
                'return': f'{episode_return:.2f}',
            }
            if 'loss' in epoch_history:
                postfix['loss'] = f"{epoch_history['loss']:.4f}"
            if 'eval_return' in epoch_history:
                postfix['eval'] = f"{epoch_history['eval_return']:.2f}"
            progress_bar.set_postfix(postfix)

            history.append(epoch_history)

        if self.run_dir is not None:
            final_path = self.save_weights(final_weights_name)
            if final_path is not None:
                print(f'Saved final weights: {final_path}')

        self.evaluator.plot_losses(history, epoch=epoch_offset + num_episodes)

        return history