"""
dreamer.py

Dreamer v1 trainer built on top of the RSSM trainer.
"""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dreamer.trainer import RSSMTrainer


def _build_mlp(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class Dreamer(RSSMTrainer):
    """
    Dreamer v1 trainer with an actor and value function trained from imagination.
    """
    def __init__(self, env, model, optimizer, loss_fn=None, loss_params=None,
                 device=None, buffer_capacity=1000, batch_size=16,
                 context_length=None, pred_length=None, grad_clip=100.0,
                 feature_fn=None, action_fn=None, eval_env=None, run_dir=None,
                 render_every_epochs=0, eval_max_steps=None,
                 imagination_horizon=15, discount=0.99, lambda_=0.95,
                 actor_hidden_size=256, value_hidden_size=256,
                 actor_lr=1e-4, value_lr=1e-4, entropy_scale=1e-3,
                 actor_grad_clip=100.0, value_grad_clip=100.0):
        super().__init__(
            env=env,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_params=loss_params,
            device=device,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            context_length=context_length,
            pred_length=pred_length,
            grad_clip=grad_clip,
            feature_fn=feature_fn,
            action_fn=action_fn,
            eval_env=eval_env,
            run_dir=run_dir,
            render_every_epochs=render_every_epochs,
            eval_max_steps=eval_max_steps,
        )

        self.imagination_horizon = int(imagination_horizon)
        self.discount = float(discount)
        self.lambda_ = float(lambda_)
        self.entropy_scale = float(entropy_scale)
        self.actor_grad_clip = actor_grad_clip
        self.value_grad_clip = value_grad_clip

        self.is_discrete = hasattr(self.env.action_space, 'n')
        self.action_dim = int(self.env.action_space.n) if self.is_discrete else int(self.env.action_space.shape[0])
        latent_size = self.model.deterministic_size + self.model.stochastic_size

        if self.is_discrete:
            self.actor = _build_mlp(latent_size, actor_hidden_size, self.action_dim).to(self.device)
        else:
            self.actor = _build_mlp(latent_size, actor_hidden_size, 2 * self.action_dim).to(self.device)

        self.value = _build_mlp(latent_size, value_hidden_size, 1).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=float(value_lr))

        if self.model.reward_decoder is None:
            raise ValueError('Dreamer requires an RSSM model with reward_size enabled')

    @contextmanager
    def _freeze_modules(self, modules):
        """Allows the RSSM to be frozen during behavior learning backprop"""
        states = []
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                states.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in states:
                parameter.requires_grad_(requires_grad)

    def _latent_features(self, h, z):
        """Convert RSSM latent state to features for actor and value function"""
        if z.dim() == 3:
            z = z[:, -1]
        return torch.cat([h[-1], z], dim=-1)

    def _history_to_tensors(self, obs_history, feature_history=None):
        """Convert obs history to tensors"""
        obs_window = np.asarray(obs_history[-self.context_length:])
        obs_tensor = torch.as_tensor(obs_window, device=self.device).float().unsqueeze(0)

        feature_tensor = None
        if self.model.has_feature:
            if feature_history is None:
                raise ValueError('feature observations are required when feature_size is set')
            feature_window = np.asarray(feature_history[-self.context_length:])
            feature_tensor = torch.as_tensor(feature_window, device=self.device).float().unsqueeze(0)

        return obs_tensor, feature_tensor

    def _posterior_state_from_history(self, obs_history, feature_history=None):
        """Compute the initial latent state of an imagined trajectory from an observation history"""
        obs_tensor, feature_tensor = self._history_to_tensors(obs_history, feature_history)

        with torch.no_grad():
            if self.model.has_feature:
                _, _, zs = self.model.encode_posterior(obs_tensor, feature_tensor)
            else:
                _, _, zs = self.model.encode_posterior(obs_tensor)

        z = zs[:, -1]
        h = torch.zeros(self.model.num_layers, 1, self.model.deterministic_size, device=self.device)
        return h, z

    def _actor_output(self, features, deterministic=False):
        """Sample action + entropy estimate from stochastic actor"""
        if self.is_discrete:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action_index = torch.argmax(logits, dim=-1)
                action = F.one_hot(action_index, num_classes=self.action_dim).float()
            else:
                action_index = dist.sample()
                action = F.one_hot(action_index, num_classes=self.action_dim).float()
            entropy = dist.entropy().unsqueeze(-1)
            return action, entropy, action_index

        actor_out = self.actor(features)
        mean, log_std = actor_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action = torch.tanh(dist.rsample())
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, entropy, action

    def _policy_action(self, obs_history, feature_history=None, deterministic=False):
        """Generate policy action from observed history"""
        h, z = self._posterior_state_from_history(obs_history, feature_history)
        features = self._latent_features(h, z)

        with torch.no_grad():
            action, _, action_index = self._actor_output(features, deterministic=deterministic)

        if self.is_discrete:
            return int(action_index.item())
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _policy_from_obs(self, obs, info, deterministic=False):
        """Wrapper for eval"""
        feature = self._get_feature(obs, info)
        feature_history = [feature] if feature is not None else None
        return self._policy_action([obs], feature_history=feature_history, deterministic=deterministic)

    def _imagined_reward(self, h, z):
        """Reward prediction from latent state"""
        return self.model.reward_decoder(self.model._reward_features(h, z))

    def _imagine(self, start_h, start_z, deterministic=False):
        """Imagine a trajectory from a given latent state using the RSSM and actor"""
        h = start_h
        z = start_z

        features = [self._latent_features(h, z)]
        rewards = []
        entropies = []

        for _ in range(self.imagination_horizon):
            feature = self._latent_features(h, z)
            action, entropy, _ = self._actor_output(feature, deterministic=deterministic)
            h, z, _, _ = self.model.rssm_step(h, z.unsqueeze(1), action)

            features.append(self._latent_features(h, z))
            rewards.append(self._imagined_reward(h, z))
            entropies.append(entropy)

        return {
            'features': torch.stack(features, dim=1),
            'rewards': torch.stack(rewards, dim=1),
            'entropies': torch.stack(entropies, dim=1),
        }

    def _lambda_returns(self, rewards, values):
        """Compute the Dreamer value estimate (Eqn. 6)"""
        next_return = values[:, -1]
        returns = []

        for t in reversed(range(rewards.size(1))):
            bootstrap = values[:, t + 1]
            target = rewards[:, t] + self.discount * (
                (1.0 - self.lambda_) * bootstrap + self.lambda_ * next_return
            )
            returns.append(target)
            next_return = target

        returns.reverse()
        return torch.stack(returns, dim=1)

    def collect_episode(self, policy_fn=None, max_steps=None):
        obs, info = self.env.reset()
        feature = self._get_feature(obs, info)
        self.buffer.start_episode(self._to_numpy(obs), self._to_numpy(feature) if feature is not None else None)

        obs_history = [self._to_numpy(obs)]
        feature_history = [self._to_numpy(feature)] if feature is not None else None

        episode_return = 0.0
        steps = 0

        while True:
            if policy_fn is None:
                action = self._policy_action(obs_history, feature_history=feature_history, deterministic=False)
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
            obs_history.append(self._to_numpy(next_obs))
            obs_history = obs_history[-self.context_length:]
            if feature_history is not None:
                feature_history.append(self._to_numpy(next_feature))
                feature_history = feature_history[-self.context_length:]

            if done or (max_steps is not None and steps >= max_steps):
                if self.buffer.current_episode is not None:
                    self.buffer.finish_episode()
                break

        return episode_return, steps

    def train_step(self, epoch=0):
        # World model update
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

        world_model_loss, loss_return = self.loss_fn(training_data, epoch)

        self.optimizer.zero_grad()
        world_model_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        # Policy update from imagination
        with torch.no_grad():
            if self.model.has_feature:
                _, _, posterior_zs = self.model.encode_posterior(batch['x'], batch.get('feature'))
            else:
                _, _, posterior_zs = self.model.encode_posterior(batch['x'])
            start_h = torch.zeros(self.model.num_layers, batch['x'].size(0), self.model.deterministic_size, device=self.device)
            start_z = posterior_zs[:, -1]

        # Don't update model weights
        with self._freeze_modules([self.model, self.value]):
            imagined = self._imagine(start_h, start_z)

            rewards = imagined['rewards'].squeeze(-1)
            entropies = imagined['entropies'].squeeze(-1)
            target_values = self.value(imagined['features']).squeeze(-1)
            returns = self._lambda_returns(rewards, target_values)
            actor_loss = -(returns + self.entropy_scale * entropies).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        with torch.no_grad():
            imagined_detached = self._imagine(start_h, start_z)
            rewards_d = imagined_detached['rewards'].squeeze(-1)
            tv_d = self.value(imagined_detached['features']).squeeze(-1)
            returns_d = self._lambda_returns(rewards_d, tv_d)

        value_pred = self.value(imagined_detached['features'].detach()).squeeze(-1)[:, :-1]
        value_loss = F.mse_loss(value_pred, returns_d.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.value_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()

        loss_return['Actor Loss'] = actor_loss.detach().cpu().item()
        loss_return['Value Loss'] = value_loss.detach().cpu().item()
        total_loss = world_model_loss.detach().cpu().item() + loss_return['Actor Loss'] + loss_return['Value Loss']
        loss_return['Dreamer Total Loss'] = total_loss
        
        # Track imagined rewards for visualization
        imagined_reward_mean = rewards_d.mean().detach().cpu().item()
        loss_return['Imagined Reward Mean'] = imagined_reward_mean

        return total_loss, loss_return

    def save_weights(self, filename):
        if self.run_dir is None:
            return None

        path = self.checkpoint_dir / filename
        torch.save(
            {
                'model': self.model.state_dict(),
                'actor': self.actor.state_dict(),
                'value': self.value.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
            },
            path,
        )
        return path

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            if 'actor' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor'])
            if 'value' in checkpoint:
                self.value.load_state_dict(checkpoint['value'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'actor_optimizer' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            if 'value_optimizer' in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            return

        self.model.load_state_dict(checkpoint)

    def train_online(self, num_episodes, policy_fn=None, exp_fn=None, eval_policy_fn=None, max_steps=None,
                     updates_per_step=1, start_training_after=1, checkpoint_every_epochs=0,
                     final_weights_name='dreamer_final.pt', plot_losses_every_epochs=None,
                     epoch_offset=0, exp_noise=0.0):
        if eval_policy_fn is None:
            eval_policy_fn = lambda obs, info: self._policy_from_obs(obs, info, deterministic=True)

        return super().train_online(
            num_episodes=num_episodes,
            policy_fn=policy_fn,
            exp_fn=exp_fn,
            eval_policy_fn=eval_policy_fn,
            max_steps=max_steps,
            updates_per_step=updates_per_step,
            start_training_after=start_training_after,
            checkpoint_every_epochs=checkpoint_every_epochs,
            final_weights_name=final_weights_name,
            plot_losses_every_epochs=plot_losses_every_epochs,
            epoch_offset=epoch_offset,
            exp_noise=exp_noise,
        )