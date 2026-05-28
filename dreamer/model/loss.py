"""
loss.py

Loss functions for RSSM training.

Author: Jared Berry
"""
import torch
from torch import nn


class RSSMLoss(nn.Module):
    """
    RSSM loss, made with PyTorch.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = loss_params['recon_mult']
        self.beta = loss_params['beta']
        self.lam = loss_params['lambda']
        self.free_nats = loss_params.get('free_nats', 0.0)
        self.anneal_mode = loss_params['kld_anneal_mode']
        self.image_loss = loss_params.get('image_loss', 'mse')
        self.reward_mult = loss_params.get('reward_mult', 1.0)

    def kld_anneal(self, epoch):
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            mult = self.beta * ((epoch + 1) / self.num_epochs)
        else:
            raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

        return mult

    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1
        ).sum(dim=-1)

    def forward(self, tr, epoch):
        # Reconstruction loss
        if self.image_loss == 'mse':
            image_recon = self.recon_mult * nn.functional.mse_loss(tr['x_next'], tr['x_pred'], reduction='mean')
            image_recon += self.recon_mult * nn.functional.mse_loss(tr['x'][:, -1], tr['x_recon'], reduction='mean')

            if 'feature_next' in tr and tr['feature_next'] is not None and 'feature_pred' in tr:
                feature_recon = self.recon_mult * nn.functional.mse_loss(tr['feature_next'], tr['feature_pred'], reduction='mean')
                if 'feature' in tr and tr['feature'] is not None and 'feature_recon' in tr:
                    feature_recon += self.recon_mult * nn.functional.mse_loss(tr['feature'][:, -1], tr['feature_recon'], reduction='mean')
            else:
                feature_recon = torch.zeros((), device=tr['x'].device)

            if 'reward_next' in tr and tr['reward_next'] is not None and 'reward_pred' in tr:
                reward_recon = self.reward_mult * nn.functional.mse_loss(tr['reward_next'], tr['reward_pred'], reduction='mean')
                if 'reward' in tr and tr['reward'] is not None and 'reward_recon' in tr:
                    reward_recon += self.reward_mult * nn.functional.mse_loss(tr['reward'], tr['reward_recon'], reduction='mean')
            else:
                reward_recon = torch.zeros((), device=tr['x'].device)
        else:
            raise NotImplementedError(f"Image loss {self.image_loss} not supported!")

        # Encoding KL Divergence
        # KL loss (posterior vs prior)
        kld = self.kl_divergence(
            tr["mu_posts"],
            tr["log_var_posts"],
            tr["mu_priors"],
            tr["log_var_priors"]
        )
        kld = torch.clamp(kld, min=self.free_nats)
        kld = kld.mean()
        kld = self.kld_anneal(epoch) * kld

        loss = image_recon + feature_recon + reward_recon + kld
        if torch.isnan(loss):
            breakpoint()

        # Make return dictionary for loss values
        loss_return = {
            "Image Reconstruction Loss": image_recon.detach().cpu().item(),
            "Feature Reconstruction Loss": feature_recon.detach().cpu().item(),
            "Reward Reconstruction Loss": reward_recon.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
        }
        return loss, loss_return