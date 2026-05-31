"""
rssm.py

RSSM model architecture made with PyTorch.

Author: Jared Berry
"""
import torch
from torch import nn
import torch.nn.functional as F

from dreamer.model.encoder import ConvEncoder
from dreamer.model.decoder import ConvDecoder, ScalarUncertaintyConvDecoder


class RSSM(nn.Module):
    """
    An RSSM with convolutional encoder-decoder and transition model.

    Args:
        enc_latent_size: Latent dimension of encoder
        stochastic_size: Stochastic state latent dimension
        deterministic_size: Deterministic state latent dimension
        control_size: Dimension of control vector
        past_length: Length of training observation history
        pred_length: Prediction horizon length
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
        uncertainty_output: Whether to use an uncertainty decoder
        feature_latent_size: Latent dimension of encoded feature vector
        feature_size: Dimension of feature vector, if present
        reward_size: Dimension of reward prediction output, if present
    """
    def __init__(self, enc_latent_size, stochastic_size, deterministic_size,
                 control_size, past_length, pred_length, conv_params, device, output_uncertainty=False,
                 img_channel_count=3, feature_latent_size=None, feature_size=None, reward_size=None):
        super().__init__()
        self.device = device
        self.output_uncertainty = output_uncertainty

        # Set number of hidden units
        self.enc_latent_size = enc_latent_size
        self.stochastic_size = stochastic_size              # Stochastic state
        self.deterministic_size = deterministic_size       # Deterministic state
        self.control_size = control_size
        self.past_length = past_length
        self.pred_length = pred_length
        self.feature_size = feature_size
        self.feature_latent_size = feature_latent_size if feature_latent_size is not None else enc_latent_size
        self.reward_size = reward_size
        self.has_feature = self.feature_size is not None

        # Dummy zero control vector
        self.dummy_u = torch.zeros((1, self.control_size)).to(self.device)

        # Encoder and decoder
        in_channels = img_channel_count
        self.encoder = ConvEncoder(enc_latent_size, in_channels, conv_params)
        if self.output_uncertainty:
            self.decoder = ScalarUncertaintyConvDecoder(stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)
        else:
            self.decoder = ConvDecoder(stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)
        self.out_image_shape = self.decoder.out_image_shape

        if self.has_feature:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_latent_size),
                nn.ReLU(),
            )

            self.feature_decoder = nn.Sequential(
                nn.Linear(self.stochastic_size, 32),
                nn.ReLU(),
                nn.Linear(32, self.feature_size),
            )
        else:
            self.feature_encoder = None
            self.feature_decoder = None

        if self.reward_size is not None:
            self.reward_decoder = nn.Sequential(
                nn.Linear(self.stochastic_size + self.deterministic_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.reward_size),
            )
        else:
            self.reward_decoder = None

        # Dreamer dynamics model definition
        self.num_layers = 2
        self.rnn = nn.GRU(
            self.stochastic_size + self.control_size,
            self.deterministic_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.prior = nn.Sequential(                     # Transition model
            nn.Linear(self.deterministic_size, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * self.stochastic_size)
        )

        post_input_size = self.enc_latent_size
        if self.has_feature:
            post_input_size += self.feature_latent_size

        self.post = nn.Sequential(                      # Representation model
            nn.Linear(post_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.stochastic_size)
        )

    def reparameterize(self, mu, log_var):
        # Get standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-5, max=1e5) # Prevent std from being too small

        # Generate random noise epsilon of same shape std
        eps = torch.randn_like(std)

        # Return reparameterized sample
        return mu + eps * std
    
    def rssm_step(self, h, z, u):
        """
        Single RSSM step
        
        Args:
            h: deterministic state
            z: stochastic latent
            u: control input
        Returns: h_next, z_next, mu, log_var
        """
        # repeat u across z's time dimension
        u = u.unsqueeze(1).repeat(1, z.size(1), 1)  # u: [B, past_length, control_size]
        rnn_input = torch.cat([z, u], dim=-1)
        _, h_next = self.rnn(rnn_input, h)  # h_next: [B, 1, deterministic]

        stats = self.prior(h_next[-1])
        mu, log_var = stats.chunk(2, dim=-1)
        # log_var = torch.clamp(log_var, min=1e-5, max=1e5)
        z_next = self.reparameterize(mu, log_var)

        return h_next, z_next, mu, log_var
    
    def encode_posterior(self, obs, feature=None, actions=None):
        B, T = obs.shape[:2]
        mus, log_vars, zs = [], [], []

        for t in range(T):
            x = obs[:, t]
            enc = self.encoder(x)

            if self.has_feature:
                if feature is None:
                    raise ValueError("feature observations are required when feature_size is set")
                feat = feature[:, t].float()
                feat_enc = self.feature_encoder(feat)
                enc = torch.cat([enc, feat_enc], dim=-1)

            stats = self.post(enc)
            mu, log_var = stats.chunk(2, dim=-1)
            # log_var = torch.clamp(log_var, min=1e-5, max=1e5)
            z = self.reparameterize(mu, log_var)

            mus.append(mu)
            log_vars.append(log_var)
            zs.append(z)
        return (
            torch.stack(mus, dim=1),
            torch.stack(log_vars, dim=1),
            torch.stack(zs, dim=1),
        )

    def _reward_features(self, h, z):
        return torch.cat([h[-1], z], dim=-1)

    def _match_image_shape(self, image, reference):
        """
        Resize decoded image to match reference spatial size if needed.
        """
        if image.shape[-2:] != reference.shape[-2:]:
            image = F.interpolate(image, size=reference.shape[-2:], mode='bilinear', align_corners=False)
        return image
    
    def transition(self, x, x_next, u, zs, feature=None, feature_next=None):
        h = torch.zeros(self.num_layers, x.size(0), self.deterministic_size, device=self.device)
        # Take last belief state as start
        z = zs[:, -1]

        # reconstruct current observation
        if self.output_uncertainty:
            x_recon, x_recon_uncertainty = self.decoder(zs[:, -1])
        else:
            x_recon = self.decoder(zs[:, -1])
        x_recon = self._match_image_shape(x_recon, x[:, -1])

        if self.reward_decoder is not None:
            reward_recon = self.reward_decoder(self._reward_features(h, zs[:, -1]))

        if self.has_feature:
            if feature is None:
                raise ValueError("feature observations are required when feature_size is set")
            feature_recon = self.feature_decoder(zs[:, -1])

        # Iterate over pred_length
        mu_priors, log_var_priors = [], []
        mu_posts, log_var_posts = [], [] # old version - i added encoded t0 mu in post? why...
        x_preds = []
        if self.output_uncertainty:
            x_pred_uncerts = []
        if self.reward_decoder is not None:
            reward_preds = []
        if self.has_feature:
            feature_preds = []

        window = x
        feature_window = feature
        for t in range(x_next.size(1)):
            # prior
            h, z_prior, mu_p, log_var_p = self.rssm_step(h, z.unsqueeze(1), u[:, t])
            
            # old way (takes into account past_length)
            # h, z_prior, mu_p, log_var_p = self.rssm_step(h, zs, u[:, t])
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)

            # decode prior (open loop prediction)
            if self.output_uncertainty:
                x_pred, x_pred_uncert = self.decoder(z_prior)
                x_pred_uncerts.append(x_pred_uncert)
            else:
                x_pred = self.decoder(z_prior)
            x_pred = self._match_image_shape(x_pred, x_next[:, t])
            x_preds.append(x_pred)

            if self.reward_decoder is not None:
                reward_pred = self.reward_decoder(self._reward_features(h, z_prior))
                reward_preds.append(reward_pred)

            if self.has_feature:
                feature_pred = self.feature_decoder(z_prior)
                feature_preds.append(feature_pred)

            if self.past_length > 1:
                window_frames = window[:, 1:]   # drop first frame
                window = torch.cat([window_frames, x_pred.unsqueeze(1).detach()], dim=1)
                if self.has_feature:
                    feature_frames = feature_window[:, 1:]
                    feature_window = torch.cat([feature_frames, feature_pred.unsqueeze(1).detach()], dim=1)
            else:
                window = x_pred.unsqueeze(1).detach()  # past_length==1, just use pred image
                if self.has_feature:
                    feature_window = feature_pred.unsqueeze(1).detach()

            if self.has_feature:
                mu_q, log_var_q, zs = self.encode_posterior(window, feature_window)
            else:
                mu_q, log_var_q, zs = self.encode_posterior(window)

            z = zs[:, -1]

            # # posterior update using real next frame
            # enc = self.encoder(x_next[:, t])
            # stats = self.post(enc)
            # mu_q, log_var_q = stats.chunk(2, dim=-1)
            # z = self.reparameterize(mu_q, log_var_q)

            mu_posts.append(mu_q[:, -1])
            log_var_posts.append(log_var_q[:, -1])

        # Stack accumulated priors + posteriors
        outputs = {
            "x_recon": x_recon,
            "x_pred": torch.stack(x_preds, dim=1),
            "mu_posts": torch.stack(mu_posts, dim=1),
            "log_var_posts": torch.stack(log_var_posts, dim=1),
            "mu_priors": torch.stack(mu_priors, dim=1),
            "log_var_priors": torch.stack(log_var_priors, dim=1),
        }

        if self.output_uncertainty:
            outputs["x_recon_uncertainty"] = x_recon_uncertainty
            outputs["x_pred_uncertainty"] = torch.stack(x_pred_uncerts, dim=1)

        if self.reward_decoder is not None:
            outputs["reward_recon"] = reward_recon
            outputs["reward_pred"] = torch.stack(reward_preds, dim=1)

        if self.has_feature:
            outputs["feature_recon"] = feature_recon
            outputs["feature_pred"] = torch.stack(feature_preds, dim=1)

        return outputs
    
    def forward(self, x, x_next, u, feature=None, feature_next=None):
        # Infer belief over past context
        _, _, zs = self.encode_posterior(x, feature)
        outputs = self.transition(x, x_next, u, zs, feature=feature, feature_next=feature_next)
        return outputs

    def sample_traj(self, x0, u_seq):
        """Generate a trajectory of predicted images given initial observation and control sequence."""
        if self.has_feature:
            raise NotImplementedError('sample_traj currently supports image-only rollouts.')

        if x0.dim() == 3:
            x0 = x0.unsqueeze(0)
        if u_seq.dim() == 2:
            u_seq = u_seq.unsqueeze(0)

        x0 = x0.to(self.device).float()
        u_seq = u_seq.to(self.device).float()

        # Keep original render size for output, but encode at model input resolution.
        x0_ref = x0
        model_h = int(self.out_image_shape[1])
        model_w = int(self.out_image_shape[2])
        if x0.shape[-2:] != (model_h, model_w):
            x0_model = F.interpolate(x0, size=(model_h, model_w), mode='bilinear', align_corners=False)
        else:
            x0_model = x0

        with torch.no_grad():
            context = x0_model.unsqueeze(1).repeat(1, self.past_length, 1, 1, 1)
            _, _, zs = self.encode_posterior(context)

            h = torch.zeros(self.num_layers, x0_model.size(0), self.deterministic_size, device=self.device)
            z = zs[:, -1]

            x_recon = self.decoder(z)
            x_recon = self._match_image_shape(x_recon, x0_ref)

            preds = [x_recon[0].permute(1, 2, 0)]
            for t in range(u_seq.size(1)):
                h, z_prior, _, _ = self.rssm_step(h, z.unsqueeze(1), u_seq[:, t])
                x_pred = self.decoder(z_prior)
                x_pred = self._match_image_shape(x_pred, x0_ref)
                preds.append(x_pred[0].permute(1, 2, 0))
                z = z_prior

            return torch.stack(preds, dim=0).clamp(0.0, 1.0)