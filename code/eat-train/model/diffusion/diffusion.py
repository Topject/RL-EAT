import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Dict, Any

from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from .utils import Progress, Silent

class Diffusion(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, model: nn.Module, max_action: float = 1.0,
                 beta_schedule: str = 'linear', n_timesteps: int = 100,
                 loss_type: str = 'l1', clip_denoised: bool = True, predict_epsilon: bool = True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = action_dim # 兼容ActorProb接口
        self.max_action = max_action
        self.model = model  # The UNet model for noise prediction

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Assumes Losses is a dictionary-like object that returns a loss function instance
        self.loss_fn = Losses['l1']() if loss_type == 'l1' else Losses.l2()

    # --- Internal Diffusion Methods (No change needed) ---

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        diffusion_history = [x] if return_diffusion else None
        
        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({'t': i})
            if return_diffusion:
                diffusion_history.append(x)
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion_history, dim=1)
        else:
            return x, None

    # --- Tianshou-facing Methods (Modified) ---

    def sample(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Generates action logits for a given state by running the reverse diffusion process.
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        # p_sample_loop returns the final denoised action `x` and the diffusion history.
        # We only need `x`, which will serve as the logits for the policy.
        action_logits, _ = self.p_sample_loop(state, shape, *args, **kwargs)
        
        # IMPORTANT: Return the raw tensor. Do NOT apply softmax.
        # Tianshou's ActorProb will interpret this as logits.
        return action_logits

    def forward(self, obs: Union[np.ndarray, torch.Tensor, Dict], state: Any = None, info: Dict = {}) -> Tuple[torch.Tensor, Any]:
        """
        Tianshou-compatible forward pass.

        :param obs: Observation from the environment. Can be a NumPy array, a tensor, or a Tianshou LazyDict.
        :param state: Hidden state for recurrent policies (unused here, passed through).
        :param info: Environment information dictionary.
        :return: A tuple of (logits, state).
        """
        # Tianshou collectors might pass obs as a LazyDict or NumPy array. Convert to a tensor.
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.betas.device)
        if not isinstance(obs, torch.Tensor):
            # This handles the common case of a LazyDict wrapper, e.g., obs.obs
            obs = obs.obs.to(self.betas.device)

        # Generate action logits using the diffusion sampling process.
        logits = self.sample(obs)

        # Return the logits and the (unused) hidden state, as required by Tianshou's API.
        return logits, state

    # --- Diffusion Model Training Methods (No change needed) ---

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)
        
        return loss * weights

    def loss(self, x, state, weights=1.0) -> torch.Tensor:
        """
        Calculates the training loss for the diffusion model.
        This should be called in your custom training loop to train the generator.
        """
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)
