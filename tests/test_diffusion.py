import numpy as np
import torch

from src.models.diffusion import GaussianDiffusion


def test_ddim_conditions_denoiser_on_transition_noise_level():
    diffusion = GaussianDiffusion(
        model_mean_type="x_0",
        schedule_kwargs={
            "schedule": "linear",
            "n_timestep": 10,
            "linear_start": 1e-4,
            "linear_end": 2e-2,
            "ddim_S": 5,
            "ddim_eta": 0,
        },
    )
    observed = []

    def denoiser(value, noise_level):
        observed.append(float(noise_level))
        return torch.zeros_like(value)

    diffusion.sample_ddim(
        denoiser,
        (1, 1, 2, 2, 2),
        noise=torch.zeros(1, 1, 2, 2, 2),
        clip_denoised=False,
    )

    steps = np.flip(diffusion.ddim_timesteps).copy()
    expected = torch.sqrt(diffusion.alphas_cumprod[torch.as_tensor(steps)]).numpy()
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=0)
