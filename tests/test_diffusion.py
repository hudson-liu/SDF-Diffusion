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


def test_ddim_transition_rederives_noise_after_x0_guidance():
    diffusion = GaussianDiffusion(
        model_mean_type="x_0",
        schedule_kwargs={
            "schedule": "linear",
            "n_timestep": 4,
            "linear_start": 1e-4,
            "linear_end": 2e-2,
            "ddim_S": 2,
            "ddim_eta": 0,
        },
    )
    denoiser_inputs = []

    def denoiser(value, _noise_level):
        denoiser_inputs.append(value.clone())
        return torch.zeros_like(value)

    def guide(value, context):
        return torch.full_like(value, 0.2) if context["inference_step"] == 0 else None

    diffusion.sample_ddim(
        denoiser,
        (1, 1, 1, 1, 1),
        noise=torch.zeros(1, 1, 1, 1, 1),
        clip_denoised=False,
        x0_guidance_fn=guide,
    )

    guided_x0 = torch.tensor(0.2)
    alpha_t = diffusion.ddim_alphas[1]
    alpha_prev = diffusion.ddim_alphas_prev[1]
    guided_noise = -torch.sqrt(alpha_t / (1 - alpha_t)) * guided_x0
    expected_next = torch.sqrt(alpha_prev) * guided_x0 + torch.sqrt(
        1 - alpha_prev
    ) * guided_noise
    torch.testing.assert_close(denoiser_inputs[1].squeeze(), expected_next)
