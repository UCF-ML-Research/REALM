import numpy as np
import torch
from .utils import diff2clf, clf2diff


def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


class PurificationForward(torch.nn.Module):
    def __init__(self, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device, amplitude_cut_range, phase_cut_range, delta, forward_noise_steps):
        super().__init__()
        self.diffusion = diffusion
        self.device = device
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.amplitude_cut_range = amplitude_cut_range
        self.phase_cut_range = phase_cut_range
        self.delta = delta
        assert sampling_method in ['ddim', 'ddpm']
        self.eta = 0 if sampling_method == 'ddim' else 1
        self.is_imagenet = is_imagenet
        self.forward_noise_steps = forward_noise_steps

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if isinstance(t, int):
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self, ori_x, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = self.amplitude_phase_exchange_torch(ori_x, x0_t)
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt

    def compute_fft(self, image):
        amplitude_channels = []
        phase_channels = []
        for channel in range(3):
            f = torch.fft.fft2(image[channel, :, :])
            fshift = torch.fft.fftshift(f)
            amplitude_channels.append(torch.abs(fshift))
            phase_channels.append(torch.angle(fshift) + torch.pi)
        return amplitude_channels, phase_channels

    def low_pass_exchange(self, amplitude_channels, amplitude_channels_0_t):
        filtered = []
        for i in range(3):
            rows, cols = amplitude_channels[i].shape
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            low_frequency = torch.from_numpy(np.sqrt(U ** 2 + V ** 2) <= self.amplitude_cut_range).to(self.device)
            amplitude_channels_0_t[i] = torch.where(low_frequency, amplitude_channels[i], amplitude_channels_0_t[i])
            filtered.append(amplitude_channels_0_t[i])
        return filtered

    def phase_low_pass_exchange(self, phase_channels, phase_channels_0_t):
        filtered = []
        for i in range(3):
            rows, cols = phase_channels[i].shape
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            low_frequency = torch.from_numpy(np.sqrt(U ** 2 + V ** 2) <= self.phase_cut_range).to(self.device)
            phase_channels_0_t[i] = torch.where(low_frequency, phase_channels[i], phase_channels_0_t[i])
            phase_channels_0_t[i][low_frequency] = torch.clip(
                phase_channels_0_t[i][low_frequency],
                phase_channels[i][low_frequency] - self.delta,
                phase_channels[i][low_frequency] + self.delta,
            )
            filtered.append(phase_channels_0_t[i])
        return filtered

    def reconstruct_image(self, filtered_amplitude_channels, phase_channels):
        reconstructed_image = []
        for channel in range(3):
            amplitude = filtered_amplitude_channels[channel]
            phase = phase_channels[channel] - torch.pi
            fshift_filtered = amplitude * torch.exp(1j * phase)
            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_reconstructed = torch.abs(torch.fft.ifft2(f_ishift))
            img_reconstructed = torch.clip(img_reconstructed, 0, 255)
            reconstructed_image.append(img_reconstructed / 255)
        return torch.stack(reconstructed_image, dim=2)

    def amplitude_phase_exchange_torch(self, x, x_0_t):
        x_t = self.get_noised_x(x, self.forward_noise_steps)
        t = (torch.ones(x.size(0)) * self.forward_noise_steps).to(x.device)
        at = self.compute_alpha(t.long())
        et = self.diffusion(x_t, t)
        if self.is_imagenet:
            et, _ = torch.split(et, 3, dim=1)
        x = torch.clip((diff2clf((x_t - et * (1 - at).sqrt()) / at.sqrt()) * 255), 0, 255)
        x_0_t = torch.clip((diff2clf(x_0_t) * 255), 0, 255)

        batch, channel, height, width = x.shape
        new_x_0_t = torch.zeros(size=(batch, height, width, channel), device=self.device)
        for batch_idx in range(batch):
            amplitude_channels, phase_channels = self.compute_fft(x[batch_idx])
            amplitude_channels_0_t, phase_channels_0_t = self.compute_fft(x_0_t[batch_idx])
            amplitude_channels_0_t_exchange = self.low_pass_exchange(amplitude_channels, amplitude_channels_0_t)
            phase_channels_0_t_exchange = self.phase_low_pass_exchange(phase_channels, phase_channels_0_t)
            new_x_0_t[batch_idx] = self.reconstruct_image(amplitude_channels_0_t_exchange, phase_channels_0_t_exchange)
        new_x_0_t = new_x_0_t.float().permute(0, 3, 1, 2).to(self.device)
        return clf2diff(new_x_0_t)
