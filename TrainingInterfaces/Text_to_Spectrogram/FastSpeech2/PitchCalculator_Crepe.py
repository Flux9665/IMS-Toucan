# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

from re import X
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
import torchcrepe

from Utility.utils import pad_list


class Crepe(torch.nn.Module):
    """
    F0 estimation with Crepe: https://github.com/maxrmorrison/torchcrepe
    """

    def __init__(self, fs=16000, n_fft=1024, hop_length=256, f0min=40, f0max=400, use_token_averaged_f0=True,
                 use_continuous_f0=True, use_log_f0=True, reduction_factor=1):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_period = 1000 * hop_length / fs
        self.f0min = f0min
        self.f0max = f0max
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0
        if use_token_averaged_f0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

    def output_size(self):
        return 1

    def get_parameters(self):
        return dict(fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, f0min=self.f0min, f0max=self.f0max,
                    use_token_averaged_f0=self.use_token_averaged_f0, use_continuous_f0=self.use_continuous_f0, use_log_f0=self.use_log_f0,
                    reduction_factor=self.reduction_factor)

    def forward(self, input_waves, input_waves_lengths=None, feats_lengths=None, durations=None,
                durations_lengths=None, norm_by_average=True):
        # If not provided, we assume that the inputs have the same length
        if input_waves_lengths is None:
            input_waves_lengths = (input_waves.new_ones(input_waves.shape[0], dtype=torch.long) * input_waves.shape[1])

        # F0 extraction
        pitch = [self._calculate_f0(x[:xl]) for x, xl in zip(input_waves, input_waves_lengths)]
        
        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            pitch = [self._adjust_num_frames(p, fl).view(-1) for p, fl in zip(pitch, feats_lengths)]

        # (Optional): Average by duration to calculate token-wise f0
        if self.use_token_averaged_f0:
            pitch = [self._average_by_duration(p, d).view(-1) for p, d in zip(pitch, durations)]
            pitch_lengths = durations_lengths
        else:
            pitch_lengths = input_waves.new_tensor([len(p) for p in pitch], dtype=torch.long)

        # Padding
        pitch = pad_list(pitch, 0.0)

        # Return with the shape (B, T, 1)
        if norm_by_average:
            average = pitch[0][pitch[0] != 0.0].mean()
            pitch = pitch / average
        return pitch.unsqueeze(-1), pitch_lengths

    def _calculate_f0(self, input):
        x = input.cpu().numpy().astype(np.double)
        x = input.unsqueeze(0)
        f0 = torchcrepe.predict(x,
                           self.fs,
                           self.hop_length,
                           self.f0min,
                           self.f0max,
                           model='full',
                           decoder=torchcrepe.decode.weighted_argmax
                           # batch_size=batch_size,
                           # device=device
                           ).squeeze(0).cpu().numpy().astype(np.double)
        f0 = np.nan_to_num(f0, nan=0.0)
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return input.new_tensor(f0.reshape(-1), dtype=torch.float)

    @staticmethod
    def _adjust_num_frames(x, num_frames):
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x

    @staticmethod
    def _convert_to_continuous_f0(f0: np.array):
        if (f0 == 0).all():
            return f0

        # padding start and end of f0 sequence
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
        f0 = interp_fn(np.arange(0, f0.shape[0]))

        return f0

    def _average_by_duration(self, x, d):
        assert 0 <= len(x) - d.sum() < self.reduction_factor
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].masked_select(x[start:end].gt(0.0)).mean(dim=0) if len(x[start:end].masked_select(x[start:end].gt(0.0))) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])]
        return torch.stack(x_avg)
