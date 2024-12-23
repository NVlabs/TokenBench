# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from copy import deepcopy
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import _compute_fid
from torchmetrics.metric import Metric
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

_STYLEGAN_TORCHSCRIPT_CKPT = (
    "./pretrained_ckpts/opensora/eval/fvd/styleganv/i3d_torchscript.pt"
)


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


class StyleGANvFeatureExtractor(Module):
    def __init__(self):
        super().__init__()
        self.model = torch.jit.load(_STYLEGAN_TORCHSCRIPT_CKPT)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        detector_kwargs = dict(
            rescale=False, resize=False, return_features=True
        )  # Return raw features before the softmax layer.
        return self.model(
            torch.stack([preprocess_single(video) for video in x]), **detector_kwargs
        )


class FVD(Metric):
    r"""
    Frechet Video Distance (FVD) is a metric to evaluate the quality of video generation models.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``videos`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor with. [0, 1]
    - ``real`` (:class:`~bool`): bool indicating if ``videos`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fvd`` (:class:`~torch.Tensor`): float scalar tensor with mean FVD value over samples

    Example:
        >>> import torch
        >>> torch.manual_seed(123)
        >>> NUMBER_OF_VIDEOS = 8
        >>> VIDEO_LENGTH = 50
        >>> CHANNEL = 3
        >>> SIZE = 64
        >>> videos1 = torch.zeros(NUMBER_OF_VIDEOS, CHANNEL, VIDEO_LENGTH, SIZE, SIZE, requires_grad=False).cuda()
        >>> videos2 = torch.ones(NUMBER_OF_VIDEOS, CHANNEL, VIDEO_LENGTH, SIZE, SIZE, requires_grad=False).cuda()
        >>> metric = FVD().cuda()
        >>> metric.update(videos1, real=True)
        >>> metric.update(videos2, real=False)
        >>> metric.compute()
        >>> tensor(232.7575)
    """
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    feature_extractor: Module
    extractor_option: str = "styleganv"

    def __init__(
        self,
        feature_extractor: Union[str, Module] = "styleganv",
        real_feature_stats: Optional[str] = None,
        reset_real_features: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(feature_extractor, str):
            # assert feature_extractor == 'styleganv', 'Only StyleGAN video is supported for now'
            if feature_extractor.lower() == "styleganv":
                self.feature_extractor = StyleGANvFeatureExtractor()
            else:
                raise NotImplementedError(
                    "Only StyleGANv and inceptionI3d are supported for now"
                )
            num_features = 400
        else:
            raise NotImplementedError()

        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.reset_real_features = reset_real_features
        self.reuse_real_stats = real_feature_stats is not None
        if self.reuse_real_stats:
            raise NotImplementedError()

    def update(self, videos: Tensor, real: bool) -> None:
        features = self.feature_extractor(videos)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += videos.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += videos.shape[0]

    def update_real_fake_batch(self, real_video: Tensor, fake_video: Tensor) -> None:
        self.update(real_video, real=True)
        self.update(fake_video, real=False)

    def compute_fvd_from_features(
        self, real_features: Tensor, fake_features: Tensor
    ) -> float:
        real_features = real_features.double()
        fake_features = fake_features.double()
        real_features_sum = real_features.sum(dim=0)
        real_features_cov_sum = real_features.t().mm(real_features)
        real_features_num_samples = real_features.shape[0]

        fake_features_sum = fake_features.sum(dim=0)
        fake_features_cov_sum = fake_features.t().mm(fake_features)
        fake_features_num_samples = fake_features.shape[0]

        if real_features_num_samples < 2 or fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (real_features_sum / real_features_num_samples).unsqueeze(0)
        mean_fake = (fake_features_sum / fake_features_num_samples).unsqueeze(0)

        cov_real_num = (
            real_features_cov_sum
            - real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (real_features_num_samples - 1)
        cov_fake_num = (
            fake_features_cov_sum
            - fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (fake_features_num_samples - 1)
        return (
            _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)
            .float()
            .item()
        )

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed
        """
        return self._plot(val, ax)
