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

"""Code to compute different metrics for tokenizer evaluation.

Assumes the reconstructed and ground truth folders contain the same number
of videos with the filenames. Compute PSNR, SSIM, LPIPS, and FVD.

Example for MP4 videos:
    python3 -m token_bench.metrics_cli \
        --mode=all  \
        --ext=mp4 \
        --gtpath <folder to ground-truth videos> \
        --targetpath <folder to reconstruction videos>

For images, set the ext to "png" or "jpg". 
"""

import argparse
import os
from typing import Callable

import json
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from glob import glob

from mediapy import read_video
from token_bench.fvd import FVD

_FLOAT32_EPS = np.finfo(np.float32).eps
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gtpath",
    type=str,
    required=True,
    help="path/to/eval/videos/<dataset-name>/",
)
parser.add_argument(
    "--targetpath",
    type=str,
    default=None,
    help="path/to/eval/videos/<dataset-name>/<target-folder>",
)
parser.add_argument("--mode", type=str, choices=["psnr", "lpips", "fvd", "all"])
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--ext", type=str, default="mp4")
args = parser.parse_args()


def PSNR(input0: np.ndarray, input1: np.ndarray) -> float:
    """Compute PSNR between two videos or two images.

    Args:
        input0: The first video or image, of shape [..., H, W, C], of [0..255].
        input1: The second video or image, of shape [..., H, W, C], of [0..255].

    Returns:
        The PSNR value.
    """
    assert input0.shape == input1.shape, "inputs should have the same shape"
    mse = ((input0 - input1) ** 2).mean()
    psnr = 20 * np.log10(_UINT8_MAX_F / (np.sqrt(mse) + _FLOAT32_EPS))
    return psnr.item()


def SSIM(input0: np.ndarray, input1: np.ndarray) -> float:
    """Compute SSIM between two videos or two images.

    Args:
        input0: The first video or image, of shape [..., H, W, C], of [0..255].
        input1: The second video or image, of shape [..., H, W, C], of [0..255].

    Returns:
        The SSIM value.
    """
    assert input0.shape == input1.shape, "inputs should have the same shape"
    if input0.ndim == 3:
        input0, input1 = np.array([input0]), np.array([input1])
    ssim_values = []
    from concurrent.futures import ThreadPoolExecutor

    def compute_ssim(pair):
        one_image0, one_image1 = pair
        return ssim(
            one_image0,
            one_image1,
            data_range=_UINT8_MAX_F,
            multichannel=True,
            channel_axis=-1,
        )

    with ThreadPoolExecutor() as executor:
        ssim_values = list(executor.map(compute_ssim, zip(input0, input1)))
    return np.mean(ssim_values)


def LPIPS(input0: np.ndarray, input1: np.ndarray, loss_fn_vgg: Callable) -> float:
    """Compute LPIPS between two videos or two images.

    Args:
        input0: The first video or image, of shape [..., H, W, C], of [0..255].
        input1: The second video or image, of shape [..., H, W, C], of [0..255].
        loss_fn_vgg: The LPIPS loss function.
        device: The device to run the computation.

    Returns:
        The LPIPS value.
    """
    assert input0.shape == input1.shape, "inputs should have the same shape"
    if input0.ndim == 3:
        input0, input1 = np.array([input0]), np.array([input1])

    # computing LPIPS needs to normalize input to [-1,1].
    input0 = torch.from_numpy(2 * (input0 / _UINT8_MAX_F - 0.5)).to(torch.float32)
    input1 = torch.from_numpy(2 * (input1 / _UINT8_MAX_F - 0.5)).to(torch.float32)

    input0 = input0.permute(0, 3, 1, 2)  # N, C, H, W
    input1 = input1.permute(0, 3, 1, 2)  # N, C, H, W

    # average LPIPS over all frames
    results = []
    for one_input0, one_input1 in zip(input0, input1):
        fm0 = one_input0.unsqueeze(0).to(args.device)
        fm1 = one_input1.unsqueeze(0).to(args.device)
        res = loss_fn_vgg(fm0, fm1).item()
        results.append(res)

    return np.mean(results)


def main_psnr_ssim() -> None:
    vfiles0 = sorted(list(set(glob(str(f"{args.gtpath}/*.{args.ext}")))))
    vfiles1 = sorted(list(set(glob(str(f"{args.targetpath}/*.{args.ext}")))))

    psnr_filename = f"{args.targetpath}/psnr.csv"
    ssim_filename = f"{args.targetpath}/ssim.csv"
    if os.path.exists(psnr_filename) and os.path.exists(ssim_filename):
        print(f"{psnr_filename} already exists. Recomputing ...")
        print(f"{ssim_filename} already exists. Recomputing ...")

    assert len(vfiles0) == len(vfiles1), "number of media files must match"

    print(f"Calculating PSNR on  {len(vfiles0)} pairs ...")
    psnr_values, ssim_values = list(), list()
    for input0_file, input1_file in tqdm(zip(vfiles0, vfiles1)):
        assert (
            input0_file.split("/")[-1] == input1_file.split("/")[-1]
        ), "file names must match"
        input0 = read_video(input0_file).astype(np.float32)
        input1 = read_video(input1_file).astype(np.float32)

        name = input0_file.split("/")[-1]
        psnr_value = PSNR(input0, input1)
        ssim_value = SSIM(input0, input1)

        psnr_values.append([name, psnr_value])
        ssim_values.append([name, ssim_value])
        print(f"{name} PSNR: {psnr_value}, SSIM: {ssim_value}")

    print(f"mean PSNR: {np.mean([el[-1] for el in psnr_values])}")
    print(f"mean SSIM: {np.mean([el[-1] for el in ssim_values])}")

    with open(psnr_filename, "w") as fw:
        json.dump(psnr_values, fw)

    with open(ssim_filename, "w") as fw:
        json.dump(ssim_values, fw)


def main_lpips() -> None:
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(args.device).eval()

    vfiles0 = sorted(list(set(glob(str(f"{args.gtpath}/*.{args.ext}")))))
    vfiles1 = sorted(list(set(glob(str(f"{args.targetpath}/*.{args.ext}")))))

    lpips_filename = f"{args.targetpath}/lpips.csv"
    if os.path.exists(lpips_filename):
        print(f"{lpips_filename} already exists. Recomputing ...")

    assert len(vfiles0) == len(vfiles1), "video files not match"

    print(f"Calculating LPIPS on {len(vfiles1)} pairs ...")
    lpips_values = list()
    for i in tqdm(range(len(vfiles0))):
        vid0 = read_video(vfiles0[i])
        vid1 = read_video(vfiles1[i])

        name = vfiles0[i].split("/")[-1]
        lpips_value = LPIPS(vid0, vid1, loss_fn_vgg)
        lpips_values.append([name, lpips_value])

    print(f"mean LPIPS: {np.mean([el[-1] for el in lpips_values])}")

    with open(lpips_filename, "w") as fw:
        json.dump(lpips_values, fw)


def main_fvd(max_n_frame: int = 300) -> None:
    fvd_model = FVD("styleganv").to(args.device).double()

    vfiles0 = sorted(list(set(glob(str(f"{args.gtpath}/*.{args.ext}")))))
    vfiles1 = sorted(list(set(glob(str(f"{args.targetpath}/*.{args.ext}")))))
    fvd_filename = f"{args.targetpath}/fvd.csv"
    if os.path.exists(fvd_filename):
        print(f"{fvd_filename} already exists. Recomputing ...")

    fvd_model.reset()

    assert len(vfiles0) == len(vfiles1), "video files not match"

    print(f"Calculating FVD on {len(vfiles1)} pairs ...")
    for i in tqdm(range(len(vfiles0))):
        vid0 = read_video(vfiles0[i])[:max_n_frame]
        vid1 = read_video(vfiles1[i])[:max_n_frame]

        if vid0.ndim == 3:
            vid0, vid1 = np.array([vid0]), np.array([vid1])

        vid0 = torch.from_numpy(vid0 / 255.0).to(args.device).float()
        vid1 = torch.from_numpy(vid1 / 255.0).to(args.device).float()
        vid0 = vid0.permute(3, 0, 1, 2).unsqueeze(0)
        vid1 = vid1.permute(3, 0, 1, 2).unsqueeze(0)

        fvd_model.update_real_fake_batch(vid0, vid1)

    fvd = fvd_model.compute().item()
    print(f"FVD: {fvd}")

    with open(fvd_filename, "w") as fw:
        json.dump([fvd], fw)


if __name__ == "__main__":
    if args.mode.lower() == "psnr" or args.mode.lower() == "all":
        main_psnr_ssim()

    if args.mode.lower() == "lpips" or args.mode.lower() == "all":
        main_lpips()

    if args.mode.lower() == "fvd" or args.mode.lower() == "all":
        main_fvd()
