<!-- # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License. -->

# TokenBench

### [Cosmos-Tokenizer Code](https://github.com/NVIDIA/Cosmos-Tokenizer) | [Technical Report](https://research.nvidia.com/labs/dir/cosmos-tokenizer/)


https://github.com/user-attachments/assets/72536cfc-5cb5-4b48-88fa-b06f3c8c4495


TokenBench is a comprehensive benchmark to standardize the evaluation for [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer), which covers a wide variety of domains including robotic manipulation, driving, egocentric, and web videos. It consists of high-resolution, long-duration videos, and is designed to evaluate the performance of video tokenizers. We resort to existing video datasets that are commonly used for various tasks, including [BDD100K](http://bdd-data.berkeley.edu/), [EgoExo-4D](https://docs.ego-exo4d-data.org/), [BridgeData V2](https://rail-berkeley.github.io/bridgedata/), and [Panda-70M](https://snap-research.github.io/Panda-70M/). This repo provides instructions on how to download and preprocess the videos for TokenBench.

## Installation
- Clone the source code
```
git clone https://github.com/NVlabs/TokenBench.git
cd TokenBench
```
- Install via pip
```
pip3 install -r requirements.txt
apt-get install -y ffmpeg
```

Preferably, build a docker image using the provided Dockerfile
```
docker build -t token-bench -f Dockerfile .

# You can run the container as:
docker run --gpus all -it --rm -v /home/${USER}:/home/${USER} \
    --workdir ${PWD} token-bench /bin/bash
```

## Download StyleGAN Checkpoints from Hugging Face


You can use this snippet to download StyleGAN checkpoints from [huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0):
```python
from huggingface_hub import login, snapshot_download
import os

login(token="<YOUR-HF-TOKEN>", add_to_git_credential=True)
model_name="LanguageBind/Open-Sora-Plan-v1.0.0"
local_dir = "pretrained_ckpts/" + model_name
os.makedirs(local_dir, exist_ok=True)
print(f"downloading `{model_name}` ...")
snapshot_download(repo_id=f"{model_name}", local_dir=local_dir)
```

Under `pretrained_ckpts/Open-Sora-Plan-v1.0.0`, you can find the StyleGAN checkpoints required for FVD metrics.
```bash 
├── opensora/eval/fvd/styleganv/
│   ├── fvd.py
│   ├── i3d_torchscript.pt
```

## Instructions to build TokenBench

1. Download the datasets from the official websites:
* EgoExo4D: <a href="https://docs.ego-exo4d-data.org/" target="_blank">https://docs.ego-exo4d-data.org/</a>
* BridgeData V2: <a href="https://rail-berkeley.github.io/bridgedata/" target="_blank">https://rail-berkeley.github.io/bridgedata/</a>
* Panda70M: <a href="https://snap-research.github.io/Panda-70M/" target="_blank">https://snap-research.github.io/Panda-70M/</a>
* BDD100K: <a href="http://bdd-data.berkeley.edu/" target="_blank">http://bdd-data.berkeley.edu/</a>

2. Pick the videos as specified in the `token_bench/video/list.txt` file.
3. Preprocess the videos using the script `token_bench/video/preprocessing_script.py`.


## Evaluation on the token-bench

We provide the basic scripts to compute the common evaluation metrics for video tokenizer reonctruction, including `PSNR`, `SSIM`, and `lpips`. Use the code to compute metrics between two folders as below

```
python3 -m token_bench.metrics_cli --mode=lpips \
        --gtpath <ground truth folder> \
        --targetpath <reconstruction folder>
```

## Continuous video tokenizer leaderboard

| Tokenizer      | Compression Ratio (T x H x W) | Formulation | PSNR  | SSIM | rFVD  |
| -------------- | ----------------- | ----------- | ----- | ---- | ----- |
| [CogVideoX](https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl_cogvideox)      | 4 × 8 × 8         | VAE         | 33.149 | 0.908 | 6.970  |
| [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer)  | 4 × 8 × 8         | VAE         | 29.705 | 0.830 | 35.867 |
| Cosmos-CV         | 4 × 8 × 8         | AE          | 37.270 | 0.928 | 6.849  |
| Cosmos-CV         | 8 × 8 × 8         | AE          | 36.856 | 0.917 | 11.624 |
| Cosmos-CV         | 8 × 16 × 16       | AE          | 35.158 | 0.875 | 43.085 |

## Discrete video tokenizer leaderboard

| Tokenizer      | Compression Ratio (T x H x W) | Quantization | PSNR  | SSIM | rFVD  |
| -------------- | ----------------- | ------------ | ----- | ---- | ----- |
| [VideoGPT](https://github.com/wilson1yan/VideoGPT)         | 4 × 4 × 4         | VQ          | 35.119 | 0.914 | 13.855 |
| [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer)  | 4 × 8 × 8         | VQ           | 30.152 | 0.827 | 53.553 |
| Cosmos-DV         | 4 × 8 × 8         | FSQ          | 35.137 | 0.887 | 19.672 |
| Cosmos-DV         | 8 × 8 × 8         | FSQ          | 34.746 | 0.872 | 43.865 |
| Cosmos-DV         | 8 × 16 × 16       | FSQ          | 33.718 | 0.828 | 113.481 |


## Core contributors

Fitsum Reda, Jinwei Gu, Xian Liu, Songwei Ge, Ting-Chun Wang, Haoxiang Wang, Ming-Yu Liu
