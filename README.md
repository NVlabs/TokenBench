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

### [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) | Huggingface Benchmark

https://github.com/user-attachments/assets/90e73525-12b5-4d41-a642-ee7570726f35

TokenBench is a comprehensive benchmark to standardize the evaluation for [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer), which covers a wide variety of domains including robotic manipulation, driving, egocentric, and web videos. It consists of high-resolution, long-duration videos, and is designed to evaluate the performance of video tokenizers. We resort to existing video datasets that are commonly used for various tasks, including [BDD100K](http://bdd-data.berkeley.edu/), [EgoExo-4D](https://docs.ego-exo4d-data.org/), [BridgeData V2](https://rail-berkeley.github.io/bridgedata/), and [Panda-70M](https://snap-research.github.io/Panda-70M/). This repo provides instructions on how to download and preprocess the videos for TokenBench.


## Instructions to build TokenBench

1. Download the datasets from the official websites:
* EgoExo4D: <a href="https://docs.ego-exo4d-data.org/" target="_blank">https://docs.ego-exo4d-data.org/</a>
* BridgeData V2: <a href="https://rail-berkeley.github.io/bridgedata/" target="_blank">https://rail-berkeley.github.io/bridgedata/</a>
* Panda70M: <a href="https://snap-research.github.io/Panda-70M/" target="_blank">https://snap-research.github.io/Panda-70M/</a>
* BDD100K: <a href="http://bdd-data.berkeley.edu/" target="_blank">http://bdd-data.berkeley.edu/</a>

2. Pick the videos as specified in the `video/list.txt` file.
3. Preprocess the videos using the script `video/preprocessing_script.py`.


## Discrete video tokenizer leaderboard

| Tokenizer      | Compression Ratio | Formulation | PSNR  | SSIM | rFVD  |
| -------------- | ----------------- | ----------- | ----- | ---- | ----- |
| Cosmos         | 4 × 8 × 8         | AE          | 36.97 | 0.92 | 7.12  |
| Cosmos         | 8 × 8 × 8         | AE          | 34.90 | 0.91 | 12.08 |
| Cosmos         | 8 × 16 × 16       | AE          | 29.71 | 0.87 | 45.08 |
| [CogVideoX](https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl_cogvideox)      | 4 × 8 × 8         | VAE         | 33.55 | 0.91 | 6.68  |
| [Omnitokenizer](https://github.com/FoundationVision/OmniTokenizer)  | 4 × 8 × 8         | VAE         | 30.08 | 0.82 | 34.79 |

## Continuous video tokenizer leaderboard

| Tokenizer      | Compression Ratio | Quantization | PSNR  | SSIM | rFVD  |
| -------------- | ----------------- | ------------ | ----- | ---- | ----- |
| Cosmos         | 4 × 8 × 8         | FSQ          | 34.89 | 0.884 | 20.11 |
| Cosmos         | 8 × 8 × 8         | FSQ          | 34.51 | 0.868 | 44.76 |
| Cosmos         | 8 × 16 × 16       | FSQ          | 33.52 | 0.823 | 118.22 |
| [Omnitokenizer](https://github.com/FoundationVision/OmniTokenizer)  | 4 × 8 × 8         | VQ           | 30.10 | 0.820 | 52.88 |


## Core contributors

Fitsum Reda, Jinwei Gu, Xian Liu, Songwei Ge, Ting-Chun Wang, Haoxiang Wang, Ming-Yu Liu
