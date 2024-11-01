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

# token-bench



## Prepare the video token-bench

1. Download the datasets from officials sites:
* EgoExo4D: https://docs.ego-exo4d-data.org/
* BridgeData V2: https://rail-berkeley.github.io/bridgedata/
* Panda70M: https://snap-research.github.io/Panda-70M/
* BDD100K: http://bdd-data.berkeley.edu/

2. Pick the videos as listed in the `video/list.txt` file.
3. Preprocess the videos using the script `video/preprocessing_script.py`.

## Citation
If you find this useful in your projects, please acknowledge it
appropriately by citing:


```
@misc{token-bench,
  title = {TokenBench: A Video Tokenizer Evaluation Dataset},
  author = {Songwei Ge and Xian Liu and Fitsum Reda and Jinwei Gu and Haoxiang Wang and Ming-Yu Liu},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nvidia/token-bench}}
}
```
