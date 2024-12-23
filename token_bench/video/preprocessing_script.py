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

import os
import imageio
import numpy as np
from glob import glob
import mediapy as media


def resize_video(video: np.ndarray, short_size: int = None) -> np.ndarray:
    """Resizes a video to have the short side of `short_size`."""
    if short_size is None:
        return video
    height, width = video.shape[-3:-1]
    if height <= width:
        height_new, width_new = short_size, int(width * short_size / height + 0.5)
    else:
        height_new, width_new = int(height * short_size / width + 0.5), short_size
    return media.resize_video(video, shape=(height_new, width_new))


raw_video_dir = "/root/dataset"

input_pattern = raw_video_dir + "/%s/*.%s"
benchmarks = ["bdd_100", "egoexo4D", "panda", "bridgev2"]
exts = ["mov", "mp4", "mp4", "mp4"]
for benchmark, ext in zip(benchmarks, exts):
    input_files = sorted(glob(str(input_pattern % (benchmark, ext))))
    print(
        "Processing", len(input_files), "videos for", input_pattern % (benchmark, ext)
    )
    for jdx, video_file in enumerate(input_files):
        video_reader = imageio.get_reader(video_file, ext)
        video_frames = []
        for frame in video_reader:
            video_frames.append(frame)

        input_video, meta_data = np.array(video_frames), video_reader.get_meta_data()

        video_fps = meta_data["fps"]
        video_duration = meta_data["duration"]
        input_video = np.array(input_video)
        T, H, W, C = input_video.shape
        print("loaded", video_file, "with", (T, H, W))
        # clip the videos to 10 seconds if they are longer
        num_frame_thres = max(int(np.ceil(video_fps * 10)), 300)
        output_video = (
            input_video[:num_frame_thres] if T > num_frame_thres else input_video
        )
        del input_video
        # resize the videos to 1080p if needed
        output_video = (
            resize_video(output_video, 1080) if min(H, W) > 1080 else output_video
        )
        print((T, H, W, C), "resized to", output_video.shape)
        video_file_tokenbench = video_file.replace(
            f"/dataset/{benchmark}/", f"/dataset/tokenbench/{benchmark}_"
        ).replace(f".{ext}", ".mp4")
        os.makedirs(os.path.dirname(video_file_tokenbench), exist_ok=True)
        print("writing to ...", video_file_tokenbench)
        media.write_video(video_file_tokenbench, output_video, fps=video_fps)
        del output_video
