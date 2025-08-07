## SAM-I2V: Upgrading SAM to Support Promptable Video Segmentation with Less than 0.2% Training Cost

**[Show Lab @ NUS](https://sites.google.com/view/showlab)**

[Haiyang Mei](https://mhaiyang.github.io/), [Pengyu Zhang](https://scholar.google.com/citations?user=GPGbDfQAAAAJ&hl=en), [Mike Zheng Shou](https://sites.google.com/view/showlab)

[[`Paper`](https://openaccess.thecvf.com/content/CVPR2025/html/Mei_SAM-I2V_Upgrading_SAM_to_Support_Promptable_Video_Segmentation_with_Less_CVPR_2025_paper.html)] [[`arXiv`](https://arxiv.org/abs/2506.01304)] [[`BibTeX`](#citation)]

- [Table of Contents](#0-SAM-I2V)
  * [1. Overview](#1-overview)
  * [2. Installation](#2-installation)
  * [3. Getting Started](#3-getting-started)
    + [3.1 Download Checkpoint](#31-download-checkpoint)
    + [3.2 Demo Use](#32-demo-use)
    + [3.3 Testing](#33-testing)
    + [3.4 Evaluation](#34-evaluation)
    + [3.5 Training](#35-training)
  * [4. Acknowledgements](#4-acknowledgements)
  * [5. Citation](#5-citation)
  * [6. License](#6-license)
  * [7. Contact](#7-contact)

### 1. Overview

**SAM-I2V** is a training-efficient method to upgrade the image-based SAM for promptable video segmentation. It achieves over **90%** of SAM 2’s performance while requiring only **0.2%** of its training cost.

<p align="center">
  <img src="assets/teaser.jpg?raw=true" width="300"/>
</p>

**SAM-I2V** takes an input video and extracts frame features via an image encoder enhanced by a temporal feature integrator to capture dynamic context. These features are processed by a memory associator and memory prompt generator to manage historical information and generate target prompts. A prompt encoder incorporates optional user inputs (e.g., masks, points, boxes). Finally, the mask decoder produces segmentation masks for each frame, enabling user-guided and memory-conditioned promptable video segmentation.

<p align="center">
  <img src="assets/pipeline.jpg?raw=true" width="600"/>
</p>

### 2. Installation

Our implementation uses `python==3.11`, `torch==2.5.0` and `torchvision==0.20.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM-I2V on a GPU machine using:

```bash
git clone https://github.com/showlab/SAM-I2V.git && cd SAM-I2V
conda create -n sam-i2v python=3.11
conda activate sam-i2v
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### 3. Getting Started

#### 3.1 Download Checkpoint

First, we need to download the SAM-I2V checkpoint. It can be downloaded from:

- [sam-i2v_8gpu.pt] 
[ [Google Drive](https://drive.google.com/drive/folders/1sRvmBf_QwCwyxppuB_5-9IMcAHcUK_K7?usp=sharing) ]
[ [OneDrive](https://1drv.ms/f/c/f6d9d790b8550d3f/Egr8_dMl2_RCgAtwiPvPEuoB_bsNfhNRwOmsMv32rjm0aA?e=JrkP6f) ]
[ [BaiduDisk](https://pan.baidu.com/s/1qJk0_QrFLoU0pL4hAZup7Q?pwd=itov) ]
- [sam-i2v_32gpu.pt]
[ [Google Drive](https://drive.google.com/drive/folders/1sRvmBf_QwCwyxppuB_5-9IMcAHcUK_K7?usp=sharing) ]
[ [OneDrive](https://1drv.ms/f/c/f6d9d790b8550d3f/Egr8_dMl2_RCgAtwiPvPEuoB_bsNfhNRwOmsMv32rjm0aA?e=JrkP6f) ]
[ [BaiduDisk](https://pan.baidu.com/s/1qJk0_QrFLoU0pL4hAZup7Q?pwd=itov) ]

**Both models were trained in one day using 24GB GPUs.** The first model (sam-i2v_8gpu.pt) was trained with 8 GPUs, while the second model (sam-i2v_32gpu.pt) was trained with 32 GPUs and offers better performance.

#### 3.2 Demo Use

SAM-I2V can be used in a few lines as follows for promptable video segmentation. Below provides a video predictor with APIs for example to add prompts and propagate masklets throughout a video. Same as SAM2, SAM-I2V supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from i2v.build_i2v import build_i2v_video_predictor

checkpoint = "./checkpoints/sam-i2v_32gpu.pt"
model_cfg = "./i2v/configs/i2v-infer.yaml"
predictor = build_i2v_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

#### 3.3 Testing

We provide instructions for testing on the SAV-Test dataset.

(a) Please refer to the [sav_dataset/README.md](sav_dataset/README.md) for detailed instructions on how to download and prepare the SAV-Test dataset before testing.

(b) Prepare the 'mask_info' for the ease of testing via:
```
python tools/save_gt_mask_multiprocess.py
```
Or you can directly download the preprocessed 'mask_info' [here](https://drive.google.com/drive/folders/1sRvmBf_QwCwyxppuB_5-9IMcAHcUK_K7?usp=sharing).

(c) Run the inference script
```
cd test_pvs
sh semi_infer.sh
```

#### 3.4 Evaluation

Run the evaluation script
```
sh semi_eval.sh
```

#### 3.5 Training

(a) Please refer to the [sav_dataset/README.md](sav_dataset/README.md) for detailed instructions on how to download and prepare the SAV-Train dataset. Totally 50,583 training videos ([train/txt/sav_train_list.txt](train/txt/sav_train_list.txt)).

(b) We follow SAM 2 to train the model on mixed video and image data. Download the [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/) and sample a subset of images, as the full dataset is too large to use in its entirety. We randomly sample 10k images ([train/txt/sa1b_10k_train_list.txt](train/txt/sa1b_10k_train_list.txt)) to train SAM-I2V.

(c) Download the SAM 1 model (i.e., [TinySAM](https://huggingface.co/xinghaochen/tinysam/tree/main)) to be upgraded and put it to `checkpoints/tinysam.pth`.

(d) Train the model:

- Single node with 8 GPUs:
```
sh train.sh
```

- Multi-node with each node has 8 GPUs (e.g., 4x8=32 GPUs):
```
sh multi_node_train_4_nodes.sh
```

### 4. Acknowledgements

Our implementation builds upon [SAM 2](https://github.com/facebookresearch/sam2) and reuses essential modules from its official codebase.

### 5. Citation

If you use SAM-I2V in your research, please use the following BibTeX entry.

```bibtex
@InProceedings{Mei_2025_CVPR,
    author    = {Mei, Haiyang and Zhang, Pengyu and Shou, Mike Zheng},
    title     = {SAM-I2V: Upgrading SAM to Support Promptable Video Segmentation with Less than 0.2% Training Cost},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3417-3426}
}
```

### 6. License

Please see `LICENSE`

### 7. Contact
E-Mail: Haiyang Mei (haiyang.mei@outlook.com)


**[⬆ back to top](#1-overview)**
