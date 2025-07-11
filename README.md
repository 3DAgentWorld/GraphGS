# Graph-Guided Scene Reconstruction from Images with 3D Gaussian Splatting

**Chong Cheng\*, Gaochao Song\*, Yiyang Yao, Qinzheng Zhou, Gangjian Zhang, Hao Wang**  
\* indicates equal contribution  
[Webpage](https://3dagentworld.github.io/graphgs/) | [ArXiv](https://arxiv.org/abs/2502.17377/)

## Abstract
*This paper investigates an open research challenge of reconstructing high-quality, large 3D open scenes from images. It is observed existing methods have various limitations, such as requiring precise camera poses for input and dense viewpoints for supervision. To perform effective and efficient 3D scene reconstruction, we propose a novel graph-guided 3D scene reconstruction framework, GraphGS. Specifically, given a set of images captured by RGB cameras on a scene, we first design a spatial prior-based scene structure estimation method. This is then used to create a camera graph that includes information about the camera topology. Further, we propose to apply the graph-guided multi-view consistency constraint and adaptive sampling strategy to the 3D Gaussian Splatting optimization process. This greatly alleviates the issue of Gaussian points overfitting to specific sparse viewpoints and expedites the 3D reconstruction process. We demonstrate GraphGS achieves high-fidelity 3D reconstruction from images, which presents state-of-the-art performance through quantitative and qualitative evaluation across multiple datasets.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{cheng2025graphguidedscenereconstructionimages,
  title={Graph-Guided Scene Reconstruction from Images with 3D Gaussian Splatting},
  author={Chong Cheng and Gaochao Song and Yiyang Yao and Qinzheng Zhou and Gangjian Zhang and Hao Wang},
  year={2025},
  eprint={2502.17377},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.17377}
}</code></pre>
  </div>
</section>

---

## 🚀 Getting Started

### Cloning the Repository
The repository contains submodules, thus please check it out with 
```bash
# SSH
git clone git@github.com:3DAgentWorld/GraphGS.git --recursive

# HTTPS
git clone https://github.com/3DAgentWorld/GraphGS.git --recursive
```

### Environment Setup
We recommend using Conda for managing dependencies.
```bash
conda env create --file environment.yml
conda activate graphgs
```

Please install [COLMAP](https://colmap.github.io/install.html) manually according to the official guide.  
**Note:** We recommend installing a version of COLMAP earlier than 3.9.1, as newer versions may lead to degraded results (the cause is currently unknown).

Refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for additional environment dependencies.

---

## 📁 Dataset Preparation

We support multiple dataset formats:

- Waymo, KITTI (Refer to [StreetSurf Docs](https://github.com/PJLab-ADG/neuralsim/blob/main/docs/data/autonomous_driving.md))
- NeRF Synthetic
- Standard COLMAP format

**Custom dataset format:**
```
path/to/your/dataset
├── dataset
│   ├── input
│   │   ├── img_01.png
│   │   ├── img_02.png
│   │   └── ...
```

---

## Sample Dataset

We provide a demo dataset from Waymo:  
**[Download Waymo Sample](https://drive.google.com/file/d/1h_GTsj3nFSWAdblQmC_4WQuNJ1jmlme-/view?usp=sharing)**


---

## Scene Structure Estimation

We provide a spatial prior-based scene structure estimation module that can work in two modes: **with** or **without** coarse camera poses.

### 1. Without Coarse Camera Pose
You can run the structure estimation module even without initial poses. This may result in slightly slower processing.
```bash
python run_modifyMatch.py -s "path/to/your/dataset" --match_mode O_neighborJump --threads 32 --gpu_id 0
```

#### Command Arguments:
- `--match_mode`: Set to `O_neighborJump` to enable CNNP.
- `--arg_r`: Neighbor number to match (radius), default is 5.
- `--arg_k`: Jump interval, default is 10.
- `--arg_w`: Concentric annulus width, default is 1.

### 2. With Coarse Camera Pose
The pipeline is fastest when a coarse spatial prior is available. The camera poses don’t have to be highly accurate — just roughly correct in planar distribution.

To extract poses in standard format:
```bash
python generate_coarse_cam.py --path "path/to/your/dataset"
```

If you don’t have camera poses, we recommend using [DUSt3R](https://github.com/naver/dust3r) to obtain an initial spatial prior. Save the output as `camera_info_opencv.json` — only the inference stage is needed, as we require only a coarse planar layout.

Expected directory structure:
```
path/to/your/dataset
├── dataset
│   ├── camera_info_opencv.json
│   ├── input
│   │   ├── img_01.png
│   │   ├── img_02.png
│   │   └── ...
```

Then run the script:
```bash
python run_modifyMatch.py -s "path/to/your/dataset" --match_mode Q --threads 32 --gpu_id 0
```

#### Command Arguments:
- `--match_mode`: Use `C` for CNNP or `Q` for CNNP + QF.
- `--arg_r`, `--arg_k`, `--arg_w`: Same as above.

### Special Cases

- If COLMAP’s bundle adjustment fails, try changing `--camera` to `SIMPLE_PINHOLE`.
- You may also modify the COLMAP parameters directly in `run_modifyMatch.py`.

### 3. Using COLMAP Default Matching

#### Exhaustive Matching (Default of COLMAP)
```bash
python run_modifyMatch.py -s "path/to/your/dataset"  --match_mode E --threads 32 --gpu_id 0
```

#### Vocab Tree Matching
Download the vocab tree from [COLMAP site](https://demuc.de/colmap/) and place it at `vocab_tree/vocab_tree_flickr100K_words32K.bin`. Then run:
```bash
python run_modifyMatch.py -s "path/to/your/dataset"  --match_mode T --threads 32 --gpu_id 0
```

---

## 🔧 Graph-Guided Optimization

To start training using the optimized graph-structured data:
```bash
python train.py -s "path/to/your/dataset/matche" -m “path/to/your/output”
```

<details>
<summary><strong>Command Line Arguments for train.py</strong></summary>

- `--source_path` / `-s`: Path to the source directory containing COLMAP or NeRF data.
- `--model_path` / `-m`: Path where the trained model should be stored (default: `output/<random>`).
- `--eval`: Add this flag to split training and test sets for evaluation.
- `--use_consistency`: Whether to use multi-view consistency loss. Default: `True`.
- `--consistency_weight`: Weight for the multi-view consistency loss. Default: `0.05`.
- `--lambda_dynamic_sampling`: Sampling adjustment rate. Default: `0.045`.
- `--importance_threshold_low`: Threshold for low-importance samples. Default: `0.7`.
- `--use_dynamic_sampling`: Whether to use dynamic importance sampling. Default: `True`.

</details>

---

## Evaluation

Generate renderings:
```bash
python render.py -m <path/to/trained/model>
```

Compute evaluation metrics:
```bash
python metrics.py -m <path/to/trained/model> -t "train"  # Use -t test for test set evaluation
```

Viewer instructions can be found in the [3DGS Viewer Guide](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

---

## 🙏 Acknowledgements

We thank the authors of 3DGS, COLMAP, StreetSurf, and DUSt3R for their inspiring contributions.
