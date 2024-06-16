<h2 align="center" style="line-height: 50px;">
    <img src="https://yueyang1996.github.io/knobo/static/images/knobo_logo.png" style="vertical-align: middle;" width="50px"/>
    A Textbook Remedy for Domain Shifts <br>
    Knowledge Priors for Medical Image Analysis
</h2>


<h4 align="center">
  <a href="https://arxiv.org/abs/2405.14839">Paper</i></a> | <a href="https://yueyang1996.github.io/knobo/">Project Page</i></a>
</h4>


## CLIP Models
We release the two CLIP models we trained for X-ray and Skin Lesion images on huggingface.
* **WhyXrayCLIP** 🩻 : https://huggingface.co/yyupenn/whyxrayclip
* **WhyLesionCLIP** 👍🏽 : https://huggingface.co/yyupenn/whylesionclip


## Installation
After cloning the repo, you can install the required dependencies and download the data by running the following commands:
```bash
git clone https://github.com/YueYANG1996/KnoBo.git
cd KnoBo
sh setup.sh
```

## Quick Start
To get the results of KnoBo on X-ray datasets, you can run the following command:
```bash
python modules/cbm.py \
    --mode binary \
    --bottleneck PubMed \
    --number_of_features 150 \
    --add_prior True \
    --modality xray \
    --model_name whyxrayclip \
```
The output will be saved to `./data/results/`. You can change the `--modality` to `skin` and `--model_name` to `whylesionclip` to get the results on Skin Lesion datasets.


## Directories
* `data/` : Contains the data for all experiments.
  - `data/bottlenecks/` : Contains the concept bottleneck created using medical documents.
  - `data/datasets/` : Contains the splits for all datasets. You may need to download the images of each dataset from their original sources. Please refer to the [DATASETS.md](DATASETS.md) for more details.
  - `data/features/` : Contains the features extracted from different models.
  - `data/grounding_functions/` : Contains the grounding functions for each concept in the bottleneck.
  - `data/results/` : Contains the results of all experiments.

* `modules/` : Contains the scripts for all experiments.
  - [`modules/cbm.py`](modules/cbm.py) : Contains the script for the running linear-based models, including KnoBo, linear probing, and PCBM.
  - [`modules/extract_features.py`](modules/extract_features.py) : Contains the script for extracting image features using different models.
  - [`modules/train_grounding.py`](modules/train_grounding.py) : Contains the script for training the grounding functions for each concept in the bottleneck.
  - [`modules/end2end.py`](modules/end2end.py) : Contains the script for training the end-to-end model, including ViT and DenseNet.
  - [`modules/LSL.py`](modules/LSL.py) : Contains the script for fine-tuning CLIP with knowledge (Language-shaped Learning).
  - [`modules/models.py`](modules/models.py) : Contains the models used in the experiments.
  - [`modules/utils.py`](modules/utils.py) : Contains the utility functions.


## Extract Features
To extract features from the images using different models, you can run the following command:
```bash
python modules/extract_features.py \
    --dataset_name <NAME OF THE DATASET> \
    --model_name <NAME OF THE MODEL> \
    --image_dir <PATH TO THE IMAGE DIRECTORY> \
```
The supported models are listed [here](https://github.com/YueYANG1996/KnoBo/blob/e3e3171b74b6c8f42046676aa6c6ae21a034deba/modules/extract_features.py#L141). We provide a bash script [`extract_features.sh`](extract_features.sh) to extract features for all datasets using the two CLIP models we trained.


## Generate Bottlenecks from Medical Documents
Working on it...


## Train Grounding Functions
To train the grounding functions for each concept in the bottleneck, you can run the following command:
```bash
python modules/train_grounding.py \
    --modality <xray or skin> \
    --bottleneck <NAME OF THE BOTTLENECK> \
```
Each grounding function is a binary classifier that predicts whether the concept is present in the image. The output will be saved to `./data/grounding_functions/<modality>/<concept>/`.


## Baselines
* **Linear Probing**: `python modules/cbm.py --mode linear_probe --modality <xray or skin> --model_name <vision backbone>`.

* **PCBM-h**: `python modules/cbm.py --mode pcbm --bottleneck PubMed --number_of_features 150 --modality <xray or skin> --model_name <vision backbone>`.

* **End-to-End**: `python modules/end2end.py --modality <xray or skin> --model_name <vit or densenet>`.

* **LSL**: You need to first fine-tune the CLIP model with knowledge using the following command:
  ```bash
  python modules/LSL.py \
      --modality <xray or skin> \
      --clip_model_name <base model, e.g., whyxrayclip> \
      --bottleneck <NAME OF THE BOTTLENECK> \
      --image_dir <PATH TO THE IMAGE DIRECTORY> \
  ```
  Then, extract the features using the fine-tuned CLIP model and get the final results same as linear probing: `python modules/cbm.py --mode linear_probe --modality <xray or skin> --model_name <fine-tuned vision backbone>`. We provide the models we fine-tuned on PubMed in the `data/model_weights/` directory.


## Citation
Please cite our paper if you find our work useful!
```bibtex
@article{yang2024textbook,
      title={A Textbook Remedy for Domain Shifts: Knowledge Priors for Medical Image Analysis}, 
      author={Yue Yang and Mona Gandhi and Yufei Wang and Yifan Wu and Michael S. Yao and Chris Callison-Burch and James C. Gee and Mark Yatskar},
      journal={arXiv preprint arXiv:2405.14839},
      year={2024}
}
```