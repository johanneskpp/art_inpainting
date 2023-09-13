# Art Image Inpainting

This project is structured in subprojects. Every subproject has an own requirements.txt file and all neccesary jupyter notebooks and scripts.
Larger folders are zipped stored with Git LFS. Datasets are not contained.

## Creating datasets

This subproject contains code to generate an own defect dataset.
The base dataset is the artbench10-dataset: https://github.com/liaopeiyuan/artbench

22 defects have been handdrawn and augmented with the albumentations library. The resulting 2300 defect-masks have been applied to the artbench images to create a defected dataset.

### smaller dataset

For later purposes the full dataset mostly was too big. Because of that a smaller dataset with 2000 images for training, 570 images for validation and 285 images for testing has been created.
A part of the dataset has been resturctured and uploaded to HuggingfaceHub for retraining of stable-diffusion for inpainting: https://huggingface.co/datasets/johanneskpp/art_defect_inpainting

### structure

```
dataset
│
└───img
│   │
│   └───train
│   │   │   a.png
│   │   │   b.png
│   │   │   ...
│   └───val
│   │   │   c.png
│   │   │   d.png
│   │   │   ...
│   └───test
│       │   e.png
│       │   f.png
│       │   ...
└───msk
│   │
│   └───train
│   │   │   a.png
│   │   │   b.png
│   │   │   ...
│   └───val
│   │   │   c.png
│   │   │   d.png
│   │   │   ...
│   └───test
│       │   e.png
│       │   f.png
│       │   ...
└───ori
│   │
│   └───train
│   │   │   a.png
│   │   │   b.png
│   │   │   ...
│   └───val
│   │   │   c.png
│   │   │   d.png
│   │   │   ...
│   └───test
│       │   e.png
│       │   f.png
│       │   ...
```

## Defect Segmentation

This subproject contains code to train an own defect segmentation.
The code should be runnable on Google Colab.

The folder contains the config .py-files for starting training with mmsegmentation.

The resulting models are stored in the content_{modelname}/-folders.
The inference-results are stored in the res_{modelname}/-folders.

## Inpainting Evaluation

This subproject contains code to do inference with different inpainting methods and evaluate their results.
Outputs are stored in results/res_{methodname}/test/-folders.
Evaluation-Logs are stored in result_logs/results_{methodname}.txt

The baseline for the evaluation script is the following git-repository: https://github.com/SayedNadim/Inpainting-Evaluation-Metrics
I had to adjust the code a little bit because it was not running as expected.

The inference of the lama model was done with this repository: https://github.com/geekyutao/inpaint-anything
I have written a script for inference on a folder which is called like this:
```shell
python lama_inference/process.py \
        --input_img_glob "dataset/img/*.png" \
        --output_dir results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
```
You need to install the inpaint-anything-repository as described in their ReadMe.

## Train Inpainting

This subproject contains code to finetune a stable diffusion 2.0 inpainting model. The code is based on the following script: https://github.com/sshh12/terrain-diffusion/blob/main/scripts/train_text_to_image_lora_sd2_inpaint.py

The evaluation is based on the diffusers-library: https://github.com/huggingface/diffusers/tree/main

The model-checkpoints are stored in the output/-folder.

