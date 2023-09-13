import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
from lama_inpaint import load_img_to_array, inpaint_img_with_lama, save_array_to_img
import time

def setup_args(parser):
    parser.add_argument(
        "--input_img_glob", type=str, required=True,
        help="Path to input imgs",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    
if __name__ == "__main__":
    """Example usage:
    python process.py \
        --input_img_glob "FA_demo/FA1_dog*.png" \
        --output_dir results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    
    stime = time.time()
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_len = len(glob.glob(args.input_img_glob))
    i = 0
    
    for input_img in glob.glob(args.input_img_glob):
        img_stem = Path(input_img).stem
        mask_img = input_img[:-8] + "_msk.png"
        out_img = input_img[:-8] + "_out.png"
        mask_ps = sorted(glob.glob(mask_img))

        img = load_img_to_array(input_img)
        for mask_p in mask_ps:
            mask = load_img_to_array(mask_p)
            if len(mask.shape) == 3:
                mask = mask[:,:,0]
            img_inpainted_p = out_img
            img_inpainted = inpaint_img_with_lama(
                img, mask, args.lama_config, args.lama_ckpt, device=device)
            save_array_to_img(img_inpainted, img_inpainted_p)
            i += 1
            print("Finished {}/{}".format(i, full_len), end='\r')
            
    duration = time.time() - stime
    
    print(duration)