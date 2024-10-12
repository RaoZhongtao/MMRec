# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    
    config_dict = {
        'gpu_id': 1,
    }

    args, _ = parser.parse_known_args()
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, ckpt_dir=args.ckpt_dir, save_model=True, mg=args.mg, mode=args.mode)


