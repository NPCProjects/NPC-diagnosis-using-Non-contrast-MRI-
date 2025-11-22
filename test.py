import argparse
import os
import time  # Import the time module
from medcam import medcam
import torch
import setproctitle
import numpy as np
from model.net import Model
from predict import evaluate_classification
from data.datasets_nii import NPC_loadall_test_nii, NPC_loadall_nii
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='liz', type=str)
parser.add_argument('--gpu', default='0,5', type=str)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

if __name__ == '__main__':
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    masks = [[False, True, True]]
    mask_name = [['T1 T2']]

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = r'path_to_test_data'
    resume = r'path_to_resume_checkpoint'
    num_cls = 2
    dataname = 'NPCCLASS'

    # Start time for test set loading and processing
    start_time = time.time()

    test_set = NPC_loadall_test_nii(transforms=test_transforms, root=datapath, label=None)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=8)

    # Time after loading dataset
    dataset_loading_time = time.time() - start_time
    print(f"Dataset loading time: {dataset_loading_time:.2f} seconds")

    model = Model(num_cls=num_cls).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # Load model checkpoint
    checkpoint_start_time = time.time()
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    checkpoint_loading_time = time.time() - checkpoint_start_time
    print(f"Model checkpoint loading time: {checkpoint_loading_time:.2f} seconds")

    # Start time for inference
    inference_start_time = time.time()

    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            med_model = medcam.inject(model, output_dir=f'attention_maps {mask_name[i]}', save_maps=True,
                                      label=None, layer=['module.cmf'])
            class_score = evaluate_classification(
                test_loader,
                model,
                dataname=dataname,
                feature_mask=mask,
                mask_name=mask_name[i])

            # Output prediction details
            for i in range(len(class_score[6])):
                name = class_score[6][i]['name']
                label = class_score[6][i]['label']
                pred = class_score[6][i]['pred']
                pred_probability = class_score[6][i]['pred_probability']
                print(name, label, pred, pred_probability)

    inference_time = time.time() - inference_start_time
    print(f"Inference time: {inference_time:.2f} seconds")

    # Total time for the entire testing procedure
    total_time = time.time() - start_time
    print(f"Total testing time: {total_time:.2f} seconds")
