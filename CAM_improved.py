from pytorch_grad_cam import GradCAM, ClassifierOutputTarget, show_cam_on_image
from torchvision.models import resnet50
import torch
import os
import numpy as np
import cv2
import argparse
import logging
from tqdm import tqdm
from model.net import Model
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader
import setproctitle

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--user', default='user', type=str)
parser.add_argument('--gpu', default='1', type=str)
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# Main function
def main():
    # Set process title and GPU configuration
    setproctitle.setproctitle(f'{args.user}: Testing!')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Configuration
    masks = [[False, True, True]]
    mask_name = ['T1 T2']
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = r'/Dataset/NPC/processed/Test'
    resume_path = r'/Dataset/NPC/Model_Train_Results/One_Encoder/model_299.pth'
    num_cls = 2
    dataname = 'NPCCLASS'

    # Load test data and model
    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=8)

    model = Model(num_cls=num_cls).cuda()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Setup target layers and output path
    target_layers = [model.module.tsf]
    attention_map_output = f'./CAM/{mask_name[0]}'
    os.makedirs(attention_map_output, exist_ok=True)

    # Process test data
    for data in tqdm(test_loader):
        process_data(data, model, target_layers, attention_map_output)


# Function to process and save CAM results for each test image
def process_data(data, model, target_layers, attention_map_output):
    target = data[1]
    input_tensor = data[0].cuda()
    name = data[-1]

    # Specify the target for CAM generation
    targets = [ClassifierOutputTarget(1)]

    # Generate CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets).squeeze()
        original = data[0].numpy().squeeze()[2]  # Using the 3rd channel for visualization
        save_cam_images(grayscale_cam, original, name, attention_map_output)


# Function to save CAM images
def save_cam_images(grayscale_cam, original, name, attention_map_output):
    os.makedirs(os.path.join(attention_map_output, name[0]), exist_ok=True)
    for i in range(grayscale_cam.shape[2]):
        rgb_img = cv2.normalize(original[:, :, i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        rgb_img = np.float32(rgb_img) / 255
        visualization = show_cam_on_image(rgb_img, grayscale_cam[:, :, i], use_rgb=True, image_weight=0.8)
        cv2.imwrite(os.path.join(attention_map_output, name[0], f'{i}_cam.jpg'), visualization)


# Run the main function
if __name__ == '__main__':
    main()
