# Split PLY and keypoints by subset

import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser(description='Split PLY and keypoints by subset')
parser.add_argument('--dataset-folder', type=str, default='dataset', help='Dataset folder')
parser.add_argument('--input-ply-directory', type=str, default='ply', help='Input PLY directory')
parser.add_argument('--input-keypoints-directory', type=str, default='keypoints', help='Input keypoints directory')
parser.add_argument('--output-directory', type=str, default='output', help='Output directory')
args = parser.parse_args()

# read train_final, val_final, and test_final
with open(os.path.join(args.dataset_folder, 'train_final.txt'), 'r') as f:
    train_final = f.readlines()
train_final = [line.strip() for line in train_final]
with open(os.path.join(args.dataset_folder, 'val_final.txt'), 'r') as f:
    val_final = f.readlines()
val_final = [line.strip() for line in val_final]
with open(os.path.join(args.dataset_folder, 'test_final.txt'), 'r') as f:
    test_final = f.readlines()
test_final = [line.strip() for line in test_final]

# create output directories
os.makedirs(args.output_directory, exist_ok=True)

for input_ply_path in glob.glob(os.path.join(args.input_ply_directory, '*.ply')):
    model_name = os.path.splitext(os.path.basename(input_ply_path))[0]
    if model_name.endswith('_remesh'):
        model_name = model_name[:-len('_remesh')]
    for (subset_name, subset) in zip(['train', 'val', 'test'], [train_final, val_final, test_final]):
        if model_name in subset:
            output_ply_path = os.path.join(args.output_directory, subset_name, model_name + '.ply')
            output_keypoints_path = os.path.join(args.output_directory, subset_name, model_name + '.json')

            # copy PLY
            os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
            shutil.copy(input_ply_path, output_ply_path)

            # copy keypoints
            input_keypoints_path = os.path.join(args.input_keypoints_directory, model_name + '.json')
            shutil.copy(input_keypoints_path, output_keypoints_path)
            