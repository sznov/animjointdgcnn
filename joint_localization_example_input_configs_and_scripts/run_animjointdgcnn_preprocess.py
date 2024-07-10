import subprocess
import os
import shutil

CODE_DIR = None # TODO Specify
CODE_DIR = os.path.abspath(CODE_DIR)

if CODE_DIR is None:
    raise Exception('Please specify CODE_DIR')

# Parameters

FORCE_TARIG_PRIM_JOINT_POSTPROCESS = False
this_dir = os.getcwd()

# animjointdgcnn preprocessing
os.chdir(CODE_DIR)
os.chdir('_scripts') 
subprocess.run(['python', 'riginfo_2_json.py',
                '--input-directory', os.path.join(this_dir, '_generated', 'rig_info'),
                '--output-directory', os.path.join(this_dir, '_generated_animjointdgcnn', 'keypoints')])

subprocess.run(['python', 'obj_remesh_2_point_cloud_trimesh.py',
                '--input-directory', os.path.join(this_dir, '_generated', 'obj_remesh'),
                '--output-directory', os.path.join(this_dir, '_generated_animjointdgcnn', 'ply')])

subprocess.run(['python', 'split_ply_and_keypoints_by_subset.py',
                '--dataset-folder', os.path.join(this_dir, '_generated'),
                '--input-ply-directory', os.path.join(this_dir, '_generated_animjointdgcnn', 'ply'),
                '--input-keypoints-directory', os.path.join(this_dir, '_generated_animjointdgcnn', 'keypoints'),
                '--output-directory', os.path.join(this_dir, '_generated_animjointdgcnn', 'ply_and_keypoints_dataset')])
os.chdir('..')

os.chdir(this_dir)