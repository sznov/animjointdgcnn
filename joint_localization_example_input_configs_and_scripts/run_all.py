import subprocess
import os
import shutil

CODE_DIR = None # TODO Specify
CODE_DIR = os.path.abspath(CODE_DIR)

if CODE_DIR is None:
    raise Exception('Please specify CODE_DIR')

# Parameters
DO_PREPROCESS = True
DO_TARIG_PRIM_JOINT_TRAIN = False
FORCE_TARIG_PRIM_JOINT_POSTPROCESS = False
this_dir = os.getcwd()

# Copy necessary input files to _generated
if DO_PREPROCESS:
    os.makedirs(os.path.join(this_dir, '_generated'), exist_ok=True)
    shutil.copyfile(os.path.join(this_dir, '_input', 'train_final.txt'), os.path.join(this_dir, '_generated', 'train_final.txt'))
    shutil.copyfile(os.path.join(this_dir, '_input', 'test_final.txt'), os.path.join(this_dir, '_generated', 'test_final.txt'))
    shutil.copyfile(os.path.join(this_dir, '_input', 'val_final.txt'), os.path.join(this_dir, '_generated', 'val_final.txt'))
    shutil.copyfile(os.path.join(this_dir, '_input', 'keypoint_names.txt'), os.path.join(this_dir, '_generated', 'keypoint_names.txt'))


# Common and TARig preprocessing
if DO_PREPROCESS:
    os.chdir(CODE_DIR)
    os.chdir('_common_preprocessing_scripts')

    subprocess.run(['python', '00_fix_leaf_joints_and_randomize_pose_and_save_to_blend.py', 
                    '--input-directory', os.path.join(this_dir, '_input', 'models'),
                    '--output-directory', os.path.join(this_dir, '_generated', 'blend'),
                    '--pose-preprocess-params-json', os.path.join(this_dir, '_input', 'pose_preprocess_params.json'),
                    '--keypoint-names-file', os.path.join(this_dir, '_input', 'keypoint_names.txt'),
                    '--rotation-range-generator-path', os.path.join(this_dir, '_input', 'rotation_range_generator.py')])

    subprocess.run(['python', '01_parallel_blend_to_obj_and_riginfo_normalized.py',
                    '--input-directory', os.path.join(this_dir, '_generated', 'blend'),
                    '--output-directory', os.path.join(this_dir, '_generated'),
                    '--keypoint-names-file', os.path.join(this_dir, '_input', 'keypoint_names.txt'),
                    '--keypoint-extraction-params-json', os.path.join(this_dir, '_input', 'keypoint_extraction_params.json')])

    if DO_TARIG_PRIM_JOINT_TRAIN:
        os.chdir('..')
        os.chdir('TARigJoint')
        subprocess.run(['python', '00_run_scripts.py', '--root-dir', os.path.join(this_dir, '_generated')]) # '--skip-pretrain-attn', '--skip-gen-binvox' if necessary
        
    os.chdir('..')

# animjointdgcnn preprocessing
if DO_PREPROCESS:
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

if DO_TARIG_PRIM_JOINT_TRAIN:
    os.chdir(CODE_DIR)
    os.chdir('TARigJoint')
    subprocess.run(['python', '01_run_train.py', '--root-dir', os.path.join(this_dir, '_generated')])
    os.chdir('..')

if DO_TARIG_PRIM_JOINT_TRAIN or FORCE_TARIG_PRIM_JOINT_POSTPROCESS:
    os.chdir(CODE_DIR)
    os.chdir('_scripts') 
    # Find epoch number from CSV with latest modified date
    csv_list = [os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', f) for f in os.listdir(os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint')) if os.path.isfile(os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', f)) and f.endswith('.csv')]
    # Remove processed csvs
    csv_list = [f for f in csv_list if not f.endswith('_processed.csv')]
    epoch_number = max(csv_list, key=os.path.getmtime)
    epoch_number = os.path.splitext(os.path.basename(epoch_number))[0]
    epoch_number = epoch_number.split('_')[-1]
    epoch_number = epoch_number.replace('epoch', '')

    # get model list from predictions folder, since they are a subset of the blend folder
    model_list = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', 'results', f'best_{epoch_number}')) if os.path.isfile(os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', 'results', f'best_{epoch_number}', f)) and f.endswith('.json')]
    
    for model in model_list:
        subprocess.run(['python', 'predictions_json_and_transforms_to_original_blend.py',
                        '--blend-file-path', os.path.join(this_dir, '_generated', 'blend', f'{model}.blend'),
                        '--predictions-json-path', os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', 'results', f'best_{epoch_number}', f'{model}.json'),
                        '--normalization-transform-json-path', os.path.join(this_dir, '_generated', 'normalization_transform', f'{model}_transform.json'),
                        '--output-blend-file-path', os.path.join(this_dir, '_generated', 'blend_with_keypoint_visualization', f'{model}.blend'),
                        '--output-predictions-json-path', os.path.join(this_dir, '_generated', 'unnormalized_prediction_jsons', f'{model}.json'),
                        '--ordered-keypoint-names-txt', os.path.join(this_dir, '_generated', 'ordered_joint_names.txt')])

    subprocess.run(['python', 'process_err_by_joint_csv.py',
                    '--input-csv', os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', f'err_by_joint_test_best_epoch{epoch_number}.csv'),
                    '--ordered-joint-names-txt', os.path.join(this_dir, '_generated', 'ordered_joint_names.txt'),
                    '--output-csv', os.path.join(this_dir, '_generated', 'checkpoints', 'tarig_prim_joint', f'err_by_joint_test_best_epoch{epoch_number}_processed.csv')])
    os.chdir('..')

os.chdir(this_dir)