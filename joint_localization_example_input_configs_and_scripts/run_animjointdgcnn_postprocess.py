import subprocess
import os
import shutil

CODE_DIR = None # TODO Specify
CODE_DIR = os.path.abspath(CODE_DIR)

this_dir = os.getcwd()

os.chdir(CODE_DIR)
os.chdir('_scripts') 
# Find directory with latest modified date in _generated_animjointdgcnn/logs
log_dir = max([os.path.join(this_dir, '_generated_animjointdgcnn', 'logs', f) 
               for f in os.listdir(os.path.join(this_dir, '_generated_animjointdgcnn', 'logs')) 
               if os.path.isdir(os.path.join(this_dir, '_generated_animjointdgcnn', 'logs', f))], 
               key=os.path.getmtime)

# Inside this directory, find the csv that ends with _test_err_by_joint_area5.csv
csv_list = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f)) and f.endswith('_test_err_by_joint_area5.csv')]

assert len(csv_list) == 1, f"Expected 1 csv file ending with _test_err_by_joint_area5.csv, but found {len(csv_list)}: {csv_list}"

csv_path = csv_list[0]

# get model list from predictions folder, since they are a subset of the blend folder (test_predictions folder inside log_dir)
model_list = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(os.path.join(log_dir, 'test_predictions')) if os.path.isfile(os.path.join(log_dir, 'test_predictions', f)) and f.endswith('.json')]
for model in model_list:
    subprocess.run(['python', 'predictions_json_and_transforms_to_original_blend.py',
                    '--blend-file-path', os.path.join(this_dir, '_generated', 'blend', f'{model}.blend'),
                    '--predictions-json-path', os.path.join(log_dir, 'test_predictions', f'{model}.json'),
                    '--normalization-transform-json-path', os.path.join(this_dir, '_generated', 'normalization_transform', f'{model}_transform.json'),
                    '--output-blend-file-path', os.path.join(this_dir, '_generated_animjointdgcnn', 'blend_with_keypoint_visualization', f'{model}.blend'),
                    '--output-predictions-json-path', os.path.join(this_dir, '_generated_animjointdgcnn', 'unnormalized_prediction_jsons', f'{model}.json'),
                    '--ordered-keypoint-names-txt', os.path.join(this_dir, '_generated_animjointdgcnn', 'ordered_keypoint_names.txt')])

subprocess.run(['python', 'process_err_by_joint_csv.py',
                '--input-csv', csv_path,
                '--ordered-joint-names-txt', os.path.join(this_dir, '_generated_animjointdgcnn', 'ordered_keypoint_names.txt'),
                '--output-csv', os.path.join(this_dir, '_generated_animjointdgcnn', 'logs', f'{os.path.splitext(os.path.basename(csv_path))[0]}_processed.csv')])
os.chdir('..')

os.chdir(this_dir)