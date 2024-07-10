"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
Author: Guocheng Qian @ 2022, guocheng.qian@kaust.edu.sa
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings
import json

SUPPORTED_REGRESSION_MODELS = ['SoftmaxKeypointDetection', 'RegressionKeypointDetection', 'SigmoidKeypointDetection', 'HumanPoseNetwork']

warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO which keypoint idx corresponds to which keypoint name?
def write_to_csv(mean_error, err_by_joint, best_epoch, cfg, write_header=True, area=5, suffix=None):
    header = ['method', 'mean_error', 'best_epoch', 'log_path']
    data = [cfg.cfg_basename, str(mean_error), str(best_epoch), cfg.run_dir]
    csv_path = cfg.csv_path
    if suffix is not None:
        csv_path = csv_path.replace('.csv', f'_{suffix}.csv')
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()
    csv_path = cfg.csv_path
    if suffix is not None:
        csv_path = csv_path.replace('.csv', f'_{suffix}.csv')
    csv_path = csv_path.replace('.csv', f'_err_by_joint_area{area}.csv')
    header = [str(i) for i in range(len(err_by_joint))]
    data = [err_by_joint[i] for i in range(len(err_by_joint))]
    with open(csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def save_predictions(all_predicted_keypoints, target_dir):
    if all_predicted_keypoints is None:
        raise ValueError('all_predicted_keypoints must be specified')
    if target_dir is None:
        raise ValueError('target_dir must be specified')
    os.makedirs(target_dir, exist_ok=True)

    if isinstance(all_predicted_keypoints, dict):
        for filename, keypoints in all_predicted_keypoints['keypoints_by_filename'].items():
            with open(os.path.join(target_dir, f'{filename}.json'), 'w') as f:
                json.dump(keypoints, f)
            save_vertex_array_to_ply_file(np.array(keypoints), os.path.join(target_dir, f'{filename}.ply'))
    elif isinstance(all_predicted_keypoints, list):
        with open(os.path.join(target_dir, 'all_predictions.json'), 'w') as f:
            json.dump(all_predicted_keypoints, f)
        save_vertex_array_to_ply_file(np.array(all_predicted_keypoints), os.path.join(target_dir, 'vis.ply'))
    else:
        raise ValueError('all_predicted_keypoints must be a dict or list')

def save_ply_file(filepath, vertex_lines):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        num_vertices = len(vertex_lines)
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_vertices}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        f.write('\n'.join(vertex_lines))


def save_vertex_array_to_ply_file(vertex_array, filepath):
    lines = [f'{vertex_array[i][0]} {vertex_array[i][1]} {vertex_array[i][2]}' 
            for i in range(0, len(vertex_array))]
    save_ply_file(filepath, lines)


def main(gpu, cfg, args=None):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        if cfg.model.get('encoder_args', None):
            if cfg.model.encoder_args.get('in_channels', None):
                cfg.model.in_channels = cfg.model.encoder_args.in_channels
        if cfg.model.get('in_channels', None) is None:
            cfg.model.in_channels = 3
        
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    if cfg.mode in ['train', 'resume']:
        optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
        if cfg.sched.lower() == 'onecycle'.lower():
            num_training_samples = len(glob.glob(os.path.join(cfg.dataset.common.data_root, 'train', '*.ply')))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, pct_start=0.0, div_factor=25.00, three_phase=False, final_div_factor=1e4, total_steps=cfg.epochs * num_training_samples)
            cfg.sched_on_epoch = False
        else:
            scheduler = build_scheduler_from_cfg(cfg, optimizer)
    
    # build val dataset
    if cfg.mode != 'process_single':
        val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='val',
                                            distributed=cfg.distributed
                                            )


        # build test dataset
        test_loader = build_dataloader_from_cfg(cfg.get('test_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )

        logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
        logging.info(f"length of test dataset: {len(test_loader.dataset)}")

        cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None

    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val_mean_error = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_mean_error, val_err_by_joint, val_all_predicted_keypoints = validate(model, val_loader, cfg)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch}, val_mean_error: {val_mean_error:.8f}')
                # Save all predictions inside a single JSON inside current logging directory
                if cfg.save_predictions:
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_val.csv')
                    write_to_csv(val_mean_error, val_err_by_joint, best_epoch, cfg, write_header=True)
                    target_dir = os.path.join(cfg.run_dir, 'val_predictions')
                    save_predictions(val_all_predicted_keypoints, target_dir)

                return val_mean_error
            elif cfg.mode == 'test':
                best_epoch, best_val_mean_error = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                logging.info(f"length of test dataset: {len(test_loader)}")
                test_mean_error, test_err_by_joint, test_all_predicted_keypoints = test(model, test_loader)
                if test_mean_error is not None:
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(
                            f'Best ckpt @E{best_epoch}, test_mean_error: {test_mean_error:.8f}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(test_mean_error, test_err_by_joint, best_epoch, cfg, write_header=True)
                    target_dir = os.path.join(cfg.run_dir, 'test_predictions')
                    save_predictions(test_all_predicted_keypoints, target_dir)
                return test_mean_error
            
            elif cfg.mode == 'process_single':
                best_epoch, best_val_mean_error = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                predicted_keypoints = process_single(model, args.ply)
                save_predictions(predicted_keypoints, 'process_single_results')  #TODO save to a directory
                return

            elif 'encoder' in cfg.mode:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))
            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
    else:
        logging.info('Training from scratch')

    if 'freeze_blocks' in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False

    train_loader, data_transform = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             return_data_transform=True
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    cfg.criterion_args.weight = None
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    # ===> start training
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    val_mean_error = float('inf')
    best_val_mean_error = float('inf')
    best_epoch = 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_mean_error, val_err_by_joint, val_all_predicted_keypoints = validate(model, val_loader, cfg)
            if val_mean_error < best_val_mean_error:
                is_best = True
                best_val_mean_error = val_mean_error
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_mean_error {val_mean_error:.8f}')
                
                # Save all predictions inside a single JSON inside current logging directory
                if cfg.save_predictions:
                    target_dir = os.path.join(cfg.run_dir, 'val_predictions')
                    save_predictions(val_all_predicted_keypoints, target_dir)
                cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '.csv')
                write_to_csv(best_val_mean_error, val_err_by_joint, best_epoch, cfg, write_header=True, suffix='val')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.7f} '
                     f'train_loss {train_loss:.8f}, val_mean_error {val_mean_error:.8f}, best_val_mean_error {best_val_mean_error:.8f}')
        if writer is not None:
            writer.add_scalar('best_val_mean_error', best_val_mean_error, epoch)
            writer.add_scalar('val_mean_error', val_mean_error, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        # TODO Debug code for printing average finger error.
        # finger_joints_1 = sum(val_err_by_joint[11:31])
        # finger_joints_2 = sum(val_err_by_joint[36:56])
        # val_mean_error_2 = finger_joints_1 + finger_joints_2
        # logging.info(f'Average finger error: {val_mean_error_2/40}')

        if cfg.sched_on_epoch:
            # TODO HACK Recompute val_mean_error to take into account only finger joints
            scheduler.step(epoch, metric=val_mean_error)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val_mean_error': best_val_mean_error},
                            is_best=is_best
                            )
            is_best = False

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch}, val_mean_error {best_val_mean_error:.8f}, ')

    if cfg.world_size < 2:  # do not support multi gpu testing
        # test
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        test_mean_error, test_err_by_joint, test_all_predicted_keypoints = test(model, test_loader)
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Best ckpt @E{best_epoch},  test_mean_error {test_mean_error:.8f}')
        if writer is not None:
            writer.add_scalar('test_mean_error', test_mean_error, epoch)
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
        write_to_csv(test_mean_error, test_err_by_joint, best_epoch, cfg, write_header=True)
        target_dir = os.path.join(cfg.run_dir, 'test_predictions')
        save_predictions(test_all_predicted_keypoints, target_dir)
        logging.info(f'save results in {cfg.csv_path}')
    else:
        logging.warning('Testing using multiple GPUs is not allowed for now. Running testing after this training is required.')
    if writer is not None:
        writer.close()
    if cfg.distributed:
        dist.destroy_process_group()
    wandb.finish(exit_code=True)

# TODO Support for criterion
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    total_loss = 0
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), ascii=True)
    num_iter = 0
    for idx, data in pbar:
        # NOTE Vertex order in data['points'] corresponds to the vertex order in the obj_remesh file
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['keypoints'].squeeze(-1) # TODO should "keypoints" be here instead of y?
        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        if 'points' not in data.keys():
            raise ValueError('`points` attribute is not founded in dataset')
            # print(data.keys()) # TODO ?
        if 'feats' not in data.keys():
            input_data = { 'pos': data['points'] }
        else:
            input_data = { 'pos': data['points'], 'feats': data['feats'] }
        input_data['x'] = get_features_by_keys(input_data, cfg.feature_keys)
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            pred = model(input_data)

            if model.__class__.__name__ not in SUPPORTED_REGRESSION_MODELS:
                raise NotImplementedError(f'Unsupported model type {model.__class__.__name__}')
            
            keypoints = pred

            # TODO HACK Scale finger keypoints loss by 50
            # errors = torch.sum((keypoints - target) ** 2, dim=2)
            # finger_joints_1 = errors[:, 11:31].sum()
            # finger_joints_2 = errors[:, 36:56].sum()
            # loss = torch.sum(errors) + 49 * (finger_joints_1 + finger_joints_2)

            loss = torch.sum((keypoints - target) ** 2) # TODO Support for external criterion

            # NOTE for debugging, save ply files
            # # create ply files in log directory
            # target_dir = os.path.join(cfg.log_dir, 'ply_files')
            # os.makedirs(target_dir, exist_ok=True)

            # # save input_data['pos'], which is a 1xNUM_VERTICESx3 tensor to a PLY file
            # save_vertex_array_to_ply_file(input_data['pos'][0], os.path.join(target_dir, f'{epoch}_{idx}_input.ply'))

            # # save target, which is a 1xNUM_KEYPOINTSx3 to a PLY file
            # save_vertex_array_to_ply_file(target[0], os.path.join(target_dir, f'{epoch}_{idx}_target.ply'))                

        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                if cfg.distributed:
                    raise NotImplementedError('Only support scheduler on epoch')
                
                scheduler.step()

        # update loss_meter
        current_loss = loss.item()
        total_loss += current_loss

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {current_loss:.3f}")
    mean_error = total_loss / len(train_loader)
    return mean_error

@torch.no_grad()
def process_single(model, ply_path):
    import open3d as o3d
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float().reshape(1, -1, 3).cuda()

    data = {'points': points}
    if 'feats' not in data.keys():
        input_data = { 'pos': data['points'] }
    else:
        input_data = { 'pos': data['points'], 'feats': data['feats'] }
    input_data['x'] = get_features_by_keys(input_data, cfg.feature_keys)

    pred = model(input_data)
    predicted_keypoints = pred.cpu().numpy().reshape(-1, 3).tolist()

    return predicted_keypoints

@torch.no_grad()
def test_or_validate(model, dataloader, cfg, desc=None):
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=dataloader.__len__(), desc=desc, ascii=True)
    total_error = 0
    mae_by_joint = None

    # TODO Debug code for saving errors
    model_names = []
    maes = []

    all_predicted_keypoints = {'keypoints_by_filename': dict()} if cfg.save_predictions else None
    for idx, data in pbar:
        batch_size = len(data['keypoints'])
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if isinstance(data[key], torch.Tensor) and key != 'keypoints':
                data[key] = data[key].cuda(non_blocking=True)
        target = data['keypoints'].squeeze(-1) # TODO Check if batch_size > 1 is supported
        if 'feats' not in data.keys():
            input_data = { 'pos': data['points'] }
        else:
            input_data = { 'pos': data['points'], 'feats': data['feats'] }
        input_data['x'] = get_features_by_keys(input_data, cfg.feature_keys)
        pred = model(input_data)
        if model.__class__.__name__ in SUPPORTED_REGRESSION_MODELS:
            predicted_keypoints = pred.cpu()
        else:
            raise NotImplementedError(f'Unsupported model type {model.__class__.__name__}')
        
        for i in range(batch_size):
            # Perform inverse transform
            for transform_data in reversed(data['transforms']):
                transform_fqn = transform_data['transform_fqn'][i]
                transform = dataloader.dataset.transforms[transform_fqn]
                if hasattr(transform, 'inverse_transform'):
                    # Retrieve state for current example in mini-batch
                    state = {key: transform_data['state'][key][i] for key in transform_data['state'].keys()}
                    predicted_keypoints[i] = transform.inverse_transform(predicted_keypoints[i], state)
        # calc mae by joint by taking the sqrt of the sum of the squared errors
        mse = torch.sum((predicted_keypoints - target) ** 2, dim=2)
        mae = torch.sqrt(mse)

        # TODO Debug code for saving errors
        maes.append(mae[0].detach().cpu().numpy()) # NOTE this supports only batch_size = 1
        model_names.append(data['filename'][0])

        if mae_by_joint is None:
            mae_by_joint = torch.zeros((predicted_keypoints.shape[1]))
        mae_by_joint += torch.sum(mae, dim=0)
        total_error += torch.sum((predicted_keypoints - target) ** 2).item()
        if cfg.save_predictions:
            for i in range(len(data['filename'])):
                filename = data['filename'][i]
                keypoints = predicted_keypoints[i].cpu().numpy()
                keypoints = keypoints.reshape(-1, 3)
                keypoints = keypoints.tolist()
                all_predicted_keypoints['keypoints_by_filename'][filename] = keypoints
    mean_error = total_error / len(dataloader.dataset)
    mae_by_joint /= len(dataloader.dataset)
    mae_by_joint = mae_by_joint.reshape(-1)
    mae_by_joint = mae_by_joint.tolist()

    # TODO Debug code for saving errors
    maes = np.array(maes)
    # save maes
    if cfg.save_predictions:
        np.save(os.path.join(cfg.run_dir, 'maes.npy'), maes)
        # save model names as json
        with open(os.path.join(cfg.run_dir, 'model_names.json'), 'w') as f:
            json.dump(model_names, f)

    return mean_error, mae_by_joint, all_predicted_keypoints

@torch.no_grad()
def validate(model, val_loader, cfg, desc=None):
    return test_or_validate(model, val_loader, cfg, desc='Val')


@torch.no_grad()
def test(model, dataloader):
    return test_or_validate(model, dataloader, cfg, desc='Test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Keypoint detection training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument('--ply', type=str, default=None, help='ply file path for process_single mode')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = 0

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test', 'process_single']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg, args)
