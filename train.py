import argparse
import logging
from pathlib import Path
import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
from tqdm import tqdm

from pytorch_toolbelt.losses import DiceLoss

from utils.PolynomialLRDecay import PolynomialLRDecay
from utils.StepLRDecay import StepLRDecay
from utils.config import parse_net_cfg, print_net_cfg, get_net_from_config, parse_train_cfg, print_train_cfg
from utils.dataset import SkinLesionDataset
from evaluate import evaluate
from utils.utils import mask_to_image


def train_net(net_cfg, net,
              device,
              train_cfg: dict,
              dataset_path: str,
              checkpoint_dir: str = "checkpoints",
              tb_writer=None,
              wandb_run=None):
    if train_cfg['checkpoint']['enable'] or train_cfg['evaluate']['enable']:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_path)

    train_set = SkinLesionDataset(
        images_file_path=str(dataset_path / 'train_images.txt'),
        masks_file_path=str(dataset_path / 'train_masks.txt'),
        width=train_cfg['dataset']['image_size'][1],
        height=train_cfg['dataset']['image_size'][0],
        mode='train',
        augmentation_cfg=train_cfg['dataset_augmentation'],
    )

    n_steps_in_one_epoch = len(train_set) // train_cfg['dataset']['batch_size']
    total_steps = n_steps_in_one_epoch * train_cfg['epoch']['count']
    n_images_in_one_epoch = n_steps_in_one_epoch * train_cfg['dataset']['batch_size']

    loader_args = dict(
        batch_size=train_cfg['dataset']['batch_size'],
        num_workers=train_cfg['dataset']['num_workers'],
        pin_memory=True, drop_last=True
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if train_cfg['optim']['type'] == 'adam':
        optimizer = optim.Adam(
            net.parameters(), lr=train_cfg['optim']['lr'],
            betas=train_cfg['optim']['betas'],
            weight_decay=train_cfg['optim']['weight_decay']
        )
    elif train_cfg['optim']['type'] == 'rmsprop':
        optimizer = optim.RMSprop(
            net.parameters(), lr=train_cfg['optim']['lr'],
            momentum=train_cfg['optim']['momentum'],
            weight_decay=train_cfg['optim']['weight_decay'])
    elif train_cfg['optim']['type'] == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=train_cfg['optim']['lr'],
            momentum=train_cfg['optim']['momentum'],
            weight_decay=train_cfg['optim']['weight_decay']
        )

    else:
        raise ValueError("optim type is invalid.")

    scheduler = None
    if train_cfg['optim_lr_schedule']['enable']:
        if train_cfg['optim_lr_schedule']['type'] == 'polynomial':
            scheduler = PolynomialLRDecay(
                optimizer,
                max_decay_steps=train_cfg['epoch']['count'],
                end_learning_rate=train_cfg['optim_lr_schedule']['eta_min'],
                power=train_cfg['optim_lr_schedule']['exponent']
            )
        elif train_cfg['optim_lr_schedule']['type'] == 'steps':
            scheduler = StepLRDecay(
                optimizer,
                max_decay_steps=train_cfg['epoch']['count'],
                steps=train_cfg['optim_lr_schedule']['steps'],
                scales=train_cfg['optim_lr_schedule']['scales']
            )
        elif train_cfg['optim_lr_schedule']['type'] == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg['epoch']['count'],
                eta_min=train_cfg['optim_lr_schedule']['eta_min']
            )
        elif train_cfg['optim_lr_schedule']['type'] == 'cosine_annealing_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=train_cfg['optim_lr_schedule']['first_restart_at'],
                T_mult=train_cfg['optim_lr_schedule']['after_restart_factor'],
                eta_min=train_cfg['optim_lr_schedule']['eta_min']
            )

    if net.n_classes == 1:
        ce_criterion = nn.BCEWithLogitsLoss()
        dice_criterion = DiceLoss(mode='binary', from_logits=True)
    else:
        ce_criterion = nn.CrossEntropyLoss()
        dice_criterion = DiceLoss(mode='multiclass', from_logits=True)

    if train_cfg['epoch']['unfreeze_at'] != 0:
        net.freeze_pretrained_wight(freeze=True)

    global_step = 0
    eval_metric_best_value = - torch.inf

    # 5. Begin training
    for epoch in range(1, train_cfg['epoch']['count'] + 1):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_images_in_one_epoch, desc=f"Epoch {epoch}/{train_cfg['epoch']['count']}",
                  unit='img') as pbar:
            for images, true_masks in train_loader:
                assert images.shape[1] == net.n_input_channels, \
                    f'Network has been defined with {net.n_input_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)

                if net.n_classes == 1:
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                else:
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                pred_masks = net(images)

                ce_loss_val = 0
                if train_cfg['loss']['ce_factor'] != 0:
                    if net.n_classes == 1:
                        ce_loss_val = ce_criterion(pred_masks, true_masks[..., None].permute(0, 3, 1, 2))
                    else:
                        ce_loss_val = ce_criterion(pred_masks, true_masks)

                dice_loss_val = 0
                if train_cfg['loss']['dice_factor'] != 0:
                    dice_loss_val = dice_criterion(pred_masks, true_masks)

                loss = train_cfg['loss']['ce_factor'] * ce_loss_val + train_cfg['loss']['dice_factor'] * dice_loss_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                global_step += 1
                epoch_loss += loss.item()

                if tb_writer:
                    tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
                    tb_writer.add_scalar('train loss', loss.item(), global_step)
                    tb_writer.add_scalar('epoch', epoch, global_step)

                if wandb_run:
                    wandb_run.log({
                        'lr': optimizer.param_groups[0]['lr'],
                        'train loss': loss.item(),
                        'epoch': epoch
                    }, step=global_step)

                if train_cfg['checkpoint']['enable'] and global_step % int(
                        total_steps * train_cfg['checkpoint']['step']) == 0:
                    torch.save(net.state_dict(), str(checkpoint_dir / 'checkpoint_step{}.pth'.format(global_step)))
                    logging.info(f'Checkpoint step={global_step} saved!')

                # Evaluation round
                if train_cfg['evaluate']['enable'] and global_step % int(
                        total_steps * train_cfg['evaluate']['step']) == 0:
                    pbar_current = pbar.n
                    pbar.close()

                    val_metrics_names, _, _, val_metrics_score, _ = evaluate(
                        net=net, net_cfg=net_cfg,
                        device=device,
                        images_file_path=str(dataset_path / 'val_images.txt'),
                        masks_file_path=str(dataset_path / 'val_masks.txt'),
                    )

                    val_metrics_dict = {}
                    for val_metric_name, val_metric_score in zip(val_metrics_names, val_metrics_score):
                        val_metrics_dict[val_metric_name] = val_metric_score.item()

                    logging.info('Validation Metrics: {}'.format(val_metrics_dict))

                    if val_metrics_dict[train_cfg['evaluate']['metric']] >= eval_metric_best_value:
                        eval_metric_best_value = val_metrics_dict[train_cfg['evaluate']['metric']]

                        torch.save(net.state_dict(), str(checkpoint_dir / 'checkpoint_best.pth'))
                        logging.info(
                            f"{train_cfg['evaluate']['metric']} metric improved and network weights saved. New value: {eval_metric_best_value} ")

                    if (tb_writer and train_cfg['log_tensorboard']['sample_image_pred']) or (
                            wandb_run and train_cfg['log_wandb']['sample_image_pred']):
                        sample_image = images[0].detach().cpu()
                        sample_true_mask = np.array(mask_to_image(true_masks[0]))
                        if net.n_classes == 1:
                            sample_pred_mask = np.array(mask_to_image((torch.sigmoid(pred_masks[0]) > 0.5)))
                        else:
                            sample_pred_mask = np.array(mask_to_image(
                                F.one_hot(pred_masks[0].argmax(dim=0), net.n_classes).permute(2, 0, 1)
                            ))

                    if tb_writer:
                        for key, value in val_metrics_dict.items():
                            tb_writer.add_scalar(f'validation_metrics/{key}', value, global_step)

                        if train_cfg['log_tensorboard']['sample_image_pred']:
                            tb_writer.add_image('sample/image', sample_image, global_step)
                            tb_writer.add_image('sample/true_mask', sample_true_mask, global_step, dataformats='HW')
                            tb_writer.add_image('sample/pred_mask', sample_pred_mask, global_step, dataformats='HW')

                        if train_cfg['log_tensorboard']['histogram']:
                            for tag, value in net.named_parameters():
                                if value is not None and value.grad is not None:
                                    tag = tag.replace('/', '.')
                                    tb_writer.add_histogram('Weights/' + tag, value.data.cpu(), global_step)
                                    tb_writer.add_histogram('Gradients/' + tag, value.grad.data.cpu(), global_step)

                    if wandb_run:
                        wandb_run.log({
                            'validation_metrics': val_metrics_dict,
                        }, step=global_step)

                        if train_cfg['log_wandb']['sample_image_pred']:
                            wandb_run.log({
                                'sample': {
                                    'image': wandb.Image(sample_image),
                                    'true_mask': wandb.Image(sample_true_mask),
                                    'pred_mask': wandb.Image(sample_pred_mask),
                                }
                            }, step=global_step)

                        if train_cfg['log_wandb']['histogram']:
                            histograms = {}

                            for tag, value in net.named_parameters():
                                if value is not None and value.grad is not None:
                                    tag = tag.replace('/', '.')
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            wandb_run.log(histograms, step=global_step)

                    pbar = tqdm(total=n_images_in_one_epoch, initial=pbar_current,
                                desc=f"Epoch {epoch}/{train_cfg['epoch']['count']}",
                                unit='img')

            if scheduler is not None:
                scheduler.step()

        if train_cfg['epoch']['unfreeze_at'] != 0 and train_cfg['epoch']['unfreeze_at'] == epoch:
            net.freeze_pretrained_wight(freeze=False)

        pbar.close()

    if train_cfg['checkpoint']['enable']:
        torch.save(net.state_dict(), str(checkpoint_dir / 'checkpoint_last.pth'))
        logging.info(f'Checkpoint last saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train a model on images and target masks.')

    parser.add_argument('--net-cfg', '-nc', type=str, default='cfg/net/dual-encoder-unet.cfg',
                        help='Specify the path to the configuration file for the network architecture.')

    parser.add_argument('--train-cfg', '-tc', type=str, default='cfg/train.cfg',
                        help='Specify the path to the configuration file for the training settings.')

    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Specify the directory where the prepared dataset is stored.')

    parser.add_argument('--load', type=str, default=False,
                        help='Specify the path to a .pth file from which a pre-trained model should be loaded.')

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Specify the directory where checkpoints should be saved during training. Default is "checkpoints".')

    parser.add_argument('--tensorboard-dir', type=str, default="",
                        help='Specify the main directory where TensorBoard logs should be saved.')

    parser.add_argument('--wandb-dir', type=str, default="",
                        help='Specify the main directory where Weights & Biases logs should be saved.')

    parser.add_argument('--wandb-run-name', type=str, default='SkinLesionSegmentation',
                        help='Specify a name for the Weights & Biases run. Default is "SkinLesionSegmentation".')

    parser.add_argument('--note', type=str, default=False,
                        help='Any note or comment about the training run that should be logged.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Use this flag if you don\'t want to use CUDA, even if it is available.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net_cfg = parse_net_cfg(args.net_cfg)
    logging.info(print_net_cfg(net_cfg, ret=True))

    train_cfg = parse_train_cfg(args.train_cfg)
    logging.info(print_train_cfg(train_cfg, ret=True))

    net = get_net_from_config(net_cfg)
    net.to(device=device)

    tb_writer = None
    if train_cfg['log_tensorboard']['enable']:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
        tensorboard_dir = Path(args.tensorboard_dir) / "tensorboard" / ("run-" + str(time_str))
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = SummaryWriter(str(tensorboard_dir))

        tb_writer.add_text("arguments",
                           "".join("\t" + line for line in json.dumps(vars(args), indent=2).splitlines(True)))

        tb_writer.add_text("net_config",
                           "".join("\t" + line for line in json.dumps(net_cfg, indent=2).splitlines(True)))

        tb_writer.add_text("train_config",
                           "".join("\t" + line for line in json.dumps(train_cfg, indent=2).splitlines(True)))

        if args.note:
            tb_writer.add_text("note", args.note)

        try:
            dummy_input = torch.randn(2, net.n_input_channels,
                                      train_cfg['dataset']['image_size'][0], train_cfg['dataset']['image_size'][1],
                                      device=device)

            tb_writer.add_graph(net, dummy_input)
        except:
            pass

    wandb_run = None
    if train_cfg['log_wandb']['enable']:
        wandb_dir = Path(args.wandb_dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)

        wandb_run = wandb.init(dir=wandb_dir, project=args.wandb_run_name, resume='allow', anonymous='must',
                               mode=train_cfg['log_wandb']['mode'])

        wandb_run.config.update({
            "arguments": vars(args),
            "net_config": net_cfg,
            "train_config": train_cfg,
        })

        if args.note:
            wandb_run.config.update({
                "note": args.note
            })

    if args.load:
        logging.info('Load weights {}.'.format(args.load))

        model_dict = net.state_dict()
        pretrained_dict = torch.load(args.load, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)

        logging.info(
            "\nSuccessful Load Key: " + str(load_key)[:500] + "...\nSuccessful Load Key Num: " + str(len(load_key)))
        logging.info(
            "\nFail To Load Key: " + str(no_load_key)[:500] + "...\nFail To Load Key Num: " + str(len(no_load_key)))

    try:
        train_net(
            net_cfg=net_cfg, net=net,
            device=device,
            train_cfg=train_cfg,
            dataset_path=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            tb_writer=tb_writer,
            wandb_run=wandb_run
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
