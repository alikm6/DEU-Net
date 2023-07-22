import argparse
import logging
from typing import Optional
import string
import random

import torch
import torch.nn.functional as f
from tqdm import tqdm
from pathlib import Path
from torchmetrics import Dice
from torchmetrics.classification import MultilabelAccuracy, MultilabelJaccardIndex, MultilabelRecall, \
    MultilabelSpecificity, MultilabelPrecision, MultilabelF1Score, MultilabelAUROC
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryRecall, \
    BinarySpecificity, BinaryPrecision, BinaryF1Score, BinaryAUROC

from predict import predict_image_ITTA, resize_pred_score, convert_pred_score_to_01, \
    transforms_names_list_to_torch
from utils.dataset import SkinLesionDataset
from utils.utils import plot_img_and_mask, mask_to_image
from utils.config import parse_net_cfg, print_net_cfg, get_net_from_config


def evaluate(
        net, net_cfg,
        device,
        images_file_path: str, masks_file_path: str,
):
    return evaluate_full(
        nets=[net, ], nets_cfg=[net_cfg, ],
        device=device,
        transforms=None, reversed_transforms=None,
        images_file_path=images_file_path,
        masks_file_path=masks_file_path,
        visualize=False,
        save_output=False,
        eval_org_dim=False,
    )


def evaluate_full(
        nets: list, nets_cfg: list,
        device,
        images_file_path: str, masks_file_path: str,
        transforms: Optional[list] = None, reversed_transforms: Optional[list] = None,
        visualize: bool = False,
        save_output: bool = False, save_dir: str = "eval_out",
        eval_org_dim: bool = False
):
    images_path = SkinLesionDataset.load_txt(images_file_path)
    masks_path = SkinLesionDataset.load_txt(masks_file_path)
    images_count = len(images_path)

    assert len(images_path) == len(masks_path) and len(images_path) != 0

    if save_output:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    n_classes = nets[0].n_classes

    input_width = nets_cfg[0]['input_size'][1]
    input_height = nets_cfg[0]['input_size'][0]
    input_channels = nets[0].n_input_channels

    def compute_score(fn, preds, target, multiclass_ignore_bg=True):
        score = fn(preds=preds, target=target)

        if n_classes != 1:
            if multiclass_ignore_bg:
                score = score[1:, ...]

            score = torch.mean(score)

        return score.item()

    def compute_score_full(pred_mask_score, true_mask, true_mask_expanded):
        if n_classes == 1:
            dice_fn = Dice().to(device=device)
            iou_fn = BinaryJaccardIndex().to(device=device)
            recall_fn = BinaryRecall().to(device=device)
            specificity_fn = BinarySpecificity().to(device=device)
            accuracy_fn = BinaryAccuracy().to(device=device)
            precision_fn = BinaryPrecision().to(device=device)
            f1_fn = BinaryF1Score().to(device=device)
            auc_fn = BinaryAUROC().to(device=device)
        else:
            dice_fn = Dice(num_classes=2, average='none', multiclass=True).to(device=device)
            iou_fn = MultilabelJaccardIndex(num_labels=2, average='none').to(device=device)
            recall_fn = MultilabelRecall(num_labels=2, average='none').to(device=device)
            specificity_fn = MultilabelSpecificity(num_labels=2, average='none').to(device=device)
            accuracy_fn = MultilabelAccuracy(num_labels=2, average='none').to(device=device)
            precision_fn = MultilabelPrecision(num_labels=2, average='none').to(device=device)
            f1_fn = MultilabelF1Score(num_labels=2, average='none').to(device=device)
            auc_fn = MultilabelAUROC(num_labels=2, average='none').to(device=device)

        metrics = torch.Tensor([
            compute_score(fn=dice_fn, preds=pred_mask_score, target=true_mask),  # dice
            compute_score(fn=iou_fn, preds=pred_mask_score, target=true_mask_expanded),  # iou
            compute_score(fn=recall_fn, preds=pred_mask_score, target=true_mask_expanded),  # recall
            compute_score(fn=specificity_fn, preds=pred_mask_score, target=true_mask_expanded),  # specificity
            compute_score(fn=accuracy_fn, preds=pred_mask_score, target=true_mask_expanded),  # accuracy
            compute_score(fn=precision_fn, preds=pred_mask_score, target=true_mask_expanded),  # precision
            compute_score(fn=f1_fn, preds=pred_mask_score, target=true_mask_expanded),  # f1
            compute_score(fn=auc_fn, preds=pred_mask_score, target=true_mask_expanded)  # auc
        ])

        return metrics

    metrics_names = ['dice', 'iou', 'recall', 'specificity', 'accuracy', 'precision', 'f1', 'auc', ]
    metrics_count = len(metrics_names)

    metrics_net_dim = torch.zeros(images_count, metrics_count, dtype=torch.float32)

    if eval_org_dim:
        metrics_org_dim = torch.zeros(images_count, metrics_count, dtype=torch.float32)

    # iterate over the validation set
    for image_id, (image_path, mask_path) in tqdm(
            enumerate(zip(images_path, masks_path)),
            total=images_count,
            desc=f'Evaluate',
            unit='img'
    ):
        image_pil = SkinLesionDataset.load(image_path)

        image_org_dim_width, image_org_dim_height = image_pil.size

        if eval_org_dim:
            image_org_dim = SkinLesionDataset.preprocess(image_pil, is_mask=False, width=None, height=None).to(
                device=device)

        image_net_dim = SkinLesionDataset.preprocess(image_pil, is_mask=False,
                                                     width=input_width, height=input_height).to(device=device)

        true_mask_pil = SkinLesionDataset.load(mask_path)

        if eval_org_dim:
            true_mask_org_dim = SkinLesionDataset.preprocess(true_mask_pil, is_mask=True, width=None, height=None).to(
                device=device)

        true_mask_net_dim = SkinLesionDataset.preprocess(true_mask_pil, is_mask=True,
                                                         width=input_width, height=input_height).to(device=device)

        if n_classes == 1:
            if eval_org_dim:
                true_mask_org_dim_expanded = true_mask_org_dim[None, ...]

            true_mask_net_dim_expanded = true_mask_net_dim[None, ...]
        else:
            if eval_org_dim:
                true_mask_org_dim_expanded = f.one_hot(true_mask_org_dim, n_classes).permute(2, 0, 1)

            true_mask_net_dim_expanded = f.one_hot(true_mask_net_dim, n_classes).permute(2, 0, 1)

        pred_mask_net_dim_score, pred_mask_net_dim = predict_image_ITTA(
            nets=nets, nets_cfg=nets_cfg,
            image=image_net_dim, device=device,
            transforms=transforms, reversed_transforms=reversed_transforms,
        )

        if eval_org_dim:
            # resize pred_mask_net_dim_score to org_dim
            pred_mask_org_dim_score = resize_pred_score(
                pred_mask_net_dim_score,
                new_width=image_org_dim_width,
                new_height=image_org_dim_height
            )

            # calculate pred_mask_org_dim from pred_mask_org_dim_score
            pred_mask_org_dim = convert_pred_score_to_01(pred_mask_org_dim_score.unsqueeze(0), n_classes).squeeze(0)

        # add batch dimension
        true_mask_net_dim = true_mask_net_dim[None, ...]
        true_mask_net_dim_expanded = true_mask_net_dim_expanded[None, ...]
        pred_mask_net_dim_score = pred_mask_net_dim_score[None, ...]
        pred_mask_net_dim = pred_mask_net_dim[None, ...]

        if eval_org_dim:
            true_mask_org_dim = true_mask_org_dim[None, ...]
            true_mask_org_dim_expanded = true_mask_org_dim_expanded[None, ...]
            pred_mask_org_dim_score = pred_mask_org_dim_score[None, ...]
            pred_mask_org_dim = pred_mask_org_dim[None, ...]

        if visualize:
            if n_classes == 1:
                plot_img_and_mask(img=image_net_dim, pred_mask=pred_mask_net_dim[0, ...],
                                  true_mask=true_mask_net_dim_expanded[0, ...])

                if eval_org_dim:
                    plot_img_and_mask(img=image_org_dim, pred_mask=pred_mask_org_dim[0, ...],
                                      true_mask=true_mask_org_dim_expanded[0, ...])
            else:
                # ignore background
                plot_img_and_mask(img=image_net_dim, pred_mask=pred_mask_net_dim[0, 1:, ...],
                                  true_mask=true_mask_net_dim_expanded[0, 1:, ...])

                if eval_org_dim:
                    plot_img_and_mask(img=image_org_dim, pred_mask=pred_mask_org_dim[0, 1:, ...],
                                      true_mask=true_mask_org_dim_expanded[0, 1:, ...])

        if save_output:
            pred_mask_net_dim_img = mask_to_image(pred_mask_net_dim[0, ...])
            pred_mask_net_dim_img.save(save_dir / (Path(image_path).stem + "_OUT_net_dim.png"))

            if eval_org_dim:
                pred_mask_org_dim_img = mask_to_image(pred_mask_org_dim[0, ...])
                pred_mask_org_dim_img.save(save_dir / (Path(image_path).stem + "_OUT_org_dim.png"))

        # calculate metrics_net_dim
        metrics_net_dim[image_id] = compute_score_full(pred_mask_score=pred_mask_net_dim_score,
                                                       true_mask=true_mask_net_dim,
                                                       true_mask_expanded=true_mask_net_dim_expanded)

        if eval_org_dim:
            # calculate metrics_org_dim
            metrics_org_dim[image_id] = compute_score_full(pred_mask_score=pred_mask_org_dim_score,
                                                           true_mask=true_mask_org_dim,
                                                           true_mask_expanded=true_mask_org_dim_expanded)

    metrics_net_dim_avg = torch.mean(metrics_net_dim, dim=0)

    if eval_org_dim:
        metrics_org_dim_avg = torch.mean(metrics_org_dim, dim=0)

    if eval_org_dim:
        return metrics_names, metrics_net_dim, metrics_org_dim, metrics_net_dim_avg, metrics_org_dim_avg
    else:
        return metrics_names, metrics_net_dim, None, metrics_net_dim_avg, None


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test images using trained models.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Use this flag if you don\'t want to use CUDA, even if it is available.')

    parser.add_argument('--cfg', '-c', nargs='+', type=str, default='cfg/net/dual-encoder-unet.cfg',
                        help='Specify the paths to the configuration files for the desired networks. Can input multiple files for ensemble evaluation.')

    parser.add_argument('--model', nargs='+', default='MODEL.pth', metavar='FILE',
                        help='Specify the paths to the files where the trained models are stored. Can input multiple models for ensemble evaluation.')

    parser.add_argument('--input-images', '-i', metavar='INPUT', required=True,
                        help='The path to a txt file containing the list of image files to be evaluated.')

    parser.add_argument('--input-masks', '-m', metavar='INPUT', required=True,
                        help='The path to a txt file containing the list of corresponding ground truth masks for the input images.')

    parser.add_argument('--transforms', '-t', nargs='+',
                        help='List of transformations to be applied on each image before evaluation. Possible transformations: vflip, hflip, rotation_90, rotation_180, rotation_270.')

    parser.add_argument('--org-dim', action='store_true', default=False,
                        help='If set, the network will also evaluate with the original dimensions of the input image, in addition to the dimensions specified in the configuration file.')

    parser.add_argument('--viz', '-v', action='store_true',
                        help='If set, the images and their corresponding segmentation results will be displayed as they are processed.')

    parser.add_argument('--save', '-s', action='store_true', default=False,
                        help='If set, the output masks from the evaluation will be saved.')

    parser.add_argument('--save-dir', '-sd', type=str, default='eval_out',
                        help='Specify the directory where the output masks should be saved. Default is "eval_out".')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    assert len(args.model) == len(args.cfg)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    nets = []
    nets_cfg = []

    input_width = None
    input_height = None
    input_channels = None
    n_classes = None

    for tmp_net_path, tmp_net_cfg_path in zip(args.model, args.cfg):
        tmp_cfg = parse_net_cfg(tmp_net_cfg_path)
        logging.info(print_net_cfg(tmp_cfg, ret=True))

        tmp_net = get_net_from_config(tmp_cfg)

        logging.info(f'Loading model {tmp_net_path}')

        tmp_net.to(device=device)
        tmp_net.load_state_dict(torch.load(tmp_net_path, map_location=device))

        logging.info('Model loaded!')

        nets.append(tmp_net)
        nets_cfg.append(tmp_cfg)

        if input_width is None:
            input_width = tmp_cfg['input_size'][1]

        if input_height is None:
            input_height = tmp_cfg['input_size'][0]

        if input_channels is None:
            input_channels = tmp_net.n_input_channels

        if n_classes is None:
            n_classes = tmp_net.n_classes

        assert input_width == tmp_cfg['input_size'][1] and \
               input_height == tmp_cfg['input_size'][0] and \
               input_channels == tmp_net.n_input_channels and \
               n_classes == tmp_net.n_classes

    transforms = reversed_transforms = None
    if args.transforms is not None:
        transforms, reversed_transforms = transforms_names_list_to_torch(args.transforms)

    evaluate_metrics_names, evaluate_metrics_net_dim, evaluate_metrics_org_dim, evaluate_metrics_net_dim_avg, evaluate_metrics_org_dim_avg = evaluate_full(
        nets=nets, nets_cfg=nets_cfg,
        device=device,
        transforms=transforms, reversed_transforms=reversed_transforms,
        images_file_path=args.input_images, masks_file_path=args.input_masks,
        visualize=args.viz,
        save_output=args.save, save_dir=args.save_dir,
        eval_org_dim=args.org_dim
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    evaluate_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    summary_csv_file = save_dir / 'evaluate_summary.csv'

    csv_keys = 'id,cfg,model,images,masks,dimension,transforms'
    for key in evaluate_metrics_names:
        csv_keys += f',{key}'

    print("\nevaluate_score_net_dim:\n")

    csv_line_net_dim = '"' + evaluate_id + '"' + ',' + \
                       '"' + ','.join(args.cfg) + '"' + ',' + \
                       '"' + ','.join(args.model) + '"' + ',' + \
                       '"' + args.input_images + '"' + ',' + \
                       '"' + args.input_masks + '"' + ',' + \
                       '"net_dim",' + \
                       '"' + ('None' if args.transforms is None else ','.join(args.transforms)) + '"'

    for key, value in zip(evaluate_metrics_names, evaluate_metrics_net_dim_avg):
        csv_line_net_dim += f',{value}'

        print(f"{key}: {value}")

    if args.org_dim:
        print("\nevaluate_score_org_dim:\n")

        csv_line_org_dim = '"' + evaluate_id + '"' + ',' + \
                           '"' + ','.join(args.cfg) + '"' + ',' + \
                           '"' + ','.join(args.model) + '"' + ',' + \
                           '"' + args.input_images + '"' + ',' + \
                           '"' + args.input_masks + '"' + ',' + \
                           '"org_dim",' + \
                           '"' + ('None' if args.transforms is None else ','.join(args.transforms)) + '"'

        for key, value in zip(evaluate_metrics_names, evaluate_metrics_org_dim_avg):
            csv_line_org_dim += f',{value}'

            print(f"{key}: {value}")

        if summary_csv_file.exists():
            with open(summary_csv_file, 'a') as file:
                file.write(f'{csv_line_net_dim}\n')
                file.write(f'{csv_line_org_dim}\n')
        else:
            with open(summary_csv_file, 'w') as file:
                file.write(f'{csv_keys}\n')
                file.write(f'{csv_line_net_dim}\n')
                file.write(f'{csv_line_org_dim}\n')
    else:
        if summary_csv_file.exists():
            with open(summary_csv_file, 'a') as file:
                file.write(f'{csv_line_net_dim}\n')
        else:
            with open(summary_csv_file, 'w') as file:
                file.write(f'{csv_keys}\n')
                file.write(f'{csv_line_net_dim}\n')

    images_path = SkinLesionDataset.load_txt(args.input_images)
    masks_path = SkinLesionDataset.load_txt(args.input_masks)

    with open(save_dir / f'evaluate_{evaluate_id}_net_dim.csv', 'w') as file:
        file.write('image,mask')
        for key in evaluate_metrics_names:
            file.write(f',{key}')
        file.write('\n')

        for image_path, mask_path, metrics in zip(images_path, masks_path, evaluate_metrics_net_dim):
            file.write(f'"{image_path}","{mask_path}"')

            for value in metrics:
                file.write(f',{value}')

            file.write('\n')

    if args.org_dim:
        with open(save_dir / f'evaluate_{evaluate_id}_org_dim.csv', 'w') as file:
            file.write('image,mask')
            for key in evaluate_metrics_names:
                file.write(f',{key}')
            file.write('\n')

            for image_path, mask_path, metrics in zip(images_path, masks_path, evaluate_metrics_org_dim):
                file.write(f'"{image_path}","{mask_path}"')

                for value in metrics:
                    file.write(f',{value}')

                file.write('\n')
