import argparse
import logging
from typing import Optional
from pathlib import Path

import torch
import torch.nn.functional as f
from torchvision import transforms as t

from utils.dataset import SkinLesionDataset
from utils.config import parse_net_cfg, print_net_cfg, get_net_from_config
from utils.utils import mask_to_image, plot_img_and_mask


def convert_pred_score_to_01(
        pred_masks_score: torch.Tensor,
        n_classes: int
):
    if n_classes == 1:
        pred_mask = (pred_masks_score > 0.5).long()
    else:
        pred_mask = f.one_hot(pred_masks_score.argmax(dim=1), n_classes).permute(0, 3, 1, 2).long()

    return pred_mask


def resize_pred_score(
        pred_mask_score: torch.Tensor,
        new_width: int,
        new_height: int,
):
    resize_tf = t.Resize((new_height, new_width), antialias=True)

    pred_mask_score_resized = resize_tf(pred_mask_score)

    return pred_mask_score_resized


def transforms_names_list_to_torch(names: list):
    transforms = reversed_transforms = []

    for transform_name in names:
        transform_name = transform_name.lower()

        assert transform_name in ['vflip', 'hflip', 'rotation_90', 'rotation_180', 'rotation_270']

        if transform_name == 'vflip':
            transforms.append(t.RandomVerticalFlip(1))
            reversed_transforms.append(t.RandomVerticalFlip(1))
        elif transform_name == 'hflip':
            transforms.append(t.RandomHorizontalFlip(1))
            reversed_transforms.append(t.RandomHorizontalFlip(1))
        elif transform_name == 'rotation_90':
            transforms.append(t.RandomRotation((90, 90)))
            reversed_transforms.append(t.RandomRotation((-90, -90)))
        elif transform_name == 'rotation_180':
            transforms.append(t.RandomRotation((180, 180)))
            reversed_transforms.append(t.RandomRotation((180, 180)))
        elif transform_name == 'rotation_270':
            transforms.append(t.RandomRotation((270, 270)))
            reversed_transforms.append(t.RandomRotation((-270, -270)))

    return transforms, reversed_transforms


def predict_images(
        net,
        images: torch.Tensor,
        device
):
    net.eval()

    images = images.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # predict the mask
        pred_masks_score = net(images)

        if net.n_classes == 1:
            pred_masks_score = torch.sigmoid(pred_masks_score)
        else:
            pred_masks_score = f.softmax(pred_masks_score, dim=1)

    return pred_masks_score


# integrated_test_time_augmentation
def predict_image_ITTA(
        nets: list, nets_cfg: list,
        image: torch.Tensor,
        device,
        transforms: Optional[list] = None, reversed_transforms: Optional[list] = None,
):
    assert len(nets) == len(nets_cfg)
    assert image.dim() == 3

    input_width = nets_cfg[0]['input_size'][1]
    input_height = nets_cfg[0]['input_size'][0]
    input_channels = nets[0].n_input_channels

    assert input_width == image.size(dim=2)
    assert input_height == image.size(dim=1)
    assert input_channels == image.size(dim=0)

    n_classes = nets[0].n_classes

    assert (transforms is None and reversed_transforms is None) or (len(transforms) == len(reversed_transforms))

    transforms_count = 0
    if transforms is not None:
        transforms_count = len(transforms)

    image = image.to(device=device)

    images = torch.zeros((transforms_count + 1, input_channels, input_height, input_width), dtype=torch.float32).to(
        device=device)
    images[0] = image

    if transforms is not None:
        for transform_i, transform in enumerate(transforms):
            images[transform_i + 1] = transform(image)

    pred_masks_score = torch.zeros((len(nets), transforms_count + 1, n_classes, input_height, input_width),
                                   dtype=torch.float32).to(device=device)

    for net_i, (net, net_cfg) in enumerate(zip(nets, nets_cfg)):
        assert input_channels == net.n_input_channels and n_classes == net.n_classes
        assert input_width == net_cfg['input_size'][1] and input_height == net_cfg['input_size'][0]

        pred_masks_score[net_i] = predict_images(net=net, images=images, device=device)

        if transforms is not None:
            for reversed_transform_i, reversed_transform in enumerate(reversed_transforms):
                pred_masks_score[net_i, reversed_transform_i + 1] = reversed_transform(
                    pred_masks_score[net_i, reversed_transform_i + 1])

    pred_mask_score = torch.mean(pred_masks_score.view(-1, n_classes, input_height, input_width), dim=0)

    pred_mask = convert_pred_score_to_01(pred_mask_score.unsqueeze(0), n_classes)
    pred_mask = pred_mask.squeeze(0)

    return pred_mask_score, pred_mask


def get_args():
    parser = argparse.ArgumentParser(description='Generate segmentation masks for input images using trained models.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Use this flag if you don\'t want to use CUDA, even if it is available.')

    parser.add_argument('--cfg', '-c', nargs='+', type=str, default='cfg/net/dual-encoder-unet.cfg',
                        help='Specify the paths to the configuration files for the desired networks. Can input multiple files for ensemble prediction.')

    parser.add_argument('--model', nargs='+', default='MODEL.pth', metavar='FILE',
                        help='Specify the paths to the files where the trained models are stored. Can input multiple models for ensemble prediction.')

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', required=True,
                        help='The paths to the image files to be segmented. Can input multiple images.')

    parser.add_argument('--transforms', '-t', nargs='+',
                        help='List of transformations to be applied on each image before prediction. Possible transformations: vflip, hflip, rotation_90, rotation_180, rotation_270.')

    parser.add_argument('--org-dim', action='store_true', default=False,
                        help='If set, the network will generate output masks with the original dimensions of the input image, in addition to the dimensions specified in the configuration file.')

    parser.add_argument('--output-dir', default="predict_out",
                        help='Specify the directory where the output masks should be saved. Default is "predict_out".')

    parser.add_argument('--save', '-s', action='store_true', default=False,
                        help='If set, the output masks from the prediction will be saved.')

    parser.add_argument('--viz', '-v', action='store_true', default=False,
                        help='If set, the images and their corresponding segmentation results will be displayed as they are processed.')

    return parser.parse_args()


def get_output_filenames(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_name(fn):
        return str(output_dir / (Path(fn).stem + '_OUT'))

    return list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

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

    for filename, out_filename in zip(in_files, out_files):
        logging.info(f'Predicting image {filename} ...')

        image_pil = SkinLesionDataset.load(filename)
        image_org_width, image_org_height = image_pil.size

        if args.org_dim:
            image_org_dim = SkinLesionDataset.preprocess(
                image_pil,
                is_mask=False,
                width=None,
                height=None
            )

        image_net_dim = SkinLesionDataset.preprocess(
            image_pil,
            is_mask=False,
            width=input_width,
            height=input_height
        )

        pred_mask_net_dim_score, pred_mask_net_dim = predict_image_ITTA(
            nets=nets, nets_cfg=nets_cfg,
            image=image_net_dim,
            device=device,
            transforms=transforms, reversed_transforms=reversed_transforms,
        )

        if args.org_dim:
            # resize pred_mask_net_dim_score to org_dim
            pred_mask_org_dim_score = resize_pred_score(
                pred_mask_net_dim_score,
                new_width=image_org_width,
                new_height=image_org_height
            )

            # calculate pred_mask_org_dim from pred_mask_org_dim_score
            pred_mask_org_dim = convert_pred_score_to_01(pred_mask_org_dim_score.unsqueeze(0), n_classes).squeeze(0)

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')

            if n_classes == 1:
                plot_img_and_mask(img=image_net_dim, pred_mask=pred_mask_net_dim)

                if args.org_dim:
                    plot_img_and_mask(img=image_org_dim, pred_mask=pred_mask_org_dim)
            else:
                # ignore background
                plot_img_and_mask(img=image_net_dim, pred_mask=pred_mask_net_dim[1:, ...])

                if args.org_dim:
                    plot_img_and_mask(img=image_org_dim, pred_mask=pred_mask_org_dim[1:, ...])

        if args.save:
            pred_mask_net_dim_img = mask_to_image(pred_mask_net_dim)
            pred_mask_net_dim_img.save(f'{out_filename}_net_dim.png')

            if args.org_dim:
                pred_mask_org_dim_img = mask_to_image(pred_mask_org_dim)
                pred_mask_org_dim_img.save(f'{out_filename}_org_dim.png')

                logging.info(f'Mask saved to {out_filename}_net_dim.png and {out_filename}_org_dim.png')
            else:
                logging.info(f'Mask saved to {out_filename}_net_dim.png')
