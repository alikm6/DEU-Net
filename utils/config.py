import configparser
from typing import Type


def decode_str_to_tuple(text, delimiter=',', _type: Type = int):
    if text.lower() == 'none':
        return None
    else:
        return tuple(_type(x) for x in text.split(delimiter))


def decode_str_to_list(text, delimiter=',', _type: Type = int):
    if text.lower() == 'none':
        return None
    else:
        return list(_type(x) for x in text.split(delimiter))


def decode_str_to_dict(text, delimiter=';', _type: Type = str):
    if text.lower() == 'none':
        return None
    else:
        out = dict()

        for item in text.split(delimiter):
            key, value = item.split(':')

            out[key] = _type(value)

        return out


def parse_net_cfg(file):
    config = configparser.ConfigParser()
    config.read(file)

    cfg = {
        'type': config['net'].get('type', 'dual-encoder-unet').lower(),
        'input_size': decode_str_to_tuple(
            config['net'].get('input_size', '224,224'),
            _type=int
        ),
        'num_output_channels': config['net'].getint('num_output_channels', 2),
    }

    if cfg['type'] == 'dual-encoder-unet':
        cfg['encoder1'] = config['net'].get('encoder1', 'resnet34').lower()
        cfg['encoder1_pretrained'] = config['net'].getboolean('encoder1_pretrained', True)
        cfg['encoder2'] = config['net'].get('encoder2', 'maxvit_t').lower()
        cfg['encoder2_pretrained'] = config['net'].getboolean('encoder2_pretrained', True)

        cfg['decoder_up_sample_bilinear'] = config['net'].getboolean('decoder_up_sample_bilinear', False)
        cfg['decoder_n_output_channels'] = decode_str_to_tuple(
            config['net'].get('decoder_n_output_channels', 'none'),
            _type=int
        )
        cfg['decoder_input_from_encoder1_blocks'] = decode_str_to_tuple(
            config['net'].get('decoder_input_from_encoder1_blocks', 'none'),
            _type=int
        )
        cfg['decoder_input_from_encoder2_blocks'] = decode_str_to_tuple(
            config['net'].get('decoder_input_from_encoder2_blocks', 'none'),
            _type=int
        )
        cfg['decoder_input_from_main_image'] = config['net'].getboolean('decoder_input_from_main_image', True)

        cfg['decoder_conv_mid_channels_scale_factor'] = config['net'].getint(
            'decoder_conv_mid_channels_scale_factor', 1)
        cfg['decoder_merge_operation'] = config['net'].get('decoder_merge_operation', 'concat').lower()

    return cfg


def print_net_cfg(cfg, ret=False):
    text = "net:\n"

    for key in cfg.keys():
        text += f"\t{key}: {cfg[key]}\n"

    if ret:
        return text
    else:
        print(text)


def get_net_from_config(net_cfg: dict):
    if net_cfg['type'] == 'dual-encoder-unet':
        from nets.dual_encoder_unet import DualEncoderUNet

        net = DualEncoderUNet(
            n_classes=net_cfg['num_output_channels'],
            encoder1=net_cfg['encoder1'],
            encoder1_pretrained=net_cfg['encoder1_pretrained'],
            encoder2=net_cfg['encoder2'],
            encoder2_pretrained=net_cfg['encoder2_pretrained'],
            decoder_up_sample_bilinear=net_cfg['decoder_up_sample_bilinear'],
            decoder_n_output_channels=net_cfg['decoder_n_output_channels'],
            decoder_input_from_encoder1_blocks=net_cfg['decoder_input_from_encoder1_blocks'],
            decoder_input_from_encoder2_blocks=net_cfg['decoder_input_from_encoder2_blocks'],
            decoder_input_from_main_image=net_cfg['decoder_input_from_main_image'],
            decoder_conv_mid_channels_scale_factor=net_cfg['decoder_conv_mid_channels_scale_factor'],
            decoder_merge_operation=net_cfg['decoder_merge_operation']
        )
    else:
        raise ValueError("net type is invalid.")

    return net


def parse_train_cfg(file):
    config = configparser.ConfigParser()
    config.read(file)

    cfg = {
        'epoch': {
            'count': config['epoch'].getint('count', 50),
            'unfreeze_at': config['epoch'].getint('unfreeze_at', 0),
        },
        'dataset': {
            'image_size': decode_str_to_tuple(
                config['dataset'].get('image_size', '224,224'),
                _type=int
            ),
            'batch_size': config['dataset'].getint('batch_size', 2),
            'num_workers': config['dataset'].getint('num_workers', 4),
        },
        'dataset_augmentation': {
            'enable': config['dataset_augmentation'].getboolean('enable', True),
            'prob': config['dataset_augmentation'].getfloat('prob', 0.5),
            'rotation_range': decode_str_to_tuple(
                config['dataset_augmentation'].get('rotation_range', '-15,15'),
                _type=int
            ),
            'hflip_prob': config['dataset_augmentation'].getfloat('hflip_prob', 0.5),
            'vflip_prob': config['dataset_augmentation'].getfloat('vflip_prob', 0.5),
            'brightness': config['dataset_augmentation'].getfloat('brightness', 0.03),
            'contrast': config['dataset_augmentation'].getfloat('contrast', 0.03),
            'saturation': config['dataset_augmentation'].getfloat('saturation', 0.03),
            'hue': config['dataset_augmentation'].getfloat('hue', 0.03),
        },
        'loss': {
            'dice_factor': config['loss'].getfloat('dice_factor', 0.4),
            'ce_factor': config['loss'].getfloat('ce_factor', 0.6),
        },
        'optim': {
            'type': config['optim'].get('type', 'adam').lower(),
            'lr': config['optim'].getfloat('lr', 0.001),
            'weight_decay': config['optim'].getfloat('weight_decay', 0),
            'betas': decode_str_to_tuple(
                config['optim'].get('betas', '0.5,0.999'),
                _type=float
            ),
            'momentum': config['optim'].getfloat('momentum', 0.9),
        },
        'optim_lr_schedule': {
            'enable': config['optim_lr_schedule'].getboolean('enable', False),
            'type': config['optim_lr_schedule'].get('type', 'steps').lower(),
            'steps': decode_str_to_tuple(
                config['optim_lr_schedule'].get('steps', '0.8,0.9'),
                _type=float
            ),
            'scales': decode_str_to_tuple(
                config['optim_lr_schedule'].get('scales', '0.1,0.1'),
                _type=float
            ),
            'exponent': config['optim_lr_schedule'].getfloat('exponent', 0.9),
            'eta_min': config['optim_lr_schedule'].getfloat('eta_min', 0),
            'first_restart_at': config['optim_lr_schedule'].getint('first_restart_at', 10),
            'after_restart_factor': config['optim_lr_schedule'].getint('after_restart_factor', 1),
        },
        'checkpoint': {
            'enable': config['checkpoint'].getboolean('enable', True),
            'step': float(eval(config['checkpoint'].get('step', "0.1"))),
        },
        'evaluate': {
            'enable': config['evaluate'].getboolean('enable', True),
            'step': float(eval(config['evaluate'].get('step', "0.1"))),
            'metric': config['evaluate'].get('metric', 'dice').lower(),
        },
        'log_tensorboard': {
            'enable': config['log_tensorboard'].getboolean('enable', True),
            'histogram': config['log_tensorboard'].getboolean('histogram', False),
            'sample_image_pred': config['log_tensorboard'].getboolean('sample_image_pred', False),
        },
        'log_wandb': {
            'enable': config['log_wandb'].getboolean('enable', True),
            'histogram': config['log_wandb'].getboolean('histogram', False),
            'sample_image_pred': config['log_wandb'].getboolean('sample_image_pred', False),
            'mode': config['log_wandb'].get('mode', 'offline').lower(),
        },
    }

    return cfg


def print_train_cfg(cfg, ret=False):
    text = ""

    for sec in cfg.keys():
        text += f"{sec}:\n"

        for key in cfg[sec].keys():
            text += f"\t{key}: {cfg[sec][key]}\n"

    if ret:
        return text
    else:
        print(text)


if __name__ == '__main__':
    print_net_cfg(parse_net_cfg('../cfg/net/dual-encoder-unet.cfg'))
    print_train_cfg(parse_train_cfg('../cfg/train.cfg'))
