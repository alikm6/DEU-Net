from typing import Optional

import timm
import torch
from torch import nn
from torchvision import models

from .blocks import Permute, View


class EncoderOutputPreprocessBlock(nn.Module):
    def __init__(self, module, pass_input_size_to_module: bool = False):
        super(EncoderOutputPreprocessBlock, self).__init__()

        self.module = module
        self.pass_input_size_to_module = pass_input_size_to_module

    def forward(self, x, input_size):
        if not self.pass_input_size_to_module:
            return self.module(x)
        else:
            return self.module(x, input_size)


class deit_OutputPreprocessBlock(nn.Module):
    def __init__(self):
        super(deit_OutputPreprocessBlock, self).__init__()

    def forward(self, x, input_size):
        x = x.permute(0, 2, 1)
        x = x.view(-1, 192, input_size[0] // 16, input_size[1] // 16)

        return x


def get_encoder(name: str, pretrained: bool = True, blocks_count: Optional[int] = None, prefix: str = 'encoder_'):
    if blocks_count is not None and blocks_count < 1:
        raise ValueError("Invalid encoder_blocks_count.")

    blocks_output_preprocess = None

    if name == 'efficientnet_b4' or name == 'efficientnet_b5' or name == 'efficientnet_b6' or name == 'efficientnet_b7':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError(f"Invalid encoder_blocks_count, maximum value for {name} is 5.")

        n_input_channels = 3

        if name == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            encoder = models.efficientnet_b4(weights=weights)
        elif name == 'efficientnet_b5':
            weights = models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            encoder = models.efficientnet_b5(weights=weights)
        elif name == 'efficientnet_b6':
            weights = models.EfficientNet_B6_Weights.DEFAULT if pretrained else None
            encoder = models.efficientnet_b6(weights=weights)
        else:
            weights = models.EfficientNet_B7_Weights.DEFAULT if pretrained else None
            encoder = models.efficientnet_b7(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.features[0],
                encoder.features[1]
            ),
            f'{prefix}block1': encoder.features[2],
            f'{prefix}block2': encoder.features[3],
            f'{prefix}block3': nn.Sequential(
                encoder.features[4],
                encoder.features[5]
            ),
            f'{prefix}block4': nn.Sequential(
                encoder.features[6],
                encoder.features[7],
                encoder.features[8]
            )
        })

        if name == 'efficientnet_b4':
            n_output_channels = [24, 32, 56, 160, 1792]
        elif name == 'efficientnet_b5':
            n_output_channels = [24, 40, 64, 176, 2048]
        elif name == 'efficientnet_b6':
            n_output_channels = [32, 40, 72, 200, 2304]
        else:
            n_output_channels = [32, 48, 80, 224, 2560]

        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'efficientnet_v2_s':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError(f"Invalid encoder_blocks_count, maximum value for {name} is 5.")

        n_input_channels = 3

        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        encoder = models.efficientnet_v2_s(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.features[0],
                encoder.features[1]
            ),
            f'{prefix}block1': encoder.features[2],
            f'{prefix}block2': encoder.features[3],
            f'{prefix}block3': nn.Sequential(
                encoder.features[4],
                encoder.features[5]
            ),
            f'{prefix}block4': nn.Sequential(
                encoder.features[6],
                encoder.features[7]
            )
        })

        n_output_channels = [24, 48, 64, 160, 1280]

        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'efficientnet_v2_m' or name == 'efficientnet_v2_l':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError(f"Invalid encoder_blocks_count, maximum value for {name} is 5.")

        n_input_channels = 3

        if name == 'efficientnet_v2_m':
            weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            encoder = models.efficientnet_v2_m(weights=weights)
        else:
            weights = models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            encoder = models.efficientnet_v2_l(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.features[0],
                encoder.features[1]
            ),
            f'{prefix}block1': encoder.features[2],
            f'{prefix}block2': encoder.features[3],
            f'{prefix}block3': nn.Sequential(
                encoder.features[4],
                encoder.features[5]
            ),
            f'{prefix}block4': nn.Sequential(
                encoder.features[6],
                encoder.features[7],
                encoder.features[8]
            )
        })

        if name == 'efficientnet_v2_m':
            n_output_channels = [24, 48, 80, 176, 1280]
        else:
            n_output_channels = [32, 64, 96, 224, 1280]

        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'maxvit_t':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError("Invalid encoder_blocks_count, maximum value for maxvit_t is 5.")

        n_input_channels = 3

        weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        encoder = models.maxvit_t(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': encoder.stem,
            f'{prefix}block1': encoder.blocks[0],
            f'{prefix}block2': encoder.blocks[1],
            f'{prefix}block3': encoder.blocks[2],
            f'{prefix}block4': encoder.blocks[3]
        })
        n_output_channels = [64, 64, 128, 256, 512]
        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'maxvit_tiny_tf_384':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError("Invalid encoder_blocks_count, maximum value for maxvit_tiny_tf_384 is 5.")

        n_input_channels = 3

        encoder = timm.create_model(
            'maxvit_tiny_tf_384.in1k',
            pretrained=pretrained,
            features_only=True,
        )

        blocks = nn.ModuleDict({
            f'{prefix}block0': encoder.stem,
            f'{prefix}block1': encoder.stages_0,
            f'{prefix}block2': encoder.stages_1,
            f'{prefix}block3': encoder.stages_2,
            f'{prefix}block4': encoder.stages_3
        })
        n_output_channels = [64, 64, 128, 256, 512]
        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'maxvit_small_tf_224':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError("Invalid encoder_blocks_count, maximum value for maxvit_small_tf_224 is 5.")

        n_input_channels = 3

        encoder = timm.create_model(
            'maxvit_small_tf_224.in1k',
            pretrained=pretrained,
            features_only=True,
        )

        blocks = nn.ModuleDict({
            f'{prefix}block0': encoder.stem,
            f'{prefix}block1': encoder.stages_0,
            f'{prefix}block2': encoder.stages_1,
            f'{prefix}block3': encoder.stages_2,
            f'{prefix}block4': encoder.stages_3
        })
        n_output_channels = [64, 96, 192, 384, 768]
        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'resnet34' or name == 'resnet50':
        if blocks_count is None:
            blocks_count = 5

        if blocks_count > 5:
            raise ValueError(f"Invalid encoder_blocks_count, maximum value for {name} is 5.")

        n_input_channels = 3

        if name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            encoder = models.resnet34(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            encoder = models.resnet50(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu
            ),
            f'{prefix}block1': nn.Sequential(
                encoder.maxpool,
                encoder.layer1
            ),
            f'{prefix}block2': encoder.layer2,
            f'{prefix}block3': encoder.layer3,
            f'{prefix}block4': encoder.layer4
        })

        if name == 'resnet34':
            n_output_channels = [64, 64, 128, 256, 512]
        else:
            n_output_channels = [64, 256, 512, 1024, 2048]

        dimension_reduction = [2, 2, 2, 2, 2]

    elif name == 'convnext_tiny':
        if blocks_count is None:
            blocks_count = 4

        if blocks_count > 4:
            raise ValueError("Invalid encoder_blocks_count, maximum value for convnext_tiny is 4.")

        n_input_channels = 3

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        encoder = models.convnext_tiny(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.features[0],
                encoder.features[1],
            ),
            f'{prefix}block1': nn.Sequential(
                encoder.features[2],
                encoder.features[3],
            ),
            f'{prefix}block2': nn.Sequential(
                encoder.features[4],
                encoder.features[5],
            ),
            f'{prefix}block3': nn.Sequential(
                encoder.features[6],
                encoder.features[7],
            ),
        })
        n_output_channels = [96, 192, 384, 768]
        dimension_reduction = [4, 2, 2, 2]

    elif name == 'swin_v2_t':
        if blocks_count is None:
            blocks_count = 4

        if blocks_count > 4:
            raise ValueError("Invalid encoder_blocks_count, maximum value for swin_v2_t is 4.")

        n_input_channels = 3

        weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
        encoder = models.swin_v2_t(weights=weights)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.features[0],
                encoder.features[1],
            ),
            f'{prefix}block1': nn.Sequential(
                encoder.features[2],
                encoder.features[3],
            ),
            f'{prefix}block2': nn.Sequential(
                encoder.features[4],
                encoder.features[5],
            ),
            f'{prefix}block3': nn.Sequential(
                encoder.features[6],
                encoder.features[7],
            ),
        })
        n_output_channels = [96, 192, 384, 768]
        dimension_reduction = [4, 2, 2, 2]

        blocks_output_preprocess = nn.ModuleDict({
            f'{prefix}block0_output_preprocess': EncoderOutputPreprocessBlock(
                module=Permute(0, 3, 1, 2)
            ),
            f'{prefix}block1_output_preprocess': EncoderOutputPreprocessBlock(
                module=Permute(0, 3, 1, 2)
            ),
            f'{prefix}block2_output_preprocess': EncoderOutputPreprocessBlock(
                module=Permute(0, 3, 1, 2)
            ),
            f'{prefix}block3_output_preprocess': EncoderOutputPreprocessBlock(
                module=nn.Sequential(
                    encoder.norm,
                    encoder.permute,
                )
            ),
        })

    elif name == 'deit_tiny_distilled_patch16_224':
        if blocks_count is None:
            blocks_count = 1

        if blocks_count > 1:
            raise ValueError("Invalid encoder_blocks_count, maximum value for swin_v2_t is 4.")

        n_input_channels = 3

        encoder = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=pretrained)

        blocks = nn.ModuleDict({
            f'{prefix}block0': nn.Sequential(
                encoder.patch_embed,
                *[encoder.blocks[i] for i in range(12)],
            ),
        })
        n_output_channels = [192]
        dimension_reduction = [16]

        blocks_output_preprocess = nn.ModuleDict({
            f'{prefix}block0_output_preprocess': EncoderOutputPreprocessBlock(
                module=deit_OutputPreprocessBlock(),
                pass_input_size_to_module=True
            ),
        })

    else:
        raise ValueError("Invalid encoder.")

    if blocks_count < len(blocks):
        blocks = nn.ModuleDict(dict(list(blocks.items())[:blocks_count]))
        n_output_channels = n_output_channels[:blocks_count]
        dimension_reduction = dimension_reduction[:blocks_count]

        if blocks_output_preprocess is not None:
            blocks_output_preprocess = nn.ModuleDict(dict(list(blocks_output_preprocess.items())[:blocks_count]))

    return blocks, n_input_channels, n_output_channels, dimension_reduction, blocks_output_preprocess
