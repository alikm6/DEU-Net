from typing import Optional, Union

import torch
import torch.nn as nn

from .utils.encoders import get_encoder
from .utils.blocks import DoubleBNReLUBlock, UpSampleBlock, ConvBNReLUBlock, SEBlock


class DualEncoderUNet(nn.Module):
    def __init__(
            self,
            n_classes: int = 2,
            encoder1: str = 'efficientnet_b4',
            encoder1_pretrained: bool = True,
            encoder2: str = 'deit_tiny_distilled_patch16_224',
            encoder2_pretrained: bool = True,
            decoder_up_sample_bilinear: bool = False,
            decoder_n_output_channels: Optional[Union[list, tuple]] = None,
            decoder_input_from_encoder1_blocks: Optional[Union[list, tuple]] = None,
            decoder_input_from_encoder2_blocks: Optional[Union[list, tuple]] = None,
            decoder_input_from_main_image: bool = True,
            decoder_conv_mid_channels_scale_factor: int = 1,
            decoder_merge_operation: str = 'concat'
    ):
        super(DualEncoderUNet, self).__init__()

        # Load Encoder 1

        (
            encoder1_blocks,
            encoder1_input_num_channels,
            encoder1_outputs_num_channels,
            encoder1_outputs_dimension_reduction,
            encoder1_outputs_preprocess_blocks
        ) = get_encoder(name=encoder1, pretrained=encoder1_pretrained, blocks_count=None, prefix='encoder1_')

        encoder1_blocks_count = len(encoder1_blocks)
        encoder1_outputs_dimension_reduction_prod = self.list_prod(encoder1_outputs_dimension_reduction)

        if decoder_input_from_encoder1_blocks is None:
            decoder_input_from_encoder1_blocks = [x for x in range(encoder1_blocks_count)]
        else:
            decoder_input_from_encoder1_blocks = sorted(list(set(decoder_input_from_encoder1_blocks)))

            if decoder_input_from_encoder1_blocks[0] < 0 or \
                    decoder_input_from_encoder1_blocks[-1] > encoder1_blocks_count - 1:
                raise ValueError("Invalid decoder_input_from_encoder1_blocks.")

        if decoder_input_from_encoder1_blocks[-1] + 1 < encoder1_blocks_count:
            encoder1_blocks_count = decoder_input_from_encoder1_blocks[-1] + 1

            encoder1_blocks = self.slice_ModuleDict(encoder1_blocks, count=encoder1_blocks_count)
            encoder1_outputs_num_channels = encoder1_outputs_num_channels[:encoder1_blocks_count]
            encoder1_outputs_dimension_reduction = encoder1_outputs_dimension_reduction[:encoder1_blocks_count]

            if encoder1_outputs_preprocess_blocks is not None:
                encoder1_outputs_preprocess_blocks = self.slice_ModuleDict(encoder1_outputs_preprocess_blocks,
                                                                           count=encoder1_blocks_count)

            encoder1_outputs_dimension_reduction_prod = encoder1_outputs_dimension_reduction_prod[
                                                        :encoder1_blocks_count]

        # Load Encoder 2
        (
            encoder2_blocks,
            encoder2_input_num_channels,
            encoder2_outputs_num_channels,
            encoder2_outputs_dimension_reduction,
            encoder2_outputs_preprocess_blocks
        ) = get_encoder(name=encoder2, pretrained=encoder2_pretrained, blocks_count=None, prefix='encoder2_')

        encoder2_blocks_count = len(encoder2_blocks)
        encoder2_outputs_dimension_reduction_prod = self.list_prod(encoder2_outputs_dimension_reduction)

        if decoder_input_from_encoder2_blocks is None:
            decoder_input_from_encoder2_blocks = [x for x in range(encoder2_blocks_count)]
        else:
            decoder_input_from_encoder2_blocks = sorted(list(set(decoder_input_from_encoder2_blocks)))

            if decoder_input_from_encoder2_blocks[0] < 0 or \
                    decoder_input_from_encoder2_blocks[-1] > encoder2_blocks_count - 1:
                raise ValueError("Invalid decoder_input_from_encoder2_blocks.")

        if decoder_input_from_encoder2_blocks[-1] + 1 < encoder2_blocks_count:
            encoder2_blocks_count = decoder_input_from_encoder2_blocks[-1] + 1

            encoder2_blocks = self.slice_ModuleDict(encoder2_blocks, count=encoder2_blocks_count)
            encoder2_outputs_num_channels = encoder2_outputs_num_channels[:encoder2_blocks_count]
            encoder2_outputs_dimension_reduction = encoder2_outputs_dimension_reduction[:encoder2_blocks_count]

            if encoder2_outputs_preprocess_blocks is not None:
                encoder2_outputs_preprocess_blocks = self.slice_ModuleDict(encoder2_outputs_preprocess_blocks,
                                                                           count=encoder2_blocks_count)

            encoder2_outputs_dimension_reduction_prod = encoder2_outputs_dimension_reduction_prod[
                                                        :encoder2_blocks_count]

        ##########

        if encoder1_input_num_channels != encoder2_input_num_channels:
            raise ValueError("Invalid encoders, n_input_channels should be equal in CNN and Transformer encoders.")

        ##########

        decoder_inputs_dimension_reduction = [encoder1_outputs_dimension_reduction_prod[i] for i in
                                              decoder_input_from_encoder1_blocks]
        decoder_inputs_dimension_reduction += [encoder2_outputs_dimension_reduction_prod[i] for i in
                                               decoder_input_from_encoder2_blocks]
        decoder_inputs_dimension_reduction = sorted(list(set(decoder_inputs_dimension_reduction)))

        ##########

        if decoder_n_output_channels is None:
            decoder_n_output_channels = []

            if decoder_inputs_dimension_reduction[0] in encoder1_outputs_dimension_reduction_prod:
                i = encoder1_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[0])

                if i in decoder_input_from_encoder1_blocks:
                    decoder_n_output_channels.append(
                        encoder1_outputs_num_channels[i] // encoder1_outputs_dimension_reduction[i])
                elif decoder_inputs_dimension_reduction[0] in encoder2_outputs_dimension_reduction_prod:
                    i = encoder2_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[0])

                    if i in decoder_input_from_encoder2_blocks:
                        decoder_n_output_channels.append(
                            encoder2_outputs_num_channels[i] // encoder2_outputs_dimension_reduction[i])
            else:
                i = encoder2_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[0])

                if i in decoder_input_from_encoder2_blocks:
                    decoder_n_output_channels.append(
                        encoder2_outputs_num_channels[i] // encoder2_outputs_dimension_reduction[i])

            for x in decoder_inputs_dimension_reduction[:-1]:
                if x in encoder1_outputs_dimension_reduction_prod:
                    i = encoder1_outputs_dimension_reduction_prod.index(x)

                    if i in decoder_input_from_encoder1_blocks:
                        decoder_n_output_channels.append(encoder1_outputs_num_channels[i])
                    elif x in encoder2_outputs_dimension_reduction_prod:
                        i = encoder2_outputs_dimension_reduction_prod.index(x)

                        if i in decoder_input_from_encoder2_blocks:
                            decoder_n_output_channels.append(encoder2_outputs_num_channels[i])
                else:
                    i = encoder2_outputs_dimension_reduction_prod.index(x)

                    if i in decoder_input_from_encoder2_blocks:
                        decoder_n_output_channels.append(encoder2_outputs_num_channels[i])
        else:
            decoder_n_output_channels = list(decoder_n_output_channels)

            if len(decoder_n_output_channels) != len(decoder_inputs_dimension_reduction):
                raise ValueError(
                    f"Invalid decoder_n_output_channels, len(decoder_n_output_channels) should be equal to {len(decoder_inputs_dimension_reduction)}.")

        ##########

        decoder_merge_operation = decoder_merge_operation.lower()
        if decoder_merge_operation not in ['concat', 'sum']:
            raise ValueError("Invalid decoder_merge_operation")

        if decoder_merge_operation == 'sum' and decoder_input_from_main_image:
            raise ValueError(
                "Invalid decoder_input_from_main_image, Because decoder_merge_operation is set to sum, the value of decoder_input_from_main_image cannot be True.")

        if decoder_merge_operation == 'sum' and decoder_input_from_main_image:
            raise ValueError("Invalid decoder_merge_operation")

        ##########

        self.n_classes = n_classes
        self.n_input_channels = encoder1_input_num_channels

        self.encoder1_name = encoder1
        self.encoder1_pretrained = encoder1_pretrained
        self.encoder1_blocks = encoder1_blocks
        self.encoder1_outputs_preprocess_blocks = encoder1_outputs_preprocess_blocks
        self.encoder1_outputs_dimension_reduction_prod = encoder1_outputs_dimension_reduction_prod

        self.encoder2_name = encoder2
        self.encoder2_pretrained = encoder2_pretrained
        self.encoder2_blocks = encoder2_blocks
        self.encoder2_outputs_preprocess_blocks = encoder2_outputs_preprocess_blocks
        self.encoder2_outputs_dimension_reduction_prod = encoder2_outputs_dimension_reduction_prod

        self.decoder_up_sample_bilinear = decoder_up_sample_bilinear
        self.decoder_n_output_channels = decoder_n_output_channels
        self.decoder_input_from_encoder1_blocks = decoder_input_from_encoder1_blocks
        self.decoder_input_from_encoder2_blocks = decoder_input_from_encoder2_blocks
        self.decoder_input_from_main_image = decoder_input_from_main_image
        self.decoder_inputs_dimension_reduction = decoder_inputs_dimension_reduction
        self.decoder_merge_operation = decoder_merge_operation

        ##########

        tmp_encoder_outputs_num_channels = []
        if decoder_input_from_main_image:
            tmp_encoder_outputs_num_channels.append(self.n_input_channels)
        else:
            tmp_encoder_outputs_num_channels.append(0)

        for x in decoder_inputs_dimension_reduction:
            if x in encoder1_outputs_dimension_reduction_prod:
                i = encoder1_outputs_dimension_reduction_prod.index(x)

                if i in decoder_input_from_encoder1_blocks:
                    tmp_encoder_outputs_num_channels.append(encoder1_outputs_num_channels[i])
                elif x in encoder2_outputs_dimension_reduction_prod:
                    i = encoder2_outputs_dimension_reduction_prod.index(x)

                    if i in decoder_input_from_encoder2_blocks:
                        tmp_encoder_outputs_num_channels.append(encoder2_outputs_num_channels[i])
            else:
                i = encoder2_outputs_dimension_reduction_prod.index(x)

                if i in decoder_input_from_encoder2_blocks:
                    tmp_encoder_outputs_num_channels.append(encoder2_outputs_num_channels[i])

        tmp_decoder_n_output_channels = list(self.decoder_n_output_channels)
        if decoder_inputs_dimension_reduction[-1] in encoder1_outputs_dimension_reduction_prod:
            i = encoder1_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[-1])

            if i in decoder_input_from_encoder1_blocks:
                tmp_decoder_n_output_channels.append(encoder1_outputs_num_channels[i])
            elif decoder_inputs_dimension_reduction[-1] in encoder2_outputs_dimension_reduction_prod:
                i = encoder2_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[-1])

                if i in decoder_input_from_encoder2_blocks:
                    tmp_decoder_n_output_channels.append(encoder2_outputs_num_channels[i])
        else:
            i = encoder2_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[-1])

            if i in decoder_input_from_encoder2_blocks:
                tmp_decoder_n_output_channels.append(encoder2_outputs_num_channels[i])

        self.encoder2_outputs_conv = nn.ModuleDict()
        self.combined_features_se_conv = nn.ModuleDict()

        self.decoder_up_sample_blocks = nn.ModuleDict()
        self.decoder_conv_blocks = nn.ModuleDict()

        for i in range(len(decoder_inputs_dimension_reduction)):
            if decoder_inputs_dimension_reduction[i] in encoder2_outputs_dimension_reduction_prod:
                tmp_i2 = encoder2_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[i])

                if tmp_i2 in decoder_input_from_encoder2_blocks:
                    self.encoder2_outputs_conv.update({
                        f'encoder2_outputs_conv_{tmp_i2}': ConvBNReLUBlock(
                            in_channels=encoder2_outputs_num_channels[tmp_i2],
                            out_channels=tmp_encoder_outputs_num_channels[i + 1]
                        )
                    })

                if decoder_inputs_dimension_reduction[i] in encoder1_outputs_dimension_reduction_prod:
                    tmp_i1 = encoder1_outputs_dimension_reduction_prod.index(decoder_inputs_dimension_reduction[i])

                    if tmp_i1 in decoder_input_from_encoder1_blocks:
                        self.combined_features_se_conv.update({
                            f'combined_features_se_conv_{i}': nn.Sequential(
                                SEBlock(channel=tmp_encoder_outputs_num_channels[i + 1] * 2),
                                nn.Conv2d(
                                    in_channels=tmp_encoder_outputs_num_channels[i + 1] * 2,
                                    out_channels=tmp_encoder_outputs_num_channels[i + 1],
                                    kernel_size=1,
                                    padding=0
                                )
                            )
                        })

            tmp_dr = decoder_inputs_dimension_reduction[i]
            if i != 0:
                tmp_dr //= decoder_inputs_dimension_reduction[i - 1]

            if decoder_merge_operation == 'concat' or i == 0:
                tmp_up_sample_out_channels = tmp_decoder_n_output_channels[i + 1] // tmp_dr
            else:
                tmp_up_sample_out_channels = tmp_encoder_outputs_num_channels[i]

            self.decoder_up_sample_blocks.update({
                f'decoder_up_sample_{i}': UpSampleBlock(
                    in_channels=tmp_decoder_n_output_channels[i + 1],
                    out_channels=tmp_up_sample_out_channels,
                    scale_factor=tmp_dr,
                    bilinear=decoder_up_sample_bilinear)
            })

            if decoder_merge_operation == 'concat' or i == 0:
                tmp_conv_in_channels = tmp_up_sample_out_channels + tmp_encoder_outputs_num_channels[i]
            else:
                tmp_conv_in_channels = tmp_encoder_outputs_num_channels[i]

            self.decoder_conv_blocks.update({
                f'decoder_conv_{i}': DoubleBNReLUBlock(
                    in_channels=tmp_conv_in_channels,
                    out_channels=tmp_decoder_n_output_channels[i],
                    mid_channels=tmp_decoder_n_output_channels[i] // decoder_conv_mid_channels_scale_factor
                )
            })

        #########

        self.out_conv = nn.Conv2d(self.decoder_n_output_channels[0], n_classes, kernel_size=1)

    @staticmethod
    def list_prod(my_list, output_type=int):
        r = []

        for i in range(1, len(my_list) + 1):
            r.append(output_type(torch.prod(torch.Tensor(my_list[:i])).item()))

        return r

    @staticmethod
    def slice_ModuleDict(module_dict: nn.ModuleDict, count: int):
        return nn.ModuleDict(dict(list(module_dict.items())[:count]))

    def freeze_pretrained_wight(self, freeze: bool):
        for element in self.encoder1_blocks.values():
            for param in element.parameters():
                param.requires_grad = not freeze

        if self.encoder1_outputs_preprocess_blocks is not None:
            for element in self.encoder1_outputs_preprocess_blocks.values():
                for param in element.parameters():
                    param.requires_grad = not freeze

        for element in self.encoder2_blocks.values():
            for param in element.parameters():
                param.requires_grad = not freeze

        if self.encoder2_outputs_preprocess_blocks is not None:
            for element in self.encoder2_outputs_preprocess_blocks.values():
                for param in element.parameters():
                    param.requires_grad = not freeze

    def forward(self, x):
        b, c, h, w = x.shape

        e1 = [x]  # encoder1 outputs
        e2 = [x]  # encoder2 outputs
        e = [x]  # double encoders outputs

        # Encoder 1
        for i in range(len(self.encoder1_blocks)):
            x = self.encoder1_blocks[f'encoder1_block{i}'](x)

            if self.encoder1_outputs_preprocess_blocks is None:
                e1.append(x)
            else:
                e1.append(self.encoder1_outputs_preprocess_blocks[f'encoder1_block{i}_output_preprocess'](x, (h, w)))

        # Encoder 2
        x = e2[0]
        for i in range(len(self.encoder2_blocks)):
            x = self.encoder2_blocks[f'encoder2_block{i}'](x)

            if self.encoder2_outputs_preprocess_blocks is None:
                e2.append(x)
            else:
                e2.append(self.encoder2_outputs_preprocess_blocks[f'encoder2_block{i}_output_preprocess'](x, (h, w)))

        # Dual Encoder
        for i in range(len(self.decoder_inputs_dimension_reduction)):
            tmp_i1 = None
            tmp_i2 = None

            if self.decoder_inputs_dimension_reduction[i] in self.encoder1_outputs_dimension_reduction_prod:
                tmp_i1 = self.encoder1_outputs_dimension_reduction_prod.index(
                    self.decoder_inputs_dimension_reduction[i])
                if tmp_i1 not in self.decoder_input_from_encoder1_blocks:
                    tmp_i1 = None

            if self.decoder_inputs_dimension_reduction[i] in self.encoder2_outputs_dimension_reduction_prod:
                tmp_i2 = self.encoder2_outputs_dimension_reduction_prod.index(
                    self.decoder_inputs_dimension_reduction[i])

                if tmp_i2 not in self.decoder_input_from_encoder2_blocks:
                    tmp_i2 = None

            if tmp_i1 is not None and tmp_i2 is not None:
                tmp_e1 = e1[tmp_i1 + 1]
                tmp_e2 = self.encoder2_outputs_conv[f'encoder2_outputs_conv_{tmp_i2}'](e2[tmp_i2 + 1])

                tmp_e = torch.cat((tmp_e1, tmp_e2), dim=1)
                tmp_e = self.combined_features_se_conv[f'combined_features_se_conv_{i}'](tmp_e)

                e.append(tmp_e)

            elif tmp_i1 is not None:
                tmp_e1 = e1[tmp_i1 + 1]

                e.append(tmp_e1)

            elif tmp_i2 is not None:
                tmp_e2 = self.encoder2_outputs_conv[f'encoder2_outputs_conv_{tmp_i2}'](e2[tmp_i2 + 1])

                e.append(tmp_e2)

        # Decoder
        x = e[-1]

        for i in range(len(self.decoder_inputs_dimension_reduction) - 1, -1, -1):
            x = self.decoder_up_sample_blocks[f'decoder_up_sample_{i}'](x)

            if i != 0 or self.decoder_input_from_main_image:
                if self.decoder_merge_operation == 'concat':
                    x = torch.cat((e[i], x), dim=1)
                else:
                    x = e[i] + x

            x = self.decoder_conv_blocks[f'decoder_conv_{i}'](x)

        x = self.out_conv(x)

        return x
