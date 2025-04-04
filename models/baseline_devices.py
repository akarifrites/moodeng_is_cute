import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

from models.helpers.utils import make_divisible


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    

class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # create the three factorized convs that make up the inverted bottleneck block
        exp_conv = Conv2dNormActivation(in_channels,
                                        exp_channels,
                                        kernel_size=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False
                                        )

        # depthwise convolution with possible stride
        depth_conv = Conv2dNormActivation(exp_channels,
                                          exp_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=exp_channels,
                                          norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.ReLU,
                                          inplace=False
                                          )

        proj_conv = Conv2dNormActivation(exp_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=None,
                                         inplace=False
                                         )
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                # average pooling required for shortcut
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(
            exp_conv,
            depth_conv,
            proj_conv
        )
        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_shortcut:
            # x = self.ff.add(self.block(x) , self.shortcut(x))
            x = self.block(x) + self.shortcut(x)
        else:
            x = self.block(x)
        x = self.after_block_activation(x)
        return x


class Network(nn.Module):
    def __init__(self, config, embed_dim=32):
        super(Network, self).__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0

        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 channels_per_stage[0] // 4,
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
            Conv2dNormActivation(channels_per_stage[0] // 4,
                                 channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        # Final feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels_per_stage[-1], 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )


        # **Device Embedding Layer**
        self.device_embedding = nn.Embedding(9, embed_dim)  # Assuming 9 device IDs

        # **Classifier that fuses extracted features with device embeddings**
        self.classifier = nn.Sequential(
            # nn.Linear(544, 128),
            nn.Linear(512 + embed_dim, 128),  # Concatenate features and device embeddings
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

        # TO save parameters, a linear layer can be replaced by a Conv1D layer (see gpt)

        # ff_list = []
        # ff_list += [nn.Conv2d(
        #     channels_per_stage[-1],
        #     n_classes,
        #     kernel_size=(1, 1),
        #     stride=(1, 1),
        #     padding=0,
        #     bias=False),
        #     nn.BatchNorm2d(n_classes),
        # ]

        # ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        # self.feed_forward = nn.Sequential(
        #     *ff_list
        # )

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate
                    ):

        block = Block(in_channels,
                      out_channels,
                      expansion_rate,
                      stride
                      )
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x, device_id):
        x = self._forward_conv(x)
        # print("After _forward_conv: min={}, max={}, mean={}".format(x.min().item(), x.max().item(), x.mean().item()))
        # x = self.feed_forward(x)
        x = self.feature_extractor(x)
        # print("After feature_extractor: min={}, max={}, mean={}".format(x.min().item(), x.max().item(), x.mean().item()))
        features = x.squeeze(2).squeeze(2)  # Flatten
        # print(f"Features shape: {features.shape}")  # Debugging

        # Ensure device_id is a tensor and has correct dtype
        if isinstance(device_id, list):
            device_id = torch.tensor(device_id, dtype=torch.long, device=features.device)
        else:
            device_id = device_id.to(torch.long).to(features.device)
        # print(f"Device ID shape: {device_id.shape}, dtype: {device_id.dtype}")  # Debugging
        # assert device_id.dtype == torch.long, "device_id must be a long tensor"
        
        # Get device embeddings
        device_features = self.device_embedding(device_id)
        # print(f"Device features shape: {device_features.shape}")  # Debugging

        assert device_id.max() < self.device_embedding.num_embeddings, "Device index out of range!"

        # Concatenate extracted features with device embeddings
        combined_features = torch.cat((features, device_features), dim=1)
        # print(f"Combined features shape: {combined_features.shape}")  # Debugging

        # Final classification
        # print(f"Classifier Input Shape: {combined_features.shape}")
        logits = self.classifier(combined_features)
        # logits = x.squeeze(2).squeeze(2)
        return logits


def get_model(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8, expansion_rate=2.1,
              n_blocks=(3, 2, 1), strides=None):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @return: full neural network model based on the specified configs
    """

    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides
    }

    m = Network(model_config)
    return m
