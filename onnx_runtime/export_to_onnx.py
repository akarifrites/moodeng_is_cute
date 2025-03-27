import torch
import torch.nn as nn
import torchaudio
from torchvision.ops.misc import Conv2dNormActivation
import onnx
import onnxruntime as ort
import os
from models.baseline_devices import get_model, initialize_weights, Block
from models.helpers.utils import make_divisible

class Network(nn.Module):
    def __init__(self, config):
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

        ff_list = []
        ff_list += [nn.Conv2d(
            channels_per_stage[-1],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]

        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

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

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        logits = x.squeeze(2).squeeze(2)
        return logits

def get_model(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=2.3, expansion_rate=3.0,
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

# Load the trained weights (if applicable)
# model = torch.load("quantized_model.pt", map_location=torch.device("cpu"))
model = get_model(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8, expansion_rate=2.1)
model_state_dict = torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt', map_location=torch.device("cpu"))
model.load_state_dict(model_state_dict)

# # **Disable quantization before exporting**
# model.quant = torch.nn.Identity()  # Remove QuantStub
# model.dequant = torch.nn.Identity()  # Remove DeQuantStub

# Set to evaluation mode
model.eval()

input_tensor = torch.rand((1, 1, 256, 65), dtype=torch.float32)

torch.onnx.export(
    model,                  # model to export
    input_tensor,        # inputs of the model,
    "baseline.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    output_names=["output"],# Rename outputs for the ONNX model
    opset_version=13,  # Ensure compatibility
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

onnx_filename = "baseline.onnx"
if not os.path.exists(onnx_filename):
    print(f"Error: {onnx_filename} was not created. Check ONNX export!")
else:
    print(f"ONNX file successfully created: {onnx_filename}")
# print(f"Model exported successfully to {onnx_filename}")

# 4. Verify the ONNX Model
onnx_model = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# 5. Run Inference Using ONNX Runtime
ort_session = ort.InferenceSession(onnx_filename)

# Convert input tensor to NumPy format
input_data = {"input": input_tensor.numpy()}

# Perform inference
outputs = ort_session.run(None, input_data)

# Display output
print("ONNX Model Output:", outputs)