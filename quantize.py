import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.quantization
from torchvision.ops.misc import Conv2dNormActivation
import torchaudio.transforms
import numpy as np
import wandb
import librosa
import os

from models.baseline import get_model, initialize_weights, Block
from models.helpers.utils import make_divisible
from dataset.dcase24 import get_training_set, get_test_set

import warnings

warnings.filterwarnings('ignore')

# model_path = 'model_state_dict.pt'
# quantized_model_save_path = 'quant_model_state_dict.pt'
# batch_size = 256
# num_classes = 10

# # module for resampling waveforms on the fly
#         resample = torchaudio.transforms.Resample(
#             orig_freq=44100, 
#             new_freq=32000
#         )

def patched_forward(self, x):
    # Compute the main branch output
    out = self.block(x)
    # If the block has a shortcut, use quantized addition
    if hasattr(self, 'shortcut'):
        try:
            residual = self.shortcut(x)
            out = self.ff.add(out, residual)
        except Exception as e:
            # If for any reason shortcut processing fails, simply skip addition
            print(f"Block {self} has no valid shortcut, skipping addition. Error: {e}")
    # Apply post-activation if available
    if hasattr(self, 'after_block_activation'):
        out = self.after_block_activation(out)
    return out

# Monkey-patch the Block forward method
Block.forward = patched_forward

# define mel spectrogram
mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                           n_fft=4096, 
                                           win_length=3072,
                                           hop_length=500, 
                                           n_mels=256)


class MelSpec(nn.Module):
    def __init__(self):
        super(MelSpec, self).__init__()
        resample = torchaudio.transforms.Resample(
            orig_freq=44100, 
            new_freq=32000
        )

        # define mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                                    n_fft=4096, 
                                                    win_length=3072,
                                                    hop_length=500, 
                                                    n_mels=256)
        
        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

    def forward(self,x):
        x = self.mel(x)
        x = (x + 1e-5).log()
        return x


class Network_1(nn.Module):
    def __init__(self, config, quantize=False, mel_forward=False):
        super(Network_1, self).__init__()
        self.quantize = quantize
        if self.quantize:
            # Quantization stubs
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            
        self.mel_forward = mel_forward
        if self.mel_forward:
            # Resample the audio clip from original frequency to a new determined sampling frequency
            resample = torchaudio.transforms.Resample(
                orig_freq=44100, 
                new_freq=32000
            )

            # Generate the Mel Spectrogram with specified parameters
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                                        n_fft=4096, 
                                                        win_length=3072,
                                                        hop_length=500, 
                                                        n_mels=256)
            
            # Sequentially perform resampling then the Mel Spec generation
            self.mel = torch.nn.Sequential(
                resample,
                mel
            )
        
        # Convolutional layers
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
        if self.mel_forward:
            x = self.mel(x)
            x = (x + 1e-5).log()
        if self.quantize:
            x = self.quant(x)  # Quantize the input
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        if self.quantize:
            x = self.dequant(x)
        logits = x.squeeze(2).squeeze(2)
        return logits
    

def get_model_1(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8, expansion_rate=2.1,
            n_blocks=(3, 2, 1), strides=None, quantize=False, mel_forward=False):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @param quantize: Determines the inclusion of Quant and Dequant stubs
    @param mel_forward: Determines the inclusion of the computation of Mel Specs within the Model
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

    m = Network_1(model_config, quantize=quantize, mel_forward=mel_forward)
    return m

# Define a helper function to fuse layers
def fuse_model(model):
    # Inspect the in_c module to see its children
    print("Before fusing, in_c module:")
    print(model.in_c)

    # Fuse layers inside each Conv2dNormActivation in the in_c Sequential
    for idx in range(len(model.in_c)):
        module = model.in_c[idx]
        keys = list(module._modules.keys())
        print(f"Fusing in_c module index {idx} with keys: {keys}")
        # Only fuse if the module has 2 or 3 submodules (typical for Conv-BN or Conv-BN-ReLU)
        if len(keys) in [2, 3]:
            try:
                torch.quantization.fuse_modules(module, keys, inplace=True)
            except Exception as e:
                print(f"Could not fuse in_c module index {idx}: {e}")

    for stage in model.stages:
        for block_name, block in stage.named_children():
            print(f"Before fusing, block {block_name}:")
            print(block)
            # For each block, focus on its 'block' Sequential module that holds Conv2dNormActivation modules.
            if hasattr(block, 'block'):
                for subname, submodule in block.block.named_children():
                    if isinstance(submodule, Conv2dNormActivation):
                        keys = list(submodule._modules.keys())
                        print(f"Fusing block {block_name} submodule {subname} with keys: {keys}")
                        if len(keys) in [2, 3]:
                            try:
                                torch.quantization.fuse_modules(submodule, keys, inplace=True)
                            except Exception as e:
                                print(f"Could not fuse block {block_name} submodule {subname}: {e}")
            else:
                print(f"Block {block_name} has no 'block' attribute to fuse.")

    print("After fusing, model structure:")
    print(model)
    return model

def evaluate(model, dataloader, mel_spec_transform):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            labels = batch[2]
            raw_waveform = batch[0]
            mel_spec = mel_spec_transform(raw_waveform)
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            print(f"Batch: {idx+1}/{len(dataloader)} -- Batch Accuracy: {(predicted == labels).sum().item()}/{labels.size(0)}", end='\r')
    accuracy = 100 * correct / total
    return accuracy

# Calibration step using a representative dataset
def calibrate(model, data_loader, MelSpecGenerator):
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            raw_waveform = inputs[0]
            mel_spec = MelSpecGenerator(raw_waveform)
            model(mel_spec)
            print(f"Calibrating: {idx+1}/{len(data_loader)}", end='\r')
            # For demonstration, calibrate on one batch only
            if idx == 0:
                break

def main():
    # Initialize wandb
    wandb.init(project="quantization_experiment", config={
        "model": "Network_1",
        "quantization": "static",
        "fused": True,
        "batch_size": 256
    })

    MelSpecGenerator = MelSpec()

    # Load your pre-trained model
    model_fp32 = get_model_1(quantize=True, mel_forward=False)
    model_fp32.load_state_dict(torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt'))
    model_fp32.eval()

    model_unquantized = get_model_1(quantize=False, mel_forward=False)
    model_unquantized.load_state_dict(torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt'))
    model_unquantized.eval()

    # Inspect the original model
    print("Original model structure:")
    print(model_fp32)

    # Fuse layers for quantization
    model_fp32 = fuse_model(model_fp32)

    # Set the quantization configuration (using fbgemm backend)
    model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')

    # Prepare the model for static quantization
    model_fp32_prepared = quantization.prepare(model_fp32, inplace=True)

    train_dataset = get_training_set()  # This accesses the original training dataset
    num_calibration_samples = int(0.05 * len(train_dataset))
    calibration_indices = np.random.choice(len(train_dataset), num_calibration_samples, replace=False)

    # Create a subset of the training data for calibration
    calibration_data = Subset(train_dataset, calibration_indices)            

    # Create a DataLoader for calibration (replace with your dataset)
    calibration_data = DataLoader(dataset=calibration_data,
                                    batch_size=256,  # Use the same batch size as train_dl
                                    num_workers=0,
                                    shuffle=False  # No need to shuffle for calibration
                                    )
    calibrate(model_fp32_prepared, calibration_data, MelSpecGenerator)

    # Convert the model to a quantized version
    model_int8 = quantization.convert(model_fp32_prepared)
    print("Quantized model:")
    print(model_int8)

    # Create a test dataloader
    test_loader = DataLoader(get_test_set(), batch_size=256, shuffle=True)

    # Evaluate both quantized and unquantized models
    quantized_accuracy = evaluate(model_int8, test_loader, MelSpecGenerator)
    unquantized_accuracy = evaluate(model_unquantized, test_loader, MelSpecGenerator)

    print(f"\nQuantized model accuracy: {quantized_accuracy:.2f}%")
    print(f"Unquantized model accuracy: {unquantized_accuracy:.2f}%")

    # Log metrics to wandb
    wandb.log({
        "quantized_accuracy": quantized_accuracy,
        "unquantized_accuracy": unquantized_accuracy
    })


# def test_single_batch(model, dataloader, mel_spec_transform=None):
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad(): 
#         # Get a single batch of data
#         for batch in dataloader:
#             labels = batch[2]
#             labels = labels.type(torch.LongTensor)
#             inputs = batch[0]   
        
#             # Print shapes to verify correct input/output sizes
#             print("Input shape:", inputs.shape)
#             print("Label shape:", labels.shape)
        
#             # Convert to Mel spectrogram if needed
#             if mel_spec_transform:
#                 inputs = mel_spec_transform(inputs)
#                 print("Transformed input shape (Mel spectrogram):", inputs.shape)
        
#             # Forward pass through the model
#             outputs = model(inputs)
#             print("Output shape:", outputs.shape)
#             print("Outputs:", outputs)
#             print("Labels:", labels)
        
#             # Calculate predictions
#             _, predicted = torch.max(outputs.data, 1)
#             print("Predicted labels:", predicted)
#             print("Actual labels:", labels)
        
#             # Calculate accuracy for this single batch
#             correct = (predicted == labels).sum().item()
#             accuracy = 100 * correct / labels.size(0)
#             print(f"Accuracy for this batch: {accuracy:.2f}%")
        
#             break  # Only process one batch for testing

# Run this function with your unquantized model and dataloader
# Replace `unquantized_model` and `test_loader` with your actual model and dataloader
# test_single_batch(model_unquantized, test_loader, mel_spec_transform=mel_spec_transform)

    # Save the quantized model
    torch.save(model_int8.state_dict(), 'quant_model_state_dict.pt')

    # To run inference with the quantized model
    model_int8.eval()
    input_tensor = torch.randn((1, 1, 256, 65))  # Example input shape
    output = model_int8(input_tensor)
    print(output)

if __name__ == "__main__":
    main()