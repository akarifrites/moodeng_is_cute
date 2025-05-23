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
        # print("Input tensor shape in model.forward:", x.shape)
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
    
        # def quantized_forward(self, x):
        # """
        # :param x: batch of spectrograms
        # :return: final model predictions
        # """
        # # quantized forward needs to be done on cpu
        # orig_device = x.device
        # x = x.cpu()
        # self.model_int8.cpu()
        # y = self.model_int8(x)
        # return y.to(orig_device)
    

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

def evaluate(model, dataloader, MelSpecGenerator):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            labels = batch[2]
            raw_waveform = batch[0]
            mel_spec = MelSpecGenerator(raw_waveform)
            print("evaluate mel_spec, tensor shape:", mel_spec.shape)
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            print(f"Batch: {idx+1}/{len(dataloader)} -- Batch Accuracy: {(predicted == labels).sum().item()}/{labels.size(0)}", end='\r')
    accuracy = 100 * correct / total
    return accuracy

def train_qat(model, train_loader, num_epochs=5, learning_rate=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Adjust loss if needed
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # inputs, _, labels = batch  # adjust if your dataset returns a different tuple
            inputs = batch[0]
            labels = batch[-1]

            # # Debug prints to inspect inputs
            # if batch_idx == 0 and epoch == 0:  # Only print once per run, so it doesn't flood your logs
            #     print("==== Debug Info (Before MelSpec) ====")
            #     print(f"Type of inputs: {type(inputs)}")
            #     print(f"Shape of inputs: {inputs.shape}")
            #     print(f"Data type: {inputs.dtype}")
            #     print(f"Min: {inputs.min().item():.4f}, Max: {inputs.max().item():.4f}")
            #     # If it's not too large, you could also print a small portion:
                # print("Sample values:", inputs[0, :10])  # for 1D data, or adapt to your shape

            inputs = MelSpec()(inputs)
            # print("After MelSpec, tensor shape:", inputs.shape)

            # # Another debug print to see the transformed shape
            # if batch_idx == 0 and epoch == 0:
            #     print("==== Debug Info (After MelSpec) ====")
            #     print(f"Shape after MelSpec: {inputs.shape}")
            #     print(f"Min: {inputs.min().item():.4f}, Max: {inputs.max().item():.4f}")
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model


def main():
    # Initialize wandb
    wandb.init(project="quantization_experiment", config={
        "model": "Network_1",
        "quantization": "QAT",
        "fused": True,
        "batch_size": 256,
        "num_epochs": 5,
        "learning_rate": 1e-3
    })

    MelSpecGenerator = MelSpec()

    # Load your pre-trained model
    model_fp32 = get_model_1(in_channels=1, quantize=True, mel_forward=False)
    model_fp32.load_state_dict(torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt'))
    model_fp32.eval()

    model_unquantized = get_model_1(in_channels=1, quantize=False, mel_forward=False)
    model_unquantized.load_state_dict(torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt'))
    model_unquantized.eval()

    # Inspect the original model
    print("Original model structure:")
    print(model_fp32)

    # Fuse layers for quantization
    model_fp32 = fuse_model(model_fp32)

    model_fp32.train()  # Set model to training mode

    # Set the quantization configuration (using fbgemm backend)
    model_fp32.qconfig = quantization.get_default_qat_qconfig('fbgemm')

    # Prepare model for QAT, inpare the model for static quantization
    model_fp32_prepared = quantization.prepare_qat(model_fp32, inplace=True)

    train_dataset = get_training_set(split=5)  # This accesses the original training dataset
    num_samples = int(0.05 * len(train_dataset))  # 5% of the dataset
    subset_indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=0)

    # Run QAT training (fine-tuning) for a few epochs
    print("Starting QAT training...")
    model_fp32_prepared = train_qat(model_fp32_prepared, train_loader,
                                    num_epochs=wandb.config.num_epochs,
                                    learning_rate=wandb.config.learning_rate)

    # Convert the model to a quantized version
    model_int8 = quantization.convert(model_fp32_prepared)
    print("Quantized model:")
    print(model_int8)

    # Create a test dataloader
    test_loader = DataLoader(get_test_set(), batch_size=256, shuffle=True)

    # import matplotlib.pyplot as plt

    # # Suppose raw_waveform is a single example audio tensor with shape [1, 44100]
    # raw_waveform = torch.randn((1, 44100))

    # # Compute mel spectrogram using the training pipeline
    # mel_spec_train = MelSpec()(raw_waveform)  # Shape: [1, 256, 65] if unsqueeze isn't applied
    # # Compute mel spectrogram using the evaluation pipeline (if it's defined differently)
    # mel_spec_eval = MelSpec()(raw_waveform)  # Replace with MelSpecGenerator(raw_waveform) if different

    # # Remove the batch and channel dimensions for visualization
    # # This assumes the spectrogram shape is [batch, channel, mel_bins, time_frames]
    # spec_train = mel_spec_train.squeeze().detach().numpy()  # Shape: [256, 65]
    # spec_eval = mel_spec_eval.squeeze().detach().numpy()     # Shape: [256, 65]

    # # Plot the two spectrograms side-by-side
    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(spec_train, aspect='auto', origin='lower')
    # plt.title('Training Mel Spectrogram')
    # plt.xlabel('Time Frames')
    # plt.ylabel('Mel Bins')
    # plt.colorbar()

    # plt.subplot(1, 2, 2)
    # plt.imshow(spec_eval, aspect='auto', origin='lower')
    # plt.title('Evaluation Mel Spectrogram')
    # plt.xlabel('Time Frames')
    # plt.ylabel('Mel Bins')
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()

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

    # Save the quantized model
    torch.save(model_int8.state_dict(), 'quant_model_state_dict.pt')

    # To run inference with the quantized model
    model_int8.eval()
    raw_input = torch.randn((1, 44100))  # Example raw audio (1 channel, 44100 samples)
    input_tensor = MelSpec()(raw_input).unsqueeze(1)  # Convert to mel spectrogram with channel dimension
    # input_tensor = torch.randn((1, 1, 256, 65))  # Example input shape
    output = model_int8(input_tensor)
    print(output)

if __name__ == "__main__":
    main()