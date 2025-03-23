import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import openl3
import pandas as pd

def generate_label_mapping_from_csv(csv_path, sep="\t"):
    """
    Reads a meta CSV file and creates a mapping from filename to label index.
    The CSV is expected to have columns: filename, scene_label, identifier, source_label.
    
    Returns:
        file_to_label: dict mapping each filename (string) to its label index (int)
        label_to_idx: dict mapping each scene_label (string) to an index (int)
    """
    df = pd.read_csv(csv_path, sep=sep)
    # Get unique scene labels and sort them to assign consistent indices
    unique_labels = sorted(df['scene_label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    file_to_label = {row['filename']: label_to_idx[row['scene_label']] for _, row in df.iterrows()}
    return file_to_label, label_to_idx

class OpenL3PrecomputedDataset(Dataset):
    def __init__(self, audio_dir, cache_dir, meta_csv, target_shape=(256, 65),
                 embedding_size=512, sr=44100, transform=None, csv_sep="\t"):
        """
        Args:
            audio_dir (str): Directory containing raw audio files (e.g., .wav).
            cache_dir (str): Directory where precomputed feature files (.npy) will be stored.
            meta_csv (str): Path to the meta CSV file with columns: filename, scene_label, identifier, source_label.
            target_shape (tuple): Desired shape (height, width) for each feature (e.g., (256, 65)).
            embedding_size (int): The embedding dimension to use in OpenL3 extraction.
            sr (int): Sample rate for loading audio.
            transform (callable, optional): Optional transform to apply to the feature tensor.
            csv_sep (str): Separator used in the CSV file (default is comma).
        """
        self.audio_dir = audio_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.transform = transform
        self.target_shape = target_shape
        self.embedding_size = embedding_size
        self.sr = sr
        
        self._openl3_model = None # Initialize the OpenL3 model attribute

        # Generate label mapping from the CSV file.
        # self.file_to_label, self.label_to_idx = generate_label_mapping_from_csv(meta_csv, sep=csv_sep)
        file_to_label, label_to_idx = generate_label_mapping_from_csv(meta_csv, sep=csv_sep)
        # List of audio files (filenames) from the CSV.
        # self.audio_files = list(self.file_to_label.keys())
        self.file_to_label = {
            (fname[len("audio/"):] if fname.startswith("audio/") else fname): label 
            for fname, label in file_to_label.items()
        }
        self.label_to_idx = label_to_idx
        # List of audio files (filenames) from the updated mapping.
        self.audio_files = list(self.file_to_label.keys())

    def __len__(self):
        return len(self.audio_files)
    
    def _compute_and_cache_feature(self, audio_file):
        audio_path = os.path.join(self.audio_dir, audio_file)
        # Load raw audio (mono) using librosa
        audio, sr = librosa.load(audio_path, sr=self.sr)
        # Lazy-load the OpenL3 model if not already loaded
        if self._openl3_model is None:
            # Load a Keras model. You can adjust 'input_repr' if needed.
            self._openl3_model = openl3.models.load_audio_embedding_model(
                input_repr="mel256", 
                content_type="music", 
                embedding_size=self.embedding_size
            )
        # Extract OpenL3 embeddings; embeddings shape: (n_frames, embedding_size)
        embeddings, _ = openl3.get_audio_embedding(audio, sr,
                                                     model=self._openl3_model,
                                                     content_type='music',
                                                     embedding_size=self.embedding_size)
        # Adjust embeddings to match target_shape (crop or pad as needed)
        n_frames, emb_dim = embeddings.shape
        target_frames, target_emb = self.target_shape
        
        # Adjust time dimension (n_frames)
        if n_frames > target_frames:
            embeddings = embeddings[:target_frames, :]
        elif n_frames < target_frames:
            pad_frames = target_frames - n_frames
            embeddings = np.pad(embeddings, ((0, pad_frames), (0, 0)), mode='constant')
        
        # Adjust embedding dimension (columns)
        if emb_dim > target_emb:
            embeddings = embeddings[:, :target_emb]
        elif emb_dim < target_emb:
            pad_emb = target_emb - emb_dim
            embeddings = np.pad(embeddings, ((0, 0), (0, pad_emb)), mode='constant')
        
        return embeddings
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        if audio_file.startswith("audio/"):
            audio_file = audio_file[len("audio/"):]
        audio_path = os.path.join(self.audio_dir, audio_file)
        # Construct cache filename (change extension from .wav to .npy)
        cache_file = os.path.join(self.cache_dir, audio_file.replace('.wav', '.npy'))
        
        if os.path.exists(cache_file):
            feature = np.load(cache_file)
        else:
            feature = self._compute_and_cache_feature(audio_file)
            np.save(cache_file, feature)
        
        # Convert to torch tensor and add a channel dimension.
        # Final shape will be (1, target_shape[0], target_shape[1]) i.e., (1, 256, 65)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        
        # Get label for this audio file.
        label = self.file_to_label[audio_file]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

if __name__ == "__main__":
    # Define paths (modify these paths to match your data locations)
    audio_dir = "C:/Users/fenel/Documents/6337421/audio"       # Directory with .wav files
    cache_dir = "C:/Users/fenel\Documents/6337421/cache"         # Directory to save/load precomputed features
    meta_csv = "C:/Users/fenel/Documents/6337421/meta.csv"                # Path to the meta CSV file

    # Create dataset instance
    dataset = OpenL3PrecomputedDataset(audio_dir=audio_dir,
                                       cache_dir=cache_dir,
                                       meta_csv=meta_csv,
                                       target_shape=(256, 65),
                                       embedding_size=512,
                                       sr=44100,
                                       csv_sep="\t")  # change csv_sep if needed

    # Get a sample to verify shapes and labels
    
    sample_feature, sample_label = dataset[61745]
    print("Sample feature shape:", sample_feature.shape)  # Should print: torch.Size([1, 256, 65])
    print("Sample label:", sample_label)
    for idx in range(len(dataset)):
        feature, label = dataset[idx]