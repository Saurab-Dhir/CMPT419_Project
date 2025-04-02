import os
import librosa
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class Wav2Vec2FeatureExtractor:
    """
    A wrapper around a Hugging Face wav2vec2 model,
    so we can easily extract a fixed embedding vector for each audio clip.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60", device=device):
        # Processor normalizes and tokenizes the raw audio
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        # The actual model that outputs hidden states
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval().to(device)
        self.sr = 16000
        self.device = device

    def extract_embedding(self, audio_path):
        """
        1) Load raw audio
        2) pitch/time augment
        3) Tokenize with Wav2Vec2Processor
        4) Forward pass through Wav2Vec2Model
        5) Return an embedding vector 
        """
        # 1) Load raw audio at 16k
        waveform, _ = librosa.load(audio_path, sr=self.sr)

        # 2) Augmentation
        if random.random() < 0.5:
            waveform = librosa.effects.time_stretch(waveform, rate=random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            waveform = librosa.effects.pitch_shift(waveform, sr=self.sr, n_steps=random.uniform(-2, 2))

        # 3) Tokenize, requesting attention_mask if supported
        inputs = self.processor(
            waveform,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True  # ask for attention_mask
        )
        input_values = inputs["input_values"].to(self.device)

        # Fallback if the processor doesn't return mask
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 4) Forward pass
        with torch.no_grad():
            if attention_mask is not None:
                outputs = self.model(input_values, attention_mask=attention_mask)
            else:
                outputs = self.model(input_values)
            hidden_states = outputs.last_hidden_state  # shape [1, T, hidden_dim]

        # 5) Mean-pool => shape [hidden_dim]
        embedding = hidden_states.mean(dim=1).squeeze()
        return embedding.cpu().numpy()


class Wav2Vec2Dataset(Dataset):
    """precomputed wav2vec2 embeddings and numeric labels """
    def __init__(self, embedding_list, label_list):
        self.embedding_list = embedding_list  # list of 1D arrays
        self.label_list = label_list

    def __len__(self):
        return len(self.embedding_list)

    def __getitem__(self, idx):
        x = self.embedding_list[idx]
        y = self.label_list[idx]
        return x, y

def collate_fn_wav2vec2(batch):
    """
    For a list of embedding - label pair, stack embeddings into [batch_size, embedding_dim]
    Return (tensor_embeddings, tensor_labels)
    """
    xs, ys = zip(*batch)
    xs = torch.tensor(np.stack(xs), dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys
