import librosa
import numpy as np

import librosa.effects
import random

def augment_audio(y, sr):
    if random.random() < 0.5:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
    if random.random() < 0.5:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-1, 1))
    return y



# def extract_features(file_path, sr=16000, max_len=5):
#     """
#     Extract audio features from a file: MFCC, pitch, energy and combine them all as a feature vector
#     """
#     y, _ = librosa.load(file_path, sr=sr, duration=max_len)
#     y = augment_audio(y, sr)


#     # Core features
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
#     spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).T
#     spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).T
#     spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).T
#     spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).T
#     zcr = librosa.feature.zero_crossing_rate(y=y).T

#     # Pitch & energy stats
#     pitch = librosa.yin(y, fmin=50, fmax=500)
#     pitch_mean = np.mean(pitch)
#     pitch_std = np.std(pitch)
#     energy = np.mean(librosa.feature.rms(y=y))

#     time_steps = mfcc.shape[0]

#     # Align time steps (clip or pad others)
#     def pad_or_truncate(arr, t):
#         if arr.shape[0] > t:
#             return arr[:t]
#         elif arr.shape[0] < t:
#             pad = np.zeros((t - arr.shape[0], arr.shape[1]))
#             return np.vstack([arr, pad])
#         return arr

#     # Match all to MFCC time length
#     chroma = pad_or_truncate(chroma, time_steps)
#     spec_centroid = pad_or_truncate(spec_centroid, time_steps)
#     spec_bandwidth = pad_or_truncate(spec_bandwidth, time_steps)
#     spec_contrast = pad_or_truncate(spec_contrast, time_steps)
#     spec_rolloff = pad_or_truncate(spec_rolloff, time_steps)
#     zcr = pad_or_truncate(zcr, time_steps)

#     extra_features = np.tile([pitch_mean, pitch_std, energy], (time_steps, 1))  # (time_steps, 3)

#     features = np.hstack([
#         mfcc,
#         chroma,
#         spec_centroid,
#         spec_bandwidth,
#         spec_contrast,
#         spec_rolloff,
#         zcr,
#         extra_features
#     ])

#     return features  # shape: (time_steps, total_feature_dim)




def extract_features(file_path, sr=16000, max_len=5, n_mels=64):
    y, _ = librosa.load(file_path, sr=sr, duration=max_len)
    y = augment_audio(y, sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T  # shape: (time_steps, n_mels)

    return mel_spec_db
