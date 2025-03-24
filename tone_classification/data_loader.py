import librosa
import numpy as np

def extract_features(file_path, sr=16000, max_len=5):
    """
    Extract audio features from a file: MFCC, pitch, energy and combine them all as a feature vector
    """
    y, _ = librosa.load(file_path, sr=sr, duration=max_len)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)  # shape (13,)

    # Pitch (use librosa’s piptrack or yin)
    pitch = librosa.yin(y, fmin=50, fmax=500)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    # Energy (intensity)
    energy = np.mean(librosa.feature.rms(y=y))

    # Combine into one feature vector
    features = np.hstack([mfcc, pitch_mean, pitch_std, energy])
    
    return features


def extract_sequence_features(file_path, sr=16000, max_len=5, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr, duration=max_len)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape: (n_mfcc, t)
    mfcc = mfcc.T  # shape: (t, n_mfcc)

    max_frames = 200
    if mfcc.shape[0] < max_frames:
        pad_width = max_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_frames, :]

    return mfcc  # shape: (max_frames, n_mfcc)
