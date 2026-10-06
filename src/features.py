import os, numpy as np, pandas as pd, librosa
from joblib import Parallel, delayed
from data import audio_path, FEATURES_DIR

SR, N_FFT, HOP, N_MELS, N_FRAMES = 22050, 2048, 512, 128, 1290

def mel(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=30)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        S = librosa.power_to_db(S, ref=np.max, top_db=80)
        S = ((S + 40) / 40).T.astype(np.float16)
        if S.shape[0] < N_FRAMES:
            S = np.pad(S, ((0, N_FRAMES - S.shape[0]), (0, 0)), constant_values=-1)
        return S[:N_FRAMES], True
    except Exception:
        return np.full((N_FRAMES, N_MELS), -1, dtype=np.float16), False

if __name__ == '__main__':
    df = pd.read_csv('splits.csv', index_col=0)
    label = {g: i for i, g in enumerate(sorted(df['genre'].unique()))}
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        part = df[df['split'] == split]; ids = part.index.tolist()
        for i in range(0, len(ids), 500):
            f = f'{FEATURES_DIR}/{split}_{i//500:03d}.npz'
            if os.path.exists(f):
                continue
            chunk = ids[i:i+500]
            res = Parallel(n_jobs=-1)(delayed(mel)(audio_path(t)) for t in chunk)
            ok = np.array([r[1] for r in res])
            np.savez(f, X=np.stack([r[0] for r in res])[ok],
                     y=part.loc[chunk, 'genre'].map(label).values[ok])
            print(f, flush=True)
