import os, glob, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get('DATA_DIR', 'data')
FEATURES_DIR = os.environ.get('FEATURES_DIR', 'features')
GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

def audio_path(t):
    s = f'{t:06d}'
    return f'{DATA_DIR}/fma_small/{s[:3]}/{s}.mp3'

def make_splits(out='splits.csv'):
    tracks = pd.read_csv(f'{DATA_DIR}/fma_metadata/tracks.csv', index_col=0, header=[0, 1])
    small = tracks[tracks['set', 'subset'] == 'small']
    df = pd.DataFrame({'genre': small['track', 'genre_top']})
    df.index.name = 'track_id'
    train, temp = train_test_split(df, test_size=0.30, stratify=df['genre'], random_state=42)
    val, test = train_test_split(temp, test_size=1/3, stratify=temp['genre'], random_state=42)
    df['split'] = 'train'
    df.loc[val.index, 'split'] = 'val'
    df.loc[test.index, 'split'] = 'test'
    df.to_csv(out)
    return df

def load_features(split):
    X, y = [], []
    for f in sorted(glob.glob(f'{FEATURES_DIR}/{split}_*.npz')):
        z = np.load(f); X.append(z['X']); y.append(z['y'])
    return np.concatenate(X), np.concatenate(y)

if __name__ == '__main__':
    print(make_splits()['split'].value_counts())
