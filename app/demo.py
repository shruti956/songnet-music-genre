import os, sys, time, tempfile
import numpy as np, pandas as pd, torch, torch.nn.functional as F, librosa, streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'src'))
from model import SongNet

GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
SR, N_FFT, HOP, N_MELS = 22050, 2048, 512, 128
STEP_SEC = 4 * HOP / SR          # each model output covers about 0.09 s

@st.cache_resource
def load_model():
    m = SongNet()
    m.load_state_dict(torch.load(os.path.join(HERE, '..', 'models', 'best.pt'), map_location='cpu'))
    return m.eval()

@st.cache_data
def analyze(data: bytes, ext: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data); path = tmp.name
    y, _ = librosa.load(path, sr=SR, mono=True, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    S = librosa.power_to_db(S, ref=np.max, top_db=80)
    S = ((S + 40) / 40).T.astype(np.float32)
    with torch.no_grad():
        return F.softmax(load_model()(torch.from_numpy(S)[None]), -1)[0].numpy()   # (time, 8)

st.title('SongNet: Real-time Music Genre Classification')
f = st.file_uploader('Upload a song (mp3 or wav)', type=['mp3', 'wav'])

if f:
    data = f.getvalue()
    st.audio(data)
    probs = analyze(data, os.path.splitext(f.name)[1])
    final = probs.mean(0)
    st.subheader(f'Prediction: {GENRES[final.argmax()]} ({final.max():.0%})')
    st.bar_chart(pd.Series(final, index=GENRES))

    st.subheader('Genre probability over time')
    st.line_chart(pd.DataFrame(probs, index=np.arange(len(probs)) * STEP_SEC, columns=GENRES))

    if st.button('Simulate real-time classification'):
        text, chart = st.empty(), st.empty()
        per_sec = int(round(1 / STEP_SEC))
        for k in range(per_sec, len(probs) + per_sec, per_sec):
            k = min(k, len(probs))
            running = probs[:k].mean(0)
            text.markdown(f'**After {k * STEP_SEC:.0f}s:** {GENRES[running.argmax()]}')
            chart.bar_chart(pd.Series(running, index=GENRES))
            time.sleep(0.2)
