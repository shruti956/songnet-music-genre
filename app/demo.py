import os, sys, math, time, tempfile
import numpy as np, pandas as pd, torch, torch.nn.functional as F, soundfile as sf, streamlit as st
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'src'))
from model import SongNet

GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
SR, N_FFT, HOP, N_MELS = 22050, 2048, 512, 128
STEP_SEC = 4 * HOP / SR          # each model output covers about 0.09 s

# --- mel filterbank (same maths as librosa, no librosa needed) ---
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64); logstep = np.log(6.4) / 27.0
    return np.where(f < 1000, f / (200 / 3), 15 + np.log(np.maximum(f, 1e-10) / 1000) / logstep)

def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64); logstep = np.log(6.4) / 27.0
    return np.where(m < 15, m * (200 / 3), 1000 * np.exp(logstep * (m - 15)))

def _mel_filterbank():
    freqs = np.linspace(0, SR / 2, 1 + N_FFT // 2)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(SR / 2), N_MELS + 2))
    fdiff = np.diff(pts); ramps = pts[:, None] - freqs[None, :]
    W = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        W[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    W *= (2.0 / (pts[2:N_MELS + 2] - pts[:N_MELS]))[:, None]
    return torch.from_numpy(W.astype(np.float32))

MEL_W = _mel_filterbank()

def mel_spectrogram(y):
    spec = torch.stft(torch.from_numpy(y), N_FFT, HOP, window=torch.hann_window(N_FFT),
                      center=True, pad_mode='constant', return_complex=True)
    S = MEL_W @ (spec.abs() ** 2)                                  # (mel, time)
    S_db = 10 * torch.log10(torch.clamp(S, min=1e-10))
    S_db = torch.clamp(S_db - S_db.max(), min=-80.0)               # same as ref=max, top_db=80
    return ((S_db + 40) / 40).T                                    # (time, mel)

def load_audio(path):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return y[:SR * 30]

@st.cache_resource
def load_model():
    m = SongNet()
    m.load_state_dict(torch.load(os.path.join(HERE, '..', 'models', 'best.pt'), map_location='cpu'))
    return m.eval()

@st.cache_data
def analyze(data: bytes, ext: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data); path = tmp.name
    y = load_audio(path); os.remove(path)
    with torch.no_grad():
        return F.softmax(load_model()(mel_spectrogram(y)[None]), -1)[0].numpy()   # (time, 8)

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
