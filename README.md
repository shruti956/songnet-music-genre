# SongNet: Real-time Music Genre Classification

UE24CS352A - Machine Learning Mini-Project

Team members: `<Name 1 (SRN)>`, `<Name 2 (SRN)>`

## Overview

This project classifies music into genres from raw audio using a Convolutional
Recurrent Neural Network (C-RNN), following the approach of *SongNet*
(Stanford CS229, 2018). The model takes a mel-spectrogram as its only input,
extracts features with 1-D convolutions along the time axis, and outputs a genre
probability distribution at every timestep, which enables real-time
classification while a song plays. The song-level prediction is the mean of the
per-timestep predictions.

We compare the C-RNN against four classical baselines (k-NN, Logistic
Regression, MLP, Linear SVM) trained on the precomputed FMA features.

## Dataset

[Free Music Archive (FMA)](https://github.com/mdeff/fma), `fma_small` subset:

- 8,000 tracks, 30-second clips
- 8 balanced genres (1,000 clips each): Electronic, Experimental, Folk,
  Hip-Hop, Instrumental, International, Pop, Rock
- Split: 70% train / 20% validation / 10% test

Downloads needed:

- `fma_small.zip` (audio)
- `fma_metadata.zip` (labels in `tracks.csv`, precomputed `features.csv`)

> The dataset is NOT stored in this repo. See the setup section below.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py          # load labels, create splits
│   ├── features.py      # mel-spectrogram extraction
│   ├── baselines.py     # k-NN, LR, MLP, SVM
│   ├── model.py         # C-RNN architecture
│   ├── train.py         # training loop
│   └── evaluate.py      # accuracy, confusion matrix
├── app/
│   └── demo.py          # real-time classification demo
├── notebooks/
│   └── colab_runner.ipynb
└── report/              # write-up and slides
```

## Setup

### Option A: Google Colab (recommended)

1. Open `notebooks/colab_runner.ipynb` in Colab and select a GPU runtime
   (Runtime > Change runtime type > T4 GPU).
2. The notebook clones this repo, installs dependencies, downloads the dataset
   to `/content`, and saves cached features and checkpoints to Google Drive.

### Option B: Local machine

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset into a `data/` folder:

```bash
mkdir data && cd data
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_metadata.zip && unzip fma_small.zip
```

(If these links have changed, use the links in the official FMA repository.)

## Usage

Scripts are added as the project progresses. Planned commands:

```bash
python src/features.py      # extract and cache mel-spectrograms
python src/baselines.py     # train and evaluate baselines
python src/train.py         # train the C-RNN
python src/evaluate.py      # results table and confusion matrix
streamlit run app/demo.py   # launch the real-time demo
```

## Results

To be filled in after training.

| Model | Test Accuracy |
|---|---|
| Random guessing | 12.5% |
| k-NN | TBD |
| Logistic Regression | TBD |
| MLP | TBD |
| Linear SVM | TBD |
| C-RNN (SongNet) | TBD |

## References

- Zhang, Zhang, Chen. *SongNet: Real-time Music Classification.* Stanford CS229, 2018.
- Defferrard, Benzi, Vandergheynst, Bresson. *FMA: A Dataset for Music Analysis.* 2016.
- Choi, Fazekas, Sandler, Cho. *Convolutional Recurrent Neural Networks for Music Classification.* 2016.
- librosa: https://librosa.org
