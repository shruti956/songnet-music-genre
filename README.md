# SongNet: Real-time Music Genre Classification

UE24CS352A - Machine Learning Mini-Project

Team members: Shruti Sridhar (PES2UG24CS498), Srikanth V Reddy (PES2UG24CS519)

## Overview

This project classifies music into 8 genres from raw audio using a Convolutional
Recurrent Neural Network (C-RNN), following the approach of *SongNet*
(Stanford CS229, 2018). The model takes a mel-spectrogram as its only input,
extracts features with 1-D convolutions along the time axis, passes them through a
GRU, and outputs a genre probability distribution at every timestep. This enables
real-time classification while a song plays. The song-level prediction is the mean
of the per-timestep predictions.

We compare the C-RNN against four classical baselines (k-NN, Logistic Regression,
MLP, Linear SVM) trained on the precomputed FMA features.

## Dataset

[Free Music Archive (FMA)](https://github.com/mdeff/fma), `fma_small` subset:
8,000 tracks (30-second clips), 8 balanced genres (Electronic, Experimental, Folk,
Hip-Hop, Instrumental, International, Pop, Rock), split 70% / 20% / 10%
(train / val / test, stratified, seed 42). A few corrupted mp3s are skipped during
feature extraction, leaving 5,599 / 1,599 / 799 clips.

The dataset is NOT stored in this repo. `splits.csv` lists the exact split used.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── splits.csv           # train/val/test assignment for every track
├── models/best.pt       # trained C-RNN weights
├── src/
│   ├── data.py          # load labels, create splits
│   ├── features.py      # mel-spectrogram extraction
│   ├── baselines.py     # k-NN, LR, MLP, SVM
│   ├── model.py         # C-RNN architecture
│   ├── train.py         # training loop (checkpoint + resume)
│   └── evaluate.py      # test accuracy, confusion matrix
├── app/demo.py          # real-time classification demo (Streamlit)
├── notebooks/colab_runner.ipynb   # Colab notebook used for extraction and training
└── report/              # write-up, slides, confusion matrix
```

## Run the demo (quickest)

No dataset needed, only `models/best.pt` from this repo.

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/demo.py
```

Upload an mp3 or wav. The app shows the predicted genre, the probability of each
genre, and how the prediction evolves over time. The "Simulate real-time
classification" button replays the song second by second using only the audio
heard so far.

## Reproduce the full pipeline

1. Download the data into a `data/` folder:
```bash
   mkdir data && cd data
   curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
   curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
   unzip fma_metadata.zip && unzip fma_small.zip && cd ..
```
   (If the links have changed, use the official FMA repository.)
2. Run the scripts from the repo root:
```bash
   python src/features.py      # extract mel-spectrograms into features/
   python src/baselines.py     # train and evaluate the baselines
   python src/train.py         # train the C-RNN (GPU recommended)
   python src/evaluate.py      # test accuracy and confusion matrix
```

Feature extraction and training were run on Google Colab (T4 GPU). The cells we
used are in `notebooks/colab_runner.ipynb`.

## Results

| Model | Test Accuracy |
|---|---|
| Random guessing | 12.5% |
| k-NN | 44.5% |
| Logistic Regression | 54.5% |
| MLP | 58.0% |
| Linear SVM | 53.5% |
| **C-RNN (SongNet)** | **60.2%** |

The C-RNN slightly outperforms the best baseline (MLP, 58.0%) while using only the
raw mel-spectrogram as input. The baselines use the 518 precomputed FMA features.

<img width="1200" height="1200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/bcf592b0-179c-439e-83f3-4c7a817ba607" />

![Confusion matrix](report/confusion_matrix.png)

International, Rock, Hip-Hop, Electronic and Folk are classified best. Pop and
Instrumental are hardest: Pop is confused with Rock, International and Folk, and
Instrumental with Experimental and Folk.

## References

- Zhang, Zhang, Chen. *SongNet: Real-time Music Classification.* Stanford CS229, 2018.
- Defferrard, Benzi, Vandergheynst, Bresson. *FMA: A Dataset for Music Analysis.* 2016.
- Choi, Fazekas, Sandler, Cho. *Convolutional Recurrent Neural Networks for Music Classification.* 2016.
- librosa: https://librosa.org
