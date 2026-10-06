import torch.nn as nn

class SongNet(nn.Module):
    def __init__(s, n_mels=128, n_classes=8):
        super().__init__()
        s.conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.3))
        s.gru = nn.GRU(128, 128, batch_first=True)
        s.fc = nn.Linear(128, n_classes)

    def forward(s, x):                       # x: (batch, time, mel)
        h = s.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = s.gru(h)
        return s.fc(h)
