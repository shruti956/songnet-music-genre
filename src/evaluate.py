import torch, torch.nn.functional as F, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import load_features, GENRES
from model import SongNet

X, y = load_features('test')
model = SongNet(); model.load_state_dict(torch.load('models/best.pt', map_location='cpu')); model.eval()
X = torch.from_numpy(X)
preds = []
with torch.no_grad():
    for i in range(0, len(X), 32):
        preds.append(F.softmax(model(X[i:i+32].float()), -1).mean(1).argmax(1))
preds = torch.cat(preds).numpy()
print('Test accuracy:', (preds == y).mean())
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(confusion_matrix(y, preds), display_labels=GENRES).plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout(); plt.savefig('report/confusion_matrix.png', dpi=150)
