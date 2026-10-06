import os, numpy as np, torch, torch.nn.functional as F
from data import load_features
from model import SongNet

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
Xtr, ytr = [torch.from_numpy(a).to(dev) for a in load_features('train')]
Xva, yva = [torch.from_numpy(a).to(dev) for a in load_features('val')]
ytr, yva = ytr.long(), yva.long()

os.makedirs('models', exist_ok=True)
CK, BEST = 'models/ckpt.pt', 'models/best.pt'
EPOCHS, BS, CROP = 30, 64, 256
model = SongNet().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
start, best = 0, 0.0
if os.path.exists(CK):
    c = torch.load(CK, map_location=dev)
    model.load_state_dict(c['model']); opt.load_state_dict(c['opt'])
    start, best = c['epoch'] + 1, c['best']

@torch.no_grad()
def evaluate(X, y):
    model.eval(); correct = 0
    for i in range(0, len(X), 64):
        p = F.softmax(model(X[i:i+64].float()), -1).mean(1)
        correct += (p.argmax(1) == y[i:i+64]).sum().item()
    return correct / len(X)

for ep in range(start, EPOCHS):
    model.train(); total = 0
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(0, len(perm) - BS + 1, BS):
        idx = perm[i:i+BS]
        s = np.random.randint(0, Xtr.shape[1] - CROP)
        out = model(Xtr[idx, s:s+CROP].float())
        loss = F.cross_entropy(out.reshape(-1, 8), ytr[idx].repeat_interleave(out.shape[1]))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    va = evaluate(Xva, yva)
    if va > best:
        best = va; torch.save(model.state_dict(), BEST)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep, 'best': best}, CK)
    print(f'epoch {ep+1}/{EPOCHS} loss {total/(len(perm)//BS):.3f} val acc {va:.3f} best {best:.3f}', flush=True)
