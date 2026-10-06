import numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from data import DATA_DIR
warnings.filterwarnings('ignore')

splits = pd.read_csv('splits.csv', index_col=0)
feats = pd.read_csv(f'{DATA_DIR}/fma_metadata/features.csv', index_col=0, header=[0, 1, 2]).loc[splits.index]
Xtr, Xva, Xte = [feats[splits['split'] == s].values for s in ['train', 'val', 'test']]
ytr, yva, yte = [splits.loc[splits['split'] == s, 'genre'].values for s in ['train', 'val', 'test']]
sc = StandardScaler().fit(np.nan_to_num(Xtr))
Xtr, Xva, Xte = [sc.transform(np.nan_to_num(a)) for a in (Xtr, Xva, Xte)]

models = {'k-NN': KNeighborsClassifier(n_neighbors=5),
          'Logistic Regression': LogisticRegression(max_iter=1000),
          'MLP': MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42),
          'Linear SVM': LinearSVC(max_iter=5000)}
rows = []
for name, m in models.items():
    m.fit(Xtr, ytr)
    rows.append((name, accuracy_score(yva, m.predict(Xva)), accuracy_score(yte, m.predict(Xte))))
    print(name, rows[-1][1:])
pd.DataFrame(rows, columns=['Model', 'Val acc', 'Test acc']).round(4).to_csv('baseline_results.csv', index=False)
