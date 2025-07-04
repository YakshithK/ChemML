#%%
import deepchem as dc
import warnings
dc.__version__
warnings.filterwarnings("ignore", message=".*GetValence.*")

#%%
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv', splitter='random')

train_dataset, valid_dataset, test_dataset = datasets

#%%
from rdkit.Chem import Draw
from rdkit import Chem

mol = Chem.MolFromSmiles(train_dataset.ids[0])
Draw.MolToImage(mol)

#%%
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw

#%%
for smiles in train_dataset.ids:
    mol = Chem.MolFromSmiles(smiles)
    display(Draw.MolToImage(mol, size=(200, 200), kekulize=True, wedgeBonds=True))

#%%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

#%%
smiles_list = []
mwts = []
logps = []
hbds = []  # H-bond donors + acceptors
tpsas = []
rbs = []
logS_values = []

for i in range(train_dataset.X.shape[0]):
    SMILES = train_dataset.ids[i]
    mol = Chem.MolFromSmiles(SMILES)

    if mol is None:
        continue  # skip invalid molecules

    smiles_list.append(SMILES)
    mwts.append(Descriptors.MolWt(mol))
    logps.append(Descriptors.MolLogP(mol))
    hbds.append(Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol))
    tpsas.append(rdMolDescriptors.CalcTPSA(mol))
    rbs.append(Lipinski.NumRotatableBonds(mol))
    logS_values.append(train_dataset.y[i][0])

#%%
df = pd.DataFrame({
    'SMILES': smiles_list,
    'MolWt': mwts,
    'LogP': logps,
    'HBD+HBA': hbds,
    'TPSA': tpsas,
    'RotatableBonds': rbs,
    'LogS': logS_values
})

df.to_csv('esol_data.csv', index=False)
df.head()

#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
X = df[['MolWt', 'LogP', 'HBD+HBA', 'TPSA', 'RotatableBonds']]
y = df['LogS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

#%%
lr = LinearRegression()
lr.fit(X_train, y_train)

#%%
y_pred_lr = lr.predict(X_test)

# Evaluate
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"MAE: {mae_lr:.3f}")
print(f"R²: {r2_lr:.3f}")

#%%
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.7, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='gray')
plt.xlabel('Actual LogS')
plt.ylabel('Predicted LogS')
plt.title('Linear Regression: Predicted vs Actual LogS')
plt.grid(True)
plt.show()

#%%
from sklearn.ensemble import RandomForestRegressor

#%%
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#%%
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"RF MAE: {mae_rf:.3f}")
print(f"RF R^2: {r2_rf:.3f}")

#%%
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
plt.xlabel("Actual logS")
plt.ylabel("Predicted logS (RF)")
plt.title("Random Forest: Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()

#%%
plt.figure(figsize=(7,7))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction')
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='red', label='Random Forest')
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='green', label='Linear Regression')
plt.xlabel('Actual LogS')
plt.ylabel('Predicted LogS')
plt.title('Predicted vs Actual: RF vs Linear Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
import numpy as np

#%%
rf_importances = rf.feature_importances_
lr_importances = np.abs(lr.coef_)
feature_names = X.columns

#%%
x = np.arange(len(feature_names))
width = 0.35
plt.barh(x - width/2, rf_importances, height=width, label='Random Forest')
plt.barh(x + width/2, lr_importances, height=width, label='Linear Regression')
plt.yticks(x, feature_names)
plt.xlabel("Importance Score")
plt.title("Feature Importance: RF vs Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()

#%%
from deepchem.models import GraphConvModel
from deepchem.metrics import Metric

#%%
model = GraphConvModel(
    n_tasks=1,
    mode='regression',
    dropout=0.2,
    model_dir='graphconv_model',
    learning_rate=0.001
)

model.fit(train_dataset, nb_epoch=150, checkpoint_interval=10, early_stopping=True)

#%%
mae_metric = Metric(dc.metrics.mean_absolute_error)
r2_metric = Metric(dc.metrics.pearson_r2_score)

print("Evaluating on test set...")
test_scores = model.evaluate(test_dataset, [mae_metric, r2_metric], transformers)
print(f"GraphConv MAE: {test_scores['mean_absolute_error']:.3f}")
print(f"GraphConv R²: {test_scores['pearson_r2_score']:.3f}")

#%%
y_pred = model.predict(test_dataset)
y_true = test_dataset.y

plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='gray')
plt.xlabel("Actual logS")
plt.ylabel("Predicted logS")
plt.title("GraphConv: Predicted vs Actual")
plt.grid(True)
plt.show()

#%%
from sklearn.linear_model import Ridge

#%%
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

#%%
y_pred_ridge = ridge.predict(X_test)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"RF MAE: {mae_ridge:.3f}")
print(f"RF R^2: {r2_ridge:.3f}")

#%%
plt.scatter(y_test, y_pred_ridge, alpha=0.7, color='green')
plt.xlabel("Actual logS")
plt.ylabel("Predicted logS (RF)")
plt.title("Random Forest: Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()

# %%
from sklearn.linear_model import LassoCV

#%%
lasso = LassoCV(cv=5).fit(X_train, y_train)

#%%
y_pred_lasso = lasso.predict(X_test)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"RF MAE: {mae_lasso:.3f}")
print(f"RF R^2: {r2_lasso:.3f}")

#%%
plt.scatter(y_test, y_pred_lasso, alpha=0.7, color='green')
plt.xlabel("Actual logS")
plt.ylabel("Predicted logS (RF)")
plt.title("Random Forest: Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()
# %%
