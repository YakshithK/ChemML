import deepchem as dc
from deepchem.models import GraphConvModel
import warnings
from deepchem.metrics import Metric
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*GetValence.*")

print("DEEPCHEM VERSION: ", dc.__version__)

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv', splitter='random')

train_dataset, valid_dataset, test_dataset = datasets

model = GraphConvModel(
    n_tasks=1,
    mode='regression',
    dropout=0.2,
    model_dir='graphconv_model',
    learning_rate=0.001
)

model.fit(train_dataset, nb_epoch=50)

mae_metric = Metric(dc.metrics.mean_absolute_error)
r2_metric = Metric(dc.metrics.pearson_r2_score)

print("Evaluating on test set...")
test_scores = model.evaluate(test_dataset, [mae_metric, r2_metric], transformers)
print(f"GraphConv MAE: {test_scores['mean_absolute_error']:.3f}")
print(f"GraphConv R²: {test_scores['pearson_r2_score']:.3f}")

y_pred = model.predict(test_dataset)
y_true = test_dataset.y

plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='gray')
plt.xlabel("Actual logS")
plt.ylabel("Predicted logS")
plt.title("GraphConv: Predicted vs Actual")
plt.grid(True)
plt.show()