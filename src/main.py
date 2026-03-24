import kagglehub
from kagglehub import KaggleDatasetAdapter

dataset = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    "creditcard.csv"
)

print(dataset.head())