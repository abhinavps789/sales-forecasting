from os.path import exists
import pandas as pd

# Input data

df_path = "train.csv.zip" if exists("train.csv.zip") else "/content/train.csv.zip"
df = pd.read_csv(df_path)
df = df.sample(n=19_000,random_state=0)

df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)
df["date"] = pd.to_datetime(df["date"])

df.sort_values("date",inplace=True)
df.reset_index(inplace=True,drop=True)
df.head()

# Splitting into training and testing

train = df[df["date"] < "2017-01-01"]
test = df[df["date"] >= "2017-01-01"]

# Features - input values
# Labels - what we want to predict

train_features = train.drop(columns=["sales"])
train_target = train["sales"]
test_features = test.drop(columns=["sales"])
test_target = test["sales"]

# Enrich features

from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType

enricher = FeaturesEnricher(
    search_keys = {
        "date":SearchKey.DATE,
    },
    cv = CVType.time_series
)
enricher.fit(train_features,
             train_target,
             eval_set=[(test_features,test_target)])

from catboost import CatBoostRegressor
from catboost.utils import eval_metric

model = CatBoostRegressor(verbose=False,allow_writing_files=False, random_state=0)

enricher.calculate_metrics(
    train_features, train_target,
    eval_set = [(test_features,test_target)],
    estimator = model,
    scoring = "mean_absolute_percentage_error"
)

enriched_train_features = enricher.transform(train_features,keep_input = True)
enriched_test_features = enricher.transform(test_features,keep_input = True)
enriched_train_features.head()

model.fit(train_features,train_target)
preds = model.predict(test_features)
eval_metric(test_target.values,preds,"SMAPE")

model.fit(enriched_train_features,train_target)
enriched_preds = model.predict(enriched_test_features)
eval_metric(test_target.values,enriched_preds,"SMAPE")
