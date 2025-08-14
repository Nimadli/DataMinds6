import warnings

import joblib
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)

df = pd.read_parquet("multisim_dataset.parquet")
drop_columns = df.columns[14:131]
for column in drop_columns:
    df.drop(column, axis=1, inplace=True)
df = df.drop(["is_smartphone", "is_featurephone"], axis=1)

df.set_index("telephone_number", inplace=True)
df.dropna(axis=0, inplace=True)

onehot_features = ["simcard_type", "is_dualsim", "gndr"]
catboost_features = ["device_os_name", "dev_man", "region", "trf"]
numerical_features = ["tenure", "age_dev", "dev_num", "age"]

for column in numerical_features:
    df[column] = df[column].astype(int)

X = df[df.columns[:11]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

onehot_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder()),
    ]
)

catboost_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", CatBoostEncoder()),
    ]
)

numerical_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)
preprocessor = ColumnTransformer(
    [
        ("onehot", onehot_pipeline, onehot_features),
        ("catboost", catboost_pipeline, catboost_features),
        ("num", numerical_pipeline, numerical_features),
    ]
)

model = Pipeline(
    [
        ("preprocessing", preprocessor),
        ("model", XGBClassifier(eval_metric="logloss", n_estimators=100)),
    ]
)

model.fit(X_train, y_train)

joblib.dump(model, "xgb_model.pkl")
print("Model Saved")
