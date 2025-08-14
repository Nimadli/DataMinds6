import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

model = joblib.load("xgb_model.pkl")


df = pd.read_parquet("multisim_dataset.parquet")
drop_columns = df.columns[14:131]
for column in drop_columns:
    df.drop(column, axis=1, inplace=True)
df = df.drop(["is_smartphone", "is_featurephone"], axis=1)

df.set_index("telephone_number", inplace=True)
df.dropna(axis=0, inplace=True)
X = df[df.columns[:11]]
y = df["target"]

# Use the original dataset and the same random_state to get the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
print("Predictions: ", y_pred)
print("accuracy: ", accuracy_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("f1: ", f1_score(y_test, y_pred))
