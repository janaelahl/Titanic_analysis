

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = Path("data/train.csv")
CLEAN_PATH = Path("data/train_clean.csv")
FIG_DIR = Path("figures")
OUT_DIR = Path("outputs")

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("\n=== STEP 1: Load Data ===")
if not DATA_PATH.exists():
    raise FileNotFoundError(" Missing data/train.csv. Download Titanic train.csv from Kaggle and place it in data/")

df = pd.read_csv(DATA_PATH)
print(" Data loaded. Shape:", df.shape)
print(df.head())


print("\n=== STEP 2: Clean Data ===")


df["Age"].fillna(df["Age"].median(), inplace=True)


df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)


df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")
df["Pclass"] = df["Pclass"].astype("category")


df.to_csv(CLEAN_PATH, index=False)
print(" Cleaned dataset saved at:", CLEAN_PATH.resolve())


print("\n=== STEP 3: Summary & Insights ===")


summary = df.describe(include="all")
summary.to_csv(OUT_DIR / "summary.csv")
print(" Summary stats saved at outputs/summary.csv")


print("\n Survival by gender:\n", df.groupby("Sex")["Survived"].mean())
df.groupby("Sex")["Survived"].mean().to_csv(OUT_DIR / "survival_by_gender.csv")


print("\n📊 Survival by class:\n", df.groupby("Pclass")["Survived"].mean())
df.groupby("Pclass")["Survived"].mean().to_csv(OUT_DIR / "survival_by_class.csv")


print("\n📊 Survival by class & gender:\n", df.groupby(["Pclass", "Sex"])["Survived"].mean())
df.groupby(["Pclass", "Sex"])["Survived"].mean().to_csv(OUT_DIR / "survival_by_class_gender.csv")


print("\n=== STEP 4: Visualizations ===")
sns.set(style="whitegrid")


plt.figure()
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Gender")
plt.savefig(FIG_DIR / "survival_by_gender.png")


plt.figure()
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Class")
plt.savefig(FIG_DIR / "survival_by_class.png")


plt.figure()
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30)
plt.title("Age Distribution by Survival")
plt.savefig(FIG_DIR / "age_distribution.png")


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(FIG_DIR / "correlation_heatmap.png")

print(" Charts saved in figures/ folder")


print("\n=== STEP 5: Simple Logistic Regression Model ===")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target]

cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(steps=[
    ("pre", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(" Accuracy:", accuracy_score(y_test, preds))
print("\n Classification Report:\n", classification_report(y_test, preds))

