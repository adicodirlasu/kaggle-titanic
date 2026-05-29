"""Titanic - Kaggle. Feature engineering + Gradient Boosting cu validare CV."""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = pd.concat([train, test], sort=False)

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Miss",
    "Mme": "Mrs", "Don": "Rare", "Dona": "Rare", "Lady": "Rare", "Countess": "Rare",
    "Jonkheer": "Rare", "Sir": "Rare", "Capt": "Rare", "Ms": "Miss",
}

def fe(df, ref):
    df = df.copy()
    df["Title"] = df["Name"].apply(lambda n: re.search(r",\s*([^\.]+)\.", n).group(1).strip())
    df["Title"] = df["Title"].map(TITLE_MAP).fillna("Rare")
    # Age: imputare pe mediana per Title (din setul complet)
    age_med = ref.assign(T=ref["Name"].apply(
        lambda n: TITLE_MAP.get(re.search(r",\s*([^\.]+)\.", n).group(1).strip(), "Rare"))
        ).groupby("T")["Age"].median()
    df["Age"] = df.apply(lambda r: age_med[r["Title"]] if pd.isna(r["Age"]) else r["Age"], axis=1)
    df["Fare"] = df["Fare"].fillna(ref["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Sex"] = (df["Sex"] == "male").astype(int)
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)
    feats = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "HasCabin"] \
        + [c for c in df.columns if c.startswith(("Embarked_", "Title_"))]
    return df[feats]

X = fe(train, full)
y = train["Survived"]
X_test = fe(test, full).reindex(columns=X.columns, fill_value=0)

model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                   learning_rate=0.05, subsample=0.9, random_state=42)
cv = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(f"CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")

model.fit(X, y)
pred = model.predict(X_test).astype(int)
out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})
out.to_csv("submission.csv", index=False)
print("submission.csv scris:", out.shape, "| supraviețuitori prezisi:", int(pred.sum()))
