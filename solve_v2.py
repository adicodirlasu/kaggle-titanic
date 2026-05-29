"""Titanic v2 - feature engineering avansat + comparatie modele + tuning + ensemble."""
import re, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = pd.concat([train, test], sort=False)

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Miss",
    "Mme": "Mrs", "Don": "Rare", "Dona": "Rare", "Lady": "Rare", "Countess": "Rare",
    "Jonkheer": "Rare", "Sir": "Rare", "Capt": "Rare", "Ms": "Miss",
}

def title_of(n):
    return TITLE_MAP.get(re.search(r",\s*([^\.]+)\.", n).group(1).strip(), "Rare")

# --- familii / grupuri de bilete pentru "survival by group" ---
full["Title"] = full["Name"].apply(title_of)
full["Surname"] = full["Name"].apply(lambda n: n.split(",")[0].strip())
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
# tarif per persoana (biletele de grup au fare totala)
tk_count = full.groupby("Ticket")["Ticket"].transform("count")
full["FarePer"] = full["Fare"] / tk_count
age_med = full.groupby("Title")["Age"].transform("median")
full["Age"] = full["Age"].fillna(age_med)
full["Embarked"] = full["Embarked"].fillna("S")

def build(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["Sex"] = (df["Sex"] == "male").astype(int)
    df["FamilyBin"] = pd.cut(df["FamilySize"], [0,1,4,20], labels=["alone","small","large"])
    df["AgeBin"] = pd.cut(df["Age"], [0,12,18,35,60,200],
                          labels=["child","teen","adult","mid","senior"])
    df["TicketGroup"] = df.groupby("Ticket")["Ticket"].transform("count")
    df = pd.get_dummies(df, columns=["Embarked","Title","Deck","FamilyBin","AgeBin"],
                        drop_first=True)
    base = ["Pclass","Sex","Age","FarePer","FamilySize","IsAlone","HasCabin","TicketGroup"]
    feats = base + [c for c in df.columns if c.startswith(
        ("Embarked_","Title_","Deck_","FamilyBin_","AgeBin_"))]
    return df[feats]

X_all = build(full)
X = X_all.iloc[:len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train):].reset_index(drop=True)
y = train["Survived"]
cv = StratifiedKFold(5, shuffle=True, random_state=42)

models = {
    "LogReg":  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
    "SVC":     make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", probability=True)),
    "KNN":     make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "RandFor": RandomForestClassifier(400, max_depth=6, min_samples_leaf=3, random_state=42),
    "ExtraTr": ExtraTreesClassifier(400, max_depth=7, min_samples_leaf=3, random_state=42),
    "GBM":     GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, subsample=0.9, random_state=42),
}
print("=== CV accuracy per model ===")
scores = {}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv)
    scores[name] = s.mean()
    print(f"{name:8s}: {s.mean():.4f} +/- {s.std():.4f}")

# --- tuning GBM ---
grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
    {"n_estimators":[150,250],"max_depth":[2,3],"learning_rate":[0.03,0.05,0.1],
     "subsample":[0.8,0.9]}, cv=cv, n_jobs=-1)
grid.fit(X, y)
print(f"\nGBM tuned: {grid.best_score_:.4f}  {grid.best_params_}")

# --- ensemble (soft voting) din top performeri ---
ens = VotingClassifier([
    ("gbm", grid.best_estimator_),
    ("rf", models["RandFor"]),
    ("svc", models["SVC"]),
    ("lr", models["LogReg"]),
], voting="soft")
es = cross_val_score(ens, X, y, cv=cv)
print(f"Ensemble : {es.mean():.4f} +/- {es.std():.4f}")

# alege cel mai bun intre GBM-tuned si ensemble
if es.mean() >= grid.best_score_:
    best, tag = ens, f"ensemble ({es.mean():.4f})"
else:
    best, tag = grid.best_estimator_, f"GBM tuned ({grid.best_score_:.4f})"
best.fit(X, y)
pred = best.predict(X_test).astype(int)
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred}).to_csv(
    "submission_v2.csv", index=False)
print(f"\nAles -> {tag}; submission_v2.csv scris ({pred.sum()} supravietuitori)")
