"""Titanic v3 - adauga XGBoost + LightGBM (cu tuning) la comparatie + ensemble final."""
import re, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
full = pd.concat([train, test], sort=False)
TITLE_MAP = {"Mr":"Mr","Miss":"Miss","Mrs":"Mrs","Master":"Master","Dr":"Rare","Rev":"Rare",
    "Col":"Rare","Major":"Rare","Mlle":"Miss","Mme":"Mrs","Don":"Rare","Dona":"Rare","Lady":"Rare",
    "Countess":"Rare","Jonkheer":"Rare","Sir":"Rare","Capt":"Rare","Ms":"Miss"}
title_of = lambda n: TITLE_MAP.get(re.search(r",\s*([^\.]+)\.", n).group(1).strip(), "Rare")

full["Title"] = full["Name"].apply(title_of)
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["FarePer"] = full["Fare"] / full.groupby("Ticket")["Ticket"].transform("count")
full["Age"] = full["Age"].fillna(full.groupby("Title")["Age"].transform("median"))
full["Embarked"] = full["Embarked"].fillna("S")

def build(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["Sex"] = (df["Sex"] == "male").astype(int)
    df["TicketGroup"] = df.groupby("Ticket")["Ticket"].transform("count")
    df = pd.get_dummies(df, columns=["Embarked","Title","Deck"], drop_first=True)
    base = ["Pclass","Sex","Age","FarePer","FamilySize","IsAlone","HasCabin","TicketGroup"]
    return df[base + [c for c in df.columns if c.startswith(("Embarked_","Title_","Deck_"))]]

X_all = build(full)
X = X_all.iloc[:len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train):].reset_index(drop=True)
y = train["Survived"]
cv = StratifiedKFold(5, shuffle=True, random_state=42)

xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
lgb = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.9, random_state=42)
rf  = RandomForestClassifier(400, max_depth=6, min_samples_leaf=3, random_state=42)

print("=== CV accuracy (default) ===")
for name, m in [("GBM",gbm),("RandFor",rf),("XGBoost",xgb),("LightGBM",lgb)]:
    s = cross_val_score(m, X, y, cv=cv); print(f"{name:9s}: {s.mean():.4f} +/- {s.std():.4f}")

print("\n=== tuning XGBoost ===")
gx = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
    {"n_estimators":[200,400],"max_depth":[2,3,4],"learning_rate":[0.02,0.05],
     "subsample":[0.8,1.0],"colsample_bytree":[0.8,1.0]}, cv=cv, n_jobs=-1)
gx.fit(X, y); print(f"XGB tuned : {gx.best_score_:.4f}  {gx.best_params_}")

print("\n=== tuning LightGBM ===")
gl = GridSearchCV(LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    {"n_estimators":[200,400],"num_leaves":[7,15,31],"learning_rate":[0.02,0.05],
     "subsample":[0.8,1.0],"colsample_bytree":[0.8,1.0]}, cv=cv, n_jobs=-1)
gl.fit(X, y); print(f"LGBM tuned: {gl.best_score_:.4f}  {gl.best_params_}")

ens = VotingClassifier([("xgb",gx.best_estimator_),("lgb",gl.best_estimator_),
                        ("gbm",gbm),("rf",rf)], voting="soft")
es = cross_val_score(ens, X, y, cv=cv); print(f"\nEnsemble  : {es.mean():.4f} +/- {es.std():.4f}")

cands = {"XGB tuned":(gx.best_score_,gx.best_estimator_),
         "LGBM tuned":(gl.best_score_,gl.best_estimator_),
         "Ensemble":(es.mean(),ens)}
tag, (sc, best) = max(cands.items(), key=lambda kv: kv[1][0])
best.fit(X, y); pred = best.predict(X_test).astype(int)
pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":pred}).to_csv("submission_v3.csv",index=False)
print(f"\nAles -> {tag} ({sc:.4f}); submission_v3.csv scris ({pred.sum()} supravietuitori)")
