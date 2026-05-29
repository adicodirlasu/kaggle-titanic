"""Titanic v4 - XGBoost tuned + Women-Child-Group survival heuristic (Chris Deotte).

Logica WCG:
- Grupuri = Surname + Ticket-prefix (familii care calatoresc impreuna).
- Barbatii adulti se prezic individual (modelul ML).
- Femeile & copiii dintr-un grup partajeaza soarta: daca grupul (vazut in train)
  a murit in totalitate -> toti mor; daca a supravietuit -> toti traiesc.
"""
import re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
full = pd.concat([train, test], sort=False).reset_index(drop=True)
TITLE_MAP = {"Mr":"Mr","Miss":"Miss","Mrs":"Mrs","Master":"Master","Dr":"Rare","Rev":"Rare",
    "Col":"Rare","Major":"Rare","Mlle":"Miss","Mme":"Mrs","Don":"Rare","Dona":"Rare","Lady":"Rare",
    "Countess":"Rare","Jonkheer":"Rare","Sir":"Rare","Capt":"Rare","Ms":"Miss"}
title_of = lambda n: TITLE_MAP.get(re.search(r",\s*([^\.]+)\.", n).group(1).strip(), "Rare")

full["Title"] = full["Name"].apply(title_of)
full["Surname"] = full["Name"].apply(lambda n: n.split(",")[0].strip())
full["Fare"] = full["Fare"].fillna(full["Fare"].median())
full["FarePer"] = full["Fare"] / full.groupby("Ticket")["Ticket"].transform("count")
full["Age"] = full["Age"].fillna(full.groupby("Title")["Age"].transform("median"))
full["Embarked"] = full["Embarked"].fillna("S")
# femeie sau copil (<16): candidat pentru regula de grup
full["WC"] = (full["Sex"] == "female") | (full["Age"] < 16)
# grup = Surname + prefix bilet (ignora ultima cifra ca sa lege bilete consecutive)
full["TickPref"] = full["Ticket"].str.replace(r"\d", "", regex=True).str.strip()
full["GroupId"] = full["Surname"] + "-" + full["Pclass"].astype(str) + "-" + \
                  full["Ticket"].str.extract(r"(\d+)$")[0].fillna("X").str[:-1]

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
n = len(train)
X, X_test = X_all.iloc[:n].reset_index(drop=True), X_all.iloc[n:].reset_index(drop=True)
y = train["Survived"]
cv = StratifiedKFold(5, shuffle=True, random_state=42)
xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=1.0)

# --- evaluare CV onesta a heuristicii ---
# baseline OOF al modelului
oof = cross_val_predict(xgb, X, y, cv=cv)
base_acc = (oof == y).mean()

# aplica WCG pe predictiile OOF: pentru fiecare femeie/copil din train,
# foloseste rata grupului calculata din CELELALTE fold-uri (leakage-safe via group leave-out aprox)
oof_wcg = oof.copy()
tr = full.iloc[:n]
for gid, g in tr.groupby("GroupId"):
    wc = g[g["WC"]]
    if len(wc) < 2:
        continue
    for i in wc.index:
        others = wc.drop(i)
        if len(others) == 0:
            continue
        r = others["Survived"].mean()
        oof_wcg[i] = 1 if r > 0.5 else 0
wcg_acc = (oof_wcg == y).mean()

print(f"XGBoost OOF accuracy        : {base_acc:.4f}")
print(f"+ WCG heuristic (leave-one) : {wcg_acc:.4f}")
print(f"Diferenta                   : {wcg_acc - base_acc:+.4f}")

# --- predictie finala pe test ---
xgb.fit(X, y)
pred = pd.Series(xgb.predict(X_test).astype(int), index=test.index)
test_idx = list(range(n, len(full)))
wcg_train = tr[tr["WC"]].groupby("GroupId")["Survived"].agg(["mean","count"])
overwritten = 0
for pos, j in enumerate(test_idx):
    if not full.loc[j, "WC"]:
        continue
    gid = full.loc[j, "GroupId"]
    if gid in wcg_train.index:
        r = wcg_train.loc[gid, "mean"]
        new = 1 if r > 0.5 else 0
        if new != pred.iloc[pos]:
            overwritten += 1
        pred.iloc[pos] = new

pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":pred.values}).to_csv(
    "submission_v4.csv", index=False)
print(f"\nWCG a suprascris {overwritten} predictii pe test; "
      f"submission_v4.csv scris ({int(pred.sum())} supravietuitori)")
