# Titanic — Kaggle ([competiția](https://www.kaggle.com/competitions/titanic))

Predicția supraviețuirii pasagerilor. Progresie de la un model de bază la o soluție
care combină gradient boosting cu o euristică de grup.

## Progresia experimentelor

| Versiune | Abordare | CV Accuracy (5-fold) | **Public LB** |
|----------|----------|----------------------|---------------|
| v1 | GradientBoosting + feature engineering de bază | 0.8451 | **0.76315** 🏆 |
| v2 | Comparație 7 modele + tuning + ensemble | 0.8451 | (nesubmis) |
| v3 | XGBoost & LightGBM cu GridSearch | 0.8496 | 0.74641 |
| v4 | XGBoost tuned + Women-Child-Group heuristic | 0.8653 | 0.76076 |

### ⚠️ Lecția principală: CV ≠ Leaderboard

Pe CV ordinea era **v4 > v3 > v1**. Pe leaderboard-ul real s-a **inversat
complet**: **v1 > v4 > v3**.

- **v3** a avut al doilea cel mai bun CV dar a generalizat cel mai prost (0.746)
  — grid search-ul a optimizat zgomotul celor 891 de rânduri (overfitting de validare).
- **v4 (WCG)** a arătat +1.57 puncte pe CV dar n-a adus nimic real — grupurile
  construite din prefixul biletului au fost prea agresive.
- **v1, cel mai simplu model, a câștigat** (0.76315). Mai puține feature-uri,
  fără tuning, fără trucuri → mai puțin overfitting.

Pe un dataset mic (891 rânduri), complexitatea s-a întors împotriva noastră.
CV-ul nu e un proxy de încredere pentru leaderboard pe Titanic.
Kaggle reține automat cel mai bun scor (v1) pentru clasamentul privat.

## Soluția câștigătoare (v1) — cea mai simplă

Pe baza scorurilor reale de leaderboard, **v1 este soluția de referință**:

1. **Feature engineering de bază**
   - `Title` extras din nume (Mr/Miss/Mrs/Master/Rare)
   - `FamilySize`, `IsAlone`, `HasCabin`
   - Imputare: Age pe mediana per titlu, Fare/Embarked pe valori comune

2. **Model**: GradientBoosting
   `n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9`

Vezi `solve.py`. Submission: `submission.csv` (public LB 0.76315).

### Ce NU a funcționat (documentat pentru transparență)

- **v3** — tuning XGBoost/LightGBM cu GridSearch: cel mai bun CV după v4, dar
  cel mai prost pe LB (0.746). Overfitting clar.
- **v4** — Women-Child-Group heuristic (Chris Deotte): grupuri = familie + bilet,
  femeile/copiii primesc soarta grupului. +1.57 puncte pe CV, dar 0 câștig real
  (0.761). Construcția grupurilor din prefixul biletului a fost prea agresivă.

## Fișiere

| Fișier | Descriere |
|--------|-----------|
| `solve.py` | v1 — baseline GBM |
| `solve_v2.py` | v2 — comparație modele + ensemble |
| `solve_v3.py` | v3 — XGBoost + LightGBM + tuning |
| `solve_v4.py` | v4 — XGBoost + WCG heuristic |
| `submission.csv` | **← cel mai bun pe LB (0.76315), de la v1** |
| `train.csv`, `test.csv` | datele (format Kaggle) |

## Rulare

```bash
pip install -r requirements.txt
python3 solve.py        # soluția câștigătoare (v1)
```

## Submitere

Toate cele 4 versiuni au fost deja submise (autentificare prin `kaggle auth login`).
Kaggle reține automat cel mai bun scor pentru clasamentul privat.

**CLI:**
```bash
kaggle competitions submit -c titanic -f submission.csv -m "GBM baseline v1"
kaggle competitions submissions -c titanic   # vezi scorurile
```

> Notă auth: Kaggle CLI 2.2.0 folosește OAuth (`kaggle auth login`), nu vechiul
> `kaggle.json` cu username+key. Dacă există un `kaggle.json` legacy în `~/.kaggle/`,
> îl poate bloca — mută-l (`kaggle.json.legacy`) și folosește `credentials.json` din OAuth.

## Note

Titanic e un dataset mic (891 rânduri train). Modelele se saturează la ~0.83–0.85 CV,
dar **CV-ul nu prezice leaderboard-ul** — vezi lecția de mai sus. Modelul cel mai simplu
a generalizat cel mai bine. Tuning agresiv și euristici complexe au dus la overfitting.
