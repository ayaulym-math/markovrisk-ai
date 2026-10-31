import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def analyze(close):
    """close: күнделікті жабылу бағасы (pd.Series).
    Қайтарады: Марков матрицасы, нәтиже кестесі, тексеру күндерінің саны."""
    ret = close.pct_change().dropna()
    absr = ret.abs()

    X = pd.DataFrame({
        "a1": absr, "a2": absr.shift(1), "a3": absr.shift(2),
        "vol5": ret.rolling(5).std(), "vol20": ret.rolling(20).std(),
        "r1": ret,
    })

    # Уақыт ретімен бөлу: алғашқы 80% оқыту, соңғы 20% тексеру
    idx = X.dropna().index
    cut_date = idx[int(len(idx) * 0.8)]

    # Тәуекел деңгейінің шектері тек оқыту бөлігінен алынады
    q1, q2 = absr[absr.index < cut_date].quantile([0.33, 0.67])
    risk = absr.apply(lambda a: 0 if a < q1 else (1 if a < q2 else 2))
    X["risk"] = risk

    d = pd.concat([X, risk.shift(-1).rename("target")], axis=1).dropna()
    d["target"] = d["target"].astype(int)
    tr = d[d.index < cut_date].copy()
    te = d[d.index >= cut_date].copy()

    # Марков өту матрицасы (оқыту бөлігінен)
    M = pd.crosstab(tr["risk"], tr["target"], normalize="index")
    M = M.reindex(index=[0, 1, 2], columns=[0, 1, 2]).fillna(0)
    for k in [0, 1, 2]:
        tr[f"pm{k}"] = tr["risk"].map(M[k])
        te[f"pm{k}"] = te["risk"].map(M[k])

    f1 = ["a1", "a2", "a3", "vol5", "vol20", "r1", "risk"]
    f2 = f1 + ["pm0", "pm1", "pm2"]
    y = te["target"]
    high = (y == 2).astype(int)

    rows = [("base", float((y == tr["target"].mode()[0]).mean()), np.nan)]
    markov_pred = te["risk"].map(M.idxmax(axis=1))
    rows.append(("markov", float((markov_pred == y).mean()),
                 float(roc_auc_score(high, te["pm2"]))))
    for key, feats in [("rf", f1), ("comb", f2)]:
        m = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        m.fit(tr[feats], tr["target"])
        p = m.predict(te[feats])
        s = m.predict_proba(te[feats])[:, list(m.classes_).index(2)]
        rows.append((key, float((p == y).mean()), float(roc_auc_score(high, s))))

    return M, pd.DataFrame(rows, columns=["key", "acc", "auc"]), len(te)
