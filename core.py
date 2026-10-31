import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score


def read_upload(file, filename):
    """CSV немесе Excel файлын оқиды."""
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file, sep=None, engine="python")


def make_close(df, date_col, price_col):
    """Кестеден күн бойынша баға қатарын жасайды."""
    idx = pd.to_datetime(df[date_col], errors="coerce")
    vals = pd.to_numeric(df[price_col], errors="coerce")
    s = pd.Series(vals.values, index=idx)
    s = s[~s.index.isna()].dropna().sort_index()
    return s[~s.index.duplicated()]


def _p2(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(2)]


def analyze(close):
    """close: күнделікті баға (pd.Series, индексі - күн).
    Қайтарады: Марков матрицасы, нәтиже кестесі, ақпарат (күндер, көлемдер)."""
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
    # Оқытудың соңғы жолының «ертеңгі күні» тексерудің бірінші күні болады,
    # сондықтан оны алып тастаймыз (жылыстауды болдырмау)
    tr = d[d.index < cut_date].iloc[:-1].copy()
    te = d[d.index >= cut_date].copy()

    # Марков өту матрицасы: тек оқыту бөлігінен
    M = pd.crosstab(tr["risk"], tr["target"], normalize="index")
    M = M.reindex(index=[0, 1, 2], columns=[0, 1, 2]).fillna(0)
    for k in [0, 1, 2]:
        tr[f"pm{k}"] = tr["risk"].map(M[k])
        te[f"pm{k}"] = te["risk"].map(M[k])

    f1 = ["a1", "a2", "a3", "vol5", "vol20", "r1", "risk"]
    f2 = f1 + ["pm0", "pm1", "pm2"]
    y = te["target"]
    high = (y == 2).astype(int)

    def row(key, pred, score):
        return (key, float((pred == y).mean()),
                float(f1_score(y, pred, average="macro")),
                float(roc_auc_score(high, score)) if score is not None else np.nan)

    mode = tr["target"].mode()[0]
    rows = [row("base", np.full(len(y), mode), None)]
    rows.append(row("markov", te["risk"].map(M.idxmax(axis=1)).values, te["pm2"].values))

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(tr[f1], tr["target"])
    rows.append(row("lr", lr.predict(te[f1]), _p2(lr, te[f1])))

    for key, feats in [("rf", f1), ("comb", f2)]:
        m = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        m.fit(tr[feats], tr["target"])
        rows.append(row(key, m.predict(te[feats]), _p2(m, te[feats])))

    fmt = lambda ts: ts.strftime("%d.%m.%Y")
    info = {
        "start": fmt(close.index[0]), "end": fmt(close.index[-1]), "n": len(close),
        "train_end": fmt(tr.index[-1]), "n_train": len(tr),
        "test_start": fmt(te.index[0]), "test_end": fmt(te.index[-1]), "n_test": len(te),
    }
    return M, pd.DataFrame(rows, columns=["key", "acc", "f1", "auc"]), info
