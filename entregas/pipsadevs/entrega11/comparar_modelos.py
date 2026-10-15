"""Compara varios clasificadores de scikit-learn sobre las features de `rafa_bot.py`.

Usa validación cruzada estratificada y repetida (RepeatedStratifiedKFold) para
obtener una estimación HONESTA del rendimiento, sin el sesgo de evaluar sobre
los mismos casos con los que se entrena. Reporta, para cada modelo:

  - la métrica oficial de la competición en el corte 0.5
    (puntuacion = 1 - (4·FN + FP) / nº de casos fieles),
  - el AUROC,
  - el mejor umbral sobre las predicciones out-of-fold y la métrica en ese umbral,
  - las mismas métricas restringidas a los casos `H` (los que más se parecen al test).

Uso:
    python3 comparar_modelos.py
    python3 comparar_modelos.py --datos train.jsonl --folds 5 --repeats 3
    python3 comparar_modelos.py --guardar resultados_modelos.json
"""
import argparse
import json
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluar_local import puntuacion
from rafa_bot import Guardian

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Definición de los modelos a comparar.
# Cada entrada: nombre -> (estimador, acepta_sample_weight)
# Todos van envueltos en un Pipeline con StandardScaler (los árboles lo ignoran).
# --------------------------------------------------------------------------- #
def construir_modelos():
    modelos = {}

    def add(nombre, est, sw=True):
        modelos[nombre] = (est, sw)

    add("logistic_regression", LogisticRegression(max_iter=2000, C=1.0, random_state=42))
    add("logistic_regression_l1", LogisticRegression(max_iter=2000, C=0.5, penalty="l1",
                                                     solver="liblinear", random_state=42))
    add("logistic_regression_bal", LogisticRegression(max_iter=2000, C=1.0,
                                                      class_weight="balanced", random_state=42))
    add("ridge_classifier", LogisticRegression(max_iter=2000, C=1.0, penalty="l2",
                                               solver="lbfgs", random_state=42))

    add("decision_tree", DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                                random_state=42))
    add("random_forest", RandomForestClassifier(n_estimators=300, max_depth=14,
                                                random_state=42, n_jobs=-1))
    add("random_forest_bal", RandomForestClassifier(n_estimators=300, max_depth=14,
                                                    class_weight="balanced_subsample",
                                                    random_state=42, n_jobs=-1))
    add("extra_trees", ExtraTreesClassifier(n_estimators=300, max_depth=14,
                                            random_state=42, n_jobs=-1))
    add("extra_trees_bal", ExtraTreesClassifier(n_estimators=300, max_depth=14,
                                                class_weight="balanced_subsample",
                                                random_state=42, n_jobs=-1))
    add("gradient_boosting", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                        random_state=42))
    add("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=300,
                                                                 random_state=42))
    add("adaboost", AdaBoostClassifier(n_estimators=200, random_state=42))
    add("bagging", BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1))

    add("knn", KNeighborsClassifier(n_neighbors=15, weights="distance"), sw=False)
    add("gaussian_nb", GaussianNB(), sw=False)
    add("lda", LinearDiscriminantAnalysis(), sw=False)
    add("qda", QuadraticDiscriminantAnalysis(reg_param=0.1), sw=False)

    add("svc_rbf", SVC(C=3.0, gamma="scale", probability=True, random_state=42))
    add("svc_rbf_bal", SVC(C=3.0, gamma="scale", probability=True,
                           class_weight="balanced", random_state=42))
    add("calibrated_svc", CalibratedClassifierCV(
        SVC(C=3.0, gamma="scale", random_state=42), method="sigmoid", cv=3))
    add("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                             early_stopping=True, random_state=42), sw=False)

    # Ensembles
    add("voting_gb_rf", VotingClassifier(
        estimators=[
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)),
        ],
        voting="soft"), sw=False)
    add("voting_top", VotingClassifier(
        estimators=[
            ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)),
            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ],
        voting="soft"), sw=False)
    add("voting_soft", VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)),
            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
            ("svc", SVC(C=3.0, gamma="scale", probability=True, random_state=42)),
        ],
        voting="soft"), sw=False)
    add("stacking", StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)),
            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=-1), sw=False)

    return modelos


def cargar_casos(ruta):
    casos = []
    with open(ruta, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                casos.append(json.loads(line))
    return casos


def matriz_features(casos):
    """Extrae features sin entrenar (evita el coste de Guardian.__init__)."""
    extractor = Guardian.__new__(Guardian)
    X = np.array([extractor._extraer_features(c) for c in casos], dtype=np.float64)
    y = np.array([c["etiqueta"] for c in casos], dtype=np.int64)
    es_h = np.array([str(c.get("id", "")).startswith("H") for c in casos])
    # Grupo = escenario base (p. ej. "H101" para H101::r1, H101::r2...). Las
    # variantes comparten pregunta/fuentes, así que no deben separarse entre folds.
    grupos = np.array([str(c.get("id", "")).split("::")[0] for c in casos])
    return X, y, es_h, grupos


def construir_splits(X, y, grupos, folds, repeats, agrupar):
    """Lista de pares (train_idx, test_idx) con CV estratificada (y agrupada)."""
    splits = []
    for rep in range(repeats):
        if agrupar:
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42 + rep)
            iterador = cv.split(X, y, grupos)
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42 + rep)
            iterador = cv.split(X, y)
        splits.extend(list(iterador))
    return splits


def _scores(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    d = est.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def _fit(est, X, y, sw, sample_weight):
    if sw and sample_weight is not None:
        est.fit(X, y, clf__sample_weight=sample_weight)
    else:
        est.fit(X, y)


def _mejor_umbral(y, scores):
    """Umbral que maximiza la métrica oficial sobre un conjunto etiquetado."""
    mejores = [(puntuacion(y.tolist(), (scores >= t).astype(int).tolist()), t)
               for t in np.unique(scores)]
    return max(mejores, key=lambda p: p[0])


def evaluar_modelo(nombre, est, sw, X, y, es_h, splits, sample_weight):
    """Devuelve métricas agregadas sobre las repeticiones de CV.

    La métrica honesta (`metrica_cal`) elige el umbral SOLO con las predicciones
    de entrenamiento de cada fold y lo aplica al fold de test, imitando lo que
    hará el bot final al calibrar el corte. `metrica_mejor` usa el umbral
    oráculo (cota superior optimista) y `auroc` mide el orden puro.
    """
    agg = {k: [] for k in ("metrica_05", "metrica_cal", "metrica_mejor", "auroc",
                           "umbral", "metrica_H_cal", "metrica_H_calH",
                           "metrica_H_mejor", "auroc_H", "umbral_H")}

    for train_idx, test_idx in splits:
        modelo = Pipeline([("scaler", StandardScaler()), ("clf", est)])
        w_tr = sample_weight[train_idx] if (sw and sample_weight is not None) else None
        _fit(modelo, X[train_idx], y[train_idx], sw, w_tr)
        s_tr = _scores(modelo, X[train_idx])
        s_te = _scores(modelo, X[test_idx])

        y_tr, y_te = y[train_idx], y[test_idx]
        met_tr, umbral = _mejor_umbral(y_tr, s_tr)

        agg["metrica_05"].append(puntuacion(y_te.tolist(), (s_te >= 0.5).astype(int).tolist()))
        agg["metrica_cal"].append(puntuacion(y_te.tolist(), (s_te >= umbral).astype(int).tolist()))
        agg["metrica_mejor"].append(_mejor_umbral(y_te, s_te)[0])
        agg["umbral"].append(umbral)
        agg["auroc"].append(roc_auc_score(y_te, s_te) if len(set(y_te)) > 1 else 0.5)

        mask_h_tr = es_h[train_idx]
        mask_h = es_h[test_idx]
        if mask_h.any() and mask_h_tr.any():
            y_h, s_h = y_te[mask_h], s_te[mask_h]
            y_h_tr, s_h_tr = y_tr[mask_h_tr], s_tr[mask_h_tr]
            if len(set(y_h)) > 1:
                umbral_h = _mejor_umbral(y_h_tr, s_h_tr)[1]
                agg["metrica_H_cal"].append(
                    puntuacion(y_h.tolist(), (s_h >= umbral).astype(int).tolist()))
                agg["metrica_H_calH"].append(
                    puntuacion(y_h.tolist(), (s_h >= umbral_h).astype(int).tolist()))
                agg["metrica_H_mejor"].append(_mejor_umbral(y_h, s_h)[0])
                agg["auroc_H"].append(roc_auc_score(y_h, s_h))
                agg["umbral_H"].append(umbral_h)

    def resumen(vals):
        vals = np.array(vals, dtype=float)
        if vals.size == 0:
            return 0.0, 0.0
        return float(vals.mean()), float(vals.std())

    out = {"modelo": nombre}
    for clave, prefijo in (("metrica_05", "metrica_05"), ("metrica_cal", "metrica"),
                           ("metrica_mejor", "metrica_mejor"), ("umbral", "umbral"),
                           ("auroc", "auroc")):
        out[prefijo], out[prefijo + "_std"] = resumen(agg[clave])
    for clave, prefijo in (("metrica_H_cal", "metrica_H"),
                           ("metrica_H_calH", "metrica_H_calH"),
                           ("metrica_H_mejor", "metrica_H_mejor"),
                           ("auroc_H", "auroc_H"), ("umbral_H", "umbral_H")):
        out[prefijo], out[prefijo + "_std"] = resumen(agg[clave])
    return out


def oof_probabilidades(est, sw, X, y, splits, sample_weight):
    """Probabilidad out-of-fold de cada caso (media sobre las repeticiones)."""
    oof = np.zeros(len(y))
    cuentas = np.zeros(len(y))
    for train_idx, test_idx in splits:
        modelo = Pipeline([("scaler", StandardScaler()), ("clf", est)])
        w_tr = sample_weight[train_idx] if (sw and sample_weight is not None) else None
        _fit(modelo, X[train_idx], y[train_idx], sw, w_tr)
        oof[test_idx] += _scores(modelo, X[test_idx])
        cuentas[test_idx] += 1
    return oof / np.maximum(cuentas, 1)


def informe_calibracion(nombre, est, sw, X, y, es_h, splits, sample_weight):
    """Calcula el umbral óptimo OOF y el factor `a` para recolocar el corte en 0.5.

    El bot final devuelve  score = a·p / (a·p + (1-p));  con a = (1-t)/t el corte
    0.5 cae justo en la probabilidad t que maximiza la métrica.
    """
    oof = oof_probabilidades(est, sw, X, y, splits, sample_weight)
    print(f"\nCalibración OOF de «{nombre}»:")
    for etiqueta, mask in (("todos", np.ones(len(y), bool)), ("solo H", es_h)):
        m, t = _mejor_umbral(y[mask], oof[mask])
        a = (1.0 - t) / t if t > 0 else float("inf")
        print(f"  {etiqueta:<8} umbral óptimo {t:.4f} · métrica {m:.3f} · "
              f"a=(1-t)/t = {a:.3f}")
    # Referencia teórica: con probabilidades calibradas y coste FN=4·FP, el corte
    # óptimo es p=0.2, que se recoloca en 0.5 con a=4.
    print(f"  {'teórico':<8} p*=0.2 · a=4.000 (si las probabilidades estuvieran calibradas)")
    return oof


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", default="train.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--peso-h", type=float, default=3.0,
                    help="sample_weight de los casos H (1.0 = sin ponderar)")
    ap.add_argument("--guardar", default=None, help="ruta JSON donde guardar los resultados")
    ap.add_argument("--solo", default=None,
                    help="lista separada por comas de modelos a comparar (por defecto, todos)")
    ap.add_argument("--sin-agrupar", action="store_true",
                    help="CV estratificada normal en vez de agrupada por escenario base")
    ap.add_argument("--calibrar", default=None,
                    help="nombre de un modelo: calcula su umbral óptimo OOF y termina")
    args = ap.parse_args(argv)

    casos = cargar_casos(args.datos)
    X, y, es_h, grupos = matriz_features(casos)
    print(f"{len(casos)} casos · {X.shape[1]} features · {int(es_h.sum())} casos H · "
          f"{int((y == 1).sum())} alucinaciones · {len(set(grupos))} escenarios base")

    sample_weight = np.where(es_h, args.peso_h, 1.0).astype(np.float64)
    splits = construir_splits(X, y, grupos, args.folds, args.repeats, not args.sin_agrupar)

    todos = construir_modelos()
    if args.solo:
        pedidos = [m.strip() for m in args.solo.split(",") if m.strip()]
        desconocidos = [m for m in pedidos if m not in todos]
        if desconocidos:
            raise SystemExit(f"Modelos desconocidos: {desconocidos}\nDisponibles: {list(todos)}")
        todos = {m: todos[m] for m in pedidos}

    if args.calibrar:
        if args.calibrar not in todos:
            raise SystemExit(f"Modelo desconocido: {args.calibrar}\nDisponibles: {list(todos)}")
        est, sw = todos[args.calibrar]
        informe_calibracion(args.calibrar, est, sw, X, y, es_h, splits, sample_weight)
        return []

    resultados = []
    for nombre, (est, sw) in todos.items():
        r = evaluar_modelo(nombre, est, sw, X, y, es_h, splits, sample_weight)
        resultados.append(r)
        print(f"  ✓ {nombre:<26} mét_H* {r['metrica_H_calH']:.3f} · "
              f"AUROC_H {r['auroc_H']:.3f} · mét_cal {r['metrica']:.3f} · "
              f"AUROC {r['auroc']:.3f} · umbral_H {r['umbral_H']:.3f}")

    # Ranking: el test se parece a los casos H. El AUROC en H es la señal más
    # estable (independiente del umbral), así que ordenamos por él; desempatamos
    # por la métrica calibrada en H y luego por la global.
    resultados.sort(key=lambda r: (r["auroc_H"], r["metrica_H_calH"], r["metrica"]),
                    reverse=True)

    print("\n" + "=" * 110)
    print(f"{'modelo':<26}{'AUROC_H':>9}{'mét_H*':>9}{'±std':>7}{'mét_cal':>9}"
          f"{'AUROC':>8}{'umbral_H':>10}{'mét@mej':>9}{'mét_H@mej':>11}")
    print("-" * 110)
    for r in resultados:
        print(f"{r['modelo']:<26}{r['auroc_H']:>9.3f}{r['metrica_H_calH']:>9.3f}"
              f"{r['metrica_H_calH_std']:>7.3f}{r['metrica']:>9.3f}{r['auroc']:>8.3f}"
              f"{r['umbral_H']:>10.3f}{r['metrica_mejor']:>9.3f}{r['metrica_H_mejor']:>11.3f}")
    print("=" * 110)
    print("  AUROC_H  = capacidad de ordenar los casos H (proxy del test), estable")
    print("  mét_H*   = métrica en H con umbral calibrado solo con H (ruidosa pero orientativa)")
    print(f"GANADOR: {resultados[0]['modelo']}  (AUROC_H {resultados[0]['auroc_H']:.3f}, "
          f"mét_H* {resultados[0]['metrica_H_calH']:.3f}, AUROC {resultados[0]['auroc']:.3f})")

    if args.guardar:
        with open(args.guardar, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        print(f"Resultados guardados en {args.guardar}")

    return resultados


if __name__ == "__main__":
    main()
