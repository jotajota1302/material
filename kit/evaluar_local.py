"""Evalúa vuestro bot (solo Python estándar, sin modelos) sobre casos etiquetados (p. ej. train.jsonl) con
la MISMA métrica que el leaderboard.

  python evaluar_local.py --bot mi_bot.py --datos train.jsonl
  python evaluar_local.py --bot mi_bot.py --datos train.jsonl --solo-escenarios

Puntuación = 1 - (4·FN + FP) / nº de casos fieles, con el corte en score >= 0.5:
  1 = perfecto · 0 = lo mismo que marcar todo como alucinación · < 0 = peor que eso.
FN = alucinación que se os escapa (pesa 4), FP = falsa alarma sobre un caso fiel.
También imprime el AUROC (si vuestro bot ordena bien los casos, al margen del corte)
y el umbral que daría la mejor puntuación con vuestros scores, para que podáis
recolocar vuestro corte en 0.5.
"""
import argparse
import json


def puntuacion(y_true, y_pred, w_fn=4.0):
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    n_neg = sum(1 for t in y_true if t == 0)
    if n_neg == 0:
        n_pos = sum(1 for t in y_true if t == 1)
        return 1.0 - fn / n_pos if n_pos else 1.0
    return 1.0 - (w_fn * fn + fp) / n_neg


def auroc(y_true, scores):
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    if not pos or not neg:
        return 0.5
    g = sum(1.0 if sp > sn else 0.5 if sp == sn else 0.0 for sp in pos for sn in neg)
    return g / (len(pos) * len(neg))


def evaluar(bot_cls, casos, umbral=0.5):
    from generar_predicciones import _caso_para_bot
    g = bot_cls()
    scores = [min(1.0, max(0.0, float(g.score(_caso_para_bot(c))))) for c in casos]
    y = [c["etiqueta"] for c in casos]
    pred = [1 if s >= umbral else 0 for s in scores]
    mejor = max(((puntuacion(y, [1 if s >= t else 0 for s in scores]), t)
                 for t in sorted(set(scores))), default=(0.0, umbral))
    return {"puntuacion": puntuacion(y, pred), "auroc": auroc(y, scores),
            "fp": sum(1 for t, p in zip(y, pred) if t == 0 and p == 1),
            "fn": sum(1 for t, p in zip(y, pred) if t == 1 and p == 0),
            "mejor_umbral": mejor[1], "puntuacion_mejor_umbral": mejor[0]}


def main(argv=None):
    from generar_predicciones import _cargar_bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="ejemplo_bot.py")
    ap.add_argument("--datos", default="train.jsonl")
    ap.add_argument("--solo-escenarios", action="store_true",
                    help="solo los casos de varias fuentes (ids N, s, T, G, R, H), los que más se parecen al test")
    ap.add_argument("--prefijo", default=None,
                    help="solo los casos cuyo id empieza por este prefijo (p. ej. H: los más parecidos al test)")
    args = ap.parse_args(argv)
    casos = [json.loads(l) for l in open(args.datos, encoding="utf-8") if l.strip()]
    if args.solo_escenarios:
        casos = [c for c in casos if c["id"][0] in "NsTGRH"]
    if args.prefijo:
        casos = [c for c in casos if c["id"].startswith(args.prefijo)]
    if casos and "etiqueta" not in casos[0]:
        raise SystemExit(
            f"{args.datos} no tiene etiquetas, así que no se puede evaluar: es lo que tenéis que predecir.\n"
            "  Para medir vuestro bot:   python evaluar_local.py --bot mi_bot.py --datos train.jsonl --prefijo H\n"
            "  Para generar la entrega:  python generar_predicciones.py --bot mi_bot.py "
            "--test test_publico.jsonl --salida predicciones.csv")
    r = evaluar(_cargar_bot(args.bot), casos)
    print(f"{len(casos)} casos · puntuación {r['puntuacion']:.3f} · AUROC {r['auroc']:.3f} · "
          f"{r['fp']} falsas alarmas · {r['fn']} alucinaciones escapadas")
    print(f"Con vuestros scores, el mejor corte sería {r['mejor_umbral']:.3f} "
          f"(puntuación {r['puntuacion_mejor_umbral']:.3f}). El organizador corta en 0.5: "
          f"si vuestro corte es otro, reescalad el score para que caiga en 0.5.")


if __name__ == "__main__":
    main()
