"""Guardian: cuatro detectores (distorsion, contradiccion, no_respaldado,
sobregeneralizacion) que alimentan, junto con otras señales, una regresión
logística entrenada en train.jsonl (coeficientes incrustados abajo).
Solo librería estándar. score() ya devuelve el corte en 0.5."""
import math
import re
import unicodedata

STOP = set("el la los las un una unos unas de del al y o en a con por para que se su sus es son fue fueron ha han lo le les como mas pero sin sobre entre".split())
QUANT_UP = set("todos todas todo toda siempre nunca ningun ninguna cualquier cada totalidad mayoria".split())
QUANT_DOWN = set("solo solamente algunos algunas varios varias aproximadamente alrededor casi".split())


def _norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _toks(s):
    return re.findall(r"[a-z0-9]+", _norm(s))


def _nums(s):
    out = set()
    for m in re.findall(r"\d+(?:[.,]\d+)*", s):
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", m):
            m = m.replace(".", "")
        out.add(m.replace(",", "."))
    return out


FQ_DOWN = set("solo solamente algunos algunas varios varias unos unas aproximadamente alrededor cerca casi mas menos".split())

MODELO = {"keys": ["cap_new", "cit_resp_max", "cit_resp_mean", "cit_resp_min", "det_contra", "det_dist", "det_nores", "det_sobre", "fr_best_contra_max", "fr_best_contra_mean", "fr_best_contra_min", "fr_best_src_max", "fr_best_src_mean", "fr_best_src_min", "fr_contra_todas_max", "fr_contra_todas_mean", "fr_contra_todas_min", "fr_resp_todas_max", "fr_resp_todas_mean", "fr_resp_todas_min", "llm_contra", "llm_none", "llm_nores", "llm_res_minconf", "nov_frac", "novc_frac", "num_all_in_cit", "num_miss", "num_miss_cit", "or_best_contra_max", "or_best_contra_mean", "or_best_contra_min", "or_best_src_max", "or_best_src_mean", "or_best_src_min", "or_contra_todas_max", "or_contra_todas_mean", "or_contra_todas_min", "or_resp_todas_max", "or_resp_todas_mean", "or_resp_todas_min", "or_sim_max", "or_sim_mean", "or_sim_min", "q_down_lost"], "mu": [0.01933, 0.35079, 0.32504, 0.28855, 0.27172, 0.23067, 0.61297, 0.06034, 0.61818, 0.45022, 0.2814, 0.76034, 0.60053, 0.39333, 0.4006, 0.25829, 0.1499, 0.72241, 0.53973, 0.33623, 0.26929, 0.00293, 0.10217, 0.97933, 0.22472, 0.39563, 0.40715, 0.24253, 0.73404, 0.45317, 0.42403, 0.39754, 0.4211, 0.38276, 0.33356, 0.35008, 0.32291, 0.30543, 0.45971, 0.4164, 0.37169, 0.8036, 0.78329, 0.76149, 0.0867], "sd": [0.14595, 0.43311, 0.40676, 0.40363, 0.36931, 0.26496, 0.32969, 0.18187, 0.40469, 0.3357, 0.35289, 0.36825, 0.33427, 0.4025, 0.42096, 0.3192, 0.31305, 0.38729, 0.34603, 0.4091, 0.43406, 0.05404, 0.28527, 0.03432, 0.19339, 0.26705, 0.4913, 0.44603, 0.8884, 0.41141, 0.39702, 0.40474, 0.41773, 0.38577, 0.38974, 0.41551, 0.40019, 0.40592, 0.43536, 0.40853, 0.41867, 0.12931, 0.12846, 0.16863, 0.29955], "coef": [0.4164, -0.0569, -0.0683, 0.0179, 0.5266, -0.0944, 0.7657, -0.1808, 0.2362, -0.265, 0.0084, 0.3301, -0.325, -0.1664, -0.2377, -0.0031, 0.0275, 0.2331, -0.3131, -0.0633, 1.3494, 0.2151, 0.8949, -0.593, -0.6928, -0.3558, 0.1171, -0.2954, -0.0699, -0.272, 0.1273, 0.1695, 0.2153, -0.0655, -0.1686, -0.2254, -0.0762, 0.0887, -0.1067, 0.0254, 0.1477, 0.0262, -0.0616, 0.0003, 0.2134], "b": 0.6464, "T": 0.15}


class Guardian:
    UMBRAL = 0.5   # el corte lo fija el organizador; reescalamos en score()

    # ---------- utilidades ----------
    def _llm(self, o):
        """(veredicto, confianza, fuente) sin romper con None."""
        l = o.get("llm") or {}
        c = l.get("confianza")
        return l.get("veredicto"), (c if c is not None else 0.0), l.get("fuente")

    def _fuentes_citadas(self, caso):
        idx = set()
        for o in caso["senales"]["oraciones"]:
            i = self._llm(o)[2]
            if i is not None and 0 <= i < len(caso["fuentes"]):
                idx.add(i)
        return " ".join(caso["fuentes"][i] for i in sorted(idx)) or " ".join(caso["fuentes"])

    # ---------- detectores ----------
    def distorsion(self, caso):
        """Cifras de la respuesta que no aparecen en las fuentes que cita el LLM."""
        nr = _nums(caso["respuesta"])
        if not nr:
            return 0.0
        faltan_cit = nr - _nums(self._fuentes_citadas(caso))
        faltan_todas = nr - _nums(" ".join(caso["fuentes"]))
        # falta en TODAS = muy sospechoso (salvo aritmética); solo en la citada = atribución cruzada
        return min(1.0, 0.45 * len(faltan_todas) + 0.25 * len(faltan_cit - faltan_todas))

    def no_respaldado(self, caso):
        """La peor oración manda; respaldo_todas solo se fía a medias (el NLI se diluye)."""
        peor = 1.0
        for o in caso["senales"]["oraciones"]:
            v, c, idx = self._llm(o)
            soporte_llm = c if v == "respaldada" else 1 - c
            mejor_fuente = max(o["respaldo_por_fuente"], default=0.0)
            frs = [f for f in caso["senales"]["fragmentos"] if f["oracion"] == caso["senales"]["oraciones"].index(o)]
            peor_frag = min((max(f["respaldo_por_fuente"] + [f["respaldo_todas"]]) for f in frs), default=mejor_fuente)
            s = 0.35 * soporte_llm + 0.25 * mejor_fuente + 0.25 * peor_frag + 0.15 * o["respaldo_todas"]
            peor = min(peor, s)
        # palabras de contenido de la respuesta sin rastro en las fuentes
        st = {t[:5] for t in _toks(" ".join(caso["fuentes"]))}
        rt = [t for t in _toks(caso["respuesta"]) if t not in STOP and not t.isdigit()]
        nov = sum(1 for t in rt if t[:5] not in st) / max(1, len(rt))
        return min(1.0, (1 - peor) + 0.5 * nov)

    def contradiccion(self, caso):
        """Solo señales de contradicción (antes se mezclaba el soporte, que es lo contrario)."""
        peor = 0.0
        for o in caso["senales"]["oraciones"]:
            v, c, idx = self._llm(o)
            llm_c = c if v == "contradicha" else 0.0
            cf = o["contradiccion_por_fuente"][idx] if idx is not None and idx < len(o["contradiccion_por_fuente"]) else max(o["contradiccion_por_fuente"], default=0.0)
            peor = max(peor, 0.6 * llm_c + 0.25 * cf + 0.15 * o["contradiccion_todas"])
        return peor

    def sobregeneralizacion(self, caso):
        """Cuantificadores que amplían alcance o matices que se pierden (era un stub que fallaba)."""
        r, f = set(_toks(caso["respuesta"])), set(_toks(self._fuentes_citadas(caso)))
        sube = len((r & QUANT_UP) - f)
        pierde = len((f & QUANT_DOWN) - r)
        return min(1.0, 0.6 * sube + 0.3 * pierde)

    # ---------- características para el modelo ----------
    def _features(self, caso):
        S = caso["senales"]; ors = S["oraciones"]; frs = S["fragmentos"]
        src = " ".join(caso["fuentes"]); resp = caso["respuesta"]
        f = {}

        def agg(p, vals):
            vals = vals or [0.0]
            f[p + "_min"] = min(vals); f[p + "_max"] = max(vals); f[p + "_mean"] = sum(vals) / len(vals)

        llm = [o.get("llm") or {} for o in ors]
        conf = lambda l: l.get("confianza") or 0
        f["llm_contra"] = max([conf(l) for l in llm if l.get("veredicto") == "contradicha"] or [0])
        f["llm_nores"] = max([conf(l) for l in llm if l.get("veredicto") == "no_respaldada"] or [0])
        f["llm_res_minconf"] = min([conf(l) for l in llm if l.get("veredicto") == "respaldada"] or [1])
        f["llm_none"] = sum(1 for l in llm if l.get("veredicto") is None)
        agg("or_resp_todas", [o["respaldo_todas"] for o in ors])
        agg("or_contra_todas", [o["contradiccion_todas"] for o in ors])
        agg("or_best_src", [max(o["respaldo_por_fuente"] or [0]) for o in ors])
        agg("or_best_contra", [max(o["contradiccion_por_fuente"] or [0]) for o in ors])
        agg("fr_resp_todas", [o["respaldo_todas"] for o in frs])
        agg("fr_contra_todas", [o["contradiccion_todas"] for o in frs])
        agg("fr_best_src", [max(o["respaldo_por_fuente"] or [0]) for o in frs])
        agg("fr_best_contra", [max(o["contradiccion_por_fuente"] or [0]) for o in frs])
        cit = []
        for o in ors:
            i = self._llm(o)[2]
            if i is not None and i < len(o["respaldo_por_fuente"]):
                cit.append(o["respaldo_por_fuente"][i])
        agg("cit_resp", cit)
        agg("or_sim", [max(o["similitud_por_fuente"] or [0]) for o in ors])

        st = {t[:5] for t in _toks(src)}
        rt = [t for t in _toks(resp) if t not in STOP and not t.isdigit()]
        f["nov_frac"] = sum(1 for t in rt if t[:5] not in st) / max(1, len(rt))
        csrc = self._fuentes_citadas(caso)
        cst = {t[:5] for t in _toks(csrc)}
        f["novc_frac"] = sum(1 for t in rt if t[:5] not in cst) / max(1, len(rt))

        nr = _nums(resp)
        f["num_miss"] = len(nr - _nums(src))
        f["num_miss_cit"] = len(nr - _nums(csrc))
        f["num_all_in_cit"] = 1.0 if nr and not (nr - _nums(csrc)) else 0.0
        rtk, ctk = set(_toks(resp)), set(_toks(csrc))
        f["q_down_lost"] = len((ctk & FQ_DOWN) - rtk)
        caps = {_norm(w) for w in re.findall(r"(?<!^)(?<![.¿¡] )\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+", resp)}
        f["cap_new"] = len([w for w in caps if w not in set(_toks(src))])

        f["det_dist"] = self.distorsion(caso)
        f["det_contra"] = self.contradiccion(caso)
        f["det_nores"] = self.no_respaldado(caso)
        f["det_sobre"] = self.sobregeneralizacion(caso)
        return f

    # ---------- combinación ----------
    def score(self, caso):
        f = self._features(caso)
        z = MODELO["b"]
        for k, mu, sd, w in zip(MODELO["keys"], MODELO["mu"], MODELO["sd"], MODELO["coef"]):
            z += w * (f[k] - mu) / sd
        z -= math.log(MODELO["T"] / (1 - MODELO["T"])) + 0.0   # el corte óptimo T pasa a 0.5
        z = max(-30.0, min(30.0, z))
        return 1 / (1 + math.exp(-z))
