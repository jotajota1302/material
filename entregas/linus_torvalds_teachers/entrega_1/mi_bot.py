import math


def fval(val, default=0.0):
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)


def flist(lista, default=0.0):
    if not lista:
        return [float(default)]
    limpia = [fval(x, default) for x in lista if x is not None]
    return limpia if limpia else [float(default)]


class Guardian:
    PESOS = [
        -0.714589,
        1.047916,
        -1.267332,
        -3.809663,
        -3.385839,
        0.257211,
        -0.171154,
        0.134662,
        -0.050681,
        -0.387112,
        -0.626785,
        0.980733,
        1.238163,
        -1.148639,
        0.107827,
        -0.980032,
        -0.916362,
        -0.088874,
        -0.434866,
        -0.637624,
    ]
    # Sesgo perfectamente calibrado para mapear el corte 0.273 exactamente a 0.500
    SESGO = 8.140002

    def extraer_features(self, caso):
        senales = caso.get("senales") or {}
        oraciones = senales.get("oraciones") or []
        fragmentos = senales.get("fragmentos") or []

        # 1. Señales detalladas del Juez LLM
        llm_contradicha = 0.0
        llm_no_respaldada = 0.0
        llm_alarma_count = 0.0
        llm_conf_respaldada_min = 1.0
        llm_score_min = 1.0
        llm_nulo = 0.0

        for o in oraciones:
            llm = o.get("llm") or {}
            veredicto = llm.get("veredicto") or ""
            conf_raw = llm.get("confianza")
            if conf_raw is None and veredicto == "":
                llm_nulo = 1.0
            conf = fval(conf_raw, 0.5)

            if veredicto == "contradicha":
                llm_contradicha = max(llm_contradicha, conf)
                llm_alarma_count += 1.0
                llm_score_min = min(llm_score_min, -conf)
            elif veredicto == "no_respaldada":
                llm_no_respaldada = max(llm_no_respaldada, conf)
                llm_alarma_count += 1.0
                llm_score_min = min(llm_score_min, 0.0)
            elif veredicto == "respaldada":
                llm_conf_respaldada_min = min(llm_conf_respaldada_min, conf)
                llm_score_min = min(llm_score_min, conf)
            else:
                llm_score_min = min(llm_score_min, 0.2)

        n_orac = max(len(oraciones), 1)
        ratio_alarma_llm = llm_alarma_count / n_orac

        # 2. Oraciones NLI + Similitud
        contra_todas_o = [
            fval(o.get("contradiccion_todas"), 0.0) for o in oraciones
        ]
        contra_fuente_o = [
            max(flist(o.get("contradiccion_por_fuente"), 0.0))
            for o in oraciones
        ]
        resp_todas_o = [
            fval(o.get("respaldo_todas"), 1.0) for o in oraciones
        ]
        resp_fuente_o = [
            max(flist(o.get("respaldo_por_fuente"), 0.0)) for o in oraciones
        ]
        sim_fuente_o = [
            max(flist(o.get("similitud_por_fuente"), 0.0)) for o in oraciones
        ]

        max_contra_todas = max(contra_todas_o, default=0.0)
        max_contra_fuente = max(contra_fuente_o, default=0.0)
        min_resp_todas = min(resp_todas_o, default=1.0)
        min_resp_fuente = min(resp_fuente_o, default=1.0)
        mean_resp_fuente = (
            sum(resp_fuente_o) / len(resp_fuente_o) if resp_fuente_o else 1.0
        )
        min_sim_fuente = min(sim_fuente_o, default=1.0)
        mean_sim_fuente = (
            sum(sim_fuente_o) / len(sim_fuente_o) if sim_fuente_o else 1.0
        )

        # 3. Fragmentos NLI + Similitud
        contra_todas_f = [
            fval(f.get("contradiccion_todas"), 0.0) for f in fragmentos
        ]
        contra_fuente_f = [
            max(flist(f.get("contradiccion_por_fuente"), 0.0))
            for f in fragmentos
        ]
        resp_todas_f = [
            fval(f.get("respaldo_todas"), 1.0) for f in fragmentos
        ]
        resp_fuente_f = [
            max(flist(f.get("respaldo_por_fuente"), 0.0)) for f in fragmentos
        ]
        sim_fuente_f = [
            max(flist(f.get("similitud_por_fuente"), 0.0)) for f in fragmentos
        ]

        max_contra_todas_f = max(contra_todas_f, default=0.0)
        max_contra_fuente_f = max(contra_fuente_f, default=0.0)
        min_resp_todas_f = min(resp_todas_f, default=1.0)
        min_resp_fuente_f = min(resp_fuente_f, default=1.0)
        min_sim_fuente_f = min(sim_fuente_f, default=1.0)

        # 4. Interacciones
        desacuerdo_juez = llm_conf_respaldada_min * (1.0 - min_resp_fuente)
        caida_respaldo_frag = max(0.0, min_resp_fuente - min_resp_fuente_f)

        return [
            llm_contradicha,
            llm_no_respaldada,
            ratio_alarma_llm,
            llm_conf_respaldada_min,
            llm_score_min,
            llm_nulo,
            max_contra_todas,
            max_contra_fuente,
            min_resp_todas,
            min_resp_fuente,
            mean_resp_fuente,
            min_sim_fuente,
            mean_sim_fuente,
            max_contra_todas_f,
            max_contra_fuente_f,
            min_resp_todas_f,
            min_resp_fuente_f,
            min_sim_fuente_f,
            desacuerdo_juez,
            caida_respaldo_frag,
        ]

    def score(self, caso) -> float:
        try:
            x = self.extraer_features(caso)
            z = sum(w * xi for w, xi in zip(self.PESOS, x)) + self.SESGO
            p = 1.0 / (1.0 + math.exp(-z))
            return float(min(max(p, 0.0), 1.0))
        except Exception:
            return 0.55