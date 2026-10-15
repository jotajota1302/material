import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class Guardian:
    def __init__(self):
        """
        Inicializa el modelo de detección de alucinaciones sin pandas.
        Carga train.jsonl, extrae características ricas de todas las oraciones,
        fragmentos y fuentes, y entrena un clasificador calibrado.
        """
        print("Initializing Guardian and training ML model (pandas-free, multi-sentence)...")

        casos = []
        with open("train.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    casos.append(json.loads(line))

        X = [self._extraer_features(c) for c in casos]
        y = [c["etiqueta"] for c in casos]

        X_mat = np.array(X, dtype=np.float64)
        y_vec = np.array(y, dtype=np.int64)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_mat)

        # Usamos class_weight ponderado porque dejar escapar una alucinación (FN)
        # penaliza 4 veces más que una falsa alarma (FP) en la métrica oficial.
        # Dejamos n_jobs=1 para evitar la sobrecarga de hilos en predicciones 1 a 1.
        self.clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            class_weight={0: 1.0, 1: 4.0},
            random_state=42
        )
        self.clf.fit(X_scaled, y_vec)
        print("Model trained and ready!")

    def _extraer_features(self, caso):
        """
        Extrae un vector denso de características numéricas a partir de:
        - Todas las oraciones de la respuesta (peor caso, promedios, discrepancias).
        - Todos los fragmentos (subfrases más sensibles a distorsiones numéricas y fechas).
        - Veredictos y confianza del LLM juez (contradicha, no_respaldada, respaldada).
        - Conflictos entre la opinión del LLM juez y el modelo NLI.
        - Conteos estructurales (nº oraciones, nº fragmentos, nº fuentes).
        """
        senales = caso.get("senales") or {}
        oraciones = senales.get("oraciones") or []
        fragmentos = senales.get("fragmentos") or []
        fuentes = caso.get("fuentes") or []

        n_oraciones = len(oraciones)
        n_fragmentos = len(fragmentos)
        n_fuentes = len(fuentes)

        # 1. Señales agregadas de todas las oraciones
        respaldo_todas = []
        contradiccion_todas = []
        max_respaldo_fuente = []
        max_contradiccion_fuente = []
        max_similitud_fuente = []

        cnt_contradicha = 0
        cnt_no_respaldada = 0
        cnt_respaldada = 0
        cnt_llm_null = 0
        min_conf_respaldada = 1.0
        min_conf_all = 1.0
        llm_nli_conflict = 0

        for o in oraciones:
            rt = float(o.get("respaldo_todas") if o.get("respaldo_todas") is not None else 0.0)
            ct = float(o.get("contradiccion_todas") if o.get("contradiccion_todas") is not None else 0.0)
            respaldo_todas.append(rt)
            contradiccion_todas.append(ct)

            rpf = o.get("respaldo_por_fuente") or [0.0]
            cpf = o.get("contradiccion_por_fuente") or [0.0]
            spf = o.get("similitud_por_fuente") or [0.0]
            max_respaldo_fuente.append(max(rpf))
            max_contradiccion_fuente.append(max(cpf))
            max_similitud_fuente.append(max(spf))

            llm = o.get("llm") or {}
            veredicto = llm.get("veredicto")
            conf = float(llm.get("confianza") if llm.get("confianza") is not None else 0.0)
            min_conf_all = min(min_conf_all, conf)

            if veredicto == "contradicha":
                cnt_contradicha += 1
            elif veredicto == "no_respaldada":
                cnt_no_respaldada += 1
            elif veredicto == "respaldada":
                cnt_respaldada += 1
                min_conf_respaldada = min(min_conf_respaldada, conf)
                # Detección de falsa confianza: LLM dice respaldada pero NLI detecta contradicción o bajo respaldo
                if rt < 0.25 or ct > 0.4:
                    llm_nli_conflict += 1
            else:
                cnt_llm_null += 1

        # 2. Señales agregadas de fragmentos (subfrases)
        frag_rt = []
        frag_ct = []
        frag_max_rpf = []
        frag_max_cpf = []
        for f in fragmentos:
            rt = float(f.get("respaldo_todas") if f.get("respaldo_todas") is not None else 0.0)
            ct = float(f.get("contradiccion_todas") if f.get("contradiccion_todas") is not None else 0.0)
            frag_rt.append(rt)
            frag_ct.append(ct)
            rpf = f.get("respaldo_por_fuente") or [0.0]
            cpf = f.get("contradiccion_por_fuente") or [0.0]
            frag_max_rpf.append(max(rpf))
            frag_max_cpf.append(max(cpf))

        prop_respaldada = cnt_respaldada / max(n_oraciones, 1)

        # 3. Vector de características final
        features = [
            # Conteos estructurales
            float(n_oraciones),
            float(n_fragmentos),
            float(n_fuentes),
            float(n_fragmentos / max(n_oraciones, 1)),

            # Oraciones: Respaldo y Contradicción global (peor caso y promedio)
            min(respaldo_todas, default=0.0),
            float(np.mean(respaldo_todas)) if respaldo_todas else 0.0,
            max(contradiccion_todas, default=0.0),
            float(np.mean(contradiccion_todas)) if contradiccion_todas else 0.0,

            # Oraciones: Soporte individual por fuente
            min(max_respaldo_fuente, default=0.0),
            float(np.mean(max_respaldo_fuente)) if max_respaldo_fuente else 0.0,
            max(max_contradiccion_fuente, default=0.0),
            float(np.mean(max_contradiccion_fuente)) if max_contradiccion_fuente else 0.0,
            min(max_similitud_fuente, default=0.0),

            # LLM Juez
            float(cnt_contradicha),
            float(cnt_no_respaldada),
            float(cnt_respaldada),
            float(cnt_llm_null),
            1.0 if cnt_contradicha > 0 else 0.0,
            1.0 if cnt_no_respaldada > 0 else 0.0,
            1.0 if (cnt_respaldada == n_oraciones and n_oraciones > 0) else 0.0,
            float(prop_respaldada),
            float(min_conf_respaldada),
            float(min_conf_all),
            float(llm_nli_conflict),

            # Fragmentos (más sensibles a detalles y números alterados)
            min(frag_rt, default=0.0),
            float(np.mean(frag_rt)) if frag_rt else 0.0,
            max(frag_ct, default=0.0),
            float(np.mean(frag_ct)) if frag_ct else 0.0,
            min(frag_max_rpf, default=0.0),
            max(frag_max_cpf, default=0.0),
        ]
        return features

    def score(self, caso):
        """
        Evalúa un caso individual. Devuelve una probabilidad [0.0, 1.0]
        de que la respuesta alucine (1.0 = alucinación, 0.0 = fiel).
        """
        feat = self._extraer_features(caso)
        x = np.array([feat], dtype=np.float64)
        x_scaled = self.scaler.transform(x)

        # Devolvemos la probabilidad de alucinación (clase 1)
        prob_alucinacion = self.clf.predict_proba(x_scaled)[0, 1]
        return float(prob_alucinacion)
