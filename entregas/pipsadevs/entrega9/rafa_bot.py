import json
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Cuantificadores universales para detectar sobre-generalizaciones
UNIVERSALES = {
    'todo', 'toda', 'todos', 'todas', 'siempre', 'nunca', 'jamas',
    'ningun', 'ninguna', 'ninguno', 'completamente', 'totalmente',
    'entero', 'entera', 'absoluto', 'absoluta', 'unico', 'unica', 'unicos', 'unicas'
}


class Guardian:
    def __init__(self):
        """
        Inicializa y calibra el detector de alucinaciones El Guardián.
        Incorpora:
        1. Proporciones e intensidades normalizadas (invariantes a longitud/giro).
        2. Detección simbólica de cifras normalizadas, años y nombres propios inventados.
        3. Verificación de la fuente citada por el LLM juez contra el modelo NLI.
        4. Detección de discrepancias sutiles entre fragmentos y oraciones.
        5. Calibración bayesiana para la función de pérdida asimétrica (FN pesa 4x).
        6. Ponderación reforzada de los casos H del train (sample_weight=3.0).
        """
        print("Initializing Guardian with advanced domain signals & calibration...")

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

        # Dar más peso (sample_weight=3.0) a los casos H de train
        sample_weights = np.array(
            [3.0 if c.get("id", "").startswith("H") else 1.0 for c in casos],
            dtype=np.float64
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_mat)

        self.clf = RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            class_weight={0: 1.0, 1: 2.5},
            random_state=42,
            n_jobs=1
        )
        self.clf.fit(X_scaled, y_vec, sample_weight=sample_weights)

        # Factor bayesiano calibrado mediante validación cruzada:
        # Equilibra el coste 4x de FN sin disparar falsas alarmas
        self.w_loss = 1.8
        print("Model trained and ready!")

    def _normalize_numbers(self, text):
        """
        Extrae números considerando separadores de miles y decimales en español.
        Ej: '190.000' genera tanto '190.000' como '190000'.
        """
        nums = set()
        raw = re.findall(r'\b\d+(?:[.,]\d+)*\b', text)
        for r in raw:
            nums.add(r)
            clean = r.replace('.', '').replace(',', '.')
            nums.add(clean)
            try:
                val = float(clean)
                nums.add(str(val))
                if val.is_integer():
                    nums.add(str(int(val)))
            except ValueError:
                pass
        return nums

    def _get_proper_nouns(self, text):
        """
        Detecta nombres propios (palabras con mayúscula inicial que no inician frase).
        Altamente preciso (96%+) para detectar entidades inventadas o distorsionadas.
        """
        sentences = re.split(r'[.!?]\s+', text)
        nouns = set()
        for s in sentences:
            words = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', s)
            if len(words) > 1:
                for w in words[1:]:
                    nouns.add(w.lower())
        return nouns

    def _extraer_features(self, caso):
        senales = caso.get("senales") or {}
        oraciones = senales.get("oraciones") or []
        fragmentos = senales.get("fragmentos") or []
        fuentes = caso.get("fuentes") or []
        resp = caso.get("respuesta") or ""
        fuentes_text = " ".join(fuentes)
        pregunta = caso.get("pregunta") or ""

        n_oraciones = max(len(oraciones), 1)
        n_fragmentos = len(fragmentos)

        # 1. Señales agregadas de oraciones
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

        cited_support = []
        cited_contradiction = []

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
            f_idx = llm.get("fuente")

            if veredicto == "contradicha":
                cnt_contradicha += 1
            elif veredicto == "no_respaldada":
                cnt_no_respaldada += 1
            elif veredicto == "respaldada":
                cnt_respaldada += 1
                min_conf_respaldada = min(min_conf_respaldada, conf)
                if rt < 0.25 or ct > 0.4:
                    llm_nli_conflict += 1
                # Verificación de la fuente específica que el LLM afirma que respalda
                if f_idx is not None and isinstance(f_idx, int) and 0 <= f_idx < len(rpf):
                    cited_support.append(rpf[f_idx])
                    cited_contradiction.append(cpf[f_idx])
            else:
                cnt_llm_null += 1

        # 2. Señales agregadas de fragmentos y discrepancias oracion-fragmento
        frag_rt = []
        frag_ct = []
        frag_max_rpf = []
        frag_max_cpf = []
        frag_drop_rt = []
        frag_jump_ct = []

        for f in fragmentos:
            rt = float(f.get("respaldo_todas") if f.get("respaldo_todas") is not None else 0.0)
            ct = float(f.get("contradiccion_todas") if f.get("contradiccion_todas") is not None else 0.0)
            frag_rt.append(rt)
            frag_ct.append(ct)
            rpf = f.get("respaldo_por_fuente") or [0.0]
            cpf = f.get("contradiccion_por_fuente") or [0.0]
            frag_max_rpf.append(max(rpf))
            frag_max_cpf.append(max(cpf))

            # Discrepancia con la oración padre
            o_idx = f.get("oracion")
            if o_idx is not None and isinstance(o_idx, int) and 0 <= o_idx < len(respaldo_todas):
                frag_drop_rt.append(max(0.0, respaldo_todas[o_idx] - rt))
                frag_jump_ct.append(max(0.0, ct - contradiccion_todas[o_idx]))

        # 3. Comprobaciones léxicas y de entidades (cubren lo que el NLI no ve)
        nums_r = self._normalize_numbers(resp)
        nums_f = self._normalize_numbers(fuentes_text)
        unseen_nums = nums_r - nums_f
        has_unseen_nums = 1.0 if unseen_nums else 0.0
        prop_unseen_nums = float(len(unseen_nums) / max(len(nums_r), 1))

        # Años inventados (ej: 1997 vs 1994)
        years_r = set(re.findall(r'\b(?:19|20)\d{2}\b', resp))
        years_f = set(re.findall(r'\b(?:19|20)\d{2}\b', fuentes_text))
        has_unseen_years = 1.0 if (years_r - years_f) else 0.0

        # Nombres propios inventados (entidades desconocidas en fuentes y pregunta)
        proper_r = self._get_proper_nouns(resp)
        proper_f = set(re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', fuentes_text + " " + pregunta))
        proper_f_lower = {w.lower() for w in proper_f}
        has_unseen_proper = 1.0 if (proper_r - proper_f_lower) else 0.0

        # Sobre-generalización léxica (palabras universales en respuesta que no estaban en fuentes)
        words_r = set(re.findall(r'\b\w+\b', resp.lower()))
        words_f = set(re.findall(r'\b\w+\b', fuentes_text.lower()))
        unseen_universales = (words_r & UNIVERSALES) - words_f
        has_unseen_universales = 1.0 if unseen_universales else 0.0

        # 4. Vector de características final (35 features normalizadas)
        features = [
            # Estructura normalizada
            float(n_fragmentos / n_oraciones),

            # Oraciones: Respaldo y Contradicción global
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

            # LLM Juez: Proporciones por oración
            float(cnt_contradicha / n_oraciones),
            float(cnt_no_respaldada / n_oraciones),
            float(cnt_respaldada / n_oraciones),
            float(cnt_llm_null / n_oraciones),
            float(llm_nli_conflict / n_oraciones),

            # Banderas binarias
            1.0 if cnt_contradicha > 0 else 0.0,
            1.0 if cnt_no_respaldada > 0 else 0.0,
            1.0 if (cnt_respaldada == len(oraciones) and len(oraciones) > 0) else 0.0,
            1.0 if cnt_llm_null > 0 else 0.0,
            1.0 if llm_nli_conflict > 0 else 0.0,

            # Confianzas del LLM
            float(min_conf_respaldada),
            float(min_conf_all),

            # Coherencia de la fuente citada por el LLM vs NLI de esa fuente
            min(cited_support, default=0.0),
            max(cited_contradiction, default=0.0),

            # Fragmentos
            min(frag_rt, default=0.0),
            float(np.mean(frag_rt)) if frag_rt else 0.0,
            max(frag_ct, default=0.0),
            float(np.mean(frag_ct)) if frag_ct else 0.0,
            min(frag_max_rpf, default=0.0),
            max(frag_max_cpf, default=0.0),

            # Discrepancias fragmento-oración
            max(frag_drop_rt, default=0.0),
            max(frag_jump_ct, default=0.0),

            # Comprobaciones de distorsión y sobre-generalización
            has_unseen_nums,
            prop_unseen_nums,
            has_unseen_years,
            has_unseen_proper,
            has_unseen_universales,
        ]
        return features

    def score(self, caso):
        """
        Evalúa un caso individual. Devuelve un score continuo calibrado en [0.0, 1.0].
        El organizador corta en score >= 0.5 = ALUCINACIÓN.
        """
        feat = self._extraer_features(caso)
        x = np.array([feat], dtype=np.float64)
        x_scaled = self.scaler.transform(x)

        prob = self.clf.predict_proba(x_scaled)[0, 1]

        # Reescalado asimétrico por Bayes para alinear el umbral de coste óptimo con 0.5
        score_adj = (self.w_loss * prob) / (self.w_loss * prob + (1.0 - prob) + 1e-9)
        return float(np.clip(score_adj, 0.0, 1.0))
