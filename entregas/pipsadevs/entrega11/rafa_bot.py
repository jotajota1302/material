import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from infernal_flatternify import infernal_flatternify


class Guardian:
    # Umbral óptimo sobre las probabilidades out-of-fold (validación cruzada
    # agrupada por escenario). El score final se recoloca con a=(1-t)/t para que
    # el corte 0.5 del evaluador caiga justo en t, donde la métrica oficial
    # (un falso negativo pesa 4x) se maximiza. Medido con comparar_modelos.py
    # (--calibrar gradient_boosting):
    #   casos H (proxy del test) -> t* ≈ 0.124 · métrica 0.126 · a ≈ 7.05
    #   conjunto completo        -> t* ≈ 0.239 · métrica 0.410 · a ≈ 3.18
    # Elegimos t=0.15: agresivo (los FN pesan 4x) pero con margen para el giro,
    # que cambia de dominio. El valor teórico con probabilidades calibradas y
    # coste FN=4·FP es p*=0.2 (a=4).
    UMBRAL_OOF = 0.15
    PESO_H = 3.0

    def __init__(self):
        """
        Inicializa el modelo de detección de alucinaciones sin pandas.
        - Utiliza proporciones e intensidades normalizadas en lugar de recuentos absolutos
          para generalizar a respuestas largas y al giro (3-4 oraciones).
        - Pondera los casos de tipo H con sample_weight=3.0 por ser los más representativos del test.
        - El clasificador (GradientBoosting) es el ganador de comparar ~25 modelos
          de scikit-learn con CV agrupada: mejor AUROC en los casos H (0.834) y
          mejor métrica en el proxy del test.
        """
        print("Initializing Guardian and training ML model (GradientBoosting)...")

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
        # porque son los que más se parecen a la distribución real del test y del giro
        sample_weights = np.array(
            [self.PESO_H if c.get("id", "").startswith("H") else 1.0 for c in casos],
            dtype=np.float64,
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_mat)

        # Modelo ganador de la comparación (comparar_modelos.py): mejor AUROC en
        # los casos H (proxy del test) y mejor métrica calibrada en ese subconjunto.
        self.clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self.clf.fit(X_scaled, y_vec, sample_weight=sample_weights)

        # Factor de calibración: recoloca el corte óptimo en 0.5
        self.a = (1.0 - self.UMBRAL_OOF) / self.UMBRAL_OOF
        print("Model trained and ready!")

    def _extraer_features(self, caso):
        """
        Extrae características normalizadas:
        - Todas las variables de recuento se convierten en PROPORCIONES divididas por el nº de oraciones.
        - De este modo, si en el giro o test hay 3 o 4 oraciones por respuesta, los valores
          permanecen dentro del rango aprendido y no se salen de distribución.
        - Señales de peor caso (min respaldo, max contradicción) independientes de la longitud.
        """

        # integrate infernal_flatternify.py features here
        infernal_features = infernal_flatternify(caso)

        senales = caso.get("senales") or {}
        oraciones = senales.get("oraciones") or []
        fragmentos = senales.get("fragmentos") or []

        n_oraciones = max(len(oraciones), 1)
        n_fragmentos = len(fragmentos)

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
                # Detección de conflicto LLM-NLI
                if rt < 0.25 or ct > 0.4:
                    llm_nli_conflict += 1
            else:
                cnt_llm_null += 1

        # 2. Señales agregadas de fragmentos
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

        # 3. Vector de características con PROPORCIONES (invariante a escala de oraciones)
        features = [
            # Ratio de fragmentos por oración (en vez de conteo absoluto de fragmentos)
            float(n_fragmentos / n_oraciones),

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

            # LLM Juez: PROPORCIONES (divididas entre n_oraciones)
            float(cnt_contradicha / n_oraciones),
            float(cnt_no_respaldada / n_oraciones),
            float(cnt_respaldada / n_oraciones),
            float(cnt_llm_null / n_oraciones),
            float(llm_nli_conflict / n_oraciones),

            # Flags binarios (0 o 1, completamente invariantes al número de oraciones)
            1.0 if cnt_contradicha > 0 else 0.0,
            1.0 if cnt_no_respaldada > 0 else 0.0,
            1.0 if (cnt_respaldada == len(oraciones) and len(oraciones) > 0) else 0.0,
            1.0 if cnt_llm_null > 0 else 0.0,
            1.0 if llm_nli_conflict > 0 else 0.0,

            # Confianzas del LLM
            float(min_conf_respaldada),
            float(min_conf_all),

            # Fragmentos (más sensibles a detalles y números alterados)
            min(frag_rt, default=0.0),
            float(np.mean(frag_rt)) if frag_rt else 0.0,
            max(frag_ct, default=0.0),
            float(np.mean(frag_ct)) if frag_ct else 0.0,
            min(frag_max_rpf, default=0.0),
            max(frag_max_cpf, default=0.0),
        ]

        # Añadir las features de infernal_flatternify.py programáticamente
        features.extend(infernal_features.values())

        return features

    def score(self, caso):
        """
        Evalúa un caso individual. Devuelve un score continuo calibrado en [0.0, 1.0].
        El organizador corta en score >= 0.5 = ALUCINACIÓN.
        """
        feat = self._extraer_features(caso)
        x = np.array([feat], dtype=np.float64)
        x_scaled = self.scaler.transform(x)

        # Probabilidad de alucinación del ensemble
        prob = self.clf.predict_proba(x_scaled)[0, 1]

        # Recoloca el corte: score=0.5 justo en el umbral óptimo t (a=(1-t)/t).
        # Así el corte 0.5 del organizador coincide con el óptimo de la métrica,
        # que penaliza un falso negativo 4x más que una falsa alarma.
        score_adj = (self.a * prob) / (self.a * prob + (1.0 - prob) + 1e-9)
        return float(np.clip(score_adj, 0.0, 1.0))
