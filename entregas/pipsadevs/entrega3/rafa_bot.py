import json
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class Guardian:
    def __init__(self):

        print("Initializing Guardian and training ML model (pandas-free)...")

        preguntas = []
        textos_oracion = []
        numericos = []
        y = []

        # 1. Cargar datos con la librería estándar json
        with open("train.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                caso = json.loads(line)
                p, t, num = self._extraer_features(caso)
                preguntas.append(p)
                textos_oracion.append(t)
                numericos.append(num)
                y.append(caso["etiqueta"])

        # 2. Inicializar y entrenar transformadores
        self.vec_pregunta = TfidfVectorizer()
        self.vec_oracion = TfidfVectorizer()
        self.scaler = StandardScaler()

        X_p = self.vec_pregunta.fit_transform(preguntas)
        X_t = self.vec_oracion.fit_transform(textos_oracion)
        X_num = self.scaler.fit_transform(np.array(numericos, dtype=np.float64))

        # 3. Concatenar matrices (texto sparse + numéricas escaladas)
        X = sp.hstack([X_p, X_t, X_num])

        # 4. Entrenar el clasificador
        self.clf = RandomForestClassifier(random_state=42)
        self.clf.fit(X, y)
        print("Model trained and ready!")

    def _extraer_features(self, caso):

        pregunta = caso.get("pregunta") or ""
        oraciones = caso.get("senales", {}).get("oraciones", [])

        if oraciones:
            o0 = oraciones[0]
            texto_oracion = o0.get("texto") or ""

            # Garantizar float y valor fallback si es None
            respaldo = float(o0.get("respaldo_todas") if o0.get("respaldo_todas") is not None else 0.0)
            contradiccion = float(o0.get("contradiccion_todas") if o0.get("contradiccion_todas") is not None else 0.0)

            llm_info = o0.get("llm") or {}
            conf = llm_info.get("confianza")
            llm_confianza = float(conf) if conf is not None else 0.0

            v_map = {"respaldada": 0.0, "contradicha": 1.0, "no_respaldada": 2.0}
            llm_veredicto = v_map.get(llm_info.get("veredicto"), -1.0)
        else:
            texto_oracion = ""
            respaldo = 0.0
            contradiccion = 0.0
            llm_confianza = 0.0
            llm_veredicto = -1.0

        return pregunta, texto_oracion, [respaldo, contradiccion, llm_confianza, llm_veredicto]

    def score(self, caso):
        """
        Se ejecuta por cada caso durante evaluación / inferencia.
        """
        p, t, num = self._extraer_features(caso)

        x_p = self.vec_pregunta.transform([p])
        x_t = self.vec_oracion.transform([t])
        x_num = self.scaler.transform(np.array([num], dtype=np.float64))

        x = sp.hstack([x_p, x_t, x_num])

        # predict_proba devuelve [prob_fiel, prob_alucinacion]
        # Devolvemos la probabilidad de clase 1 (alucinación)
        prob_alucinacion = self.clf.predict_proba(x)[0, 1]
        return float(prob_alucinacion)
