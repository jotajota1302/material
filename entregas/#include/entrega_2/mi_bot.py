"""Detector de alucinaciones independiente del idioma.

Solo combina las señales semánticas precalculadas del reto. No inspecciona
palabras, sinónimos ni expresiones de ningún idioma concreto.
"""


class Guardian:
    def __init__(self):
        import json, os
        import numpy as np
        self.np = np
        path = os.path.join(os.path.dirname(__file__), "train.jsonl")
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        X = np.array([self._features(x) for x in rows], float)
        y = np.array([x["etiqueta"] for x in rows], float)
        self.mean = X.mean(0); self.std = X.std(0) + 1e-6
        X = (X-self.mean)/self.std
        self.w = np.zeros(X.shape[1]+1)
        for _ in range(800):
            z = self.w[0] + X @ self.w[1:]
            p = 1/(1+np.exp(-np.clip(z,-30,30)))
            g = p-y
            self.w[0] -= .12*g.mean()
            self.w[1:] -= .12*((X.T@g)/len(y)+.003*self.w[1:])

    @staticmethod
    def _features(caso):
        s=caso["senales"]; o=s["oraciones"]; f=s["fragmentos"]
        def v(xs,k,d): return [float(x.get(k) or 0) for x in xs] or [d]
        fr=v(f,"respaldo_todas",1); fc=v(f,"contradiccion_todas",0)
        ll=[x.get("llm") or {} for x in o]
        q=lambda name:max([x.get("confianza") or 0 for x in ll if x.get("veredicto")==name] or [0])
        return [1-min(fr),max(fc),q("no_respaldada"),q("contradicha"),q("respaldada"),len(o),len(f)]

    def score(self, caso):
        x=(self.np.array(self._features(caso))-self.mean)/self.std
        z=self.w[0]+x@self.w[1:]
        probabilidad = float(1 / (1 + self.np.exp(-max(-30, min(30, z)))))

        # Calibración monótona: el umbral H equivalente a 0.236 pasa a 0.5
        # sin cambiar el orden relativo de los casos.
        return probabilidad / (probabilidad + 0.180)
        """Devuelve el riesgo de alucinación, siempre entre 0 y 1."""
        senales = caso.get("senales", {})
        oraciones = senales.get("oraciones", [])
        fragmentos = senales.get("fragmentos", [])
        confianza_respaldo = []
        riesgo_maximo = 0.0

        # El juez LLM funciona bien en contradicciones explícitas, pero no se
        # usa como única fuente de decisión: puede aceptar detalles añadidos.
        for oracion in oraciones:
            llm = oracion.get("llm") or {}
            veredicto = llm.get("veredicto")
            confianza = float(llm.get("confianza") or 0.0)

            if veredicto == "contradicha":
                riesgo_maximo = max(riesgo_maximo, 0.92 * confianza)
            elif veredicto == "no_respaldada":
                riesgo_maximo = max(riesgo_maximo, 0.75 * confianza)
            confianza_respaldo.append(confianza if veredicto == "respaldada" else 0.0)

        # Los fragmentos evitan que una afirmación falsa se diluya dentro de
        # una oración con otros datos correctos.
        for fragmento in fragmentos:
            respaldo = float(fragmento.get("respaldo_todas") or 0.0)
            # El respaldo semántico conjunto es más estable entre idiomas que
            # una contradicción NLI aislada, que aquí añade ruido.
            indice = fragmento.get("oracion")
            confianza = confianza_respaldo[indice] if isinstance(indice, int) and 0 <= indice < len(confianza_respaldo) else 0.0
            factor = 0.35 if confianza >= 0.90 else 0.70 if confianza >= 0.70 else 1.0
            riesgo_maximo = max(riesgo_maximo, factor * 0.45 * (1.0 - respaldo))

        # Los escapes penalizan cuatro veces más que las falsas alarmas. La
        # calibración sobre los casos H coloca el umbral interno conservador
        # 0.011 en el corte oficial score >= 0.5.
        return min(1.0, max(0.0, riesgo_maximo / 0.022))
