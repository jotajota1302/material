"""Guardian coste-sensible (FN pesa 4x): combina NLI + LLM juez + chequeos lexicos.

Solo Python estandar. score() devuelve P(alucinacion) en [0,1], calibrada para
que el corte 0.5 sea el optimo con coste 4:1 (fugas vs falsas alarmas).
"""

CONDICIONALES = [
    "podr\u00eda", "podr\u00edan", "posible", "posibles", "siempre que",
    "salvo", "excepto", "solo ", "s\u00f3lo", "\u00fanicamente", "al menos",
    "en caso de", "si se aprueba", "si llueve", "si ",
    "condici\u00f3n", "condicionado",
]
CAUSALES = ["gracias a", "debido a", "porque", "ya que", "puesto que"]
UNIVERSALES = [
    "todos ", "todas ", "todo ", "siempre", "nunca",
    "ning\u00fan", "ninguna", "nadie",
]

# Umbral bruto elegido para coste 4:1 (agresivo: mejor recall aunque suba FP).
# Se reescala a 0.5 de forma monotona para cumplir el corte del organizador.
UMBRAL_BRUTO = 0.20


class Guardian:
    def _riesgo_oracion(self, oracion, fuentes):
        texto = (oracion.get("texto") or "").lower()
        resp = oracion.get("respaldo_por_fuente") or [0.0]
        base = 1.0 - max(resp, default=0.0)

        llm = oracion.get("llm") or {}
        veredicto = llm.get("veredicto")
        boost = 0.0
        if veredicto == "contradicha":
            boost += 0.55
        elif veredicto == "no_respaldada":
            boost += 0.35

        # La fuente citada trae un matiz condicional que la respuesta omite.
        f = llm.get("fuente")
        if isinstance(f, int) and 0 <= f < len(fuentes):
            fs = (fuentes[f] or "").lower()
            if any(k in fs for k in CONDICIONALES) and not any(k in texto for k in CONDICIONALES):
                boost += 0.50

        # Causas / generalizaciones anadidas por la respuesta.
        if any(k in texto for k in CAUSALES):
            boost += 0.30
        if any(k in texto for k in UNIVERSALES):
            boost += 0.30

        # Contradiccion fuerte del NLI (con peso bajo: tambien falla en fieles).
        contra = oracion.get("contradiccion_por_fuente") or [0.0]
        try:
            mc = max(list(contra) + [oracion.get("contradiccion_todas", 0.0)])
        except Exception:
            mc = 0.0
        if mc > 0.9 and base > 0.5:
            boost += 0.15

        return base + boost

    def _riesgo_fragmento(self, frag):
        resp = frag.get("respaldo_por_fuente") or [0.0]
        return 1.0 - max(resp, default=0.0)

    def _riesgo_bruto(self, caso):
        fuentes = caso.get("fuentes", [])
        ors = (caso.get("senales") or {}).get("oraciones", [])
        frags = (caso.get("senales") or {}).get("fragmentos", [])
        if not ors:
            return 0.0
        peor_or = max(self._riesgo_oracion(o, fuentes) for o in ors)
        peor_frag = max((self._riesgo_fragmento(f) for f in frags), default=0.0)
        # Los fragmentos son ruidosos: cuentan con peso menor, pero pillan
        # incisos sutiles dentro de frases por lo demas correctas.
        return max(peor_or, 0.8 * peor_frag)

    def _reescalar(self, s):
        t = UMBRAL_BRUTO
        s = min(1.5, max(0.0, float(s)))
        if s < t:
            return (s / t) * 0.5 if t > 0 else 0.0
        return 0.5 + ((s - t) / (1.5 - t)) * 0.5

    def score(self, caso):
        return min(1.0, max(0.0, self._reescalar(self._riesgo_bruto(caso))))
