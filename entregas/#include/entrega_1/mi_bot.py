class Guardian:
    def score(self, caso):
        senales = caso.get("senales", {})
        oraciones = senales.get("oraciones", [])
        fragmentos = senales.get("fragmentos", [])
        riesgo_maximo = 0.0
        for oracion in oraciones:
            llm = oracion.get("llm") or {}
            veredicto = llm.get("veredicto")
            confianza = float(llm.get("confianza") or 0.0)

            if veredicto == "contradicha":
                riesgo_maximo = max(riesgo_maximo, 0.92 * confianza)
            elif veredicto == "no_respaldada":
                riesgo_maximo = max(riesgo_maximo, 0.75 * confianza)
        for fragmento in fragmentos:
            respaldo = float(fragmento.get("respaldo_todas") or 0.0)
            riesgo_maximo = max(riesgo_maximo, 0.45 * (1.0 - respaldo))
        return min(1.0, max(0.0, riesgo_maximo / 0.162))
