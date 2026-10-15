

from re import match



#
# example: "senales": {"oraciones": [{"texto": "Están exentos del turno mensual de trabajo los socios de más de 70 años y las personas con discapacidad", "respaldo_por_fuente": [0.0, 0.0, 0.294, 0.0, 0.001, 0.0, 0.001, 0.002, 0.0], "contradiccion_por_fuente": [0.0, 0.001, 0.002, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001], "similitud_por_fuente": [0.182, 0.447, 0.934, 0.24, 0.275, 0.164, 0.046, 0.067, 0.123], "respaldo_todas": 0.081, "contradiccion_todas": 0.888, "llm": {"veredicto": "respaldada", "fuente": 2, "confianza": 0.98}}], "fragmentos": [{"oracion": 0, "texto": "Están exentos del turno mensual de trabajo los socios de más de 70 años", "respaldo_por_fuente": [0.0, 0.0, 0.288, 0.0, 0.001, 0.0, 0.001, 0.001, 0.0], "contradiccion_por_fuente": [0.0, 0.002, 0.004, 0.0, 0.001, 0.001, 0.001, 0.002, 0.0], "similitud_por_fuente": [0.223, 0.552, 0.803, 0.238, 0.341, 0.198, 0.059, 0.075, 0.179], "respaldo_todas": 0.639, "contradiccion_todas": 0.352}, {"oracion": 0, "texto": "las personas con discapacidad", "respaldo_por_fuente": [0.003, 0.009, 0.593, 0.058, 0.003, 0.018, 0.006, 0.056, 0.045], "contradiccion_por_fuente": [0.001, 0.02, 0.015, 0.02, 0.006, 0.011, 0.003, 0.034, 0.107], "similitud_por_fuente": [0.026, -0.006, 0.639, 0.011, -0.043, -0.003, -0.056, -0.065, -0.061], "respaldo_todas": 0.03, "contradiccion_todas": 0.008}]}}
#

def infernal_flatternify(caso):
    """
    Convierte un caso de El Guardián en un caso de Infernal Flatternify.
    """
    peor_respaldo = 1.0
    mejor_fuente = 0.0
    peor_contradiccion = 1.0
    mejor_contradiccion = 0.0
    mejor_similitud = float("-inf")
    peor_similitud = float("inf")
    respaldo_todas = 0.0
    contradiccion_todas = 0.0

    senales = caso.get("senales", {})
    for oracion in senales.get("oraciones", []):
        respaldos = oracion.get("respaldo_por_fuente", [])
        contradicciones = oracion.get("contradiccion_por_fuente", [])
        similitudes = oracion.get("similitud_por_fuente", [])

        mejor_fuente = max(mejor_fuente, max(respaldos, default=0.0))
        peor_respaldo = min(peor_respaldo, min(respaldos, default=1.0))
        mejor_contradiccion = max(mejor_contradiccion, max(contradicciones, default=0.0))
        peor_contradiccion = min(peor_contradiccion, min(contradicciones, default=1.0))
        if similitudes:
            mejor_similitud = max(mejor_similitud, max(similitudes))
            peor_similitud = min(peor_similitud, min(similitudes))
        respaldo_todas = max(respaldo_todas, oracion.get("respaldo_todas", 0.0))
        contradiccion_todas = max(contradiccion_todas, oracion.get("contradiccion_todas", 0.0))

    if mejor_similitud == float("-inf"):
        mejor_similitud = 0.0
    if peor_similitud == float("inf"):
        peor_similitud = 0.0

    return {
        "tiene_interrogacion": 1 if match(".*\?",caso["pregunta"]) else 0,
        "tiene_como": 1 if match("¿?C[oó]mo",caso["pregunta"]) else 0,
        "tiene_que": 1 if match("¿?Qu[eé]",caso["pregunta"]) else 0,
        "pregunta": caso["pregunta"],
        "peor_respaldo": peor_respaldo,
        "mejor_fuente": mejor_fuente,
        "peor_contradiccion": peor_contradiccion,
        "mejor_contradiccion": mejor_contradiccion,
        "mejor_similitud": mejor_similitud,
        "peor_similitud": peor_similitud,
        "respaldo_todas": respaldo_todas,
        "contradiccion_todas": contradiccion_todas,
        #"EXPECTED": True if caso["etiqueta"] == 1 else False # alucina???
        #"fuentes": caso["fuentes"],
        #"respuesta": caso["respuesta"],
        #"senales": caso["senales"]
    }
