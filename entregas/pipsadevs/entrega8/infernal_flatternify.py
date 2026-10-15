

from re import findall, match, search


def infernal_flatternify(caso):
    """
    Convierte un caso de El Guardián en un caso de Infernal Flatternify.
    """

    number_concordancia = False
    no_number_concordancia = False
    concordancia_simbolos = False
    concordancia_no_simbolos = False
    concordancia_mayusculas = False

    for fuente in caso["fuentes"]:
        if search(r"\d+", fuente) and search(r"\d+", caso["respuesta"]):
            number_concordancia = 1

        if not search(r"\d+", fuente) and not search(r"\d+", caso["respuesta"]):
            no_number_concordancia = 1

        if search(r"[%€\$]", fuente) and search(r"[%€\$]", caso["respuesta"]):
            concordancia_simbolos = 1

        if not search(r"[%€\$]", fuente) and not search(r"[%€\$]", caso["respuesta"]):
            concordancia_no_simbolos = 1

        if not search(r"[A-ZÁÉÍÓÚÑ]", fuente) and search(r"[A-ZÁÉÍÓÚÑ]", caso["respuesta"]):
            concordancia_mayusculas = 1

    # todos los numeros en las respuestas deben aparecer en fuentes
    all_fuentes_text = " ".join(caso["fuentes"])
    nums_fuentes = set(findall(r"\d+", all_fuentes_text))
    nums_respuesta = set(findall(r"\d+", caso["respuesta"]))
    sin_numeros_inventados = 1 if nums_respuesta.issubset(nums_fuentes) else 0

    # hacemos lo mismo con las divisas
    euros_fuentes = set(findall(r"\d+[\.,]?\d*\s*(?:euros|€)", all_fuentes_text.lower()))
    euros_respuesta = set(findall(r"\d+[\.,]?\d*\s*(?:euros|€)", caso["respuesta"].lower()))
    sin_euros_inventados = 1 if euros_respuesta.issubset(euros_fuentes) else 0

    # con los porcentajes
    porcentajes_fuentes = set(findall(r"\d+\s*%", all_fuentes_text))
    porcentajes_respuesta = set(findall(r"\d+\s*%", caso["respuesta"]))
    sin_porcentajes_inventados = 1 if porcentajes_respuesta.issubset(porcentajes_fuentes) else 0

    # con los años
    anos_fuentes = set(findall(r"\b(19|20)\d{2}\b", all_fuentes_text))
    anos_respuesta = set(findall(r"\b(19|20)\d{2}\b", caso["respuesta"]))
    sin_anos_inventados = 1 if anos_respuesta.issubset(anos_fuentes) else 0

    # Detectar palabras con mayúscula inicial (excluyendo la primera palabra de una frase si es necesario, o general)
    mayus_fuentes = set(findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b", all_fuentes_text))
    mayus_respuesta = set(findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b", caso["respuesta"]))
    sin_mayusculas_inventadas = 1 if mayus_respuesta.issubset(mayus_fuentes) else 0


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
        "tiene_interrogacion": True if search(".*\?",caso["pregunta"]) else False,
        "tiene_como": True if search("¿?C[oó]mo",caso["pregunta"]) else False,
        "tiene_que": True if search("¿?Qu[eé]",caso["pregunta"]) else False,
        "peor_respaldo": peor_respaldo,
        "mejor_fuente": mejor_fuente,
        "peor_contradiccion": peor_contradiccion,
        "mejor_contradiccion": mejor_contradiccion,
        "mejor_similitud": mejor_similitud,
        "peor_similitud": peor_similitud,
        "respaldo_todas": respaldo_todas,
        "contradiccion_todas": contradiccion_todas,
        #"EXPECTED": True if caso.get("etiqueta") == 1 else False, # alucina???
        
        #"fuentes": caso["fuentes"],
        #"respuesta": caso["respuesta"]
        #"senales": caso["senales"]

        "tiene_en": True if search(r"¿?En",caso["pregunta"]) else False,
        "tiene_quien": True if search(r"¿?Qui[eé]n",caso["pregunta"]) else False,
        "tiene_cuales": True if search(r"¿?Cu[aá]les",caso["pregunta"]) else False,


        
        "tiene_concordancia_numerica": number_concordancia,
        "tiene_concordancia_no_numerica" : no_number_concordancia,
        "tiene_concordancia_simbolos" : concordancia_simbolos,
        "tiene_concordancia_no_simbolos" : concordancia_no_simbolos,
        "contiene_mayusculas_fuente": concordancia_mayusculas,
        "sin_numeros_inventados" : sin_numeros_inventados,
        "sin_euros_inventados": sin_euros_inventados,
        "sin_porcentajes_inventados": sin_porcentajes_inventados,
        "sin_anos_inventados": sin_anos_inventados,
        "sin_mayusculas_inventadas": sin_mayusculas_inventadas,

        }
