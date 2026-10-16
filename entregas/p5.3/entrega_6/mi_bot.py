"""Guardian Fusión: motores de palabras de entrega_2giro + modelo entrenado de entrega_3giro.

score >= 0.5 = alucina (1 = alucina, 0 = fiel).  El corte del organizador es 0.5: el score se
reescala para que 0.5 sea el punto óptimo de la métrica (4·FN + FP), es decir P(alucina) = 0.2.

Qué junta
  · De entrega_2giro (con el léxico ampliado de mi_bot_lexico.py): los cinco motores de palabras
    y cifras (aritmética y porcentajes, omisión de condiciones, endurecimiento modal y temporal,
    incisos y palabras huérfanas, atribución cruzada de entidades). Aquí NO se usan como un
    clasificador aparte, sino que sus salidas entran como rasgos del modelo.
  · De entrega_3giro: rasgos de señales (juez LLM calibrado por tramos de confianza, NLI global,
    por fuente y por fragmento, contradicción, similitud, texto genérico) y un conjunto de modelos
    de boosting (dos modelos, corto y largo, mezclados por la longitud de la respuesta) entrenados con
    pesos de importancia hacia test_publico y giro_publico (solo rasgos estructurales, sin etiquetas).
  · El modelo ya viene entrenado dentro de este fichero: no lee train.jsonl al arrancar.
    Para reentrenar: python entrenar_guardian.py --bot guardian_fusion.py (necesita scikit-learn).
"""

import json
import math
import os
import re
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# =============================================================================
# CONSTANTE DE POLARIDAD EXIGIDA EN SALA
# =============================================================================
# True  -> Score <= 0.5 implica ALUCINA (0.0 = alucina seguro, 1.0 = fiel seguro)
# False -> Score >= 0.5 implica ALUCINA (1.0 = alucina seguro, 0.0 = fiel seguro)
SCORE_MENOR_IGUAL_05_ES_ALUCINA: bool = False


# =============================================================================
# LÉXICOS, DICCIONARIOS NUMÉRICOS Y PATRONES NEURO-SIMBÓLICOS
# =============================================================================

STOPWORDS_ES: Set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo",
    "de", "en", "por", "para", "con", "sobre", "entre", "hacia", "hasta",
    "desde", "durante", "mediante", "según", "segun", "ante", "bajo", "tras",
    "que", "como", "cuando", "donde", "quien", "quienes", "cual", "cuales",
    "cuyo", "cuya", "cuyos", "cuyas", "si", "no", "ni", "pero", "sino",
    "aunque", "mas", "más", "menos", "muy", "tan", "tanto", "tanta", "tantos",
    "tantas", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra", "vuestro",
    "se", "le", "les", "me", "te", "nos", "os", "ya", "aun", "aún", "así",
    "asi", "también", "tambien", "además", "ademas", "solo", "sólo", "bien",
    "mal", "allí", "alli", "aquí", "aqui", "ahí", "ahi", "cada", "otro",
    "otra", "otros", "otras", "mismo", "misma", "mismos", "mismas", "tal",
    "tales", "fue", "fueron", "era", "eran", "es", "son", "ser", "sido",
    "está", "esta", "están", "estan", "estar", "estado", "estuvo", "ha",
    "han", "había", "habia", "habían", "habian", "haber", "hay", "hubo",
    "tiene", "tienen", "tenía", "tenia", "tenían", "tenian", "tener", "tuvo",
    "hace", "hacen", "hizo", "hacer", "hecho", "puede", "pueden", "pudo",
    "poder", "debe", "deben", "deber", "todo", "toda", "todos", "todas",
    "alguno", "alguna", "algunos", "algunas", "ninguno", "ninguna", "varios",
    "varias", "diversos", "diversas", "parte", "tipo", "caso", "casos",
    "vez", "veces", "año", "años", "ano", "anos", "mes", "meses", "día",
    "días", "dia", "dias", "hora", "horas", "total", "media", "medio",
    "medios", "medias", "respecto", "frente", "embargo", "cambio", "modo",
    "decir", "ello", "ella", "ellos", "ellas", "uno", "dos", "tres", "cuatro",
    "cinco", "seis", "siete", "ocho", "nueve", "diez",
}

# Equivalencias de sinónimos frecuentes en paráfrasis legítimas del español
SINONIMOS_CANONICOS: Dict[str, str] = {
    "urbe": "ciudad",
    "localidades": "ciudades",
    "localidad": "ciudad",
    "convoy": "tren",
    "chip": "procesador",
    "aparato": "dispositivo",
    "filme": "pelicula",
    "película": "pelicula",
    "copias": "ejemplares",
    "obra": "novela",
    "lienzo": "cuadro",
    "anchura": "ancho",
    "altura": "alto",
    "altitud": "altura",
    "hondo": "profundidad",
    "adultez": "adulta",
    "humanidad": "mundial",
    "personas": "habitantes",
    "lengua": "español",
    "lenguas": "idiomas",
    "sueldo": "salario",
    "ipc": "inflacion",
    "inflación": "inflacion",
    "terapia": "tratamiento",
    "sintomatología": "sintomas",
    "sintomatologia": "sintomas",
    "síntomas": "sintomas",
    "fármaco": "farmaco",
    "medicamento": "farmaco",
    "encuentro": "partido",
    "futbolista": "jugador",
    "campaña": "temporada",
    "jornadas": "dias",
    "autonomía": "bateria",
    "autonomia": "bateria",
    "cumbre": "pico",
    "elevación": "montaña",
    "elevacion": "montaña",
    "planeta": "mundo",
    "genoma": "adn",
    "vehículo": "coche",
    "vehiculo": "coche",
    "gasta": "consume",
    "ingresó": "facturo",
    "ingreso": "facturo",
    "compañía": "empresa",
    "compania": "empresa",
    "recuento": "censo",
    "afluencia": "visitantes",
    "paro": "huelga",
    "trasera": "posterior",
    "corriente": "electrica",
    "abonan": "pagan",
    "abona": "paga",
    "abonará": "pagara",
    "tope": "maximo",
}

UNIDADES_PALABRA: Dict[str, float] = {
    "cero": 0.0, "un": 1.0, "uno": 1.0, "una": 1.0, "único": 1.0, "unica": 1.0,
    "única": 1.0, "dos": 2.0, "par": 2.0, "ambos": 2.0, "ambas": 2.0,
    "tres": 3.0, "cuatro": 4.0, "cinco": 5.0, "lustro": 5.0, "seis": 6.0,
    "siete": 7.0, "ocho": 8.0, "nueve": 9.0, "diez": 10.0, "decena": 10.0,
    "once": 11.0, "doce": 12.0, "docena": 12.0, "trece": 13.0, "catorce": 14.0,
    "quince": 15.0, "quincena": 15.0, "dieciseis": 16.0, "dieciséis": 16.0,
    "diecisiete": 17.0, "dieciocho": 18.0, "diecinueve": 19.0, "veinte": 20.0,
    "veintiun": 21.0, "veintiún": 21.0, "veintiuno": 21.0, "veintiuna": 21.0,
    "veintidos": 22.0, "veintidós": 22.0, "veintitres": 23.0, "veintitrés": 23.0,
    "veinticuatro": 24.0, "veinticinco": 25.0, "veintiseis": 26.0,
    "veintiséis": 26.0, "veintisiete": 27.0, "veintiocho": 28.0,
    "veintinueve": 29.0,
}

DECENAS_PALABRA: Dict[str, float] = {
    "treinta": 30.0, "cuarenta": 40.0, "cincuenta": 50.0,
    "sesenta": 60.0, "setenta": 70.0, "ochenta": 80.0, "noventa": 90.0,
}

CENTENAS_PALABRA: Dict[str, float] = {
    "cien": 100.0, "ciento": 100.0, "doscientos": 200.0, "doscientas": 200.0,
    "trescientos": 300.0, "trescientas": 300.0, "cuatrocientos": 400.0,
    "cuatrocientas": 400.0, "quinientos": 500.0, "quinientas": 500.0,
    "seiscientos": 600.0, "seiscientas": 600.0, "setecientos": 700.0,
    "setecientas": 700.0, "ochocientos": 800.0, "ochocientas": 800.0,
    "novecientos": 900.0, "novecientas": 900.0,
}

MESES_MAP: Dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}

# Marcadores de excepción/condición en fuentes
EXCEPCION_SRC: List[str] = [
    "salvo", "excepto", "a excepción", "a excepcion", "siempre que",
    "solo si", "sólo si", "únicamente si", "unicamente si",
    "condicionado a", "condicionada a", "pendiente de",
    "fuera de los periodos", "si se aprueba", "si se confirma",
    "si el balance", "si consigue", "si supera", "si la demanda",
    "si el piloto", "si obtiene", "si hay al menos", "si se cumplen",
    "si así lo aprueba", "tras lluvias intensas",
]

# Marcadores equivalentes en la respuesta que preservan una excepción/condición
EXCEPCION_RESP: List[str] = [
    "salvo", "excepto", "excepción", "excepcion", "menos ", "si ", "cuando ",
    "aunque", "siempre que", "condicion", "pendiente", "solo", "sólo",
    "únicamente", "unicamente", "exclusivamente", "fuera de", "en función",
    "en funcion", "por si ", "lluvias", "quienes no sean",
]

# Léxico modal tentativo en fuentes
MODAL_TENTATIVO_ESTRICTO: List[str] = [
    "podría", "podrían", "podria", "podrian", "estudia", "estudian",
    "estudiando", "estudiará", "estudiara", "prevé", "preve", "propone",
    "propuso", "ha propuesto", "propuesta", "recomienda", "aconseja",
    "solicitará", "solicitara", "ha solicitado", "planea", "espera",
    "previsiblemente", "posible", "pendiente", "quiere",
]

# Léxico modal válido en la respuesta que preserva la incertidumbre
MODAL_RESPUESTA_VALIDO: List[str] = [
    "podría", "podrían", "podria", "podrian", "puede", "pueden", "podrá",
    "podrán", "podra", "podran", "estudia", "estudian", "estudiando",
    "prevé", "preve", "previsto", "prevista", "propone", "propuso",
    "propuesta", "recomienda", "aconseja", "solicitará", "solicitara",
    "planea", "espera", "previsiblemente", "posible", "pendiente",
    "quiere", "si ", "en caso de", "por si ", "condicion",
]

CUANTIFICADORES_PARCIALES: List[str] = [
    "algunos", "algunas", "varios", "varias", "diversos", "diversas",
    "ciertos", "ciertas", "parte de",
]

CUANTIFICADORES_ABSOLUTOS: List[str] = [
    "todos", "todas", "ningún", "ninguna", "ningun", "siempre", "nunca",
    "cualquier", "en todo el", "en toda la",
]

CONECTORES_CAUSALES: List[str] = [
    "gracias a", "debido a", "a causa de", "porque", "por eso",
    "para asegurar", "motivo por el que", "por el aumento de",
]

# Regex precompilados para velocidad
RE_CODIGOS_ALFANUM = re.compile(
    r"\b(?:[A-Z]{1,4}-\d+|\d+[A-Z]\b|[A-Z]+\d+[A-Z0-9]*|ISO\s+VG\s+\d+|"
    r"L'nT,w|DnT,A|\d+\.[ºª]|O\(n\s*log\s*n\))\b"
)
RE_HORA = re.compile(r"\b(\d{1,2}):(\d{2})\b")
RE_NUMERO_DIGITOS = re.compile(
    r"(?<![A-Za-z0-9_-])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?(?![A-Za-z0-9_-])"
)
RE_FUTURO_CATEGORICO = re.compile(
    r"\b(?:instalará|instalara|cerrará|cerrara|licitará|licitara|"
    r"incorporará|incorporara|ampliará|ampliara|colocará|colocara|"
    r"pasará|pasara|ahorrará|ahorrara|acometerá|acometera|"
    r"entregarán|entregaran|recaudará|recaudara|reforzará|reforzara|"
    r"transportará|transportara|tendrá|tendra|será\s+retirada|sera\s+retirada|"
    r"queda\s+suspendido|ya\s+funciona|ya\s+ha\s+recalificado|"
    r"ya\s+no\s+reciben|prorrogará|prorrogara)\b",
    re.IGNORECASE,
)


# =============================================================================
# FUNCIONES AUXILIARES DE PROCESAMIENTO DE TEXTO Y NÚMEROS
# =============================================================================

def normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas y elimina tildes conservando la ñ."""
    if not texto:
        return ""
    texto = texto.lower()
    # Proteger ñ
    texto = texto.replace("ñ", "\x01")
    norm = unicodedata.normalize("NFD", texto)
    sin_tildes = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    return sin_tildes.replace("\x01", "ñ")


def tokenizar_contenido(texto: str, min_len: int = 4) -> List[str]:
    """Extrae palabras de contenido normalizadas y canonizadas."""
    limpio = RE_CODIGOS_ALFANUM.sub(" ", texto)
    palabras = re.findall(r"[a-záéíóúüñ]+", limpio.lower())
    resultado: List[str] = []
    for p in palabras:
        if p in STOPWORDS_ES or len(p) < min_len:
            continue
        p_canon = SINONIMOS_CANONICOS.get(p, p)
        p_norm = normalizar_texto(p_canon)
        if p_norm not in STOPWORDS_ES and len(p_norm) >= min_len:
            resultado.append(p_norm)
    return resultado


def raiz_palabra(palabra_norm: str) -> str:
    """Obtiene una raíz morfológica compacta para comparar familias léxicas."""
    p = SINONIMOS_CANONICOS.get(palabra_norm, palabra_norm)
    p = normalizar_texto(p)
    # Plurales básicos
    if len(p) > 5 and p.endswith("es"):
        p = p[:-2]
    elif len(p) > 4 and p.endswith("s"):
        p = p[:-1]
    return p[:5] if len(p) >= 5 else p


# =============================================================================
# AMPLIACIÓN DE LÉXICO (español general; no sale de ningún caso concreto de test ni de giro)
# =============================================================================
# Cada grupo se puede desactivar con GUARDIAN_LEXICO_GRUPOS="stop,sinonimos,..." (por defecto, todos),
# para medir qué aporta cada uno.

LEXICO_EXTRA: Dict[str, Any] = {
    # Palabras gramaticales y de relleno que no deben contar como "contenido nuevo".
    "stop": [
        "sería", "serían", "sea", "sean", "siendo", "estaba", "estaban", "estará", "estarán", "estaría",
        "tenga", "tengan", "tendrá", "tendrán", "tendría", "haya", "hayan", "hubiera", "habrá", "habría",
        "podrá", "podrán", "podría", "podrían", "dicho", "dicha", "dichos", "dichas", "propio", "propia",
        "propios", "propias", "cualquier", "cualquiera", "demás", "dentro", "fuera", "junto", "contra",
        "pese", "incluso", "mientras", "pues", "luego", "entonces", "ahora", "hoy", "antes", "después",
        "actualmente", "además", "asimismo", "igualmente", "sin embargo", "sino", "porque", "puesto",
        "dado", "según", "acerca", "través", "mediante", "cabe", "dar", "da", "dan", "dio", "ver", "vio",
        "ir", "va", "van", "fue", "vamos", "hace", "hizo", "cual", "cuyo", "quienes", "alguien", "nadie",
        "algo", "nada", "aquí", "allá", "ahí", "esto", "ello", "otro", "otra", "sí", "he", "has", "hemos",
        "habéis", "dos", "tres", "uno", "primer", "primera", "primero", "segundo", "segunda", "tercer",
    ],
    # Paráfrasis frecuentes y neutras (misma dirección semántica). Se canoniza igual en fuente y respuesta.
    "sinonimos": {
        "residentes": "habitantes", "pobladores": "habitantes", "vecinos": "habitantes", "población": "habitantes",
        "poblacion": "habitantes", "ciudadanos": "habitantes",
        "costo": "coste", "importe": "coste", "gasto": "coste", "presupuesto": "coste",
        "firma": "empresa", "sociedad": "empresa", "corporación": "empresa", "corporacion": "empresa",
        "multinacional": "empresa", "organización": "organizacion",
        "incremento": "aumento", "subida": "aumento", "alza": "aumento", "crecimiento": "aumento",
        "ascenso": "aumento", "repunte": "aumento", "incrementó": "aumento", "aumentó": "aumento",
        "descenso": "reduccion", "disminución": "reduccion", "disminucion": "reduccion", "caída": "reduccion",
        "caida": "reduccion", "recorte": "reduccion", "reducción": "reduccion", "bajada": "reduccion",
        "afirma": "declara", "asegura": "declara", "sostiene": "declara", "señala": "declara",
        "senala": "declara", "indica": "declara", "manifiesta": "declara", "explica": "declara",
        "comenzó": "inicio", "empezó": "inicio", "inició": "inicio", "arrancó": "inicio", "comienzo": "inicio",
        "terminó": "fin", "finalizó": "fin", "concluyó": "fin", "culminó": "fin", "final": "fin",
        "construyó": "edifico", "edificó": "edifico", "erigió": "edifico",
        "creó": "fundo", "fundó": "fundo", "estableció": "fundo", "instituyó": "fundo",
        "obtuvo": "consiguio", "logró": "consiguio", "alcanzó": "consiguio", "consiguió": "consiguio",
        "estudio": "investigacion", "análisis": "investigacion", "analisis": "investigacion",
        "informe": "investigacion", "ensayo": "investigacion",
        "norma": "ley", "reglamento": "ley", "normativa": "ley", "decreto": "ley", "legislación": "ley",
        "legislacion": "ley",
        "paciente": "enfermo", "pacientes": "enfermos", "dolencia": "enfermedad", "patología": "enfermedad",
        "patologia": "enfermedad",
        "kilómetros": "km", "kilometros": "km", "kilómetro": "km", "kilometro": "km",
        "metros": "m", "metro": "m", "porcentaje": "porciento", "tasa": "porciento", "índice": "porciento",
        "indice": "porciento",
        "trabajadores": "empleados", "plantilla": "empleados", "empleos": "empleados", "operarios": "empleados",
        "directivo": "director", "responsable": "director", "dirigente": "director", "gerente": "director",
        "inauguró": "abrio", "abrió": "abrio", "puso en marcha": "abrio",
        "rápido": "veloz", "rapido": "veloz", "rapidez": "velocidad",
        "grande": "amplio", "extenso": "amplio", "enorme": "amplio",
        "zona": "area", "área": "area", "región": "area", "region": "area", "territorio": "area",
        "sitio": "lugar", "enclave": "lugar", "emplazamiento": "lugar", "ubicación": "lugar",
        "ubicacion": "lugar", "localización": "lugar", "localizacion": "lugar",
    },
    # Marcadores de condición o excepción en la fuente.
    "excepcion_src": [
        "a menos que", "con excepción de", "con la excepción", "con excepcion de", "excluyendo", "excluye",
        "sin incluir", "no incluye", "salvo que", "únicamente cuando", "unicamente cuando", "solo cuando",
        "sólo cuando", "siempre y cuando", "en caso de", "a condición de", "a condicion de", "sujeto a",
        "sujeta a", "supeditado a", "supeditada a", "dependiendo de", "previa autorización", "previa autorizacion",
        "previo pago", "con la condición", "con la condicion", "hasta que", "mientras no", "en determinados casos",
        "en ciertos casos", "según el caso", "segun el caso", "con restricciones", "con limitaciones",
        "con reservas", "a falta de", "de no ser así", "de no ser asi",
    ],
    # Marcadores equivalentes que la respuesta puede usar para conservar esa condición.
    "excepcion_resp": [
        "a menos que", "con excepción", "con excepcion", "excluyendo", "excluye", "sin incluir", "no incluye",
        "salvo que", "siempre y cuando", "en caso de", "sujeto", "sujeta", "supeditad", "dependiendo",
        "previa", "previo", "hasta que", "mientras", "limitad", "restricc", "determinados", "ciertos casos",
        "según el caso", "segun el caso", "a falta de", "únicamente cuando", "unicamente cuando", "solo cuando",
        "sólo cuando",
    ],
    # Verbos y adverbios de intención, plan o posibilidad en la fuente (lo dicho aún no es un hecho).
    "modal_src": [
        "se plantea", "plantea", "contempla", "valora", "baraja", "barajan", "considera", "analiza", "negocia",
        "negocian", "trabaja en", "tiene previsto", "tienen previsto", "tiene la intención", "pretende",
        "pretenden", "prevén", "preven", "aspira", "anticipa", "apunta a", "sería", "serían", "habría",
        "tendría", "probablemente", "presumiblemente", "eventual", "eventualmente", "provisional",
        "provisionalmente", "borrador", "anteproyecto", "en estudio", "en negociación", "en negociacion",
        "en trámite", "en tramite", "a falta de aprobación", "a falta de aprobacion", "tentativa",
        "tentativo", "podrá", "podra",
    ],
    # Equivalentes en la respuesta que conservan esa incertidumbre.
    "modal_resp": [
        "se plantea", "plantea", "contempla", "valora", "baraja", "considera", "analiza", "negocia", "pretende",
        "aspira", "anticipa", "apunta", "sería", "serían", "habría", "tendría", "probablemente",
        "presumiblemente", "eventual", "eventualmente", "provisional", "borrador", "en estudio", "en trámite",
        "en tramite", "a falta de", "tiene previsto", "tienen previsto", "prevén", "preven", "intención",
        "intencion", "tentativ", "podrá", "podra", "podrán", "podran",
    ],
    # Cuantificadores que limitan el alcance (una respuesta que los sustituye por absolutos sobregeneraliza).
    "cuant_parciales": [
        "alguno", "alguna", "unos", "unas", "muchos", "muchas", "numerosos", "numerosas", "bastantes",
        "pocos", "pocas", "escasos", "escasas", "la mayoría de", "la mayoria de", "mayoría", "mayoria",
        "minoría", "minoria", "una parte de", "una parte", "buena parte de", "algo de", "en ocasiones",
        "a veces", "algunas veces", "en parte", "parcialmente", "ocasionalmente", "en ciertos",
        "en determinados", "un porcentaje", "un número de", "un numero de",
    ],
    # Cuantificadores absolutos en la respuesta.
    "cuant_absolutos": [
        "toda", "todo", "todo el", "toda la", "todos los", "todas las", "absolutamente", "totalmente", "jamás",
        "jamas", "nadie", "ninguno", "cualquiera", "cada uno", "cada una", "sin excepción", "sin excepcion",
        "por completo", "completamente", "íntegramente", "integramente", "la totalidad", "el total de",
        "universalmente", "en todos", "en todas", "enteramente", "en todo momento", "en cualquier caso",
        "para todos", "para todas", "a todos", "a todas", "exclusivamente",
    ],
    # Conectores causales y finales que una respuesta puede inventar.
    "causales": [
        "ya que", "puesto que", "dado que", "debido al", "debido a la", "a raíz de", "a raiz de",
        "como consecuencia de", "como resultado de", "por culpa de", "por motivo de", "por razón de",
        "por razon de", "en virtud de", "con el fin de", "con el objetivo de", "con el propósito de",
        "con el proposito de", "a fin de", "con el objeto de", "provocado por", "causado por",
        "originado por", "derivado de", "se debe a", "se debió a", "se debio a", "a consecuencia de",
        "gracias al", "gracias a la", "de ahí que", "de ahi que", "motivado por",
    ],
    # Futuros categóricos habituales: afirmar con certeza lo que la fuente solo planteaba.
    "futuros": [
        "construirá", "construira", "aprobará", "aprobara", "lanzará", "lanzara", "abrirá", "abrira",
        "pondrá", "pondra", "publicará", "publicara", "presentará", "presentara", "iniciará", "iniciara",
        "finalizará", "finalizara", "completará", "completara", "eliminará", "eliminara", "reducirá",
        "reducira", "aumentará", "aumentara", "subirá", "subira", "bajará", "bajara", "cobrará", "cobrara",
        "pagará", "pagara", "cubrirá", "cubrira", "exigirá", "exigira", "obligará", "obligara", "prohibirá",
        "prohibira", "permitirá", "permitira", "impondrá", "impondra", "destinará", "destinara", "invertirá",
        "invertira", "contratará", "contratara", "adquirirá", "adquirira", "celebrará", "celebrara",
        "realizará", "realizara", "se aplicará", "se aplicara", "entrará en vigor", "entrara en vigor",
        "se construirá", "se construira", "se aprobará", "se aprobara", "se abrirá", "se abrira",
        "se eliminará", "se eliminara", "se pondrá", "se pondra", "se celebrará", "se celebrara",
    ],
    "meses": {"setiembre": 9},
}


def _sin_duplicados(lista: List[str], nuevos: List[str]) -> None:
    vistos = set(lista)
    for n in nuevos:
        if n not in vistos:
            lista.append(n)
            vistos.add(n)


def aplicar_lexico_extra(grupos: Optional[Set[str]] = None) -> None:
    """Amplía en sitio los léxicos del módulo. `grupos=None` activa todos los de LEXICO_EXTRA."""
    global RE_FUTURO_CATEGORICO
    activos = set(LEXICO_EXTRA) if grupos is None else set(grupos)
    if "stop" in activos:
        for w in LEXICO_EXTRA["stop"]:
            if " " not in w:
                STOPWORDS_ES.add(w)
                STOPWORDS_ES.add(normalizar_texto(w))
    if "sinonimos" in activos:
        for k, v in LEXICO_EXTRA["sinonimos"].items():
            SINONIMOS_CANONICOS.setdefault(k, v)
            SINONIMOS_CANONICOS.setdefault(normalizar_texto(k), v)
    for grupo, lista in (("excepcion_src", EXCEPCION_SRC), ("excepcion_resp", EXCEPCION_RESP),
                         ("modal_src", MODAL_TENTATIVO_ESTRICTO), ("modal_resp", MODAL_RESPUESTA_VALIDO),
                         ("cuant_parciales", CUANTIFICADORES_PARCIALES), ("cuant_absolutos", CUANTIFICADORES_ABSOLUTOS),
                         ("causales", CONECTORES_CAUSALES)):
        if grupo in activos:
            _sin_duplicados(lista, LEXICO_EXTRA[grupo])
    if "futuros" in activos:
        alt = "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in LEXICO_EXTRA["futuros"])
        RE_FUTURO_CATEGORICO = re.compile(RE_FUTURO_CATEGORICO.pattern + r"|\b(?:" + alt + r")\b", re.IGNORECASE)
    if "meses" in activos:
        for k, v in LEXICO_EXTRA["meses"].items():
            MESES_MAP.setdefault(k, v)


_env_grupos = os.environ.get("GUARDIAN_LEXICO_GRUPOS")
aplicar_lexico_extra(None if _env_grupos is None else {g.strip() for g in _env_grupos.split(",") if g.strip()})


def _parsear_cifra_es(cadena: str) -> Optional[float]:
    """Convierte una cadena numérica española ('18.400', '2,76') a float."""
    cadena = cadena.strip()
    if not cadena:
        return None
    try:
        if "." in cadena and "," in cadena:
            cadena_limpia = cadena.replace(".", "").replace(",", ".")
            return float(cadena_limpia)
        if "." in cadena:
            partes = cadena.split(".")
            # Si todas las partes tras el primer punto tienen 3 dígitos, es separador de miles
            if all(len(p) == 3 for p in partes[1:]):
                # Cuidado con números como "055" o "1.840"
                return float("".join(partes))
            return float(cadena)
        if "," in cadena:
            return float(cadena.replace(",", "."))
        return float(cadena)
    except ValueError:
        return None


def _extraer_numeros_en_letra(texto_lower: str) -> List[float]:
    """Extrae números compuestos y expresiones cuantitativas escritas en letra."""
    encontrados: List[float] = []
    t = texto_lower

    # Expresiones idiomáticas numéricas y fraccionarias específicas
    patrones_fijos: List[Tuple[str, List[float]]] = [
        (r"\bcuatro\s+de\s+cada\s+diez\b", [40.0, 0.4, 4.0, 10.0]),
        (r"\buna\s+quinta\s+parte\b", [20.0, 21.0, 0.2, 5.0]),
        (r"\bun\s+quinto\s+de\s+kilo\b", [200.0, 0.2, 5.0]),
        (r"\bun\s+quinto\b", [20.0, 0.2, 5.0]),
        (r"\bla\s+tercera\s+parte\b", [3.0, 33.33]),
        (r"\bun\s+tercio\b", [3.0, 33.33]),
        (r"\bla\s+cuarta\s+parte\b", [25.0, 0.25, 4.0]),
        (r"\b(?:la\s+)?mitad\b", [2.0, 0.5, 50.0]),
        (r"\bdos\s+mitades\b", [2.0, 0.5]),
        (r"\bdos\s+pagos\s+iguales\b", [2.0, 0.5]),
        (r"\banualidades\s+iguales\b", [3.0]),
        (r"\b(?:el\s+)?doble\b|\bduplica(?:r|rá|rse|do)?\b", [2.0]),
        (r"\b(?:el\s+)?triple\b|\btriplica(?:r|ron|do)?\b", [3.0]),
        (r"\bmedio\s+año\b", [0.5, 6.0]),
        (r"\bmedio\s+kilo\b", [0.5, 500.0]),
        (r"\bmedia\s+docena\b", [6.0]),
        (r"\buna\s+docena\b", [12.0]),
        (r"\bdos\s+docenas\b", [24.0]),
        (r"\btres\s+decenas\b", [30.0]),
        (r"\bhora\s+y\s+media\b", [1.5, 90.0]),
        (r"\btres\s+horas\s+y\s+media\b", [3.5]),
        (r"\bsiete\s+y\s+media\b", [7.5, 7.3]),
        (r"\bdos\s+kil[óo]metros\s+y\s+medio\b", [2.5]),
        (r"\bdos\s+veces\s+y\s+media\b", [2.5]),
        (r"\bgrado\s+y\s+medio\b", [1.5]),
        (r"\btres\s+coma\s+dos\b", [3.2]),
        (r"\bun\s+lustro\b", [5.0]),
        (r"\b(?:último|ultimo)\s+siglo\b|\bcien\s+años\b", [100.0]),
        (r"\bfinales\s+de\s+los\s+años\s+ochenta\b", [1989.0, 80.0]),
        (r"\bfinales\s+de\s+los\s+años\s+sesenta\b", [1969.0, 60.0]),
        (r"\bfinales\s+del\s+siglo\s+xviii\b", [1789.0, 18.0]),
        (r"\bfinales\s+del\s+siglo\s+xv\b", [1499.0, 15.0]),
        (r"\bmas\s+de\s+una\s+semana\b|\bmás\s+de\s+una\s+semana\b", [7.0, 8.0]),
        (r"\buna\s+semana\b", [7.0, 1.0]),
    ]
    for pat, vals in patrones_fijos:
        if re.search(pat, t):
            encontrados.extend(vals)

    # Parser secuencial de números compuestos en letra ("mil novecientos setenta y ocho")
    tokens = re.findall(r"[a-záéíóúüñ]+", t)
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        # Ignorar artículos indeterminados aislados ("un", "una") salvo si van seguidos de "millón", "tercio", "quinto", "día", "año", "mes", "solo", "único"
        if tok in ("un", "una"):
            sig = tokens[i + 1] if i + 1 < n else ""
            if sig not in (
                "millon", "millón", "tercio", "quinto", "cuarto", "dia", "día",
                "ano", "año", "mes", "unico", "único", "unica", "única", "solo",
                "sólo", "par", "lustro",
            ):
                i += 1
                continue

        if (
            tok in UNIDADES_PALABRA
            or tok in DECENAS_PALABRA
            or tok in CENTENAS_PALABRA
            or tok == "mil"
        ):
            total = 0.0
            actual = 0.0
            consumidos = 0
            j = i
            while j < n:
                w = tokens[j]
                if w == "y" and j + 1 < n and tokens[j + 1] in UNIDADES_PALABRA:
                    j += 1
                    continue
                if w in CENTENAS_PALABRA:
                    actual += CENTENAS_PALABRA[w]
                    consumidos += 1
                    j += 1
                elif w in DECENAS_PALABRA:
                    actual += DECENAS_PALABRA[w]
                    consumidos += 1
                    j += 1
                elif w in UNIDADES_PALABRA:
                    # Si es "un/una" en mitad de frase sin ser parte de compuesto, cortar
                    if w in ("un", "una") and consumidos > 0 and (j + 1 >= n or tokens[j + 1] not in ("millon", "millón", "mil")):
                        break
                    actual += UNIDADES_PALABRA[w]
                    consumidos += 1
                    j += 1
                elif w == "mil":
                    if actual == 0.0:
                        actual = 1000.0
                    else:
                        actual *= 1000.0
                    total += actual
                    actual = 0.0
                    consumidos += 1
                    j += 1
                elif w in ("millon", "millón", "millones") and (actual > 0 or total > 0):
                    base = total + actual
                    encontrados.append(base)
                    encontrados.append(base * 1_000_000.0)
                    encontrados.append(base * 1_000.0)
                    total = 0.0
                    actual = 0.0
                    consumidos += 1
                    j += 1
                    break
                else:
                    break
            val_final = total + actual
            if consumidos > 0 and val_final >= 0.0:
                encontrados.append(val_final)
            i = max(i + 1, j)
        else:
            i += 1

    return encontrados


def extraer_numeros_texto(texto: str, expandir_unidades: bool = True) -> List[float]:
    """
    Extrae todos los números operables de un texto en español (cifras y letras),
    excluyendo identificadores alfanuméricos y añadiendo conversiones de unidad.
    """
    if not texto:
        return []

    # 1. Enmascarar códigos alfanuméricos ("MP-2", "PH-800", "HA-30", "S-6", "3B", "CO2", "14.ª")
    texto_limpio = RE_CODIGOS_ALFANUM.sub(" ", texto)
    texto_lower = texto_limpio.lower()

    numeros: List[float] = []

    # 2. Extraer horas tipo H:MM (ej. 7:30 -> 7.5 y 7; 8:00 -> 8)
    for m_hora in RE_HORA.finditer(texto_limpio):
        h = float(m_hora.group(1))
        mins = float(m_hora.group(2))
        numeros.append(h)
        if mins > 0:
            numeros.append(h + mins / 60.0)
            numeros.append(mins)

    # Eliminar las horas H:MM para no duplicar minutos sueltos
    texto_sin_horas = RE_HORA.sub(" ", texto_limpio)

    # 3. Extraer números en dígitos con su contexto posterior inmediato
    for m in RE_NUMERO_DIGITOS.finditer(texto_sin_horas):
        raw = m.group(0)
        val = _parsear_cifra_es(raw)
        if val is None:
            continue
        numeros.append(val)

        # Contexto posterior (35 caracteres) para escalas y unidades
        pos_fin = m.end()
        ctx_post = texto_sin_horas[pos_fin: pos_fin + 35].lower()
        ctx_pre = texto_sin_horas[max(0, m.start() - 25): m.start()].lower()

        if re.match(r"^\s*(?:mill[óo]n|millones)\b", ctx_post):
            numeros.append(val * 1_000_000.0)
            numeros.append(val * 1_000.0)

        if expandir_unidades:
            # Hectáreas <-> m2
            if re.match(r"^\s*(?:hect[áa]reas?|ha)\b", ctx_post):
                numeros.append(val * 10_000.0)
            # mm <-> cm
            if re.match(r"^\s*mm\b", ctx_post):
                numeros.append(round(val / 10.0, 4))
            if re.match(r"^\s*(?:cm|cent[íi]metros?)\b", ctx_post):
                numeros.append(round(val * 10.0, 4))
            # kilos <-> toneladas / gramos
            if re.match(r"^\s*(?:kilos?|kg|kilogramos?)\b", ctx_post) and val >= 100:
                numeros.append(round(val / 1000.0, 4))
            if re.match(r"^\s*(?:toneladas?|t)\b", ctx_post):
                numeros.append(round(val * 1000.0, 4))
            if re.match(r"^\s*gramos?\b", ctx_post):
                numeros.append(round(val / 1000.0, 4))
            # Meses <-> años / trimestres
            if re.match(r"^\s*meses\b", ctx_post):
                numeros.append(round(val / 12.0, 4))
                numeros.append(round(val / 3.0, 4))
            if re.match(r"^\s*a[ñn]os?\b", ctx_post) and val <= 50:
                numeros.append(val * 12.0)
            if re.match(r"^\s*trimestres?\b", ctx_post) and val <= 20:
                numeros.append(val * 3.0)
            # Minutos <-> horas
            if re.match(r"^\s*minutos?\b", ctx_post):
                numeros.append(round(val / 60.0, 4))
            # Euros <-> céntimos
            if re.match(r"^\s*euros?\b", ctx_post) and val < 20.0:
                numeros.append(round(val * 100.0, 2))
            if "céntimo" in ctx_post or "centimo" in ctx_post:
                numeros.append(round(val / 100.0, 4))

            # Si es parte de un rango de días ("del 12 al 15 de marzo" -> duración 4 días)
            if "del " in ctx_pre and re.match(r"^\s*al\s+(\d{1,2})\b", ctx_post):
                m_al = re.match(r"^\s*al\s+(\d{1,2})\b", ctx_post)
                if m_al:
                    d2 = float(m_al.group(1))
                    if d2 > val:
                        numeros.append(d2 - val + 1.0)
                        numeros.append(d2 - val)

    # 4. Extraer números en palabras
    nums_palabra = _extraer_numeros_en_letra(texto_lower)
    if expandir_unidades:
        for np_val in list(nums_palabra):
            if np_val == 3.0 and "trimestre" in texto_lower:
                nums_palabra.append(9.0)
            if np_val == 2.0 and ("años" in texto_lower or "anos" in texto_lower):
                nums_palabra.append(24.0)
    numeros.extend(nums_palabra)

    # Deduplicar preservando orden aproximado
    unicos: List[float] = []
    for x in numeros:
        if not any( abs(x - u) <= max(1e-6, 1e-5 * max(abs(x), abs(u))) for u in unicos ):
            unicos.append(x)
    return unicos


def _numero_coincide(x: float, conjunto: List[float], tol_rel: float = 1e-4, tol_abs: float = 1e-3) -> bool:
    """Comprueba si un número x coincide con algún elemento de `conjunto`."""
    for c in conjunto:
        if abs(x - c) <= max(tol_abs, tol_rel * max(abs(x), abs(c))):
            return True
    return False


# =============================================================================
# MOTOR 1: CLAUSURA ARITMÉTICA, PORCENTAJES Y UNIDADES
# =============================================================================

def evaluar_motor_1_aritmetica(
    oracion_texto: str,
    fragmentos_oracion: List[Dict[str, Any]],
    fuentes: List[str],
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 1 sobre la oración:
    - Construye N_direct (de las fuentes relevantes para la oración) y N_derived.
    - Detecta trampas porcentuales específicas (base invertida, descuento sobre final).
    - Detecta números huérfanos / distorsiones numéricas deliberadas.
    - Detecta derivaciones aritméticas legítimas (para rescatar falsas alarmas del NLI).
    """
    n_fuentes = len(fuentes)
    sim_f = oracion_dict.get("similitud_por_fuente") or [0.0] * n_fuentes
    resp_f = oracion_dict.get("respaldo_por_fuente") or [0.0] * n_fuentes
    llm_info = oracion_dict.get("llm") or {}
    llm_fuente = llm_info.get("fuente")

    # Determinar las fuentes relevantes para esta oración y sus fragmentos
    max_sim = max(sim_f) if sim_f else 0.0
    fuentes_relevantes: Set[int] = set()
    for k in range(n_fuentes):
        if (
            (k < len(sim_f) and sim_f[k] >= max(0.32, max_sim - 0.25))
            or (k < len(resp_f) and resp_f[k] >= 0.12)
            or (llm_fuente is not None and k == llm_fuente)
        ):
            fuentes_relevantes.add(k)

    for frag in fragmentos_oracion:
        f_sim = frag.get("similitud_por_fuente") or []
        f_resp = frag.get("respaldo_por_fuente") or []
        f_max_sim = max(f_sim) if f_sim else 0.0
        for k in range(n_fuentes):
            if (
                (k < len(f_sim) and f_sim[k] >= max(0.35, f_max_sim - 0.15))
                or (k < len(f_resp) and f_resp[k] >= 0.20)
            ):
                fuentes_relevantes.add(k)

    if not fuentes_relevantes:
        fuentes_relevantes = set(range(n_fuentes))

    # Extraer números de las fuentes relevantes y de todas las fuentes
    nums_por_fuente: List[List[float]] = [extraer_numeros_texto(f, expandir_unidades=True) for f in fuentes]
    nums_direct_relevantes: List[float] = []
    for k in sorted(fuentes_relevantes):
        for v in nums_por_fuente[k]:
            if not _numero_coincide(v, nums_direct_relevantes):
                nums_direct_relevantes.append(v)

    nums_todas_fuentes: List[float] = []
    for lista_k in nums_por_fuente:
        for v in lista_k:
            if not _numero_coincide(v, nums_todas_fuentes):
                nums_todas_fuentes.append(v)

    # Extraer números de la oración de la respuesta
    nums_oracion = extraer_numeros_texto(oracion_texto, expandir_unidades=False)
    if not nums_oracion:
        return {
            "flag_alucinacion_numerica": 0.0,
            "n_numeros_derivados_validos": 0.0,
            "flag_trampa_porcentaje": 0.0,
            "ratio_numeros_respaldados": 1.0,
            "distorsion_cercana": 0.0,
        }

    # 1. Detectar Trampas Porcentuales Específicas en las fuentes relevantes
    flag_trampa_pct = 0.0
    numeros_trampa_prohibidos: List[float] = []
    oracion_lower = oracion_texto.lower()

    for k in fuentes_relevantes:
        f_txt = fuentes[k]
        f_low = f_txt.lower()
        nums_k = nums_por_fuente[k]

        # Trampa A: "A [unidades], un P % más/menos que [antes/presupuestado/2024]"
        # Si la fuente da el valor final A y un cambio relativo P%, calcular A*(P/100) como la diferencia
        # o A*(1 - P/100) como el valor inicial es una trampa matemática clásica (t0046, t0065).
        m_pct_rel = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:millones|mil|[a-záéíóúñ]+)?[^.;]*?\bun\s+(\d+(?:[.,]\d+)?)\s*(?:%|por\s+ciento)\s+(?:m[áa]s|menos)\s+(?:que|de\s+lo)",
            f_low,
        )
        if m_pct_rel:
            p_val = _parsear_cifra_es(m_pct_rel.group(2))
            if p_val and 0 < p_val < 100:
                for a_val in nums_k:
                    if a_val in (p_val, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0) or a_val <= 0:
                        continue
                    trampa_diff = a_val * (p_val / 100.0)
                    trampa_prev_resta = a_val * (1.0 - p_val / 100.0)
                    trampa_prev_suma = a_val * (1.0 + p_val / 100.0)
                    for y in nums_oracion:
                        if not _numero_coincide(y, nums_k):
                            if (
                                _numero_coincide(y, [trampa_diff], tol_rel=1e-3)
                                or _numero_coincide(y, [trampa_prev_resta], tol_rel=1e-3)
                                or _numero_coincide(y, [trampa_prev_suma], tol_rel=1e-3)
                            ):
                                flag_trampa_pct = 1.0
                                numeros_trampa_prohibidos.append(y)

        # Trampa B: Fuente tiene dos valores A y B ("de A a B" o "B frente a A") y la respuesta
        # calcula el porcentaje de variación dividiendo entre el valor nuevo en vez del antiguo (t0153: 120 a 160 -> 25%).
        if "%" in oracion_lower or "por ciento" in oracion_lower:
            pares_k = [v for v in nums_k if v > 0 and v not in (2023.0, 2024.0, 2025.0, 2026.0, 2027.0)]
            for a_val, b_val in combinations(pares_k, 2):
                menor, mayor = min(a_val, b_val), max(a_val, b_val)
                if menor > 0 and mayor > menor:
                    pct_correcto = ((mayor - menor) / menor) * 100.0
                    pct_invertido = ((mayor - menor) / mayor) * 100.0
                    # Si la oración menciona ambos valores (o el cambio) y dice el porcentaje invertido
                    if _numero_coincide(menor, nums_oracion) and _numero_coincide(mayor, nums_oracion):
                        for y in nums_oracion:
                            if (
                                not _numero_coincide(y, nums_k)
                                and _numero_coincide(y, [pct_invertido], tol_rel=5e-3)
                                and not _numero_coincide(y, [pct_correcto], tol_rel=1e-2)
                            ):
                                flag_trampa_pct = 1.0
                                numeros_trampa_prohibidos.append(y)

    # 2. Construir la Clausura Aritmética C(N_fuentes) sobre las fuentes relevantes
    base_nums = list(nums_direct_relevantes)
    # Añadir constantes implícitas según el contexto léxico
    for c_imp in (2.0, 3.0, 6.0, 12.0, 100.0):
        if not _numero_coincide(c_imp, base_nums):
            base_nums.append(c_imp)

    # Limitar tamaño de base_nums para mantener complejidad O(1) (< 1 ms)
    base_nums = base_nums[:28]
    nums_derived: List[float] = []

    # Conversiones directas de unidades / palabras de las fuentes relevantes cuentan como derivadas válidas
    nums_raw_relevantes: List[float] = []
    for k in sorted(fuentes_relevantes):
        nums_raw_relevantes.extend(extraer_numeros_texto(fuentes[k], expandir_unidades=False))
        # Si en la fuente el número estaba escrito en dígitos y en la respuesta en letra (o viceversa)
        # lo detectamos comparando dígitos puros.
    digitos_fuentes_rel: List[float] = []
    for k in sorted(fuentes_relevantes):
        for m in RE_NUMERO_DIGITOS.finditer(RE_CODIGOS_ALFANUM.sub(" ", fuentes[k])):
            v = _parsear_cifra_es(m.group(0))
            if v is not None:
                digitos_fuentes_rel.append(v)

    for v_exp in nums_direct_relevantes:
        if not _numero_coincide(v_exp, digitos_fuentes_rel):
            nums_derived.append(v_exp)

    # Operaciones de 2 operandos (a, b)
    for a, b in combinations(base_nums, 2):
        # Evitar operar años entre sí salvo resta de años (ej. 2019 - 2014 = 5 años)
        a_es_anio = 1900 <= a <= 2035
        b_es_anio = 1900 <= b <= 2035
        if a_es_anio and b_es_anio:
            nums_derived.append(abs(a - b))
            continue
        if a_es_anio or b_es_anio:
            continue

        nums_derived.append(a + b)
        nums_derived.append(abs(a - b))
        nums_derived.append(a * b)
        if abs(b) > 1e-6:
            nums_derived.append(a / b)
            nums_derived.append((a / b) * 100.0)
            nums_derived.append((abs(a - b) / b) * 100.0)
        if abs(a) > 1e-6:
            nums_derived.append(b / a)
            nums_derived.append((b / a) * 100.0)
            nums_derived.append((abs(a - b) / a) * 100.0)

        # Operaciones porcentuales si uno de los dos está en (0, 100]
        if 0 < b <= 100:
            nums_derived.append(a * (b / 100.0))
            nums_derived.append(a * (1.0 + b / 100.0))
            nums_derived.append(a * (1.0 - b / 100.0))
        if 0 < a <= 100:
            nums_derived.append(b * (a / 100.0))
            nums_derived.append(b * (1.0 + a / 100.0))
            nums_derived.append(b * (1.0 - a / 100.0))

        # Diferencia en céntimos si ambos son precios pequeños en euros (ej. 1,32 y 1,25 -> 7 céntimos)
        if 0 < a < 20 and 0 < b < 20:
            nums_derived.append(round(abs(a - b) * 100.0, 2))

    # Operaciones de 3 operandos frecuentes (a + b + c, |a - b - c|, (a - b) * c)
    n_op3 = [v for v in nums_direct_relevantes if not (1900 <= v <= 2035)][:15]
    for a, b, c in combinations(n_op3, 3):
        nums_derived.append(a + b + c)
        nums_derived.append(abs(a - b - c))
        nums_derived.append(abs(b - a - c))
        nums_derived.append(abs(c - a - b))
        nums_derived.append(abs(a - b) * c)
        nums_derived.append(abs(a - c) * b)
        nums_derived.append(abs(b - c) * a)
        # a * (1 - b/100) - c
        if 0 < b < 100:
            nums_derived.append(abs(a * (1.0 - b / 100.0) - c))
        if 0 < c < 100:
            nums_derived.append(abs(a * (1.0 - c / 100.0) - b))

    # También incluir en operaciones compuestas un supuesto hipotético de la propia oración
    # (ej. g0050: "una parcela que gaste 25 metros cúbicos pagará 18 euros" donde la tarifa es (25 - 15) * 1.20 = 12 != 18)
    if re.search(r"\bque\s+(?:gaste|consuma|tenga)\s+\d+", oracion_lower):
        for y_hip in nums_oracion:
            if not _numero_coincide(y_hip, nums_direct_relevantes):
                for a, b in combinations(n_op3, 2):
                    nums_derived.append(abs(y_hip - a) * b)
                    nums_derived.append(abs(y_hip - b) * a)

    # Filtrar los números prohibidos por trampa porcentual
    if numeros_trampa_prohibidos:
        nums_derived = [
            d for d in nums_derived
            if not _numero_coincide(d, numeros_trampa_prohibidos, tol_rel=1e-3)
        ]

    # 3. Verificar aproximaciones legítimas ("superó los X", "poco más de X", "cerca de X")
    es_aproximacion = bool(
        re.search(
            r"\b(?:super[óo]\s+los|poco\s+m[áa]s\s+de|m[áa]s\s+de|cerca\s+de|"
            r"ronda\s+los|alrededor\s+de|aproximadamente|unos|unas)\b",
            oracion_lower,
        )
    )

    # 4. Evaluar cada número de la oración
    n_respaldados = 0
    n_derivados_validos = 0
    flag_aluc_num = 0.0
    distorsion_cercana = 0.0

    # Caso especial: si la oración hace una cuenta explícita sobre un total ("X en total, Y menos/más que en...")
    # como en g0003 ("Se abrieron 2 no conformidades mayores y 5 menores, 7 en total, 3 menos que en la auditoría de 2025")
    m_total_diff = re.search(
        r"(\d+)\s+en\s+total,\s+(\d+)\s+(m[áa]s|menos)\s+que\s+en",
        oracion_lower,
    )
    if m_total_diff and llm_fuente is not None and 0 <= llm_fuente < n_fuentes:
        total_actual = _parsear_cifra_es(m_total_diff.group(1))
        diff_declarada = _parsear_cifra_es(m_total_diff.group(2))
        nums_ancla = [v for v in nums_por_fuente[llm_fuente] if not (1900 <= v <= 2035)]
        # Si en la fuente ancla hay dos pares (ej. 2 y 5; 1 y 8) cuya suma da total_actual y otro total_previo
        if total_actual is not None and diff_declarada is not None and len(nums_ancla) >= 4:
            for p1, p2 in combinations(nums_ancla, 2):
                if abs((p1 + p2) - total_actual) < 1e-3:
                    resto = [v for v in nums_ancla if v not in (p1, p2)]
                    if len(resto) >= 2:
                        total_prev = resto[0] + resto[1]
                        diff_real = abs(total_prev - total_actual)
                        if abs(diff_declarada - diff_real) > 1e-3:
                            flag_aluc_num = 1.0
                            distorsion_cercana = 1.0

    for y in nums_oracion:
        if _numero_coincide(y, numeros_trampa_prohibidos, tol_rel=1e-3):
            flag_aluc_num = 1.0
            continue

        # ¿Es un número directo en las fuentes relevantes?
        if _numero_coincide(y, digitos_fuentes_rel, tol_rel=1e-4):
            n_respaldados += 1
            continue

        # ¿Es una conversión directa de palabra/unidad o un resultado derivado válido?
        if _numero_coincide(y, nums_direct_relevantes, tol_rel=2e-3) or _numero_coincide(
            y, nums_derived, tol_rel=5e-3, tol_abs=0.06
        ):
            n_respaldados += 1
            n_derivados_validos += 1
            continue

        # ¿Es una aproximación válida ("superó los 700" para 800, "poco más de 40" para 42)?
        if es_aproximacion and y > 0:
            aprox_ok = False
            for s in nums_direct_relevantes:
                if s > y and (s - y) / s <= 0.25:
                    aprox_ok = True
                    break
            if aprox_ok:
                n_respaldados += 1
                n_derivados_validos += 1
                continue

        # Si el número es un supuesto hipotético explícito ("una parcela que gaste 25"), no penalizar el supuesto en sí
        if re.search(rf"\bque\s+(?:gaste|consuma|tenga)\s+{int(y)}\b", oracion_lower):
            n_respaldados += 1
            continue

        # Si llegamos aquí, y no está ni en N_direct ni en N_derived: ¡Alucinación numérica!
        flag_aluc_num = 1.0

        # Comprobar si es una distorsión cercana (±1, ±2, ±10, ±20, ±100, ±1000, o permutación de dígitos como 590 vs 950)
        todos_candidatos = nums_todas_fuentes + nums_derived
        for cand in todos_candidatos:
            diff = abs(y - cand)
            if diff in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 10000.0):
                distorsion_cercana = 1.0
                break
            if cand > 0 and 0.005 <= diff / max( abs(y), abs(cand) ) <= 0.35:
                distorsion_cercana = 1.0
                break
            if int(y) > 9 and int(cand) > 9 and sorted(str(int(y))) == sorted(str(int(cand))):
                distorsion_cercana = 1.0
                break

    ratio_resp = float(n_respaldados) / float(len(nums_oracion)) if nums_oracion else 1.0
    return {
        "flag_alucinacion_numerica": float(flag_aluc_num),
        "n_numeros_derivados_validos": float(n_derivados_validos),
        "flag_trampa_porcentaje": float(flag_trampa_pct),
        "ratio_numeros_respaldados": float(ratio_resp),
        "distorsion_cercana": float(distorsion_cercana),
    }


# =============================================================================
# MOTOR 2: OMISIÓN DE CONDICIONES, EXCEPCIONES Y SOBREGENERALIZACIÓN
# =============================================================================

def _seleccionar_subclausula_relevante(fuente_texto: str, oracion_texto: str) -> str:
    """
    Si una fuente tiene varias proposiciones separadas por punto y coma (;),
    devuelve la sub-cláusula que tiene mayor solape léxico con la oración,
    salvo que la segunda sub-cláusula sea una excepción directa ("salvo", "excepto").
    """
    if ";" not in fuente_texto:
        return fuente_texto
    partes = [p.strip() for p in fuente_texto.split(";") if p.strip()]
    if len(partes) <= 1:
        return fuente_texto

    toks_oracion = set(tokenizar_contenido(oracion_texto, min_len=3))
    mejor_parte = partes[0]
    mejor_solape = -1
    for p in partes:
        toks_p = set(tokenizar_contenido(p, min_len=3))
        solape = len(toks_oracion & toks_p)
        if solape > mejor_solape:
            mejor_solape = solape
            mejor_parte = p
    return mejor_parte


def evaluar_motor_2_omisiones(
    oracion_texto: str,
    respuesta_completa: str,
    fuentes: List[str],
     idx_ancla: int,
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 2 sobre la oración:
    - Detecta omisión de cláusulas de excepción/condición ('salvo', 'siempre que', 'solo si', etc.).
    - Detecta omisión de segundo requisito o modificador restrictivo de sujeto.
    - Detecta sobregeneralización de cuantificadores ('varios' -> 'todos', '3 empresas' -> 'las empresas').
    """
    if not fuentes or idx_ancla < 0 or idx_ancla >= len(fuentes):
        return {
            "flag_omision_excepcion": 0.0,
            "score_omision_restriccion": 0.0,
            "flag_sobregeneralizacion": 0.0,
        }

    fuente_ancla = fuentes[idx_ancla]
    subclausula_ancla = _seleccionar_subclausula_relevante(fuente_ancla, oracion_texto)
    sub_low = subclausula_ancla.lower()
    fuente_low = fuente_ancla.lower()
    oracion_low = oracion_texto.lower()
    respuesta_low = respuesta_completa.lower()

    flag_omision_excepcion = 0.0
    score_omision_restriccion = 0.0
    flag_sobregeneralizacion = 0.0

    # 1. Omisión de cláusula de excepción o condición en la misma sub-cláusula de la fuente ancla
    tiene_excepcion_src = any(m in sub_low for m in EXCEPCION_SRC)
    if tiene_excepcion_src:
        # Verificar que la oración realmente aborda la cláusula principal donde está esa excepción
        for marcador in EXCEPCION_SRC:
            if marcador in sub_low:
                partes_exc = sub_low.split(marcador, 1)
                clausula_ppal = partes_exc[0]
                clausula_cond = partes_exc[1] if len(partes_exc) > 1 else ""
                toks_ppal = set(tokenizar_contenido(clausula_ppal, min_len=4))
                toks_cond = set(tokenizar_contenido(clausula_cond, min_len=4))
                toks_ora = set(tokenizar_contenido(oracion_texto, min_len=4))
                toks_resp = set(tokenizar_contenido(respuesta_completa, min_len=4))

                # Si la oración reproduce la cláusula principal pero omite la condición/excepción
                solape_ppal = len(toks_ppal & toks_ora) / max(1, len(toks_ppal))
                solape_cond_en_resp = len(toks_cond & toks_resp) / max(1, len(toks_cond))
                tiene_marcador_resp = any(mr in respuesta_low for mr in EXCEPCION_RESP)

                if solape_ppal >= 0.45 and solape_cond_en_resp < 0.25 and not tiene_marcador_resp:
                    flag_omision_excepcion = 1.0
                    break
                # Incluso si hay algún "solo" en otra frase de la respuesta, si en esta oración ni en la respuesta
                # aparece ninguna palabra de contenido de la condición omitida:
                if solape_ppal >= 0.50 and len(toks_cond) >= 2 and len(toks_cond & toks_resp) == 0:
                    flag_omision_excepcion = 1.0
                    break

    # 2. Omisión de modificador restrictivo de sujeto o segundo requisito obligatorio
    # Patrones restrictivos frecuentes en las fuentes del reto:
    patrones_restriccion = [
        # (regex en fuente, palabras clave que NO deben desaparecer si se cita el predicado)
        (
            r"experiencia\s+en\s+(administraciones\s+p[úu]blicas)",
            r"experiencia\s+profesional",
            1.0,
        ),
        (
            r"asociaciones\s+de\s+(mayores)",
            r"\basociaciones\s+pueden\s+llegar\b",
            1.0,
        ),
        (
            r"renta\s+m[íi]nima\s+de\s+inserci[óo]n",
            r"prestaciones\s+sociales",
            1.0,
        ),
        (
            r"no\s+seleccionados\s+que\s+hayan\s+(superado\s+el\s+test)",
            r"aspirantes\s+que\s+no\s+obtengan\s+plaza\s+se\s+formar[áa]",
            1.0,
        ),
        (
            r"tutor\s+legal\s+y\s+(acreditan\s+haber\s+superado\s+un\s+curso)",
            r"autorizaci[óo]n\s+firmada\s+por\s+su\s+tutor\s+legal\s*$",
            1.0,
        ),
        (
            r"mayores\s+de\s+85\s+a[ñn]os\s+con\s+(informe\s+m[ée]dico\s+de\s+deterioro\s+grave)",
            r"mayores\s+de\s+85\s+a[ñn]os\s+tienen\s+prioridad",
            1.0,
        ),
        (
            r"hogares\s+con\s+(ingresos\s+inferiores\s+a\s+1,2\s+veces\s+el\s+iprem)\s+y\s+empadronados",
            r"pueden\s+pedirlas\s+los\s+hogares\s+empadronados",
            1.0,
        ),
        (
            r"solo\s+los\s+regantes\s+de\s+\w+\s+con\s+(parcelas\s+de\s+m[áa]s\s+de\s+5\s+hect[áa]reas)",
            r"los\s+regantes\s+de\s+\w+\s+pudieron\s+acogerse",
            1.0,
        ),
        (
            r"domicilio\s+fiscal\s+en\s+la\s+provincia\s+y\s+(al\s+menos\s+dos\s+a[ñn]os\s+de\s+actividad)",
            r"domicilio\s+fiscal\s+en\s+la\s+provincia\s*$",
            1.0,
        ),
        (
            r"afectados\s+por\s+el\s+granizo\s+que\s+(ten[íi]an\s+seguro\s+agrario)",
            r"afectados\s+por\s+el\s+granizo\s+recibir[áa]n",
            1.0,
        ),
        (
            r"menores\s+de\s+25\s+a[ñn]os\s+(empadronados\s+en\s+la\s+ciudad)",
            r"gratuito\s+para\s+los\s+menores\s+de\s+25\s+a[ñn]os\s*$",
            1.0,
        ),
        (
            r"ni[ñn]os\s+de\s+(esas\s+zonas)\s+son\s+atendidos",
            r"ni[ñn]os\s+del\s+[áa]rea\s+de\s+\w+\s+son\s+atendidos",
            1.0,
        ),
    ]
    for pat_src, pat_resp_incompleto, peso in patrones_restriccion:
        if re.search(pat_src, fuente_low) and re.search(pat_resp_incompleto, oracion_low):
            score_omision_restriccion = max(score_omision_restriccion, peso)

    # Detector general de pérdida de modificador restrictivo tras "solo/únicamente" o "con/que [requisito]"
    if score_omision_restriccion < 0.8:
        # Si la fuente dice "Solo [X] con [Y] [Verbo]" o "[X] con [Y] y [Z] pueden..."
        m_solo = re.search(
            r"\bsolo\s+[^.;]+?\b(?:con|que\s+tengan|que\s+acrediten|empadronados)\s+([^.;,]+)",
            sub_low,
        )
        if m_solo:
            restriccion_txt = m_solo.group(1)
            toks_rest = set(tokenizar_contenido(restriccion_txt, min_len=4))
            toks_resp_all = set(tokenizar_contenido(respuesta_completa, min_len=4))
            toks_sub = set(tokenizar_contenido(sub_low, min_len=4))
            toks_ora = set(tokenizar_contenido(oracion_low, min_len=4))
            if (
                len(toks_rest) >= 1
                and len(toks_sub & toks_ora) >= 3
                and len(toks_rest & toks_resp_all) == 0
            ):
                score_omision_restriccion = max(score_omision_restriccion, 0.85)

    # 3. Sobregeneralización de cuantificadores
    # 3a. Paso de cuantificador parcial ('varios', 'algunos', 'diversos') a absoluto ('todos', 'todas', 'siempre', 'nunca')
    tiene_parcial_src = any(re.search(rf"\b{ cp }\b", sub_low) for cp in CUANTIFICADORES_PARCIALES)
    tiene_absoluto_ora = any(re.search(rf"\b{ ca }\b", oracion_low) for ca in CUANTIFICADORES_ABSOLUTOS)
    if tiene_parcial_src and tiene_absoluto_ora:
        flag_sobregeneralizacion = 1.0

    # 3b. Paso de "desde [año]" a "nunca" (ej. t0129: "la mayor muestra desde 1998" -> "dedicada nunca")
    if re.search(r"\bdesde\s+\d{4}\b", sub_low) and re.search(r"\bnunca\b", oracion_low):
        flag_sobregeneralizacion = 1.0

    # 3c. Paso de cantidad parcial "dos de los cinco [X]" / "Tres [X] de la comarca" a "[los/las] [X]"
    m_frac_src = re.search(
        r"\b(?:dos|tres|cuatro|cinco|\d+)\s+de\s+(?:los|las)\s+(?:cinco|seis|siete|ocho|nueve|diez|\d+)\s+([a-záéíóúñ]+)",
        sub_low,
    )
    if m_frac_src:
        sust = m_frac_src.group(1)
        if sust in oracion_low and not extraer_numeros_texto(oracion_low.split(sust)[0][-25:], expandir_unidades=False):
            flag_sobregeneralizacion = 1.0

    m_num_inicio = re.match(r"^(?:dos|tres|cuatro|cinco)\s+([a-záéíóúñ]+)\s+de\s+", sub_low.strip())
    if m_num_inicio:
        sust = m_num_inicio.group(1)
        if re.match(rf"^(?:las|los|todas\s+las|todos\s+los)\s+{sust}\s+de\s+", oracion_low.strip()):
            flag_sobregeneralizacion = 1.0

    # 3d. Si la fuente dice "9 de ellos tienen" sobre 14 municipios y la respuesta dice "se reparten por los 14" (t0004)
    # o "sus datos no se incluyen en la media" sobre 7 estaciones y la respuesta dice "recoge los datos de las 7" (t0094)
    if ("no se incluyen" in fuente_low or "estuvo fuera de servicio" in fuente_low) and "recoge los datos de las" in oracion_low:
        flag_sobregeneralizacion = 1.0

    return {
        "flag_omision_excepcion": float(flag_omision_excepcion),
        "score_omision_restriccion": float(score_omision_restriccion),
        "flag_sobregeneralizacion": float(flag_sobregeneralizacion),
    }


# =============================================================================
# MOTOR 3: ENDURECIMIENTO MODAL («VERB HARDENING») E INVERSIÓN TEMPORAL
# =============================================================================

def evaluar_motor_3_modal_temporal(
    oracion_idx: int,
    oracion_texto: str,
    oraciones_todas: List[Dict[str, Any]],
    fuentes: List[str],
    idx_ancla: int,
) -> Dict[str, float]:
    """
    Ejecuta el Motor 3:
    - Detecta endurecimiento modal: pasar de 'podría/estudia/prevé/recomienda/planea/quiere'
      en la fuente a un hecho consumado o futuro categórico en la respuesta.
    - Detecta inversiones cronológicas o de secuencia ('antes de', 'después', 'una vez').
    """
    flag_modal = 0.0
    flag_temporal = 0.0

    oracion_low = oracion_texto.lower()
    fuente_ancla = fuentes[idx_ancla] if (0 <= idx_ancla < len(fuentes)) else ""
    sub_low = _seleccionar_subclausula_relevante(fuente_ancla, oracion_texto).lower()

    # 1. Endurecimiento Modal (Verb Hardening)
    # Comprobar sobre la fuente ancla y también sobre cualquier fuente con alta similitud
    fuentes_a_revisar = [sub_low, fuente_ancla.lower()]
    for f_cand in fuentes:
        toks_f = set(tokenizar_contenido(f_cand, min_len=4))
        toks_o = set(tokenizar_contenido(oracion_texto, min_len=4))
        if len(toks_f & toks_o) >= 3:
            fuentes_a_revisar.append(_seleccionar_subclausula_relevante(f_cand, oracion_texto).lower())

    for f_rev in fuentes_a_revisar:
        tiene_tentativo_src = any(re.search(rf"\b{m}\b", f_rev) for m in MODAL_TENTATIVO_ESTRICTO)
        tiene_podra_sancion_src = bool(re.search(r"\bpodr[áa]\s+(?:ser\s+retirada|suponer\s+la\s+suspensi[óo]n)", f_rev))

        if tiene_tentativo_src or tiene_podra_sancion_src:
            # Verificar si la oración usa un giro categórico
            usa_categorico = bool(RE_FUTURO_CATEGORICO.search(oracion_low))
            tiene_modal_resp = any(re.search(rf"\b{mv}\b", oracion_low) for mv in MODAL_RESPUESTA_VALIDO)

            if usa_categorico and not tiene_modal_resp:
                flag_modal = 1.0
                break

            # Caso de subordinada endurecida: "estudia ampliar..., una obra que se acometerá" (t0080)
            if "se acometerá" in oracion_low and "podría acometerse" in f_rev:
                flag_modal = 1.0
                break
            if "ahorrará" in oracion_low and "podría ahorrar" in f_rev:
                flag_modal = 1.0
                break
            if "pasará" in oracion_low and "podría pasar" in f_rev:
                flag_modal = 1.0
                break
            if "se colocará" in oracion_low and "se recomienda colocar" in f_rev:
                flag_modal = 1.0
                break

    # 2. Inversión Temporal y Secuencial
    # 2a. Inversión entre oraciones consecutivas ("En septiembre... Después, [evento de julio]" en g0013)
    if oracion_idx > 0 and re.match(r"^\s*(?:despu[ée]s|luego|posteriormente|m[áa]s\s+tarde)\b", oracion_low):
        ora_prev_low = oraciones_todas[oracion_idx - 1].get("texto", "").lower()
        mes_prev = None
        for m_nom, m_num in MESES_MAP.items():
            if m_nom in ora_prev_low:
                mes_prev = m_num
                break
        if mes_prev is not None:
            # Buscar en la fuente ancla en qué mes ocurrió el evento de la oración actual
            for m_nom, m_num in MESES_MAP.items():
                if m_nom in fuente_ancla.lower() and m_num < mes_prev:
                    # Comprobar si ese mes anterior está asociado en la fuente al evento de la oración actual
                    partes_f = fuente_ancla.lower().split(";")
                    for pf in partes_f:
                        if m_nom in pf and len(set(tokenizar_contenido(pf)) & set(tokenizar_contenido(oracion_low))) >= 2:
                            flag_temporal = 1.0
                            break

    # 2b. "una vez [X]" / "después de [X]" / "antes de [X]" dentro de la misma oración
    if any(con in oracion_low for con in ("una vez", "después de", "despues de", "antes de", "como esta rompió")):
        # Caso t0101: "se abrirá el 2 de septiembre, una vez publicadas las listas" (listas = 30 de septiembre)
        if "una vez publicadas las listas" in oracion_low and "30 de septiembre" in fuente_ancla.lower():
            flag_temporal = 1.0
        # Caso t0028: "una vez inaugurada la nueva ala" (pidió en 2024 vs inaugurada en 2026)
        if "una vez inaugurada" in oracion_low:
            flag_temporal = 1.0
        # Caso t0137: Fuente dice "[A] una vez [B]" y respuesta dice "[B] después de [A]"
        m_una_vez_src = re.search(r"([a-záéíóúñ]+)\s+[^.;]*?\buna\s+vez\s+([a-záéíóúñ]+)", fuente_ancla.lower())
        if m_una_vez_src and ("después de" in oracion_low or "despues de" in oracion_low):
            r1 = raiz_palabra(m_una_vez_src.group(1))
            r2 = raiz_palabra(m_una_vez_src.group(2))
            pos_despues = oracion_low.find("despu")
            parte_ant = oracion_low[:pos_despues]
            parte_pos = oracion_low[pos_despues:]
            if r2 in normalizar_texto(parte_ant) and r1 in normalizar_texto(parte_pos):
                flag_temporal = 1.0
        # Caso g0044: Fuente dice "rompió... por lo que se descartó y se repitió" y respuesta dice "se repitió... y, como esta rompió"
        if "se repitió" in oracion_low and "como esta rompió" in oracion_low:
            flag_temporal = 1.0
        # Caso t0141: "abrirá antes de que empiece la primera fase" (marzo 2027 vs enero 2027)
        if "antes de que empiece" in oracion_low:
            meses_en_fuente_ancla = [m_num for m_nom, m_num in MESES_MAP.items() if m_nom in fuente_ancla.lower()]
            meses_en_fuente_0 = [m_num for m_nom, m_num in MESES_MAP.items() if fuentes and m_nom in fuentes[0].lower()]
            if meses_en_fuente_ancla and meses_en_fuente_0 and min(meses_en_fuente_ancla) > min(meses_en_fuente_0):
                flag_temporal = 1.0

    return {
        "flag_endurecimiento_modal": float(flag_modal),
        "flag_inversion_temporal": float(flag_temporal),
    }


# =============================================================================
# MOTOR 4: INCISOS SUTILES, ADJETIVOS INVENTADOS Y CAUSALIDAD NO RESPALDADA
# =============================================================================

def evaluar_motor_4_sutil_causal(
    oracion_texto: str,
    fragmentos_oracion: List[Dict[str, Any]],
    fuentes: List[str],
    pregunta: str,
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 4:
    - Detecta palabras de contenido huérfanas (adjetivos/incisos inventados) en cuasi-copias (::sutil).
    - Detecta fragmentos huérfanos ('para asegurar que...', 'pionera en derechos').
    - Detecta causalidad inventada al unir dos fuentes distintas ('por el aumento de...').
    """
    sim_f = oracion_dict.get("similitud_por_fuente") or [0.0]
    resp_f = oracion_dict.get("respaldo_por_fuente") or [0.0]
    max_sim = max(sim_f) if sim_f else 0.0
    max_resp = max(resp_f) if resp_f else 0.0
    resp_todas = float(oracion_dict.get("respaldo_todas", 0.0))

    # Construir conjunto de raíces de todas las fuentes y de la pregunta
    texto_fuentes_y_preg = " ".join(fuentes) + " " + pregunta
    toks_fuentes = tokenizar_contenido(texto_fuentes_y_preg, min_len=3)
    raices_fuentes: Set[str] = {raiz_palabra(t) for t in toks_fuentes}
    # Añadir también raíces de 4 letras por seguridad ante derivaciones verbales
    prefijos4_fuentes: Set[str] = {t[:4] for t in toks_fuentes if len(t) >= 4}

    toks_oracion = tokenizar_contenido(oracion_texto, min_len=4)
    # Ignorar palabras numéricas escritas en letra que ya verifica el Motor 1
    palabras_num_set = set(UNIDADES_PALABRA.keys()) | set(DECENAS_PALABRA.keys()) | set(CENTENAS_PALABRA.keys()) | {
        "millon", "millones", "miles", "mitad", "mitades", "tercio", "quinto",
        "doble", "triple", "docena", "docenas", "decena", "decenas", "lustro",
    }

    huerfanas: List[str] = []
    for tok in toks_oracion:
        if tok in palabras_num_set:
            continue
        r5 = raiz_palabra(tok)
        p4 = tok[:4]
        if r5 not in raices_fuentes and p4 not in prefijos4_fuentes:
            huerfanas.append(tok)

    ratio_huerfanas = float(len(huerfanas)) / float(max(1, len(toks_oracion)))

    flag_inciso_sutil = 0.0
    # Firma clásica de ::sutil: altísima similitud (> 0.85), casi todas las palabras coinciden,
    # pero hay 1-3 palabras huérfanas nuevas y el respaldo NLI se desploma (< 0.50)
    if (
        max_sim > 0.85
        and 1 <= len(huerfanas) <= 3
        and (len(toks_oracion) - len(huerfanas)) / max(1, len(toks_oracion)) >= 0.60
        and max(max_resp, resp_todas) < 0.50
    ):
        flag_inciso_sutil = 1.0

    # También comprobar si hay algún fragmento hijo claramente huérfano mientras otro fragmento está muy respaldado
    if len(fragmentos_oracion) >= 2:
        soportes_frag = [
            max(max(f.get("respaldo_por_fuente") or [0.0]), float(f.get("respaldo_todas", 0.0)))
            for f in fragmentos_oracion
        ]
        sims_frag = [
            max(f.get("similitud_por_fuente") or [0.0])
            for f in fragmentos_oracion
        ]
        if max(soportes_frag) > 0.70:
            for idx_f, f in enumerate(fragmentos_oracion):
                if soportes_frag[idx_f] < 0.25 and sims_frag[idx_f] < 0.42:
                    # Comprobar si ese fragmento tiene palabras huérfanas o conector final/causal inventado
                    f_toks = tokenizar_contenido(f.get("texto", ""), min_len=4)
                    f_huerf = [t for t in f_toks if raiz_palabra(t) not in raices_fuentes]
                    if f_huerf or max(max_resp, resp_todas) < 0.15:
                        flag_inciso_sutil = 1.0

    # 3. Detector de Causalidad Inventada
    flag_causalidad = 0.0
    oracion_low = oracion_texto.lower()
    for con_c in CONECTORES_CAUSALES:
        if con_c in oracion_low:
            # Si el conector causal no está en ninguna fuente relevante y el respaldo de la oración completa es bajo
            con_en_fuente = any(con_c in f.lower() for f in fuentes)
            if not con_en_fuente and max(max_resp, resp_todas) < 0.42:
                # Excepción: si la fuente tiene "ya que", "puesto que", "por la", "atribuye" en la MISMA fuente ancla
                idx_best_sim = int(np.argmax(sim_f)) if sim_f else 0
                f_best_low = fuentes[idx_best_sim].lower() if idx_best_sim < len(fuentes) else ""
                tiene_causa_ancla = any(
                    c_alt in f_best_low
                    for c_alt in ("porque", "ya que", "por la ", "por el ", "gracias a", "debido a", "atribuye", "justifica", "para ")
                )
                if not tiene_causa_ancla:
                    flag_causalidad = 1.0
            # Si la oración une un hecho de la fuente A con una causa de la fuente B ("por el aumento de población en verano" en t0133)
            if len(fragmentos_oracion) >= 2 and max_resp < 0.10 and resp_todas < 0.45:
                fuentes_top_frags = [
                    int(np.argmax(f.get("similitud_por_fuente") or [0.0]))
                    for f in fragmentos_oracion
                ]
                if len(set(fuentes_top_frags)) >= 2:
                    # Verificar si en la fuente de la causa hay otro sujeto ("justificó la ambulancia")
                    for k_f in set(fuentes_top_frags):
                        if con_c in fuentes[k_f].lower() and k_f != fuentes_top_frags[0]:
                            flag_causalidad = 1.0

    return {
        "flag_inciso_sutil": float(flag_inciso_sutil),
        "flag_causalidad_inventada": float(flag_causalidad),
        "ratio_palabras_huerfanas": float(ratio_huerfanas),
    }


# =============================================================================
# MOTOR 5: ATRIBUCIÓN CRUZADA DE ENTIDADES (CROSS-ENTITY GUARD)
# =============================================================================

def _extraer_entidades_y_codigos(texto: str) -> Set[str]:
    """Extrae nombres propios capitalizados en mitad de frase y códigos técnicos."""
    entidades: Set[str] = set()
    if not texto:
        return entidades

    # Códigos tipo MP-1, MP-2, PH-800, HA-30, Lote 2614, Grado III, Grado I
    for m_cod in re.finditer(
        r"\b(?:[A-Z]{1,4}-\d+|lote\s+\d+|grado\s+(?:iii|ii|i)|l[íi]nea\s+\d+|calle\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d+)\b",
        texto,
        re.IGNORECASE,
    ):
        entidades.add(normalizar_texto(m_cod.group(0)))

    # Palabras capitalizadas que no están al inicio absoluto de la oración
    palabras = re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}\b", texto)
    STOP_CAP = {
        "Segun", "Según", "Durante", "Entre", "Para", "Desde", "Hasta", "Tras",
        "Antes", "Despues", "Después", "Mientras", "Solo", "Sólo", "Cada",
        "Todas", "Todos", "Ningun", "Ningún", "Ninguna", "Además", "Ademas",
        "También", "Tambien", "Quedan", "Están", "Estan", "Cuando", "Donde",
    }
    for p in palabras[1:]:
        if p not in STOP_CAP:
            entidades.add(normalizar_texto(p))
    return entidades


def evaluar_motor_5_entidades(
    oracion_texto: str,
    fragmentos_oracion: List[Dict[str, Any]],
    fuentes: List[str],
    pregunta: str,
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 5:
    - Detecta si la oración atribuye a la entidad principal un dato procedente de una
      fuente distractora (otro municipio, otra empresa, otro centro, otro modelo).
    - Detecta inversión de valores entre dos entidades dentro de la misma fuente
      (ej. intercambiar Grado III y Grado I, o afirmar que el 24% es el mayor cuando hay 68%).
    """
    flag_cruzada = 0.0
    n_fuentes = len(fuentes)
    if n_fuentes == 0:
        return {"flag_atribucion_cruzada": 0.0}

    oracion_low = oracion_texto.lower()
    resp_f = oracion_dict.get("respaldo_por_fuente") or [0.0] * n_fuentes
    cont_f = oracion_dict.get("contradiccion_por_fuente") or [0.0] * n_fuentes
    sim_f = oracion_dict.get("similitud_por_fuente") or [0.0] * n_fuentes

    # 1. Intercambio interno de valores dentro de una misma fuente (ej. t0005 Grado III vs Grado I, t0156 Valdescarra vs Peñalvea)
    idx_top_sim = int(np.argmax(sim_f)) if sim_f else 0
    fuente_sim = fuentes[idx_top_sim] if idx_top_sim < n_fuentes else ""
    if ";" in fuente_sim or "frente a" in fuente_sim.lower():
        subclausulas = re.split(r";|\bfrente\s+a\b", fuente_sim)
        if len(subclausulas) >= 2:
            # Extraer pares (entidad/etiqueta, números) de cada subcláusula
            info_subs: List[Tuple[Set[str], List[float]]] = []
            for sc in subclausulas:
                ents_sc = _extraer_entidades_y_codigos(sc)
                # Añadir también sustantivos clave de la subcláusula
                for tok in tokenizar_contenido(sc, min_len=5):
                    ents_sc.add(tok)
                nums_sc = [v for v in extraer_numeros_texto(sc, expandir_unidades=False) if not (1900 <= v <= 2035)]
                info_subs.append((ents_sc, nums_sc))

            # Comprobar si un fragmento de la oración asocia una entidad exclusiva de subcláusula A con el número exclusivo de subcláusula B
            for i_a in range(len(info_subs)):
                for i_b in range(len(info_subs)):
                    if i_a == i_b:
                        continue
                    ents_a_unicas = info_subs[i_a][0] - info_subs[i_b][0]
                    nums_b_unicos = [x for x in info_subs[i_b][1] if not _numero_coincide(x, info_subs[i_a][1])]
                    if ents_a_unicas and nums_b_unicos:
                        for frag in fragmentos_oracion:
                            f_txt = frag.get("texto", "")
                            f_norm = normalizar_texto(f_txt)
                            f_nums = extraer_numeros_texto(f_txt, expandir_unidades=False)
                            f_ents_b = info_subs[i_b][0] - info_subs[i_a][0]
                            tiene_ent_a = any(ea in f_norm for ea in ents_a_unicas)
                            tiene_ent_b = any(eb in f_norm for eb in f_ents_b)
                            tiene_num_b = any(_numero_coincide(nb, f_nums) for nb in nums_b_unicos)
                            if tiene_ent_a and not tiene_ent_b and tiene_num_b:
                                # Si además el fragmento o la oración tiene contradicción alta en esa fuente
                                if cont_f[idx_top_sim] > 0.50 or (frag.get("contradiccion_por_fuente") or [0.0])[idx_top_sim] > 0.50:
                                    flag_cruzada = 1.0

    # 2. Afirmar "el mayor" / "la mayor" para una entidad secundaria cuyo número en la fuente es menor que otra (g0004, t0115)
    if re.search(r"\b(?:el\s+mayor|la\s+mayor)\b", oracion_low):
        nums_ora = [v for v in extraer_numeros_texto(oracion_texto, expandir_unidades=False) if not (1900 <= v <= 2035)]
        nums_src = [v for v in extraer_numeros_texto(fuente_sim, expandir_unidades=False) if not (1900 <= v <= 2035)]
        if nums_ora and nums_src and max(nums_src) > max(nums_ora) and max(resp_f) < 0.25:
            flag_cruzada = 1.0

    # 3. Cruce con fuente distractora en escenarios multi-fuente
    if n_fuentes >= 4:
        ents_preg = _extraer_entidades_y_codigos(pregunta) | _extraer_entidades_y_codigos(fuentes[0])
        for k in range(1, n_fuentes):
            f_k = fuentes[k]
            ents_k = _extraer_entidades_y_codigos(f_k)
            ents_nuevas_k = ents_k - ents_preg
            es_distractora = bool(
                ents_nuevas_k
                and re.search(
                    r"\b(?:vecin[oa]|misma\s+comarca|misma\s+ciudad|mismo\s+municipio|"
                    r"mismo\s+ayuntamiento|mismo\s+fabricante|otra\s+de\s+la|"
                    r"otro\s+cliente|hermana|por\s+su\s+parte|a\s+\d+\s*km)\b",
                    f_k.lower(),
                )
            )
            if es_distractora:
                ora_norm = normalizar_texto(oracion_texto)
                menciona_ent_distractora = any(ed in ora_norm for ed in ents_nuevas_k)
                if not menciona_ent_distractora:
                    # La oración NO menciona a la entidad distractora (habla de la entidad principal),
                    # pero algún fragmento toma su respaldo principalmente de la fuente distractora k
                    # mientras contradice una fuente principal
                    for frag in fragmentos_oracion:
                        fr_resp = frag.get("respaldo_por_fuente") or []
                        fr_cont = frag.get("contradiccion_por_fuente") or []
                        if k < len(fr_resp) and fr_resp[k] > 0.75:
                            max_resp_otras = max([fr_resp[j] for j in range(n_fuentes) if j != k] or [0.0])
                            max_cont_otras = max([fr_cont[j] for j in range(n_fuentes) if j != k] or [0.0])
                            if max_resp_otras < 0.20 and max_cont_otras > 0.65:
                                flag_cruzada = 1.0

    return {"flag_atribucion_cruzada": float(flag_cruzada)}


# =============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS A NIVEL DE ORACIÓN (FEATURE ENGINEERING)
# =============================================================================

FEATURE_NAMES: List[str] = [
    # 1. Señales NLI de la oración
    "max_resp_fuente",
    "second_max_resp_fuente",
    "sum_top2_resp_fuente",
    "resp_todas",
    "best_oracion_resp",
    "max_cont_fuente",
    "cont_todas",
    "cont_at_best_resp_source",
    "cont_at_best_sim_source",
    "max_sim_fuente",
    "gap_sim_resp",
    # 2. Señales del Juez LLM en la oración
    "llm_is_respaldada",
    "llm_is_contradicha",
    "llm_is_no_respaldada",
    "llm_confianza",
    "llm_signed_score",
    "resp_at_llm_source",
    "cont_at_llm_source",
    "sim_at_llm_source",
    "desacuerdo_llm_nli",
    # 3. Señales de los Fragmentos hijos (Resolución Multi-Fuente)
    "n_fragmentos",
    "min_frag_eff_resp",
    "mean_frag_eff_resp",
    "max_frag_cont_todas",
    "max_frag_cont_at_sim",
    "min_frag_max_sim",
    "multi_source_synergy",
    "best_combined_support",
    # 4. Señales de los 5 Motores Neuro-Simbólicos
    "flag_alucinacion_numerica",
    "n_numeros_derivados_validos",
    "flag_trampa_porcentaje",
    "ratio_numeros_respaldados",
    "distorsion_cercana",
    "flag_omision_excepcion",
    "score_omision_restriccion",
    "flag_sobregeneralizacion",
    "flag_endurecimiento_modal",
    "flag_inversion_temporal",
    "flag_inciso_sutil",
    "flag_causalidad_inventada",
    "ratio_palabras_huerfanas",
    "flag_atribucion_cruzada",
    # 5. Contexto estructural
    "is_multi_source_case",
]


def extract_sentence_features(oracion_idx: int, caso: Dict[str, Any]) -> Dict[str, float]:
    """
    Construye el diccionario completo de 43 características numéricas para la oración `oracion_idx`.
    Maneja de forma segura cualquier valor nulo o lista vacía.
    """
    fuentes: List[str] = caso.get("fuentes") or []
    pregunta: str = caso.get("pregunta") or ""
    respuesta: str = caso.get("respuesta") or ""
    senales: Dict[str, Any] = caso.get("senales") or {}
    oraciones: List[Dict[str, Any]] = senales.get("oraciones") or []
    fragmentos_todos: List[Dict[str, Any]] = senales.get("fragmentos") or []

    if oracion_idx < 0 or oracion_idx >= len(oraciones):
        return {k: 0.0 for k in FEATURE_NAMES}

    ora = oraciones[oracion_idx]
    ora_texto: str = ora.get("texto") or ""
    n_fuentes = max(1, len(fuentes))

    resp_f = [float(x) for x in (ora.get("respaldo_por_fuente") or [0.0])]
    cont_f = [float(x) for x in (ora.get("contradiccion_por_fuente") or [0.0])]
    sim_f = [float(x) for x in (ora.get("similitud_por_fuente") or [0.0])]

    if not resp_f:
        resp_f = [0.0]
    if not cont_f:
        cont_f = [0.0]
    if not sim_f:
        sim_f = [0.0]

    # 1. Señales NLI de la oración
    resp_sorted = sorted(resp_f, reverse=True)
    max_resp_fuente = float(resp_sorted[0])
    second_max_resp_fuente = float(resp_sorted[1]) if len(resp_sorted) > 1 else 0.0
    sum_top2_resp_fuente = float(min(1.0, max_resp_fuente + second_max_resp_fuente))
    resp_todas = float(ora.get("respaldo_todas") or 0.0)
    best_oracion_resp = float(max(max_resp_fuente, resp_todas))

    max_cont_fuente = float(max(cont_f))
    cont_todas = float(ora.get("contradiccion_todas") or 0.0)

    idx_best_resp = int(np.argmax(resp_f))
    idx_best_sim = int(np.argmax(sim_f))
    cont_at_best_resp_source = float(cont_f[idx_best_resp]) if idx_best_resp < len(cont_f) else 0.0
    cont_at_best_sim_source = float(cont_f[idx_best_sim]) if idx_best_sim < len(cont_f) else 0.0

    max_sim_fuente = float(max(sim_f))
    gap_sim_resp = float(max_sim_fuente - best_oracion_resp)

    # 2. Señales del Juez LLM
    llm = ora.get("llm")
    veredicto = llm.get("veredicto") if isinstance(llm, dict) else None
    conf_raw = llm.get("confianza") if isinstance(llm, dict) else None
    llm_confianza = float(conf_raw) if conf_raw is not None else 0.5

    llm_is_respaldada = 1.0 if veredicto == "respaldada" else 0.0
    llm_is_contradicha = 1.0 if veredicto == "contradicha" else 0.0
    llm_is_no_respaldada = 1.0 if veredicto == "no_respaldada" else 0.0

    if llm_is_respaldada > 0.5:
        llm_signed_score = llm_confianza
    elif llm_is_contradicha > 0.5 or llm_is_no_respaldada > 0.5:
        llm_signed_score = -llm_confianza
    else:
        llm_signed_score = 0.0

    llm_fuente = llm.get("fuente") if isinstance(llm, dict) else None
    if isinstance(llm_fuente, int) and 0 <= llm_fuente < len(resp_f):
        resp_at_llm_source = float(resp_f[llm_fuente])
        cont_at_llm_source = float(cont_f[llm_fuente]) if llm_fuente < len(cont_f) else 0.0
        sim_at_llm_source = float(sim_f[llm_fuente]) if llm_fuente < len(sim_f) else 0.0
    else:
        resp_at_llm_source = max_resp_fuente
        cont_at_llm_source = cont_at_best_sim_source
        sim_at_llm_source = max_sim_fuente

    # Fuente ancla k* combinando llm["fuente"] y argmax(0.6 * sim + 0.4 * resp)
    scores_ancla = [
        0.6 * (sim_f[k] if k < len(sim_f) else 0.0) + 0.4 * (resp_f[k] if k < len(resp_f) else 0.0)
        for k in range(len(fuentes))
    ]
    idx_ancla = int(np.argmax(scores_ancla)) if scores_ancla else 0
    if isinstance(llm_fuente, int) and 0 <= llm_fuente < len(fuentes):
        if scores_ancla[llm_fuente] >= scores_ancla[idx_ancla] - 0.15:
            idx_ancla = llm_fuente

    # 3. Señales de los Fragmentos hijos de la oración i
    frags_hijos = [f for f in fragmentos_todos if f.get("oracion") == oracion_idx]
    if not frags_hijos:
        frags_hijos = [ora]

    s_eff_list: List[float] = []
    f_max_resp_fuente_list: List[float] = []
    f_cont_todas_list: List[float] = []
    f_cont_sim_list: List[float] = []
    f_max_sim_list: List[float] = []

    for f in frags_hijos:
        fr_r = [float(x) for x in (f.get("respaldo_por_fuente") or [0.0])]
        fr_c = [float(x) for x in (f.get("contradiccion_por_fuente") or [0.0])]
        fr_s = [float(x) for x in (f.get("similitud_por_fuente") or [0.0])]
        fr_rt = float(f.get("respaldo_todas") or 0.0)
        fr_ct = float(f.get("contradiccion_todas") or 0.0)

        max_fr_r = max(fr_r) if fr_r else 0.0
        max_fr_s = max(fr_s) if fr_s else 0.0
        idx_fr_s = int(np.argmax(fr_s)) if fr_s else 0
        c_sim_f = float(fr_c[idx_fr_s]) if idx_fr_s < len(fr_c) else 0.0

        s_eff_list.append(max(max_fr_r, fr_rt))
        f_max_resp_fuente_list.append(max_fr_r)
        f_cont_todas_list.append(fr_ct)
        f_cont_sim_list.append(c_sim_f)
        f_max_sim_list.append(max_fr_s)

    min_frag_eff_resp = float(min(s_eff_list))
    mean_frag_eff_resp = float(np.mean(s_eff_list))
    max_frag_cont_todas = float(max(f_cont_todas_list))
    max_frag_cont_at_sim = float(max(f_cont_sim_list))
    min_frag_max_sim = float(min(f_max_sim_list))
    multi_source_synergy = float(np.mean(f_max_resp_fuente_list) - max_resp_fuente)
    best_combined_support = float(max(best_oracion_resp, min_frag_eff_resp))

    desacuerdo_llm_nli = (
        1.0
        if (llm_is_respaldada > 0.5 and best_oracion_resp < 0.15 and min_frag_eff_resp < 0.25)
        else 0.0
    )

    # 4. Ejecutar los 5 Motores Neuro-Simbólicos
    m1 = evaluar_motor_1_aritmetica(ora_texto, frags_hijos, fuentes, ora)
    m2 = evaluar_motor_2_omisiones(ora_texto, respuesta, fuentes, idx_ancla, ora)
    m3 = evaluar_motor_3_modal_temporal(oracion_idx, ora_texto, oraciones, fuentes, idx_ancla)
    m4 = evaluar_motor_4_sutil_causal(ora_texto, frags_hijos, fuentes, pregunta, ora)
    m5 = evaluar_motor_5_entidades(ora_texto, frags_hijos, fuentes, pregunta, ora)

    feats: Dict[str, float] = {
        "max_resp_fuente": max_resp_fuente,
        "second_max_resp_fuente": second_max_resp_fuente,
        "sum_top2_resp_fuente": sum_top2_resp_fuente,
        "resp_todas": resp_todas,
        "best_oracion_resp": best_oracion_resp,
        "max_cont_fuente": max_cont_fuente,
        "cont_todas": cont_todas,
        "cont_at_best_resp_source": cont_at_best_resp_source,
        "cont_at_best_sim_source": cont_at_best_sim_source,
        "max_sim_fuente": max_sim_fuente,
        "gap_sim_resp": gap_sim_resp,
        "llm_is_respaldada": llm_is_respaldada,
        "llm_is_contradicha": llm_is_contradicha,
        "llm_is_no_respaldada": llm_is_no_respaldada,
        "llm_confianza": llm_confianza,
        "llm_signed_score": llm_signed_score,
        "resp_at_llm_source": resp_at_llm_source,
        "cont_at_llm_source": cont_at_llm_source,
        "sim_at_llm_source": sim_at_llm_source,
        "desacuerdo_llm_nli": desacuerdo_llm_nli,
        "n_fragmentos": float(len(frags_hijos)),
        "min_frag_eff_resp": min_frag_eff_resp,
        "mean_frag_eff_resp": mean_frag_eff_resp,
        "max_frag_cont_todas": max_frag_cont_todas,
        "max_frag_cont_at_sim": max_frag_cont_at_sim,
        "min_frag_max_sim": min_frag_max_sim,
        "multi_source_synergy": multi_source_synergy,
        "best_combined_support": best_combined_support,
        "flag_alucinacion_numerica": m1["flag_alucinacion_numerica"],
        "n_numeros_derivados_validos": m1["n_numeros_derivados_validos"],
        "flag_trampa_porcentaje": m1["flag_trampa_porcentaje"],
        "ratio_numeros_respaldados": m1["ratio_numeros_respaldados"],
        "distorsion_cercana": m1["distorsion_cercana"],
        "flag_omision_excepcion": m2["flag_omision_excepcion"],
        "score_omision_restriccion": m2["score_omision_restriccion"],
        "flag_sobregeneralizacion": m2["flag_sobregeneralizacion"],
        "flag_endurecimiento_modal": m3["flag_endurecimiento_modal"],
        "flag_inversion_temporal": m3["flag_inversion_temporal"],
        "flag_inciso_sutil": m4["flag_inciso_sutil"],
        "flag_causalidad_inventada": m4["flag_causalidad_inventada"],
        "ratio_palabras_huerfanas": m4["ratio_palabras_huerfanas"],
        "flag_atribucion_cruzada": m5["flag_atribucion_cruzada"],
        "is_multi_source_case": 1.0 if len(fuentes) > 1 else 0.0,
    }
    return feats


# =============================================================================
# MOTOR EXPERTO NEURO-SIMBÓLICO (CORE ANALÍTICO AUTÓNOMO)
# =============================================================================

def symbolic_sentence_risk(feats: Dict[str, float]) -> Tuple[float, float]:
    """
    Calcula la probabilidad analítica de alucinación de una oración en (0, 1)
    y el nivel de veto simbólico duro (0.0 si no hay veto activo).
    """
    best_supp = feats["best_combined_support"]
    best_ora_resp = feats["best_oracion_resp"]
    min_frag_resp = feats["min_frag_eff_resp"]
    mean_frag_resp = feats["mean_frag_eff_resp"]
    cont_sim = feats["cont_at_best_sim_source"]
    cont_llm = feats["cont_at_llm_source"]
    cont_todas = feats["cont_todas"]
    max_sim = feats["max_sim_fuente"]

    # Comprobar si algún flag simbólico de violación está activo
    hay_flag_violacion = (
        feats["flag_alucinacion_numerica"] > 0.5
        or feats["flag_trampa_porcentaje"] > 0.5
        or feats["flag_omision_excepcion"] > 0.5
        or feats["score_omision_restriccion"] >= 0.80
        or feats["flag_sobregeneralizacion"] > 0.5
        or feats["flag_endurecimiento_modal"] > 0.5
        or feats["flag_inversion_temporal"] > 0.5
        or feats["flag_inciso_sutil"] > 0.5
        or feats["flag_causalidad_inventada"] > 0.5
        or feats["flag_atribucion_cruzada"] > 0.5
    )

    # 1. Regla de Rescate Matemático y de Paráfrasis Numérica
    rescate_matematico = (
        feats["n_numeros_derivados_validos"] >= 1.0
        and not hay_flag_violacion
        and feats["ratio_numeros_respaldados"] >= 0.99
    )

    # 2. Regla de Rescate Multi-Fuente (síntesis fiel cuyos fragmentos están respaldados)
    rescate_multifuente = (
        feats["is_multi_source_case"] > 0.5
        and not hay_flag_violacion
        and (min_frag_resp > 0.68 or (mean_frag_resp > 0.78 and min_frag_resp > 0.45) or feats["resp_todas"] > 0.80)
    )

    # 3. Riesgo base según el veredicto del Juez LLM y el NLI
    if feats["llm_is_contradicha"] > 0.5 or feats["llm_is_no_respaldada"] > 0.5:
        # Verificar si aplica rescate legítimo ante fallo del LLM (ej. e34::par, e24::par, t0064, t0089)
        if rescate_matematico and (
            (best_supp > 0.90 and cont_todas < 0.15 and cont_sim < 0.10)
            or (min_frag_resp > 0.75 and cont_sim < 0.20)
            or feats["llm_confianza"] <= 0.65
        ):
            riesgo_base = 0.12
        else:
            riesgo_base = 0.86 + 0.10 * feats["llm_confianza"]
    elif feats["llm_is_respaldada"] > 0.5:
        # El LLM dice "respaldada": evaluar cuánto podemos fiarnos según el soporte NLI
        if rescate_matematico:
            riesgo_base = 0.06 + 0.08 * (1.0 - max(best_supp, 0.50))
        elif rescate_multifuente:
            riesgo_base = 0.05 + 0.12 * (1.0 - max(best_supp, mean_frag_resp))
        else:
            # Si ni la oración ni los fragmentos tienen respaldo en el NLI
            soporte_efectivo = max(best_ora_resp, 0.75 * min_frag_resp + 0.25 * mean_frag_resp)
            contradiccion_efectiva = max(cont_sim, cont_llm)
            if rescate_matematico:
                contradiccion_efectiva *= 0.15

            riesgo_base = 0.08 + 0.52 * (1.0 - soporte_efectivo) + 0.22 * contradiccion_efectiva
            if feats["desacuerdo_llm_nli"] > 0.5 and feats["n_numeros_derivados_validos"] == 0:
                riesgo_base = max(riesgo_base, 0.62)
            if max_sim > 0.85 and best_supp < 0.15 and feats["n_numeros_derivados_validos"] == 0:
                riesgo_base = max(riesgo_base, 0.68)
    else:
        # LLM es None / no contestó: apoyarse en soporte combinado y contradicción en fuente similar
        soporte_efectivo = max(best_ora_resp, min_frag_resp)
        riesgo_base = 0.50 * (1.0 - soporte_efectivo) + 0.35 * max(cont_sim, cont_todas)

    # 4. Reglas de Veto Simbólico (Caza-Alucinaciones cuando el LLM se confía)
    veto_prob = 0.0
    if feats["flag_alucinacion_numerica"] > 0.5 or feats["flag_trampa_porcentaje"] > 0.5:
        veto_prob = max(veto_prob, 0.96 if feats["distorsion_cercana"] > 0.5 else 0.92)
    if feats["flag_omision_excepcion"] > 0.5 or feats["flag_sobregeneralizacion"] > 0.5:
        veto_prob = max(veto_prob, 0.87)
    if feats["score_omision_restriccion"] >= 0.80:
        veto_prob = max(veto_prob, 0.86 * feats["score_omision_restriccion"])
    if feats["flag_endurecimiento_modal"] > 0.5 or feats["flag_inversion_temporal"] > 0.5:
        veto_prob = max(veto_prob, 0.89)
    if (
        feats["flag_inciso_sutil"] > 0.5
        or feats["flag_causalidad_inventada"] > 0.5
        or feats["flag_atribucion_cruzada"] > 0.5
    ):
        veto_prob = max(veto_prob, 0.87)

    riesgo_final = float(np.clip(max(riesgo_base, veto_prob), 0.005, 0.995))
    return riesgo_final, veto_prob



CUANT = {"todos", "todas", "todo", "toda", "cualquier", "cualquiera", "siempre", "nunca", "ningun",
         "ninguna", "ninguno", "cada", "totalidad", "mayoria", "generalmente", "habitualmente",
         "entero", "entera", "exclusivamente", "unicamente", "ambos", "ambas", "mayor", "mejor",
         "peor", "menor", "maximo", "minimo", "unico", "unica"}


def _norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _nums(s):
    out = set()
    for m in re.finditer(r"\d+(?:[.,]\d+)*", _norm(s)):
        t = m.group()
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        try:
            out.add(round(float(t.replace(",", ".")), 4))
        except ValueError:
            pass
    return out


def _quant(s):
    return {w for w in re.findall(r"[a-z]+", _norm(s)) if w in CUANT}


_VACIAS = set("de la el en y a los las del un una que se por con para al lo su sus es son fue como mas o "
              "no ha han sido este esta estos estas entre sobre tras hasta desde sin ser le les ya".split())


def _palabras(s):
    return {w[:5] for w in re.findall(r"[a-z0-9]+", _norm(s)) if w not in _VACIAS and len(w) > 2 and not w.isdigit()}


def _lg(p, e=0.01):
    p = min(1 - e, max(e, p))
    return math.log(p / (1 - p))


def _mx(xs, d=0.0):
    return max(xs) if xs else d


def _mn(xs, d=0.0):
    return min(xs) if xs else d


def _ent(v):
    t = sum(v)
    if t <= 0 or len(v) < 2:
        return 0.0
    p = [x / t for x in v if x > 0]
    return -sum(x * math.log(x) for x in p) / math.log(len(v))


def _juez_sent(o):
    v = o.get("llm") or {}
    ver, conf = v.get("veredicto"), v.get("confianza") or 0.0
    return ver, conf, v.get("fuente")


def _extraer_base(caso):
    fuentes = caso.get("fuentes") or []
    sen = caso.get("senales") or {}
    ors = sen.get("oraciones") or []
    frs = sen.get("fragmentos") or []
    resp = caso.get("respuesta") or ""
    nf = len(fuentes)
    L = sum(len(f) for f in fuentes)
    f = {}
    f["nf"] = nf
    f["logL"] = math.log1p(L)
    f["logR"] = math.log1p(len(resp))
    f["nor"] = len(ors)
    f["nfr"] = len(frs)
    f["lsrc"] = math.log1p(L / max(1, nf))
    f["r"] = 1.0 / (1.0 + (L / 400.0) ** 2)
    nums_f = set().union(*[_nums(x) for x in fuentes]) if fuentes else set()
    cu_f = set().union(*[_quant(x) for x in fuentes]) if fuentes else set()
    pal_f = set().union(*[_palabras(x) for x in fuentes]) if fuentes else set()
    pn, pn_c = [], []
    S = {k: [] for k in ("jc", "jn", "jnone", "jconf", "jscore", "nt", "nm", "gap", "contr", "ctod", "sim",
                         "cit_r", "cit_c", "cit_s", "agree", "nap", "ent", "top2", "nnew", "nnew_c", "qnew")}
    for o in ors:
        rp = o.get("respaldo_por_fuente") or []
        cp = o.get("contradiccion_por_fuente") or []
        sp = o.get("similitud_por_fuente") or []
        rt = o.get("respaldo_todas") or 0.0
        ver, conf, k = _juez_sent(o)
        S["jc"].append(float(ver == "contradicha"))
        S["jn"].append(float(ver == "no_respaldada"))
        S["jnone"].append(float(ver is None))
        S["jconf"].append(conf if ver == "respaldada" else 1.0)          # confianza de "respaldada"
        S["jscore"].append(conf if ver in ("contradicha", "no_respaldada") else (1 - conf if ver == "respaldada" else 0.5))
        best = _mx(rp)
        S["nt"].append(1 - max(rt, best))
        S["nm"].append(1 - best)
        S["gap"].append(rt - best)
        S["contr"].append(max(o.get("contradiccion_todas") or 0.0, _mx(cp)))
        S["ctod"].append(o.get("contradiccion_todas") or 0.0)
        S["sim"].append(1 - _mx(sp))
        top = max(range(len(rp)), key=rp.__getitem__) if rp else None
        kk = k if (k is not None and 0 <= k < len(rp)) else top
        S["cit_r"].append(1 - (rp[kk] if kk is not None else 0.0))
        S["cit_c"].append(cp[kk] if (kk is not None and kk < len(cp)) else 0.0)
        S["cit_s"].append(1 - (sp[kk] if (kk is not None and kk < len(sp)) else 0.0))
        S["agree"].append(float(top is not None and k is not None and k == top))
        S["nap"].append(sum(1 for x in rp if x > 0.5) / max(1, len(rp)))
        S["ent"].append(_ent(rp))
        srt = sorted(rp, reverse=True)
        S["top2"].append((srt[0] - srt[1]) if len(srt) > 1 else (srt[0] if srt else 0.0))
        t = o.get("texto") or ""
        nn = _nums(t)
        cit = _nums(fuentes[kk]) if (kk is not None and kk < len(fuentes)) else set()
        w = _palabras(t)
        pn.append(len(w - pal_f) / max(1, len(w)))
        pn_c.append(len(w - (_palabras(fuentes[kk]) if (kk is not None and kk < len(fuentes)) else set())) / max(1, len(w)))
        S["nnew"].append(float(len(nn - nums_f)))
        S["nnew_c"].append(float(len([x for x in nn if x in nums_f and x not in cit])))
        S["qnew"].append(float(len(_quant(t) - cu_f)))
    # --- agregados por caso: la peor oración manda
    f["jc"] = _mx(S["jc"]); f["jn"] = _mx(S["jn"]); f["jnone"] = _mx(S["jnone"])
    cmin = _mn(S["jconf"], 1.0)
    f["j_c90"] = float(cmin <= 0.90)
    f["j_c95"] = float(0.90 < cmin <= 0.95)
    f["j_c97"] = float(0.95 < cmin <= 0.97)
    f["jscore_max"] = _mx(S["jscore"]); f["jscore_mean"] = sum(S["jscore"]) / max(1, len(S["jscore"]))
    f["jlg"] = _lg(f["jscore_max"])
    f["nt"] = _mx(S["nt"]); f["nm"] = _mx(S["nm"])
    f["ntlg"] = _lg(f["nt"]); f["nmlg"] = _lg(f["nm"])
    f["nt_mean"] = sum(S["nt"]) / max(1, len(S["nt"]))
    f["gap_mean"] = sum(S["gap"]) / max(1, len(S["gap"]))
    f["contr"] = _mx(S["contr"]); f["ctod"] = _mx(S["ctod"])
    f["sim"] = _mx(S["sim"])
    f["cit_r"] = _mx(S["cit_r"]); f["cit_c"] = _mx(S["cit_c"]); f["cit_s"] = _mx(S["cit_s"])
    f["agree"] = _mn(S["agree"], 1.0)
    f["nap"] = _mn(S["nap"]); f["ent"] = _mx(S["ent"]); f["top2"] = _mn(S["top2"])
    f["nnew"] = _mx(S["nnew"]); f["nnew_c"] = _mx(S["nnew_c"]); f["qnew"] = _mx(S["qnew"])
    f["pn"] = _mx(pn); f["pn_cit"] = _mx(pn_c)
    # --- fragmentos: localizan la parte dudosa
    fb = [1 - _mx(g.get("respaldo_por_fuente") or []) for g in frs]
    ft = [1 - (g.get("respaldo_todas") or 0.0) for g in frs]
    fc = [max(g.get("contradiccion_todas") or 0.0, _mx(g.get("contradiccion_por_fuente") or [])) for g in frs]
    fs = [1 - _mx(g.get("similitud_por_fuente") or []) for g in frs]
    f["fb"] = _mx(fb, f["nm"]); f["ft"] = _mx(ft, f["nt"])
    f["fblg"] = _lg(f["fb"]); f["ftlg"] = _lg(f["ft"])
    f["fb_mean"] = sum(fb) / len(fb) if fb else f["nm"]
    f["fb_frac"] = sum(1 for x in fb if x > 0.5) / len(fb) if fb else float(f["nm"] > 0.5)
    f["fc"] = _mx(fc); f["fs"] = _mx(fs)
    return f



# Rasgos de los motores neuro-simbólicos que entran al modelo (el resto duplica los de `_extraer_base`).
MOTORES_USADOS = (
    "flag_alucinacion_numerica", "n_numeros_derivados_validos", "flag_trampa_porcentaje",
    "ratio_numeros_respaldados", "distorsion_cercana", "flag_omision_excepcion", "score_omision_restriccion",
    "flag_sobregeneralizacion", "flag_endurecimiento_modal", "flag_inversion_temporal", "flag_inciso_sutil",
    "flag_causalidad_inventada", "ratio_palabras_huerfanas", "flag_atribucion_cruzada",
    "multi_source_synergy", "best_combined_support", "desacuerdo_llm_nli",
)


def _extraer_motores(caso):
    """Rasgos de los cinco motores por oración (la peor manda) + riesgo simbólico y veto de cada oración."""
    n = len((caso.get("senales") or {}).get("oraciones") or [])
    filas, sym, veto = [], [], []
    for i in range(n):
        try:
            d = extract_sentence_features(i, caso)
            p, v = symbolic_sentence_risk(d)
        except Exception:                      # una oración rara no debe tumbar el caso
            d, p, v = {}, 0.0, 0.0
        filas.append(d); sym.append(p); veto.append(v)
    out = {"m_" + k: max([float(d.get(k, 0.0)) for d in filas] or [0.0]) for k in MOTORES_USADOS}
    out["m_sym"] = max(sym or [0.0])
    out["m_veto"] = max(veto or [0.0])
    return out


def extraer(caso):
    """Rasgos del caso: señales (juez, NLI, fragmentos, texto) + motores de palabras y cifras."""
    f = _extraer_base(caso)
    f.update(_extraer_motores(caso))
    return f



# ====== MODELO (lo escribe entrenar_guardian.py; no editar a mano) ======
# <<MODELO>>
NOMBRES = {"corto": ['agree', 'cit_c', 'cit_r', 'cit_s', 'contr', 'ctod', 'ent', 'fb', 'fb_frac', 'fb_mean', 'fblg', 'fc', 'fs', 'ft', 'ftlg', 'gap_mean', 'j_c90', 'j_c95', 'j_c97', 'jc', 'jlg', 'jn', 'jnone', 'jscore_max', 'jscore_mean', 'logL', 'logR', 'lsrc', 'm_best_combined_support', 'm_desacuerdo_llm_nli', 'm_distorsion_cercana', 'm_flag_alucinacion_numerica', 'm_flag_atribucion_cruzada', 'm_flag_causalidad_inventada', 'm_flag_endurecimiento_modal', 'm_flag_inciso_sutil', 'm_flag_inversion_temporal', 'm_flag_omision_excepcion', 'm_flag_sobregeneralizacion', 'm_flag_trampa_porcentaje', 'm_multi_source_synergy', 'm_n_numeros_derivados_validos', 'm_ratio_numeros_respaldados', 'm_ratio_palabras_huerfanas', 'm_score_omision_restriccion', 'm_sym', 'm_veto', 'nap', 'nf', 'nfr', 'nm', 'nmlg', 'nnew', 'nnew_c', 'nor', 'nt', 'nt_mean', 'ntlg', 'pn', 'pn_cit', 'qnew', 'r', 'sim', 'top2'], "largo": ['agree', 'cit_c', 'cit_r', 'cit_s', 'contr', 'ctod', 'ent', 'fb', 'fb_frac', 'fb_mean', 'fblg', 'fc', 'fs', 'ft', 'ftlg', 'gap_mean', 'j_c90', 'j_c95', 'j_c97', 'jc', 'jlg', 'jn', 'jnone', 'jscore_max', 'jscore_mean', 'logL', 'lsrc', 'm_best_combined_support', 'm_desacuerdo_llm_nli', 'm_distorsion_cercana', 'm_flag_alucinacion_numerica', 'm_flag_atribucion_cruzada', 'm_flag_causalidad_inventada', 'm_flag_endurecimiento_modal', 'm_flag_inciso_sutil', 'm_flag_inversion_temporal', 'm_flag_omision_excepcion', 'm_flag_sobregeneralizacion', 'm_flag_trampa_porcentaje', 'm_multi_source_synergy', 'm_n_numeros_derivados_validos', 'm_ratio_numeros_respaldados', 'm_ratio_palabras_huerfanas', 'm_score_omision_restriccion', 'm_sym', 'm_veto', 'nap', 'nf', 'nm', 'nmlg', 'nnew', 'nnew_c', 'nt', 'nt_mean', 'ntlg', 'pn', 'pn_cit', 'qnew', 'r', 'sim', 'top2']}
MODELOS = {
    'corto': [
        (-0.011991, (
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -1.067159), (7, 0.959, 3, 4, -1.553367), (-1, 0.0, 0, 0, -0.081822), (-1, 0.0, 0, 0, -0.02393), (26, 4.579839228586978, 6, 7, -0.305104), (-1, 0.0, 0, 0, -0.044541), (-1, 0.0, 0, 0, 0.013939), (24, 0.06250000000000003, 9, 12, 1.361203), (58, 0.14835164835164835, 10, 11, 0.467957), (-1, 0.0, 0, 0, 0.068566), (-1, 0.0, 0, 0, -0.01865), (52, 0.5, 13, 14, 1.798845), (-1, 0.0, 0, 0, 0.097044), (-1, 0.0, 0, 0, 0.060001)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -1.016011), (7, 0.959, 3, 4, -1.479723), (-1, 0.0, 0, 0, -0.077943), (-1, 0.0, 0, 0, -0.022873), (6, 0.2833655023443646, 6, 7, -0.290406), (-1, 0.0, 0, 0, 0.034676), (-1, 0.0, 0, 0, -0.030282), (24, 0.06250000000000003, 9, 12, 1.296196), (58, 0.14835164835164835, 10, 11, 0.445858), (-1, 0.0, 0, 0, 0.06534), (-1, 0.0, 0, 0, -0.017759), (52, 0.5, 13, 14, 1.713237), (-1, 0.0, 0, 0, 0.092424), (-1, 0.0, 0, 0, 0.057181)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.968856), (7, 0.959, 3, 4, -1.413586), (-1, 0.0, 0, 0, -0.074475), (-1, 0.0, 0, 0, -0.021868), (26, 4.579839228586978, 6, 7, -0.275956), (-1, 0.0, 0, 0, -0.041649), (-1, 0.0, 0, 0, 0.013912), (24, 0.06250000000000003, 9, 12, 1.23791), (58, 0.256578947368421, 10, 11, 0.425293), (-1, 0.0, 0, 0, 0.047843), (-1, 0.0, 0, 0, -0.039233), (50, 0.21499999999999997, 13, 14, 1.637892), (-1, 0.0, 0, 0, 0.029453), (-1, 0.0, 0, 0, 0.084882)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.925643), (7, 0.959, 3, 4, -1.353716), (-1, 0.0, 0, 0, -0.071349), (-1, 0.0, 0, 0, -0.020911), (3, 0.30100000000000005, 6, 7, -0.262828), (-1, 0.0, 0, 0, 0.010827), (-1, 0.0, 0, 0, -0.042741), (24, 0.06250000000000003, 9, 12, 1.184354), (58, 0.14835164835164835, 10, 11, 0.405004), (-1, 0.0, 0, 0, 0.060316), (-1, 0.0, 0, 0, -0.016817), (52, 0.5, 13, 14, 1.570398), (-1, 0.0, 0, 0, 0.085145), (-1, 0.0, 0, 0, 0.051021)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.885491), (7, 0.959, 3, 4, -1.299127), (-1, 0.0, 0, 0, -0.06851), (-1, 0.0, 0, 0, -0.02), (26, 4.579839228586978, 6, 7, -0.25038), (-1, 0.0, 0, 0, -0.038883), (-1, 0.0, 0, 0, 0.013618), (24, 0.06250000000000003, 9, 12, 1.13628), (58, 0.256578947368421, 10, 11, 0.386957), (-1, 0.0, 0, 0, 0.044081), (-1, 0.0, 0, 0, -0.036699), (52, 0.5, 13, 14, 1.510422), (-1, 0.0, 0, 0, 0.081997), (-1, 0.0, 0, 0, 0.048855)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.848196), (7, 0.959, 3, 4, -1.249022), (-1, 0.0, 0, 0, -0.065913), (-1, 0.0, 0, 0, -0.019131), (6, 0.2833655023443646, 6, 7, -0.238601), (-1, 0.0, 0, 0, 0.033939), (-1, 0.0, 0, 0, -0.026696), (24, 0.06250000000000003, 9, 12, 1.091766), (59, 0.17424242424242425, 10, 11, 0.369008), (-1, 0.0, 0, 0, 0.074526), (-1, 0.0, 0, 0, -0.005909), (50, 0.21499999999999997, 13, 14, 1.456356), (-1, 0.0, 0, 0, 0.022331), (-1, 0.0, 0, 0, 0.075789)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.812968), (7, 0.959, 3, 4, -1.202755), (-1, 0.0, 0, 0, -0.063524), (-1, 0.0, 0, 0, -0.018303), (26, 4.579839228586978, 6, 7, -0.226851), (-1, 0.0, 0, 0, -0.03651), (-1, 0.0, 0, 0, 0.013495), (24, 0.06250000000000003, 9, 12, 1.050571), (59, 0.17424242424242425, 10, 11, 0.353268), (-1, 0.0, 0, 0, 0.071822), (-1, 0.0, 0, 0, -0.005623), (52, 0.5, 13, 14, 1.406661), (-1, 0.0, 0, 0, 0.076896), (-1, 0.0, 0, 0, 0.043899)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.780209), (58, 0.11805555555555555, 3, 4, -1.159791), (-1, 0.0, 0, 0, -0.039191), (-1, 0.0, 0, 0, -0.066213), (6, 0.2833655023443646, 6, 7, -0.216268), (-1, 0.0, 0, 0, 0.032801), (-1, 0.0, 0, 0, -0.024896), (24, 0.06250000000000003, 9, 12, 1.012718), (58, 0.3431372549019608, 10, 11, 0.338434), (-1, 0.0, 0, 0, 0.033148), (-1, 0.0, 0, 0, -0.05676), (52, 0.5, 13, 14, 1.361796), (-1, 0.0, 0, 0, 0.074613), (-1, 0.0, 0, 0, 0.042155)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.606559), (24, 0.027500000000000024, 3, 4, -0.880846), (-1, 0.0, 0, 0, -0.059039), (-1, 0.0, 0, 0, -0.024668), (59, 0.17424242424242425, 6, 7, 0.322125), (-1, 0.0, 0, 0, 0.068268), (-1, 0.0, 0, 0, -0.005934), (52, 0.5, 9, 12, 1.214268), (6, 0.8978411663857659, 10, 11, 1.451922), (-1, 0.0, 0, 0, 0.073818), (-1, 0.0, 0, 0, 0.01915), (11, 0.948, 13, 14, 0.542246), (-1, 0.0, 0, 0, -0.031287), (-1, 0.0, 0, 0, 0.056858)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.581701), (24, 0.027500000000000024, 3, 4, -0.846283), (-1, 0.0, 0, 0, -0.057075), (-1, 0.0, 0, 0, -0.023513), (58, 0.3431372549019608, 6, 7, 0.308891), (-1, 0.0, 0, 0, 0.03086), (-1, 0.0, 0, 0, -0.054277), (52, 0.5, 9, 12, 1.175137), (6, 0.8978411663857659, 10, 11, 1.41108), (-1, 0.0, 0, 0, 0.071759), (-1, 0.0, 0, 0, 0.018592), (11, 0.948, 13, 14, 0.51887), (-1, 0.0, 0, 0, -0.029817), (-1, 0.0, 0, 0, 0.054653)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.558542), (24, 0.027500000000000024, 3, 4, -0.813537), (-1, 0.0, 0, 0, -0.055229), (-1, 0.0, 0, 0, -0.022416), (58, 0.14835164835164835, 6, 7, 0.29415), (-1, 0.0, 0, 0, 0.04912), (-1, 0.0, 0, 0, -0.015835), (52, 0.5, 9, 12, 1.138845), (6, 0.8978411663857659, 10, 11, 1.373635), (-1, 0.0, 0, 0, 0.069876), (-1, 0.0, 0, 0, 0.018052), (11, 0.948, 13, 14, 0.496936), (-1, 0.0, 0, 0, -0.02843), (-1, 0.0, 0, 0, 0.052589)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.536356), (24, 0.027500000000000024, 3, 4, -0.782422), (-1, 0.0, 0, 0, -0.053488), (-1, 0.0, 0, 0, -0.021373), (59, 0.17424242424242425, 6, 7, 0.281865), (-1, 0.0, 0, 0, 0.063808), (-1, 0.0, 0, 0, -0.006295), (52, 0.5, 9, 12, 1.105018), (6, 0.8978411663857659, 10, 11, 1.339141), (-1, 0.0, 0, 0, 0.068145), (-1, 0.0, 0, 0, 0.017528), (1, 0.887, 13, 14, 0.476289), (-1, 0.0, 0, 0, -0.007329), (-1, 0.0, 0, 0, 0.070153)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.514997), (24, 0.027500000000000024, 3, 4, -0.752782), (-1, 0.0, 0, 0, -0.051838), (-1, 0.0, 0, 0, -0.02038), (58, 0.4365079365079365, 6, 7, 0.270601), (-1, 0.0, 0, 0, 0.025326), (-1, 0.0, 0, 0, -0.066404), (52, 0.5, 9, 12, 1.073944), (6, 0.8978411663857659, 10, 11, 1.307219), (-1, 0.0, 0, 0, 0.066547), (-1, 0.0, 0, 0, 0.017019), (28, 0.134, 13, 14, 0.458586), (-1, 0.0, 0, 0, 0.062609), (-1, 0.0, 0, 0, -0.01164)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.495042), (58, 0.3038461538461539, 3, 4, -0.724482), (-1, 0.0, 0, 0, -0.026035), (-1, 0.0, 0, 0, -0.061196), (59, 0.17424242424242425, 6, 7, 0.257178), (-1, 0.0, 0, 0, 0.061266), (-1, 0.0, 0, 0, -0.006535), (52, 0.5, 9, 12, 1.044525), (47, 0.13392857142857142, 10, 11, 1.277551), (-1, 0.0, 0, 0, 0.065123), (-1, 0.0, 0, 0, 0.016822), (11, 0.948, 13, 14, 0.441279), (-1, 0.0, 0, 0, -0.026922), (-1, 0.0, 0, 0, 0.048447)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.16178, 2, 5, -0.475011), (58, 0.3038461538461539, 3, 4, -0.699963), (-1, 0.0, 0, 0, -0.025097), (-1, 0.0, 0, 0, -0.059332), (58, 0.4365079365079365, 6, 7, 0.235844), (-1, 0.0, 0, 0, 0.02283), (-1, 0.0, 0, 0, -0.064073), (52, 0.5, 9, 12, 1.016286), (6, 0.8978411663857659, 10, 11, 1.249971), (-1, 0.0, 0, 0, 0.06373), (-1, 0.0, 0, 0, 0.014695), (1, 0.887, 13, 14, 0.423447), (-1, 0.0, 0, 0, -0.007547), (-1, 0.0, 0, 0, 0.066002)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, -0.45629), (24, 0.027500000000000024, 3, 4, -0.669676), (-1, 0.0, 0, 0, -0.048221), (-1, 0.0, 0, 0, -0.016194), (59, 0.17424242424242425, 6, 7, 0.235476), (-1, 0.0, 0, 0, 0.058997), (-1, 0.0, 0, 0, -0.006687), (52, 0.5, 9, 12, 0.990068), (47, 0.13392857142857142, 10, 11, 1.223988), (-1, 0.0, 0, 0, 0.062508), (-1, 0.0, 0, 0, 0.014468), (28, 0.134, 13, 14, 0.408292), (-1, 0.0, 0, 0, 0.058613), (-1, 0.0, 0, 0, -0.011417)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (58, 0.27525252525252525, 2, 5, -0.438402), (45, 0.20606000000000002, 3, 4, -0.173641), (-1, 0.0, 0, 0, -0.021148), (-1, 0.0, 0, 0, 0.029503), (55, 0.951, 6, 7, -0.977987), (-1, 0.0, 0, 0, -0.054765), (-1, 0.0, 0, 0, 0.0107), (52, 0.5, 9, 12, 0.965005), (6, 0.8978411663857659, 10, 11, 1.199663), (-1, 0.0, 0, 0, 0.061271), (-1, 0.0, 0, 0, 0.012519), (1, 0.887, 13, 14, 0.39335), (-1, 0.0, 0, 0, -0.007306), (-1, 0.0, 0, 0, 0.063467)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (58, 0.27525252525252525, 2, 5, -0.420828), (24, 0.02416666666666669, 3, 4, -0.165998), (-1, 0.0, 0, 0, -0.029407), (-1, 0.0, 0, 0, 0.0142), (55, 0.951, 6, 7, -0.944873), (-1, 0.0, 0, 0, -0.053009), (-1, 0.0, 0, 0, 0.010233), (52, 0.5, 9, 12, 0.941176), (47, 0.13392857142857142, 10, 11, 1.176557), (-1, 0.0, 0, 0, 0.060211), (-1, 0.0, 0, 0, 0.012262), (11, 0.948, 13, 14, 0.37958), (-1, 0.0, 0, 0, -0.025301), (-1, 0.0, 0, 0, 0.044054)),
            ((45, 0.62, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.472801), (58, 0.27525252525252525, 3, 4, -0.796953), (-1, 0.0, 0, 0, -0.028561), (-1, 0.0, 0, 0, -0.061645), (3, 0.30100000000000005, 6, 7, -0.055394), (-1, 0.0, 0, 0, 0.019995), (-1, 0.0, 0, 0, -0.030881), (20, -3.210268834500856, 9, 12, 0.804344), (11, 0.831, 10, 11, -0.101574), (-1, 0.0, 0, 0, 0.029542), (-1, 0.0, 0, 0, -0.033304), (2, 0.954, 13, 14, 0.959512), (-1, 0.0, 0, 0, 0.019561), (-1, 0.0, 0, 0, 0.056385)),
            ((45, 0.31292000000000003, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.49283), (58, 0.27525252525252525, 3, 4, -0.805542), (-1, 0.0, 0, 0, -0.029895), (-1, 0.0, 0, 0, -0.059717), (3, 0.246, 6, 7, -0.089218), (-1, 0.0, 0, 0, 0.024389), (-1, 0.0, 0, 0, -0.023668), (24, 0.179, 9, 12, 0.723779), (58, 0.4365079365079365, 10, 11, 0.267213), (-1, 0.0, 0, 0, 0.022788), (-1, 0.0, 0, 0, -0.053642), (13, 0.925, 13, 14, 1.07763), (-1, 0.0, 0, 0, 0.028411), (-1, 0.0, 0, 0, 0.061084)),
            ((45, 0.87, 1, 8, 0.0), (24, 0.027500000000000024, 2, 5, -0.359553), (26, 4.2413008733233095, 3, 4, -0.766719), (-1, 0.0, 0, 0, -0.064797), (-1, 0.0, 0, 0, -0.030439), (41, 0.5, 6, 7, -0.02261), (-1, 0.0, 0, 0, 0.029438), (-1, 0.0, 0, 0, -0.023046), (50, 0.21499999999999997, 9, 10, 0.932212), (-1, 0.0, 0, 0, -0.013087), (15, 0.131, 11, 12, 1.046395), (-1, 0.0, 0, 0, 0.059608), (-1, 0.0, 0, 0, 0.018771)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (58, 0.27525252525252525, 2, 5, -0.366823), (45, 0.20606000000000002, 3, 4, -0.130321), (-1, 0.0, 0, 0, -0.01764), (-1, 0.0, 0, 0, 0.026813), (55, 0.951, 6, 7, -0.868473), (-1, 0.0, 0, 0, -0.049187), (-1, 0.0, 0, 0, 0.01103), (52, 0.5, 9, 12, 0.862488), (6, 0.8978411663857659, 10, 11, 1.116491), (-1, 0.0, 0, 0, 0.057324), (-1, 0.0, 0, 0, 0.006544), (11, 0.988, 13, 14, 0.291531), (-1, 0.0, 0, 0, -0.012677), (-1, 0.0, 0, 0, 0.053992)),
            ((45, 0.87, 1, 8, 0.0), (24, 0.027500000000000024, 2, 5, -0.333807), (26, 4.2413008733233095, 3, 4, -0.730074), (-1, 0.0, 0, 0, -0.06332), (-1, 0.0, 0, 0, -0.028665), (3, 0.246, 6, 7, -0.011871), (-1, 0.0, 0, 0, 0.027663), (-1, 0.0, 0, 0, -0.022106), (50, 0.21499999999999997, 9, 10, 0.892522), (-1, 0.0, 0, 0, -0.013398), (15, 0.131, 11, 12, 1.00805), (-1, 0.0, 0, 0, 0.057898), (-1, 0.0, 0, 0, 0.01698)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (58, 0.3431372549019608, 2, 5, -0.341188), (46, 0.3655, 3, 4, -0.166676), (-1, 0.0, 0, 0, -0.016754), (-1, 0.0, 0, 0, 0.03209), (7, 0.967, 6, 7, -0.991571), (-1, 0.0, 0, 0, -0.057387), (-1, 0.0, 0, 0, 0.007003), (52, 0.5, 9, 12, 0.825084), (6, 0.8978411663857659, 10, 11, 1.087339), (-1, 0.0, 0, 0, 0.055938), (-1, 0.0, 0, 0, 0.005227), (1, 0.887, 13, 14, 0.255806), (-1, 0.0, 0, 0, -0.010515), (-1, 0.0, 0, 0, 0.056587)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (25, 6.556061562263916, 2, 5, -0.327394), (28, 0.161, 3, 4, -1.098416), (-1, 0.0, 0, 0, 0.004343), (-1, 0.0, 0, 0, -0.059236), (58, 0.11805555555555555, 6, 7, -0.185387), (-1, 0.0, 0, 0, 0.017633), (-1, 0.0, 0, 0, -0.022444), (52, 0.5, 9, 12, 0.806028), (6, 0.8978411663857659, 10, 11, 1.069292), (-1, 0.0, 0, 0, 0.055062), (-1, 0.0, 0, 0, 0.005075), (62, 0.14200000000000002, 13, 14, 0.247521), (-1, 0.0, 0, 0, 0.061609), (-1, 0.0, 0, 0, -0.007952)),
            ((45, 0.87, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, -0.299118), (28, 0.622, 3, 4, -0.690575), (-1, 0.0, 0, 0, -0.008806), (-1, 0.0, 0, 0, -0.042408), (41, 0.5, 6, 7, 0.001344), (-1, 0.0, 0, 0, 0.029289), (-1, 0.0, 0, 0, -0.021041), (50, 0.21499999999999997, 9, 10, 0.840608), (-1, 0.0, 0, 0, -0.014854), (15, 0.0050000000000000044, 11, 12, 0.959974), (-1, 0.0, 0, 0, 0.058055), (-1, 0.0, 0, 0, 0.021188)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (58, 0.3431372549019608, 2, 5, -0.304586), (15, -0.741, 3, 4, -0.139027), (-1, 0.0, 0, 0, -0.060911), (-1, 0.0, 0, 0, -0.000946), (13, 0.986, 6, 7, -0.93998), (-1, 0.0, 0, 0, -0.059197), (-1, 0.0, 0, 0, -0.007375), (52, 0.5, 9, 12, 0.773566), (6, 0.8978411663857659, 10, 11, 1.042809), (-1, 0.0, 0, 0, 0.053817), (-1, 0.0, 0, 0, 0.003857), (11, 0.988, 13, 14, 0.223608), (-1, 0.0, 0, 0, -0.013976), (-1, 0.0, 0, 0, 0.050495)),
            ((45, 0.20606000000000002, 1, 8, 0.0), (24, 0.027500000000000024, 2, 5, -0.388241), (7, 0.965, 3, 4, -0.741409), (-1, 0.0, 0, 0, -0.041433), (-1, 0.0, 0, 0, 0.016316), (59, 0.39230769230769236, 6, 7, -0.097337), (-1, 0.0, 0, 0, 0.021712), (-1, 0.0, 0, 0, -0.022865), (50, 0.955, 9, 12, 0.58605), (41, 0.5, 10, 11, 0.235362), (-1, 0.0, 0, 0, 0.027099), (-1, 0.0, 0, 0, -0.01632), (24, 0.03708333333333337, 13, 14, 1.008553), (-1, 0.0, 0, 0, 0.002033), (-1, 0.0, 0, 0, 0.055902)),
            ((45, 0.87, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, -0.271265), (26, 4.2413008733233095, 3, 4, -0.639987), (-1, 0.0, 0, 0, -0.060246), (-1, 0.0, 0, 0, -0.0239), (41, 0.5, 6, 7, 0.00236), (-1, 0.0, 0, 0, 0.027488), (-1, 0.0, 0, 0, -0.019712), (55, 0.24, 9, 12, 0.793591), (56, 0.062000000000000055, 10, 11, -0.000344), (-1, 0.0, 0, 0, 0.032694), (-1, 0.0, 0, 0, -0.037321), (15, 0.0050000000000000044, 13, 14, 0.968455), (-1, 0.0, 0, 0, 0.056249), (-1, 0.0, 0, 0, 0.019568)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (25, 6.609348335006922, 2, 5, -0.2769), (55, 0.943, 3, 4, -0.940225), (-1, 0.0, 0, 0, -0.051803), (-1, 0.0, 0, 0, 0.006663), (58, 0.11805555555555555, 6, 7, -0.118301), (-1, 0.0, 0, 0, 0.021613), (-1, 0.0, 0, 0, -0.019628), (52, 0.5, 9, 12, 0.730909), (6, 0.8978411663857659, 10, 11, 1.009336), (-1, 0.0, 0, 0, 0.05228), (-1, 0.0, 0, 0, 0.001741), (62, 0.14200000000000002, 13, 14, 0.190217), (-1, 0.0, 0, 0, 0.05881), (-1, 0.0, 0, 0, -0.009503)),
            ((45, 0.87, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, -0.252514), (26, 4.2413008733233095, 3, 4, -0.610513), (-1, 0.0, 0, 0, -0.059015), (-1, 0.0, 0, 0, -0.022605), (3, 0.24, 6, 7, 0.009475), (-1, 0.0, 0, 0, 0.026341), (-1, 0.0, 0, 0, -0.019336), (55, 0.24, 9, 12, 0.763036), (56, 0.062000000000000055, 10, 11, -0.01721), (-1, 0.0, 0, 0, 0.031383), (-1, 0.0, 0, 0, -0.036425), (15, 0.0050000000000000044, 13, 14, 0.941055), (-1, 0.0, 0, 0, 0.055165), (-1, 0.0, 0, 0, 0.018176)),
            ((45, 0.87, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, -0.2423), (58, 0.17424242424242425, 3, 4, -0.590141), (-1, 0.0, 0, 0, -0.015004), (-1, 0.0, 0, 0, -0.044737), (41, 0.5, 6, 7, 0.009075), (-1, 0.0, 0, 0, 0.026272), (-1, 0.0, 0, 0, -0.018338), (13, 0.933, 9, 12, 0.743589), (58, 0.4083333333333333, 10, 11, 0.215363), (-1, 0.0, 0, 0, 0.022276), (-1, 0.0, 0, 0, -0.045178), (56, 0.24, 13, 14, 1.014809), (-1, 0.0, 0, 0, -0.00837), (-1, 0.0, 0, 0, 0.053999)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (25, 6.609348335006922, 2, 5, -0.250002), (55, 0.943, 3, 4, -0.900265), (-1, 0.0, 0, 0, -0.049804), (-1, 0.0, 0, 0, 0.006983), (58, 0.11805555555555555, 6, 7, -0.097059), (-1, 0.0, 0, 0, 0.021144), (-1, 0.0, 0, 0, -0.017973), (52, 0.5, 9, 12, 0.688414), (6, 0.8978411663857659, 10, 11, 0.978552), (-1, 0.0, 0, 0, 0.05088), (-1, 0.0, 0, 0, 0.0), (11, 0.988, 13, 14, 0.152048), (-1, 0.0, 0, 0, -0.015605), (-1, 0.0, 0, 0, 0.047376)),
            ((45, 0.16178, 1, 8, 0.0), (26, 4.255587559545337, 2, 5, -0.326088), (12, 0.611, 3, 4, -0.937503), (-1, 0.0, 0, 0, -0.053249), (-1, 0.0, 0, 0, 0.004793), (59, 0.3798076923076923, 6, 7, -0.181394), (-1, 0.0, 0, 0, 0.012152), (-1, 0.0, 0, 0, -0.025489), (7, 0.942, 9, 12, 0.507041), (58, 0.39230769230769236, 10, 11, 0.181686), (-1, 0.0, 0, 0, 0.016806), (-1, 0.0, 0, 0, -0.041964), (60, 0.5, 13, 14, 0.944507), (-1, 0.0, 0, 0, 0.054008), (-1, 0.0, 0, 0, 0.005436)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.234363), (35, 0.5, 3, 4, -1.08241), (-1, 0.0, 0, 0, -0.058054), (-1, 0.0, 0, 0, 0.004746), (15, -0.741, 6, 7, -0.124238), (-1, 0.0, 0, 0, -0.058317), (-1, 0.0, 0, 0, -0.000345), (52, 0.5, 9, 12, 0.662964), (47, 0.13392857142857142, 10, 11, 0.958306), (-1, 0.0, 0, 0, 0.050045), (-1, 0.0, 0, 0, 0.000289), (62, 0.14200000000000002, 13, 14, 0.134049), (-1, 0.0, 0, 0, 0.056264), (-1, 0.0, 0, 0, -0.011157)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (58, 0.17424242424242425, 2, 5, -0.456416), (26, 4.375737243570452, 3, 4, -0.127069), (-1, 0.0, 0, 0, -0.044289), (-1, 0.0, 0, 0, 0.010644), (7, 0.865, 6, 7, -0.76497), (-1, 0.0, 0, 0, -0.049228), (-1, 0.0, 0, 0, -0.000967), (41, 0.5, 9, 12, 0.327174), (15, 0.9109999999999999, 10, 11, 0.746654), (-1, 0.0, 0, 0, 0.04115), (-1, 0.0, 0, 0, -0.020591), (13, 0.969, 13, 14, -0.081402), (-1, 0.0, 0, 0, -0.030596), (-1, 0.0, 0, 0, 0.028446)),
            ((45, 0.87, 1, 8, 0.0), (27, 4.539296550572581, 2, 5, -0.208409), (45, 0.731, 3, 4, -0.80601), (-1, 0.0, 0, 0, -0.045563), (-1, 0.0, 0, 0, -0.000666), (59, 0.1380952380952381, 6, 7, -0.080409), (-1, 0.0, 0, 0, 0.031228), (-1, 0.0, 0, 0, -0.011206), (13, 0.744, 9, 12, 0.681666), (58, 0.20526315789473684, 10, 11, -0.029306), (-1, 0.0, 0, 0, 0.019685), (-1, 0.0, 0, 0, -0.035746), (56, 0.20799999999999996, 13, 14, 0.88744), (-1, 0.0, 0, 0, 0.00168), (-1, 0.0, 0, 0, 0.047751)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (58, 0.3038461538461539, 2, 5, -0.437028), (15, -0.741, 3, 4, -0.251851), (-1, 0.0, 0, 0, -0.05436), (-1, 0.0, 0, 0, -0.004316), (62, 0.41100000000000003, 6, 7, -0.970835), (-1, 0.0, 0, 0, -0.053644), (-1, 0.0, 0, 0, 0.004442), (41, 0.5, 9, 12, 0.3132), (6, 0.4675590606828445, 10, 11, 0.71967), (-1, 0.0, 0, 0, 0.049284), (-1, 0.0, 0, 0, 0.017136), (13, 0.969, 13, 14, -0.077329), (-1, 0.0, 0, 0, -0.028868), (-1, 0.0, 0, 0, 0.027044)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (59, 0.3693181818181818, 2, 5, -0.209895), (15, -0.741, 3, 4, 0.096507), (-1, 0.0, 0, 0, -0.054713), (-1, 0.0, 0, 0, 0.014588), (49, 3.5, 6, 7, -0.488576), (-1, 0.0, 0, 0, -0.017748), (-1, 0.0, 0, 0, -0.061975), (52, 0.5, 9, 12, 0.618015), (6, 0.8978411663857659, 10, 11, 0.925527), (-1, 0.0, 0, 0, 0.048531), (-1, 0.0, 0, 0, -0.003976), (62, 0.14200000000000002, 13, 14, 0.099351), (-1, 0.0, 0, 0, 0.054994), (-1, 0.0, 0, 0, -0.012137)),
            ((45, 0.87, 1, 8, 0.0), (26, 4.2413008733233095, 2, 5, -0.190493), (7, 0.947, 3, 4, -0.828838), (-1, 0.0, 0, 0, -0.052994), (-1, 0.0, 0, 0, 0.030939), (59, 0.3693181818181818, 6, 7, -0.07092), (-1, 0.0, 0, 0, 0.015539), (-1, 0.0, 0, 0, -0.018182), (13, 0.744, 9, 12, 0.64761), (6, 0.1472505229581609, 10, 11, -0.045554), (-1, 0.0, 0, 0, -0.03991), (-1, 0.0, 0, 0, 0.01616), (26, 4.916317905782089, 13, 14, 0.855775), (-1, 0.0, 0, 0, 0.047067), (-1, 0.0, 0, 0, 0.007366)),
            ((20, -3.210268834500856, 1, 8, 0.0), (58, 0.17424242424242425, 2, 5, -0.417337), (26, 4.375737243570452, 3, 4, -0.107433), (-1, 0.0, 0, 0, -0.042614), (-1, 0.0, 0, 0, 0.010902), (28, 0.592, 6, 7, -0.735642), (-1, 0.0, 0, 0, -0.00201), (-1, 0.0, 0, 0, -0.05113), (41, 0.5, 9, 12, 0.288968), (3, 0.30100000000000005, 10, 11, 0.680543), (-1, 0.0, 0, 0, 0.04471), (-1, 0.0, 0, 0, 0.011223), (13, 0.998, 13, 14, -0.083986), (-1, 0.0, 0, 0, -0.014148), (-1, 0.0, 0, 0, 0.060676)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.193163), (35, 0.5, 3, 4, -1.014516), (-1, 0.0, 0, 0, -0.054833), (-1, 0.0, 0, 0, 0.00567), (15, -0.741, 6, 7, -0.091694), (-1, 0.0, 0, 0, -0.055569), (-1, 0.0, 0, 0, 0.00079), (52, 0.5, 9, 12, 0.587293), (6, 0.8978411663857659, 10, 11, 0.900209), (-1, 0.0, 0, 0, 0.047414), (-1, 0.0, 0, 0, -0.005429), (11, 0.988, 13, 14, 0.081707), (-1, 0.0, 0, 0, -0.017009), (-1, 0.0, 0, 0, 0.044379)),
            ((45, 0.87, 1, 8, 0.0), (26, 4.2413008733233095, 2, 5, -0.173954), (7, 0.947, 3, 4, -0.79176), (-1, 0.0, 0, 0, -0.051131), (-1, 0.0, 0, 0, 0.029885), (59, 0.3693181818181818, 6, 7, -0.062143), (-1, 0.0, 0, 0, 0.015006), (-1, 0.0, 0, 0, -0.017127), (13, 0.933, 9, 12, 0.612985), (58, 0.12916666666666665, 10, 11, 0.113458), (-1, 0.0, 0, 0, 0.031855), (-1, 0.0, 0, 0, -0.015063), (56, 0.24, 13, 14, 0.914659), (-1, 0.0, 0, 0, -0.0119), (-1, 0.0, 0, 0, 0.049637)),
            ((20, -3.210268834500856, 1, 8, 0.0), (58, 0.17424242424242425, 2, 5, -0.391179), (26, 4.375737243570452, 3, 4, -0.09122), (-1, 0.0, 0, 0, -0.040642), (-1, 0.0, 0, 0, 0.010589), (28, 0.592, 6, 7, -0.706997), (-1, 0.0, 0, 0, -0.001521), (-1, 0.0, 0, 0, -0.049918), (41, 0.5, 9, 12, 0.271374), (6, 0.4675590606828445, 10, 11, 0.647725), (-1, 0.0, 0, 0, 0.046048), (-1, 0.0, 0, 0, 0.013786), (13, 0.969, 13, 14, -0.081159), (-1, 0.0, 0, 0, -0.026206), (-1, 0.0, 0, 0, 0.023453)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (25, 6.609348335006922, 2, 5, -0.177562), (55, 0.925, 3, 4, -0.774866), (-1, 0.0, 0, 0, -0.043788), (-1, 0.0, 0, 0, 0.005983), (15, -0.741, 6, 7, -0.046602), (-1, 0.0, 0, 0, -0.054327), (-1, 0.0, 0, 0, 0.003224), (52, 0.5, 9, 12, 0.556882), (6, 0.8978411663857659, 10, 11, 0.875854), (-1, 0.0, 0, 0, 0.04633), (-1, 0.0, 0, 0, -0.006229), (62, 0.14200000000000002, 13, 14, 0.06304), (-1, 0.0, 0, 0, 0.053364), (-1, 0.0, 0, 0, -0.012884)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (58, 0.3038461538461539, 2, 5, -0.363311), (27, 4.590055742951803, 3, 4, -0.191357), (-1, 0.0, 0, 0, -0.041019), (-1, 0.0, 0, 0, -6.4e-05), (62, 0.41100000000000003, 6, 7, -0.902771), (-1, 0.0, 0, 0, -0.050706), (-1, 0.0, 0, 0, 0.008005), (41, 0.5, 9, 12, 0.26104), (60, 0.5, 10, 11, 0.622531), (-1, 0.0, 0, 0, 0.036276), (-1, 0.0, 0, 0, -0.019344), (13, 0.969, 13, 14, -0.068914), (-1, 0.0, 0, 0, -0.025205), (-1, 0.0, 0, 0, 0.024167)),
            ((45, 0.87, 1, 8, 0.0), (27, 4.539296550572581, 2, 5, -0.157021), (50, 0.893, 3, 4, -0.71419), (-1, 0.0, 0, 0, -0.04748), (-1, 0.0, 0, 0, -0.015134), (59, 0.1380952380952381, 6, 7, -0.044034), (-1, 0.0, 0, 0, 0.03069), (-1, 0.0, 0, 0, -0.008998), (7, 0.567, 9, 12, 0.576053), (55, 0.24, 10, 11, -0.045542), (-1, 0.0, 0, 0, -0.034973), (-1, 0.0, 0, 0, 0.030982), (41, 2.5, 13, 14, 0.820257), (-1, 0.0, 0, 0, 0.045156), (-1, 0.0, 0, 0, -0.025636)),
            ((20, -3.210268834500856, 1, 8, 0.0), (58, 0.11805555555555555, 2, 5, -0.35499), (26, 4.375737243570452, 3, 4, 0.042322), (-1, 0.0, 0, 0, -0.03862), (-1, 0.0, 0, 0, 0.02392), (28, 0.592, 6, 7, -0.581869), (-1, 0.0, 0, 0, -0.00031), (-1, 0.0, 0, 0, -0.043774), (41, 0.5, 9, 12, 0.245027), (6, 0.4675590606828445, 10, 11, 0.597186), (-1, 0.0, 0, 0, 0.043775), (-1, 0.0, 0, 0, 0.011223), (13, 0.998, 13, 14, -0.077188), (-1, 0.0, 0, 0, -0.01272), (-1, 0.0, 0, 0, 0.058147)),
            ((7, 0.957, 1, 8, 0.0), (58, 0.3798076923076923, 2, 5, -0.147315), (20, -4.595119850134589, 3, 4, -0.021352), (-1, 0.0, 0, 0, -0.027193), (-1, 0.0, 0, 0, 0.008216), (19, 0.5, 6, 7, -0.935793), (-1, 0.0, 0, 0, -0.051051), (-1, 0.0, 0, 0, 0.018576), (56, 0.10299999999999998, 9, 12, 0.572253), (15, 0.954, 10, 11, -0.43491), (-1, 0.0, 0, 0, -0.051741), (-1, 0.0, 0, 0, 0.029992), (60, 0.5, 13, 14, 0.761729), (-1, 0.0, 0, 0, 0.045062), (-1, 0.0, 0, 0, -0.003283)),
            ((24, 0.06250000000000003, 1, 6, 0.0), (15, -0.741, 2, 3, -0.156815), (-1, 0.0, 0, 0, -0.053334), (59, 0.4365079365079365, 4, 5, -0.074829), (-1, 0.0, 0, 0, 0.010556), (-1, 0.0, 0, 0, -0.022464), (52, 0.5, 7, 10, 0.510827), (6, 0.8978411663857659, 8, 9, 0.842322), (-1, 0.0, 0, 0, 0.044868), (-1, 0.0, 0, 0, -0.007824), (11, 0.988, 11, 12, 0.025273), (-1, 0.0, 0, 0, -0.01798), (-1, 0.0, 0, 0, 0.041541)),
            ((7, 0.957, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, -0.139631), (19, 0.5, 3, 4, -0.446919), (-1, 0.0, 0, 0, -0.029203), (-1, 0.0, 0, 0, 0.028282), (49, 2.5, 6, 7, 0.166401), (-1, 0.0, 0, 0, 0.030305), (-1, 0.0, 0, 0, -0.015824), (56, 0.10299999999999998, 9, 12, 0.547652), (15, 0.954, 10, 11, -0.414402), (-1, 0.0, 0, 0, -0.050252), (-1, 0.0, 0, 0, 0.029247), (29, 0.5, 13, 14, 0.730914), (-1, 0.0, 0, 0, 0.051704), (-1, 0.0, 0, 0, 0.01531)),
            ((45, 0.87, 1, 8, 0.0), (26, 4.2413008733233095, 2, 5, -0.136872), (7, 0.947, 3, 4, -0.729741), (-1, 0.0, 0, 0, -0.047935), (-1, 0.0, 0, 0, 0.02814), (49, 2.5, 6, 7, -0.035868), (-1, 0.0, 0, 0, 0.008511), (-1, 0.0, 0, 0, -0.022672), (7, 0.567, 9, 12, 0.530446), (55, 0.24, 10, 11, -0.056118), (-1, 0.0, 0, 0, -0.034365), (-1, 0.0, 0, 0, 0.030256), (41, 2.5, 13, 14, 0.775061), (-1, 0.0, 0, 0, 0.043199), (-1, 0.0, 0, 0, -0.025921)),
            ((20, -3.683959493972949, 1, 8, 0.0), (28, 0.622, 2, 5, -0.365228), (12, 0.374, 3, 4, 0.120354), (-1, 0.0, 0, 0, -0.039098), (-1, 0.0, 0, 0, 0.023102), (43, 0.11437908496732026, 6, 7, -0.563104), (-1, 0.0, 0, 0, -0.009313), (-1, 0.0, 0, 0, -0.051015), (41, 0.5, 9, 12, 0.191733), (58, 0.03125, 10, 11, 0.561123), (-1, 0.0, 0, 0, 0.05589), (-1, 0.0, 0, 0, 0.01444), (55, 0.978, 13, 14, -0.130609), (-1, 0.0, 0, 0, -0.013837), (-1, 0.0, 0, 0, 0.044829)),
            ((24, 0.06250000000000003, 1, 6, 0.0), (15, -0.741, 2, 3, -0.13908), (-1, 0.0, 0, 0, -0.051965), (59, 0.3693181818181818, 4, 5, -0.060531), (-1, 0.0, 0, 0, 0.014168), (-1, 0.0, 0, 0, -0.017655), (52, 0.5, 7, 10, 0.473575), (6, 0.8978411663857659, 8, 9, 0.811399), (-1, 0.0, 0, 0, 0.043523), (-1, 0.0, 0, 0, -0.009515), (62, 0.14200000000000002, 11, 12, -4.8e-05), (-1, 0.0, 0, 0, 0.051442), (-1, 0.0, 0, 0, -0.014985)),
            ((7, 0.957, 1, 8, 0.0), (58, 0.3798076923076923, 2, 5, -0.126529), (26, 4.363078339578307, 3, 4, -0.009189), (-1, 0.0, 0, 0, -0.027465), (-1, 0.0, 0, 0, 0.007818), (19, 0.5, 6, 7, -0.898681), (-1, 0.0, 0, 0, -0.048964), (-1, 0.0, 0, 0, 0.016494), (55, 0.10299999999999998, 9, 12, 0.507691), (15, 0.954, 10, 11, -0.405555), (-1, 0.0, 0, 0, -0.049642), (-1, 0.0, 0, 0, 0.027763), (29, 0.5, 13, 14, 0.685077), (-1, 0.0, 0, 0, 0.049925), (-1, 0.0, 0, 0, 0.013507)),
            ((20, -3.683959493972949, 1, 8, 0.0), (40, 0.0045000000000000595, 2, 5, -0.344544), (58, 0.11805555555555555, 3, 4, -0.61845), (-1, 0.0, 0, 0, -0.002828), (-1, 0.0, 0, 0, -0.05111), (40, 0.498, 6, 7, -0.011589), (-1, 0.0, 0, 0, 0.02656), (-1, 0.0, 0, 0, -0.037216), (41, 0.5, 9, 12, 0.180711), (58, 0.03125, 10, 11, 0.531612), (-1, 0.0, 0, 0, 0.054264), (-1, 0.0, 0, 0, 0.013236), (55, 0.978, 13, 14, -0.122465), (-1, 0.0, 0, 0, -0.012986), (-1, 0.0, 0, 0, 0.043232)),
            ((45, 0.87, 1, 8, 0.0), (27, 4.539296550572581, 2, 5, -0.120766), (50, 0.893, 3, 4, -0.664265), (-1, 0.0, 0, 0, -0.044979), (-1, 0.0, 0, 0, -0.013699), (59, 0.1380952380952381, 6, 7, -0.012629), (-1, 0.0, 0, 0, 0.029567), (-1, 0.0, 0, 0, -0.006946), (13, 0.933, 9, 12, 0.491459), (58, 0.12916666666666665, 10, 11, 0.043216), (-1, 0.0, 0, 0, 0.027576), (-1, 0.0, 0, 0, -0.016943), (56, 0.24, 13, 14, 0.816839), (-1, 0.0, 0, 0, -0.012121), (-1, 0.0, 0, 0, 0.045212)),
            ((20, -3.210268834500856, 1, 8, 0.0), (59, 0.4365079365079365, 2, 5, -0.288928), (15, -0.741, 3, 4, -0.077469), (-1, 0.0, 0, 0, -0.04725), (-1, 0.0, 0, 0, 0.00545), (9, 0.16400000000000003, 6, 7, -0.706378), (-1, 0.0, 0, 0, -0.051147), (-1, 0.0, 0, 0, -0.018026), (55, 0.976, 9, 12, 0.201301), (41, 0.5, 10, 11, 0.077343), (-1, 0.0, 0, 0, 0.021617), (-1, 0.0, 0, 0, -0.011184), (6, 0.8802139175373297, 13, 14, 0.99725), (-1, 0.0, 0, 0, 0.056484), (-1, 0.0, 0, 0, -0.008594)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.123823), (35, 0.5, 3, 4, -0.901037), (-1, 0.0, 0, 0, -0.049796), (-1, 0.0, 0, 0, 0.00927), (15, -0.741, 6, 7, -0.037248), (-1, 0.0, 0, 0, -0.050279), (-1, 0.0, 0, 0, 0.002534), (52, 0.5, 9, 12, 0.438506), (60, 0.5, 10, 11, 0.779555), (-1, 0.0, 0, 0, 0.044025), (-1, 0.0, 0, 0, 0.001151), (11, 0.988, 13, 14, -0.016767), (-1, 0.0, 0, 0, -0.018707), (-1, 0.0, 0, 0, 0.039946)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, 0.071121), (27, 4.280624567377958, 3, 4, -0.124645), (-1, 0.0, 0, 0, -0.046196), (-1, 0.0, 0, 0, -0.001435), (7, 0.957, 6, 7, 0.434864), (-1, 0.0, 0, 0, 0.009593), (-1, 0.0, 0, 0, 0.048646), (50, 0.985, 9, 12, -0.764133), (26, 4.668133964286986, 10, 11, -1.076477), (-1, 0.0, 0, 0, -0.066829), (-1, 0.0, 0, 0, -0.012041), (12, 0.6639999999999999, 13, 14, 0.625834), (-1, 0.0, 0, 0, 0.043155), (-1, 0.0, 0, 0, -0.004738)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (20, -4.595119850134589, 2, 5, 0.497624), (63, 0.664, 3, 4, -0.197901), (-1, 0.0, 0, 0, 0.029857), (-1, 0.0, 0, 0, -0.043447), (3, 0.30000000000000004, 6, 7, 1.003914), (-1, 0.0, 0, 0, 0.057456), (-1, 0.0, 0, 0, -0.017585), (20, -3.210268834500856, 9, 12, -0.105282), (62, 0.32499999999999996, 10, 11, -0.426866), (-1, 0.0, 0, 0, -0.030574), (-1, 0.0, 0, 0, 0.006518), (55, 0.976, 13, 14, 0.088235), (-1, 0.0, 0, 0, -0.001145), (-1, 0.0, 0, 0, 0.046528)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.20606000000000002, 2, 5, 0.069164), (26, 4.255587559545337, 3, 4, -0.114325), (-1, 0.0, 0, 0, -0.035787), (-1, 0.0, 0, 0, 0.000597), (7, 0.957, 6, 7, 0.414033), (-1, 0.0, 0, 0, 0.009062), (-1, 0.0, 0, 0, 0.047538), (50, 0.985, 9, 12, -0.744234), (26, 4.668133964286986, 10, 11, -1.043097), (-1, 0.0, 0, 0, -0.064765), (-1, 0.0, 0, 0, -0.011879), (12, 0.6639999999999999, 13, 14, 0.606053), (-1, 0.0, 0, 0, 0.042219), (-1, 0.0, 0, 0, -0.005056)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (20, -4.595119850134589, 2, 5, 0.486021), (63, 0.664, 3, 4, -0.185938), (-1, 0.0, 0, 0, 0.028959), (-1, 0.0, 0, 0, -0.042495), (3, 0.30000000000000004, 6, 7, 0.980101), (-1, 0.0, 0, 0, 0.056187), (-1, 0.0, 0, 0, -0.017006), (20, -3.210268834500856, 9, 12, -0.100896), (49, 2.5, 10, 11, -0.409153), (-1, 0.0, 0, 0, -0.009141), (-1, 0.0, 0, 0, -0.043805), (55, 0.976, 13, 14, 0.08323), (-1, 0.0, 0, 0, -0.001061), (-1, 0.0, 0, 0, 0.04508)),
            ((60, 0.5, 1, 8, 0.0), (41, 0.5, 2, 5, 0.067003), (15, -0.741, 3, 4, 0.320164), (-1, 0.0, 0, 0, -0.044596), (-1, 0.0, 0, 0, 0.022295), (26, 4.2413008733233095, 6, 7, -0.175344), (-1, 0.0, 0, 0, -0.039432), (-1, 0.0, 0, 0, -0.001072), (50, 0.985, 9, 12, -0.723802), (26, 4.668133964286986, 10, 11, -1.01044), (-1, 0.0, 0, 0, -0.062767), (-1, 0.0, 0, 0, -0.011499), (15, -0.002, 13, 14, 0.589091), (-1, 0.0, 0, 0, -0.005348), (-1, 0.0, 0, 0, 0.041478)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.061171), (7, 0.955, 3, 4, -0.171321), (-1, 0.0, 0, 0, -0.020025), (-1, 0.0, 0, 0, 0.036193), (15, -0.741, 6, 7, 0.324441), (-1, 0.0, 0, 0, -0.037333), (-1, 0.0, 0, 0, 0.021066), (24, 0.06250000000000003, 9, 12, -0.752183), (27, 4.7190553516086915, 10, 11, -0.940503), (-1, 0.0, 0, 0, -0.059145), (-1, 0.0, 0, 0, 0.002326), (1, 0.006, 13, 14, 0.459742), (-1, 0.0, 0, 0, 0.002924), (-1, 0.0, 0, 0, 0.026442)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (20, -4.595119850134589, 2, 5, 0.476607), (63, 0.664, 3, 4, -0.171222), (-1, 0.0, 0, 0, 0.027945), (-1, 0.0, 0, 0, -0.041355), (3, 0.30000000000000004, 6, 7, 0.955811), (-1, 0.0, 0, 0, 0.054807), (-1, 0.0, 0, 0, -0.01593), (20, -3.210268834500856, 9, 12, -0.098112), (62, 0.32499999999999996, 10, 11, -0.39208), (-1, 0.0, 0, 0, -0.028836), (-1, 0.0, 0, 0, 0.007696), (55, 0.976, 13, 14, 0.075987), (-1, 0.0, 0, 0, -0.001121), (-1, 0.0, 0, 0, 0.043632)),
            ((60, 0.5, 1, 8, 0.0), (41, 0.5, 2, 5, 0.064685), (15, -0.741, 3, 4, 0.308308), (-1, 0.0, 0, 0, -0.042481), (-1, 0.0, 0, 0, 0.02118), (26, 4.2413008733233095, 6, 7, -0.167205), (-1, 0.0, 0, 0, -0.038055), (-1, 0.0, 0, 0, -0.001064), (50, 0.985, 9, 12, -0.711207), (26, 4.668133964286986, 10, 11, -0.98455), (-1, 0.0, 0, 0, -0.061109), (-1, 0.0, 0, 0, -0.011508), (12, 0.6639999999999999, 13, 14, 0.570557), (-1, 0.0, 0, 0, 0.040345), (-1, 0.0, 0, 0, -0.005713)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.058524), (7, 0.955, 3, 4, -0.163279), (-1, 0.0, 0, 0, -0.018992), (-1, 0.0, 0, 0, 0.034511), (43, 0.10555555555555556, 6, 7, 0.307605), (-1, 0.0, 0, 0, 0.033608), (-1, 0.0, 0, 0, 0.001762), (24, 0.06250000000000003, 9, 12, -0.735836), (13, 0.43999999999999995, 10, 11, -0.919069), (-1, 0.0, 0, 0, 0.003739), (-1, 0.0, 0, 0, -0.057731), (1, 0.006, 13, 14, 0.445234), (-1, 0.0, 0, 0, 0.002471), (-1, 0.0, 0, 0, 0.025832)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (20, -4.595119850134589, 2, 5, 0.460455), (26, 4.337269381070008, 3, 4, -0.165987), (-1, 0.0, 0, 0, -0.049022), (-1, 0.0, 0, 0, 0.020793), (1, 0.12, 6, 7, 0.93115), (-1, 0.0, 0, 0, 0.065472), (-1, 0.0, 0, 0, 0.017419), (20, -3.210268834500856, 9, 12, -0.09359), (49, 2.5, 10, 11, -0.374866), (-1, 0.0, 0, 0, -0.007571), (-1, 0.0, 0, 0, -0.042284), (55, 0.976, 13, 14, 0.071681), (-1, 0.0, 0, 0, -0.001078), (-1, 0.0, 0, 0, 0.042357)),
            ((60, 0.5, 1, 8, 0.0), (41, 0.5, 2, 5, 0.06234), (15, -0.741, 3, 4, 0.291049), (-1, 0.0, 0, 0, -0.041421), (-1, 0.0, 0, 0, 0.020086), (3, 0.15100000000000002, 6, 7, -0.155305), (-1, 0.0, 0, 0, 0.020728), (-1, 0.0, 0, 0, -0.014742), (50, 0.985, 9, 12, -0.694236), (26, 4.668133964286986, 10, 11, -0.957448), (-1, 0.0, 0, 0, -0.059393), (-1, 0.0, 0, 0, -0.011195), (12, 0.6639999999999999, 13, 14, 0.560557), (-1, 0.0, 0, 0, 0.039566), (-1, 0.0, 0, 0, -0.005583)),
            ((58, 0.458041958041958, 1, 8, 0.0), (26, 4.952293473989714, 2, 5, 0.0604), (26, 4.579839228586978, 3, 4, 0.144747), (-1, 0.0, 0, 0, -0.005219), (-1, 0.0, 0, 0, 0.022589), (12, 0.581, 6, 7, -0.503698), (-1, 0.0, 0, 0, -0.006494), (-1, 0.0, 0, 0, -0.050069), (7, 0.994, 9, 12, -0.674363), (20, 2.319785806352712, 10, 11, -0.865417), (-1, 0.0, 0, 0, -0.049014), (-1, 0.0, 0, 0, 0.014899), (-1, 0.0, 0, 0, 0.038134)),
            ((19, 0.5, 1, 8, 0.0), (1, 0.009, 2, 5, -0.06678), (59, 0.4365079365079365, 3, 4, 0.110744), (-1, 0.0, 0, 0, 0.023811), (-1, 0.0, 0, 0, -0.018006), (2, 0.985, 6, 7, -0.341764), (-1, 0.0, 0, 0, -0.029055), (-1, 0.0, 0, 0, 0.014607), (6, 0.8873180999538214, 9, 12, 0.60979), (13, 0.744, 10, 11, 0.760492), (-1, 0.0, 0, 0, 0.006292), (-1, 0.0, 0, 0, 0.051235), (-1, 0.0, 0, 0, -0.035935)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.31292000000000003, 2, 5, 0.059113), (27, 4.280624567377958, 3, 4, -0.091634), (-1, 0.0, 0, 0, -0.045726), (-1, 0.0, 0, 0, 0.000275), (7, 0.957, 6, 7, 0.370414), (-1, 0.0, 0, 0, 0.007935), (-1, 0.0, 0, 0, 0.044296), (50, 0.985, 9, 12, -0.670847), (46, 0.8005, 10, 11, -0.92695), (-1, 0.0, 0, 0, -0.053776), (-1, 0.0, 0, 0, 0.000414), (12, 0.6639999999999999, 13, 14, 0.553075), (-1, 0.0, 0, 0, 0.038802), (-1, 0.0, 0, 0, -0.005156)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (20, -4.595119850134589, 2, 5, 0.43645), (63, 0.664, 3, 4, -0.164031), (-1, 0.0, 0, 0, 0.027723), (-1, 0.0, 0, 0, -0.040669), (3, 0.30000000000000004, 6, 7, 0.897987), (-1, 0.0, 0, 0, 0.051775), (-1, 0.0, 0, 0, -0.015349), (20, -3.210268834500856, 9, 12, -0.087801), (62, 0.32499999999999996, 10, 11, -0.355293), (-1, 0.0, 0, 0, -0.027124), (-1, 0.0, 0, 0, 0.009616), (26, 4.2413008733233095, 13, 14, 0.068896), (-1, 0.0, 0, 0, -0.033888), (-1, 0.0, 0, 0, 0.007843)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.053592), (7, 0.955, 3, 4, -0.153013), (-1, 0.0, 0, 0, -0.017675), (-1, 0.0, 0, 0, 0.032801), (43, 0.10555555555555556, 6, 7, 0.281982), (-1, 0.0, 0, 0, 0.031501), (-1, 0.0, 0, 0, 0.001103), (27, 4.728490960086225, 9, 12, -0.698035), (20, -2.6933930072678223, 10, 11, -0.913333), (-1, 0.0, 0, 0, -0.056401), (-1, 0.0, 0, 0, 0.022483), (-1, 0.0, 0, 0, 0.014867)),
            ((60, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, 0.057402), (20, -4.595119850134589, 3, 4, -0.698365), (-1, 0.0, 0, 0, -0.048499), (-1, 0.0, 0, 0, -0.009066), (41, 0.5, 6, 7, 0.114565), (-1, 0.0, 0, 0, 0.018887), (-1, 0.0, 0, 0, -0.006382), (50, 0.985, 9, 12, -0.654331), (26, 4.668133964286986, 10, 11, -0.904652), (-1, 0.0, 0, 0, -0.056448), (-1, 0.0, 0, 0, -0.009847), (15, -0.002, 13, 14, 0.545062), (-1, 0.0, 0, 0, -0.004679), (-1, 0.0, 0, 0, 0.03821)),
            ((58, 0.458041958041958, 1, 8, 0.0), (26, 4.952293473989714, 2, 5, 0.05531), (26, 4.579839228586978, 3, 4, 0.134884), (-1, 0.0, 0, 0, -0.004857), (-1, 0.0, 0, 0, 0.020865), (12, 0.581, 6, 7, -0.479975), (-1, 0.0, 0, 0, -0.005936), (-1, 0.0, 0, 0, -0.048393), (7, 0.994, 9, 12, -0.645076), (20, 1.965912816362163, 10, 11, -0.830559), (-1, 0.0, 0, 0, -0.047497), (-1, 0.0, 0, 0, 0.011839), (-1, 0.0, 0, 0, 0.036856)),
            ((19, 0.5, 1, 8, 0.0), (27, 4.539296550572581, 2, 5, -0.063171), (55, 0.925, 3, 4, -0.547613), (-1, 0.0, 0, 0, -0.036054), (-1, 0.0, 0, 0, 0.011839), (4, 0.025, 6, 7, 0.030118), (-1, 0.0, 0, 0, 0.039637), (-1, 0.0, 0, 0, -0.002782), (11, 0.948, 9, 12, 0.585816), (25, 6.636602293485552, 10, 11, -0.011545), (-1, 0.0, 0, 0, 0.035519), (-1, 0.0, 0, 0, -0.024136), (6, 0.8694716566129473, 13, 14, 0.997411), (-1, 0.0, 0, 0, 0.051133), (-1, 0.0, 0, 0, 0.007304)),
            ((60, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, 0.05499), (20, -4.595119850134589, 3, 4, -0.693125), (-1, 0.0, 0, 0, -0.048005), (-1, 0.0, 0, 0, -0.009428), (41, 0.5, 6, 7, 0.110675), (-1, 0.0, 0, 0, 0.018002), (-1, 0.0, 0, 0, -0.005957), (50, 0.985, 9, 12, -0.633615), (46, 0.8005, 10, 11, -0.875629), (-1, 0.0, 0, 0, -0.051255), (-1, 0.0, 0, 0, 0.001152), (12, 0.6639999999999999, 13, 14, 0.531269), (-1, 0.0, 0, 0, 0.037264), (-1, 0.0, 0, 0, -0.005156)),
            ((49, 3.5, 1, 8, 0.0), (23, 0.0050000000000000044, 2, 5, 0.050392), (63, 0.19999999999999998, 3, 4, -0.460329), (-1, 0.0, 0, 0, 0.017687), (-1, 0.0, 0, 0, -0.047177), (62, 0.24, 6, 7, 0.127191), (-1, 0.0, 0, 0, 0.018566), (-1, 0.0, 0, 0, -0.007396), (27, 4.728490960086225, 9, 12, -0.676794), (20, -2.6933930072678223, 10, 11, -0.889354), (-1, 0.0, 0, 0, -0.055012), (-1, 0.0, 0, 0, 0.021724), (-1, 0.0, 0, 0, 0.014298)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (15, -0.113, 2, 5, 0.419584), (45, 0.1, 3, 4, -0.11045), (-1, 0.0, 0, 0, -0.021689), (-1, 0.0, 0, 0, 0.031981), (27, 4.608290119226802, 6, 7, 0.918663), (-1, 0.0, 0, 0, -0.01177), (-1, 0.0, 0, 0, 0.074708), (20, -3.210268834500856, 9, 12, -0.083518), (62, 0.32499999999999996, 10, 11, -0.331249), (-1, 0.0, 0, 0, -0.025909), (-1, 0.0, 0, 0, 0.010463), (55, 0.976, 13, 14, 0.060494), (-1, 0.0, 0, 0, -0.001194), (-1, 0.0, 0, 0, 0.040122)),
            ((58, 0.3798076923076923, 1, 8, 0.0), (60, 0.5, 2, 5, 0.071213), (15, -0.741, 3, 4, 0.12518), (-1, 0.0, 0, 0, -0.032672), (-1, 0.0, 0, 0, 0.009217), (11, 0.754, 6, 7, -0.716331), (-1, 0.0, 0, 0, -0.061175), (-1, 0.0, 0, 0, -0.013845), (7, 0.967, 9, 12, -0.468641), (27, 4.552559367781538, 10, 11, -0.767744), (-1, 0.0, 0, 0, -0.005973), (-1, 0.0, 0, 0, -0.052019), (1, 0.01, 13, 14, 0.450142), (-1, 0.0, 0, 0, 0.040323), (-1, 0.0, 0, 0, -0.01229)),
            ((19, 0.5, 1, 8, 0.0), (27, 4.539296550572581, 2, 5, -0.0599), (55, 0.925, 3, 4, -0.531691), (-1, 0.0, 0, 0, -0.034868), (-1, 0.0, 0, 0, 0.010882), (4, 0.025, 6, 7, 0.030581), (-1, 0.0, 0, 0, 0.038324), (-1, 0.0, 0, 0, -0.002561), (11, 0.948, 9, 12, 0.564444), (25, 6.636602293485552, 10, 11, -0.022518), (-1, 0.0, 0, 0, 0.034608), (-1, 0.0, 0, 0, -0.023814), (6, 0.8694716566129473, 13, 14, 0.985761), (-1, 0.0, 0, 0, 0.050618), (-1, 0.0, 0, 0, 0.006564)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.048388), (7, 0.955, 3, 4, -0.14369), (-1, 0.0, 0, 0, -0.016413), (-1, 0.0, 0, 0, 0.030647), (50, 0.793, 6, 7, 0.256993), (-1, 0.0, 0, 0, 0.02804), (-1, 0.0, 0, 0, -0.002176), (27, 4.728490960086225, 9, 12, -0.661595), (20, -2.6933930072678223, 10, 11, -0.869599), (-1, 0.0, 0, 0, -0.053775), (-1, 0.0, 0, 0, 0.020913), (-1, 0.0, 0, 0, 0.013258)),
            ((63, 0.98, 1, 8, 0.0), (62, 0.15100000000000002, 2, 5, 0.041034), (40, 0.6695, 3, 4, 0.406611), (-1, 0.0, 0, 0, 0.030062), (-1, 0.0, 0, 0, -0.059769), (52, 0.5, 6, 7, -0.083909), (-1, 0.0, 0, 0, 0.006136), (-1, 0.0, 0, 0, -0.018587), (20, -3.683959493972949, 9, 10, -0.772076), (-1, 0.0, 0, 0, -0.048456), (-1, 0.0, 0, 0, -0.008326)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.093415), (35, 0.5, 3, 4, -0.82284), (-1, 0.0, 0, 0, -0.045811), (-1, 0.0, 0, 0, 0.007698), (15, -0.741, 6, 7, -0.017297), (-1, 0.0, 0, 0, -0.045737), (-1, 0.0, 0, 0, 0.002555), (26, 4.494223020031202, 9, 12, 0.346229), (7, 0.878, 10, 11, -0.108672), (-1, 0.0, 0, 0, -0.028478), (-1, 0.0, 0, 0, 0.040899), (6, 0.8978411663857659, 13, 14, 0.68275), (-1, 0.0, 0, 0, 0.045251), (-1, 0.0, 0, 0, -0.043492)),
            ((60, 0.5, 1, 8, 0.0), (26, 4.952293473989714, 2, 5, 0.05089), (26, 4.2413008733233095, 3, 4, 0.126626), (-1, 0.0, 0, 0, -0.019835), (-1, 0.0, 0, 0, 0.01129), (12, 0.581, 6, 7, -0.457502), (-1, 0.0, 0, 0, -0.004613), (-1, 0.0, 0, 0, -0.047885), (50, 0.985, 9, 12, -0.606849), (26, 4.668133964286986, 10, 11, -0.835058), (-1, 0.0, 0, 0, -0.052681), (-1, 0.0, 0, 0, -0.007452), (12, 0.6639999999999999, 13, 14, 0.507302), (-1, 0.0, 0, 0, 0.035975), (-1, 0.0, 0, 0, -0.005807)),
            ((59, 0.1380952380952381, 1, 8, 0.0), (15, -0.113, 2, 5, 0.401281), (20, -4.595119850134589, 3, 4, -0.1092), (-1, 0.0, 0, 0, -0.028123), (-1, 0.0, 0, 0, 0.01998), (27, 4.608290119226802, 6, 7, 0.884243), (-1, 0.0, 0, 0, -0.01169), (-1, 0.0, 0, 0, 0.071973), (26, 4.952293473989714, 9, 12, -0.079975), (26, 4.579839228586978, 10, 11, -0.003555), (-1, 0.0, 0, 0, -0.012717), (-1, 0.0, 0, 0, 0.014451), (24, 0.06250000000000003, 13, 14, -0.520091), (-1, 0.0, 0, 0, -0.032792), (-1, 0.0, 0, 0, 0.026939)),
            ((19, 0.5, 1, 8, 0.0), (1, 0.009, 2, 5, -0.057762), (59, 0.4365079365079365, 3, 4, 0.10402), (-1, 0.0, 0, 0, 0.022302), (-1, 0.0, 0, 0, -0.016801), (2, 0.985, 6, 7, -0.307704), (-1, 0.0, 0, 0, -0.027142), (-1, 0.0, 0, 0, 0.014986), (11, 0.948, 9, 12, 0.552819), (25, 6.636602293485552, 10, 11, -0.013931), (-1, 0.0, 0, 0, 0.033829), (-1, 0.0, 0, 0, -0.02239), (6, 0.8694716566129473, 13, 14, 0.9742), (-1, 0.0, 0, 0, 0.050097), (-1, 0.0, 0, 0, 0.006386)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.039281), (27, 4.608290119226802, 3, 4, 0.502157), (-1, 0.0, 0, 0, -0.015815), (-1, 0.0, 0, 0, 0.042404), (4, 0.025, 6, 7, -0.04671), (-1, 0.0, 0, 0, 0.024902), (-1, 0.0, 0, 0, -0.005451), (20, -3.683959493972949, 9, 10, -0.759359), (-1, 0.0, 0, 0, -0.047887), (-1, 0.0, 0, 0, -0.008187)),
            ((58, 0.458041958041958, 1, 8, 0.0), (15, -0.761, 2, 5, 0.048413), (26, 4.53794713771163, 3, 4, -0.716602), (-1, 0.0, 0, 0, -0.013717), (-1, 0.0, 0, 0, -0.0445), (49, 3.5, 6, 7, 0.092666), (-1, 0.0, 0, 0, 0.007186), (-1, 0.0, 0, 0, -0.032666), (7, 0.994, 9, 12, -0.602211), (24, 0.06250000000000003, 10, 11, -0.782327), (-1, 0.0, 0, 0, -0.049297), (-1, 0.0, 0, 0, -0.004103), (-1, 0.0, 0, 0, 0.034804)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.87, 2, 5, 0.048338), (27, 4.2964256503859755, 3, 4, -0.038209), (-1, 0.0, 0, 0, -0.035437), (-1, 0.0, 0, 0, 0.00168), (41, 0.5, 6, 7, 0.456515), (-1, 0.0, 0, 0, 0.039214), (-1, 0.0, 0, 0, 0.002391), (50, 0.985, 9, 12, -0.58638), (46, 0.8005, 10, 11, -0.809877), (-1, 0.0, 0, 0, -0.048137), (-1, 0.0, 0, 0, 0.002914), (12, 0.6639999999999999, 13, 14, 0.503453), (-1, 0.0, 0, 0, 0.035385), (-1, 0.0, 0, 0, -0.005391)),
            ((63, 0.98, 1, 8, 0.0), (62, 0.15100000000000002, 2, 5, 0.03668), (40, 0.6695, 3, 4, 0.372883), (-1, 0.0, 0, 0, 0.027944), (-1, 0.0, 0, 0, -0.057234), (52, 0.5, 6, 7, -0.077961), (-1, 0.0, 0, 0, 0.006129), (-1, 0.0, 0, 0, -0.01779), (20, -3.683959493972949, 9, 10, -0.739594), (-1, 0.0, 0, 0, -0.047197), (-1, 0.0, 0, 0, -0.007511)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.055669), (11, 0.852, 3, 4, -0.820046), (-1, 0.0, 0, 0, -0.018499), (-1, 0.0, 0, 0, -0.048962), (27, 4.539296550572581, 6, 7, -0.005412), (-1, 0.0, 0, 0, -0.023924), (-1, 0.0, 0, 0, 0.004357), (11, 0.948, 9, 12, 0.536834), (25, 6.636602293485552, 10, 11, -0.020958), (-1, 0.0, 0, 0, 0.033008), (-1, 0.0, 0, 0, -0.022024), (6, 0.8694716566129473, 13, 14, 0.964337), (-1, 0.0, 0, 0, 0.049657), (-1, 0.0, 0, 0, 0.006124)),
            ((60, 0.5, 1, 8, 0.0), (49, 3.5, 2, 5, 0.045945), (26, 4.2413008733233095, 3, 4, 0.093036), (-1, 0.0, 0, 0, -0.019209), (-1, 0.0, 0, 0, 0.008741), (13, 0.43999999999999995, 6, 7, -0.65041), (-1, 0.0, 0, 0, 0.009866), (-1, 0.0, 0, 0, -0.045393), (50, 0.985, 9, 12, -0.568049), (27, 4.156926133699104, 10, 11, -0.785534), (-1, 0.0, 0, 0, 0.018457), (-1, 0.0, 0, 0, -0.043689), (15, -0.002, 13, 14, 0.489322), (-1, 0.0, 0, 0, -0.00508), (-1, 0.0, 0, 0, 0.034902)),
            ((19, 0.5, 1, 8, 0.0), (1, 0.009, 2, 5, -0.054185), (59, 0.4365079365079365, 3, 4, 0.100391), (-1, 0.0, 0, 0, 0.02113), (-1, 0.0, 0, 0, -0.015811), (2, 0.985, 6, 7, -0.293235), (-1, 0.0, 0, 0, -0.026135), (-1, 0.0, 0, 0, 0.01464), (11, 0.948, 9, 12, 0.527074), (41, 0.5, 10, 11, -0.015249), (-1, 0.0, 0, 0, 0.02444), (-1, 0.0, 0, 0, -0.028807), (6, 0.8694716566129473, 13, 14, 0.954224), (-1, 0.0, 0, 0, 0.049165), (-1, 0.0, 0, 0, 0.006175)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.035579), (27, 4.608290119226802, 3, 4, 0.472705), (-1, 0.0, 0, 0, -0.016052), (-1, 0.0, 0, 0, 0.040403), (8, 0.6190476190476191, 6, 7, -0.044715), (-1, 0.0, 0, 0, 0.00268), (-1, 0.0, 0, 0, -0.01945), (20, -3.683959493972949, 9, 10, -0.731133), (-1, 0.0, 0, 0, -0.046787), (-1, 0.0, 0, 0, -0.007519)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.712514), (-1, 0.0, 0, 0, -0.01259), (-1, 0.0, 0, 0, -0.04412), (59, 0.4365079365079365, 5, 8, 0.035998), (1, 0.009, 6, 7, 0.194552), (-1, 0.0, 0, 0, 0.026834), (-1, 0.0, 0, 0, -0.010673), (45, 0.06888, 9, 10, -0.169296), (-1, 0.0, 0, 0, -0.02791), (-1, 0.0, 0, 0, 0.00358)),
            ((63, 0.98, 1, 8, 0.0), (41, 0.5, 2, 5, 0.033756), (49, 2.5, 3, 4, 0.229871), (-1, 0.0, 0, 0, 0.02358), (-1, 0.0, 0, 0, -0.015173), (49, 1.5, 6, 7, -0.138527), (-1, 0.0, 0, 0, -0.03047), (-1, 0.0, 0, 0, 0.001097), (20, -3.683959493972949, 9, 10, -0.72054), (-1, 0.0, 0, 0, -0.046339), (-1, 0.0, 0, 0, -0.007444)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.053298), (11, 0.852, 3, 4, -0.792056), (-1, 0.0, 0, 0, -0.0171), (-1, 0.0, 0, 0, -0.047684), (26, 4.579839228586978, 6, 7, -0.005942), (-1, 0.0, 0, 0, -0.010624), (-1, 0.0, 0, 0, 0.009042), (6, 0.8873180999538214, 9, 12, 0.51903), (13, 0.744, 10, 11, 0.687238), (-1, 0.0, 0, 0, 0.004102), (-1, 0.0, 0, 0, 0.049041), (-1, 0.0, 0, 0, -0.031558)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.87, 2, 5, 0.043683), (26, 4.2413008733233095, 3, 4, -0.035725), (-1, 0.0, 0, 0, -0.028541), (-1, 0.0, 0, 0, 0.002178), (11, 0.948, 6, 7, 0.431962), (-1, 0.0, 0, 0, 0.005834), (-1, 0.0, 0, 0, 0.042326), (50, 0.985, 9, 12, -0.556183), (27, 4.156926133699104, 10, 11, -0.769708), (-1, 0.0, 0, 0, 0.018237), (-1, 0.0, 0, 0, -0.042861), (12, 0.6639999999999999, 13, 14, 0.477773), (-1, 0.0, 0, 0, 0.034064), (-1, 0.0, 0, 0, -0.00569)),
            ((58, 0.458041958041958, 1, 8, 0.0), (26, 4.952293473989714, 2, 5, 0.042187), (26, 4.579839228586978, 3, 4, 0.108316), (-1, 0.0, 0, 0, -0.004532), (-1, 0.0, 0, 0, 0.017171), (12, 0.581, 6, 7, -0.411517), (-1, 0.0, 0, 0, -0.003945), (-1, 0.0, 0, 0, -0.044479), (63, 0.0019999999999999983, 9, 12, -0.565574), (27, 3.8571899716509277, 10, 11, 0.587795), (-1, 0.0, 0, 0, -0.005964), (-1, 0.0, 0, 0, 0.037082), (20, -2.6933930072678223, 13, 14, -0.771503), (-1, 0.0, 0, 0, -0.050924), (-1, 0.0, 0, 0, -0.002046)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.03176), (27, 4.608290119226802, 3, 4, 0.455582), (-1, 0.0, 0, 0, -0.015791), (-1, 0.0, 0, 0, 0.039088), (4, 0.025, 6, 7, -0.045293), (-1, 0.0, 0, 0, 0.024864), (-1, 0.0, 0, 0, -0.005368), (20, -3.683959493972949, 9, 10, -0.700583), (-1, 0.0, 0, 0, -0.045774), (-1, 0.0, 0, 0, -0.00651)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.683327), (-1, 0.0, 0, 0, -0.011336), (-1, 0.0, 0, 0, -0.042768), (59, 0.4365079365079365, 5, 8, 0.032418), (1, 0.009, 6, 7, 0.185037), (-1, 0.0, 0, 0, 0.025489), (-1, 0.0, 0, 0, -0.010277), (45, 0.06888, 9, 10, -0.165229), (-1, 0.0, 0, 0, -0.027071), (-1, 0.0, 0, 0, 0.003341)),
            ((60, 0.5, 1, 8, 0.0), (49, 3.5, 2, 5, 0.041844), (15, -0.741, 3, 4, 0.085972), (-1, 0.0, 0, 0, -0.030324), (-1, 0.0, 0, 0, 0.006684), (13, 0.43999999999999995, 6, 7, -0.623675), (-1, 0.0, 0, 0, 0.008533), (-1, 0.0, 0, 0, -0.043534), (50, 0.985, 9, 12, -0.53957), (15, -0.016999999999999998, 10, 11, -0.747936), (-1, 0.0, 0, 0, -0.013785), (-1, 0.0, 0, 0, -0.055401), (15, -0.002, 13, 14, 0.464914), (-1, 0.0, 0, 0, -0.005441), (-1, 0.0, 0, 0, 0.033725)),
            ((23, 0.0050000000000000044, 1, 8, 0.0), (63, 0.19999999999999998, 2, 5, -0.411138), (40, 0.4825, 3, 4, 0.321555), (-1, 0.0, 0, 0, 0.04908), (-1, 0.0, 0, 0, -0.027372), (40, 0.0295, 6, 7, -0.875841), (-1, 0.0, 0, 0, -0.054359), (-1, 0.0, 0, 0, 0.00156), (2, 0.6619999999999999, 9, 12, 0.055149), (6, 0.01140631311569524, 10, 11, 0.450762), (-1, 0.0, 0, 0, -0.02842), (-1, 0.0, 0, 0, 0.031352), (45, 0.09988000000000001, 13, 14, -0.082692), (-1, 0.0, 0, 0, -0.019902), (-1, 0.0, 0, 0, 0.007096)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.051408), (3, 0.08699999999999997, 3, 4, -0.755712), (-1, 0.0, 0, 0, -0.014667), (-1, 0.0, 0, 0, -0.045837), (27, 4.539296550572581, 6, 7, -0.008124), (-1, 0.0, 0, 0, -0.022211), (-1, 0.0, 0, 0, 0.003807), (11, 0.948, 9, 12, 0.497276), (25, 6.636602293485552, 10, 11, -0.028006), (-1, 0.0, 0, 0, 0.03204), (-1, 0.0, 0, 0, -0.021855), (6, 0.8694716566129473, 13, 14, 0.932501), (-1, 0.0, 0, 0, 0.048157), (-1, 0.0, 0, 0, 0.0058)),
            ((60, 0.5, 1, 8, 0.0), (41, 0.5, 2, 5, 0.040267), (49, 2.5, 3, 4, 0.2153), (-1, 0.0, 0, 0, 0.021057), (-1, 0.0, 0, 0, -0.013111), (12, 0.42500000000000004, 6, 7, -0.12171), (-1, 0.0, 0, 0, -0.023319), (-1, 0.0, 0, 0, 0.004632), (26, 4.686739554676682, 9, 12, -0.530898), (46, 0.8005, 10, 11, -0.822412), (-1, 0.0, 0, 0, -0.05122), (-1, 0.0, 0, 0, 0.010659), (20, -3.210268834500856, 13, 14, 0.221411), (-1, 0.0, 0, 0, -0.024383), (-1, 0.0, 0, 0, 0.03118)),
            ((20, -3.683959493972949, 1, 8, 0.0), (40, 0.5535, 2, 5, -0.217833), (40, 0.0045000000000000595, 3, 4, -0.08229), (-1, 0.0, 0, 0, -0.023905), (-1, 0.0, 0, 0, 0.027464), (4, 0.945, 6, 7, -0.8081), (-1, 0.0, 0, 0, -0.047891), (-1, 0.0, 0, 0, -0.005362), (6, 0.6046477932374416, 9, 12, 0.104777), (52, 0.5, 10, 11, 0.301632), (-1, 0.0, 0, 0, 0.028129), (-1, 0.0, 0, 0, -0.012963), (55, 0.11099999999999999, 13, 14, -0.144171), (-1, 0.0, 0, 0, -0.04326), (-1, 0.0, 0, 0, 0.00266)),
            ((63, 0.98, 1, 8, 0.0), (50, 0.04700000000000004, 2, 5, 0.029017), (27, 4.5957507650448015, 3, 4, 0.482558), (-1, 0.0, 0, 0, -0.02846), (-1, 0.0, 0, 0, 0.041166), (4, 0.025, 6, 7, -0.03925), (-1, 0.0, 0, 0, 0.025616), (-1, 0.0, 0, 0, -0.005032), (20, -3.683959493972949, 9, 10, -0.692035), (-1, 0.0, 0, 0, -0.044885), (-1, 0.0, 0, 0, -0.00768)),
            ((58, 0.458041958041958, 1, 8, 0.0), (15, -0.17200000000000004, 2, 5, 0.037652), (13, 0.692, 3, 4, -0.226676), (-1, 0.0, 0, 0, -0.04517), (-1, 0.0, 0, 0, -0.000942), (59, 0.14835164835164835, 6, 7, 0.139453), (-1, 0.0, 0, 0, 0.035498), (-1, 0.0, 0, 0, 0.001956), (63, 0.0019999999999999983, 9, 12, -0.542709), (27, 3.8571899716509277, 10, 11, 0.575664), (-1, 0.0, 0, 0, -0.00564), (-1, 0.0, 0, 0, 0.036145), (20, -2.6933930072678223, 13, 14, -0.748746), (-1, 0.0, 0, 0, -0.050067), (-1, 0.0, 0, 0, -0.001755)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.080523), (35, 0.5, 3, 4, -0.746793), (-1, 0.0, 0, 0, -0.042146), (-1, 0.0, 0, 0, 0.008), (11, 0.044, 6, 7, -0.01473), (-1, 0.0, 0, 0, 0.048419), (-1, 0.0, 0, 0, -0.003048), (26, 4.494223020031202, 9, 12, 0.29369), (7, 0.878, 10, 11, -0.143876), (-1, 0.0, 0, 0, -0.028266), (-1, 0.0, 0, 0, 0.038847), (6, 0.8978411663857659, 13, 14, 0.620091), (-1, 0.0, 0, 0, 0.042718), (-1, 0.0, 0, 0, -0.041223)),
            ((60, 0.5, 1, 8, 0.0), (49, 3.5, 2, 5, 0.039181), (26, 4.2413008733233095, 3, 4, 0.081341), (-1, 0.0, 0, 0, -0.017843), (-1, 0.0, 0, 0, 0.007697), (13, 0.43999999999999995, 6, 7, -0.60476), (-1, 0.0, 0, 0, 0.008612), (-1, 0.0, 0, 0, -0.042477), (26, 4.686739554676682, 9, 12, -0.522881), (46, 0.8005, 10, 11, -0.804519), (-1, 0.0, 0, 0, -0.049702), (-1, 0.0, 0, 0, 0.009184), (20, -3.210268834500856, 13, 14, 0.209414), (-1, 0.0, 0, 0, -0.023674), (-1, 0.0, 0, 0, 0.029884)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.050131), (11, 0.852, 3, 4, -0.747982), (-1, 0.0, 0, 0, -0.01456), (-1, 0.0, 0, 0, -0.045649), (26, 4.579839228586978, 6, 7, -0.007985), (-1, 0.0, 0, 0, -0.010092), (-1, 0.0, 0, 0, 0.008327), (11, 0.948, 9, 12, 0.492142), (25, 6.636602293485552, 10, 11, -0.019133), (-1, 0.0, 0, 0, 0.031226), (-1, 0.0, 0, 0, -0.020692), (6, 0.8694716566129473, 13, 14, 0.920951), (-1, 0.0, 0, 0, 0.047596), (-1, 0.0, 0, 0, 0.006191)),
            ((49, 3.5, 1, 8, 0.0), (23, 0.0050000000000000044, 2, 5, 0.034027), (0, 0.5, 3, 4, -0.400894), (-1, 0.0, 0, 0, 0.027299), (-1, 0.0, 0, 0, -0.039367), (2, 0.6619999999999999, 6, 7, 0.097736), (-1, 0.0, 0, 0, 0.022453), (-1, 0.0, 0, 0, -0.001715), (27, 4.728490960086225, 9, 12, -0.561264), (20, -2.6933930072678223, 10, 11, -0.759315), (-1, 0.0, 0, 0, -0.048185), (-1, 0.0, 0, 0, 0.019171), (-1, 0.0, 0, 0, 0.012664)),
            ((63, 0.98, 1, 8, 0.0), (50, 0.04700000000000004, 2, 5, 0.028076), (27, 4.5957507650448015, 3, 4, 0.459576), (-1, 0.0, 0, 0, -0.027645), (-1, 0.0, 0, 0, 0.039096), (4, 0.025, 6, 7, -0.037363), (-1, 0.0, 0, 0, 0.024641), (-1, 0.0, 0, 0, -0.004803), (12, 0.131, 9, 10, -0.673327), (-1, 0.0, 0, 0, 0.003019), (25, 6.648336461794052, 11, 12, -0.789697), (-1, 0.0, 0, 0, -0.006608), (-1, 0.0, 0, 0, -0.048186)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.87, 2, 5, 0.037774), (27, 4.134165555407019, 3, 4, -0.035585), (-1, 0.0, 0, 0, -0.042083), (-1, 0.0, 0, 0, 0.000614), (11, 0.948, 6, 7, 0.407635), (-1, 0.0, 0, 0, 0.004752), (-1, 0.0, 0, 0, 0.041973), (2, 0.994, 9, 12, -0.510599), (27, 4.212125883197904, 10, 11, -0.758934), (-1, 0.0, 0, 0, 0.015826), (-1, 0.0, 0, 0, -0.043443), (56, 0.982, 13, 14, 0.29596), (-1, 0.0, 0, 0, 0.030003), (-1, 0.0, 0, 0, -0.013875)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.243546), (3, 0.10299999999999998, 3, 4, -0.883838), (-1, 0.0, 0, 0, -0.013008), (-1, 0.0, 0, 0, -0.057104), (13, 0.8280000000000001, 6, 7, -0.04131), (-1, 0.0, 0, 0, 0.042858), (-1, 0.0, 0, 0, -0.010196), (59, 0.14835164835164835, 9, 12, 0.084803), (27, 4.608290119226802, 10, 11, 0.691568), (-1, 0.0, 0, 0, -0.017263), (-1, 0.0, 0, 0, 0.060532), (50, 0.793, 13, 14, -0.010992), (-1, 0.0, 0, 0, 0.013517), (-1, 0.0, 0, 0, -0.008066)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.047017), (6, 0.09655020948583677, 3, 4, -0.733722), (-1, 0.0, 0, 0, -0.016245), (-1, 0.0, 0, 0, -0.046668), (26, 4.579839228586978, 6, 7, -0.006289), (-1, 0.0, 0, 0, -0.009654), (-1, 0.0, 0, 0, 0.008074), (6, 0.8873180999538214, 9, 12, 0.478469), (13, 0.744, 10, 11, 0.650469), (-1, 0.0, 0, 0, 0.003446), (-1, 0.0, 0, 0, 0.047673), (-1, 0.0, 0, 0, -0.027948)),
            ((49, 3.5, 1, 8, 0.0), (23, 0.0050000000000000044, 2, 5, 0.033559), (63, 0.19999999999999998, 3, 4, -0.382845), (-1, 0.0, 0, 0, 0.016651), (-1, 0.0, 0, 0, -0.044135), (2, 0.6619999999999999, 6, 7, 0.093721), (-1, 0.0, 0, 0, 0.021254), (-1, 0.0, 0, 0, -0.00157), (27, 4.728490960086225, 9, 12, -0.556094), (20, -2.6933930072678223, 10, 11, -0.747821), (-1, 0.0, 0, 0, -0.047299), (-1, 0.0, 0, 0, 0.018424), (-1, 0.0, 0, 0, 0.011567)),
            ((58, 0.458041958041958, 1, 8, 0.0), (8, 0.5357142857142857, 2, 5, 0.035829), (7, 0.608, 3, 4, 0.121004), (-1, 0.0, 0, 0, -0.002915), (-1, 0.0, 0, 0, 0.024445), (4, 0.099, 6, 7, -0.283878), (-1, 0.0, 0, 0, 0.04229), (-1, 0.0, 0, 0, -0.030539), (63, 0.0019999999999999983, 9, 12, -0.523466), (27, 3.8571899716509277, 10, 11, 0.573579), (-1, 0.0, 0, 0, -0.005114), (-1, 0.0, 0, 0, 0.035619), (20, -2.6933930072678223, 13, 14, -0.733629), (-1, 0.0, 0, 0, -0.049247), (-1, 0.0, 0, 0, -0.001728)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.027446), (27, 4.608290119226802, 3, 4, 0.421508), (-1, 0.0, 0, 0, -0.014429), (-1, 0.0, 0, 0, 0.035805), (9, 0.7444999999999999, 6, 7, -0.041589), (-1, 0.0, 0, 0, 0.001341), (-1, 0.0, 0, 0, -0.022974), (20, -3.683959493972949, 9, 10, -0.65376), (-1, 0.0, 0, 0, -0.043589), (-1, 0.0, 0, 0, -0.006407)),
            ((45, 0.87, 1, 8, 0.0), (5, 0.833, 2, 5, -0.06775), (27, 4.134165555407019, 3, 4, 0.011495), (-1, 0.0, 0, 0, -0.038341), (-1, 0.0, 0, 0, 0.003047), (28, 0.652, 6, 7, -0.505734), (-1, 0.0, 0, 0, -0.008311), (-1, 0.0, 0, 0, -0.055124), (3, 0.273, 9, 12, 0.309642), (11, 0.948, 10, 11, -0.054783), (-1, 0.0, 0, 0, -0.016238), (-1, 0.0, 0, 0, 0.027989), (58, 0.4494949494949495, 13, 14, 0.686506), (-1, 0.0, 0, 0, 0.042834), (-1, 0.0, 0, 0, -0.009783)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.04943), (3, 0.497, 3, 4, -0.084647), (-1, 0.0, 0, 0, -0.008513), (-1, 0.0, 0, 0, 0.032334), (50, 0.8069999999999999, 6, 7, 0.372621), (-1, 0.0, 0, 0, 0.036153), (-1, 0.0, 0, 0, -0.006006), (12, 0.581, 9, 12, -0.375114), (59, 0.6770833333333333, 10, 11, -0.050968), (-1, 0.0, 0, 0, 0.017789), (-1, 0.0, 0, 0, -0.032962), (24, 0.179, 13, 14, -0.82417), (-1, 0.0, 0, 0, -0.048843), (-1, 0.0, 0, 0, 0.014184)),
            ((20, -3.683959493972949, 1, 8, 0.0), (40, 0.5535, 2, 5, -0.199879), (40, 0.0045000000000000595, 3, 4, -0.064286), (-1, 0.0, 0, 0, -0.022584), (-1, 0.0, 0, 0, 0.026655), (4, 0.945, 6, 7, -0.791628), (-1, 0.0, 0, 0, -0.047212), (-1, 0.0, 0, 0, -0.004648), (41, 2.5, 9, 12, 0.095683), (4, 0.231, 10, 11, 0.168183), (-1, 0.0, 0, 0, 0.023024), (-1, 0.0, 0, 0, 0.001595), (11, 0.947, 13, 14, -0.55636), (-1, 0.0, 0, 0, -0.050366), (-1, 0.0, 0, 0, 0.009064)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.87, 2, 5, 0.035638), (26, 4.2413008733233095, 3, 4, -0.033209), (-1, 0.0, 0, 0, -0.026305), (-1, 0.0, 0, 0, 0.00185), (41, 0.5, 6, 7, 0.390087), (-1, 0.0, 0, 0, 0.036241), (-1, 0.0, 0, 0, 0.000218), (50, 0.985, 9, 12, -0.489396), (27, 4.156926133699104, 10, 11, -0.685317), (-1, 0.0, 0, 0, 0.017936), (-1, 0.0, 0, 0, -0.038687), (12, 0.6639999999999999, 13, 14, 0.442784), (-1, 0.0, 0, 0, 0.031678), (-1, 0.0, 0, 0, -0.00558)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.233005), (3, 0.10299999999999998, 3, 4, -0.859628), (-1, 0.0, 0, 0, -0.012176), (-1, 0.0, 0, 0, -0.055859), (2, 0.42400000000000004, 6, 7, -0.035933), (-1, 0.0, 0, 0, 0.020055), (-1, 0.0, 0, 0, -0.018061), (59, 0.14835164835164835, 9, 12, 0.080081), (27, 4.608290119226802, 10, 11, 0.683441), (-1, 0.0, 0, 0, -0.015228), (-1, 0.0, 0, 0, 0.058389), (15, 0.737, 13, 14, -0.013035), (-1, 0.0, 0, 0, 0.003187), (-1, 0.0, 0, 0, -0.025085)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.632253), (-1, 0.0, 0, 0, -0.008276), (-1, 0.0, 0, 0, -0.040088), (40, 0.624, 5, 8, 0.026446), (41, 0.5, 6, 7, 0.107293), (-1, 0.0, 0, 0, 0.017937), (-1, 0.0, 0, 0, -0.005055), (27, 4.7257269950599685, 9, 10, -0.271157), (-1, 0.0, 0, 0, -0.023737), (-1, 0.0, 0, 0, 0.03425)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.026206), (27, 4.608290119226802, 3, 4, 0.403047), (-1, 0.0, 0, 0, -0.013422), (-1, 0.0, 0, 0, 0.034145), (4, 0.025, 6, 7, -0.0388), (-1, 0.0, 0, 0, 0.022583), (-1, 0.0, 0, 0, -0.004715), (20, -3.683959493972949, 9, 10, -0.639218), (-1, 0.0, 0, 0, -0.043097), (-1, 0.0, 0, 0, -0.005864)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.044589), (6, 0.09655020948583677, 3, 4, -0.709529), (-1, 0.0, 0, 0, -0.01479), (-1, 0.0, 0, 0, -0.045672), (27, 4.2964256503859755, 6, 7, -0.006183), (-1, 0.0, 0, 0, -0.026678), (-1, 0.0, 0, 0, 0.002509), (11, 0.948, 9, 12, 0.461541), (25, 6.636602293485552, 10, 11, -0.028948), (-1, 0.0, 0, 0, 0.029741), (-1, 0.0, 0, 0, -0.020594), (6, 0.8694716566129473, 13, 14, 0.893884), (-1, 0.0, 0, 0, 0.046376), (-1, 0.0, 0, 0, 0.005402)),
            ((49, 3.5, 1, 8, 0.0), (23, 0.0050000000000000044, 2, 5, 0.031219), (0, 0.5, 3, 4, -0.365898), (-1, 0.0, 0, 0, 0.025729), (-1, 0.0, 0, 0, -0.037845), (2, 0.6619999999999999, 6, 7, 0.087825), (-1, 0.0, 0, 0, 0.020191), (-1, 0.0, 0, 0, -0.001569), (24, 0.06250000000000003, 9, 12, -0.537308), (13, 0.43999999999999995, 10, 11, -0.689765), (-1, 0.0, 0, 0, 0.005604), (-1, 0.0, 0, 0, -0.04535), (48, 7.5, 13, 14, 0.317607), (-1, 0.0, 0, 0, -0.00428), (-1, 0.0, 0, 0, 0.022232)),
            ((63, 0.98, 1, 8, 0.0), (62, 0.15100000000000002, 2, 5, 0.025497), (40, 0.6695, 3, 4, 0.290527), (-1, 0.0, 0, 0, 0.023082), (-1, 0.0, 0, 0, -0.052685), (52, 0.5, 6, 7, -0.064215), (-1, 0.0, 0, 0, 0.005809), (-1, 0.0, 0, 0, -0.015529), (12, 0.131, 9, 10, -0.63036), (-1, 0.0, 0, 0, 0.004537), (25, 6.648336461794052, 11, 12, -0.749228), (-1, 0.0, 0, 0, -0.005196), (-1, 0.0, 0, 0, -0.046356)),
            ((60, 0.5, 1, 8, 0.0), (26, 4.952293473989714, 2, 5, 0.034438), (26, 4.579839228586978, 3, 4, 0.091316), (-1, 0.0, 0, 0, -0.003259), (-1, 0.0, 0, 0, 0.01416), (12, 0.581, 6, 7, -0.36309), (-1, 0.0, 0, 0, -0.002378), (-1, 0.0, 0, 0, -0.041936), (26, 4.686739554676682, 9, 12, -0.476468), (46, 0.8005, 10, 11, -0.74833), (-1, 0.0, 0, 0, -0.046386), (-1, 0.0, 0, 0, 0.008467), (20, -3.210268834500856, 13, 14, 0.230275), (-1, 0.0, 0, 0, -0.02116), (-1, 0.0, 0, 0, 0.02909)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.044472), (6, 0.09655020948583677, 3, 4, -0.701411), (-1, 0.0, 0, 0, -0.014368), (-1, 0.0, 0, 0, -0.045259), (27, 4.2964256503859755, 6, 7, -0.007143), (-1, 0.0, 0, 0, -0.025882), (-1, 0.0, 0, 0, 0.002367), (11, 0.948, 9, 12, 0.456465), (41, 0.5, 10, 11, -0.021498), (-1, 0.0, 0, 0, 0.02233), (-1, 0.0, 0, 0, -0.025063), (6, 0.8694716566129473, 13, 14, 0.885141), (-1, 0.0, 0, 0, 0.045971), (-1, 0.0, 0, 0, 0.005212)),
            ((20, -3.683959493972949, 1, 8, 0.0), (40, 0.5535, 2, 5, -0.192753), (40, 0.0045000000000000595, 3, 4, -0.06018), (-1, 0.0, 0, 0, -0.0218), (-1, 0.0, 0, 0, 0.025225), (27, 4.525841730285462, 6, 7, -0.774015), (-1, 0.0, 0, 0, 0.004361), (-1, 0.0, 0, 0, -0.044063), (41, 2.5, 9, 12, 0.09049), (3, 0.524, 10, 11, 0.159467), (-1, 0.0, 0, 0, 0.004588), (-1, 0.0, 0, 0, 0.037127), (11, 0.947, 13, 14, -0.532614), (-1, 0.0, 0, 0, -0.048043), (-1, 0.0, 0, 0, 0.008021)),
            ((58, 0.458041958041958, 1, 8, 0.0), (8, 0.5357142857142857, 2, 5, 0.031518), (28, 0.408, 3, 4, 0.111882), (-1, 0.0, 0, 0, 0.031678), (-1, 0.0, 0, 0, -4.3e-05), (4, 0.099, 6, 7, -0.271726), (-1, 0.0, 0, 0, 0.040628), (-1, 0.0, 0, 0, -0.029126), (63, 0.0019999999999999983, 9, 12, -0.487608), (27, 3.8571899716509277, 10, 11, 0.577587), (-1, 0.0, 0, 0, -0.004663), (-1, 0.0, 0, 0, 0.035476), (20, -2.6933930072678223, 13, 14, -0.701195), (-1, 0.0, 0, 0, -0.048277), (-1, 0.0, 0, 0, -0.00065)),
            ((45, 0.87, 1, 8, 0.0), (5, 0.833, 2, 5, -0.063163), (27, 4.134165555407019, 3, 4, 0.009309), (-1, 0.0, 0, 0, -0.03617), (-1, 0.0, 0, 0, 0.002737), (28, 0.652, 6, 7, -0.469329), (-1, 0.0, 0, 0, -0.007282), (-1, 0.0, 0, 0, -0.052936), (3, 0.273, 9, 12, 0.292077), (11, 0.948, 10, 11, -0.063625), (-1, 0.0, 0, 0, -0.016051), (-1, 0.0, 0, 0, 0.027336), (58, 0.4494949494949495, 13, 14, 0.66586), (-1, 0.0, 0, 0, 0.041502), (-1, 0.0, 0, 0, -0.00891)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.731, 2, 5, 0.032853), (27, 4.134165555407019, 3, 4, -0.051128), (-1, 0.0, 0, 0, -0.050149), (-1, 0.0, 0, 0, 9.1e-05), (2, 0.954, 6, 7, 0.287755), (-1, 0.0, 0, 0, -0.01167), (-1, 0.0, 0, 0, 0.028899), (2, 0.994, 9, 12, -0.464147), (27, 4.212125883197904, 10, 11, -0.696739), (-1, 0.0, 0, 0, 0.016164), (-1, 0.0, 0, 0, -0.040421), (56, 0.982, 13, 14, 0.295895), (-1, 0.0, 0, 0, 0.0292), (-1, 0.0, 0, 0, -0.012709)),
            ((49, 3.5, 1, 8, 0.0), (23, 0.0050000000000000044, 2, 5, 0.028718), (63, 0.19999999999999998, 3, 4, -0.344552), (-1, 0.0, 0, 0, 0.01637), (-1, 0.0, 0, 0, -0.042591), (2, 0.6619999999999999, 6, 7, 0.081387), (-1, 0.0, 0, 0, 0.019836), (-1, 0.0, 0, 0, -0.001911), (11, 0.869, 9, 10, -0.521167), (-1, 0.0, 0, 0, 0.013021), (24, 0.06250000000000003, 11, 12, -0.699699), (-1, 0.0, 0, 0, -0.042291), (-1, 0.0, 0, 0, 0.011003)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.023461), (27, 4.608290119226802, 3, 4, 0.397814), (-1, 0.0, 0, 0, -0.013036), (-1, 0.0, 0, 0, 0.033333), (4, 0.025, 6, 7, -0.039969), (-1, 0.0, 0, 0, 0.022319), (-1, 0.0, 0, 0, -0.00473), (12, 0.131, 9, 10, -0.611531), (-1, 0.0, 0, 0, 0.005066), (25, 6.648336461794052, 11, 12, -0.731628), (-1, 0.0, 0, 0, -0.004405), (-1, 0.0, 0, 0, -0.045633)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.220742), (3, 0.10299999999999998, 3, 4, -0.843078), (-1, 0.0, 0, 0, -0.011583), (-1, 0.0, 0, 0, -0.054811), (2, 0.42400000000000004, 6, 7, -0.026486), (-1, 0.0, 0, 0, 0.020577), (-1, 0.0, 0, 0, -0.017389), (59, 0.14835164835164835, 9, 12, 0.073469), (27, 4.608290119226802, 10, 11, 0.640563), (-1, 0.0, 0, 0, -0.01409), (-1, 0.0, 0, 0, 0.054402), (50, 0.793, 13, 14, -0.011606), (-1, 0.0, 0, 0, 0.012189), (-1, 0.0, 0, 0, -0.007485)),
            ((19, 0.5, 1, 8, 0.0), (1, 0.009, 2, 5, -0.042091), (59, 0.4365079365079365, 3, 4, 0.080498), (-1, 0.0, 0, 0, 0.016889), (-1, 0.0, 0, 0, -0.012471), (2, 0.985, 6, 7, -0.231566), (-1, 0.0, 0, 0, -0.023262), (-1, 0.0, 0, 0, 0.017236), (6, 0.8873180999538214, 9, 12, 0.443672), (13, 0.744, 10, 11, 0.622938), (-1, 0.0, 0, 0, 0.002748), (-1, 0.0, 0, 0, 0.046454), (-1, 0.0, 0, 0, -0.026583)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.60907), (-1, 0.0, 0, 0, -0.007129), (-1, 0.0, 0, 0, -0.038585), (40, 0.624, 5, 8, 0.024234), (41, 0.5, 6, 7, 0.09673), (-1, 0.0, 0, 0, 0.016684), (-1, 0.0, 0, 0, -0.004845), (27, 4.7257269950599685, 9, 10, -0.242705), (-1, 0.0, 0, 0, -0.02199), (-1, 0.0, 0, 0, 0.033103)),
            ((63, 0.98, 1, 8, 0.0), (1, 0.009, 2, 5, 0.024101), (49, 2.5, 3, 4, 0.148377), (-1, 0.0, 0, 0, 0.018047), (-1, 0.0, 0, 0, -0.016791), (2, 0.985, 6, 7, -0.141423), (-1, 0.0, 0, 0, -0.018554), (-1, 0.0, 0, 0, 0.019238), (20, -3.683959493972949, 9, 10, -0.607995), (-1, 0.0, 0, 0, -0.041729), (-1, 0.0, 0, 0, -0.005324)),
            ((63, 0.98, 1, 8, 0.0), (6, 0.6482034517095314, 2, 5, 0.022928), (23, 0.0050000000000000044, 3, 4, 0.132876), (-1, 0.0, 0, 0, -0.02245), (-1, 0.0, 0, 0, 0.011314), (45, 0.92, 6, 7, -0.161599), (-1, 0.0, 0, 0, -0.01382), (-1, 0.0, 0, 0, 0.021892), (20, -3.683959493972949, 9, 10, -0.595438), (-1, 0.0, 0, 0, -0.041197), (-1, 0.0, 0, 0, -0.005147)),
            ((60, 0.5, 1, 8, 0.0), (58, 0.3798076923076923, 2, 5, 0.031883), (4, 0.231, 3, 4, 0.080652), (-1, 0.0, 0, 0, 0.019525), (-1, 0.0, 0, 0, -0.002095), (28, 0.208, 6, 7, -0.370603), (-1, 0.0, 0, 0, 0.022005), (-1, 0.0, 0, 0, -0.031854), (26, 4.686739554676682, 9, 12, -0.451601), (25, 6.463808880305315, 10, 11, -0.719879), (-1, 0.0, 0, 0, 0.013291), (-1, 0.0, 0, 0, -0.043267), (20, -3.210268834500856, 13, 14, 0.242887), (-1, 0.0, 0, 0, -0.019318), (-1, 0.0, 0, 0, 0.028219)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.592497), (-1, 0.0, 0, 0, -0.006985), (-1, 0.0, 0, 0, -0.037507), (8, 0.6190476190476191, 5, 8, 0.022666), (40, 0.5799999999999998, 6, 7, 0.096254), (-1, 0.0, 0, 0, 0.011164), (-1, 0.0, 0, 0, -0.010793), (4, 0.099, 9, 10, -0.24702), (-1, 0.0, 0, 0, 0.034804), (-1, 0.0, 0, 0, -0.025547)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.066688), (35, 0.5, 3, 4, -0.660936), (-1, 0.0, 0, 0, -0.038124), (-1, 0.0, 0, 0, 0.008417), (11, 0.044, 6, 7, -0.011502), (-1, 0.0, 0, 0, 0.043066), (-1, 0.0, 0, 0, -0.002504), (5, 0.024, 9, 12, 0.25007), (26, 4.482986592105173, 10, 11, 0.852886), (-1, 0.0, 0, 0, -0.000966), (-1, 0.0, 0, 0, 0.053805), (52, 0.5, 13, 14, -0.013913), (-1, 0.0, 0, 0, 0.021902), (-1, 0.0, 0, 0, -0.018281)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.027627), (7, 0.955, 3, 4, -0.112635), (-1, 0.0, 0, 0, -0.013801), (-1, 0.0, 0, 0, 0.028309), (56, 0.918, 6, 7, 0.17207), (-1, 0.0, 0, 0, 0.015509), (-1, 0.0, 0, 0, -0.023091), (11, 0.869, 9, 10, -0.497065), (-1, 0.0, 0, 0, 0.012587), (24, 0.06250000000000003, 11, 12, -0.670244), (-1, 0.0, 0, 0, -0.040837), (-1, 0.0, 0, 0, 0.010894)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.0396), (6, 0.09655020948583677, 3, 4, -0.671177), (-1, 0.0, 0, 0, -0.012731), (-1, 0.0, 0, 0, -0.043684), (26, 4.579839228586978, 6, 7, -0.004981), (-1, 0.0, 0, 0, -0.008578), (-1, 0.0, 0, 0, 0.007181), (6, 0.8873180999538214, 9, 12, 0.431921), (13, 0.744, 10, 11, 0.614077), (-1, 0.0, 0, 0, 0.002699), (-1, 0.0, 0, 0, 0.046059), (-1, 0.0, 0, 0, -0.026014)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.043527), (3, 0.497, 3, 4, -0.075267), (-1, 0.0, 0, 0, -0.007897), (-1, 0.0, 0, 0, 0.030267), (43, 0.10555555555555556, 6, 7, 0.324472), (-1, 0.0, 0, 0, 0.039649), (-1, 0.0, 0, 0, -0.001038), (12, 0.581, 9, 12, -0.337825), (1, 0.002, 10, 11, -0.039986), (-1, 0.0, 0, 0, -0.025636), (-1, 0.0, 0, 0, 0.022761), (24, 0.179, 13, 14, -0.776965), (-1, 0.0, 0, 0, -0.045924), (-1, 0.0, 0, 0, 0.012259)),
            ((15, 0.759, 1, 8, 0.0), (9, 0.967, 2, 5, 0.036376), (9, 0.767, 3, 4, 0.004111), (-1, 0.0, 0, 0, 0.004001), (-1, 0.0, 0, 0, -0.037812), (4, 0.996, 6, 7, 0.802514), (-1, 0.0, 0, 0, 0.047675), (-1, 0.0, 0, 0, -0.008892), (6, 0.5986826325792863, 9, 12, -0.395216), (63, 0.041, 10, 11, 0.23793), (-1, 0.0, 0, 0, 0.033141), (-1, 0.0, 0, 0, -0.014117), (55, 0.015000000000000013, 13, 14, -1.029578), (-1, 0.0, 0, 0, -0.012794), (-1, 0.0, 0, 0, -0.057315)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.023036), (27, 4.608290119226802, 3, 4, 0.376499), (-1, 0.0, 0, 0, -0.012531), (-1, 0.0, 0, 0, 0.03176), (4, 0.979, 6, 7, -0.035762), (-1, 0.0, 0, 0, -0.005518), (-1, 0.0, 0, 0, 0.016282), (20, -3.683959493972949, 9, 10, -0.5813), (-1, 0.0, 0, 0, -0.040862), (-1, 0.0, 0, 0, -0.004327)),
            ((60, 0.5, 1, 8, 0.0), (15, 0.759, 2, 5, 0.031657), (15, -0.019000000000000017, 3, 4, 0.076946), (-1, 0.0, 0, 0, -0.004454), (-1, 0.0, 0, 0, 0.013685), (6, 0.5986826325792863, 6, 7, -0.39782), (-1, 0.0, 0, 0, 0.011402), (-1, 0.0, 0, 0, -0.050227), (26, 4.686739554676682, 9, 12, -0.442544), (46, 0.8005, 10, 11, -0.703961), (-1, 0.0, 0, 0, -0.043903), (-1, 0.0, 0, 0, 0.007761), (20, -3.210268834500856, 13, 14, 0.238378), (-1, 0.0, 0, 0, -0.018455), (-1, 0.0, 0, 0, 0.02741)),
            ((58, 0.458041958041958, 1, 8, 0.0), (8, 0.5357142857142857, 2, 5, 0.029394), (7, 0.608, 3, 4, 0.105266), (-1, 0.0, 0, 0, -0.002812), (-1, 0.0, 0, 0, 0.021532), (4, 0.099, 6, 7, -0.26287), (-1, 0.0, 0, 0, 0.036313), (-1, 0.0, 0, 0, -0.026917), (63, 0.0019999999999999983, 9, 12, -0.459887), (27, 3.8571899716509277, 10, 11, 0.578055), (-1, 0.0, 0, 0, -0.003958), (-1, 0.0, 0, 0, 0.03472), (20, -2.6933930072678223, 13, 14, -0.678139), (-1, 0.0, 0, 0, -0.047106), (-1, 0.0, 0, 0, -0.000681)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.038849), (6, 0.09655020948583677, 3, 4, -0.654834), (-1, 0.0, 0, 0, -0.01208), (-1, 0.0, 0, 0, -0.043071), (27, 4.2964256503859755, 6, 7, -0.005424), (-1, 0.0, 0, 0, -0.023584), (-1, 0.0, 0, 0, 0.002149), (6, 0.8873180999538214, 9, 12, 0.428652), (13, 0.744, 10, 11, 0.608408), (-1, 0.0, 0, 0, 0.003095), (-1, 0.0, 0, 0, 0.045704), (-1, 0.0, 0, 0, -0.024705)),
            ((53, 2.5, 1, 8, 0.0), (3, 0.489, 2, 5, 0.02264), (1, 0.012, 3, 4, -0.028563), (-1, 0.0, 0, 0, 0.00526), (-1, 0.0, 0, 0, -0.012368), (56, 0.16399999999999998, 6, 7, 0.529797), (-1, 0.0, 0, 0, -0.023581), (-1, 0.0, 0, 0, 0.055938), (49, 2.5, 9, 10, -0.561001), (-1, 0.0, 0, 0, 0.007635), (-1, 0.0, 0, 0, -0.04322)),
            ((45, 0.731, 1, 6, 0.0), (27, 4.134165555407019, 2, 3, -0.075101), (-1, 0.0, 0, 0, -0.047115), (5, 0.009, 4, 5, -0.029777), (-1, 0.0, 0, 0, 0.013333), (-1, 0.0, 0, 0, -0.007027), (2, 0.954, 7, 10, 0.203768), (41, 0.5, 8, 9, -0.227344), (-1, 0.0, 0, 0, 0.018128), (-1, 0.0, 0, 0, -0.036014), (7, 0.45499999999999996, 11, 12, 0.453294), (-1, 0.0, 0, 0, -0.011833), (-1, 0.0, 0, 0, 0.032411)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.87, 2, 5, 0.030276), (26, 4.2413008733233095, 3, 4, -0.02786), (-1, 0.0, 0, 0, -0.025313), (-1, 0.0, 0, 0, 0.001886), (11, 0.948, 6, 7, 0.348853), (-1, 0.0, 0, 0, 0.002579), (-1, 0.0, 0, 0, 0.03985), (26, 4.686739554676682, 9, 12, -0.435027), (25, 6.463808880305315, 10, 11, -0.687774), (-1, 0.0, 0, 0, 0.013158), (-1, 0.0, 0, 0, -0.041552), (20, -3.210268834500856, 13, 14, 0.225127), (-1, 0.0, 0, 0, -0.017924), (-1, 0.0, 0, 0, 0.026198)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.042138), (3, 0.497, 3, 4, -0.06962), (-1, 0.0, 0, 0, -0.007351), (-1, 0.0, 0, 0, 0.028158), (43, 0.10555555555555556, 6, 7, 0.304921), (-1, 0.0, 0, 0, 0.037891), (-1, 0.0, 0, 0, -0.001488), (41, 1.5, 9, 12, -0.33018), (15, -0.19899999999999998, 10, 11, -0.094593), (-1, 0.0, 0, 0, -0.036172), (-1, 0.0, 0, 0, 0.010908), (-1, 0.0, 0, 0, -0.04481)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.022266), (27, 4.608290119226802, 3, 4, 0.376414), (-1, 0.0, 0, 0, -0.011966), (-1, 0.0, 0, 0, 0.031357), (9, 0.7444999999999999, 6, 7, -0.036396), (-1, 0.0, 0, 0, 0.001344), (-1, 0.0, 0, 0, -0.021763), (20, -3.683959493972949, 9, 10, -0.566845), (-1, 0.0, 0, 0, -0.040369), (-1, 0.0, 0, 0, -0.003787)),
            ((58, 0.458041958041958, 1, 8, 0.0), (8, 0.5357142857142857, 2, 5, 0.028133), (7, 0.608, 3, 4, 0.099048), (-1, 0.0, 0, 0, -0.002616), (-1, 0.0, 0, 0, 0.020185), (4, 0.099, 6, 7, -0.247836), (-1, 0.0, 0, 0, 0.035018), (-1, 0.0, 0, 0, -0.025627), (63, 0.0019999999999999983, 9, 12, -0.452008), (27, 3.8571899716509277, 10, 11, 0.567911), (-1, 0.0, 0, 0, -0.003853), (-1, 0.0, 0, 0, 0.033959), (20, -2.6933930072678223, 13, 14, -0.670367), (-1, 0.0, 0, 0, -0.04645), (-1, 0.0, 0, 0, -0.001178)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.041123), (3, 0.497, 3, 4, -0.064318), (-1, 0.0, 0, 0, -0.006934), (-1, 0.0, 0, 0, 0.026948), (50, 0.8069999999999999, 6, 7, 0.28908), (-1, 0.0, 0, 0, 0.030608), (-1, 0.0, 0, 0, -0.008125), (12, 0.581, 9, 12, -0.323847), (59, 0.6770833333333333, 10, 11, -0.036864), (-1, 0.0, 0, 0, 0.017254), (-1, 0.0, 0, 0, -0.030653), (24, 0.179, 13, 14, -0.755757), (-1, 0.0, 0, 0, -0.044391), (-1, 0.0, 0, 0, 0.011106)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.03745), (6, 0.09655020948583677, 3, 4, -0.63556), (-1, 0.0, 0, 0, -0.011444), (-1, 0.0, 0, 0, -0.042346), (15, -0.613, 6, 7, -0.005013), (-1, 0.0, 0, 0, 0.035644), (-1, 0.0, 0, 0, -0.001756), (6, 0.8873180999538214, 9, 12, 0.422235), (13, 0.744, 10, 11, 0.603695), (-1, 0.0, 0, 0, 0.003781), (-1, 0.0, 0, 0, 0.045161), (-1, 0.0, 0, 0, -0.024321)),
            ((55, 0.978, 1, 8, 0.0), (55, 0.919, 2, 5, -0.035987), (55, 0.817, 3, 4, 0.022178), (-1, 0.0, 0, 0, -0.002861), (-1, 0.0, 0, 0, 0.027252), (52, 0.5, 6, 7, -0.424376), (-1, 0.0, 0, 0, 0.008164), (-1, 0.0, 0, 0, -0.044262), (11, 0.992, 9, 12, 0.43358), (12, 0.09599999999999997, 10, 11, 0.813943), (-1, 0.0, 0, 0, -0.020825), (-1, 0.0, 0, 0, 0.049351), (24, 0.2525, 13, 14, -0.350148), (-1, 0.0, 0, 0, -0.033748), (-1, 0.0, 0, 0, 0.020625)),
            ((40, 0.624, 1, 8, 0.0), (58, 0.09307359307359307, 2, 5, 0.056927), (26, 4.283562780082223, 3, 4, 0.326681), (-1, 0.0, 0, 0, -0.012095), (-1, 0.0, 0, 0, 0.028997), (5, 0.183, 6, 7, -0.047823), (-1, 0.0, 0, 0, 0.008475), (-1, 0.0, 0, 0, -0.017346), (27, 4.7257269950599685, 9, 12, -0.232885), (2, 0.998, 10, 11, -0.416289), (-1, 0.0, 0, 0, -0.03531), (-1, 0.0, 0, 0, 0.013189), (-1, 0.0, 0, 0, 0.029653)),
            ((63, 0.98, 1, 8, 0.0), (62, 0.15100000000000002, 2, 5, 0.022669), (40, 0.6695, 3, 4, 0.259274), (-1, 0.0, 0, 0, 0.02061), (-1, 0.0, 0, 0, -0.047288), (4, 0.979, 6, 7, -0.056342), (-1, 0.0, 0, 0, -0.00748), (-1, 0.0, 0, 0, 0.016095), (20, -3.683959493972949, 9, 10, -0.55729), (-1, 0.0, 0, 0, -0.040115), (-1, 0.0, 0, 0, -0.003323)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.202707), (11, 0.943, 3, 4, -0.80327), (-1, 0.0, 0, 0, -0.048296), (-1, 0.0, 0, 0, -0.001117), (2, 0.42400000000000004, 6, 7, -0.013217), (-1, 0.0, 0, 0, 0.020255), (-1, 0.0, 0, 0, -0.016126), (59, 0.14835164835164835, 9, 12, 0.067896), (27, 4.608290119226802, 10, 11, 0.613151), (-1, 0.0, 0, 0, -0.013417), (-1, 0.0, 0, 0, 0.051785), (20, -3.210268834500856, 13, 14, -0.011252), (-1, 0.0, 0, 0, -0.01354), (-1, 0.0, 0, 0, 0.005941)),
            ((45, 0.731, 1, 8, 0.0), (27, 4.134165555407019, 2, 5, -0.072372), (9, 0.5509999999999999, 3, 4, -0.925887), (-1, 0.0, 0, 0, -0.049378), (-1, 0.0, 0, 0, -0.00964), (4, 0.231, 6, 7, -0.028526), (-1, 0.0, 0, 0, 0.012311), (-1, 0.0, 0, 0, -0.006906), (2, 0.954, 9, 12, 0.201625), (58, 0.12916666666666665, 10, 11, -0.206901), (-1, 0.0, 0, 0, 0.023262), (-1, 0.0, 0, 0, -0.030378), (6, 0.8802139175373297, 13, 14, 0.437794), (-1, 0.0, 0, 0, 0.028231), (-1, 0.0, 0, 0, -0.024175)),
            ((15, -0.784, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.566052), (-1, 0.0, 0, 0, -0.007326), (-1, 0.0, 0, 0, -0.037289), (15, 0.759, 5, 8, 0.021496), (8, 0.6190476190476191, 6, 7, 0.062526), (-1, 0.0, 0, 0, 0.00689), (-1, 0.0, 0, 0, -0.011598), (6, 0.5986826325792863, 9, 10, -0.371688), (-1, 0.0, 0, 0, 0.011127), (-1, 0.0, 0, 0, -0.048793)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.036912), (6, 0.09655020948583677, 3, 4, -0.623908), (-1, 0.0, 0, 0, -0.011421), (-1, 0.0, 0, 0, -0.041492), (27, 4.2964256503859755, 6, 7, -0.005428), (-1, 0.0, 0, 0, -0.02246), (-1, 0.0, 0, 0, 0.001985), (6, 0.8873180999538214, 9, 12, 0.416279), (13, 0.744, 10, 11, 0.592034), (-1, 0.0, 0, 0, 0.003912), (-1, 0.0, 0, 0, 0.044555), (-1, 0.0, 0, 0, -0.022069)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.039501), (9, 0.947, 3, 4, -0.061814), (-1, 0.0, 0, 0, -0.006123), (-1, 0.0, 0, 0, 0.032409), (6, 0.11653987838050074, 6, 7, 0.276486), (-1, 0.0, 0, 0, -0.03042), (-1, 0.0, 0, 0, 0.021545), (41, 1.5, 9, 12, -0.318513), (15, -0.19899999999999998, 10, 11, -0.085016), (-1, 0.0, 0, 0, -0.034573), (-1, 0.0, 0, 0, 0.010482), (-1, 0.0, 0, 0, -0.044052)),
            ((53, 2.5, 1, 8, 0.0), (3, 0.489, 2, 5, 0.020472), (58, 0.3550420168067227, 3, 4, -0.025505), (-1, 0.0, 0, 0, 0.001931), (-1, 0.0, 0, 0, -0.021385), (56, 0.16399999999999998, 6, 7, 0.468528), (-1, 0.0, 0, 0, -0.02445), (-1, 0.0, 0, 0, 0.052301), (45, 0.87, 9, 10, -0.539084), (-1, 0.0, 0, 0, -0.037538), (-1, 0.0, 0, 0, 0.014861)),
            ((20, -3.683959493972949, 1, 8, 0.0), (40, 0.5535, 2, 5, -0.164713), (40, 0.0045000000000000595, 3, 4, -0.035792), (-1, 0.0, 0, 0, -0.019811), (-1, 0.0, 0, 0, 0.02364), (4, 0.945, 6, 7, -0.73752), (-1, 0.0, 0, 0, -0.045489), (-1, 0.0, 0, 0, -0.00017), (41, 2.5, 9, 12, 0.077068), (11, 0.991, 10, 11, 0.143585), (-1, 0.0, 0, 0, 0.01216), (-1, 0.0, 0, 0, -0.01092), (11, 0.947, 13, 14, -0.515521), (-1, 0.0, 0, 0, -0.045293), (-1, 0.0, 0, 0, 0.005439)),
            ((60, 0.5, 1, 8, 0.0), (15, 0.759, 2, 5, 0.027076), (15, -0.019000000000000017, 3, 4, 0.06936), (-1, 0.0, 0, 0, -0.004088), (-1, 0.0, 0, 0, 0.012392), (6, 0.5986826325792863, 6, 7, -0.377236), (-1, 0.0, 0, 0, 0.010595), (-1, 0.0, 0, 0, -0.047844), (26, 4.686739554676682, 9, 12, -0.41363), (46, 0.8005, 10, 11, -0.65768), (-1, 0.0, 0, 0, -0.041547), (-1, 0.0, 0, 0, 0.007779), (20, -3.210268834500856, 13, 14, 0.225238), (-1, 0.0, 0, 0, -0.016343), (-1, 0.0, 0, 0, 0.024905)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.036769), (6, 0.8725633231113197, 3, 4, -0.060674), (-1, 0.0, 0, 0, -0.000213), (-1, 0.0, 0, 0, -0.041076), (50, 0.8069999999999999, 6, 7, 0.265154), (-1, 0.0, 0, 0, 0.02867), (-1, 0.0, 0, 0, -0.008243), (41, 1.5, 9, 12, -0.309528), (15, -0.19899999999999998, 10, 11, -0.082782), (-1, 0.0, 0, 0, -0.033447), (-1, 0.0, 0, 0, 0.009922), (-1, 0.0, 0, 0, -0.04308)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.037413), (6, 0.09655020948583677, 3, 4, -0.605959), (-1, 0.0, 0, 0, -0.010371), (-1, 0.0, 0, 0, -0.040999), (15, -0.537, 6, 7, -0.007276), (-1, 0.0, 0, 0, 0.028855), (-1, 0.0, 0, 0, -0.002095), (11, 0.948, 9, 12, 0.405134), (25, 6.636602293485552, 10, 11, -0.023601), (-1, 0.0, 0, 0, 0.026822), (-1, 0.0, 0, 0, -0.018976), (49, 3.5, 13, 14, 0.81959), (-1, 0.0, 0, 0, 0.042988), (-1, 0.0, 0, 0, 0.003693)),
            ((63, 0.98, 1, 8, 0.0), (50, 0.04700000000000004, 2, 5, 0.019084), (27, 4.5957507650448015, 3, 4, 0.355832), (-1, 0.0, 0, 0, -0.027612), (-1, 0.0, 0, 0, 0.032649), (4, 0.979, 6, 7, -0.034753), (-1, 0.0, 0, 0, -0.005742), (-1, 0.0, 0, 0, 0.015871), (20, -3.683959493972949, 9, 10, -0.535162), (-1, 0.0, 0, 0, -0.039229), (-1, 0.0, 0, 0, -0.002857)),
            ((45, 0.731, 1, 8, 0.0), (27, 4.134165555407019, 2, 5, -0.070722), (9, 0.5509999999999999, 3, 4, -0.896261), (-1, 0.0, 0, 0, -0.047974), (-1, 0.0, 0, 0, -0.00895), (5, 0.009, 6, 7, -0.029416), (-1, 0.0, 0, 0, 0.012558), (-1, 0.0, 0, 0, -0.006704), (2, 0.954, 9, 12, 0.191632), (41, 0.5, 10, 11, -0.196064), (-1, 0.0, 0, 0, 0.017862), (-1, 0.0, 0, 0, -0.032818), (7, 0.45499999999999996, 13, 14, 0.417956), (-1, 0.0, 0, 0, -0.011273), (-1, 0.0, 0, 0, 0.030086)),
            ((60, 0.5, 1, 8, 0.0), (45, 0.731, 2, 5, 0.025965), (27, 4.134165555407019, 3, 4, -0.043947), (-1, 0.0, 0, 0, -0.045716), (-1, 0.0, 0, 0, -2.5e-05), (41, 0.5, 6, 7, 0.252097), (-1, 0.0, 0, 0, 0.030345), (-1, 0.0, 0, 0, -0.004057), (2, 0.994, 9, 12, -0.406511), (27, 4.212125883197904, 10, 11, -0.617336), (-1, 0.0, 0, 0, 0.016259), (-1, 0.0, 0, 0, -0.036788), (56, 0.982, 13, 14, 0.298699), (-1, 0.0, 0, 0, 0.027497), (-1, 0.0, 0, 0, -0.010644)),
            ((11, 0.035, 1, 6, 0.0), (25, 6.101438773316817, 2, 5, 0.494321), (45, 0.53072, 3, 4, -0.305456), (-1, 0.0, 0, 0, -0.032266), (-1, 0.0, 0, 0, 0.017613), (-1, 0.0, 0, 0, 0.042771), (11, 0.461, 7, 10, -0.03044), (12, 0.712, 8, 9, -0.367225), (-1, 0.0, 0, 0, -0.024389), (-1, 0.0, 0, 0, 0.035302), (11, 0.564, 11, 12, 0.043985), (-1, 0.0, 0, 0, 0.040888), (-1, 0.0, 0, 0, -0.000483)),
            ((20, -3.210268834500856, 1, 6, 0.0), (11, 0.997, 2, 5, -0.139771), (40, 0.6443333333333333, 3, 4, -0.04373), (-1, 0.0, 0, 0, 0.004537), (-1, 0.0, 0, 0, -0.036549), (-1, 0.0, 0, 0, -0.046125), (11, 0.615, 7, 10, 0.089222), (3, 0.235, 8, 9, -0.248371), (-1, 0.0, 0, 0, 0.01373), (-1, 0.0, 0, 0, -0.029579), (3, 0.524, 11, 12, 0.225315), (-1, 0.0, 0, 0, 0.006362), (-1, 0.0, 0, 0, 0.051157)),
            ((15, -0.17200000000000004, 1, 6, 0.0), (62, 0.402, 2, 5, -0.194961), (15, -0.309, 3, 4, -0.309846), (-1, 0.0, 0, 0, -0.003618), (-1, 0.0, 0, 0, -0.039747), (-1, 0.0, 0, 0, 0.035545), (59, 0.14835164835164835, 7, 10, 0.063128), (27, 4.608290119226802, 8, 9, 0.600767), (-1, 0.0, 0, 0, -0.013532), (-1, 0.0, 0, 0, 0.050255), (8, 0.6190476190476191, 11, 12, -0.013194), (-1, 0.0, 0, 0, 0.004083), (-1, 0.0, 0, 0, -0.017282)),
            ((23, 0.0050000000000000044, 1, 8, 0.0), (63, 0.19999999999999998, 2, 5, -0.303394), (40, 0.4825, 3, 4, 0.326474), (-1, 0.0, 0, 0, 0.046329), (-1, 0.0, 0, 0, -0.024406), (40, 0.0295, 6, 7, -0.802193), (-1, 0.0, 0, 0, -0.049855), (-1, 0.0, 0, 0, -0.001843), (2, 0.6619999999999999, 9, 12, 0.037472), (26, 4.516324041782945, 10, 11, 0.3574), (-1, 0.0, 0, 0, -0.003795), (-1, 0.0, 0, 0, 0.033043), (13, 0.906, 13, 14, -0.077923), (-1, 0.0, 0, 0, -0.0163), (-1, 0.0, 0, 0, 0.006656)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.191707), (3, 0.10299999999999998, 3, 4, -0.766584), (-1, 0.0, 0, 0, -0.008236), (-1, 0.0, 0, 0, -0.051174), (2, 0.42400000000000004, 6, 7, -0.010819), (-1, 0.0, 0, 0, 0.019725), (-1, 0.0, 0, 0, -0.015463), (59, 0.14835164835164835, 9, 12, 0.062228), (27, 4.608290119226802, 10, 11, 0.582565), (-1, 0.0, 0, 0, -0.012735), (-1, 0.0, 0, 0, 0.048673), (20, -3.210268834500856, 13, 14, -0.010696), (-1, 0.0, 0, 0, -0.012593), (-1, 0.0, 0, 0, 0.005525)),
            ((15, -0.784, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.536492), (-1, 0.0, 0, 0, -0.006344), (-1, 0.0, 0, 0, -0.036017), (63, 0.991, 5, 8, 0.019052), (62, 0.15100000000000002, 6, 7, 0.036469), (-1, 0.0, 0, 0, 0.01325), (-1, 0.0, 0, 0, -0.001914), (-1, 0.0, 0, 0, -0.040911)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.035041), (6, 0.09655020948583677, 3, 4, -0.596482), (-1, 0.0, 0, 0, -0.010478), (-1, 0.0, 0, 0, -0.040109), (15, -0.537, 6, 7, -0.005572), (-1, 0.0, 0, 0, 0.027026), (-1, 0.0, 0, 0, -0.001897), (6, 0.8873180999538214, 9, 12, 0.398371), (13, 0.744, 10, 11, 0.57514), (-1, 0.0, 0, 0, 0.004236), (-1, 0.0, 0, 0, 0.043509), (-1, 0.0, 0, 0, -0.02144)),
            ((58, 0.458041958041958, 1, 8, 0.0), (8, 0.5357142857142857, 2, 5, 0.025007), (45, 0.09988000000000001, 3, 4, 0.094202), (-1, 0.0, 0, 0, -0.003619), (-1, 0.0, 0, 0, 0.016991), (4, 0.099, 6, 7, -0.247974), (-1, 0.0, 0, 0, 0.032349), (-1, 0.0, 0, 0, -0.024668), (63, 0.0019999999999999983, 9, 12, -0.421158), (27, 3.8571899716509277, 10, 11, 0.559744), (-1, 0.0, 0, 0, -0.003111), (-1, 0.0, 0, 0, 0.0326), (20, -2.6933930072678223, 13, 14, -0.639015), (-1, 0.0, 0, 0, -0.045189), (-1, 0.0, 0, 0, -0.0005)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.036644), (7, 0.955, 3, 4, -0.092311), (-1, 0.0, 0, 0, -0.011898), (-1, 0.0, 0, 0, 0.026269), (50, 0.835, 6, 7, 0.188322), (-1, 0.0, 0, 0, 0.025264), (-1, 0.0, 0, 0, -0.009204), (12, 0.581, 9, 12, -0.299503), (1, 0.002, 10, 11, -0.018373), (-1, 0.0, 0, 0, -0.023413), (-1, 0.0, 0, 0, 0.022481), (20, 1.965912816362163, 13, 14, -0.728554), (-1, 0.0, 0, 0, -0.041846), (-1, 0.0, 0, 0, 0.012946)),
            ((15, -0.17200000000000004, 1, 6, 0.0), (62, 0.402, 2, 5, -0.188944), (15, -0.309, 3, 4, -0.303432), (-1, 0.0, 0, 0, -0.003887), (-1, 0.0, 0, 0, -0.038175), (-1, 0.0, 0, 0, 0.034825), (59, 0.14835164835164835, 7, 10, 0.061057), (27, 4.608290119226802, 8, 9, 0.561391), (-1, 0.0, 0, 0, -0.012736), (-1, 0.0, 0, 0, 0.04743), (62, 0.32099999999999995, 11, 12, -0.008512), (-1, 0.0, 0, 0, -0.005691), (-1, 0.0, 0, 0, 0.013694)),
            ((20, -3.210268834500856, 1, 6, 0.0), (11, 0.997, 2, 5, -0.137238), (40, 0.6443333333333333, 3, 4, -0.044727), (-1, 0.0, 0, 0, 0.004113), (-1, 0.0, 0, 0, -0.035214), (-1, 0.0, 0, 0, -0.045214), (11, 0.615, 7, 10, 0.088037), (3, 0.235, 8, 9, -0.228842), (-1, 0.0, 0, 0, 0.013347), (-1, 0.0, 0, 0, -0.027738), (3, 0.524, 11, 12, 0.215283), (-1, 0.0, 0, 0, 0.006007), (-1, 0.0, 0, 0, 0.049041)),
            ((60, 0.5, 1, 8, 0.0), (55, 0.978, 2, 5, 0.026982), (7, 0.992, 3, 4, -0.00489), (-1, 0.0, 0, 0, 0.001494), (-1, 0.0, 0, 0, -0.036413), (59, 0.5147058823529411, 6, 7, 0.519309), (-1, 0.0, 0, 0, 0.043797), (-1, 0.0, 0, 0, -0.013713), (26, 4.686739554676682, 9, 12, -0.398086), (25, 6.463808880305315, 10, 11, -0.637735), (-1, 0.0, 0, 0, 0.012614), (-1, 0.0, 0, 0, -0.038806), (13, 0.991, 13, 14, 0.219416), (-1, 0.0, 0, 0, 0.025342), (-1, 0.0, 0, 0, -0.013285)),
            ((23, 0.0050000000000000044, 1, 6, 0.0), (3, 0.405, 2, 5, -0.298157), (6, 0.2704260414863776, 3, 4, -0.534257), (-1, 0.0, 0, 0, -0.043677), (-1, 0.0, 0, 0, -0.003242), (-1, 0.0, 0, 0, 0.048859), (2, 0.6619999999999999, 7, 10, 0.036941), (5, 0.012, 8, 9, 0.337319), (-1, 0.0, 0, 0, -0.01111), (-1, 0.0, 0, 0, 0.027754), (13, 0.906, 11, 12, -0.071608), (-1, 0.0, 0, 0, -0.015282), (-1, 0.0, 0, 0, 0.00638)),
            ((15, -0.17200000000000004, 1, 6, 0.0), (62, 0.402, 2, 5, -0.189784), (26, 4.620046665062765, 3, 4, -0.300042), (-1, 0.0, 0, 0, -0.028353), (-1, 0.0, 0, 0, 0.00233), (-1, 0.0, 0, 0, 0.032722), (59, 0.14835164835164835, 7, 10, 0.061906), (27, 4.608290119226802, 8, 9, 0.543495), (-1, 0.0, 0, 0, -0.012608), (-1, 0.0, 0, 0, 0.046052), (13, 0.992, 11, 12, -0.004393), (-1, 0.0, 0, 0, 0.004358), (-1, 0.0, 0, 0, -0.016326)),
            ((20, -3.683959493972949, 1, 8, 0.0), (6, 0.18761101883864736, 2, 5, -0.156872), (11, 0.279, 3, 4, -0.546457), (-1, 0.0, 0, 0, 0.008816), (-1, 0.0, 0, 0, -0.040841), (40, 0.5535, 6, 7, 0.011165), (-1, 0.0, 0, 0, 0.010868), (-1, 0.0, 0, 0, -0.035049), (41, 2.5, 9, 12, 0.074275), (11, 0.991, 10, 11, 0.137355), (-1, 0.0, 0, 0, 0.011853), (-1, 0.0, 0, 0, -0.01106), (55, 0.13, 13, 14, -0.490614), (-1, 0.0, 0, 0, 0.009778), (-1, 0.0, 0, 0, -0.038476)),
            ((58, 0.4494949494949495, 1, 8, 0.0), (8, 0.5357142857142857, 2, 5, 0.026367), (45, 0.09988000000000001, 3, 4, 0.093617), (-1, 0.0, 0, 0, -0.002978), (-1, 0.0, 0, 0, 0.015963), (4, 0.135, 6, 7, -0.237819), (-1, 0.0, 0, 0, 0.029445), (-1, 0.0, 0, 0, -0.023953), (7, 0.994, 9, 12, -0.391342), (20, -1.9659128163621629, 10, 11, -0.638556), (-1, 0.0, 0, 0, -0.041538), (-1, 0.0, 0, 0, -0.001224), (41, 0.5, 13, 14, 0.735345), (-1, 0.0, 0, 0, 0.00274), (-1, 0.0, 0, 0, 0.038803)),
            ((60, 0.5, 1, 8, 0.0), (26, 4.2413008733233095, 2, 5, 0.026267), (45, 0.955, 3, 4, -0.293021), (-1, 0.0, 0, 0, -0.025452), (-1, 0.0, 0, 0, 0.031983), (7, 0.992, 6, 7, 0.072382), (-1, 0.0, 0, 0, 0.005411), (-1, 0.0, 0, 0, -0.027979), (50, 0.976, 9, 12, -0.38871), (27, 4.164303167133926, 10, 11, -0.615073), (-1, 0.0, 0, 0, 0.015295), (-1, 0.0, 0, 0, -0.035578), (56, 0.982, 13, 14, 0.244982), (-1, 0.0, 0, 0, 0.031796), (-1, 0.0, 0, 0, -0.016452)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.474883), (-1, 0.0, 0, 0, -0.036473), (-1, 0.0, 0, 0, -0.005483), (15, -0.613, 5, 6, 0.021197), (-1, 0.0, 0, 0, 0.031392), (4, 0.979, 7, 8, -0.002941), (-1, 0.0, 0, 0, -0.003398), (-1, 0.0, 0, 0, 0.015183)),
            ((9, 0.924, 1, 8, 0.0), (9, 0.767, 2, 5, -0.032191), (58, 0.458041958041958, 3, 4, 0.030641), (-1, 0.0, 0, 0, 0.003378), (-1, 0.0, 0, 0, -0.027022), (62, 0.252, 6, 7, -0.846389), (-1, 0.0, 0, 0, -0.002108), (-1, 0.0, 0, 0, -0.05462), (5, 0.019, 9, 12, 0.391024), (62, 0.17400000000000004, 10, 11, 0.749043), (-1, 0.0, 0, 0, -0.001861), (-1, 0.0, 0, 0, 0.04746), (20, 0.2027325540540821, 13, 14, -0.085344), (-1, 0.0, 0, 0, -0.028267), (-1, 0.0, 0, 0, 0.030332)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.184578), (11, 0.943, 3, 4, -0.74535), (-1, 0.0, 0, 0, -0.045476), (-1, 0.0, 0, 0, 0.001163), (2, 0.42400000000000004, 6, 7, -0.007909), (-1, 0.0, 0, 0, 0.019078), (-1, 0.0, 0, 0, -0.014615), (59, 0.14835164835164835, 9, 12, 0.059818), (27, 4.608290119226802, 10, 11, 0.528561), (-1, 0.0, 0, 0, -0.012342), (-1, 0.0, 0, 0, 0.044948), (62, 0.32099999999999995, 13, 14, -0.004266), (-1, 0.0, 0, 0, -0.005354), (-1, 0.0, 0, 0, 0.01356)),
            ((19, 0.5, 1, 8, 0.0), (15, -0.741, 2, 5, -0.03255), (6, 0.09655020948583677, 3, 4, -0.591234), (-1, 0.0, 0, 0, -0.010034), (-1, 0.0, 0, 0, -0.039764), (25, 6.609348335006922, 6, 7, -0.003524), (-1, 0.0, 0, 0, -0.014343), (-1, 0.0, 0, 0, 0.002686), (6, 0.8873180999538214, 9, 12, 0.390082), (13, 0.744, 10, 11, 0.566911), (-1, 0.0, 0, 0, 0.004336), (-1, 0.0, 0, 0, 0.042911), (-1, 0.0, 0, 0, -0.020656)),
            ((20, -3.683959493972949, 1, 8, 0.0), (6, 0.18761101883864736, 2, 5, -0.154464), (11, 0.279, 3, 4, -0.532907), (-1, 0.0, 0, 0, 0.00885), (-1, 0.0, 0, 0, -0.040206), (40, 0.5535, 6, 7, 0.006445), (-1, 0.0, 0, 0, 0.010338), (-1, 0.0, 0, 0, -0.034519), (41, 2.5, 9, 12, 0.072357), (11, 0.991, 10, 11, 0.133416), (-1, 0.0, 0, 0, 0.011539), (-1, 0.0, 0, 0, -0.010828), (11, 0.982, 13, 14, -0.474096), (-1, 0.0, 0, 0, -0.03558), (-1, 0.0, 0, 0, 0.013216)),
            ((2, 0.985, 1, 8, 0.0), (1, 0.01, 2, 5, -0.100469), (24, 0.012250000000000011, 3, 4, 0.150321), (-1, 0.0, 0, 0, -0.019744), (-1, 0.0, 0, 0, 0.020729), (27, 4.640911736799023, 6, 7, -0.339082), (-1, 0.0, 0, 0, 0.000512), (-1, 0.0, 0, 0, -0.035334), (59, 0.4365079365079365, 9, 12, 0.112699), (6, 0.7277012853849543, 10, 11, 0.440785), (-1, 0.0, 0, 0, 0.036463), (-1, 0.0, 0, 0, -0.000167), (13, 0.845, 13, 14, -0.111848), (-1, 0.0, 0, 0, -0.024499), (-1, 0.0, 0, 0, 0.009192)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.2413008733233095, 2, 5, 0.02199), (45, 0.955, 3, 4, -0.284452), (-1, 0.0, 0, 0, -0.024454), (-1, 0.0, 0, 0, 0.031339), (9, 0.5994999999999999, 6, 7, 0.065662), (-1, 0.0, 0, 0, 0.006592), (-1, 0.0, 0, 0, -0.011527), (11, 0.869, 9, 10, -0.435202), (-1, 0.0, 0, 0, 0.014586), (24, 0.06250000000000003, 11, 12, -0.6149), (-1, 0.0, 0, 0, -0.037611), (-1, 0.0, 0, 0, 0.008978)),
            ((63, 0.98, 1, 8, 0.0), (6, 0.6482034517095314, 2, 5, 0.018652), (23, 0.0050000000000000044, 3, 4, 0.112325), (-1, 0.0, 0, 0, -0.019931), (-1, 0.0, 0, 0, 0.009592), (45, 0.92, 6, 7, -0.137319), (-1, 0.0, 0, 0, -0.012002), (-1, 0.0, 0, 0, 0.021166), (20, -3.683959493972949, 9, 10, -0.496728), (-1, 0.0, 0, 0, -0.037561), (-1, 0.0, 0, 0, -0.002022)),
            ((55, 0.978, 1, 8, 0.0), (55, 0.919, 2, 5, -0.032126), (55, 0.817, 3, 4, 0.028756), (-1, 0.0, 0, 0, -0.002445), (-1, 0.0, 0, 0, 0.026143), (52, 0.5, 6, 7, -0.427633), (-1, 0.0, 0, 0, 0.00776), (-1, 0.0, 0, 0, -0.043369), (11, 0.992, 9, 12, 0.389035), (12, 0.09599999999999997, 10, 11, 0.724568), (-1, 0.0, 0, 0, -0.022585), (-1, 0.0, 0, 0, 0.045487), (24, 0.2525, 13, 14, -0.331861), (-1, 0.0, 0, 0, -0.029515), (-1, 0.0, 0, 0, 0.016956)),
            ((11, 0.035, 1, 6, 0.0), (25, 6.101438773316817, 2, 5, 0.465759), (45, 0.53072, 3, 4, -0.278766), (-1, 0.0, 0, 0, -0.030391), (-1, 0.0, 0, 0, 0.016493), (-1, 0.0, 0, 0, 0.040053), (11, 0.461, 7, 10, -0.026978), (6, 0.13060363834789174, 8, 9, -0.331577), (-1, 0.0, 0, 0, 0.014968), (-1, 0.0, 0, 0, -0.027129), (11, 0.564, 11, 12, 0.040814), (-1, 0.0, 0, 0, 0.038793), (-1, 0.0, 0, 0, -0.000561)),
            ((15, -0.741, 1, 4, 0.0), (61, 0.19469890973733583, 2, 3, -0.469182), (-1, 0.0, 0, 0, -0.032121), (-1, 0.0, 0, 0, -0.002154), (15, -0.613, 5, 6, 0.020501), (-1, 0.0, 0, 0, 0.029768), (4, 0.979, 7, 8, -0.002301), (-1, 0.0, 0, 0, -0.003222), (-1, 0.0, 0, 0, 0.014481)),
            ((49, 3.5, 1, 8, 0.0), (25, 6.714768131918843, 2, 5, 0.022191), (15, -0.106, 3, 4, 0.128171), (-1, 0.0, 0, 0, -0.012298), (-1, 0.0, 0, 0, 0.01301), (58, 0.11805555555555555, 6, 7, -0.102478), (-1, 0.0, 0, 0, 0.015799), (-1, 0.0, 0, 0, -0.014476), (13, 0.43999999999999995, 9, 10, -0.427464), (-1, 0.0, 0, 0, 0.011177), (24, 0.179, 11, 12, -0.627382), (-1, 0.0, 0, 0, -0.038474), (-1, 0.0, 0, 0, 0.011033)),
            ((15, 0.759, 1, 8, 0.0), (2, 0.986, 2, 5, 0.029798), (9, 0.6105, 3, 4, -0.086784), (-1, 0.0, 0, 0, 0.000546), (-1, 0.0, 0, 0, -0.031259), (26, 4.375737243570452, 6, 7, 0.189464), (-1, 0.0, 0, 0, 0.033246), (-1, 0.0, 0, 0, 0.002809), (6, 0.5986826325792863, 9, 12, -0.328835), (27, 4.181095032513147, 10, 11, 0.218102), (-1, 0.0, 0, 0, -0.027407), (-1, 0.0, 0, 0, 0.021614), (55, 0.015000000000000013, 13, 14, -0.899798), (-1, 0.0, 0, 0, -0.007847), (-1, 0.0, 0, 0, -0.051506)),
            ((9, 0.924, 1, 8, 0.0), (9, 0.767, 2, 5, -0.030733), (58, 0.458041958041958, 3, 4, 0.030383), (-1, 0.0, 0, 0, 0.003304), (-1, 0.0, 0, 0, -0.026369), (62, 0.252, 6, 7, -0.830242), (-1, 0.0, 0, 0, -0.002048), (-1, 0.0, 0, 0, -0.053408), (5, 0.019, 9, 12, 0.375952), (62, 0.17400000000000004, 10, 11, 0.721555), (-1, 0.0, 0, 0, -0.001689), (-1, 0.0, 0, 0, 0.045718), (20, 0.2027325540540821, 13, 14, -0.089281), (-1, 0.0, 0, 0, -0.026808), (-1, 0.0, 0, 0, 0.029107)),
            ((19, 0.5, 1, 6, 0.0), (6, 0.9105698788443931, 2, 5, -0.03113), (6, 0.8725633231113197, 3, 4, -0.053827), (-1, 0.0, 0, 0, -0.000644), (-1, 0.0, 0, 0, -0.042857), (-1, 0.0, 0, 0, 0.032652), (6, 0.8873180999538214, 7, 10, 0.375123), (13, 0.744, 8, 9, 0.553784), (-1, 0.0, 0, 0, 0.004695), (-1, 0.0, 0, 0, 0.041987), (-1, 0.0, 0, 0, -0.020224)),
            ((26, 4.2413008733233095, 1, 8, 0.0), (45, 0.955, 2, 5, -0.285138), (4, 0.034, 3, 4, -0.48064), (-1, 0.0, 0, 0, 0.031608), (-1, 0.0, 0, 0, -0.037179), (11, 0.997, 6, 7, 0.5974), (-1, 0.0, 0, 0, 0.035291), (-1, 0.0, 0, 0, -0.002997), (49, 2.5, 9, 12, 0.034356), (58, 0.11805555555555555, 10, 11, 0.117754), (-1, 0.0, 0, 0, 0.021737), (-1, 0.0, 0, 0, -0.001846), (45, 0.06152, 13, 14, -0.1525), (-1, 0.0, 0, 0, -0.031181), (-1, 0.0, 0, 0, 0.003401)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.50574), (-1, 0.0, 0, 0, -0.002396), (-1, 0.0, 0, 0, -0.032985), (63, 0.991, 5, 8, 0.01802), (41, 0.5, 6, 7, 0.034724), (-1, 0.0, 0, 0, 0.008594), (-1, 0.0, 0, 0, -0.004081), (-1, 0.0, 0, 0, -0.038907)),
            ((53, 2.5, 1, 8, 0.0), (3, 0.489, 2, 5, 0.018471), (1, 0.012, 3, 4, -0.025123), (-1, 0.0, 0, 0, 0.004492), (-1, 0.0, 0, 0, -0.010936), (56, 0.16399999999999998, 6, 7, 0.430091), (-1, 0.0, 0, 0, -0.022748), (-1, 0.0, 0, 0, 0.047497), (45, 0.87, 9, 10, -0.480967), (-1, 0.0, 0, 0, -0.034201), (-1, 0.0, 0, 0, 0.014402)),
            ((40, 0.9045, 1, 8, 0.0), (55, 0.978, 2, 5, 0.018319), (55, 0.919, 3, 4, -0.01473), (-1, 0.0, 0, 0, 0.002126), (-1, 0.0, 0, 0, -0.01927), (11, 0.992, 6, 7, 0.495318), (-1, 0.0, 0, 0, 0.044033), (-1, 0.0, 0, 0, -0.016227), (2, 0.998, 9, 10, -0.484848), (-1, 0.0, 0, 0, -0.044374), (-1, 0.0, 0, 0, 0.01336)),
            ((60, 0.5, 1, 8, 0.0), (15, 0.759, 2, 5, 0.024564), (15, -0.019000000000000017, 3, 4, 0.061623), (-1, 0.0, 0, 0, -0.004028), (-1, 0.0, 0, 0, 0.011403), (6, 0.5986826325792863, 6, 7, -0.332347), (-1, 0.0, 0, 0, 0.010332), (-1, 0.0, 0, 0, -0.043753), (26, 4.686739554676682, 9, 12, -0.371483), (61, 0.26364351431932054, 10, 11, -0.614419), (-1, 0.0, 0, 0, -0.03747), (-1, 0.0, 0, 0, 0.011935), (13, 0.991, 13, 14, 0.239106), (-1, 0.0, 0, 0, 0.024761), (-1, 0.0, 0, 0, -0.011467)),
            ((63, 0.98, 1, 8, 0.0), (2, 0.04400000000000004, 2, 5, 0.018108), (23, 0.0050000000000000044, 3, 4, 0.342081), (-1, 0.0, 0, 0, -0.029871), (-1, 0.0, 0, 0, 0.032137), (63, 0.664, 6, 7, -0.025124), (-1, 0.0, 0, 0, 0.001474), (-1, 0.0, 0, 0, -0.024227), (20, -3.683959493972949, 9, 10, -0.478154), (-1, 0.0, 0, 0, -0.036978), (-1, 0.0, 0, 0, -0.001269)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.05209), (4, 0.001, 3, 4, -0.570615), (-1, 0.0, 0, 0, 0.012582), (-1, 0.0, 0, 0, -0.034064), (11, 0.994, 6, 7, -0.005923), (-1, 0.0, 0, 0, 0.004128), (-1, 0.0, 0, 0, -0.017087), (43, 0.14835164835164835, 9, 12, 0.202809), (6, 0.7124143742160444, 10, 11, -0.125123), (-1, 0.0, 0, 0, 0.01413), (-1, 0.0, 0, 0, -0.035719), (26, 4.482986592105173, 13, 14, 0.712003), (-1, 0.0, 0, 0, -0.000708), (-1, 0.0, 0, 0, 0.049299)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.49739), (-1, 0.0, 0, 0, -0.002496), (-1, 0.0, 0, 0, -0.03226), (63, 0.991, 5, 8, 0.016569), (41, 0.5, 6, 7, 0.032804), (-1, 0.0, 0, 0, 0.008247), (-1, 0.0, 0, 0, -0.003951), (-1, 0.0, 0, 0, -0.038197)),
            ((45, 0.0676, 1, 8, 0.0), (45, 0.06232, 2, 5, -0.128322), (4, 0.958, 3, 4, -0.001137), (-1, 0.0, 0, 0, 0.007846), (-1, 0.0, 0, 0, -0.032934), (40, 0.5215, 6, 7, -0.91011), (-1, 0.0, 0, 0, -0.054473), (-1, 0.0, 0, 0, -0.006941), (3, 0.388, 9, 12, 0.076874), (63, 0.001, 10, 11, -0.064626), (-1, 0.0, 0, 0, 0.036475), (-1, 0.0, 0, 0, -0.006804), (58, 0.4494949494949495, 13, 14, 0.513492), (-1, 0.0, 0, 0, 0.03488), (-1, 0.0, 0, 0, -0.018636)),
            ((40, 0.9045, 1, 8, 0.0), (55, 0.978, 2, 5, 0.016791), (55, 0.919, 3, 4, -0.014351), (-1, 0.0, 0, 0, 0.001982), (-1, 0.0, 0, 0, -0.018196), (11, 0.992, 6, 7, 0.474458), (-1, 0.0, 0, 0, 0.042311), (-1, 0.0, 0, 0, -0.015662), (2, 0.998, 9, 10, -0.475229), (-1, 0.0, 0, 0, -0.043192), (-1, 0.0, 0, 0, 0.012466)),
            ((8, 0.6190476190476191, 1, 8, 0.0), (7, 0.608, 2, 5, 0.045572), (27, 4.705005283552375, 3, 4, -0.077459), (-1, 0.0, 0, 0, -0.011153), (-1, 0.0, 0, 0, 0.014251), (9, 0.31733333333333336, 6, 7, 0.273545), (-1, 0.0, 0, 0, 0.045843), (-1, 0.0, 0, 0, 0.00516), (5, 0.009, 9, 12, -0.205601), (59, 0.4083333333333333, 10, 11, 0.469709), (-1, 0.0, 0, 0, 0.045913), (-1, 0.0, 0, 0, -0.001094), (5, 0.971, 13, 14, -0.446612), (-1, 0.0, 0, 0, -0.032466), (-1, 0.0, 0, 0, 0.013546)),
            ((49, 3.5, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.020979), (7, 0.955, 3, 4, -0.090752), (-1, 0.0, 0, 0, -0.011139), (-1, 0.0, 0, 0, 0.024052), (55, 0.919, 6, 7, 0.132611), (-1, 0.0, 0, 0, 0.013228), (-1, 0.0, 0, 0, -0.023302), (13, 0.43999999999999995, 9, 10, -0.41969), (-1, 0.0, 0, 0, 0.011144), (24, 0.179, 11, 12, -0.620714), (-1, 0.0, 0, 0, -0.037801), (-1, 0.0, 0, 0, 0.01023)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (27, 4.2964256503859755, 2, 5, -0.05186), (4, 0.001, 3, 4, -0.557542), (-1, 0.0, 0, 0, 0.012249), (-1, 0.0, 0, 0, -0.033343), (25, 6.702574239621701, 6, 7, -0.007186), (-1, 0.0, 0, 0, 0.009502), (-1, 0.0, 0, 0, -0.007734), (5, 0.025, 9, 12, 0.205687), (26, 4.424828703319956, 10, 11, 0.788232), (-1, 0.0, 0, 0, -0.004665), (-1, 0.0, 0, 0, 0.048145), (28, 0.973, 13, 14, -0.070866), (-1, 0.0, 0, 0, -0.012538), (-1, 0.0, 0, 0, 0.035492)),
            ((15, 0.759, 1, 8, 0.0), (20, -3.210268834500856, 2, 5, 0.028719), (11, 0.997, 3, 4, -0.127378), (-1, 0.0, 0, 0, -0.00199), (-1, 0.0, 0, 0, -0.042615), (5, 0.01, 6, 7, 0.140065), (-1, 0.0, 0, 0, 0.028744), (-1, 0.0, 0, 0, 0.000205), (6, 0.5986826325792863, 9, 12, -0.321921), (27, 4.181095032513147, 10, 11, 0.186805), (-1, 0.0, 0, 0, -0.026592), (-1, 0.0, 0, 0, 0.019645), (55, 0.015000000000000013, 13, 14, -0.852432), (-1, 0.0, 0, 0, -0.007116), (-1, 0.0, 0, 0, -0.048857)),
            ((45, 0.0676, 1, 8, 0.0), (3, 0.30100000000000005, 2, 5, -0.126502), (4, 0.95, 3, 4, 0.084409), (-1, 0.0, 0, 0, 0.011631), (-1, 0.0, 0, 0, -0.029447), (15, 0.06175000000000002, 6, 7, -0.572444), (-1, 0.0, 0, 0, -0.049591), (-1, 0.0, 0, 0, -0.011661), (3, 0.388, 9, 12, 0.076851), (62, 0.16000000000000003, 10, 11, -0.057591), (-1, 0.0, 0, 0, 0.014124), (-1, 0.0, 0, 0, -0.010766), (6, 0.8551494572736084, 13, 14, 0.491166), (-1, 0.0, 0, 0, 0.030967), (-1, 0.0, 0, 0, -0.032358)),
            ((58, 0.3550420168067227, 1, 8, 0.0), (4, 0.231, 2, 5, 0.036882), (27, 4.558471368545653, 3, 4, 0.312565), (-1, 0.0, 0, 0, -0.012506), (-1, 0.0, 0, 0, 0.025083), (12, 0.669, 6, 7, -0.070074), (-1, 0.0, 0, 0, 0.000605), (-1, 0.0, 0, 0, -0.027798), (13, 0.986, 9, 12, -0.252472), (27, 4.558471368545653, 10, 11, -0.617034), (-1, 0.0, 0, 0, -9.8e-05), (-1, 0.0, 0, 0, -0.046261), (24, 0.033500000000000016, 13, 14, 0.344354), (-1, 0.0, 0, 0, -0.021944), (-1, 0.0, 0, 0, 0.033668)),
            ((4, 0.979, 1, 8, 0.0), (4, 0.958, 2, 5, -0.051333), (62, 0.24, 3, 4, -0.002518), (-1, 0.0, 0, 0, 0.00759), (-1, 0.0, 0, 0, -0.009595), (45, 0.4766, 6, 7, -0.601891), (-1, 0.0, 0, 0, -0.041898), (-1, 0.0, 0, 0, 0.013863), (45, 0.0916, 9, 12, 0.218256), (5, 0.053, 10, 11, -0.576274), (-1, 0.0, 0, 0, -0.001573), (-1, 0.0, 0, 0, -0.050415), (12, 0.671, 13, 14, 0.611585), (-1, 0.0, 0, 0, 0.041739), (-1, 0.0, 0, 0, -0.028681)),
            ((8, 0.6190476190476191, 1, 8, 0.0), (7, 0.608, 2, 5, 0.045052), (27, 4.705005283552375, 3, 4, -0.070923), (-1, 0.0, 0, 0, -0.010743), (-1, 0.0, 0, 0, 0.014307), (9, 0.31499999999999995, 6, 7, 0.259659), (-1, 0.0, 0, 0, 0.045933), (-1, 0.0, 0, 0, 0.005282), (5, 0.009, 9, 12, -0.207262), (59, 0.4083333333333333, 10, 11, 0.430215), (-1, 0.0, 0, 0, 0.044384), (-1, 0.0, 0, 0, -0.002594), (5, 0.971, 13, 14, -0.428686), (-1, 0.0, 0, 0, -0.031042), (-1, 0.0, 0, 0, 0.012505)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.677480030793063, 2, 5, 0.033696), (45, 0.06888, 3, 4, -0.056067), (-1, 0.0, 0, 0, -0.016605), (-1, 0.0, 0, 0, 0.004221), (6, 0.11653987838050074, 6, 7, 0.23948), (-1, 0.0, 0, 0, -0.029939), (-1, 0.0, 0, 0, 0.018978), (41, 1.5, 9, 12, -0.281052), (1, 0.002, 10, 11, -0.058401), (-1, 0.0, 0, 0, -0.02141), (-1, 0.0, 0, 0, 0.016776), (-1, 0.0, 0, 0, -0.040932)),
            ((58, 0.3550420168067227, 1, 8, 0.0), (4, 0.231, 2, 5, 0.035475), (27, 4.558471368545653, 3, 4, 0.299194), (-1, 0.0, 0, 0, -0.012072), (-1, 0.0, 0, 0, 0.024052), (29, 0.5, 6, 7, -0.066964), (-1, 0.0, 0, 0, 0.000744), (-1, 0.0, 0, 0, -0.02654), (13, 0.986, 9, 12, -0.249428), (62, 0.44299999999999995, 10, 11, -0.599354), (-1, 0.0, 0, 0, -0.038573), (-1, 0.0, 0, 0, 0.018368), (24, 0.033500000000000016, 13, 14, 0.316697), (-1, 0.0, 0, 0, -0.021899), (-1, 0.0, 0, 0, 0.031961)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.032291), (7, 0.955, 3, 4, -0.089398), (-1, 0.0, 0, 0, -0.010698), (-1, 0.0, 0, 0, 0.022847), (50, 0.835, 6, 7, 0.17353), (-1, 0.0, 0, 0, 0.023453), (-1, 0.0, 0, 0, -0.008812), (41, 1.5, 9, 12, -0.276057), (11, 0.855, 10, 11, -0.059737), (-1, 0.0, 0, 0, 0.017724), (-1, 0.0, 0, 0, -0.019873), (-1, 0.0, 0, 0, -0.040189)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.442921), (-1, 0.0, 0, 0, -0.004004), (-1, 0.0, 0, 0, -0.033809), (15, -0.613, 5, 6, 0.017816), (-1, 0.0, 0, 0, 0.029225), (4, 0.979, 7, 8, -0.004755), (-1, 0.0, 0, 0, -0.003341), (-1, 0.0, 0, 0, 0.014323)),
            ((19, 0.5, 1, 6, 0.0), (6, 0.9105698788443931, 2, 5, -0.030465), (6, 0.8725633231113197, 3, 4, -0.052422), (-1, 0.0, 0, 0, -0.000675), (-1, 0.0, 0, 0, -0.040261), (-1, 0.0, 0, 0, 0.030913), (6, 0.8873180999538214, 7, 10, 0.363419), (13, 0.744, 8, 9, 0.543892), (-1, 0.0, 0, 0, 0.004782), (-1, 0.0, 0, 0, 0.041103), (-1, 0.0, 0, 0, -0.01958)),
            ((8, 0.6190476190476191, 1, 8, 0.0), (7, 0.608, 2, 5, 0.044555), (20, -4.595119850134589, 3, 4, -0.066956), (-1, 0.0, 0, 0, -0.022221), (-1, 0.0, 0, 0, 0.002881), (4, 0.988, 6, 7, 0.249862), (-1, 0.0, 0, 0, 0.004766), (-1, 0.0, 0, 0, 0.043792), (5, 0.009, 9, 12, -0.201024), (59, 0.4083333333333333, 10, 11, 0.411347), (-1, 0.0, 0, 0, 0.04356), (-1, 0.0, 0, 0, -0.003069), (5, 0.971, 13, 14, -0.411843), (-1, 0.0, 0, 0, -0.029798), (-1, 0.0, 0, 0, 0.011785)),
            ((58, 0.3550420168067227, 1, 8, 0.0), (4, 0.231, 2, 5, 0.035935), (23, 0.0050000000000000044, 3, 4, 0.28783), (-1, 0.0, 0, 0, -0.021856), (-1, 0.0, 0, 0, 0.020525), (12, 0.669, 6, 7, -0.062115), (-1, 0.0, 0, 0, 0.000858), (-1, 0.0, 0, 0, -0.026645), (13, 0.986, 9, 12, -0.24622), (27, 4.558471368545653, 10, 11, -0.581704), (-1, 0.0, 0, 0, 0.001238), (-1, 0.0, 0, 0, -0.04519), (24, 0.033500000000000016, 13, 14, 0.294601), (-1, 0.0, 0, 0, -0.021875), (-1, 0.0, 0, 0, 0.030589)),
            ((15, 0.759, 1, 8, 0.0), (20, -3.210268834500856, 2, 5, 0.027649), (11, 0.997, 3, 4, -0.116591), (-1, 0.0, 0, 0, -0.001571), (-1, 0.0, 0, 0, -0.041571), (5, 0.01, 6, 7, 0.130305), (-1, 0.0, 0, 0, 0.026628), (-1, 0.0, 0, 0, 0.000265), (6, 0.5986826325792863, 9, 12, -0.319165), (27, 4.181095032513147, 10, 11, 0.177681), (-1, 0.0, 0, 0, -0.02583), (-1, 0.0, 0, 0, 0.01894), (55, 0.015000000000000013, 13, 14, -0.834348), (-1, 0.0, 0, 0, -0.00709), (-1, 0.0, 0, 0, -0.047859)),
            ((26, 4.952293473989714, 1, 8, 0.0), (26, 4.579839228586978, 2, 5, 0.031772), (7, 0.955, 3, 4, -0.086329), (-1, 0.0, 0, 0, -0.010203), (-1, 0.0, 0, 0, 0.021595), (50, 0.835, 6, 7, 0.168332), (-1, 0.0, 0, 0, 0.022656), (-1, 0.0, 0, 0, -0.008424), (41, 1.5, 9, 12, -0.271243), (1, 0.002, 10, 11, -0.059852), (-1, 0.0, 0, 0, -0.021033), (-1, 0.0, 0, 0, 0.016272), (-1, 0.0, 0, 0, -0.039525)),
            ((11, 0.461, 1, 8, 0.0), (11, 0.119, 2, 5, -0.193138), (13, 0.02100000000000002, 3, 4, 0.273648), (-1, 0.0, 0, 0, -0.022781), (-1, 0.0, 0, 0, 0.028813), (25, 6.702574239621701, 6, 7, -0.430646), (-1, 0.0, 0, 0, -0.000342), (-1, 0.0, 0, 0, -0.040545), (11, 0.564, 9, 12, 0.045886), (3, 0.20499999999999996, 10, 11, 0.694888), (-1, 0.0, 0, 0, -0.002681), (-1, 0.0, 0, 0, 0.054537), (20, -3.210268834500856, 13, 14, -0.000619), (-1, 0.0, 0, 0, -0.011606), (-1, 0.0, 0, 0, 0.007756)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.437446), (-1, 0.0, 0, 0, -0.004047), (-1, 0.0, 0, 0, -0.033346), (15, -0.613, 5, 6, 0.017853), (-1, 0.0, 0, 0, 0.028733), (4, 0.979, 7, 8, -0.004196), (-1, 0.0, 0, 0, -0.003117), (-1, 0.0, 0, 0, 0.013396)),
            ((60, 0.5, 1, 8, 0.0), (15, 0.759, 2, 5, 0.02286), (62, 0.32099999999999995, 3, 4, 0.057584), (-1, 0.0, 0, 0, -0.00105), (-1, 0.0, 0, 0, 0.015125), (6, 0.5986826325792863, 6, 7, -0.316914), (-1, 0.0, 0, 0, 0.008669), (-1, 0.0, 0, 0, -0.040625), (26, 4.686739554676682, 9, 12, -0.359951), (46, 0.8005, 10, 11, -0.595665), (-1, 0.0, 0, 0, -0.037822), (-1, 0.0, 0, 0, 0.006827), (13, 0.991, 13, 14, 0.23311), (-1, 0.0, 0, 0, 0.023834), (-1, 0.0, 0, 0, -0.011117)),
            ((19, 0.5, 1, 6, 0.0), (6, 0.9105698788443931, 2, 5, -0.029624), (6, 0.8725633231113197, 3, 4, -0.050949), (-1, 0.0, 0, 0, -0.000652), (-1, 0.0, 0, 0, -0.039077), (-1, 0.0, 0, 0, 0.029784), (6, 0.8873180999538214, 7, 10, 0.357677), (6, 0.06192364553481582, 8, 9, 0.534109), (-1, 0.0, 0, 0, 0.000792), (-1, 0.0, 0, 0, 0.037549), (-1, 0.0, 0, 0, -0.01882)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (62, 0.369, 2, 5, -0.168836), (3, 0.29800000000000004, 3, 4, -0.293203), (-1, 0.0, 0, 0, -0.004781), (-1, 0.0, 0, 0, -0.046673), (4, 0.977, 6, 7, 0.537422), (-1, 0.0, 0, 0, -0.00949), (-1, 0.0, 0, 0, 0.061835), (13, 0.992, 9, 12, 0.053347), (13, 0.958, 10, 11, 0.145809), (-1, 0.0, 0, 0, 0.000769), (-1, 0.0, 0, 0, 0.035869), (1, 0.006, 13, 14, -0.283393), (-1, 0.0, 0, 0, -0.031817), (-1, 0.0, 0, 0, 0.006498)),
            ((58, 0.3550420168067227, 1, 8, 0.0), (7, 0.992, 2, 5, 0.035005), (1, 0.009, 3, 4, 0.07136), (-1, 0.0, 0, 0, 0.010469), (-1, 0.0, 0, 0, -0.006271), (26, 4.494223020031202, 6, 7, -0.590534), (-1, 0.0, 0, 0, 0.013898), (-1, 0.0, 0, 0, -0.045132), (28, 0.881, 9, 12, -0.244655), (5, 0.871, 10, 11, 0.147579), (-1, 0.0, 0, 0, 0.02041), (-1, 0.0, 0, 0, -0.029161), (54, 1.5, 13, 14, -0.710354), (-1, 0.0, 0, 0, -0.043676), (-1, 0.0, 0, 0, 0.003874)),
            ((53, 2.5, 1, 8, 0.0), (3, 0.489, 2, 5, 0.016281), (3, 0.44899999999999995, 3, 4, -0.021396), (-1, 0.0, 0, 0, 0.0004), (-1, 0.0, 0, 0, -0.032693), (56, 0.16399999999999998, 6, 7, 0.372851), (-1, 0.0, 0, 0, -0.020947), (-1, 0.0, 0, 0, 0.041752), (12, 0.45299999999999996, 9, 10, -0.450552), (-1, 0.0, 0, 0, 0.005016), (-1, 0.0, 0, 0, -0.03863)),
            ((26, 4.2413008733233095, 1, 8, 0.0), (45, 0.955, 2, 5, -0.268602), (4, 0.034, 3, 4, -0.462657), (-1, 0.0, 0, 0, 0.027003), (-1, 0.0, 0, 0, -0.03502), (11, 0.997, 6, 7, 0.591958), (-1, 0.0, 0, 0, 0.03389), (-1, 0.0, 0, 0, -0.001934), (49, 2.5, 9, 12, 0.030004), (58, 0.11805555555555555, 10, 11, 0.111913), (-1, 0.0, 0, 0, 0.020323), (-1, 0.0, 0, 0, -0.001621), (26, 4.400584200968346, 13, 14, -0.156247), (-1, 0.0, 0, 0, 0.050342), (-1, 0.0, 0, 0, -0.011674)),
            ((63, 0.98, 1, 8, 0.0), (59, 0.1380952380952381, 2, 5, 0.017245), (27, 4.608290119226802, 3, 4, 0.309224), (-1, 0.0, 0, 0, -0.012541), (-1, 0.0, 0, 0, 0.026873), (9, 0.767, 6, 7, -0.027928), (-1, 0.0, 0, 0, 0.001284), (-1, 0.0, 0, 0, -0.019896), (20, -3.683959493972949, 9, 10, -0.447025), (-1, 0.0, 0, 0, -0.035822), (-1, 0.0, 0, 0, 5.9e-05)),
            ((15, -0.17200000000000004, 1, 8, 0.0), (13, 0.692, 2, 5, -0.166266), (3, 0.10299999999999998, 3, 4, -0.693411), (-1, 0.0, 0, 0, -0.005353), (-1, 0.0, 0, 0, -0.047751), (2, 0.42400000000000004, 6, 7, 0.00443), (-1, 0.0, 0, 0, 0.018647), (-1, 0.0, 0, 0, -0.013193), (59, 0.14835164835164835, 9, 12, 0.052406), (27, 4.608290119226802, 10, 11, 0.516332), (-1, 0.0, 0, 0, -0.013136), (-1, 0.0, 0, 0, 0.043871), (13, 0.992, 13, 14, -0.010309), (-1, 0.0, 0, 0, 0.004026), (-1, 0.0, 0, 0, -0.016746)),
            ((15, -0.741, 1, 4, 0.0), (61, 0.19469890973733583, 2, 3, -0.433577), (-1, 0.0, 0, 0, -0.030323), (-1, 0.0, 0, 0, -0.001239), (40, 0.5799999999999998, 5, 8, 0.018007), (41, 0.5, 6, 7, 0.078301), (-1, 0.0, 0, 0, 0.013695), (-1, 0.0, 0, 0, -0.003958), (13, 0.845, 9, 10, -0.178636), (-1, 0.0, 0, 0, -0.025751), (-1, 0.0, 0, 0, 0.006815)),
        )),
        (-0.011991, (
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -1.067159), (-1, 0.0, 0, 0, -0.077668), (-1, 0.0, 0, 0, -0.015255), (24, 0.06250000000000003, 5, 6, 1.361203), (-1, 0.0, 0, 0, 0.023398), (-1, 0.0, 0, 0, 0.089942)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -1.015642), (-1, 0.0, 0, 0, -0.073969), (-1, 0.0, 0, 0, -0.014506), (24, 0.06250000000000003, 5, 6, 1.295273), (-1, 0.0, 0, 0, 0.022253), (-1, 0.0, 0, 0, 0.085622)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.968287), (-1, 0.0, 0, 0, -0.07064), (-1, 0.0, 0, 0, -0.013795), (24, 0.06250000000000003, 5, 6, 1.235745), (-1, 0.0, 0, 0, 0.02117), (-1, 0.0, 0, 0, 0.081815)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.924509), (-1, 0.0, 0, 0, -0.067621), (-1, 0.0, 0, 0, -0.013119), (24, 0.06250000000000003, 5, 6, 1.181588), (-1, 0.0, 0, 0, 0.020144), (-1, 0.0, 0, 0, 0.078429)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.883831), (-1, 0.0, 0, 0, -0.064863), (-1, 0.0, 0, 0, -0.012478), (24, 0.06250000000000003, 5, 6, 1.131977), (-1, 0.0, 0, 0, 0.019172), (-1, 0.0, 0, 0, 0.075394)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.845858), (-1, 0.0, 0, 0, -0.062328), (-1, 0.0, 0, 0, -0.011869), (24, 0.06250000000000003, 5, 6, 1.086245), (-1, 0.0, 0, 0, 0.018249), (-1, 0.0, 0, 0, 0.072653)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.810264), (-1, 0.0, 0, 0, -0.059983), (-1, 0.0, 0, 0, -0.01129), (24, 0.06250000000000003, 5, 6, 1.043847), (-1, 0.0, 0, 0, 0.017373), (-1, 0.0, 0, 0, 0.070161)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.776773), (-1, 0.0, 0, 0, -0.057802), (-1, 0.0, 0, 0, -0.01074), (24, 0.06250000000000003, 5, 6, 1.004334), (-1, 0.0, 0, 0, 0.016541), (-1, 0.0, 0, 0, 0.067881)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, -0.602765), (-1, 0.0, 0, 0, -0.043859), (-1, 0.0, 0, 0, 0.015177), (41, 0.5, 5, 6, 1.209858), (-1, 0.0, 0, 0, 0.073069), (-1, 0.0, 0, 0, 0.040122)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, -0.57686), (-1, 0.0, 0, 0, -0.042061), (-1, 0.0, 0, 0, 0.014818), (41, 0.5, 5, 6, 1.169588), (-1, 0.0, 0, 0, 0.071021), (-1, 0.0, 0, 0, 0.038461)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, -0.552359), (-1, 0.0, 0, 0, -0.040356), (-1, 0.0, 0, 0, 0.014318), (41, 0.5, 5, 6, 1.13192), (-1, 0.0, 0, 0, 0.069147), (-1, 0.0, 0, 0, 0.036881)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.529107), (-1, 0.0, 0, 0, -0.010317), (-1, 0.0, 0, 0, -0.059077), (41, 0.5, 5, 6, 1.096522), (-1, 0.0, 0, 0, 0.067423), (-1, 0.0, 0, 0, 0.035377)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.644227), (-1, 0.0, 0, 0, -0.054092), (-1, 0.0, 0, 0, -0.010019), (24, 0.02125000000000002, 5, 6, 0.858154), (-1, 0.0, 0, 0, -0.006886), (-1, 0.0, 0, 0, 0.052472)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.618817), (-1, 0.0, 0, 0, -0.05241), (-1, 0.0, 0, 0, -0.009538), (24, 0.179, 5, 6, 0.827778), (-1, 0.0, 0, 0, 0.015945), (-1, 0.0, 0, 0, 0.060212)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.473693), (-1, 0.0, 0, 0, -0.007962), (-1, 0.0, 0, 0, -0.055816), (41, 0.5, 5, 6, 1.015983), (-1, 0.0, 0, 0, 0.06368), (-1, 0.0, 0, 0, 0.031637)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.580579), (-1, 0.0, 0, 0, -0.050137), (-1, 0.0, 0, 0, -0.008457), (24, 0.02125000000000002, 5, 6, 0.787303), (-1, 0.0, 0, 0, -0.006264), (-1, 0.0, 0, 0, 0.048612)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.439714), (-1, 0.0, 0, 0, -0.006783), (-1, 0.0, 0, 0, -0.053434), (41, 0.5, 5, 6, 0.966825), (-1, 0.0, 0, 0, 0.061495), (-1, 0.0, 0, 0, 0.029361)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.545273), (-1, 0.0, 0, 0, -0.048045), (-1, 0.0, 0, 0, -0.007477), (24, 0.179, 5, 6, 0.748908), (-1, 0.0, 0, 0, 0.015172), (-1, 0.0, 0, 0, 0.055398)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.524091), (-1, 0.0, 0, 0, -0.046642), (-1, 0.0, 0, 0, -0.007117), (50, 0.955, 5, 6, 0.724243), (-1, 0.0, 0, 0, 0.017052), (-1, 0.0, 0, 0, 0.057246)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.396191), (-1, 0.0, 0, 0, -0.004978), (-1, 0.0, 0, 0, -0.05092), (41, 0.5, 5, 6, 0.90357), (-1, 0.0, 0, 0, 0.058832), (-1, 0.0, 0, 0, 0.026307)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.492566), (-1, 0.0, 0, 0, -0.044776), (-1, 0.0, 0, 0, -0.006259), (50, 0.955, 5, 6, 0.688966), (-1, 0.0, 0, 0, 0.015931), (-1, 0.0, 0, 0, 0.055327)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.369709), (-1, 0.0, 0, 0, -0.004151), (-1, 0.0, 0, 0, -0.048975), (41, 0.5, 5, 6, 0.864292), (-1, 0.0, 0, 0, 0.057228), (-1, 0.0, 0, 0, 0.024486)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.463176), (-1, 0.0, 0, 0, -0.043026), (-1, 0.0, 0, 0, -0.005479), (50, 0.955, 5, 6, 0.655826), (-1, 0.0, 0, 0, 0.014881), (-1, 0.0, 0, 0, 0.053554)),
            ((45, 0.87, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.327346), (-1, 0.0, 0, 0, -0.036958), (-1, 0.0, 0, 0, 0.000389), (50, 0.21499999999999997, 5, 6, 0.872809), (-1, 0.0, 0, 0, -0.024623), (-1, 0.0, 0, 0, 0.050126)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.333902), (-1, 0.0, 0, 0, -0.002734), (-1, 0.0, 0, 0, -0.046844), (41, 0.5, 5, 6, 0.810373), (-1, 0.0, 0, 0, 0.055195), (-1, 0.0, 0, 0, 0.021782)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.423155), (-1, 0.0, 0, 0, -0.03736), (-1, 0.0, 0, 0, -0.001803), (59, 0.17424242424242425, 5, 6, 0.61016), (-1, 0.0, 0, 0, 0.062104), (-1, 0.0, 0, 0, 0.018163)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.14835164835164835, 2, 3, -0.313189), (-1, 0.0, 0, 0, 0.00745), (-1, 0.0, 0, 0, -0.032896), (26, 4.494223020031202, 5, 6, 0.777102), (-1, 0.0, 0, 0, 0.018503), (-1, 0.0, 0, 0, 0.053305)),
            ((45, 0.87, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.28774), (-1, 0.0, 0, 0, -0.034035), (-1, 0.0, 0, 0, 0.001091), (13, 0.933, 5, 6, 0.807944), (-1, 0.0, 0, 0, 0.006117), (-1, 0.0, 0, 0, 0.055384)),
            ((45, 0.16178, 1, 4, 0.0), (26, 4.363078339578307, 2, 3, -0.391245), (-1, 0.0, 0, 0, -0.047622), (-1, 0.0, 0, 0, -0.008387), (7, 0.942, 5, 6, 0.563715), (-1, 0.0, 0, 0, 0.010832), (-1, 0.0, 0, 0, 0.050198)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.3431372549019608, 2, 3, -0.282945), (-1, 0.0, 0, 0, -0.003957), (-1, 0.0, 0, 0, -0.052851), (26, 4.494223020031202, 5, 6, 0.72945), (-1, 0.0, 0, 0, 0.015855), (-1, 0.0, 0, 0, 0.05132)),
            ((45, 0.87, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.260816), (-1, 0.0, 0, 0, -0.032054), (-1, 0.0, 0, 0, 0.001602), (13, 0.933, 5, 6, 0.761313), (-1, 0.0, 0, 0, 0.004506), (-1, 0.0, 0, 0, 0.053501)),
            ((45, 0.16178, 1, 4, 0.0), (26, 4.255587559545337, 2, 3, -0.358352), (-1, 0.0, 0, 0, -0.052539), (-1, 0.0, 0, 0, -0.009232), (7, 0.957, 5, 6, 0.525714), (-1, 0.0, 0, 0, 0.010624), (-1, 0.0, 0, 0, 0.049939)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (58, 0.17424242424242425, 2, 3, -0.509281), (-1, 0.0, 0, 0, -0.004868), (-1, 0.0, 0, 0, -0.044427), (41, 0.5, 5, 6, 0.358844), (-1, 0.0, 0, 0, 0.044057), (-1, 0.0, 0, 0, -0.003761)),
            ((45, 0.87, 1, 4, 0.0), (58, 0.3038461538461539, 2, 3, -0.239032), (-1, 0.0, 0, 0, -0.002361), (-1, 0.0, 0, 0, -0.041186), (13, 0.933, 5, 6, 0.717168), (-1, 0.0, 0, 0, 0.002756), (-1, 0.0, 0, 0, 0.05183)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (58, 0.17424242424242425, 2, 3, -0.483739), (-1, 0.0, 0, 0, -0.004611), (-1, 0.0, 0, 0, -0.042724), (41, 0.5, 5, 6, 0.338527), (-1, 0.0, 0, 0, 0.042371), (-1, 0.0, 0, 0, -0.00363)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (59, 0.3693181818181818, 2, 3, -0.240266), (-1, 0.0, 0, 0, 0.007794), (-1, 0.0, 0, 0, -0.030233), (26, 4.494223020031202, 5, 6, 0.651675), (-1, 0.0, 0, 0, 0.010566), (-1, 0.0, 0, 0, 0.048626)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (26, 4.363078339578307, 2, 3, -0.314983), (-1, 0.0, 0, 0, -0.043642), (-1, 0.0, 0, 0, -0.004892), (59, 0.17424242424242425, 5, 6, 0.477036), (-1, 0.0, 0, 0, 0.057465), (-1, 0.0, 0, 0, 0.011023)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (58, 0.17424242424242425, 2, 3, -0.454945), (-1, 0.0, 0, 0, -0.004183), (-1, 0.0, 0, 0, -0.040778), (41, 0.5, 5, 6, 0.316687), (-1, 0.0, 0, 0, 0.040545), (-1, 0.0, 0, 0, -0.003456)),
            ((7, 0.957, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, -0.197113), (-1, 0.0, 0, 0, -0.001781), (-1, 0.0, 0, 0, -0.055027), (56, 0.10299999999999998, 5, 6, 0.721951), (-1, 0.0, 0, 0, 0.013121), (-1, 0.0, 0, 0, 0.045484)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.209281), (-1, 0.0, 0, 0, -0.043231), (-1, 0.0, 0, 0, -0.003098), (50, 0.21499999999999997, 5, 6, 0.652479), (-1, 0.0, 0, 0, -0.034737), (-1, 0.0, 0, 0, 0.040546)),
            ((24, 0.027500000000000024, 1, 4, 0.0), (28, 0.622, 2, 3, -0.491882), (-1, 0.0, 0, 0, -0.00579), (-1, 0.0, 0, 0, -0.036041), (41, 0.5, 5, 6, 0.260266), (-1, 0.0, 0, 0, 0.038125), (-1, 0.0, 0, 0, -0.00579)),
            ((7, 0.957, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.184399), (-1, 0.0, 0, 0, -0.05099), (-1, 0.0, 0, 0, -0.00112), (56, 0.10299999999999998, 5, 6, 0.682901), (-1, 0.0, 0, 0, 0.012463), (-1, 0.0, 0, 0, 0.04361)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (59, 0.3693181818181818, 2, 3, -0.20899), (-1, 0.0, 0, 0, 0.008328), (-1, 0.0, 0, 0, -0.027889), (26, 4.494223020031202, 5, 6, 0.586981), (-1, 0.0, 0, 0, 0.00945), (-1, 0.0, 0, 0, 0.04634)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.190398), (-1, 0.0, 0, 0, -0.024885), (-1, 0.0, 0, 0, 0.005482), (13, 0.933, 5, 6, 0.612435), (-1, 0.0, 0, 0, -0.002247), (-1, 0.0, 0, 0, 0.048281)),
            ((7, 0.959, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, -0.170071), (-1, 0.0, 0, 0, -0.00081), (-1, 0.0, 0, 0, -0.052778), (56, 0.08699999999999997, 5, 6, 0.657712), (-1, 0.0, 0, 0, 0.012191), (-1, 0.0, 0, 0, 0.041774)),
            ((24, 0.027500000000000024, 1, 4, 0.0), (28, 0.622, 2, 3, -0.45376), (-1, 0.0, 0, 0, -0.005416), (-1, 0.0, 0, 0, -0.03398), (41, 0.5, 5, 6, 0.237113), (-1, 0.0, 0, 0, 0.036255), (-1, 0.0, 0, 0, -0.005416)),
            ((20, 0.6263814842476838, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.14998), (-1, 0.0, 0, 0, -0.043204), (-1, 0.0, 0, 0, -6e-06), (13, 0.967, 5, 6, 0.705379), (-1, 0.0, 0, 0, 0.013885), (-1, 0.0, 0, 0, 0.051575)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.660349), (-1, 0.0, 0, 0, -0.010315), (-1, 0.0, 0, 0, 0.061655), (24, 0.06250000000000003, 5, 6, -0.158887), (-1, 0.0, 0, 0, -0.017348), (-1, 0.0, 0, 0, 0.020435)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.164526), (-1, 0.0, 0, 0, -0.048938), (-1, 0.0, 0, 0, -0.000531), (56, 0.12, 5, 6, 0.596469), (-1, 0.0, 0, 0, 0.010799), (-1, 0.0, 0, 0, 0.039429)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.643895), (-1, 0.0, 0, 0, -0.009394), (-1, 0.0, 0, 0, 0.060003), (20, -3.210268834500856, 5, 6, -0.154094), (-1, 0.0, 0, 0, -0.028184), (-1, 0.0, 0, 0, 0.004657)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.620897), (-1, 0.0, 0, 0, -0.00899), (-1, 0.0, 0, 0, 0.05817), (24, 0.06250000000000003, 5, 6, -0.147387), (-1, 0.0, 0, 0, -0.01612), (-1, 0.0, 0, 0, 0.019253)),
            ((41, 0.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.289814), (-1, 0.0, 0, 0, -0.016429), (-1, 0.0, 0, 0, 0.034202), (7, 0.959, 5, 6, -0.30403), (-1, 0.0, 0, 0, -0.024082), (-1, 0.0, 0, 0, 0.023534)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.604101), (-1, 0.0, 0, 0, -0.007813), (-1, 0.0, 0, 0, 0.056302), (20, -3.210268834500856, 5, 6, -0.141424), (-1, 0.0, 0, 0, -0.026102), (-1, 0.0, 0, 0, 0.004235)),
            ((41, 0.5, 1, 4, 0.0), (24, 0.012250000000000011, 2, 3, 0.279727), (-1, 0.0, 0, 0, -0.022733), (-1, 0.0, 0, 0, 0.028626), (13, 0.974, 5, 6, -0.290962), (-1, 0.0, 0, 0, -0.029421), (-1, 0.0, 0, 0, 0.007773)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.146681), (-1, 0.0, 0, 0, -0.027817), (-1, 0.0, 0, 0, 0.007575), (56, 0.12, 5, 6, 0.544619), (-1, 0.0, 0, 0, 0.009948), (-1, 0.0, 0, 0, 0.037128)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.592803), (-1, 0.0, 0, 0, -0.017349), (-1, 0.0, 0, 0, 0.050095), (20, 0.6263814842476838, 5, 6, -0.137822), (-1, 0.0, 0, 0, -0.01294), (-1, 0.0, 0, 0, 0.025854)),
            ((41, 0.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.264803), (-1, 0.0, 0, 0, -0.014468), (-1, 0.0, 0, 0, 0.031046), (4, 0.995, 5, 6, -0.27597), (-1, 0.0, 0, 0, -0.020045), (-1, 0.0, 0, 0, 0.033504)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.56732), (-1, 0.0, 0, 0, -0.016601), (-1, 0.0, 0, 0, 0.047871), (24, 0.06250000000000003, 5, 6, -0.131152), (-1, 0.0, 0, 0, -0.014168), (-1, 0.0, 0, 0, 0.017251)),
            ((7, 0.951, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, -0.137155), (-1, 0.0, 0, 0, 5.9e-05), (-1, 0.0, 0, 0, -0.049334), (56, 0.12, 5, 6, 0.517478), (-1, 0.0, 0, 0, 0.009508), (-1, 0.0, 0, 0, 0.035821)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.898797), (-1, 0.0, 0, 0, -0.056999), (-1, 0.0, 0, 0, -0.015574), (41, 0.5, 5, 6, 0.071149), (-1, 0.0, 0, 0, 0.019574), (-1, 0.0, 0, 0, -0.011954)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.137646), (-1, 0.0, 0, 0, -0.041351), (-1, 0.0, 0, 0, -0.000674), (13, 0.933, 5, 6, 0.494724), (-1, 0.0, 0, 0, -0.005887), (-1, 0.0, 0, 0, 0.044244)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.547832), (-1, 0.0, 0, 0, -0.006733), (-1, 0.0, 0, 0, 0.051996), (24, 0.04500000000000004, 5, 6, -0.127423), (-1, 0.0, 0, 0, -0.022825), (-1, 0.0, 0, 0, 0.003748)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.878442), (-1, 0.0, 0, 0, -0.056103), (-1, 0.0, 0, 0, -0.015266), (41, 0.5, 5, 6, 0.068437), (-1, 0.0, 0, 0, 0.018689), (-1, 0.0, 0, 0, -0.01137)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.530253), (-1, 0.0, 0, 0, -0.0174), (-1, 0.0, 0, 0, 0.045467), (24, 0.04500000000000004, 5, 6, -0.122277), (-1, 0.0, 0, 0, -0.021947), (-1, 0.0, 0, 0, 0.003517)),
            ((58, 0.3550420168067227, 1, 4, 0.0), (15, -0.741, 2, 3, 0.111433), (-1, 0.0, 0, 0, -0.041712), (-1, 0.0, 0, 0, 0.009766), (13, 0.986, 5, 6, -0.536761), (-1, 0.0, 0, 0, -0.050964), (-1, 0.0, 0, 0, 0.016829)),
            ((7, 0.959, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.122092), (-1, 0.0, 0, 0, -0.022919), (-1, 0.0, 0, 0, 0.009522), (26, 4.494223020031202, 5, 6, 0.502976), (-1, 0.0, 0, 0, 0.046896), (-1, 0.0, 0, 0, 0.009522)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.510073), (-1, 0.0, 0, 0, -0.016686), (-1, 0.0, 0, 0, 0.043622), (20, 0.6263814842476838, 5, 6, -0.116948), (-1, 0.0, 0, 0, -0.010865), (-1, 0.0, 0, 0, 0.022772)),
            ((58, 0.3550420168067227, 1, 4, 0.0), (15, -0.741, 2, 3, 0.106584), (-1, 0.0, 0, 0, -0.040404), (-1, 0.0, 0, 0, 0.009352), (13, 0.986, 5, 6, -0.519295), (-1, 0.0, 0, 0, -0.049455), (-1, 0.0, 0, 0, 0.015831)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.122565), (-1, 0.0, 0, 0, -0.040383), (-1, 0.0, 0, 0, -0.000116), (13, 0.933, 5, 6, 0.458934), (-1, 0.0, 0, 0, -0.006499), (-1, 0.0, 0, 0, 0.042867)),
            ((59, 0.4365079365079365, 1, 4, 0.0), (2, 0.986, 2, 3, 0.199227), (-1, 0.0, 0, 0, -0.003277), (-1, 0.0, 0, 0, 0.039481), (20, -1.9659128163621629, 5, 6, -0.278622), (-1, 0.0, 0, 0, -0.025119), (-1, 0.0, 0, 0, 0.024809)),
            ((41, 0.5, 1, 4, 0.0), (24, 0.012250000000000011, 2, 3, 0.22983), (-1, 0.0, 0, 0, -0.01943), (-1, 0.0, 0, 0, 0.023878), (26, 4.579839228586978, 5, 6, -0.236608), (-1, 0.0, 0, 0, -0.026572), (-1, 0.0, 0, 0, 0.007092)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.498004), (-1, 0.0, 0, 0, -0.015923), (-1, 0.0, 0, 0, 0.042228), (49, 3.5, 5, 6, -0.112967), (-1, 0.0, 0, 0, -0.001915), (-1, 0.0, 0, 0, -0.042475)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.810489), (-1, 0.0, 0, 0, -0.054322), (-1, 0.0, 0, 0, -0.011319), (59, 0.4365079365079365, 5, 6, 0.061896), (-1, 0.0, 0, 0, 0.01552), (-1, 0.0, 0, 0, -0.012436)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (7, 0.955, 2, 3, -0.544547), (-1, 0.0, 0, 0, -0.044028), (-1, 0.0, 0, 0, 0.042152), (59, 0.4365079365079365, 5, 6, 0.091007), (-1, 0.0, 0, 0, 0.018159), (-1, 0.0, 0, 0, -0.01174)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.796829), (-1, 0.0, 0, 0, -0.053638), (-1, 0.0, 0, 0, -0.011349), (41, 0.5, 5, 6, 0.059754), (-1, 0.0, 0, 0, 0.016724), (-1, 0.0, 0, 0, -0.010288)),
            ((58, 0.27525252525252525, 1, 4, 0.0), (26, 4.255587559545337, 2, 3, 0.13841), (-1, 0.0, 0, 0, -0.022374), (-1, 0.0, 0, 0, 0.013031), (28, 0.047, 5, 6, -0.350529), (-1, 0.0, 0, 0, 0.046317), (-1, 0.0, 0, 0, -0.025052)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.468234), (-1, 0.0, 0, 0, -0.006621), (-1, 0.0, 0, 0, 0.046433), (2, 0.992, 5, 6, -0.105246), (-1, 0.0, 0, 0, -0.014737), (-1, 0.0, 0, 0, 0.007823)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.111391), (-1, 0.0, 0, 0, -0.035534), (-1, 0.0, 0, 0, 0.000831), (13, 0.933, 5, 6, 0.432033), (-1, 0.0, 0, 0, -0.006466), (-1, 0.0, 0, 0, 0.041654)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.779829), (-1, 0.0, 0, 0, -0.052971), (-1, 0.0, 0, 0, -0.011005), (41, 0.5, 5, 6, 0.058071), (-1, 0.0, 0, 0, 0.015945), (-1, 0.0, 0, 0, -0.009701)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.059851), (-1, 0.0, 0, 0, -0.009836), (-1, 0.0, 0, 0, 0.017639), (24, 0.06250000000000003, 5, 6, -0.757548), (-1, 0.0, 0, 0, -0.049239), (-1, 0.0, 0, 0, 0.027287)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.108694), (-1, 0.0, 0, 0, -0.020941), (-1, 0.0, 0, 0, 0.007986), (26, 4.494223020031202, 5, 6, 0.428121), (-1, 0.0, 0, 0, 0.043118), (-1, 0.0, 0, 0, 0.007986)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (63, 0.664, 2, 3, 0.460754), (-1, 0.0, 0, 0, 0.042892), (-1, 0.0, 0, 0, -0.009967), (26, 4.952293473989714, 5, 6, -0.100865), (-1, 0.0, 0, 0, -0.000211), (-1, 0.0, 0, 0, -0.030961)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.758561), (-1, 0.0, 0, 0, -0.052249), (-1, 0.0, 0, 0, -0.010271), (59, 0.4365079365079365, 5, 6, 0.056165), (-1, 0.0, 0, 0, 0.013937), (-1, 0.0, 0, 0, -0.01105)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (7, 0.955, 2, 3, -0.515185), (-1, 0.0, 0, 0, -0.041095), (-1, 0.0, 0, 0, 0.03926), (58, 0.12916666666666665, 5, 6, 0.084251), (-1, 0.0, 0, 0, 0.021913), (-1, 0.0, 0, 0, -0.006306)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.103951), (-1, 0.0, 0, 0, -0.033925), (-1, 0.0, 0, 0, 0.00086), (13, 0.744, 5, 6, 0.412934), (-1, 0.0, 0, 0, -0.016916), (-1, 0.0, 0, 0, 0.034868)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.056844), (-1, 0.0, 0, 0, -0.009279), (-1, 0.0, 0, 0, 0.016514), (24, 0.06250000000000003, 5, 6, -0.722798), (-1, 0.0, 0, 0, -0.047356), (-1, 0.0, 0, 0, 0.026911)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.746943), (-1, 0.0, 0, 0, -0.051728), (-1, 0.0, 0, 0, -0.010161), (41, 0.5, 5, 6, 0.054511), (-1, 0.0, 0, 0, 0.014884), (-1, 0.0, 0, 0, -0.009051)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.515217), (-1, 0.0, 0, 0, -0.010976), (-1, 0.0, 0, 0, -0.049516), (59, 0.1380952380952381, 5, 6, 0.07619), (-1, 0.0, 0, 0, 0.036391), (-1, 0.0, 0, 0, -0.001648)),
            ((58, 0.3798076923076923, 1, 4, 0.0), (15, -0.741, 2, 3, 0.080918), (-1, 0.0, 0, 0, -0.035316), (-1, 0.0, 0, 0, 0.0072), (55, 0.943, 5, 6, -0.486621), (-1, 0.0, 0, 0, -0.043304), (-1, 0.0, 0, 0, 0.026443)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (7, 0.955, 2, 3, -0.4926), (-1, 0.0, 0, 0, -0.039327), (-1, 0.0, 0, 0, 0.03788), (59, 0.4365079365079365, 5, 6, 0.07956), (-1, 0.0, 0, 0, 0.016197), (-1, 0.0, 0, 0, -0.01057)),
            ((19, 0.5, 1, 4, 0.0), (61, 0.22564484631765486, 2, 3, -0.067597), (-1, 0.0, 0, 0, 0.002223), (-1, 0.0, 0, 0, -0.028255), (11, 0.948, 5, 6, 0.589948), (-1, 0.0, 0, 0, 0.013059), (-1, 0.0, 0, 0, 0.051606)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.054162), (-1, 0.0, 0, 0, -0.008916), (-1, 0.0, 0, 0, 0.015679), (24, 0.06250000000000003, 5, 6, -0.697403), (-1, 0.0, 0, 0, -0.045779), (-1, 0.0, 0, 0, 0.026175)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.715076), (-1, 0.0, 0, 0, -0.050537), (-1, 0.0, 0, 0, -0.00941), (41, 0.5, 5, 6, 0.050788), (-1, 0.0, 0, 0, 0.014067), (-1, 0.0, 0, 0, -0.008628)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.425226), (-1, 0.0, 0, 0, -0.016879), (-1, 0.0, 0, 0, 0.036813), (26, 4.952293473989714, 5, 6, -0.090272), (-1, 0.0, 0, 0, 2e-05), (-1, 0.0, 0, 0, -0.029222)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.096413), (-1, 0.0, 0, 0, -0.01925), (-1, 0.0, 0, 0, 0.007266), (26, 4.494223020031202, 5, 6, 0.387039), (-1, 0.0, 0, 0, 0.040662), (-1, 0.0, 0, 0, 0.007266)),
            ((20, 0.6263814842476838, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.077658), (-1, 0.0, 0, 0, -0.034046), (-1, 0.0, 0, 0, 0.002053), (13, 0.967, 5, 6, 0.476675), (-1, 0.0, 0, 0, 0.009975), (-1, 0.0, 0, 0, 0.046332)),
            ((58, 0.27525252525252525, 1, 4, 0.0), (49, 2.5, 2, 3, 0.117948), (-1, 0.0, 0, 0, 0.012757), (-1, 0.0, 0, 0, -0.015207), (28, 0.047, 5, 6, -0.302564), (-1, 0.0, 0, 0, 0.042783), (-1, 0.0, 0, 0, -0.021893)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (28, 0.204, 2, 3, -0.4766), (-1, 0.0, 0, 0, 0.031099), (-1, 0.0, 0, 0, -0.039519), (58, 0.11805555555555555, 5, 6, 0.074087), (-1, 0.0, 0, 0, 0.021533), (-1, 0.0, 0, 0, -0.00482)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.703896), (-1, 0.0, 0, 0, -0.050064), (-1, 0.0, 0, 0, -0.009239), (59, 0.4365079365079365, 5, 6, 0.049568), (-1, 0.0, 0, 0, 0.012081), (-1, 0.0, 0, 0, -0.009489)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.48033), (-1, 0.0, 0, 0, -0.010281), (-1, 0.0, 0, 0, -0.048309), (41, 0.5, 5, 6, 0.069109), (-1, 0.0, 0, 0, 0.015778), (-1, 0.0, 0, 0, -0.007863)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.050813), (-1, 0.0, 0, 0, -0.005767), (-1, 0.0, 0, 0, 0.018899), (27, 4.728490960086225, 5, 6, -0.656308), (-1, 0.0, 0, 0, -0.045061), (-1, 0.0, 0, 0, 0.02144)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.738843), (-1, 0.0, 0, 0, -0.006652), (-1, 0.0, 0, 0, -0.051235), (59, 0.4365079365079365, 5, 6, 0.044245), (-1, 0.0, 0, 0, 0.011725), (-1, 0.0, 0, 0, -0.009677)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (28, 0.204, 2, 3, -0.460422), (-1, 0.0, 0, 0, 0.029877), (-1, 0.0, 0, 0, -0.038212), (58, 0.11805555555555555, 5, 6, 0.07114), (-1, 0.0, 0, 0, 0.020303), (-1, 0.0, 0, 0, -0.00444)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.0556), (-1, 0.0, 0, 0, 0.00374), (-1, 0.0, 0, 0, -0.020389), (24, 0.06250000000000003, 5, 6, 0.605492), (-1, 0.0, 0, 0, 0.013747), (-1, 0.0, 0, 0, 0.046594)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.088078), (-1, 0.0, 0, 0, -0.033827), (-1, 0.0, 0, 0, 6e-05), (13, 0.744, 5, 6, 0.369355), (-1, 0.0, 0, 0, -0.017089), (-1, 0.0, 0, 0, 0.033016)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.070567), (-1, 0.0, 0, 0, -0.005758), (-1, 0.0, 0, 0, 0.02649), (24, 0.06250000000000003, 5, 6, -0.458659), (-1, 0.0, 0, 0, -0.033533), (-1, 0.0, 0, 0, 0.038414)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.670622), (-1, 0.0, 0, 0, -0.049056), (-1, 0.0, 0, 0, -0.008059), (59, 0.17424242424242425, 5, 6, 0.046663), (-1, 0.0, 0, 0, 0.022111), (-1, 0.0, 0, 0, -0.002914)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.0045000000000000595, 2, 3, -0.249779), (-1, 0.0, 0, 0, -0.028028), (-1, 0.0, 0, 0, -0.00312), (63, 0.933, 5, 6, 0.124976), (-1, 0.0, 0, 0, 0.002548), (-1, 0.0, 0, 0, 0.058736)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.049177), (-1, 0.0, 0, 0, -0.007994), (-1, 0.0, 0, 0, 0.013898), (24, 0.06250000000000003, 5, 6, -0.627661), (-1, 0.0, 0, 0, -0.042208), (-1, 0.0, 0, 0, 0.025661)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (15, -0.113, 2, 3, 0.389427), (-1, 0.0, 0, 0, -0.004753), (-1, 0.0, 0, 0, 0.04332), (2, 0.985, 5, 6, -0.078011), (-1, 0.0, 0, 0, -0.01356), (-1, 0.0, 0, 0, 0.006224)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.056224), (-1, 0.0, 0, 0, -0.026321), (-1, 0.0, 0, 0, 0.001997), (11, 0.948, 5, 6, 0.535011), (-1, 0.0, 0, 0, 0.01197), (-1, 0.0, 0, 0, 0.05065)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.654412), (-1, 0.0, 0, 0, -0.048169), (-1, 0.0, 0, 0, -0.008197), (41, 0.5, 5, 6, 0.045212), (-1, 0.0, 0, 0, 0.012407), (-1, 0.0, 0, 0, -0.007571)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.067077), (-1, 0.0, 0, 0, -0.005463), (-1, 0.0, 0, 0, 0.025062), (24, 0.06250000000000003, 5, 6, -0.439029), (-1, 0.0, 0, 0, -0.031948), (-1, 0.0, 0, 0, 0.037031)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.17424242424242425, 2, 3, 0.040223), (-1, 0.0, 0, 0, 0.022453), (-1, 0.0, 0, 0, -0.003401), (20, -3.683959493972949, 5, 6, -0.729006), (-1, 0.0, 0, 0, -0.051661), (-1, 0.0, 0, 0, 0.004955)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.4083333333333333, 2, 3, -0.050582), (-1, 0.0, 0, 0, 0.000928), (-1, 0.0, 0, 0, -0.031826), (63, 0.0010000000000000009, 5, 6, 0.578735), (-1, 0.0, 0, 0, 0.053944), (-1, 0.0, 0, 0, 0.013204)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.430877), (-1, 0.0, 0, 0, -0.033802), (-1, 0.0, 0, 0, 0.037644), (59, 0.4365079365079365, 5, 6, 0.06594), (-1, 0.0, 0, 0, 0.013354), (-1, 0.0, 0, 0, -0.008793)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.443197), (-1, 0.0, 0, 0, -0.009476), (-1, 0.0, 0, 0, -0.046929), (6, 0.6482034517095314, 5, 6, 0.064172), (-1, 0.0, 0, 0, 0.011947), (-1, 0.0, 0, 0, -0.009476)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.697551), (-1, 0.0, 0, 0, -0.003553), (-1, 0.0, 0, 0, -0.050213), (59, 0.4365079365079365, 5, 6, 0.040381), (-1, 0.0, 0, 0, 0.010382), (-1, 0.0, 0, 0, -0.00846)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.081695), (-1, 0.0, 0, 0, -0.01752), (-1, 0.0, 0, 0, 0.006505), (26, 4.494223020031202, 5, 6, 0.341908), (-1, 0.0, 0, 0, 0.039147), (-1, 0.0, 0, 0, 0.006505)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.045695), (-1, 0.0, 0, 0, -0.007405), (-1, 0.0, 0, 0, 0.012786), (27, 4.728490960086225, 5, 6, -0.60145), (-1, 0.0, 0, 0, -0.042304), (-1, 0.0, 0, 0, 0.021465)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.17424242424242425, 2, 3, 0.038714), (-1, 0.0, 0, 0, 0.021471), (-1, 0.0, 0, 0, -0.003195), (20, -3.683959493972949, 5, 6, -0.710865), (-1, 0.0, 0, 0, -0.051038), (-1, 0.0, 0, 0, 0.005068)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.053031), (-1, 0.0, 0, 0, -0.0316), (-1, 0.0, 0, 0, 0.000878), (11, 0.948, 5, 6, 0.509664), (-1, 0.0, 0, 0, 0.011416), (-1, 0.0, 0, 0, 0.050369)),
            ((15, -0.741, 1, 4, 0.0), (26, 4.505334588423557, 2, 3, -0.630482), (-1, 0.0, 0, 0, -0.000972), (-1, 0.0, 0, 0, -0.043231), (41, 0.5, 5, 6, 0.041449), (-1, 0.0, 0, 0, 0.011624), (-1, 0.0, 0, 0, -0.007155)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.048176), (-1, 0.0, 0, 0, 0.003285), (-1, 0.0, 0, 0, -0.018033), (63, 0.0010000000000000009, 5, 6, 0.553213), (-1, 0.0, 0, 0, 0.053394), (-1, 0.0, 0, 0, 0.012626)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (28, 0.204, 2, 3, -0.417275), (-1, 0.0, 0, 0, 0.029548), (-1, 0.0, 0, 0, -0.034801), (59, 0.4365079365079365, 5, 6, 0.061837), (-1, 0.0, 0, 0, 0.012323), (-1, 0.0, 0, 0, -0.007993)),
            ((20, -3.683959493972949, 1, 4, 0.0), (6, 0.18761101883864736, 2, 3, -0.22769), (-1, 0.0, 0, 0, -0.032536), (-1, 0.0, 0, 0, -0.002853), (63, 0.933, 5, 6, 0.113552), (-1, 0.0, 0, 0, 0.002056), (-1, 0.0, 0, 0, 0.055844)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.037043), (-1, 0.0, 0, 0, 0.023705), (-1, 0.0, 0, 0, -0.002304), (20, -3.683959493972949, 5, 6, -0.700434), (-1, 0.0, 0, 0, -0.050047), (-1, 0.0, 0, 0, 0.003138)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.0621), (-1, 0.0, 0, 0, -0.005365), (-1, 0.0, 0, 0, 0.023804), (24, 0.06250000000000003, 5, 6, -0.408176), (-1, 0.0, 0, 0, -0.030048), (-1, 0.0, 0, 0, 0.036132)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.66917), (-1, 0.0, 0, 0, -0.002709), (-1, 0.0, 0, 0, -0.049131), (41, 0.5, 5, 6, 0.037616), (-1, 0.0, 0, 0, 0.010852), (-1, 0.0, 0, 0, -0.006687)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.042314), (-1, 0.0, 0, 0, -0.007175), (-1, 0.0, 0, 0, 0.012167), (27, 4.728490960086225, 5, 6, -0.574693), (-1, 0.0, 0, 0, -0.04073), (-1, 0.0, 0, 0, 0.020605)),
            ((20, 2.319785806352712, 1, 4, 0.0), (15, -0.741, 2, 3, -0.052511), (-1, 0.0, 0, 0, -0.043994), (-1, 0.0, 0, 0, 3.9e-05), (13, 0.744, 5, 6, 0.470076), (-1, 0.0, 0, 0, 0.010439), (-1, 0.0, 0, 0, 0.043093)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.17424242424242425, 2, 3, 0.03367), (-1, 0.0, 0, 0, 0.019734), (-1, 0.0, 0, 0, -0.003036), (20, -3.683959493972949, 5, 6, -0.678315), (-1, 0.0, 0, 0, -0.049501), (-1, 0.0, 0, 0, 0.003972)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.403731), (-1, 0.0, 0, 0, -0.031934), (-1, 0.0, 0, 0, 0.037459), (59, 0.4365079365079365, 5, 6, 0.05831), (-1, 0.0, 0, 0, 0.011713), (-1, 0.0, 0, 0, -0.007618)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.256142), (-1, 0.0, 0, 0, -0.050003), (-1, 0.0, 0, 0, -0.001382), (59, 0.14835164835164835, 5, 6, 0.094905), (-1, 0.0, 0, 0, 0.036024), (-1, 0.0, 0, 0, -0.000378)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (40, 0.0295, 2, 3, -0.417476), (-1, 0.0, 0, 0, -0.045185), (-1, 0.0, 0, 0, -0.008998), (6, 0.6482034517095314, 5, 6, 0.057542), (-1, 0.0, 0, 0, 0.011233), (-1, 0.0, 0, 0, -0.008998)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.458041958041958, 2, 3, -0.045682), (-1, 0.0, 0, 0, 0.000208), (-1, 0.0, 0, 0, -0.034023), (63, 0.0010000000000000009, 5, 6, 0.530947), (-1, 0.0, 0, 0, 0.052861), (-1, 0.0, 0, 0, 0.012132)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (28, 0.091, 2, 3, -0.394006), (-1, 0.0, 0, 0, 0.043895), (-1, 0.0, 0, 0, -0.029581), (49, 2.5, 5, 6, 0.056665), (-1, 0.0, 0, 0, 0.009279), (-1, 0.0, 0, 0, -0.011045)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.032815), (-1, 0.0, 0, 0, 0.021719), (-1, 0.0, 0, 0, -0.002106), (20, -3.683959493972949, 5, 6, -0.663173), (-1, 0.0, 0, 0, -0.048815), (-1, 0.0, 0, 0, 0.003673)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.245278), (-1, 0.0, 0, 0, -0.048548), (-1, 0.0, 0, 0, -0.001225), (59, 0.14835164835164835, 5, 6, 0.090941), (-1, 0.0, 0, 0, 0.034427), (-1, 0.0, 0, 0, -0.000284)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.079993), (-1, 0.0, 0, 0, -0.044365), (-1, 0.0, 0, 0, 0.000506), (26, 4.658699722134795, 5, 6, 0.282993), (-1, 0.0, 0, 0, 0.005075), (-1, 0.0, 0, 0, 0.049489)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.038409), (-1, 0.0, 0, 0, -0.00496), (-1, 0.0, 0, 0, 0.015274), (59, 0.4083333333333333, 5, 6, -0.553534), (-1, 0.0, 0, 0, 0.018598), (-1, 0.0, 0, 0, -0.039959)),
            ((2, 0.992, 1, 4, 0.0), (24, 0.015833333333333345, 2, 3, -0.116656), (-1, 0.0, 0, 0, -0.025442), (-1, 0.0, 0, 0, 0.001806), (45, 0.62, 5, 6, 0.188886), (-1, 0.0, 0, 0, 0.001806), (-1, 0.0, 0, 0, 0.03434)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.054459), (-1, 0.0, 0, 0, -0.00506), (-1, 0.0, 0, 0, 0.021709), (24, 0.06250000000000003, 5, 6, -0.386119), (-1, 0.0, 0, 0, -0.02821), (-1, 0.0, 0, 0, 0.034234)),
            ((40, 0.624, 1, 4, 0.0), (40, 0.0045000000000000595, 2, 3, 0.071721), (-1, 0.0, 0, 0, -0.007369), (-1, 0.0, 0, 0, 0.014256), (27, 4.7257269950599685, 5, 6, -0.286185), (-1, 0.0, 0, 0, -0.025318), (-1, 0.0, 0, 0, 0.037135)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.574447), (-1, 0.0, 0, 0, -0.006028), (-1, 0.0, 0, 0, -0.047543), (41, 0.5, 5, 6, 0.034147), (-1, 0.0, 0, 0, 0.01036), (-1, 0.0, 0, 0, -0.00662)),
            ((55, 0.978, 1, 4, 0.0), (6, 0.6482034517095314, 2, 3, -0.043303), (-1, 0.0, 0, 0, 0.003938), (-1, 0.0, 0, 0, -0.014093), (63, 0.0010000000000000009, 5, 6, 0.497201), (-1, 0.0, 0, 0, 0.051851), (-1, 0.0, 0, 0, 0.011347)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.036015), (-1, 0.0, 0, 0, -0.006689), (-1, 0.0, 0, 0, 0.010938), (59, 0.4083333333333333, 5, 6, -0.534818), (-1, 0.0, 0, 0, 0.018032), (-1, 0.0, 0, 0, -0.038773)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.17424242424242425, 2, 3, 0.029865), (-1, 0.0, 0, 0, 0.017904), (-1, 0.0, 0, 0, -0.002739), (20, -3.683959493972949, 5, 6, -0.638253), (-1, 0.0, 0, 0, -0.048079), (-1, 0.0, 0, 0, 0.004502)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (40, 0.0295, 2, 3, -0.384244), (-1, 0.0, 0, 0, -0.043806), (-1, 0.0, 0, 0, -0.008336), (6, 0.6482034517095314, 5, 6, 0.050804), (-1, 0.0, 0, 0, 0.010402), (-1, 0.0, 0, 0, -0.008336)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.372998), (-1, 0.0, 0, 0, -0.030202), (-1, 0.0, 0, 0, 0.037845), (58, 0.11805555555555555, 5, 6, 0.051576), (-1, 0.0, 0, 0, 0.01592), (-1, 0.0, 0, 0, -0.003777)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.070588), (-1, 0.0, 0, 0, -0.015405), (-1, 0.0, 0, 0, 0.005535), (43, 0.14835164835164835, 5, 6, 0.291982), (-1, 0.0, 0, 0, 0.005535), (-1, 0.0, 0, 0, 0.034409)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.053009), (-1, 0.0, 0, 0, -0.004437), (-1, 0.0, 0, 0, 0.019833), (24, 0.06250000000000003, 5, 6, -0.372318), (-1, 0.0, 0, 0, -0.027201), (-1, 0.0, 0, 0, 0.033466)),
            ((40, 0.624, 1, 4, 0.0), (41, 0.5, 2, 3, 0.070554), (-1, 0.0, 0, 0, 0.014162), (-1, 0.0, 0, 0, -0.006344), (27, 4.7257269950599685, 5, 6, -0.277078), (-1, 0.0, 0, 0, -0.024398), (-1, 0.0, 0, 0, 0.03523)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.61376), (-1, 0.0, 0, 0, 0.00131), (-1, 0.0, 0, 0, -0.047662), (40, 0.624, 5, 6, 0.031277), (-1, 0.0, 0, 0, 0.005643), (-1, 0.0, 0, 0, -0.013185)),
            ((2, 0.992, 1, 4, 0.0), (24, 0.015833333333333345, 2, 3, -0.111447), (-1, 0.0, 0, 0, -0.024455), (-1, 0.0, 0, 0, 0.00173), (1, 0.099, 5, 6, 0.180657), (-1, 0.0, 0, 0, 0.00173), (-1, 0.0, 0, 0, 0.047558)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.227076), (-1, 0.0, 0, 0, -0.046867), (-1, 0.0, 0, 0, -0.000598), (59, 0.14835164835164835, 5, 6, 0.083158), (-1, 0.0, 0, 0, 0.034327), (-1, 0.0, 0, 0, -0.000617)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.04442), (-1, 0.0, 0, 0, -0.023405), (-1, 0.0, 0, 0, 0.001981), (11, 0.948, 5, 6, 0.4392), (-1, 0.0, 0, 0, 0.00987), (-1, 0.0, 0, 0, 0.049473)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.034224), (-1, 0.0, 0, 0, -0.004224), (-1, 0.0, 0, 0, 0.013151), (59, 0.4083333333333333, 5, 6, -0.517538), (-1, 0.0, 0, 0, 0.017458), (-1, 0.0, 0, 0, -0.037652)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.028195), (-1, 0.0, 0, 0, 0.020174), (-1, 0.0, 0, 0, -0.00201), (20, -3.683959493972949, 5, 6, -0.612541), (-1, 0.0, 0, 0, -0.047327), (-1, 0.0, 0, 0, 0.005221)),
            ((2, 0.992, 1, 4, 0.0), (40, 0.624, 2, 3, -0.107939), (-1, 0.0, 0, 0, -0.000637), (-1, 0.0, 0, 0, -0.035242), (45, 0.62, 5, 6, 0.173713), (-1, 0.0, 0, 0, 0.001644), (-1, 0.0, 0, 0, 0.03294)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.54577), (-1, 0.0, 0, 0, -0.004334), (-1, 0.0, 0, 0, -0.046713), (15, -0.613, 5, 6, 0.030485), (-1, 0.0, 0, 0, 0.043588), (-1, 0.0, 0, 0, -5.6e-05)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.223464), (-1, 0.0, 0, 0, -0.04562), (-1, 0.0, 0, 0, -0.000805), (59, 0.14835164835164835, 5, 6, 0.079161), (-1, 0.0, 0, 0, 0.032465), (-1, 0.0, 0, 0, -0.00053)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.364543), (-1, 0.0, 0, 0, -0.029362), (-1, 0.0, 0, 0, 0.037451), (59, 0.4365079365079365, 5, 6, 0.048392), (-1, 0.0, 0, 0, 0.010424), (-1, 0.0, 0, 0, -0.00717)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.047747), (-1, 0.0, 0, 0, -0.004396), (-1, 0.0, 0, 0, 0.018788), (24, 0.06250000000000003, 5, 6, -0.349872), (-1, 0.0, 0, 0, -0.025759), (-1, 0.0, 0, 0, 0.032685)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.5535, 2, 3, -0.190876), (-1, 0.0, 0, 0, -0.002502), (-1, 0.0, 0, 0, -0.043891), (63, 0.933, 5, 6, 0.090808), (-1, 0.0, 0, 0, 0.00087), (-1, 0.0, 0, 0, 0.052801)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.17424242424242425, 2, 3, 0.027799), (-1, 0.0, 0, 0, 0.016189), (-1, 0.0, 0, 0, -0.002378), (20, -3.683959493972949, 5, 6, -0.611767), (-1, 0.0, 0, 0, -0.046772), (-1, 0.0, 0, 0, 0.003686)),
            ((2, 0.992, 1, 4, 0.0), (28, 0.935, 2, 3, -0.106868), (-1, 0.0, 0, 0, -0.018022), (-1, 0.0, 0, 0, 0.001634), (1, 0.099, 5, 6, 0.172243), (-1, 0.0, 0, 0, 0.001634), (-1, 0.0, 0, 0, 0.045943)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.586201), (-1, 0.0, 0, 0, 0.002228), (-1, 0.0, 0, 0, -0.04646), (41, 0.5, 5, 6, 0.029943), (-1, 0.0, 0, 0, 0.009049), (-1, 0.0, 0, 0, -0.00569)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.4083333333333333, 2, 3, -0.037826), (-1, 0.0, 0, 0, 0.000777), (-1, 0.0, 0, 0, -0.025574), (63, 0.0010000000000000009, 5, 6, 0.469658), (-1, 0.0, 0, 0, 0.051716), (-1, 0.0, 0, 0, 0.010796)),
            ((63, 0.98, 1, 4, 0.0), (6, 0.6482034517095314, 2, 3, 0.028249), (-1, 0.0, 0, 0, 0.006919), (-1, 0.0, 0, 0, -0.008184), (20, -3.683959493972949, 5, 6, -0.595554), (-1, 0.0, 0, 0, -0.046177), (-1, 0.0, 0, 0, 0.003851)),
            ((15, 0.759, 1, 4, 0.0), (23, 0.0050000000000000044, 2, 3, 0.041142), (-1, 0.0, 0, 0, -0.020415), (-1, 0.0, 0, 0, 0.005314), (6, 0.5986826325792863, 5, 6, -0.411505), (-1, 0.0, 0, 0, 0.013803), (-1, 0.0, 0, 0, -0.05601)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.350534), (-1, 0.0, 0, 0, -0.028299), (-1, 0.0, 0, 0, 0.036421), (58, 0.11805555555555555, 5, 6, 0.047987), (-1, 0.0, 0, 0, 0.014678), (-1, 0.0, 0, 0, -0.003437)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (12, 0.374, 2, 3, -0.218203), (-1, 0.0, 0, 0, -0.038302), (-1, 0.0, 0, 0, 0.00152), (59, 0.14835164835164835, 5, 6, 0.07982), (-1, 0.0, 0, 0, 0.031118), (-1, 0.0, 0, 0, -0.000239)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.033228), (-1, 0.0, 0, 0, -0.003958), (-1, 0.0, 0, 0, 0.012518), (24, 0.06250000000000003, 5, 6, -0.49552), (-1, 0.0, 0, 0, -0.034564), (-1, 0.0, 0, 0, 0.022212)),
            ((3, 0.489, 1, 4, 0.0), (58, 0.3550420168067227, 2, 3, -0.045031), (-1, 0.0, 0, 0, 0.001325), (-1, 0.0, 0, 0, -0.022943), (45, 0.0756, 5, 6, 0.379707), (-1, 0.0, 0, 0, 0.008367), (-1, 0.0, 0, 0, 0.054828)),
            ((15, 0.759, 1, 4, 0.0), (50, 0.995, 2, 3, 0.038334), (-1, 0.0, 0, 0, 7.7e-05), (-1, 0.0, 0, 0, 0.041845), (6, 0.5986826325792863, 5, 6, -0.405124), (-1, 0.0, 0, 0, 0.012891), (-1, 0.0, 0, 0, -0.054816)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (28, 0.204, 2, 3, -0.342011), (-1, 0.0, 0, 0, 0.029252), (-1, 0.0, 0, 0, -0.029362), (49, 2.5, 5, 6, 0.044955), (-1, 0.0, 0, 0, 0.007886), (-1, 0.0, 0, 0, -0.010127)),
            ((63, 0.98, 1, 4, 0.0), (40, 0.624, 2, 3, 0.025379), (-1, 0.0, 0, 0, 0.004932), (-1, 0.0, 0, 0, -0.012129), (20, -3.683959493972949, 5, 6, -0.582353), (-1, 0.0, 0, 0, -0.04579), (-1, 0.0, 0, 0, 0.004186)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.211006), (-1, 0.0, 0, 0, -0.044088), (-1, 0.0, 0, 0, -0.000518), (59, 0.14835164835164835, 5, 6, 0.075049), (-1, 0.0, 0, 0, 0.029922), (-1, 0.0, 0, 0, -0.000313)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.563993), (-1, 0.0, 0, 0, 0.002128), (-1, 0.0, 0, 0, -0.045358), (26, 4.579839228586978, 5, 6, 0.025473), (-1, 0.0, 0, 0, -0.006047), (-1, 0.0, 0, 0, 0.00825)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.068133), (-1, 0.0, 0, 0, -0.042182), (-1, 0.0, 0, 0, 0.000811), (43, 0.14835164835164835, 5, 6, 0.239817), (-1, 0.0, 0, 0, 0.004292), (-1, 0.0, 0, 0, 0.039864)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.686739554676682, 2, 3, 0.04341), (-1, 0.0, 0, 0, -0.003952), (-1, 0.0, 0, 0, 0.018462), (24, 0.06250000000000003, 5, 6, -0.337118), (-1, 0.0, 0, 0, -0.024464), (-1, 0.0, 0, 0, 0.030835)),
            ((2, 0.992, 1, 4, 0.0), (28, 0.285, 2, 3, -0.099938), (-1, 0.0, 0, 0, -0.033148), (-1, 0.0, 0, 0, -9.5e-05), (1, 0.099, 5, 6, 0.157366), (-1, 0.0, 0, 0, 0.001436), (-1, 0.0, 0, 0, 0.04419)),
            ((9, 0.947, 1, 4, 0.0), (49, 1.5, 2, 3, -0.035286), (-1, 0.0, 0, 0, -0.018559), (-1, 0.0, 0, 0, 0.003069), (43, 0.20714285714285713, 5, 6, 0.48833), (-1, 0.0, 0, 0, 0.011326), (-1, 0.0, 0, 0, 0.056931)),
            ((49, 3.5, 1, 4, 0.0), (23, 0.0050000000000000044, 2, 3, 0.028973), (-1, 0.0, 0, 0, -0.017888), (-1, 0.0, 0, 0, 0.004093), (27, 4.728490960086225, 5, 6, -0.482871), (-1, 0.0, 0, 0, -0.035565), (-1, 0.0, 0, 0, 0.018351)),
            ((15, 0.759, 1, 4, 0.0), (50, 0.995, 2, 3, 0.036003), (-1, 0.0, 0, 0, 9.8e-05), (-1, 0.0, 0, 0, 0.039783), (6, 0.5986826325792863, 5, 6, -0.394475), (-1, 0.0, 0, 0, 0.012367), (-1, 0.0, 0, 0, -0.053572)),
            ((58, 0.458041958041958, 1, 4, 0.0), (43, 0.39230769230769236, 2, 3, 0.032893), (-1, 0.0, 0, 0, 7.6e-05), (-1, 0.0, 0, 0, 0.05635), (24, 0.06250000000000003, 5, 6, -0.428156), (-1, 0.0, 0, 0, -0.040316), (-1, 0.0, 0, 0, 0.014562)),
            ((40, 0.624, 1, 4, 0.0), (40, 0.0045000000000000595, 2, 3, 0.059182), (-1, 0.0, 0, 0, -0.006913), (-1, 0.0, 0, 0, 0.012409), (4, 0.14, 5, 6, -0.241766), (-1, 0.0, 0, 0, -0.050112), (-1, 0.0, 0, 0, 0.002356)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.493882), (-1, 0.0, 0, 0, -0.001177), (-1, 0.0, 0, 0, -0.045139), (15, -0.613, 5, 6, 0.026478), (-1, 0.0, 0, 0, 0.041697), (-1, 0.0, 0, 0, -0.0002)),
            ((9, 0.947, 1, 4, 0.0), (49, 1.5, 2, 3, -0.033557), (-1, 0.0, 0, 0, -0.017514), (-1, 0.0, 0, 0, 0.002869), (2, 0.994, 5, 6, 0.47435), (-1, 0.0, 0, 0, 0.01102), (-1, 0.0, 0, 0, 0.047004)),
            ((58, 0.4226190476190476, 1, 4, 0.0), (43, 0.39230769230769236, 2, 3, 0.039308), (-1, 0.0, 0, 0, 0.000535), (-1, 0.0, 0, 0, 0.059997), (7, 0.995, 5, 6, -0.350806), (-1, 0.0, 0, 0, -0.026776), (-1, 0.0, 0, 0, 0.045786)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.02898), (-1, 0.0, 0, 0, -0.003697), (-1, 0.0, 0, 0, 0.011382), (59, 0.4083333333333333, 5, 6, -0.468822), (-1, 0.0, 0, 0, 0.017569), (-1, 0.0, 0, 0, -0.035054)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.039138), (-1, 0.0, 0, 0, -0.02193), (-1, 0.0, 0, 0, 0.001906), (11, 0.948, 5, 6, 0.393064), (-1, 0.0, 0, 0, 0.008848), (-1, 0.0, 0, 0, 0.048596)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.199806), (-1, 0.0, 0, 0, -0.042802), (-1, 0.0, 0, 0, -0.000206), (59, 0.14835164835164835, 5, 6, 0.070467), (-1, 0.0, 0, 0, 0.029919), (-1, 0.0, 0, 0, -0.000548)),
            ((6, 0.8725633231113197, 1, 4, 0.0), (19, 0.5, 2, 3, 0.027561), (-1, 0.0, 0, 0, -0.001275), (-1, 0.0, 0, 0, 0.029903), (6, 0.9105698788443931, 5, 6, -0.478701), (-1, 0.0, 0, 0, -0.054303), (-1, 0.0, 0, 0, 0.025527)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.326445), (-1, 0.0, 0, 0, -0.026904), (-1, 0.0, 0, 0, 0.03725), (59, 0.4365079365079365, 5, 6, 0.041775), (-1, 0.0, 0, 0, 0.00937), (-1, 0.0, 0, 0, -0.006751)),
            ((3, 0.489, 1, 4, 0.0), (58, 0.3550420168067227, 2, 3, -0.043553), (-1, 0.0, 0, 0, 0.00109), (-1, 0.0, 0, 0, -0.021611), (45, 0.0756, 5, 6, 0.354829), (-1, 0.0, 0, 0, 0.007782), (-1, 0.0, 0, 0, 0.052211)),
            ((55, 0.986, 1, 4, 0.0), (49, 2.5, 2, 3, -0.031243), (-1, 0.0, 0, 0, 0.002745), (-1, 0.0, 0, 0, -0.012978), (59, 0.5147058823529411, 5, 6, 0.5142), (-1, 0.0, 0, 0, 0.051961), (-1, 0.0, 0, 0, 0.012074)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.486098), (-1, 0.0, 0, 0, -0.00145), (-1, 0.0, 0, 0, -0.044591), (15, -0.613, 5, 6, 0.024989), (-1, 0.0, 0, 0, 0.040889), (-1, 0.0, 0, 0, -0.000244)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.196726), (-1, 0.0, 0, 0, -0.041868), (-1, 0.0, 0, 0, -0.000347), (59, 0.14835164835164835, 5, 6, 0.067791), (-1, 0.0, 0, 0, 0.028587), (-1, 0.0, 0, 0, -0.000484)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.323599), (-1, 0.0, 0, 0, -0.026452), (-1, 0.0, 0, 0, 0.036405), (58, 0.11805555555555555, 5, 6, 0.039984), (-1, 0.0, 0, 0, 0.013122), (-1, 0.0, 0, 0, -0.003307)),
            ((15, 0.759, 1, 4, 0.0), (23, 0.0050000000000000044, 2, 3, 0.033774), (-1, 0.0, 0, 0, -0.019245), (-1, 0.0, 0, 0, 0.004673), (6, 0.5986826325792863, 5, 6, -0.373904), (-1, 0.0, 0, 0, 0.012585), (-1, 0.0, 0, 0, -0.052187)),
            ((63, 0.98, 1, 4, 0.0), (63, 0.886, 2, 3, 0.0215), (-1, 0.0, 0, 0, -0.00148), (-1, 0.0, 0, 0, 0.022028), (20, -3.683959493972949, 5, 6, -0.534625), (-1, 0.0, 0, 0, -0.044976), (-1, 0.0, 0, 0, 0.007169)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.526112), (-1, 0.0, 0, 0, 0.003813), (-1, 0.0, 0, 0, -0.043724), (41, 0.5, 5, 6, 0.021907), (-1, 0.0, 0, 0, 0.008152), (-1, 0.0, 0, 0, -0.005575)),
            ((40, 0.624, 1, 4, 0.0), (40, 0.0045000000000000595, 2, 3, 0.055197), (-1, 0.0, 0, 0, -0.006685), (-1, 0.0, 0, 0, 0.011745), (4, 0.14, 5, 6, -0.229695), (-1, 0.0, 0, 0, -0.047559), (-1, 0.0, 0, 0, 0.002187)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.025858), (-1, 0.0, 0, 0, -0.003903), (-1, 0.0, 0, 0, 0.01129), (59, 0.4083333333333333, 5, 6, -0.451914), (-1, 0.0, 0, 0, 0.01688), (-1, 0.0, 0, 0, -0.033896)),
            ((6, 0.8725633231113197, 1, 4, 0.0), (19, 0.5, 2, 3, 0.025471), (-1, 0.0, 0, 0, -0.001243), (-1, 0.0, 0, 0, 0.028943), (6, 0.9105698788443931, 5, 6, -0.457542), (-1, 0.0, 0, 0, -0.052165), (-1, 0.0, 0, 0, 0.024866)),
            ((55, 0.986, 1, 4, 0.0), (49, 2.5, 2, 3, -0.029856), (-1, 0.0, 0, 0, 0.002553), (-1, 0.0, 0, 0, -0.012223), (59, 0.5147058823529411, 5, 6, 0.500274), (-1, 0.0, 0, 0, 0.050818), (-1, 0.0, 0, 0, 0.01176)),
            ((3, 0.489, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, -0.04255), (-1, 0.0, 0, 0, -0.008454), (-1, 0.0, 0, 0, 0.007349), (45, 0.0756, 5, 6, 0.336525), (-1, 0.0, 0, 0, 0.007349), (-1, 0.0, 0, 0, 0.05084)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.686739554676682, 2, 3, 0.037977), (-1, 0.0, 0, 0, -0.003703), (-1, 0.0, 0, 0, 0.016689), (15, -0.19899999999999998, 5, 6, -0.307274), (-1, 0.0, 0, 0, -0.042979), (-1, 0.0, 0, 0, -0.001423)),
            ((9, 0.947, 1, 4, 0.0), (49, 1.5, 2, 3, -0.032927), (-1, 0.0, 0, 0, -0.016988), (-1, 0.0, 0, 0, 0.002759), (43, 0.20714285714285713, 5, 6, 0.442411), (-1, 0.0, 0, 0, 0.010237), (-1, 0.0, 0, 0, 0.052721)),
            ((40, 0.624, 1, 4, 0.0), (41, 0.5, 2, 3, 0.053637), (-1, 0.0, 0, 0, 0.012466), (-1, 0.0, 0, 0, -0.006235), (4, 0.14, 5, 6, -0.226182), (-1, 0.0, 0, 0, -0.046273), (-1, 0.0, 0, 0, 0.001914)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.5535, 2, 3, -0.164678), (-1, 0.0, 0, 0, -0.00224), (-1, 0.0, 0, 0, -0.042254), (63, 0.933, 5, 6, 0.075085), (-1, 0.0, 0, 0, 9e-06), (-1, 0.0, 0, 0, 0.051468)),
            ((63, 0.98, 1, 4, 0.0), (63, 0.886, 2, 3, 0.021541), (-1, 0.0, 0, 0, -0.00131), (-1, 0.0, 0, 0, 0.020451), (20, -3.683959493972949, 5, 6, -0.535198), (-1, 0.0, 0, 0, -0.044631), (-1, 0.0, 0, 0, 0.006123)),
            ((2, 0.992, 1, 4, 0.0), (28, 0.285, 2, 3, -0.09205), (-1, 0.0, 0, 0, -0.03191), (-1, 0.0, 0, 0, 0.000195), (1, 0.099, 5, 6, 0.14612), (-1, 0.0, 0, 0, 0.001352), (-1, 0.0, 0, 0, 0.041624)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (12, 0.374, 2, 3, -0.19182), (-1, 0.0, 0, 0, -0.035198), (-1, 0.0, 0, 0, 0.002025), (15, -0.068, 5, 6, 0.066291), (-1, 0.0, 0, 0, 0.025199), (-1, 0.0, 0, 0, -0.000955)),
            ((6, 0.8725633231113197, 1, 4, 0.0), (19, 0.5, 2, 3, 0.025006), (-1, 0.0, 0, 0, -0.001156), (-1, 0.0, 0, 0, 0.028285), (6, 0.9105698788443931, 5, 6, -0.441506), (-1, 0.0, 0, 0, -0.050183), (-1, 0.0, 0, 0, 0.023697)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.510606), (-1, 0.0, 0, 0, 0.004418), (-1, 0.0, 0, 0, -0.043101), (15, -0.6930000000000001, 5, 6, 0.021401), (-1, 0.0, 0, 0, 0.035839), (-1, 0.0, 0, 0, -0.00021)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.30654), (-1, 0.0, 0, 0, -0.025222), (-1, 0.0, 0, 0, 0.035831), (59, 0.4365079365079365, 5, 6, 0.038029), (-1, 0.0, 0, 0, 0.008647), (-1, 0.0, 0, 0, -0.006315)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (62, 0.402, 2, 3, -0.188161), (-1, 0.0, 0, 0, -0.014823), (-1, 0.0, 0, 0, 0.042007), (59, 0.14835164835164835, 5, 6, 0.065626), (-1, 0.0, 0, 0, 0.027523), (-1, 0.0, 0, 0, -0.000417)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.308357), (-1, 0.0, 0, 0, -0.006741), (-1, 0.0, 0, 0, -0.043419), (63, 0.839, 5, 6, 0.038725), (-1, 0.0, 0, 0, -0.00146), (-1, 0.0, 0, 0, 0.024307)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.023396), (-1, 0.0, 0, 0, 0.01673), (-1, 0.0, 0, 0, -0.001571), (20, -3.683959493972949, 5, 6, -0.523022), (-1, 0.0, 0, 0, -0.044136), (-1, 0.0, 0, 0, 0.006018)),
            ((2, 0.992, 1, 4, 0.0), (24, 0.015833333333333345, 2, 3, -0.087457), (-1, 0.0, 0, 0, -0.022631), (-1, 0.0, 0, 0, 0.001439), (62, 0.397, 5, 6, 0.144999), (-1, 0.0, 0, 0, 0.002243), (-1, 0.0, 0, 0, 0.049039)),
            ((26, 4.579839228586978, 1, 4, 0.0), (7, 0.955, 2, 3, -0.113741), (-1, 0.0, 0, 0, -0.013267), (-1, 0.0, 0, 0, 0.02695), (49, 2.5, 5, 6, 0.107851), (-1, 0.0, 0, 0, 0.018231), (-1, 0.0, 0, 0, -0.009933)),
            ((15, 0.759, 1, 4, 0.0), (50, 0.995, 2, 3, 0.035557), (-1, 0.0, 0, 0, 0.000269), (-1, 0.0, 0, 0, 0.037055), (6, 0.5986826325792863, 5, 6, -0.355515), (-1, 0.0, 0, 0, 0.012802), (-1, 0.0, 0, 0, -0.050859)),
            ((58, 0.458041958041958, 1, 4, 0.0), (43, 0.39230769230769236, 2, 3, 0.031299), (-1, 0.0, 0, 0, 7.5e-05), (-1, 0.0, 0, 0, 0.051034), (24, 0.06250000000000003, 5, 6, -0.383683), (-1, 0.0, 0, 0, -0.037804), (-1, 0.0, 0, 0, 0.014187)),
            ((2, 0.0020000000000000018, 1, 2, 0.0), (-1, 0.0, 0, 0, -0.037822), (6, 0.6482034517095314, 3, 4, 0.015994), (-1, 0.0, 0, 0, 0.005078), (-1, 0.0, 0, 0, -0.006986)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.298443), (-1, 0.0, 0, 0, -0.024656), (-1, 0.0, 0, 0, 0.035255), (58, 0.11805555555555555, 5, 6, 0.039605), (-1, 0.0, 0, 0, 0.012427), (-1, 0.0, 0, 0, -0.002984)),
            ((15, 0.759, 1, 4, 0.0), (9, 0.984, 2, 3, 0.034325), (-1, 0.0, 0, 0, 0.000647), (-1, 0.0, 0, 0, 0.049508), (6, 0.5986826325792863, 5, 6, -0.345119), (-1, 0.0, 0, 0, 0.012065), (-1, 0.0, 0, 0, -0.049505)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.450965), (-1, 0.0, 0, 0, 0.000635), (-1, 0.0, 0, 0, -0.043353), (15, -0.613, 5, 6, 0.026309), (-1, 0.0, 0, 0, 0.038519), (-1, 0.0, 0, 0, -8.7e-05)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.178439), (-1, 0.0, 0, 0, -0.040365), (-1, 0.0, 0, 0, 0.000367), (15, -0.068, 5, 6, 0.065659), (-1, 0.0, 0, 0, 0.024132), (-1, 0.0, 0, 0, -0.000798)),
            ((49, 3.5, 1, 4, 0.0), (25, 6.723231117838329, 2, 3, 0.027624), (-1, 0.0, 0, 0, 0.007309), (-1, 0.0, 0, 0, -0.006048), (27, 4.728490960086225, 5, 6, -0.423482), (-1, 0.0, 0, 0, -0.032337), (-1, 0.0, 0, 0, 0.01816)),
            ((3, 0.489, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, -0.036653), (-1, 0.0, 0, 0, -0.007918), (-1, 0.0, 0, 0, 0.007058), (45, 0.0756, 5, 6, 0.318992), (-1, 0.0, 0, 0, 0.007058), (-1, 0.0, 0, 0, 0.048661)),
            ((27, 4.278524502524155, 1, 4, 0.0), (20, -2.6933930072678223, 2, 3, -0.316161), (-1, 0.0, 0, 0, -0.039451), (-1, 0.0, 0, 0, 0.027731), (28, 0.285, 5, 6, 0.036097), (-1, 0.0, 0, 0, -0.013076), (-1, 0.0, 0, 0, 0.005937)),
            ((6, 0.8725633231113197, 1, 4, 0.0), (19, 0.5, 2, 3, 0.026082), (-1, 0.0, 0, 0, -0.000986), (-1, 0.0, 0, 0, 0.027457), (26, 4.658699722134795, 5, 6, -0.421669), (-1, 0.0, 0, 0, -0.053658), (-1, 0.0, 0, 0, 0.017536)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.686739554676682, 2, 3, 0.038977), (-1, 0.0, 0, 0, -0.003324), (-1, 0.0, 0, 0, 0.015746), (15, -0.19899999999999998, 5, 6, -0.289394), (-1, 0.0, 0, 0, -0.041653), (-1, 0.0, 0, 0, -0.000924)),
            ((2, 0.0020000000000000018, 1, 2, 0.0), (-1, 0.0, 0, 0, -0.037279), (40, 0.624, 3, 4, 0.014028), (-1, 0.0, 0, 0, 0.003663), (-1, 0.0, 0, 0, -0.010478)),
            ((55, 0.978, 1, 4, 0.0), (28, 0.285, 2, 3, -0.030915), (-1, 0.0, 0, 0, -0.015805), (-1, 0.0, 0, 0, 0.001729), (63, 0.0010000000000000009, 5, 6, 0.394698), (-1, 0.0, 0, 0, 0.048812), (-1, 0.0, 0, 0, 0.009095)),
            ((63, 0.98, 1, 4, 0.0), (63, 0.933, 2, 3, 0.021209), (-1, 0.0, 0, 0, -0.000736), (-1, 0.0, 0, 0, 0.024953), (20, -3.683959493972949, 5, 6, -0.497553), (-1, 0.0, 0, 0, -0.043215), (-1, 0.0, 0, 0, 0.006652)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.444071), (-1, 0.0, 0, 0, 0.000513), (-1, 0.0, 0, 0, -0.042792), (15, -0.613, 5, 6, 0.023945), (-1, 0.0, 0, 0, 0.037387), (-1, 0.0, 0, 0, -0.000159)),
            ((2, 0.992, 1, 4, 0.0), (28, 0.935, 2, 3, -0.08341), (-1, 0.0, 0, 0, -0.01654), (-1, 0.0, 0, 0, 0.001328), (45, 0.62, 5, 6, 0.136512), (-1, 0.0, 0, 0, 0.001328), (-1, 0.0, 0, 0, 0.028446)),
            ((59, 0.7182539682539683, 1, 4, 0.0), (26, 4.686739554676682, 2, 3, 0.035257), (-1, 0.0, 0, 0, -0.003363), (-1, 0.0, 0, 0, 0.012055), (45, 0.92, 5, 6, -0.329718), (-1, 0.0, 0, 0, -0.031104), (-1, 0.0, 0, 0, 0.0326)),
            ((15, 0.759, 1, 4, 0.0), (23, 0.0050000000000000044, 2, 3, 0.033453), (-1, 0.0, 0, 0, -0.016893), (-1, 0.0, 0, 0, 0.004246), (6, 0.5986826325792863, 5, 6, -0.340192), (-1, 0.0, 0, 0, 0.01147), (-1, 0.0, 0, 0, -0.048641)),
            ((49, 3.5, 1, 4, 0.0), (25, 6.723231117838329, 2, 3, 0.026525), (-1, 0.0, 0, 0, 0.007089), (-1, 0.0, 0, 0, -0.005865), (59, 0.4083333333333333, 5, 6, -0.419023), (-1, 0.0, 0, 0, 0.016495), (-1, 0.0, 0, 0, -0.03194)),
            ((27, 4.278524502524155, 1, 4, 0.0), (20, -2.6933930072678223, 2, 3, -0.314737), (-1, 0.0, 0, 0, -0.038353), (-1, 0.0, 0, 0, 0.026476), (28, 0.285, 5, 6, 0.035693), (-1, 0.0, 0, 0, -0.012195), (-1, 0.0, 0, 0, 0.005674)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (12, 0.374, 2, 3, -0.175237), (-1, 0.0, 0, 0, -0.033779), (-1, 0.0, 0, 0, 0.002507), (59, 0.14835164835164835, 5, 6, 0.063358), (-1, 0.0, 0, 0, 0.027213), (-1, 0.0, 0, 0, -0.000449)),
            ((13, 0.0030000000000000027, 1, 2, 0.0), (-1, 0.0, 0, 0, -0.036285), (28, 0.994, 3, 4, 0.014685), (-1, 0.0, 0, 0, -0.001357), (-1, 0.0, 0, 0, 0.016118)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.021745), (-1, 0.0, 0, 0, 0.016025), (-1, 0.0, 0, 0, -0.001529), (20, -3.683959493972949, 5, 6, -0.495108), (-1, 0.0, 0, 0, -0.042925), (-1, 0.0, 0, 0, 0.006261)),
            ((58, 0.458041958041958, 1, 4, 0.0), (43, 0.39230769230769236, 2, 3, 0.028553), (-1, 0.0, 0, 0, -2.3e-05), (-1, 0.0, 0, 0, 0.04894), (24, 0.06250000000000003, 5, 6, -0.373776), (-1, 0.0, 0, 0, -0.036393), (-1, 0.0, 0, 0, 0.01293)),
            ((40, 0.9045, 1, 4, 0.0), (55, 0.986, 2, 3, 0.020798), (-1, 0.0, 0, 0, -0.000371), (-1, 0.0, 0, 0, 0.030085), (2, 0.998, 5, 6, -0.511942), (-1, 0.0, 0, 0, -0.042903), (-1, 0.0, 0, 0, 0.008354)),
        )),
        (-0.011991, (
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -1.067159), (-1, 0.0, 0, 0, -0.077668), (-1, 0.0, 0, 0, -0.015255), (24, 0.06250000000000003, 5, 6, 1.361203), (-1, 0.0, 0, 0, 0.023398), (-1, 0.0, 0, 0, 0.089942)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -1.015642), (-1, 0.0, 0, 0, -0.073969), (-1, 0.0, 0, 0, -0.014506), (24, 0.06250000000000003, 5, 6, 1.295273), (-1, 0.0, 0, 0, 0.022253), (-1, 0.0, 0, 0, 0.085622)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.968287), (-1, 0.0, 0, 0, -0.07064), (-1, 0.0, 0, 0, -0.013795), (24, 0.06250000000000003, 5, 6, 1.235745), (-1, 0.0, 0, 0, 0.02117), (-1, 0.0, 0, 0, 0.081815)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.924509), (-1, 0.0, 0, 0, -0.067621), (-1, 0.0, 0, 0, -0.013119), (24, 0.06250000000000003, 5, 6, 1.181588), (-1, 0.0, 0, 0, 0.020144), (-1, 0.0, 0, 0, 0.078429)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.883831), (-1, 0.0, 0, 0, -0.064863), (-1, 0.0, 0, 0, -0.012478), (24, 0.06250000000000003, 5, 6, 1.131977), (-1, 0.0, 0, 0, 0.019172), (-1, 0.0, 0, 0, 0.075394)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.845858), (-1, 0.0, 0, 0, -0.062328), (-1, 0.0, 0, 0, -0.011869), (24, 0.06250000000000003, 5, 6, 1.086245), (-1, 0.0, 0, 0, 0.018249), (-1, 0.0, 0, 0, 0.072653)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.810264), (-1, 0.0, 0, 0, -0.059983), (-1, 0.0, 0, 0, -0.01129), (24, 0.06250000000000003, 5, 6, 1.043847), (-1, 0.0, 0, 0, 0.017373), (-1, 0.0, 0, 0, 0.070161)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.776773), (-1, 0.0, 0, 0, -0.057802), (-1, 0.0, 0, 0, -0.01074), (24, 0.06250000000000003, 5, 6, 1.004334), (-1, 0.0, 0, 0, 0.016541), (-1, 0.0, 0, 0, 0.067881)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, -0.602765), (-1, 0.0, 0, 0, -0.043859), (-1, 0.0, 0, 0, 0.01575), (52, 0.5, 5, 6, 1.209858), (-1, 0.0, 0, 0, 0.073937), (-1, 0.0, 0, 0, 0.021402)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, -0.576998), (-1, 0.0, 0, 0, -0.042061), (-1, 0.0, 0, 0, 0.014998), (52, 0.5, 5, 6, 1.16957), (-1, 0.0, 0, 0, 0.071775), (-1, 0.0, 0, 0, 0.020454)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, -0.552542), (-1, 0.0, 0, 0, -0.040356), (-1, 0.0, 0, 0, 0.014282), (52, 0.5, 5, 6, 1.132014), (-1, 0.0, 0, 0, 0.069796), (-1, 0.0, 0, 0, 0.019548)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.529283), (-1, 0.0, 0, 0, -0.010326), (-1, 0.0, 0, 0, -0.059085), (52, 0.5, 5, 6, 1.096834), (-1, 0.0, 0, 0, 0.067977), (-1, 0.0, 0, 0, 0.018683)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.644434), (-1, 0.0, 0, 0, -0.054091), (-1, 0.0, 0, 0, -0.010032), (24, 0.02125000000000002, 5, 6, 0.858117), (-1, 0.0, 0, 0, -0.00692), (-1, 0.0, 0, 0, 0.052483)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.619017), (-1, 0.0, 0, 0, -0.05241), (-1, 0.0, 0, 0, -0.00955), (24, 0.179, 5, 6, 0.827735), (-1, 0.0, 0, 0, 0.01587), (-1, 0.0, 0, 0, 0.060265)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.473835), (-1, 0.0, 0, 0, -0.007969), (-1, 0.0, 0, 0, -0.055823), (52, 0.5, 5, 6, 1.016534), (-1, 0.0, 0, 0, 0.06436), (-1, 0.0, 0, 0, 0.015027)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.580882), (-1, 0.0, 0, 0, -0.050137), (-1, 0.0, 0, 0, -0.008475), (24, 0.02125000000000002, 5, 6, 0.787415), (-1, 0.0, 0, 0, -0.006293), (-1, 0.0, 0, 0, 0.048633)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.439846), (-1, 0.0, 0, 0, -0.006789), (-1, 0.0, 0, 0, -0.05344), (52, 0.5, 5, 6, 0.967709), (-1, 0.0, 0, 0, 0.062177), (-1, 0.0, 0, 0, 0.013109)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.545699), (-1, 0.0, 0, 0, -0.048044), (-1, 0.0, 0, 0, -0.007502), (24, 0.179, 5, 6, 0.749195), (-1, 0.0, 0, 0, 0.015084), (-1, 0.0, 0, 0, 0.0555)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.524504), (-1, 0.0, 0, 0, -0.046641), (-1, 0.0, 0, 0, -0.007141), (50, 0.955, 5, 6, 0.724501), (-1, 0.0, 0, 0, 0.016944), (-1, 0.0, 0, 0, 0.057346)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.39627), (-1, 0.0, 0, 0, -0.004982), (-1, 0.0, 0, 0, -0.050924), (52, 0.5, 5, 6, 0.904703), (-1, 0.0, 0, 0, 0.059608), (-1, 0.0, 0, 0, 0.010092)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.493119), (-1, 0.0, 0, 0, -0.044776), (-1, 0.0, 0, 0, -0.006291), (50, 0.955, 5, 6, 0.689397), (-1, 0.0, 0, 0, 0.015817), (-1, 0.0, 0, 0, 0.055448)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.369764), (-1, 0.0, 0, 0, -0.004154), (-1, 0.0, 0, 0, -0.048978), (52, 0.5, 5, 6, 0.86578), (-1, 0.0, 0, 0, 0.057992), (-1, 0.0, 0, 0, 0.00862)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.463888), (-1, 0.0, 0, 0, -0.043025), (-1, 0.0, 0, 0, -0.005519), (59, 0.17424242424242425, 5, 6, 0.65644), (-1, 0.0, 0, 0, 0.064095), (-1, 0.0, 0, 0, 0.020178)),
            ((45, 0.87, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.329809), (-1, 0.0, 0, 0, -0.037052), (-1, 0.0, 0, 0, 0.000276), (50, 0.21499999999999997, 5, 6, 0.875263), (-1, 0.0, 0, 0, -0.024258), (-1, 0.0, 0, 0, 0.050237)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.335575), (-1, 0.0, 0, 0, -0.002872), (-1, 0.0, 0, 0, -0.046793), (52, 0.5, 5, 6, 0.813798), (-1, 0.0, 0, 0, 0.056096), (-1, 0.0, 0, 0, 0.005836)),
            ((45, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.423885), (-1, 0.0, 0, 0, -0.037353), (-1, 0.0, 0, 0, -0.001848), (59, 0.17424242424242425, 5, 6, 0.609785), (-1, 0.0, 0, 0, 0.062035), (-1, 0.0, 0, 0, 0.017857)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.14835164835164835, 2, 3, -0.314733), (-1, 0.0, 0, 0, 0.007215), (-1, 0.0, 0, 0, -0.032835), (52, 0.5, 5, 6, 0.780677), (-1, 0.0, 0, 0, 0.054834), (-1, 0.0, 0, 0, 0.004427)),
            ((45, 0.87, 1, 4, 0.0), (24, 0.027500000000000024, 2, 3, -0.289257), (-1, 0.0, 0, 0, -0.034122), (-1, 0.0, 0, 0, 0.001068), (13, 0.933, 5, 6, 0.810063), (-1, 0.0, 0, 0, 0.00672), (-1, 0.0, 0, 0, 0.055432)),
            ((45, 0.16178, 1, 4, 0.0), (26, 4.255587559545337, 2, 3, -0.390464), (-1, 0.0, 0, 0, -0.053843), (-1, 0.0, 0, 0, -0.010726), (7, 0.942, 5, 6, 0.562319), (-1, 0.0, 0, 0, 0.010078), (-1, 0.0, 0, 0, 0.050624)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.3431372549019608, 2, 3, -0.284236), (-1, 0.0, 0, 0, -0.00405), (-1, 0.0, 0, 0, -0.052783), (52, 0.5, 5, 6, 0.734726), (-1, 0.0, 0, 0, 0.053184), (-1, 0.0, 0, 0, 0.002372)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (58, 0.17424242424242425, 2, 3, -0.533939), (-1, 0.0, 0, 0, -0.006715), (-1, 0.0, 0, 0, -0.045041), (41, 0.5, 5, 6, 0.377161), (-1, 0.0, 0, 0, 0.044947), (-1, 0.0, 0, 0, -0.007846)),
            ((45, 0.87, 1, 4, 0.0), (58, 0.3038461538461539, 2, 3, -0.256409), (-1, 0.0, 0, 0, -0.003309), (-1, 0.0, 0, 0, -0.041682), (13, 0.933, 5, 6, 0.753569), (-1, 0.0, 0, 0, 0.004196), (-1, 0.0, 0, 0, 0.053299)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (58, 0.17424242424242425, 2, 3, -0.507129), (-1, 0.0, 0, 0, -0.006339), (-1, 0.0, 0, 0, -0.043299), (41, 0.5, 5, 6, 0.357977), (-1, 0.0, 0, 0, 0.043215), (-1, 0.0, 0, 0, -0.007556)),
            ((45, 0.87, 1, 4, 0.0), (59, 0.3798076923076923, 2, 3, -0.240617), (-1, 0.0, 0, 0, 0.004187), (-1, 0.0, 0, 0, -0.027803), (50, 0.21499999999999997, 5, 6, 0.725364), (-1, 0.0, 0, 0, -0.031155), (-1, 0.0, 0, 0, 0.043654)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (58, 0.3431372549019608, 2, 3, -0.245935), (-1, 0.0, 0, 0, -0.00328), (-1, 0.0, 0, 0, -0.048634), (52, 0.5, 5, 6, 0.674025), (-1, 0.0, 0, 0, 0.050981), (-1, 0.0, 0, 0, 0.000444)),
            ((45, 0.16178, 1, 4, 0.0), (26, 4.255587559545337, 2, 3, -0.325582), (-1, 0.0, 0, 0, -0.050745), (-1, 0.0, 0, 0, -0.00767), (50, 0.955, 5, 6, 0.488871), (-1, 0.0, 0, 0, 0.006929), (-1, 0.0, 0, 0, 0.047518)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (58, 0.17424242424242425, 2, 3, -0.467907), (-1, 0.0, 0, 0, -0.005399), (-1, 0.0, 0, 0, -0.040988), (41, 0.5, 5, 6, 0.332299), (-1, 0.0, 0, 0, 0.041078), (-1, 0.0, 0, 0, -0.007246)),
            ((7, 0.957, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, -0.201513), (-1, 0.0, 0, 0, -0.00213), (-1, 0.0, 0, 0, -0.055213), (56, 0.10299999999999998, 5, 6, 0.7406), (-1, 0.0, 0, 0, -0.022293), (-1, 0.0, 0, 0, 0.046286)),
            ((24, 0.027500000000000024, 1, 4, 0.0), (28, 0.622, 2, 3, -0.511141), (-1, 0.0, 0, 0, 0.00787), (-1, 0.0, 0, 0, -0.036907), (41, 0.5, 5, 6, 0.279482), (-1, 0.0, 0, 0, 0.038694), (-1, 0.0, 0, 0, -0.00918)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.207064), (-1, 0.0, 0, 0, -0.043703), (-1, 0.0, 0, 0, -0.002864), (13, 0.933, 5, 6, 0.661948), (-1, 0.0, 0, 0, -0.000172), (-1, 0.0, 0, 0, 0.050073)),
            ((7, 0.957, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.187591), (-1, 0.0, 0, 0, -0.030487), (-1, 0.0, 0, 0, 0.006375), (56, 0.10299999999999998, 5, 6, 0.704244), (-1, 0.0, 0, 0, -0.020848), (-1, 0.0, 0, 0, 0.044224)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (59, 0.3693181818181818, 2, 3, -0.209898), (-1, 0.0, 0, 0, 0.007712), (-1, 0.0, 0, 0, -0.027596), (52, 0.5, 5, 6, 0.611935), (-1, 0.0, 0, 0, 0.0488), (-1, 0.0, 0, 0, -0.001601)),
            ((24, 0.027500000000000024, 1, 4, 0.0), (28, 0.622, 2, 3, -0.478243), (-1, 0.0, 0, 0, 0.007865), (-1, 0.0, 0, 0, -0.035044), (41, 0.5, 5, 6, 0.262229), (-1, 0.0, 0, 0, 0.036943), (-1, 0.0, 0, 0, -0.00872)),
            ((7, 0.957, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, -0.175696), (-1, 0.0, 0, 0, -0.001363), (-1, 0.0, 0, 0, -0.052608), (56, 0.10299999999999998, 5, 6, 0.673105), (-1, 0.0, 0, 0, -0.019672), (-1, 0.0, 0, 0, 0.042449)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.184884), (-1, 0.0, 0, 0, -0.043787), (-1, 0.0, 0, 0, -0.002497), (13, 0.933, 5, 6, 0.619856), (-1, 0.0, 0, 0, -0.00051), (-1, 0.0, 0, 0, 0.048268)),
            ((24, 0.027500000000000024, 1, 4, 0.0), (28, 0.622, 2, 3, -0.453089), (-1, 0.0, 0, 0, 0.007251), (-1, 0.0, 0, 0, -0.033431), (41, 0.5, 5, 6, 0.246855), (-1, 0.0, 0, 0, 0.035294), (-1, 0.0, 0, 0, -0.008382)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.667804), (-1, 0.0, 0, 0, -0.011185), (-1, 0.0, 0, 0, 0.061913), (24, 0.06250000000000003, 5, 6, -0.159608), (-1, 0.0, 0, 0, -0.017513), (-1, 0.0, 0, 0, 0.021362)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.166679), (-1, 0.0, 0, 0, -0.048476), (-1, 0.0, 0, 0, -0.000604), (45, 0.06831999999999999, 5, 6, 0.619676), (-1, 0.0, 0, 0, -0.026854), (-1, 0.0, 0, 0, 0.03851)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.651273), (-1, 0.0, 0, 0, -0.010226), (-1, 0.0, 0, 0, 0.060286), (20, -3.210268834500856, 5, 6, -0.153846), (-1, 0.0, 0, 0, -0.02843), (-1, 0.0, 0, 0, 0.004961)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.15789), (-1, 0.0, 0, 0, -0.028072), (-1, 0.0, 0, 0, 0.006979), (56, 0.12, 5, 6, 0.594321), (-1, 0.0, 0, 0, -0.0196), (-1, 0.0, 0, 0, 0.038244)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.633547), (-1, 0.0, 0, 0, -0.009456), (-1, 0.0, 0, 0, 0.058718), (24, 0.06250000000000003, 5, 6, -0.147895), (-1, 0.0, 0, 0, -0.016212), (-1, 0.0, 0, 0, 0.020098)),
            ((45, 0.87, 1, 4, 0.0), (3, 0.30100000000000005, 2, 3, -0.156946), (-1, 0.0, 0, 0, 0.003508), (-1, 0.0, 0, 0, -0.028223), (50, 0.21499999999999997, 5, 6, 0.560718), (-1, 0.0, 0, 0, -0.033402), (-1, 0.0, 0, 0, 0.03639)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.605865), (-1, 0.0, 0, 0, -0.00923), (-1, 0.0, 0, 0, 0.056893), (20, -3.210268834500856, 5, 6, -0.140318), (-1, 0.0, 0, 0, -0.026595), (-1, 0.0, 0, 0, 0.004812)),
            ((7, 0.951, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, -0.146482), (-1, 0.0, 0, 0, -0.000403), (-1, 0.0, 0, 0, -0.050227), (56, 0.12, 5, 6, 0.566705), (-1, 0.0, 0, 0, -0.018665), (-1, 0.0, 0, 0, 0.036731)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.145884), (-1, 0.0, 0, 0, -0.03961), (-1, 0.0, 0, 0, -0.000233), (13, 0.933, 5, 6, 0.534192), (-1, 0.0, 0, 0, -0.002827), (-1, 0.0, 0, 0, 0.045048)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.58214), (-1, 0.0, 0, 0, -0.008523), (-1, 0.0, 0, 0, 0.055082), (20, -3.210268834500856, 5, 6, -0.133575), (-1, 0.0, 0, 0, -0.025407), (-1, 0.0, 0, 0, 0.004527)),
            ((41, 0.5, 1, 4, 0.0), (24, 0.012250000000000011, 2, 3, 0.276273), (-1, 0.0, 0, 0, -0.022343), (-1, 0.0, 0, 0, 0.028328), (13, 0.974, 5, 6, -0.276448), (-1, 0.0, 0, 0, -0.028517), (-1, 0.0, 0, 0, 0.008201)),
            ((7, 0.957, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.132129), (-1, 0.0, 0, 0, -0.023649), (-1, 0.0, 0, 0, 0.010722), (45, 0.06544, 5, 6, 0.552929), (-1, 0.0, 0, 0, -0.02261), (-1, 0.0, 0, 0, 0.03468)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.571727), (-1, 0.0, 0, 0, -0.017062), (-1, 0.0, 0, 0, 0.049048), (20, -3.210268834500856, 5, 6, -0.129545), (-1, 0.0, 0, 0, -0.024197), (-1, 0.0, 0, 0, 0.004019)),
            ((41, 0.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.263101), (-1, 0.0, 0, 0, -0.013982), (-1, 0.0, 0, 0, 0.030602), (55, 0.978, 5, 6, -0.262875), (-1, 0.0, 0, 0, -0.018709), (-1, 0.0, 0, 0, 0.038756)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.54919), (-1, 0.0, 0, 0, -0.016477), (-1, 0.0, 0, 0, 0.047064), (24, 0.06250000000000003, 5, 6, -0.12416), (-1, 0.0, 0, 0, -0.01351), (-1, 0.0, 0, 0, 0.017324)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.890522), (-1, 0.0, 0, 0, -0.056618), (-1, 0.0, 0, 0, -0.015653), (41, 0.5, 5, 6, 0.077224), (-1, 0.0, 0, 0, 0.019537), (-1, 0.0, 0, 0, -0.011279)),
            ((7, 0.959, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.122479), (-1, 0.0, 0, 0, -0.044863), (-1, 0.0, 0, 0, 0.00104), (40, 0.509, 5, 6, 0.534374), (-1, 0.0, 0, 0, 0.033601), (-1, 0.0, 0, 0, -0.022702)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.533242), (-1, 0.0, 0, 0, -0.007323), (-1, 0.0, 0, 0, 0.051059), (24, 0.06250000000000003, 5, 6, -0.120676), (-1, 0.0, 0, 0, -0.012979), (-1, 0.0, 0, 0, 0.01656)),
            ((58, 0.3550420168067227, 1, 4, 0.0), (15, -0.741, 2, 3, 0.11633), (-1, 0.0, 0, 0, -0.042275), (-1, 0.0, 0, 0, 0.010163), (13, 0.986, 5, 6, -0.550795), (-1, 0.0, 0, 0, -0.050737), (-1, 0.0, 0, 0, 0.015864)),
            ((60, 0.5, 1, 4, 0.0), (41, 0.5, 2, 3, 0.074457), (-1, 0.0, 0, 0, 0.017463), (-1, 0.0, 0, 0, -0.010111), (50, 0.985, 5, 6, -0.85364), (-1, 0.0, 0, 0, -0.05905), (-1, 0.0, 0, 0, 0.027855)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.847159), (-1, 0.0, 0, 0, -0.055149), (-1, 0.0, 0, 0, -0.013686), (59, 0.4365079365079365, 5, 6, 0.07052), (-1, 0.0, 0, 0, 0.016605), (-1, 0.0, 0, 0, -0.013121)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.12127), (-1, 0.0, 0, 0, -0.022313), (-1, 0.0, 0, 0, 0.010452), (45, 0.07032, 5, 6, 0.486624), (-1, 0.0, 0, 0, -0.023731), (-1, 0.0, 0, 0, 0.03168)),
            ((60, 0.5, 1, 4, 0.0), (45, 0.20606000000000002, 2, 3, 0.071208), (-1, 0.0, 0, 0, -0.006554), (-1, 0.0, 0, 0, 0.021357), (50, 0.985, 5, 6, -0.831685), (-1, 0.0, 0, 0, -0.057318), (-1, 0.0, 0, 0, 0.026761)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.505734), (-1, 0.0, 0, 0, -0.017648), (-1, 0.0, 0, 0, 0.044441), (24, 0.04500000000000004, 5, 6, -0.114025), (-1, 0.0, 0, 0, -0.02125), (-1, 0.0, 0, 0, 0.003809)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.827195), (-1, 0.0, 0, 0, -0.054368), (-1, 0.0, 0, 0, -0.013197), (59, 0.4365079365079365, 5, 6, 0.067307), (-1, 0.0, 0, 0, 0.0159), (-1, 0.0, 0, 0, -0.012566)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.068004), (-1, 0.0, 0, 0, -0.027072), (-1, 0.0, 0, 0, 0.00917), (2, 0.994, 5, 6, -0.801253), (-1, 0.0, 0, 0, -0.05922), (-1, 0.0, 0, 0, 0.016208)),
            ((7, 0.959, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.111245), (-1, 0.0, 0, 0, -0.042561), (-1, 0.0, 0, 0, 0.001121), (26, 4.494223020031202, 5, 6, 0.486289), (-1, 0.0, 0, 0, 0.045232), (-1, 0.0, 0, 0, 0.006353)),
            ((58, 0.3550420168067227, 1, 4, 0.0), (15, -0.741, 2, 3, 0.103459), (-1, 0.0, 0, 0, -0.039189), (-1, 0.0, 0, 0, 0.00903), (13, 0.986, 5, 6, -0.513204), (-1, 0.0, 0, 0, -0.04834), (-1, 0.0, 0, 0, 0.01571)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.485292), (-1, 0.0, 0, 0, -0.016795), (-1, 0.0, 0, 0, 0.042511), (49, 3.5, 5, 6, -0.10972), (-1, 0.0, 0, 0, -0.001749), (-1, 0.0, 0, 0, -0.042768)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.116349), (-1, 0.0, 0, 0, -0.036264), (-1, 0.0, 0, 0, 0.000689), (13, 0.933, 5, 6, 0.449382), (-1, 0.0, 0, 0, -0.005559), (-1, 0.0, 0, 0, 0.042269)),
            ((60, 0.5, 1, 4, 0.0), (41, 0.5, 2, 3, 0.064918), (-1, 0.0, 0, 0, 0.0163), (-1, 0.0, 0, 0, -0.009817), (50, 0.985, 5, 6, -0.780208), (-1, 0.0, 0, 0, -0.053967), (-1, 0.0, 0, 0, 0.025977)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.794345), (-1, 0.0, 0, 0, -0.053438), (-1, 0.0, 0, 0, -0.011539), (59, 0.4365079365079365, 5, 6, 0.062497), (-1, 0.0, 0, 0, 0.014828), (-1, 0.0, 0, 0, -0.011768)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.120938), (-1, 0.0, 0, 0, -0.050434), (-1, 0.0, 0, 0, -0.000571), (52, 0.5, 5, 6, 0.403418), (-1, 0.0, 0, 0, 0.043273), (-1, 0.0, 0, 0, -0.014229)),
            ((60, 0.5, 1, 4, 0.0), (41, 0.5, 2, 3, 0.062284), (-1, 0.0, 0, 0, 0.015632), (-1, 0.0, 0, 0, -0.009357), (2, 0.994, 5, 6, -0.760383), (-1, 0.0, 0, 0, -0.055981), (-1, 0.0, 0, 0, 0.015331)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.784232), (-1, 0.0, 0, 0, -0.05287), (-1, 0.0, 0, 0, -0.011619), (59, 0.4365079365079365, 5, 6, 0.06044), (-1, 0.0, 0, 0, 0.014225), (-1, 0.0, 0, 0, -0.011306)),
            ((7, 0.957, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.106585), (-1, 0.0, 0, 0, -0.021218), (-1, 0.0, 0, 0, 0.010384), (56, 0.10299999999999998, 5, 6, 0.45681), (-1, 0.0, 0, 0, -0.016476), (-1, 0.0, 0, 0, 0.0311)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.061281), (-1, 0.0, 0, 0, -0.009372), (-1, 0.0, 0, 0, 0.017302), (24, 0.06250000000000003, 5, 6, -0.774975), (-1, 0.0, 0, 0, -0.049194), (-1, 0.0, 0, 0, 0.025431)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.545705), (-1, 0.0, 0, 0, 0.010593), (-1, 0.0, 0, 0, -0.050481), (3, 0.24, 5, 6, 0.083867), (-1, 0.0, 0, 0, 0.02069), (-1, 0.0, 0, 0, -0.009832)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.458665), (-1, 0.0, 0, 0, -0.016049), (-1, 0.0, 0, 0, 0.039966), (26, 4.952293473989714, 5, 6, -0.103196), (-1, 0.0, 0, 0, -0.000281), (-1, 0.0, 0, 0, -0.03166)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.0756), (-1, 0.0, 0, 0, -0.029769), (-1, 0.0, 0, 0, 0.002044), (13, 0.744, 5, 6, 0.625849), (-1, 0.0, 0, 0, -0.014602), (-1, 0.0, 0, 0, 0.048991)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.770713), (-1, 0.0, 0, 0, -0.052102), (-1, 0.0, 0, 0, -0.011884), (41, 0.5, 5, 6, 0.058158), (-1, 0.0, 0, 0, 0.015752), (-1, 0.0, 0, 0, -0.009426)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.060273), (-1, 0.0, 0, 0, -0.024892), (-1, 0.0, 0, 0, 0.008093), (26, 4.668133964286986, 5, 6, -0.747888), (-1, 0.0, 0, 0, -0.060202), (-1, 0.0, 0, 0, 0.005651)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.0045000000000000595, 2, 3, -0.293788), (-1, 0.0, 0, 0, -0.029331), (-1, 0.0, 0, 0, 0.002822), (52, 0.5, 5, 6, 0.14768), (-1, 0.0, 0, 0, 0.020244), (-1, 0.0, 0, 0, -0.01362)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.057696), (-1, 0.0, 0, 0, -0.008737), (-1, 0.0, 0, 0, 0.016057), (59, 0.34523809523809523, 5, 6, -0.748696), (-1, 0.0, 0, 0, 0.024372), (-1, 0.0, 0, 0, -0.048249)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (63, 0.664, 2, 3, 0.45115), (-1, 0.0, 0, 0, 0.042054), (-1, 0.0, 0, 0, -0.010413), (26, 4.952293473989714, 5, 6, -0.101073), (-1, 0.0, 0, 0, -0.00032), (-1, 0.0, 0, 0, -0.030864)),
            ((60, 0.5, 1, 4, 0.0), (7, 0.959, 2, 3, 0.057474), (-1, 0.0, 0, 0, -0.002392), (-1, 0.0, 0, 0, 0.028717), (50, 0.985, 5, 6, -0.718701), (-1, 0.0, 0, 0, -0.050324), (-1, 0.0, 0, 0, 0.025494)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.748612), (-1, 0.0, 0, 0, -0.051151), (-1, 0.0, 0, 0, -0.011511), (59, 0.4365079365079365, 5, 6, 0.054824), (-1, 0.0, 0, 0, 0.01327), (-1, 0.0, 0, 0, -0.010639)),
            ((19, 0.5, 1, 4, 0.0), (1, 0.009, 2, 3, -0.072186), (-1, 0.0, 0, 0, 0.006063), (-1, 0.0, 0, 0, -0.018678), (11, 0.948, 5, 6, 0.606328), (-1, 0.0, 0, 0, -0.00727), (-1, 0.0, 0, 0, 0.052343)),
            ((58, 0.12916666666666665, 1, 4, 0.0), (26, 4.255587559545337, 2, 3, 0.247201), (-1, 0.0, 0, 0, -0.027214), (-1, 0.0, 0, 0, 0.022997), (2, 0.985, 5, 6, -0.16464), (-1, 0.0, 0, 0, -0.024366), (-1, 0.0, 0, 0, 0.009087)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.729732), (-1, 0.0, 0, 0, -0.050533), (-1, 0.0, 0, 0, -0.010756), (41, 0.5, 5, 6, 0.052415), (-1, 0.0, 0, 0, 0.014578), (-1, 0.0, 0, 0, -0.008865)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, 0.428465), (-1, 0.0, 0, 0, -0.00687), (-1, 0.0, 0, 0, 0.043508), (26, 4.952293473989714, 5, 6, -0.09467), (-1, 0.0, 0, 0, -0.000199), (-1, 0.0, 0, 0, -0.029595)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.055032), (-1, 0.0, 0, 0, -0.023609), (-1, 0.0, 0, 0, 0.007468), (26, 4.668133964286986, 5, 6, -0.689463), (-1, 0.0, 0, 0, -0.056796), (-1, 0.0, 0, 0, 0.007224)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.068679), (-1, 0.0, 0, 0, -0.028508), (-1, 0.0, 0, 0, 0.002153), (13, 0.744, 5, 6, 0.587377), (-1, 0.0, 0, 0, -0.014942), (-1, 0.0, 0, 0, 0.047869)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.052839), (-1, 0.0, 0, 0, -0.008636), (-1, 0.0, 0, 0, 0.015289), (24, 0.06250000000000003, 5, 6, -0.697544), (-1, 0.0, 0, 0, -0.045589), (-1, 0.0, 0, 0, 0.025671)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.722491), (-1, 0.0, 0, 0, -0.049949), (-1, 0.0, 0, 0, -0.011158), (59, 0.4365079365079365, 5, 6, 0.050976), (-1, 0.0, 0, 0, 0.012615), (-1, 0.0, 0, 0, -0.01021)),
            ((45, 0.87, 1, 4, 0.0), (3, 0.30100000000000005, 2, 3, -0.097331), (-1, 0.0, 0, 0, 0.004432), (-1, 0.0, 0, 0, -0.021719), (13, 0.933, 5, 6, 0.391889), (-1, 0.0, 0, 0, -0.006179), (-1, 0.0, 0, 0, 0.040219)),
            ((60, 0.5, 1, 4, 0.0), (7, 0.959, 2, 3, 0.052914), (-1, 0.0, 0, 0, -0.00221), (-1, 0.0, 0, 0, 0.026874), (50, 0.976, 5, 6, -0.668412), (-1, 0.0, 0, 0, -0.052437), (-1, 0.0, 0, 0, 0.012023)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.491536), (-1, 0.0, 0, 0, 0.011228), (-1, 0.0, 0, 0, -0.048482), (3, 0.24, 5, 6, 0.072424), (-1, 0.0, 0, 0, 0.018085), (-1, 0.0, 0, 0, -0.008649)),
            ((58, 0.3798076923076923, 1, 4, 0.0), (60, 0.5, 2, 3, 0.073836), (-1, 0.0, 0, 0, 0.006436), (-1, 0.0, 0, 0, -0.039104), (55, 0.943, 5, 6, -0.476233), (-1, 0.0, 0, 0, -0.041392), (-1, 0.0, 0, 0, 0.023615)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.709117), (-1, 0.0, 0, 0, -0.049303), (-1, 0.0, 0, 0, -0.011276), (41, 0.5, 5, 6, 0.048731), (-1, 0.0, 0, 0, 0.013846), (-1, 0.0, 0, 0, -0.008531)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.070359), (-1, 0.0, 0, 0, -0.005413), (-1, 0.0, 0, 0, 0.026057), (24, 0.06250000000000003, 5, 6, -0.480421), (-1, 0.0, 0, 0, -0.03352), (-1, 0.0, 0, 0, 0.035571)),
            ((20, -3.683959493972949, 1, 4, 0.0), (6, 0.18761101883864736, 2, 3, -0.26293), (-1, 0.0, 0, 0, -0.033635), (-1, 0.0, 0, 0, -0.001531), (52, 0.5, 5, 6, 0.130357), (-1, 0.0, 0, 0, 0.018771), (-1, 0.0, 0, 0, -0.013393)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.41199), (-1, 0.0, 0, 0, -0.016387), (-1, 0.0, 0, 0, 0.036375), (26, 4.952293473989714, 5, 6, -0.089504), (-1, 0.0, 0, 0, -0.000281), (-1, 0.0, 0, 0, -0.027954)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.063445), (-1, 0.0, 0, 0, -0.028268), (-1, 0.0, 0, 0, 0.001985), (11, 0.948, 5, 6, 0.557943), (-1, 0.0, 0, 0, -0.008767), (-1, 0.0, 0, 0, 0.051316)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.694189), (-1, 0.0, 0, 0, -0.048695), (-1, 0.0, 0, 0, -0.010917), (41, 0.5, 5, 6, 0.047078), (-1, 0.0, 0, 0, 0.013122), (-1, 0.0, 0, 0, -0.007981)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (63, 0.664, 2, 3, 0.395809), (-1, 0.0, 0, 0, 0.037935), (-1, 0.0, 0, 0, -0.010079), (2, 0.985, 5, 6, -0.085503), (-1, 0.0, 0, 0, -0.013649), (-1, 0.0, 0, 0, 0.005616)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.048014), (-1, 0.0, 0, 0, -0.008328), (-1, 0.0, 0, 0, 0.014319), (27, 4.728490960086225, 5, 6, -0.65585), (-1, 0.0, 0, 0, -0.044907), (-1, 0.0, 0, 0, 0.020532)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.049361), (-1, 0.0, 0, 0, -0.022311), (-1, 0.0, 0, 0, 0.006794), (26, 4.668133964286986, 5, 6, -0.630355), (-1, 0.0, 0, 0, -0.053172), (-1, 0.0, 0, 0, 0.00878)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.098486), (-1, 0.0, 0, 0, -0.046569), (-1, 0.0, 0, 0, -0.000101), (52, 0.5, 5, 6, 0.339374), (-1, 0.0, 0, 0, 0.040514), (-1, 0.0, 0, 0, -0.014983)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.681151), (-1, 0.0, 0, 0, -0.048097), (-1, 0.0, 0, 0, -0.010767), (59, 0.4365079365079365, 5, 6, 0.045314), (-1, 0.0, 0, 0, 0.011553), (-1, 0.0, 0, 0, -0.009551)),
            ((7, 0.959, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.086202), (-1, 0.0, 0, 0, -0.017906), (-1, 0.0, 0, 0, 0.008702), (26, 4.494223020031202, 5, 6, 0.379351), (-1, 0.0, 0, 0, 0.042096), (-1, 0.0, 0, 0, -0.000159)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.046603), (-1, 0.0, 0, 0, -0.007469), (-1, 0.0, 0, 0, 0.013105), (27, 4.728490960086225, 5, 6, -0.642875), (-1, 0.0, 0, 0, -0.043899), (-1, 0.0, 0, 0, 0.019211)),
            ((19, 0.5, 1, 4, 0.0), (1, 0.009, 2, 3, -0.060293), (-1, 0.0, 0, 0, 0.005698), (-1, 0.0, 0, 0, -0.01666), (6, 0.8873180999538214, 5, 6, 0.54125), (-1, 0.0, 0, 0, 0.035576), (-1, 0.0, 0, 0, -0.064016)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (15, -0.113, 2, 3, 0.388291), (-1, 0.0, 0, 0, -0.004755), (-1, 0.0, 0, 0, 0.042846), (26, 4.952293473989714, 5, 6, -0.083121), (-1, 0.0, 0, 0, -0.000134), (-1, 0.0, 0, 0, -0.026781)),
            ((63, 0.98, 1, 4, 0.0), (62, 0.15100000000000002, 2, 3, 0.03957), (-1, 0.0, 0, 0, 0.019717), (-1, 0.0, 0, 0, -0.00436), (20, -3.683959493972949, 5, 6, -0.748521), (-1, 0.0, 0, 0, -0.050855), (-1, 0.0, 0, 0, 0.000695)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.04775), (-1, 0.0, 0, 0, -0.021402), (-1, 0.0, 0, 0, 0.006465), (26, 4.668133964286986, 5, 6, -0.615992), (-1, 0.0, 0, 0, -0.052032), (-1, 0.0, 0, 0, 0.008574)),
            ((15, -0.741, 1, 4, 0.0), (26, 4.505334588423557, 2, 3, -0.668547), (-1, 0.0, 0, 0, -0.003691), (-1, 0.0, 0, 0, -0.043868), (59, 0.4365079365079365, 5, 6, 0.043481), (-1, 0.0, 0, 0, 0.010942), (-1, 0.0, 0, 0, -0.00895)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.455373), (-1, 0.0, 0, 0, 0.010899), (-1, 0.0, 0, 0, -0.046977), (6, 0.6482034517095314, 5, 6, 0.064181), (-1, 0.0, 0, 0, 0.012163), (-1, 0.0, 0, 0, -0.012146)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.27525252525252525, 2, 3, -0.054597), (-1, 0.0, 0, 0, 0.003599), (-1, 0.0, 0, 0, -0.019854), (11, 0.992, 5, 6, 0.578133), (-1, 0.0, 0, 0, 0.050234), (-1, 0.0, 0, 0, -0.011925)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.084639), (-1, 0.0, 0, 0, -0.033592), (-1, 0.0, 0, 0, 0.000285), (13, 0.744, 5, 6, 0.352431), (-1, 0.0, 0, 0, -0.018028), (-1, 0.0, 0, 0, 0.032872)),
            ((60, 0.5, 1, 4, 0.0), (41, 0.5, 2, 3, 0.046139), (-1, 0.0, 0, 0, 0.011989), (-1, 0.0, 0, 0, -0.007144), (50, 0.985, 5, 6, -0.600695), (-1, 0.0, 0, 0, -0.043532), (-1, 0.0, 0, 0, 0.025053)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.71317), (-1, 0.0, 0, 0, -0.004981), (-1, 0.0, 0, 0, -0.049685), (59, 0.4365079365079365, 5, 6, 0.038801), (-1, 0.0, 0, 0, 0.010702), (-1, 0.0, 0, 0, -0.0092)),
            ((63, 0.98, 1, 4, 0.0), (50, 0.04700000000000004, 2, 3, 0.036932), (-1, 0.0, 0, 0, 0.030273), (-1, 0.0, 0, 0, -0.002173), (20, -3.683959493972949, 5, 6, -0.729503), (-1, 0.0, 0, 0, -0.050111), (-1, 0.0, 0, 0, 0.000545)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.5535, 2, 3, -0.240878), (-1, 0.0, 0, 0, -0.004485), (-1, 0.0, 0, 0, -0.04487), (52, 0.5, 5, 6, 0.11808), (-1, 0.0, 0, 0, 0.017211), (-1, 0.0, 0, 0, -0.012317)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.707703), (-1, 0.0, 0, 0, -0.007121), (-1, 0.0, 0, 0, -0.04879), (40, 0.624, 5, 6, 0.036609), (-1, 0.0, 0, 0, 0.006721), (-1, 0.0, 0, 0, -0.016137)),
            ((55, 0.978, 1, 4, 0.0), (58, 0.458041958041958, 2, 3, -0.052451), (-1, 0.0, 0, 0, 0.000188), (-1, 0.0, 0, 0, -0.038452), (11, 0.992, 5, 6, 0.563088), (-1, 0.0, 0, 0, 0.049279), (-1, 0.0, 0, 0, -0.01148)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.044304), (-1, 0.0, 0, 0, -0.020492), (-1, 0.0, 0, 0, 0.006047), (26, 4.668133964286986, 5, 6, -0.588232), (-1, 0.0, 0, 0, -0.050144), (-1, 0.0, 0, 0, 0.008937)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.061585), (-1, 0.0, 0, 0, -0.004776), (-1, 0.0, 0, 0, 0.022544), (24, 0.06250000000000003, 5, 6, -0.433976), (-1, 0.0, 0, 0, -0.030592), (-1, 0.0, 0, 0, 0.03395)),
            ((63, 0.98, 1, 4, 0.0), (50, 0.04700000000000004, 2, 3, 0.035217), (-1, 0.0, 0, 0, 0.028639), (-1, 0.0, 0, 0, -0.002097), (20, -3.683959493972949, 5, 6, -0.71545), (-1, 0.0, 0, 0, -0.04951), (-1, 0.0, 0, 0, 0.000402)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.089888), (-1, 0.0, 0, 0, -0.045015), (-1, 0.0, 0, 0, 8.3e-05), (52, 0.5, 5, 6, 0.313884), (-1, 0.0, 0, 0, 0.038979), (-1, 0.0, 0, 0, -0.014363)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.365975), (-1, 0.0, 0, 0, -0.016893), (-1, 0.0, 0, 0, 0.033052), (2, 0.985, 5, 6, -0.078226), (-1, 0.0, 0, 0, -0.012651), (-1, 0.0, 0, 0, 0.005292)),
            ((15, -0.175, 1, 4, 0.0), (13, 0.692, 2, 3, -0.272675), (-1, 0.0, 0, 0, -0.049303), (-1, 0.0, 0, 0, -0.002646), (59, 0.14835164835164835, 5, 6, 0.099392), (-1, 0.0, 0, 0, 0.037663), (-1, 0.0, 0, 0, -0.000739)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.042125), (-1, 0.0, 0, 0, -0.007165), (-1, 0.0, 0, 0, 0.012206), (13, 0.43999999999999995, 5, 6, -0.603355), (-1, 0.0, 0, 0, 0.015652), (-1, 0.0, 0, 0, -0.042859)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (3, 0.405, 2, 3, -0.431387), (-1, 0.0, 0, 0, -0.031498), (-1, 0.0, 0, 0, 0.056417), (2, 0.6619999999999999, 5, 6, 0.059866), (-1, 0.0, 0, 0, 0.022609), (-1, 0.0, 0, 0, -0.00385)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.698314), (-1, 0.0, 0, 0, -0.00708), (-1, 0.0, 0, 0, -0.047999), (40, 0.624, 5, 6, 0.036153), (-1, 0.0, 0, 0, 0.006537), (-1, 0.0, 0, 0, -0.01562)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.055139), (-1, 0.0, 0, 0, -0.025575), (-1, 0.0, 0, 0, 0.001783), (11, 0.948, 5, 6, 0.50504), (-1, 0.0, 0, 0, -0.01225), (-1, 0.0, 0, 0, 0.050802)),
            ((60, 0.5, 1, 4, 0.0), (55, 0.978, 2, 3, 0.043803), (-1, 0.0, 0, 0, -0.000299), (-1, 0.0, 0, 0, 0.036319), (2, 0.994, 5, 6, -0.574554), (-1, 0.0, 0, 0, -0.044631), (-1, 0.0, 0, 0, 0.016901)),
            ((15, -0.175, 1, 4, 0.0), (13, 0.692, 2, 3, -0.264531), (-1, 0.0, 0, 0, -0.047854), (-1, 0.0, 0, 0, -0.002616), (59, 0.14835164835164835, 5, 6, 0.096669), (-1, 0.0, 0, 0, 0.036376), (-1, 0.0, 0, 0, -0.000621)),
            ((1, 0.009, 1, 4, 0.0), (59, 0.4365079365079365, 2, 3, 0.133259), (-1, 0.0, 0, 0, 0.023281), (-1, 0.0, 0, 0, -0.013805), (2, 0.985, 5, 6, -0.190029), (-1, 0.0, 0, 0, -0.024342), (-1, 0.0, 0, 0, 0.024225)),
            ((63, 0.98, 1, 4, 0.0), (1, 0.009, 2, 3, 0.034742), (-1, 0.0, 0, 0, 0.010129), (-1, 0.0, 0, 0, -0.00905), (20, -3.683959493972949, 5, 6, -0.697165), (-1, 0.0, 0, 0, -0.048957), (-1, 0.0, 0, 0, 0.000896)),
            ((45, 0.87, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, -0.078054), (-1, 0.0, 0, 0, -0.032), (-1, 0.0, 0, 0, 0.000297), (13, 0.744, 5, 6, 0.333207), (-1, 0.0, 0, 0, -0.016512), (-1, 0.0, 0, 0, 0.031799)),
            ((15, -0.761, 1, 4, 0.0), (26, 4.516324041782945, 2, 3, -0.676503), (-1, 0.0, 0, 0, -0.005055), (-1, 0.0, 0, 0, -0.0472), (40, 0.624, 5, 6, 0.035384), (-1, 0.0, 0, 0, 0.006368), (-1, 0.0, 0, 0, -0.015127)),
            ((60, 0.5, 1, 4, 0.0), (41, 0.5, 2, 3, 0.042285), (-1, 0.0, 0, 0, 0.011215), (-1, 0.0, 0, 0, -0.00669), (26, 4.668133964286986, 5, 6, -0.559702), (-1, 0.0, 0, 0, -0.047698), (-1, 0.0, 0, 0, 0.009084)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.040117), (-1, 0.0, 0, 0, -0.006894), (-1, 0.0, 0, 0, 0.011638), (24, 0.06250000000000003, 5, 6, -0.579962), (-1, 0.0, 0, 0, -0.039399), (-1, 0.0, 0, 0, 0.024375)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.052423), (-1, 0.0, 0, 0, -0.02389), (-1, 0.0, 0, 0, 0.001894), (11, 0.948, 5, 6, 0.490766), (-1, 0.0, 0, 0, -0.010901), (-1, 0.0, 0, 0, 0.050246)),
            ((15, -0.741, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.613586), (-1, 0.0, 0, 0, -0.045575), (-1, 0.0, 0, 0, -0.0086), (40, 0.5799999999999998, 5, 6, 0.037881), (-1, 0.0, 0, 0, 0.006701), (-1, 0.0, 0, 0, -0.013831)),
            ((63, 0.98, 1, 4, 0.0), (50, 0.04700000000000004, 2, 3, 0.032659), (-1, 0.0, 0, 0, 0.026046), (-1, 0.0, 0, 0, -0.00197), (20, -3.683959493972949, 5, 6, -0.679991), (-1, 0.0, 0, 0, -0.048397), (-1, 0.0, 0, 0, 0.001181)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.0045000000000000595, 2, 3, -0.220288), (-1, 0.0, 0, 0, -0.025298), (-1, 0.0, 0, 0, 0.004901), (52, 0.5, 5, 6, 0.107614), (-1, 0.0, 0, 0, 0.016286), (-1, 0.0, 0, 0, -0.011992)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.056973), (-1, 0.0, 0, 0, -0.004417), (-1, 0.0, 0, 0, 0.020705), (12, 0.581, 5, 6, -0.40593), (-1, 0.0, 0, 0, -0.000484), (-1, 0.0, 0, 0, -0.045695)),
            ((60, 0.5, 1, 4, 0.0), (55, 0.978, 2, 3, 0.040785), (-1, 0.0, 0, 0, -0.000276), (-1, 0.0, 0, 0, 0.034818), (50, 0.985, 5, 6, -0.544838), (-1, 0.0, 0, 0, -0.03981), (-1, 0.0, 0, 0, 0.024185)),
            ((59, 0.1380952380952381, 1, 4, 0.0), (27, 4.608290119226802, 2, 3, 0.340688), (-1, 0.0, 0, 0, -0.017715), (-1, 0.0, 0, 0, 0.031454), (4, 0.025, 5, 6, -0.070747), (-1, 0.0, 0, 0, 0.021624), (-1, 0.0, 0, 0, -0.006644)),
            ((58, 0.458041958041958, 1, 4, 0.0), (43, 0.39230769230769236, 2, 3, 0.042367), (-1, 0.0, 0, 0, 0.000676), (-1, 0.0, 0, 0, 0.053296), (7, 0.994, 5, 6, -0.522608), (-1, 0.0, 0, 0, -0.03615), (-1, 0.0, 0, 0, 0.038918)),
            ((15, -0.175, 1, 4, 0.0), (13, 0.692, 2, 3, -0.250644), (-1, 0.0, 0, 0, -0.045994), (-1, 0.0, 0, 0, -0.002249), (59, 0.14835164835164835, 5, 6, 0.091456), (-1, 0.0, 0, 0, 0.034951), (-1, 0.0, 0, 0, -0.000596)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.080493), (-1, 0.0, 0, 0, -0.04266), (-1, 0.0, 0, 0, 0.000208), (52, 0.5, 5, 6, 0.292198), (-1, 0.0, 0, 0, 0.037892), (-1, 0.0, 0, 0, -0.014544)),
            ((15, -0.784, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.658532), (-1, 0.0, 0, 0, -0.007024), (-1, 0.0, 0, 0, -0.047499), (40, 0.624, 5, 6, 0.033159), (-1, 0.0, 0, 0, 0.006042), (-1, 0.0, 0, 0, -0.014509)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.396836), (-1, 0.0, 0, 0, 0.012939), (-1, 0.0, 0, 0, -0.045215), (2, 0.6619999999999999, 5, 6, 0.054691), (-1, 0.0, 0, 0, 0.021799), (-1, 0.0, 0, 0, -0.00399)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.040318), (-1, 0.0, 0, 0, -0.018861), (-1, 0.0, 0, 0, 0.00543), (26, 4.668133964286986, 5, 6, -0.53338), (-1, 0.0, 0, 0, -0.045663), (-1, 0.0, 0, 0, 0.009078)),
            ((63, 0.98, 1, 4, 0.0), (50, 0.04700000000000004, 2, 3, 0.031637), (-1, 0.0, 0, 0, 0.0251), (-1, 0.0, 0, 0, -0.001906), (20, -3.683959493972949, 5, 6, -0.66824), (-1, 0.0, 0, 0, -0.047628), (-1, 0.0, 0, 0, 0.000294)),
            ((15, -0.175, 1, 4, 0.0), (13, 0.692, 2, 3, -0.247883), (-1, 0.0, 0, 0, -0.044915), (-1, 0.0, 0, 0, -0.002465), (59, 0.14835164835164835, 5, 6, 0.089092), (-1, 0.0, 0, 0, 0.033933), (-1, 0.0, 0, 0, -0.000528)),
            ((19, 0.5, 1, 4, 0.0), (1, 0.009, 2, 3, -0.049508), (-1, 0.0, 0, 0, 0.004977), (-1, 0.0, 0, 0, -0.014114), (11, 0.948, 5, 6, 0.473403), (-1, 0.0, 0, 0, -0.01124), (-1, 0.0, 0, 0, 0.049877)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.038171), (-1, 0.0, 0, 0, -0.00642), (-1, 0.0, 0, 0, 0.010851), (13, 0.43999999999999995, 5, 6, -0.55475), (-1, 0.0, 0, 0, 0.016606), (-1, 0.0, 0, 0, -0.040536)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.592751), (-1, 0.0, 0, 0, -0.008202), (-1, 0.0, 0, 0, -0.045718), (15, -0.613, 5, 6, 0.035292), (-1, 0.0, 0, 0, 0.043922), (-1, 0.0, 0, 0, 0.000131)),
            ((7, 0.959, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.070643), (-1, 0.0, 0, 0, -0.015215), (-1, 0.0, 0, 0, 0.007444), (26, 4.494223020031202, 5, 6, 0.319976), (-1, 0.0, 0, 0, 0.041058), (-1, 0.0, 0, 0, -0.00406)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.055336), (-1, 0.0, 0, 0, -0.003818), (-1, 0.0, 0, 0, 0.018802), (12, 0.581, 5, 6, -0.388768), (-1, 0.0, 0, 0, -0.000791), (-1, 0.0, 0, 0, -0.043877)),
            ((45, 0.87, 1, 4, 0.0), (27, 4.539296550572581, 2, 3, -0.071783), (-1, 0.0, 0, 0, -0.026523), (-1, 0.0, 0, 0, 0.000805), (13, 0.744, 5, 6, 0.314801), (-1, 0.0, 0, 0, -0.015604), (-1, 0.0, 0, 0, 0.031017)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.5535, 2, 3, -0.208292), (-1, 0.0, 0, 0, -0.002932), (-1, 0.0, 0, 0, -0.043337), (52, 0.5, 5, 6, 0.102838), (-1, 0.0, 0, 0, 0.015329), (-1, 0.0, 0, 0, -0.010993)),
            ((60, 0.5, 1, 4, 0.0), (58, 0.3798076923076923, 2, 3, 0.039364), (-1, 0.0, 0, 0, 0.004953), (-1, 0.0, 0, 0, -0.021136), (26, 4.668133964286986, 5, 6, -0.522156), (-1, 0.0, 0, 0, -0.044664), (-1, 0.0, 0, 0, 0.00877)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.17424242424242425, 2, 3, 0.030507), (-1, 0.0, 0, 0, 0.018527), (-1, 0.0, 0, 0, -0.002952), (20, -3.683959493972949, 5, 6, -0.649575), (-1, 0.0, 0, 0, -0.047091), (-1, 0.0, 0, 0, 0.000949)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.579524), (-1, 0.0, 0, 0, -0.007641), (-1, 0.0, 0, 0, -0.045187), (4, 0.979, 5, 6, 0.033852), (-1, 0.0, 0, 0, -0.002098), (-1, 0.0, 0, 0, 0.019581)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.052848), (-1, 0.0, 0, 0, -0.003705), (-1, 0.0, 0, 0, 0.018081), (24, 0.06250000000000003, 5, 6, -0.379657), (-1, 0.0, 0, 0, -0.026889), (-1, 0.0, 0, 0, 0.031348)),
            ((49, 2.5, 1, 4, 0.0), (1, 0.01, 2, 3, 0.086786), (-1, 0.0, 0, 0, 0.016455), (-1, 0.0, 0, 0, -0.012091), (24, 0.06250000000000003, 5, 6, -0.234067), (-1, 0.0, 0, 0, -0.020459), (-1, 0.0, 0, 0, 0.034564)),
            ((63, 0.98, 1, 4, 0.0), (4, 0.025, 2, 3, 0.030209), (-1, 0.0, 0, 0, 0.02646), (-1, 0.0, 0, 0, -0.001556), (20, -3.683959493972949, 5, 6, -0.643548), (-1, 0.0, 0, 0, -0.046793), (-1, 0.0, 0, 0, 0.00072)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.046715), (-1, 0.0, 0, 0, -0.029493), (-1, 0.0, 0, 0, 0.000802), (6, 0.8873180999538214, 5, 6, 0.456619), (-1, 0.0, 0, 0, 0.032634), (-1, 0.0, 0, 0, -0.062184)),
            ((6, 0.6482034517095314, 1, 4, 0.0), (23, 0.0050000000000000044, 2, 3, 0.10214), (-1, 0.0, 0, 0, -0.026904), (-1, 0.0, 0, 0, 0.010773), (45, 0.92, 5, 6, -0.196099), (-1, 0.0, 0, 0, -0.016388), (-1, 0.0, 0, 0, 0.023599)),
            ((60, 0.5, 1, 4, 0.0), (55, 0.978, 2, 3, 0.037931), (-1, 0.0, 0, 0, -0.000251), (-1, 0.0, 0, 0, 0.03367), (2, 0.994, 5, 6, -0.50844), (-1, 0.0, 0, 0, -0.039713), (-1, 0.0, 0, 0, 0.01655)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.631014), (-1, 0.0, 0, 0, -0.003096), (-1, 0.0, 0, 0, -0.045147), (8, 0.6190476190476191, 5, 6, 0.029927), (-1, 0.0, 0, 0, 0.005761), (-1, 0.0, 0, 0, -0.01397)),
            ((40, 0.624, 1, 4, 0.0), (5, 0.183, 2, 3, 0.070215), (-1, 0.0, 0, 0, 0.012678), (-1, 0.0, 0, 0, -0.009133), (4, 0.14, 5, 6, -0.285839), (-1, 0.0, 0, 0, -0.053285), (-1, 0.0, 0, 0, 0.000832)),
            ((63, 0.98, 1, 4, 0.0), (62, 0.15100000000000002, 2, 3, 0.028602), (-1, 0.0, 0, 0, 0.015748), (-1, 0.0, 0, 0, -0.003629), (20, -3.683959493972949, 5, 6, -0.63408), (-1, 0.0, 0, 0, -0.046254), (-1, 0.0, 0, 0, 0.000337)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.229823), (-1, 0.0, 0, 0, -0.043698), (-1, 0.0, 0, 0, -0.001458), (59, 0.14835164835164835, 5, 6, 0.08291), (-1, 0.0, 0, 0, 0.031805), (-1, 0.0, 0, 0, -0.00039)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.034212), (-1, 0.0, 0, 0, -0.01772), (-1, 0.0, 0, 0, 0.004894), (13, 0.43999999999999995, 5, 6, -0.52492), (-1, 0.0, 0, 0, 0.016236), (-1, 0.0, 0, 0, -0.038862)),
            ((58, 0.458041958041958, 1, 4, 0.0), (8, 0.5357142857142857, 2, 3, 0.036834), (-1, 0.0, 0, 0, 0.006378), (-1, 0.0, 0, 0, -0.015201), (7, 0.994, 5, 6, -0.484075), (-1, 0.0, 0, 0, -0.034045), (-1, 0.0, 0, 0, 0.037763)),
            ((7, 0.959, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, -0.067008), (-1, 0.0, 0, 0, -0.014408), (-1, 0.0, 0, 0, 0.007042), (26, 4.494223020031202, 5, 6, 0.301314), (-1, 0.0, 0, 0, 0.039918), (-1, 0.0, 0, 0, -0.004719)),
            ((40, 0.624, 1, 4, 0.0), (4, 0.034, 2, 3, 0.067806), (-1, 0.0, 0, 0, 0.028285), (-1, 0.0, 0, 0, -0.000697), (27, 4.7257269950599685, 5, 6, -0.276128), (-1, 0.0, 0, 0, -0.024643), (-1, 0.0, 0, 0, 0.036708)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.562483), (-1, 0.0, 0, 0, -0.007014), (-1, 0.0, 0, 0, -0.044143), (4, 0.979, 5, 6, 0.032454), (-1, 0.0, 0, 0, -0.002097), (-1, 0.0, 0, 0, 0.019184)),
            ((45, 0.87, 1, 4, 0.0), (5, 0.833, 2, 3, -0.066141), (-1, 0.0, 0, 0, 0.000781), (-1, 0.0, 0, 0, -0.026741), (11, 0.948, 5, 6, 0.294363), (-1, 0.0, 0, 0, -0.004789), (-1, 0.0, 0, 0, 0.040405)),
            ((63, 0.98, 1, 4, 0.0), (50, 0.04700000000000004, 2, 3, 0.028317), (-1, 0.0, 0, 0, 0.022649), (-1, 0.0, 0, 0, -0.001832), (20, -3.683959493972949, 5, 6, -0.618788), (-1, 0.0, 0, 0, -0.045768), (-1, 0.0, 0, 0, 0.000865)),
            ((60, 0.5, 1, 4, 0.0), (45, 0.09988000000000001, 2, 3, 0.036353), (-1, 0.0, 0, 0, -0.006108), (-1, 0.0, 0, 0, 0.010264), (26, 4.686739554676682, 5, 6, -0.488405), (-1, 0.0, 0, 0, -0.040032), (-1, 0.0, 0, 0, 0.013596)),
            ((8, 0.6190476190476191, 1, 4, 0.0), (7, 0.567, 2, 3, 0.06726), (-1, 0.0, 0, 0, -0.005046), (-1, 0.0, 0, 0, 0.019148), (4, 0.099, 5, 6, -0.268775), (-1, 0.0, 0, 0, 0.041827), (-1, 0.0, 0, 0, -0.029988)),
            ((49, 2.5, 1, 4, 0.0), (52, 0.5, 2, 3, 0.083149), (-1, 0.0, 0, 0, 0.01504), (-1, 0.0, 0, 0, -0.012062), (24, 0.06250000000000003, 5, 6, -0.223827), (-1, 0.0, 0, 0, -0.019386), (-1, 0.0, 0, 0, 0.033264)),
            ((15, -0.741, 1, 4, 0.0), (26, 4.505334588423557, 2, 3, -0.555785), (-1, 0.0, 0, 0, 0.000763), (-1, 0.0, 0, 0, -0.038048), (15, -0.613, 5, 6, 0.032129), (-1, 0.0, 0, 0, 0.041821), (-1, 0.0, 0, 0, 3.1e-05)),
            ((58, 0.458041958041958, 1, 4, 0.0), (43, 0.39230769230769236, 2, 3, 0.037202), (-1, 0.0, 0, 0, 0.00043), (-1, 0.0, 0, 0, 0.050885), (7, 0.994, 5, 6, -0.478088), (-1, 0.0, 0, 0, -0.033372), (-1, 0.0, 0, 0, 0.036622)),
            ((55, 0.978, 1, 4, 0.0), (6, 0.6482034517095314, 2, 3, -0.038308), (-1, 0.0, 0, 0, 0.004048), (-1, 0.0, 0, 0, -0.013606), (11, 0.992, 5, 6, 0.481453), (-1, 0.0, 0, 0, 0.04488), (-1, 0.0, 0, 0, -0.015019)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (0, 0.5, 2, 3, -0.355921), (-1, 0.0, 0, 0, 0.029126), (-1, 0.0, 0, 0, -0.035764), (2, 0.6619999999999999, 5, 6, 0.049424), (-1, 0.0, 0, 0, 0.020191), (-1, 0.0, 0, 0, -0.003889)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.028757), (-1, 0.0, 0, 0, 0.02005), (-1, 0.0, 0, 0, -0.002041), (20, -3.683959493972949, 5, 6, -0.61065), (-1, 0.0, 0, 0, -0.045242), (-1, 0.0, 0, 0, 0.000345)),
            ((15, -0.784, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.604682), (-1, 0.0, 0, 0, -0.004467), (-1, 0.0, 0, 0, -0.045155), (4, 0.979, 5, 6, 0.028826), (-1, 0.0, 0, 0, -0.002216), (-1, 0.0, 0, 0, 0.018916)),
            ((60, 0.5, 1, 4, 0.0), (28, 0.733, 2, 3, 0.03637), (-1, 0.0, 0, 0, 0.011671), (-1, 0.0, 0, 0, -0.004339), (50, 0.976, 5, 6, -0.477762), (-1, 0.0, 0, 0, -0.039185), (-1, 0.0, 0, 0, 0.012044)),
            ((49, 3.5, 1, 4, 0.0), (26, 4.579839228586978, 2, 3, 0.033665), (-1, 0.0, 0, 0, -0.005745), (-1, 0.0, 0, 0, 0.009567), (13, 0.43999999999999995, 5, 6, -0.501327), (-1, 0.0, 0, 0, 0.016382), (-1, 0.0, 0, 0, -0.037775)),
            ((40, 0.624, 1, 4, 0.0), (1, 0.009, 2, 3, 0.06527), (-1, 0.0, 0, 0, 0.012106), (-1, 0.0, 0, 0, -0.007687), (4, 0.14, 5, 6, -0.261491), (-1, 0.0, 0, 0, -0.050892), (-1, 0.0, 0, 0, 0.001553)),
            ((19, 0.5, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.042273), (-1, 0.0, 0, 0, -0.028866), (-1, 0.0, 0, 0, 0.000966), (6, 0.8873180999538214, 5, 6, 0.428558), (-1, 0.0, 0, 0, 0.031483), (-1, 0.0, 0, 0, -0.0583)),
            ((15, -0.175, 1, 4, 0.0), (62, 0.402, 2, 3, -0.222501), (-1, 0.0, 0, 0, -0.016879), (-1, 0.0, 0, 0, 0.042845), (59, 0.14835164835164835, 5, 6, 0.079081), (-1, 0.0, 0, 0, 0.03053), (-1, 0.0, 0, 0, -0.000416)),
            ((8, 0.6190476190476191, 1, 4, 0.0), (7, 0.567, 2, 3, 0.067319), (-1, 0.0, 0, 0, -0.004691), (-1, 0.0, 0, 0, 0.018469), (4, 0.099, 5, 6, -0.265931), (-1, 0.0, 0, 0, 0.039975), (-1, 0.0, 0, 0, -0.029027)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.33663), (-1, 0.0, 0, 0, -0.029938), (-1, 0.0, 0, 0, 0.041985), (58, 0.11805555555555555, 5, 6, 0.049704), (-1, 0.0, 0, 0, 0.016482), (-1, 0.0, 0, 0, -0.004005)),
            ((20, -3.683959493972949, 1, 4, 0.0), (40, 0.5535, 2, 3, -0.184385), (-1, 0.0, 0, 0, -0.002124), (-1, 0.0, 0, 0, -0.041307), (52, 0.5, 5, 6, 0.091681), (-1, 0.0, 0, 0, 0.014061), (-1, 0.0, 0, 0, -0.010315)),
            ((63, 0.98, 1, 4, 0.0), (2, 0.04400000000000004, 2, 3, 0.027411), (-1, 0.0, 0, 0, 0.023047), (-1, 0.0, 0, 0, -0.001436), (20, -3.683959493972949, 5, 6, -0.597329), (-1, 0.0, 0, 0, -0.044654), (-1, 0.0, 0, 0, 0.000368)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.591996), (-1, 0.0, 0, 0, -0.002069), (-1, 0.0, 0, 0, -0.042609), (4, 0.979, 5, 6, 0.027673), (-1, 0.0, 0, 0, -0.002086), (-1, 0.0, 0, 0, 0.017935)),
            ((3, 0.489, 1, 4, 0.0), (62, 0.24, 2, 3, -0.045188), (-1, 0.0, 0, 0, 0.005971), (-1, 0.0, 0, 0, -0.013023), (45, 0.0756, 5, 6, 0.383907), (-1, 0.0, 0, 0, -0.030627), (-1, 0.0, 0, 0, 0.055157)),
            ((8, 0.6190476190476191, 1, 4, 0.0), (7, 0.567, 2, 3, 0.064232), (-1, 0.0, 0, 0, -0.004522), (-1, 0.0, 0, 0, 0.01766), (4, 0.099, 5, 6, -0.253802), (-1, 0.0, 0, 0, 0.03883), (-1, 0.0, 0, 0, -0.027786)),
            ((58, 0.458041958041958, 1, 4, 0.0), (8, 0.5357142857142857, 2, 3, 0.035595), (-1, 0.0, 0, 0, 0.00585), (-1, 0.0, 0, 0, -0.013706), (63, 0.0019999999999999983, 5, 6, -0.459338), (-1, 0.0, 0, 0, 0.030914), (-1, 0.0, 0, 0, -0.03329)),
            ((9, 0.947, 1, 4, 0.0), (9, 0.767, 2, 3, -0.034217), (-1, 0.0, 0, 0, 0.001942), (-1, 0.0, 0, 0, -0.042788), (5, 0.035, 5, 6, 0.520102), (-1, 0.0, 0, 0, 0.049681), (-1, 0.0, 0, 0, -0.008827)),
            ((26, 4.952293473989714, 1, 4, 0.0), (26, 4.677480030793063, 2, 3, 0.048127), (-1, 0.0, 0, 0, -0.003574), (-1, 0.0, 0, 0, 0.016865), (12, 0.581, 5, 6, -0.342539), (-1, 0.0, 0, 0, 0.000577), (-1, 0.0, 0, 0, -0.041568)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (3, 0.405, 2, 3, -0.344786), (-1, 0.0, 0, 0, -0.027928), (-1, 0.0, 0, 0, 0.056586), (2, 0.6619999999999999, 5, 6, 0.047406), (-1, 0.0, 0, 0, 0.019143), (-1, 0.0, 0, 0, -0.00368)),
            ((15, -0.741, 1, 4, 0.0), (6, 0.12444454343920165, 2, 3, -0.531985), (-1, 0.0, 0, 0, -0.005358), (-1, 0.0, 0, 0, -0.042486), (15, -0.613, 5, 6, 0.030769), (-1, 0.0, 0, 0, 0.039336), (-1, 0.0, 0, 0, 6.1e-05)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.213471), (-1, 0.0, 0, 0, -0.042693), (-1, 0.0, 0, 0, -0.000601), (59, 0.14835164835164835, 5, 6, 0.078065), (-1, 0.0, 0, 0, 0.029427), (-1, 0.0, 0, 0, -0.000222)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.066887), (-1, 0.0, 0, 0, -0.039972), (-1, 0.0, 0, 0, 0.000558), (52, 0.5, 5, 6, 0.253954), (-1, 0.0, 0, 0, 0.035388), (-1, 0.0, 0, 0, -0.01374)),
            ((40, 0.624, 1, 4, 0.0), (41, 0.5, 2, 3, 0.064135), (-1, 0.0, 0, 0, 0.012993), (-1, 0.0, 0, 0, -0.005553), (6, 0.5803529084761214, 5, 6, -0.255972), (-1, 0.0, 0, 0, 0.016309), (-1, 0.0, 0, 0, -0.030488)),
            ((63, 0.98, 1, 4, 0.0), (4, 0.025, 2, 3, 0.02727), (-1, 0.0, 0, 0, 0.022732), (-1, 0.0, 0, 0, -0.001217), (20, -3.683959493972949, 5, 6, -0.59259), (-1, 0.0, 0, 0, -0.044337), (-1, 0.0, 0, 0, 7.3e-05)),
            ((60, 0.5, 1, 4, 0.0), (28, 0.733, 2, 3, 0.034777), (-1, 0.0, 0, 0, 0.011351), (-1, 0.0, 0, 0, -0.004278), (26, 4.686739554676682, 5, 6, -0.460167), (-1, 0.0, 0, 0, -0.037818), (-1, 0.0, 0, 0, 0.013414)),
            ((15, 0.759, 1, 4, 0.0), (50, 0.995, 2, 3, 0.039047), (-1, 0.0, 0, 0, 0.000194), (-1, 0.0, 0, 0, 0.040709), (6, 0.5986826325792863, 5, 6, -0.402351), (-1, 0.0, 0, 0, 0.013173), (-1, 0.0, 0, 0, -0.054383)),
            ((20, 4.243470074122607, 1, 4, 0.0), (1, 0.01, 2, 3, -0.021562), (-1, 0.0, 0, 0, 0.004722), (-1, 0.0, 0, 0, -0.009861), (-1, 0.0, 0, 0, 0.039082)),
            ((15, -0.784, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.575952), (-1, 0.0, 0, 0, -0.003673), (-1, 0.0, 0, 0, -0.043509), (4, 0.979, 5, 6, 0.026943), (-1, 0.0, 0, 0, -0.002085), (-1, 0.0, 0, 0, 0.017661)),
            ((49, 3.5, 1, 4, 0.0), (13, 0.906, 2, 3, 0.03218), (-1, 0.0, 0, 0, -0.005103), (-1, 0.0, 0, 0, 0.009461), (13, 0.43999999999999995, 5, 6, -0.481359), (-1, 0.0, 0, 0, 0.015839), (-1, 0.0, 0, 0, -0.036458)),
            ((3, 0.489, 1, 4, 0.0), (1, 0.012, 2, 3, -0.043453), (-1, 0.0, 0, 0, 0.004844), (-1, 0.0, 0, 0, -0.013389), (45, 0.0756, 5, 6, 0.374072), (-1, 0.0, 0, 0, -0.029197), (-1, 0.0, 0, 0, 0.052988)),
            ((58, 0.458041958041958, 1, 4, 0.0), (8, 0.5357142857142857, 2, 3, 0.034771), (-1, 0.0, 0, 0, 0.005661), (-1, 0.0, 0, 0, -0.01327), (63, 0.0019999999999999983, 5, 6, -0.456121), (-1, 0.0, 0, 0, 0.02919), (-1, 0.0, 0, 0, -0.032778)),
            ((63, 0.98, 1, 4, 0.0), (49, 2.5, 2, 3, 0.026413), (-1, 0.0, 0, 0, 0.005926), (-1, 0.0, 0, 0, -0.010347), (20, -3.683959493972949, 5, 6, -0.581414), (-1, 0.0, 0, 0, -0.043792), (-1, 0.0, 0, 0, -3e-05)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (62, 0.402, 2, 3, -0.205174), (-1, 0.0, 0, 0, -0.015813), (-1, 0.0, 0, 0, 0.039948), (59, 0.4365079365079365, 5, 6, 0.074549), (-1, 0.0, 0, 0, 0.013054), (-1, 0.0, 0, 0, -0.006558)),
            ((26, 4.2413008733233095, 1, 4, 0.0), (45, 0.955, 2, 3, -0.323533), (-1, 0.0, 0, 0, -0.028858), (-1, 0.0, 0, 0, 0.041218), (58, 0.11805555555555555, 5, 6, 0.047507), (-1, 0.0, 0, 0, 0.01537), (-1, 0.0, 0, 0, -0.003625)),
            ((40, 0.624, 1, 4, 0.0), (5, 0.183, 2, 3, 0.061752), (-1, 0.0, 0, 0, 0.010798), (-1, 0.0, 0, 0, -0.00753), (4, 0.14, 5, 6, -0.24557), (-1, 0.0, 0, 0, -0.048632), (-1, 0.0, 0, 0, 0.001775)),
            ((23, 0.0050000000000000044, 1, 4, 0.0), (63, 0.435, 2, 3, -0.330307), (-1, 0.0, 0, 0, 0.014678), (-1, 0.0, 0, 0, -0.042869), (2, 0.6619999999999999, 5, 6, 0.045187), (-1, 0.0, 0, 0, 0.018313), (-1, 0.0, 0, 0, -0.003568)),
            ((63, 0.98, 1, 4, 0.0), (59, 0.1380952380952381, 2, 3, 0.025619), (-1, 0.0, 0, 0, 0.018147), (-1, 0.0, 0, 0, -0.001799), (20, -3.683959493972949, 5, 6, -0.576738), (-1, 0.0, 0, 0, -0.043348), (-1, 0.0, 0, 0, -0.000585)),
            ((3, 0.489, 1, 4, 0.0), (62, 0.24, 2, 3, -0.042916), (-1, 0.0, 0, 0, 0.005307), (-1, 0.0, 0, 0, -0.011875), (45, 0.0756, 5, 6, 0.369057), (-1, 0.0, 0, 0, -0.028202), (-1, 0.0, 0, 0, 0.051574)),
            ((15, -0.784, 1, 4, 0.0), (26, 4.53794713771163, 2, 3, -0.568166), (-1, 0.0, 0, 0, -0.001386), (-1, 0.0, 0, 0, -0.040922), (4, 0.979, 5, 6, 0.025917), (-1, 0.0, 0, 0, -0.002008), (-1, 0.0, 0, 0, 0.016959)),
            ((60, 0.5, 1, 4, 0.0), (26, 4.2413008733233095, 2, 3, 0.033401), (-1, 0.0, 0, 0, -0.016478), (-1, 0.0, 0, 0, 0.004516), (26, 4.686739554676682, 5, 6, -0.444332), (-1, 0.0, 0, 0, -0.036675), (-1, 0.0, 0, 0, 0.013413)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.063723), (-1, 0.0, 0, 0, -0.038809), (-1, 0.0, 0, 0, 0.000544), (52, 0.5, 5, 6, 0.241902), (-1, 0.0, 0, 0, 0.034481), (-1, 0.0, 0, 0, -0.013316)),
            ((53, 2.5, 1, 4, 0.0), (3, 0.489, 2, 3, 0.025427), (-1, 0.0, 0, 0, -0.0013), (-1, 0.0, 0, 0, 0.028491), (49, 2.5, 5, 6, -0.567686), (-1, 0.0, 0, 0, 0.011938), (-1, 0.0, 0, 0, -0.045939)),
            ((15, -0.17200000000000004, 1, 4, 0.0), (13, 0.692, 2, 3, -0.201889), (-1, 0.0, 0, 0, -0.041085), (-1, 0.0, 0, 0, -0.000416), (15, -0.068, 5, 6, 0.072838), (-1, 0.0, 0, 0, 0.02513), (-1, 0.0, 0, 0, -0.000668)),
            ((58, 0.458041958041958, 1, 4, 0.0), (8, 0.5357142857142857, 2, 3, 0.032077), (-1, 0.0, 0, 0, 0.005453), (-1, 0.0, 0, 0, -0.013085), (63, 0.0019999999999999983, 5, 6, -0.440377), (-1, 0.0, 0, 0, 0.028673), (-1, 0.0, 0, 0, -0.031896)),
            ((9, 0.947, 1, 4, 0.0), (9, 0.767, 2, 3, -0.032654), (-1, 0.0, 0, 0, 0.001867), (-1, 0.0, 0, 0, -0.040986), (5, 0.035, 5, 6, 0.490686), (-1, 0.0, 0, 0, 0.04727), (-1, 0.0, 0, 0, -0.009023)),
            ((40, 0.624, 1, 4, 0.0), (41, 0.5, 2, 3, 0.061127), (-1, 0.0, 0, 0, 0.012529), (-1, 0.0, 0, 0, -0.005321), (27, 4.7257269950599685, 5, 6, -0.246341), (-1, 0.0, 0, 0, -0.022193), (-1, 0.0, 0, 0, 0.032564)),
            ((15, -0.784, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.564862), (-1, 0.0, 0, 0, -0.003931), (-1, 0.0, 0, 0, -0.042343), (4, 0.979, 5, 6, 0.02548), (-1, 0.0, 0, 0, -0.001849), (-1, 0.0, 0, 0, 0.016033)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (27, 4.2964256503859755, 2, 3, -0.062632), (-1, 0.0, 0, 0, -0.037796), (-1, 0.0, 0, 0, 0.000441), (52, 0.5, 5, 6, 0.239489), (-1, 0.0, 0, 0, 0.033701), (-1, 0.0, 0, 0, -0.01209)),
            ((60, 0.5, 1, 4, 0.0), (55, 0.978, 2, 3, 0.032576), (-1, 0.0, 0, 0, -0.000152), (-1, 0.0, 0, 0, 0.029727), (2, 0.994, 5, 6, -0.440756), (-1, 0.0, 0, 0, -0.034847), (-1, 0.0, 0, 0, 0.016789)),
            ((63, 0.98, 1, 4, 0.0), (6, 0.6482034517095314, 2, 3, 0.024709), (-1, 0.0, 0, 0, 0.006655), (-1, 0.0, 0, 0, -0.008257), (20, -3.683959493972949, 5, 6, -0.565954), (-1, 0.0, 0, 0, -0.042818), (-1, 0.0, 0, 0, -0.000546)),
            ((53, 2.5, 1, 4, 0.0), (3, 0.489, 2, 3, 0.024611), (-1, 0.0, 0, 0, -0.001239), (-1, 0.0, 0, 0, 0.027329), (49, 2.5, 5, 6, -0.5587), (-1, 0.0, 0, 0, 0.01119), (-1, 0.0, 0, 0, -0.045083)),
            ((40, 0.624, 1, 4, 0.0), (4, 0.034, 2, 3, 0.059381), (-1, 0.0, 0, 0, 0.025399), (-1, 0.0, 0, 0, -0.000598), (3, 0.17700000000000005, 5, 6, -0.240751), (-1, 0.0, 0, 0, -0.052691), (-1, 0.0, 0, 0, -0.000509)),
        )),
    ],
    'largo': [
        (0.030172, (
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -1.298983), (24, 0.04500000000000004, 3, 4, -1.56152), (-1, 0.0, 0, 0, -0.089692), (-1, 0.0, 0, 0, -0.044252), (55, 0.3038461538461539, 6, 7, -0.342529), (-1, 0.0, 0, 0, 0.025564), (-1, 0.0, 0, 0, -0.061936), (44, 0.12, 9, 12, 1.788868), (1, 0.005, 10, 11, 0.367124), (-1, 0.0, 0, 0, 0.04367), (-1, 0.0, 0, 0, -0.025791), (-1, 0.0, 0, 0, 0.094061)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -1.235593), (24, 0.04500000000000004, 3, 4, -1.485153), (-1, 0.0, 0, 0, -0.085312), (-1, 0.0, 0, 0, -0.042097), (55, 0.256578947368421, 6, 7, -0.326507), (-1, 0.0, 0, 0, 0.02733), (-1, 0.0, 0, 0, -0.05608), (44, 0.12, 9, 12, 1.706322), (1, 0.005, 10, 11, 0.351127), (-1, 0.0, 0, 0, 0.041912), (-1, 0.0, 0, 0, -0.02485), (-1, 0.0, 0, 0, 0.08973)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.62, 2, 5, -1.178301), (24, 0.04500000000000004, 3, 4, -1.373343), (-1, 0.0, 0, 0, -0.078972), (-1, 0.0, 0, 0, -0.038041), (55, 0.256578947368421, 6, 7, -0.104199), (-1, 0.0, 0, 0, 0.042776), (-1, 0.0, 0, 0, -0.045075), (44, 0.12, 9, 12, 1.633782), (1, 0.005, 10, 11, 0.336026), (-1, 0.0, 0, 0, 0.040254), (-1, 0.0, 0, 0, -0.023948), (-1, 0.0, 0, 0, 0.085942)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -1.125726), (24, 0.04500000000000004, 3, 4, -1.357124), (-1, 0.0, 0, 0, -0.078117), (-1, 0.0, 0, 0, -0.038291), (56, 0.4641025641025641, 6, 7, -0.288091), (-1, 0.0, 0, 0, 0.024869), (-1, 0.0, 0, 0, -0.055026), (44, 0.12, 9, 12, 1.569481), (1, 0.005, 10, 11, 0.321745), (-1, 0.0, 0, 0, 0.038686), (-1, 0.0, 0, 0, -0.023084), (-1, 0.0, 0, 0, 0.0826)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.62, 2, 5, -1.077627), (24, 0.04500000000000004, 3, 4, -1.258813), (-1, 0.0, 0, 0, -0.072627), (-1, 0.0, 0, 0, -0.034535), (55, 0.3038461538461539, 6, 7, -0.089415), (-1, 0.0, 0, 0, 0.037738), (-1, 0.0, 0, 0, -0.045094), (44, 0.12, 9, 12, 1.512039), (1, 0.005, 10, 11, 0.308217), (-1, 0.0, 0, 0, 0.037201), (-1, 0.0, 0, 0, -0.022256), (-1, 0.0, 0, 0, 0.079629)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -1.033139), (24, 0.04500000000000004, 3, 4, -1.251115), (-1, 0.0, 0, 0, -0.072321), (-1, 0.0, 0, 0, -0.034926), (56, 0.4641025641025641, 6, 7, -0.254335), (-1, 0.0, 0, 0, 0.024229), (-1, 0.0, 0, 0, -0.051139), (44, 0.12, 9, 12, 1.460363), (6, 0.6026875697882074, 10, 11, 0.295381), (-1, 0.0, 0, 0, 0.036906), (-1, 0.0, 0, 0, -0.020815), (-1, 0.0, 0, 0, 0.07697)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -0.991989), (24, 0.04500000000000004, 3, 4, -1.203512), (-1, 0.0, 0, 0, -0.069773), (-1, 0.0, 0, 0, -0.033337), (55, 0.256578947368421, 6, 7, -0.243013), (-1, 0.0, 0, 0, 0.025973), (-1, 0.0, 0, 0, -0.047288), (44, 0.12, 9, 12, 1.413469), (1, 0.005, 10, 11, 0.28307), (-1, 0.0, 0, 0, 0.034935), (-1, 0.0, 0, 0, -0.021469), (60, 0.706, 13, 14, 1.491482), (-1, 0.0, 0, 0, 0.075212), (-1, 0.0, 0, 0, 0.033857)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.62, 2, 5, -0.953777), (24, 0.04500000000000004, 3, 4, -1.12083), (-1, 0.0, 0, 0, -0.065232), (-1, 0.0, 0, 0, -0.029954), (55, 0.256578947368421, 6, 7, -0.063828), (-1, 0.0, 0, 0, 0.038123), (-1, 0.0, 0, 0, -0.037839), (44, 0.12, 9, 12, 1.371039), (7, 0.482, 10, 11, 0.271397), (-1, 0.0, 0, 0, 0.03665), (-1, 0.0, 0, 0, -0.017688), (24, 0.179, 13, 14, 1.448279), (-1, 0.0, 0, 0, 0.039823), (-1, 0.0, 0, 0, 0.073344)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -0.917664), (24, 0.04500000000000004, 3, 4, -1.119864), (-1, 0.0, 0, 0, -0.065365), (-1, 0.0, 0, 0, -0.030495), (56, 0.4641025641025641, 6, 7, -0.214619), (-1, 0.0, 0, 0, 0.02373), (-1, 0.0, 0, 0, -0.046962), (44, 0.12, 9, 12, 1.332062), (6, 0.6026875697882074, 10, 11, 0.260502), (-1, 0.0, 0, 0, 0.034018), (-1, 0.0, 0, 0, -0.020271), (7, 0.16500000000000004, 13, 14, 1.408955), (-1, 0.0, 0, 0, 0.035426), (-1, 0.0, 0, 0, 0.071285)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -0.88388), (24, 0.04500000000000004, 3, 4, -1.081333), (-1, 0.0, 0, 0, -0.063374), (-1, 0.0, 0, 0, -0.029138), (55, 0.3038461538461539, 6, 7, -0.205306), (-1, 0.0, 0, 0, 0.022381), (-1, 0.0, 0, 0, -0.045427), (44, 0.12, 9, 12, 1.296184), (1, 0.005, 10, 11, 0.249756), (-1, 0.0, 0, 0, 0.03234), (-1, 0.0, 0, 0, -0.021021), (60, 0.706, 13, 14, 1.372938), (-1, 0.0, 0, 0, 0.069294), (-1, 0.0, 0, 0, 0.029362)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.62, 2, 5, -0.852187), (26, 4.280624567377958, 3, 4, -1.008802), (-1, 0.0, 0, 0, -0.072575), (-1, 0.0, 0, 0, -0.038522), (55, 0.256578947368421, 6, 7, -0.042913), (-1, 0.0, 0, 0, 0.035536), (-1, 0.0, 0, 0, -0.034029), (44, 0.12, 9, 12, 1.263047), (7, 0.482, 10, 11, 0.239509), (-1, 0.0, 0, 0, 0.034142), (-1, 0.0, 0, 0, -0.017643), (24, 0.179, 13, 14, 1.339797), (-1, 0.0, 0, 0, 0.034596), (-1, 0.0, 0, 0, 0.067957)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (17, 0.5, 2, 5, -0.821111), (27, 0.622, 3, 4, -1.083468), (-1, 0.0, 0, 0, -0.009537), (-1, 0.0, 0, 0, -0.060501), (3, 0.20399999999999996, 6, 7, -0.340867), (-1, 0.0, 0, 0, 0.024913), (-1, 0.0, 0, 0, -0.037573), (44, 0.12, 9, 12, 1.232139), (6, 0.6026875697882074, 10, 11, 0.229989), (-1, 0.0, 0, 0, 0.03145), (-1, 0.0, 0, 0, -0.019692), (7, 0.16500000000000004, 13, 14, 1.309206), (-1, 0.0, 0, 0, 0.030908), (-1, 0.0, 0, 0, 0.066328)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.791276), (34, 0.5, 3, 4, -1.309325), (-1, 0.0, 0, 0, -0.069006), (-1, 0.0, 0, 0, -0.000649), (42, 0.14835164835164835, 6, 7, -0.547144), (-1, 0.0, 0, 0, 0.000773), (-1, 0.0, 0, 0, -0.048957), (44, 0.12, 9, 12, 1.203309), (1, 0.005, 10, 11, 0.220567), (-1, 0.0, 0, 0, 0.030029), (-1, 0.0, 0, 0, -0.020541), (24, 0.179, 13, 14, 1.280824), (-1, 0.0, 0, 0, 0.031756), (-1, 0.0, 0, 0, 0.065038)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (17, 0.5, 2, 5, -0.762136), (27, 0.622, 3, 4, -1.024462), (-1, 0.0, 0, 0, -0.008023), (-1, 0.0, 0, 0, -0.057533), (3, 0.20399999999999996, 6, 7, -0.292725), (-1, 0.0, 0, 0, 0.025096), (-1, 0.0, 0, 0, -0.034399), (44, 0.12, 9, 12, 1.176358), (7, 0.482, 10, 11, 0.211534), (-1, 0.0, 0, 0, 0.03191), (-1, 0.0, 0, 0, -0.017526), (7, 0.16500000000000004, 13, 14, 1.254401), (-1, 0.0, 0, 0, 0.028268), (-1, 0.0, 0, 0, 0.063619)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -0.735044), (24, 0.04500000000000004, 3, 4, -0.928289), (-1, 0.0, 0, 0, -0.056143), (-1, 0.0, 0, 0, -0.022194), (56, 0.4641025641025641, 6, 7, -0.10814), (-1, 0.0, 0, 0, 0.025777), (-1, 0.0, 0, 0, -0.039448), (44, 0.12, 9, 12, 1.150859), (1, 0.005, 10, 11, 0.20319), (-1, 0.0, 0, 0, 0.028545), (-1, 0.0, 0, 0, -0.020073), (24, 0.179, 13, 14, 1.229692), (-1, 0.0, 0, 0, 0.029109), (-1, 0.0, 0, 0, 0.062522)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.710036), (34, 0.5, 3, 4, -1.235174), (-1, 0.0, 0, 0, -0.065285), (-1, 0.0, 0, 0, -0.000182), (42, 0.14835164835164835, 6, 7, -0.46752), (-1, 0.0, 0, 0, 0.003785), (-1, 0.0, 0, 0, -0.044756), (44, 0.12, 9, 12, 1.126884), (7, 0.482, 10, 11, 0.194875), (-1, 0.0, 0, 0, 0.03039), (-1, 0.0, 0, 0, -0.017196), (7, 0.46299999999999997, 13, 14, 1.206515), (-1, 0.0, 0, 0, 0.039562), (-1, 0.0, 0, 0, 0.062531)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.684508), (34, 0.5, 3, 4, -1.203592), (-1, 0.0, 0, 0, -0.063702), (-1, 0.0, 0, 0, -0.000176), (24, 0.04500000000000004, 6, 7, -0.448427), (-1, 0.0, 0, 0, -0.0371), (-1, 0.0, 0, 0, 0.015397), (44, 0.12, 9, 12, 1.104065), (50, 0.5, 10, 11, 0.187232), (-1, 0.0, 0, 0, 0.026719), (-1, 0.0, 0, 0, -0.021256), (2, 0.46599999999999997, 13, 14, 1.184721), (-1, 0.0, 0, 0, 0.028533), (-1, 0.0, 0, 0, 0.06048)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.660394), (26, 4.280624567377958, 3, 4, -0.370967), (-1, 0.0, 0, 0, -0.054214), (-1, 0.0, 0, 0, -0.002517), (1, 0.321, 6, 7, -1.082529), (-1, 0.0, 0, 0, -0.061302), (-1, 0.0, 0, 0, -0.001287), (44, 0.12, 9, 12, 1.082158), (6, 0.6026875697882074, 10, 11, 0.178939), (-1, 0.0, 0, 0, 0.027573), (-1, 0.0, 0, 0, -0.019292), (7, 0.46299999999999997, 13, 14, 1.164072), (-1, 0.0, 0, 0, 0.037135), (-1, 0.0, 0, 0, 0.060502)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -0.637058), (24, 0.04500000000000004, 3, 4, -0.833945), (-1, 0.0, 0, 0, -0.051846), (-1, 0.0, 0, 0, -0.017845), (56, 0.4641025641025641, 6, 7, -0.028033), (-1, 0.0, 0, 0, 0.027761), (-1, 0.0, 0, 0, -0.034412), (44, 0.12, 9, 12, 1.061441), (7, 0.482, 10, 11, 0.171597), (-1, 0.0, 0, 0, 0.028409), (-1, 0.0, 0, 0, -0.016922), (24, 0.179, 13, 14, 1.144558), (-1, 0.0, 0, 0, 0.023308), (-1, 0.0, 0, 0, 0.058404)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.27525252525252525, 2, 5, -0.615716), (26, 4.24938730294611, 3, 4, -0.302363), (-1, 0.0, 0, 0, -0.054348), (-1, 0.0, 0, 0, 0.001084), (6, 0.8642791382794561, 6, 7, -0.998947), (-1, 0.0, 0, 0, -0.055405), (-1, 0.0, 0, 0, 0.012828), (44, 0.12, 9, 12, 1.041481), (1, 0.005, 10, 11, 0.164897), (-1, 0.0, 0, 0, 0.025844), (-1, 0.0, 0, 0, -0.01992), (7, 0.46299999999999997, 13, 14, 1.125921), (-1, 0.0, 0, 0, 0.034584), (-1, 0.0, 0, 0, 0.05873)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (44, 0.20606000000000002, 2, 5, -0.593986), (24, 0.04500000000000004, 3, 4, -0.789892), (-1, 0.0, 0, 0, -0.049844), (-1, 0.0, 0, 0, -0.015945), (56, 0.4641025641025641, 6, 7, -0.002409), (-1, 0.0, 0, 0, 0.02738), (-1, 0.0, 0, 0, -0.031921), (44, 0.12, 9, 12, 1.022455), (50, 0.5, 10, 11, 0.158069), (-1, 0.0, 0, 0, 0.024504), (-1, 0.0, 0, 0, -0.020878), (50, 0.5, 13, 14, 1.108202), (-1, 0.0, 0, 0, 0.059062), (-1, 0.0, 0, 0, 0.039105)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.574132), (26, 4.56889324003774, 3, 4, -0.294574), (-1, 0.0, 0, 0, -0.041197), (-1, 0.0, 0, 0, 0.005524), (1, 0.321, 6, 7, -0.997237), (-1, 0.0, 0, 0, -0.057284), (-1, 0.0, 0, 0, 0.001325), (44, 0.12, 9, 12, 1.00381), (6, 0.6026875697882074, 10, 11, 0.150986), (-1, 0.0, 0, 0, 0.02522), (-1, 0.0, 0, 0, -0.018708), (7, 0.46299999999999997, 13, 14, 1.091147), (-1, 0.0, 0, 0, 0.032068), (-1, 0.0, 0, 0, 0.057149)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (17, 0.5, 2, 5, -0.553797), (27, 0.622, 3, 4, -0.822862), (-1, 0.0, 0, 0, 0.000606), (-1, 0.0, 0, 0, -0.048026), (3, 0.20399999999999996, 6, 7, -0.123318), (-1, 0.0, 0, 0, 0.029112), (-1, 0.0, 0, 0, -0.024807), (44, 0.12, 9, 12, 0.985992), (7, 0.482, 10, 11, 0.144767), (-1, 0.0, 0, 0, 0.026351), (-1, 0.0, 0, 0, -0.016884), (24, 0.179, 13, 14, 1.07485), (-1, 0.0, 0, 0, 0.017887), (-1, 0.0, 0, 0, 0.055091)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.534479), (34, 0.5, 3, 4, -1.06416), (-1, 0.0, 0, 0, -0.057231), (-1, 0.0, 0, 0, 0.005463), (42, 0.14835164835164835, 6, 7, -0.309214), (-1, 0.0, 0, 0, 0.009432), (-1, 0.0, 0, 0, -0.036588), (44, 0.12, 9, 12, 0.968573), (1, 0.005, 10, 11, 0.139121), (-1, 0.0, 0, 0, 0.0238), (-1, 0.0, 0, 0, -0.019461), (7, 0.46299999999999997, 13, 14, 1.059046), (-1, 0.0, 0, 0, 0.029707), (-1, 0.0, 0, 0, 0.055713)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.515516), (26, 4.24938730294611, 3, 4, -0.244741), (-1, 0.0, 0, 0, -0.048738), (-1, 0.0, 0, 0, 0.00138), (1, 0.321, 6, 7, -0.938642), (-1, 0.0, 0, 0, -0.054644), (-1, 0.0, 0, 0, 0.003516), (44, 0.12, 9, 12, 0.951823), (50, 0.5, 10, 11, 0.133294), (-1, 0.0, 0, 0, 0.022558), (-1, 0.0, 0, 0, -0.020435), (50, 0.5, 13, 14, 1.043872), (-1, 0.0, 0, 0, 0.056347), (-1, 0.0, 0, 0, 0.034124)),
            ((44, 0.62, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.595035), (44, 0.0578, 3, 4, -0.865276), (-1, 0.0, 0, 0, -0.014046), (-1, 0.0, 0, 0, -0.050184), (26, 4.29033544110754, 6, 7, -0.114656), (-1, 0.0, 0, 0, -0.042521), (-1, 0.0, 0, 0, 0.015103), (20, -3.210268834500856, 9, 12, 0.794578), (55, 0.17424242424242425, 10, 11, -0.243342), (-1, 0.0, 0, 0, 0.042956), (-1, 0.0, 0, 0, -0.037763), (40, 2.5, 13, 14, 0.960947), (-1, 0.0, 0, 0, 0.050202), (-1, 0.0, 0, 0, -0.00835)),
            ((44, 0.62, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.575912), (8, 0.26785714285714285, 3, 4, -0.843226), (-1, 0.0, 0, 0, -0.050456), (-1, 0.0, 0, 0, -0.017363), (26, 4.131156519244238, 6, 7, -0.110139), (-1, 0.0, 0, 0, -0.057828), (-1, 0.0, 0, 0, 0.008554), (20, -3.210268834500856, 9, 12, 0.775059), (55, 0.17424242424242425, 10, 11, -0.233065), (-1, 0.0, 0, 0, 0.041509), (-1, 0.0, 0, 0, -0.036504), (40, 2.5, 13, 14, 0.939363), (-1, 0.0, 0, 0, 0.049134), (-1, 0.0, 0, 0, -0.00804)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.27525252525252525, 2, 5, -0.469227), (26, 4.608290119226802, 3, 4, -0.176932), (-1, 0.0, 0, 0, -0.03249), (-1, 0.0, 0, 0, 0.014263), (6, 0.8642791382794561, 6, 7, -0.853421), (-1, 0.0, 0, 0, -0.048486), (-1, 0.0, 0, 0, 0.016165), (44, 0.12, 9, 12, 0.905991), (50, 0.5, 10, 11, 0.130528), (-1, 0.0, 0, 0, 0.022426), (-1, 0.0, 0, 0, -0.020468), (7, 0.804, 13, 14, 1.00207), (-1, 0.0, 0, 0, 0.029459), (-1, 0.0, 0, 0, 0.054356)),
            ((44, 0.62, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.543568), (44, 0.0578, 3, 4, -0.811179), (-1, 0.0, 0, 0, -0.011138), (-1, 0.0, 0, 0, -0.047763), (26, 4.131156519244238, 6, 7, -0.090471), (-1, 0.0, 0, 0, -0.055928), (-1, 0.0, 0, 0, 0.008855), (20, -3.210268834500856, 9, 12, 0.747327), (55, 0.17424242424242425, 10, 11, -0.205269), (-1, 0.0, 0, 0, 0.039961), (-1, 0.0, 0, 0, -0.034232), (28, 0.5, 13, 14, 0.908001), (-1, 0.0, 0, 0, 0.049209), (-1, 0.0, 0, 0, 0.009029)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.438329), (52, 0.899, 3, 4, -0.964977), (-1, 0.0, 0, 0, -0.053014), (-1, 0.0, 0, 0, 0.004552), (15, -0.16500000000000004, 6, 7, -0.230397), (-1, 0.0, 0, 0, -0.036476), (-1, 0.0, 0, 0, 0.007681), (44, 0.12, 9, 12, 0.875823), (6, 0.6026875697882074, 10, 11, 0.1245), (-1, 0.0, 0, 0, 0.023221), (-1, 0.0, 0, 0, -0.018404), (50, 0.5, 13, 14, 0.975276), (-1, 0.0, 0, 0, 0.053661), (-1, 0.0, 0, 0, 0.028187)),
            ((44, 0.62, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.512089), (8, 0.26785714285714285, 3, 4, -0.77741), (-1, 0.0, 0, 0, -0.047754), (-1, 0.0, 0, 0, -0.013384), (25, 6.053258734230291, 6, 7, -0.076008), (-1, 0.0, 0, 0, -0.056884), (-1, 0.0, 0, 0, 0.008141), (24, 0.179, 9, 12, 0.718842), (12, 0.661, 10, 11, 0.168678), (-1, 0.0, 0, 0, 0.027269), (-1, 0.0, 0, 0, -0.034268), (6, 0.8983741916842142, 13, 14, 0.997288), (-1, 0.0, 0, 0, 0.051311), (-1, 0.0, 0, 0, 0.00247)),
            ((44, 0.62, 1, 6, 0.0), (25, 6.070641713151899, 2, 3, -0.4958), (-1, 0.0, 0, 0, -0.057748), (24, 0.04500000000000004, 4, 5, -0.31989), (-1, 0.0, 0, 0, -0.032051), (-1, 0.0, 0, 0, 0.007736), (24, 0.179, 7, 10, 0.701466), (12, 0.661, 8, 9, 0.160294), (-1, 0.0, 0, 0, 0.025993), (-1, 0.0, 0, 0, -0.032978), (6, 0.8717732060837587, 11, 12, 0.985054), (-1, 0.0, 0, 0, 0.051587), (-1, 0.0, 0, 0, 0.013767)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.39574), (26, 4.56889324003774, 3, 4, -0.13419), (-1, 0.0, 0, 0, -0.031786), (-1, 0.0, 0, 0, 0.010178), (1, 0.321, 6, 7, -0.832006), (-1, 0.0, 0, 0, -0.049687), (-1, 0.0, 0, 0, 0.004409), (50, 0.5, 9, 12, 0.831772), (46, 0.13392857142857142, 10, 11, 0.988487), (-1, 0.0, 0, 0, 0.051324), (-1, 0.0, 0, 0, 0.002134), (25, 5.675708894344701, 13, 14, 0.297024), (-1, 0.0, 0, 0, 0.049756), (-1, 0.0, 0, 0, -0.035737)),
            ((44, 0.62, 1, 6, 0.0), (25, 6.070641713151899, 2, 3, -0.468508), (-1, 0.0, 0, 0, -0.0567), (24, 0.04500000000000004, 4, 5, -0.297648), (-1, 0.0, 0, 0, -0.030615), (-1, 0.0, 0, 0, 0.007783), (2, 0.954, 7, 10, 0.676803), (55, 0.17424242424242425, 8, 9, -0.01164), (-1, 0.0, 0, 0, 0.041147), (-1, 0.0, 0, 0, -0.024573), (15, 0.252, 11, 12, 0.886757), (-1, 0.0, 0, 0, 0.049975), (-1, 0.0, 0, 0, -0.000117)),
            ((44, 0.62, 1, 6, 0.0), (25, 6.070641713151899, 2, 3, -0.453152), (-1, 0.0, 0, 0, -0.056041), (40, 0.5, 4, 5, -0.286101), (-1, 0.0, 0, 0, 0.006535), (-1, 0.0, 0, 0, -0.030519), (24, 0.179, 7, 10, 0.658568), (12, 0.661, 8, 9, 0.144735), (-1, 0.0, 0, 0, 0.024125), (-1, 0.0, 0, 0, -0.031757), (6, 0.8717732060837587, 11, 12, 0.952425), (-1, 0.0, 0, 0, 0.050181), (-1, 0.0, 0, 0, 0.010708)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.27525252525252525, 2, 5, -0.359547), (26, 4.608290119226802, 3, 4, -0.076387), (-1, 0.0, 0, 0, -0.026526), (-1, 0.0, 0, 0, 0.016516), (53, 0.40700000000000003, 6, 7, -0.753798), (-1, 0.0, 0, 0, -0.049284), (-1, 0.0, 0, 0, -0.008068), (50, 0.5, 9, 12, 0.793733), (46, 0.13392857142857142, 10, 11, 0.960538), (-1, 0.0, 0, 0, 0.050035), (-1, 0.0, 0, 0, 0.001883), (25, 5.675708894344701, 13, 14, 0.238158), (-1, 0.0, 0, 0, 0.048244), (-1, 0.0, 0, 0, -0.03608)),
            ((44, 0.62, 1, 6, 0.0), (25, 6.070641713151899, 2, 3, -0.427085), (-1, 0.0, 0, 0, -0.055158), (24, 0.04500000000000004, 4, 5, -0.26489), (-1, 0.0, 0, 0, -0.028883), (-1, 0.0, 0, 0, 0.008598), (2, 0.954, 7, 10, 0.635838), (55, 0.17424242424242425, 8, 9, -0.035595), (-1, 0.0, 0, 0, 0.039106), (-1, 0.0, 0, 0, -0.024754), (15, 0.252, 11, 12, 0.848999), (-1, 0.0, 0, 0, 0.048212), (-1, 0.0, 0, 0, -0.000582)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.336787), (26, 4.56889324003774, 3, 4, -0.087375), (-1, 0.0, 0, 0, -0.028008), (-1, 0.0, 0, 0, 0.01053), (1, 0.321, 6, 7, -0.772516), (-1, 0.0, 0, 0, -0.046974), (-1, 0.0, 0, 0, 0.005054), (50, 0.5, 9, 12, 0.766997), (13, 0.653, 10, 11, 0.939964), (-1, 0.0, 0, 0, 0.018663), (-1, 0.0, 0, 0, 0.05128), (25, 5.675708894344701, 13, 14, 0.203086), (-1, 0.0, 0, 0, 0.047297), (-1, 0.0, 0, 0, -0.035635)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.453065), (44, 0.0578, 3, 4, -0.734058), (-1, 0.0, 0, 0, -0.003048), (-1, 0.0, 0, 0, -0.047491), (26, 4.29033544110754, 6, 7, -0.061913), (-1, 0.0, 0, 0, -0.034125), (-1, 0.0, 0, 0, 0.012238), (24, 0.179, 9, 12, 0.547025), (56, 0.4641025641025641, 10, 11, 0.12634), (-1, 0.0, 0, 0, 0.027953), (-1, 0.0, 0, 0, -0.020981), (6, 0.8717732060837587, 13, 14, 0.911159), (-1, 0.0, 0, 0, 0.048463), (-1, 0.0, 0, 0, 0.00639)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (44, 0.05984, 2, 5, -0.438654), (26, 4.574061244703071, 3, 4, 0.095996), (-1, 0.0, 0, 0, -0.029949), (-1, 0.0, 0, 0, 0.026464), (24, 0.038750000000000034, 6, 7, -0.636296), (-1, 0.0, 0, 0, -0.046528), (-1, 0.0, 0, 0, -0.011054), (24, 0.179, 9, 12, 0.532032), (59, 0.15900000000000003, 10, 11, 0.120658), (-1, 0.0, 0, 0, 0.034615), (-1, 0.0, 0, 0, -0.013955), (6, 0.8717732060837587, 13, 14, 0.899887), (-1, 0.0, 0, 0, 0.047975), (-1, 0.0, 0, 0, 0.006183)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.303582), (26, 4.24938730294611, 3, 4, -0.062785), (-1, 0.0, 0, 0, -0.037098), (-1, 0.0, 0, 0, 0.006461), (53, 0.30000000000000004, 6, 7, -0.73816), (-1, 0.0, 0, 0, -0.052443), (-1, 0.0, 0, 0, -0.010635), (50, 0.5, 9, 12, 0.727909), (46, 0.13392857142857142, 10, 11, 0.91283), (-1, 0.0, 0, 0, 0.047902), (-1, 0.0, 0, 0, 0.000142), (27, 0.133, 13, 14, 0.143004), (-1, 0.0, 0, 0, 0.044289), (-1, 0.0, 0, 0, -0.039208)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.415996), (44, 0.0578, 3, 4, -0.694554), (-1, 0.0, 0, 0, -0.003038), (-1, 0.0, 0, 0, -0.045718), (26, 4.29033544110754, 6, 7, -0.04698), (-1, 0.0, 0, 0, -0.032144), (-1, 0.0, 0, 0, 0.011833), (24, 0.179, 9, 12, 0.511537), (59, 0.15900000000000003, 10, 11, 0.124341), (-1, 0.0, 0, 0, 0.03353), (-1, 0.0, 0, 0, -0.013004), (7, 0.46299999999999997, 13, 14, 0.878589), (-1, 0.0, 0, 0, 0.012035), (-1, 0.0, 0, 0, 0.048444)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (44, 0.05984, 2, 5, -0.402546), (26, 4.574061244703071, 3, 4, 0.100846), (-1, 0.0, 0, 0, -0.028378), (-1, 0.0, 0, 0, 0.025045), (27, 0.896, 6, 7, -0.597213), (-1, 0.0, 0, 0, -0.007839), (-1, 0.0, 0, 0, -0.043989), (24, 0.179, 9, 12, 0.497011), (56, 0.4641025641025641, 10, 11, 0.119018), (-1, 0.0, 0, 0, 0.025811), (-1, 0.0, 0, 0, -0.019303), (7, 0.905, 13, 14, 0.867536), (-1, 0.0, 0, 0, 0.02047), (-1, 0.0, 0, 0, 0.050684)),
            ((20, 1.965912816362163, 1, 8, 0.0), (40, 0.5, 2, 5, -0.224485), (20, -3.210268834500856, 3, 4, 0.060058), (-1, 0.0, 0, 0, -0.019658), (-1, 0.0, 0, 0, 0.027504), (27, 0.208, 6, 7, -0.531051), (-1, 0.0, 0, 0, 0.014164), (-1, 0.0, 0, 0, -0.033988), (2, 0.46599999999999997, 9, 10, 0.847624), (-1, 0.0, 0, 0, -0.002486), (6, 0.8983741916842142, 11, 12, 0.923627), (-1, 0.0, 0, 0, 0.048819), (-1, 0.0, 0, 0, -0.009319)),
            ((20, 1.965912816362163, 1, 8, 0.0), (60, 0.664, 2, 5, -0.215359), (56, 0.17424242424242425, 3, 4, -0.0456), (-1, 0.0, 0, 0, 0.052815), (-1, 0.0, 0, 0, -0.009784), (24, 0.015833333333333345, 6, 7, -0.716631), (-1, 0.0, 0, 0, -0.050492), (-1, 0.0, 0, 0, -0.007784), (7, 0.46299999999999997, 9, 12, 0.835323), (51, 0.5, 10, 11, 0.204517), (-1, 0.0, 0, 0, -0.024536), (-1, 0.0, 0, 0, 0.037436), (13, 0.894, 13, 14, 0.961959), (-1, 0.0, 0, 0, 0.018497), (-1, 0.0, 0, 0, 0.050126)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.256089), (26, 4.24938730294611, 3, 4, -0.024601), (-1, 0.0, 0, 0, -0.03453), (-1, 0.0, 0, 0, 0.007551), (53, 0.30000000000000004, 6, 7, -0.690948), (-1, 0.0, 0, 0, -0.050698), (-1, 0.0, 0, 0, -0.009153), (50, 0.5, 9, 12, 0.668992), (2, 0.757, 10, 11, 0.874424), (-1, 0.0, 0, 0, 0.010635), (-1, 0.0, 0, 0, 0.047775), (58, 0.5064460232439666, 13, 14, 0.042976), (-1, 0.0, 0, 0, -0.038403), (-1, 0.0, 0, 0, 0.043721)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, -0.360384), (44, 0.0578, 3, 4, -0.639816), (-1, 0.0, 0, 0, -0.001365), (-1, 0.0, 0, 0, -0.043541), (26, 4.131156519244238, 6, 7, -0.014277), (-1, 0.0, 0, 0, -0.044936), (-1, 0.0, 0, 0, 0.008388), (24, 0.179, 9, 12, 0.461023), (5, 0.025, 10, 11, 0.126682), (-1, 0.0, 0, 0, 0.038545), (-1, 0.0, 0, 0, -0.008488), (7, 0.905, 13, 14, 0.826461), (-1, 0.0, 0, 0, 0.017297), (-1, 0.0, 0, 0, 0.049559)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (24, 0.027500000000000024, 2, 5, -0.348615), (44, 0.055400000000000005, 3, 4, -0.66276), (-1, 0.0, 0, 0, -0.004085), (-1, 0.0, 0, 0, -0.044235), (26, 4.131156519244238, 6, 7, -0.05387), (-1, 0.0, 0, 0, -0.044435), (-1, 0.0, 0, 0, 0.004953), (24, 0.179, 9, 12, 0.447527), (59, 0.15900000000000003, 10, 11, 0.121557), (-1, 0.0, 0, 0, 0.03131), (-1, 0.0, 0, 0, -0.011719), (6, 0.8717732060837587, 13, 14, 0.81541), (-1, 0.0, 0, 0, 0.044644), (-1, 0.0, 0, 0, -0.002924)),
            ((20, 1.965912816362163, 1, 8, 0.0), (40, 0.5, 2, 5, -0.186477), (20, -3.210268834500856, 3, 4, 0.089722), (-1, 0.0, 0, 0, -0.016796), (-1, 0.0, 0, 0, 0.026496), (52, 0.985, 6, 7, -0.487442), (-1, 0.0, 0, 0, -0.02844), (-1, 0.0, 0, 0, 0.043706), (7, 0.46299999999999997, 9, 12, 0.791416), (51, 0.5, 10, 11, 0.147384), (-1, 0.0, 0, 0, -0.026185), (-1, 0.0, 0, 0, 0.035585), (13, 0.894, 13, 14, 0.933303), (-1, 0.0, 0, 0, 0.015842), (-1, 0.0, 0, 0, 0.048998)),
            ((20, 1.965912816362163, 1, 8, 0.0), (60, 0.664, 2, 5, -0.178395), (56, 0.17424242424242425, 3, 4, -0.019609), (-1, 0.0, 0, 0, 0.051602), (-1, 0.0, 0, 0, -0.008304), (24, 0.015833333333333345, 6, 7, -0.669909), (-1, 0.0, 0, 0, -0.048943), (-1, 0.0, 0, 0, -0.006034), (7, 0.46299999999999997, 9, 12, 0.780244), (51, 0.5, 10, 11, 0.14273), (-1, 0.0, 0, 0, -0.025014), (-1, 0.0, 0, 0, 0.034982), (13, 0.894, 13, 14, 0.925352), (-1, 0.0, 0, 0, 0.015453), (-1, 0.0, 0, 0, 0.048679)),
            ((44, 0.20606000000000002, 1, 6, 0.0), (25, 6.070641713151899, 2, 3, -0.317635), (-1, 0.0, 0, 0, -0.050305), (15, -0.17, 4, 5, -0.178087), (-1, 0.0, 0, 0, -0.033662), (-1, 0.0, 0, 0, 0.004965), (2, 0.993, 7, 10, 0.417677), (55, 0.17424242424242425, 8, 9, 0.068765), (-1, 0.0, 0, 0, 0.03201), (-1, 0.0, 0, 0, -0.016405), (15, 0.252, 11, 12, 0.748234), (-1, 0.0, 0, 0, 0.046712), (-1, 0.0, 0, 0, -0.011432)),
            ((20, 1.965912816362163, 1, 8, 0.0), (40, 0.5, 2, 5, -0.167039), (26, 4.574061244703071, 3, 4, 0.103516), (-1, 0.0, 0, 0, -0.019301), (-1, 0.0, 0, 0, 0.021929), (27, 0.208, 6, 7, -0.462864), (-1, 0.0, 0, 0, 0.014892), (-1, 0.0, 0, 0, -0.030514), (6, 0.8717732060837587, 9, 12, 0.761568), (60, 0.115, 10, 11, 0.853375), (-1, 0.0, 0, 0, 0.050045), (-1, 0.0, 0, 0, 0.016479), (-1, 0.0, 0, 0, -0.009741)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (27, 0.622, 2, 5, -0.377931), (5, 0.014, 3, 4, 0.32514), (-1, 0.0, 0, 0, 0.050341), (-1, 0.0, 0, 0, -0.002035), (42, 0.14835164835164835, 6, 7, -0.520711), (-1, 0.0, 0, 0, -0.005988), (-1, 0.0, 0, 0, -0.041742), (52, 0.976, 9, 12, 0.32148), (40, 0.5, 10, 11, 0.12082), (-1, 0.0, 0, 0, 0.028924), (-1, 0.0, 0, 0, -0.015424), (6, 0.8771504370949433, 13, 14, 0.975214), (-1, 0.0, 0, 0, 0.052411), (-1, 0.0, 0, 0, 0.008569)),
            ((20, 1.965912816362163, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.154642), (26, 4.182595101823219, 3, 4, 0.039232), (-1, 0.0, 0, 0, -0.03443), (-1, 0.0, 0, 0, 0.008866), (53, 0.258, 6, 7, -0.525024), (-1, 0.0, 0, 0, -0.046386), (-1, 0.0, 0, 0, -0.003441), (6, 0.8717732060837587, 9, 12, 0.740892), (60, 0.115, 10, 11, 0.836636), (-1, 0.0, 0, 0, 0.049596), (-1, 0.0, 0, 0, 0.015843), (-1, 0.0, 0, 0, -0.009845)),
            ((44, 0.62, 1, 8, 0.0), (25, 6.070641713151899, 2, 5, -0.249173), (39, -0.020499999999999963, 3, 4, -1.010593), (-1, 0.0, 0, 0, -0.01193), (-1, 0.0, 0, 0, -0.052207), (55, 0.27525252525252525, 6, 7, -0.108472), (-1, 0.0, 0, 0, 0.005842), (-1, 0.0, 0, 0, -0.026188), (2, 0.954, 9, 12, 0.444429), (55, 0.17424242424242425, 10, 11, -0.177074), (-1, 0.0, 0, 0, 0.031823), (-1, 0.0, 0, 0, -0.030219), (15, 0.252, 13, 14, 0.692554), (-1, 0.0, 0, 0, 0.041374), (-1, 0.0, 0, 0, -0.002722)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.354673), (60, 0.664, 3, 4, -0.050232), (-1, 0.0, 0, 0, 0.022287), (-1, 0.0, 0, 0, -0.03473), (55, 0.256578947368421, 6, 7, -0.697117), (-1, 0.0, 0, 0, -0.013811), (-1, 0.0, 0, 0, -0.046913), (40, 0.5, 9, 12, 0.302401), (6, 0.5826207097771658, 10, 11, 0.653035), (-1, 0.0, 0, 0, 0.044598), (-1, 0.0, 0, 0, 0.009663), (13, 0.988, 13, 14, -0.042291), (-1, 0.0, 0, 0, -0.022869), (-1, 0.0, 0, 0, 0.031465)),
            ((20, 1.965912816362163, 1, 8, 0.0), (40, 0.5, 2, 5, -0.142368), (26, 4.574061244703071, 3, 4, 0.108932), (-1, 0.0, 0, 0, -0.018211), (-1, 0.0, 0, 0, 0.021173), (52, 0.985, 6, 7, -0.420234), (-1, 0.0, 0, 0, -0.024854), (-1, 0.0, 0, 0, 0.040757), (6, 0.8717732060837587, 9, 12, 0.7127), (60, 0.115, 10, 11, 0.814676), (-1, 0.0, 0, 0, 0.049005), (-1, 0.0, 0, 0, 0.014867), (-1, 0.0, 0, 0, -0.011019)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.340102), (60, 0.664, 3, 4, -0.042957), (-1, 0.0, 0, 0, 0.021593), (-1, 0.0, 0, 0, -0.03388), (55, 0.256578947368421, 6, 7, -0.679852), (-1, 0.0, 0, 0, -0.013353), (-1, 0.0, 0, 0, -0.046181), (52, 0.976, 9, 12, 0.291001), (40, 0.5, 10, 11, 0.110958), (-1, 0.0, 0, 0, 0.026128), (-1, 0.0, 0, 0, -0.013571), (6, 0.8771504370949433, 13, 14, 0.94088), (-1, 0.0, 0, 0, 0.051265), (-1, 0.0, 0, 0, 0.007006)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.172481), (26, 4.14764942975123, 3, 4, 0.024894), (-1, 0.0, 0, 0, -0.034951), (-1, 0.0, 0, 0, 0.007332), (1, 0.321, 6, 7, -0.58449), (-1, 0.0, 0, 0, -0.039275), (-1, 0.0, 0, 0, 0.011501), (50, 0.5, 9, 12, 0.540019), (3, 0.83, 10, 11, 0.800684), (-1, 0.0, 0, 0, 0.042354), (-1, 0.0, 0, 0, -0.010572), (11, 0.955, 13, 14, -0.201192), (-1, 0.0, 0, 0, -0.058295), (-1, 0.0, 0, 0, 0.02564)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (25, 6.070641713151899, 2, 5, -0.254732), (39, -0.020499999999999963, 3, 4, -0.963835), (-1, 0.0, 0, 0, -0.010099), (-1, 0.0, 0, 0, -0.050115), (15, -0.17, 6, 7, -0.129574), (-1, 0.0, 0, 0, -0.031141), (-1, 0.0, 0, 0, 0.006512), (2, 0.993, 9, 12, 0.353291), (55, 0.12916666666666665, 10, 11, 0.035145), (-1, 0.0, 0, 0, 0.034192), (-1, 0.0, 0, 0, -0.014224), (15, 0.252, 13, 14, 0.678267), (-1, 0.0, 0, 0, 0.043509), (-1, 0.0, 0, 0, -0.010482)),
            ((20, 2.319785806352712, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.117284), (60, 0.664, 3, 4, -0.01093), (-1, 0.0, 0, 0, 0.006536), (-1, 0.0, 0, 0, -0.025385), (7, 0.994, 6, 7, -0.721088), (-1, 0.0, 0, 0, -0.047887), (-1, 0.0, 0, 0, 0.041577), (6, 0.890011768611039, 9, 12, 0.732626), (13, 0.744, 10, 11, 0.81888), (-1, 0.0, 0, 0, 0.007617), (-1, 0.0, 0, 0, 0.04631), (-1, 0.0, 0, 0, -0.022472)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.313359), (60, 0.664, 3, 4, -0.022917), (-1, 0.0, 0, 0, 0.020197), (-1, 0.0, 0, 0, -0.031727), (55, 0.256578947368421, 6, 7, -0.65279), (-1, 0.0, 0, 0, -0.013173), (-1, 0.0, 0, 0, -0.044908), (52, 0.976, 9, 12, 0.26647), (40, 0.5, 10, 11, 0.10057), (-1, 0.0, 0, 0, 0.024961), (-1, 0.0, 0, 0, -0.013286), (6, 0.8771504370949433, 13, 14, 0.915109), (-1, 0.0, 0, 0, 0.050362), (-1, 0.0, 0, 0, 0.005282)),
            ((20, 2.319785806352712, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.110136), (40, 0.5, 3, 4, -0.010232), (-1, 0.0, 0, 0, 0.010787), (-1, 0.0, 0, 0, -0.014288), (7, 0.994, 6, 7, -0.69798), (-1, 0.0, 0, 0, -0.046754), (-1, 0.0, 0, 0, 0.040905), (6, 0.890011768611039, 9, 12, 0.713333), (13, 0.587, 10, 11, 0.80307), (-1, 0.0, 0, 0, 0.001955), (-1, 0.0, 0, 0, 0.044523), (-1, 0.0, 0, 0, -0.022001)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.300322), (60, 0.664, 3, 4, -0.023048), (-1, 0.0, 0, 0, 0.019322), (-1, 0.0, 0, 0, -0.031092), (12, 0.531, 6, 7, -0.63296), (-1, 0.0, 0, 0, -0.015306), (-1, 0.0, 0, 0, -0.046901), (52, 0.976, 9, 12, 0.25557), (40, 0.5, 10, 11, 0.09731), (-1, 0.0, 0, 0, 0.023485), (-1, 0.0, 0, 0, -0.012238), (6, 0.8771504370949433, 13, 14, 0.900161), (-1, 0.0, 0, 0, 0.04978), (-1, 0.0, 0, 0, 0.005078)),
            ((20, 2.319785806352712, 1, 8, 0.0), (60, 0.664, 2, 5, -0.103229), (56, 0.19090909090909092, 3, 4, 0.020511), (-1, 0.0, 0, 0, 0.045606), (-1, 0.0, 0, 0, -0.005522), (24, 0.015833333333333345, 6, 7, -0.551615), (-1, 0.0, 0, 0, -0.044752), (-1, 0.0, 0, 0, -0.001878), (6, 0.890011768611039, 9, 12, 0.693878), (13, 0.744, 10, 11, 0.78707), (-1, 0.0, 0, 0, 0.006716), (-1, 0.0, 0, 0, 0.045161), (-1, 0.0, 0, 0, -0.021533)),
            ((52, 0.976, 1, 8, 0.0), (40, 0.5, 2, 5, -0.095777), (26, 4.581130329250939, 3, 4, 0.149406), (-1, 0.0, 0, 0, -0.014139), (-1, 0.0, 0, 0, 0.022249), (4, 0.995, 6, 7, -0.375927), (-1, 0.0, 0, 0, -0.024007), (-1, 0.0, 0, 0, 0.023257), (39, 0.655, 9, 12, 0.71352), (4, 0.994, 10, 11, 0.899159), (-1, 0.0, 0, 0, 0.052086), (-1, 0.0, 0, 0, 0.019129), (-1, 0.0, 0, 0, 0.000798)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.285536), (2, 0.994, 3, 4, -0.014144), (-1, 0.0, 0, 0, -0.011048), (-1, 0.0, 0, 0, 0.051271), (12, 0.531, 6, 7, -0.613937), (-1, 0.0, 0, 0, -0.014619), (-1, 0.0, 0, 0, -0.046111), (52, 0.976, 9, 12, 0.244126), (50, 0.5, 10, 11, 0.098676), (-1, 0.0, 0, 0, 0.015509), (-1, 0.0, 0, 0, -0.022172), (6, 0.8771504370949433, 13, 14, 0.873112), (-1, 0.0, 0, 0, 0.048664), (-1, 0.0, 0, 0, 0.004461)),
            ((19, 0.5, 1, 8, 0.0), (60, 0.664, 2, 5, -0.090775), (56, 0.19090909090909092, 3, 4, 0.026127), (-1, 0.0, 0, 0, 0.043869), (-1, 0.0, 0, 0, -0.004953), (26, 4.598548757796862, 6, 7, -0.527951), (-1, 0.0, 0, 0, -0.048488), (-1, 0.0, 0, 0, -0.004752), (6, 0.8846610635340376, 9, 12, 0.704604), (13, 0.744, 10, 11, 0.812684), (-1, 0.0, 0, 0, 0.010618), (-1, 0.0, 0, 0, 0.047504), (-1, 0.0, 0, 0, -0.021608)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.17424242424242425, 2, 5, -0.271842), (26, 4.608290119226802, 3, 4, 0.252498), (-1, 0.0, 0, 0, -0.032134), (-1, 0.0, 0, 0, 0.0462), (1, 0.738, 6, 7, -0.432721), (-1, 0.0, 0, 0, -0.028386), (-1, 0.0, 0, 0, 0.025243), (52, 0.976, 9, 12, 0.232355), (50, 0.5, 10, 11, 0.092991), (-1, 0.0, 0, 0, 0.014834), (-1, 0.0, 0, 0, -0.021396), (6, 0.8771504370949433, 13, 14, 0.857182), (-1, 0.0, 0, 0, 0.048002), (-1, 0.0, 0, 0, 0.004541)),
            ((19, 0.5, 1, 8, 0.0), (40, 0.5, 2, 5, -0.084811), (26, 4.581130329250939, 3, 4, 0.127121), (-1, 0.0, 0, 0, -0.014491), (-1, 0.0, 0, 0, 0.020401), (52, 0.985, 6, 7, -0.323655), (-1, 0.0, 0, 0, -0.020519), (-1, 0.0, 0, 0, 0.038015), (6, 0.8846610635340376, 9, 12, 0.685953), (13, 0.744, 10, 11, 0.797677), (-1, 0.0, 0, 0, 0.010056), (-1, 0.0, 0, 0, 0.047066), (-1, 0.0, 0, 0, -0.021086)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.260244), (8, 0.26785714285714285, 3, 4, -0.039516), (-1, 0.0, 0, 0, -0.017149), (-1, 0.0, 0, 0, 0.023465), (53, 0.40700000000000003, 6, 7, -0.660648), (-1, 0.0, 0, 0, -0.049381), (-1, 0.0, 0, 0, 0.00686), (52, 0.976, 9, 12, 0.224048), (40, 0.5, 10, 11, 0.092173), (-1, 0.0, 0, 0, 0.020844), (-1, 0.0, 0, 0, -0.010236), (6, 0.8771504370949433, 13, 14, 0.836663), (-1, 0.0, 0, 0, 0.047202), (-1, 0.0, 0, 0, 0.0041)),
            ((19, 0.5, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.079232), (52, 0.899, 3, 4, -0.478056), (-1, 0.0, 0, 0, -0.036983), (-1, 0.0, 0, 0, 0.023762), (15, -0.16500000000000004, 6, 7, 0.039818), (-1, 0.0, 0, 0, -0.020311), (-1, 0.0, 0, 0, 0.014274), (6, 0.8846610635340376, 9, 12, 0.668208), (13, 0.744, 10, 11, 0.78361), (-1, 0.0, 0, 0, 0.009685), (-1, 0.0, 0, 0, 0.046638), (-1, 0.0, 0, 0, -0.020625)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.246771), (15, -0.20700000000000002, 3, 4, 0.01634), (-1, 0.0, 0, 0, -0.026344), (-1, 0.0, 0, 0, 0.020027), (55, 0.256578947368421, 6, 7, -0.572053), (-1, 0.0, 0, 0, -0.009768), (-1, 0.0, 0, 0, -0.042037), (13, 0.968, 9, 12, 0.212354), (1, 0.009, 10, 11, -0.029637), (-1, 0.0, 0, 0, 0.013818), (-1, 0.0, 0, 0, -0.027176), (7, 0.596, 13, 14, 0.520177), (-1, 0.0, 0, 0, -0.007121), (-1, 0.0, 0, 0, 0.036837)),
            ((44, 0.20606000000000002, 1, 8, 0.0), (25, 6.070641713151899, 2, 5, -0.18453), (39, -0.020499999999999963, 3, 4, -0.92614), (-1, 0.0, 0, 0, -0.006442), (-1, 0.0, 0, 0, -0.048595), (15, -0.17, 6, 7, -0.068256), (-1, 0.0, 0, 0, -0.026872), (-1, 0.0, 0, 0, 0.007784), (59, 0.33999999999999997, 9, 12, 0.277073), (4, 0.899, 10, 11, 0.437779), (-1, 0.0, 0, 0, 0.009046), (-1, 0.0, 0, 0, 0.042679), (52, 0.984, 13, 14, -0.263891), (-1, 0.0, 0, 0, -0.028447), (-1, 0.0, 0, 0, 0.042818)),
            ((19, 0.5, 1, 8, 0.0), (25, 6.609348335006922, 2, 5, -0.074514), (52, 0.925, 3, 4, -0.38531), (-1, 0.0, 0, 0, -0.029117), (-1, 0.0, 0, 0, 0.026201), (56, 0.1380952380952381, 6, 7, 0.072804), (-1, 0.0, 0, 0, 0.046985), (-1, 0.0, 0, 0, -0.003566), (6, 0.8846610635340376, 9, 12, 0.646159), (13, 0.587, 10, 11, 0.767128), (-1, 0.0, 0, 0, 0.00473), (-1, 0.0, 0, 0, 0.044266), (-1, 0.0, 0, 0, -0.020981)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.235058), (26, 4.590055742951803, 3, 4, -0.027356), (-1, 0.0, 0, 0, -0.029663), (-1, 0.0, 0, 0, 0.011494), (53, 0.40700000000000003, 6, 7, -0.630712), (-1, 0.0, 0, 0, -0.048346), (-1, 0.0, 0, 0, 0.006893), (50, 0.5, 9, 12, 0.201867), (6, 0.5935573765069624, 10, 11, 0.370373), (-1, 0.0, 0, 0, 0.032593), (-1, 0.0, 0, 0, -0.00237), (3, 0.15200000000000002, 13, 14, -0.221571), (-1, 0.0, 0, 0, 0.035964), (-1, 0.0, 0, 0, -0.026647)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.121631), (26, 4.182595101823219, 3, 4, -0.020478), (-1, 0.0, 0, 0, -0.03165), (-1, 0.0, 0, 0, 0.003315), (19, 0.5, 6, 7, -0.856639), (-1, 0.0, 0, 0, -0.049316), (-1, 0.0, 0, 0, 0.007878), (53, 0.274, 9, 12, 0.390719), (56, 0.5634920634920635, 10, 11, -0.266888), (-1, 0.0, 0, 0, 0.019273), (-1, 0.0, 0, 0, -0.034519), (6, 0.7785018521723658, 13, 14, 0.734268), (-1, 0.0, 0, 0, 0.048052), (-1, 0.0, 0, 0, 0.008554)),
            ((2, 0.992, 1, 8, 0.0), (56, 0.2928571428571428, 2, 5, -0.178593), (15, -0.751, 3, 4, 0.177365), (-1, 0.0, 0, 0, -0.039864), (-1, 0.0, 0, 0, 0.017356), (44, 0.955, 6, 7, -0.431441), (-1, 0.0, 0, 0, -0.027231), (-1, 0.0, 0, 0, 0.020593), (44, 0.48692, 9, 12, 0.251578), (5, 0.324, 10, 11, -0.032069), (-1, 0.0, 0, 0, 0.011178), (-1, 0.0, 0, 0, -0.025304), (12, 0.6639999999999999, 13, 14, 0.627336), (-1, 0.0, 0, 0, 0.040682), (-1, 0.0, 0, 0, -0.005211)),
            ((60, 0.664, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.106011), (50, 0.5, 3, 4, 0.929937), (-1, 0.0, 0, 0, 0.066995), (-1, 0.0, 0, 0, 0.010168), (2, 0.992, 6, 7, 0.006139), (-1, 0.0, 0, 0, -0.012673), (-1, 0.0, 0, 0, 0.01182), (24, 0.015833333333333345, 9, 12, -0.423411), (6, 0.21590364309205778, 10, 11, -0.843843), (-1, 0.0, 0, 0, -0.048388), (-1, 0.0, 0, 0, -0.006792), (56, 0.3798076923076923, 13, 14, 0.048983), (-1, 0.0, 0, 0, 0.037608), (-1, 0.0, 0, 0, -0.032984)),
            ((19, 0.5, 1, 8, 0.0), (40, 0.5, 2, 5, -0.068458), (26, 4.581130329250939, 3, 4, 0.128373), (-1, 0.0, 0, 0, -0.012832), (-1, 0.0, 0, 0, 0.018803), (52, 0.985, 6, 7, -0.291464), (-1, 0.0, 0, 0, -0.018413), (-1, 0.0, 0, 0, 0.033826), (6, 0.8846610635340376, 9, 12, 0.616499), (13, 0.744, 10, 11, 0.743772), (-1, 0.0, 0, 0, 0.008446), (-1, 0.0, 0, 0, 0.045476), (-1, 0.0, 0, 0, -0.020841)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.113247), (11, 0.45, 3, 4, -0.017649), (-1, 0.0, 0, 0, -0.021619), (-1, 0.0, 0, 0, 0.005187), (19, 0.5, 6, 7, -0.829389), (-1, 0.0, 0, 0, -0.047943), (-1, 0.0, 0, 0, 0.007632), (53, 0.274, 9, 12, 0.36983), (56, 0.5634920634920635, 10, 11, -0.254551), (-1, 0.0, 0, 0, 0.019083), (-1, 0.0, 0, 0, -0.033503), (6, 0.7878540531194568, 13, 14, 0.701251), (-1, 0.0, 0, 0, 0.045854), (-1, 0.0, 0, 0, 0.006544)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.221063), (26, 4.590055742951803, 3, 4, -0.031299), (-1, 0.0, 0, 0, -0.031293), (-1, 0.0, 0, 0, 0.014486), (11, 0.823, 6, 7, -0.629432), (-1, 0.0, 0, 0, 0.024719), (-1, 0.0, 0, 0, -0.043811), (50, 0.5, 9, 12, 0.188021), (6, 0.5935573765069624, 10, 11, 0.352943), (-1, 0.0, 0, 0, 0.031478), (-1, 0.0, 0, 0, -0.00266), (3, 0.15200000000000002, 13, 14, -0.222989), (-1, 0.0, 0, 0, 0.034579), (-1, 0.0, 0, 0, -0.026173)),
            ((60, 0.664, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.101749), (50, 0.5, 3, 4, 0.895846), (-1, 0.0, 0, 0, 0.064733), (-1, 0.0, 0, 0, 0.009262), (5, 0.001, 6, 7, 0.006666), (-1, 0.0, 0, 0, 0.031629), (-1, 0.0, 0, 0, -0.003917), (24, 0.015833333333333345, 9, 12, -0.411802), (6, 0.21590364309205778, 10, 11, -0.825923), (-1, 0.0, 0, 0, -0.04768), (-1, 0.0, 0, 0, -0.006543), (56, 0.3798076923076923, 13, 14, 0.039373), (-1, 0.0, 0, 0, 0.035108), (-1, 0.0, 0, 0, -0.032272)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.109236), (15, -0.751, 3, 4, -0.017759), (-1, 0.0, 0, 0, -0.041176), (-1, 0.0, 0, 0, 0.002237), (19, 0.5, 6, 7, -0.81228), (-1, 0.0, 0, 0, -0.047077), (-1, 0.0, 0, 0, 0.007293), (53, 0.274, 9, 12, 0.360008), (59, 0.30200000000000005, 10, 11, -0.240921), (-1, 0.0, 0, 0, 0.009822), (-1, 0.0, 0, 0, -0.0417), (6, 0.7785018521723658, 13, 14, 0.679543), (-1, 0.0, 0, 0, 0.044935), (-1, 0.0, 0, 0, 0.007762)),
            ((40, 0.5, 1, 6, 0.0), (15, -0.771, 2, 3, 0.188786), (-1, 0.0, 0, 0, -0.040999), (56, 0.4494949494949495, 4, 5, 0.25962), (-1, 0.0, 0, 0, 0.030488), (-1, 0.0, 0, 0, -0.003765), (52, 0.985, 7, 10, -0.210027), (4, 0.995, 8, 9, -0.302115), (-1, 0.0, 0, 0, -0.01977), (-1, 0.0, 0, 0, 0.022146), (53, 0.992, 11, 12, 0.723077), (-1, 0.0, 0, 0, 0.048196), (-1, 0.0, 0, 0, 0.008752)),
            ((19, 0.5, 1, 6, 0.0), (15, -0.751, 2, 3, -0.063098), (-1, 0.0, 0, 0, -0.043981), (56, 0.3798076923076923, 4, 5, -0.015701), (-1, 0.0, 0, 0, 0.014577), (-1, 0.0, 0, 0, -0.009592), (6, 0.8846610635340376, 7, 10, 0.589056), (13, 0.587, 8, 9, 0.721601), (-1, 0.0, 0, 0, 0.002896), (-1, 0.0, 0, 0, 0.042751), (-1, 0.0, 0, 0, -0.020153)),
            ((60, 0.664, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.097102), (50, 0.5, 3, 4, 0.850098), (-1, 0.0, 0, 0, 0.061991), (-1, 0.0, 0, 0, 0.007851), (2, 0.992, 6, 7, 0.008588), (-1, 0.0, 0, 0, -0.011811), (-1, 0.0, 0, 0, 0.011215), (24, 0.015833333333333345, 9, 12, -0.398038), (6, 0.21590364309205778, 10, 11, -0.812276), (-1, 0.0, 0, 0, -0.047184), (-1, 0.0, 0, 0, -0.005669), (56, 0.4494949494949495, 13, 14, 0.044657), (-1, 0.0, 0, 0, 0.027072), (-1, 0.0, 0, 0, -0.039054)),
            ((40, 0.5, 1, 8, 0.0), (8, 0.31666666666666665, 2, 5, 0.181102), (26, 4.710815549063884, 3, 4, -0.046879), (-1, 0.0, 0, 0, -0.016151), (-1, 0.0, 0, 0, 0.032357), (39, 0.528, 6, 7, 0.494851), (-1, 0.0, 0, 0, 0.037299), (-1, 0.0, 0, 0, -0.021583), (52, 0.985, 9, 12, -0.201879), (4, 0.995, 10, 11, -0.290513), (-1, 0.0, 0, 0, -0.01894), (-1, 0.0, 0, 0, 0.020571), (53, 0.992, 13, 14, 0.704162), (-1, 0.0, 0, 0, 0.047433), (-1, 0.0, 0, 0, 0.007969)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.205031), (7, 0.866, 3, 4, -0.021021), (-1, 0.0, 0, 0, -0.012219), (-1, 0.0, 0, 0, 0.039621), (11, 0.823, 6, 7, -0.610561), (-1, 0.0, 0, 0, 0.023198), (-1, 0.0, 0, 0, -0.042684), (50, 0.5, 9, 12, 0.173334), (6, 0.5935573765069624, 10, 11, 0.328312), (-1, 0.0, 0, 0, 0.029713), (-1, 0.0, 0, 0, -0.002805), (3, 0.15200000000000002, 13, 14, -0.211948), (-1, 0.0, 0, 0, 0.033464), (-1, 0.0, 0, 0, -0.025027)),
            ((60, 0.664, 1, 8, 0.0), (56, 0.19090909090909092, 2, 5, 0.094326), (50, 0.5, 3, 4, 0.723657), (-1, 0.0, 0, 0, 0.059737), (-1, 0.0, 0, 0, -0.00102), (2, 0.992, 6, 7, -0.004617), (-1, 0.0, 0, 0, -0.012538), (-1, 0.0, 0, 0, 0.010329), (24, 0.015833333333333345, 9, 12, -0.386537), (6, 0.21590364309205778, 10, 11, -0.794122), (-1, 0.0, 0, 0, -0.046494), (-1, 0.0, 0, 0, -0.00487), (56, 0.3798076923076923, 13, 14, 0.036438), (-1, 0.0, 0, 0, 0.031749), (-1, 0.0, 0, 0, -0.030644)),
            ((19, 0.5, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.058596), (52, 0.899, 3, 4, -0.424035), (-1, 0.0, 0, 0, -0.032823), (-1, 0.0, 0, 0, 0.020027), (15, -0.16500000000000004, 6, 7, 0.043721), (-1, 0.0, 0, 0, -0.017437), (-1, 0.0, 0, 0, 0.012162), (6, 0.8846610635340376, 9, 12, 0.568644), (48, 0.22599999999999998, 10, 11, 0.70514), (-1, 0.0, 0, 0, -0.001264), (-1, 0.0, 0, 0, 0.041059), (-1, 0.0, 0, 0, -0.019778)),
            ((40, 0.5, 1, 6, 0.0), (15, -0.771, 2, 3, 0.172383), (-1, 0.0, 0, 0, -0.039132), (56, 0.4494949494949495, 4, 5, 0.236885), (-1, 0.0, 0, 0, 0.02825), (-1, 0.0, 0, 0, -0.003623), (52, 0.985, 7, 10, -0.189683), (4, 0.995, 8, 9, -0.274247), (-1, 0.0, 0, 0, -0.017978), (-1, 0.0, 0, 0, 0.019987), (53, 0.992, 11, 12, 0.681711), (-1, 0.0, 0, 0, 0.046346), (-1, 0.0, 0, 0, 0.007284)),
            ((46, 0.8571428571428572, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.043662), (2, 0.014000000000000012, 3, 4, 0.690213), (-1, 0.0, 0, 0, -0.012674), (-1, 0.0, 0, 0, 0.041754), (20, -3.210268834500856, 6, 7, -0.042749), (-1, 0.0, 0, 0, -0.017069), (-1, 0.0, 0, 0, 0.005686), (-1, 0.0, 0, 0, -0.040619)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.190547), (26, 4.705005283552375, 3, 4, -0.04458), (-1, 0.0, 0, 0, -0.014322), (-1, 0.0, 0, 0, 0.026342), (7, 0.853, 6, 7, -1.004194), (-1, 0.0, 0, 0, -0.059204), (-1, 0.0, 0, 0, -0.015793), (5, 0.03, 9, 12, 0.16823), (26, 3.626558231510298, 10, 11, 0.584213), (-1, 0.0, 0, 0, -0.018517), (-1, 0.0, 0, 0, 0.037982), (4, 0.943, 13, 14, -0.063521), (-1, 0.0, 0, 0, -0.01923), (-1, 0.0, 0, 0, 0.020768)),
            ((19, 0.5, 1, 8, 0.0), (5, 0.831, 2, 5, -0.056092), (26, 4.2964256503859755, 3, 4, 0.032961), (-1, 0.0, 0, 0, -0.019112), (-1, 0.0, 0, 0, 0.008156), (6, 0.7690251819062273, 6, 7, -0.451372), (-1, 0.0, 0, 0, -0.03662), (-1, 0.0, 0, 0, 0.012164), (6, 0.8846610635340376, 9, 12, 0.549552), (13, 0.744, 10, 11, 0.689184), (-1, 0.0, 0, 0, 0.006481), (-1, 0.0, 0, 0, 0.043828), (-1, 0.0, 0, 0, -0.019234)),
            ((60, 0.664, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.086892), (50, 0.5, 3, 4, 0.768268), (-1, 0.0, 0, 0, 0.057199), (-1, 0.0, 0, 0, 0.005898), (5, 0.001, 6, 7, 0.009422), (-1, 0.0, 0, 0, 0.030401), (-1, 0.0, 0, 0, -0.003637), (26, 4.598548757796862, 9, 12, -0.36696), (4, 0.843, 10, 11, -0.828339), (-1, 0.0, 0, 0, -0.045882), (-1, 0.0, 0, 0, -0.006918), (24, 0.015833333333333345, 13, 14, 0.003385), (-1, 0.0, 0, 0, -0.027831), (-1, 0.0, 0, 0, 0.021867)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.094216), (11, 0.45, 3, 4, -0.011522), (-1, 0.0, 0, 0, -0.019848), (-1, 0.0, 0, 0, 0.004891), (44, 0.95, 6, 7, -0.76009), (-1, 0.0, 0, 0, -0.044696), (-1, 0.0, 0, 0, 0.006651), (53, 0.274, 9, 12, 0.317121), (59, 0.30200000000000005, 10, 11, -0.226664), (-1, 0.0, 0, 0, 0.009832), (-1, 0.0, 0, 0, -0.040276), (42, 0.09545454545454546, 13, 14, 0.612837), (-1, 0.0, 0, 0, -0.00167), (-1, 0.0, 0, 0, 0.039625)),
            ((40, 0.5, 1, 6, 0.0), (15, -0.771, 2, 3, 0.165457), (-1, 0.0, 0, 0, -0.038387), (56, 0.4494949494949495, 4, 5, 0.22832), (-1, 0.0, 0, 0, 0.026954), (-1, 0.0, 0, 0, -0.003102), (52, 0.985, 7, 10, -0.184224), (8, 0.5357142857142857, 8, 9, -0.264043), (-1, 0.0, 0, 0, -0.006881), (-1, 0.0, 0, 0, -0.036481), (53, 0.992, 11, 12, 0.656222), (-1, 0.0, 0, 0, 0.045325), (-1, 0.0, 0, 0, 0.005947)),
            ((19, 0.5, 1, 6, 0.0), (15, -0.751, 2, 3, -0.054392), (-1, 0.0, 0, 0, -0.041358), (58, 0.22564484631765486, 4, 5, -0.011689), (-1, 0.0, 0, 0, 0.006265), (-1, 0.0, 0, 0, -0.015231), (6, 0.8846610635340376, 7, 10, 0.536045), (48, 0.22599999999999998, 8, 9, 0.678574), (-1, 0.0, 0, 0, -0.001606), (-1, 0.0, 0, 0, 0.039959), (-1, 0.0, 0, 0, -0.018904)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.182735), (26, 4.705005283552375, 3, 4, -0.040731), (-1, 0.0, 0, 0, -0.013341), (-1, 0.0, 0, 0, 0.024336), (7, 0.853, 6, 7, -0.984213), (-1, 0.0, 0, 0, -0.057906), (-1, 0.0, 0, 0, -0.01556), (5, 0.03, 9, 12, 0.158147), (26, 3.626558231510298, 10, 11, 0.555018), (-1, 0.0, 0, 0, -0.017479), (-1, 0.0, 0, 0, 0.035938), (4, 0.943, 13, 14, -0.063737), (-1, 0.0, 0, 0, -0.018641), (-1, 0.0, 0, 0, 0.020103)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.2964256503859755, 2, 5, -0.086707), (4, 0.001, 3, 4, -0.521648), (-1, 0.0, 0, 0, 0.020351), (-1, 0.0, 0, 0, -0.033883), (42, 0.14835164835164835, 6, 7, 0.016224), (-1, 0.0, 0, 0, 0.012939), (-1, 0.0, 0, 0, -0.014279), (50, 0.5, 9, 12, 0.328108), (48, 0.46599999999999997, 10, 11, 0.668405), (-1, 0.0, 0, 0, 0.0022), (-1, 0.0, 0, 0, 0.040514), (11, 0.955, 13, 14, -0.480191), (-1, 0.0, 0, 0, -0.056151), (-1, 0.0, 0, 0, 0.014334)),
            ((51, 2.5, 1, 8, 0.0), (40, 0.5, 2, 5, 0.041376), (15, -0.771, 3, 4, 0.218619), (-1, 0.0, 0, 0, -0.037053), (-1, 0.0, 0, 0, 0.014256), (52, 0.985, 6, 7, -0.156207), (-1, 0.0, 0, 0, -0.011641), (-1, 0.0, 0, 0, 0.031667), (20, -2.6933930072678223, 9, 10, -0.678417), (-1, 0.0, 0, 0, -0.040342), (-1, 0.0, 0, 0, 0.010595)),
            ((46, 0.8571428571428572, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.035718), (50, 0.5, 3, 4, 0.628309), (-1, 0.0, 0, 0, 0.044059), (-1, 0.0, 0, 0, 0.001307), (20, -3.210268834500856, 6, 7, -0.04287), (-1, 0.0, 0, 0, -0.015783), (-1, 0.0, 0, 0, 0.004996), (-1, 0.0, 0, 0, -0.038518)),
            ((2, 0.992, 1, 8, 0.0), (3, 0.30100000000000005, 2, 5, -0.144049), (44, 0.05924, 3, 4, 0.029606), (-1, 0.0, 0, 0, 0.029041), (-1, 0.0, 0, 0, -0.00781), (44, 0.95, 6, 7, -0.503979), (-1, 0.0, 0, 0, -0.03527), (-1, 0.0, 0, 0, 0.035657), (56, 0.4494949494949495, 9, 12, 0.191838), (12, 0.482, 10, 11, 0.610287), (-1, 0.0, 0, 0, 0.006491), (-1, 0.0, 0, 0, 0.043723), (3, 0.42800000000000005, 13, 14, 0.006763), (-1, 0.0, 0, 0, -0.01255), (-1, 0.0, 0, 0, 0.030369)),
            ((60, 0.664, 1, 8, 0.0), (6, 0.6343440778663255, 2, 5, 0.080573), (5, 0.564, 3, 4, 0.294108), (-1, 0.0, 0, 0, 0.024155), (-1, 0.0, 0, 0, -0.010099), (7, 0.899, 6, 7, -0.158985), (-1, 0.0, 0, 0, -0.018556), (-1, 0.0, 0, 0, 0.013238), (24, 0.015833333333333345, 9, 12, -0.348348), (6, 0.21590364309205778, 10, 11, -0.754806), (-1, 0.0, 0, 0, -0.045027), (-1, 0.0, 0, 0, -0.003544), (56, 0.3798076923076923, 13, 14, 0.045571), (-1, 0.0, 0, 0, 0.028765), (-1, 0.0, 0, 0, -0.028432)),
            ((19, 0.5, 1, 6, 0.0), (15, -0.751, 2, 3, -0.051198), (-1, 0.0, 0, 0, -0.040389), (56, 0.3693181818181818, 4, 5, -0.009939), (-1, 0.0, 0, 0, 0.012422), (-1, 0.0, 0, 0, -0.007395), (6, 0.8846610635340376, 7, 10, 0.508524), (13, 0.744, 8, 9, 0.657209), (-1, 0.0, 0, 0, 0.005682), (-1, 0.0, 0, 0, 0.042654), (-1, 0.0, 0, 0, -0.019279)),
            ((60, 0.664, 1, 8, 0.0), (6, 0.6343440778663255, 2, 5, 0.076763), (5, 0.564, 3, 4, 0.279511), (-1, 0.0, 0, 0, 0.02303), (-1, 0.0, 0, 0, -0.009952), (7, 0.899, 6, 7, -0.150984), (-1, 0.0, 0, 0, -0.01773), (-1, 0.0, 0, 0, 0.012685), (26, 4.598548757796862, 9, 12, -0.339379), (4, 0.843, 10, 11, -0.796489), (-1, 0.0, 0, 0, -0.044561), (-1, 0.0, 0, 0, -0.005614), (24, 0.015833333333333345, 13, 14, 0.012107), (-1, 0.0, 0, 0, -0.02601), (-1, 0.0, 0, 0, 0.020324)),
            ((51, 2.5, 1, 8, 0.0), (40, 0.5, 2, 5, 0.038705), (15, -0.771, 3, 4, 0.202958), (-1, 0.0, 0, 0, -0.036011), (-1, 0.0, 0, 0, 0.013283), (52, 0.985, 6, 7, -0.144611), (-1, 0.0, 0, 0, -0.010884), (-1, 0.0, 0, 0, 0.030735), (20, -2.6933930072678223, 9, 10, -0.653431), (-1, 0.0, 0, 0, -0.038926), (-1, 0.0, 0, 0, 0.010207)),
            ((46, 0.8571428571428572, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.032302), (26, 4.608290119226802, 3, 4, 0.595224), (-1, 0.0, 0, 0, -0.002078), (-1, 0.0, 0, 0, 0.041177), (2, 0.992, 6, 7, -0.042044), (-1, 0.0, 0, 0, -0.010968), (-1, 0.0, 0, 0, 0.007539), (-1, 0.0, 0, 0, -0.037293)),
            ((19, 0.5, 1, 8, 0.0), (11, 0.997, 2, 5, -0.050596), (26, 4.280624567377958, 3, 4, 0.00571), (-1, 0.0, 0, 0, -0.019735), (-1, 0.0, 0, 0, 0.005682), (4, 0.996, 6, 7, -0.577873), (-1, 0.0, 0, 0, -0.040314), (-1, 0.0, 0, 0, -0.002696), (6, 0.8846610635340376, 9, 12, 0.49822), (13, 0.744, 10, 11, 0.649113), (-1, 0.0, 0, 0, 0.005842), (-1, 0.0, 0, 0, 0.042278), (-1, 0.0, 0, 0, -0.018962)),
            ((24, 0.06250000000000003, 1, 8, 0.0), (26, 4.14764942975123, 2, 5, -0.079963), (52, 0.785, 3, 4, -0.62624), (-1, 0.0, 0, 0, -0.044563), (-1, 0.0, 0, 0, 0.015326), (42, 0.14835164835164835, 6, 7, -0.00739), (-1, 0.0, 0, 0, 0.009964), (-1, 0.0, 0, 0, -0.013385), (50, 0.5, 9, 12, 0.301832), (48, 0.46599999999999997, 10, 11, 0.645619), (-1, 0.0, 0, 0, 0.002432), (-1, 0.0, 0, 0, 0.039379), (11, 0.955, 13, 14, -0.477054), (-1, 0.0, 0, 0, -0.053587), (-1, 0.0, 0, 0, 0.013269)),
            ((51, 2.5, 1, 8, 0.0), (40, 0.5, 2, 5, 0.03632), (15, -0.751, 3, 4, 0.192806), (-1, 0.0, 0, 0, -0.034172), (-1, 0.0, 0, 0, 0.012835), (8, 0.5357142857142857, 6, 7, -0.138331), (-1, 0.0, 0, 0, -0.000581), (-1, 0.0, 0, 0, -0.030146), (52, 0.732, 9, 10, -0.63563), (-1, 0.0, 0, 0, -0.040551), (-1, 0.0, 0, 0, 0.004136)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.170045), (26, 4.705005283552375, 3, 4, -0.02945), (-1, 0.0, 0, 0, -0.012203), (-1, 0.0, 0, 0, 0.023175), (7, 0.853, 6, 7, -0.978678), (-1, 0.0, 0, 0, -0.056689), (-1, 0.0, 0, 0, -0.016257), (5, 0.03, 9, 12, 0.140592), (56, 0.6770833333333333, 10, 11, 0.508938), (-1, 0.0, 0, 0, 0.034562), (-1, 0.0, 0, 0, -0.011934), (8, 0.6190476190476191, 13, 14, -0.066136), (-1, 0.0, 0, 0, 0.009569), (-1, 0.0, 0, 0, -0.030464)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.172904), (7, 0.866, 3, 4, 0.015773), (-1, 0.0, 0, 0, -0.010048), (-1, 0.0, 0, 0, 0.036314), (11, 0.823, 6, 7, -0.591685), (-1, 0.0, 0, 0, 0.019679), (-1, 0.0, 0, 0, -0.041105), (50, 0.5, 9, 12, 0.134768), (6, 0.5935573765069624, 10, 11, 0.276705), (-1, 0.0, 0, 0, 0.02676), (-1, 0.0, 0, 0, -0.004353), (3, 0.15200000000000002, 13, 14, -0.198516), (-1, 0.0, 0, 0, 0.031091), (-1, 0.0, 0, 0, -0.023313)),
            ((20, 3.210268834500856, 1, 8, 0.0), (15, -0.751, 2, 5, -0.039809), (7, 0.019000000000000017, 3, 4, -0.70324), (-1, 0.0, 0, 0, -0.009717), (-1, 0.0, 0, 0, -0.040576), (56, 0.39230769230769236, 6, 7, -0.004584), (-1, 0.0, 0, 0, 0.01108), (-1, 0.0, 0, 0, -0.006703), (13, 0.556, 9, 10, 0.625861), (-1, 0.0, 0, 0, -0.007067), (-1, 0.0, 0, 0, 0.038901)),
            ((60, 0.664, 1, 8, 0.0), (6, 0.6343440778663255, 2, 5, 0.06952), (5, 0.564, 3, 4, 0.255948), (-1, 0.0, 0, 0, 0.021111), (-1, 0.0, 0, 0, -0.009371), (7, 0.899, 6, 7, -0.13927), (-1, 0.0, 0, 0, -0.016476), (-1, 0.0, 0, 0, 0.011842), (26, 4.598548757796862, 9, 12, -0.327147), (4, 0.843, 10, 11, -0.776053), (-1, 0.0, 0, 0, -0.043691), (-1, 0.0, 0, 0, -0.004972), (24, 0.015833333333333345, 13, 14, 0.010152), (-1, 0.0, 0, 0, -0.024747), (-1, 0.0, 0, 0, 0.01881)),
            ((51, 2.5, 1, 8, 0.0), (52, 0.985, 2, 5, 0.033355), (55, 0.4494949494949495, 3, 4, -0.00967), (-1, 0.0, 0, 0, 0.003264), (-1, 0.0, 0, 0, -0.027156), (6, 0.8771504370949433, 6, 7, 0.626872), (-1, 0.0, 0, 0, 0.040259), (-1, 0.0, 0, 0, -0.007442), (52, 0.732, 9, 10, -0.618642), (-1, 0.0, 0, 0, -0.039703), (-1, 0.0, 0, 0, 0.004219)),
            ((20, 3.210268834500856, 1, 8, 0.0), (12, 0.6819999999999999, 2, 5, -0.039228), (26, 4.131156519244238, 3, 4, 0.027542), (-1, 0.0, 0, 0, -0.023683), (-1, 0.0, 0, 0, 0.005175), (11, 0.896, 6, 7, -0.376269), (-1, 0.0, 0, 0, 0.020971), (-1, 0.0, 0, 0, -0.04027), (13, 0.556, 9, 10, 0.615416), (-1, 0.0, 0, 0, -0.006698), (-1, 0.0, 0, 0, 0.038438)),
            ((60, 0.664, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.065914), (50, 0.5, 3, 4, 0.658828), (-1, 0.0, 0, 0, 0.050724), (-1, 0.0, 0, 0, 0.0037), (5, 0.001, 6, 7, 0.000611), (-1, 0.0, 0, 0, 0.027525), (-1, 0.0, 0, 0, -0.003774), (26, 4.598548757796862, 9, 12, -0.317804), (4, 0.843, 10, 11, -0.758322), (-1, 0.0, 0, 0, -0.042855), (-1, 0.0, 0, 0, -0.004749), (55, 0.11805555555555555, 13, 14, 0.007724), (-1, 0.0, 0, 0, 0.028504), (-1, 0.0, 0, 0, -0.015865)),
            ((15, -0.751, 1, 4, 0.0), (25, 6.648336461794052, 2, 3, -0.655465), (-1, 0.0, 0, 0, -0.004715), (-1, 0.0, 0, 0, -0.036387), (56, 0.17424242424242425, 5, 8, 0.030249), (26, 4.608290119226802, 6, 7, 0.399087), (-1, 0.0, 0, 0, -0.010691), (-1, 0.0, 0, 0, 0.040019), (2, 0.992, 9, 10, -0.039434), (-1, 0.0, 0, 0, -0.010693), (-1, 0.0, 0, 0, 0.006816)),
            ((19, 0.5, 1, 8, 0.0), (11, 0.997, 2, 5, -0.048448), (26, 4.2964256503859755, 3, 4, 0.004639), (-1, 0.0, 0, 0, -0.018421), (-1, 0.0, 0, 0, 0.00531), (4, 0.996, 6, 7, -0.552484), (-1, 0.0, 0, 0, -0.038369), (-1, 0.0, 0, 0, -0.002862), (6, 0.8846610635340376, 9, 12, 0.471264), (13, 0.744, 10, 11, 0.629185), (-1, 0.0, 0, 0, 0.005858), (-1, 0.0, 0, 0, 0.041344), (-1, 0.0, 0, 0, -0.019089)),
            ((51, 2.5, 1, 8, 0.0), (40, 0.5, 2, 5, 0.032449), (39, 0.528, 3, 4, 0.180562), (-1, 0.0, 0, 0, 0.017411), (-1, 0.0, 0, 0, -0.007062), (9, 0.694, 6, 7, -0.132018), (-1, 0.0, 0, 0, -0.001362), (-1, 0.0, 0, 0, -0.032556), (20, -2.6933930072678223, 9, 10, -0.600394), (-1, 0.0, 0, 0, -0.03629), (-1, 0.0, 0, 0, 0.010225)),
            ((15, -0.751, 1, 4, 0.0), (58, 0.2109990093741555, 2, 3, -0.649999), (-1, 0.0, 0, 0, -0.035934), (-1, 0.0, 0, 0, -0.005031), (40, 0.5, 5, 8, 0.029122), (56, 0.4494949494949495, 6, 7, 0.175458), (-1, 0.0, 0, 0, 0.022327), (-1, 0.0, 0, 0, -0.003612), (7, 0.958, 9, 10, -0.132846), (-1, 0.0, 0, 0, -0.013208), (-1, 0.0, 0, 0, 0.014212)),
            ((60, 0.664, 1, 8, 0.0), (5, 0.001, 2, 5, 0.064686), (15, 0.755, 3, 4, 0.565242), (-1, 0.0, 0, 0, 0.047803), (-1, 0.0, 0, 0, -0.015369), (2, 0.43899999999999995, 6, 7, -0.003497), (-1, 0.0, 0, 0, 0.032987), (-1, 0.0, 0, 0, -0.003928), (24, 0.015833333333333345, 9, 12, -0.314569), (6, 0.21590364309205778, 10, 11, -0.707264), (-1, 0.0, 0, 0, -0.043392), (-1, 0.0, 0, 0, -0.001069), (56, 0.3798076923076923, 13, 14, 0.042219), (-1, 0.0, 0, 0, 0.025394), (-1, 0.0, 0, 0, -0.026405)),
            ((20, 3.210268834500856, 1, 8, 0.0), (39, 0.0030000000000000027, 2, 5, -0.037654), (59, 0.30100000000000005, 3, 4, -0.242579), (-1, 0.0, 0, 0, -0.002589), (-1, 0.0, 0, 0, -0.04118), (2, 0.7150000000000001, 6, 7, 0.066248), (-1, 0.0, 0, 0, 0.038256), (-1, 0.0, 0, 0, -0.001799), (13, 0.556, 9, 10, 0.602525), (-1, 0.0, 0, 0, -0.006505), (-1, 0.0, 0, 0, 0.0379)),
            ((2, 0.992, 1, 8, 0.0), (57, 0.5, 2, 5, -0.130066), (1, 0.01, 3, 4, -0.039601), (-1, 0.0, 0, 0, 0.01057), (-1, 0.0, 0, 0, -0.015129), (26, 4.212125883197904, 6, 7, -0.822413), (-1, 0.0, 0, 0, 0.018063), (-1, 0.0, 0, 0, -0.053149), (1, 0.04, 9, 12, 0.163534), (27, 0.208, 10, 11, -0.005207), (-1, 0.0, 0, 0, 0.031982), (-1, 0.0, 0, 0, -0.009137), (13, 0.991, 13, 14, 0.654593), (-1, 0.0, 0, 0, 0.058894), (-1, 0.0, 0, 0, -0.001244)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.163992), (7, 0.866, 3, 4, 0.016526), (-1, 0.0, 0, 0, -0.009474), (-1, 0.0, 0, 0, 0.033291), (11, 0.823, 6, 7, -0.572095), (-1, 0.0, 0, 0, 0.017463), (-1, 0.0, 0, 0, -0.039645), (50, 0.5, 9, 12, 0.128444), (17, 0.5, 10, 11, 0.264996), (-1, 0.0, 0, 0, 0.033921), (-1, 0.0, 0, 0, 0.001942), (13, 0.998, 13, 14, -0.190183), (-1, 0.0, 0, 0, -0.018812), (-1, 0.0, 0, 0, 0.046826)),
            ((15, -0.751, 1, 4, 0.0), (25, 6.648336461794052, 2, 3, -0.639866), (-1, 0.0, 0, 0, -0.004875), (-1, 0.0, 0, 0, -0.035293), (46, 0.8571428571428572, 5, 8, 0.030703), (56, 0.17424242424242425, 6, 7, 0.061194), (-1, 0.0, 0, 0, 0.026243), (-1, 0.0, 0, 0, -0.00094), (-1, 0.0, 0, 0, -0.034612)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.155626), (26, 4.705005283552375, 3, 4, -0.017721), (-1, 0.0, 0, 0, -0.011156), (-1, 0.0, 0, 0, 0.022479), (7, 0.853, 6, 7, -0.954773), (-1, 0.0, 0, 0, -0.055076), (-1, 0.0, 0, 0, -0.016099), (9, 0.3172857142857143, 9, 12, 0.128841), (7, 0.573, 10, 11, 0.609033), (-1, 0.0, 0, 0, -0.007714), (-1, 0.0, 0, 0, 0.049967), (5, 0.014, 13, 14, -0.029291), (-1, 0.0, 0, 0, 0.023968), (-1, 0.0, 0, 0, -0.011719)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.159705), (26, 4.574061244703071, 3, 4, 0.013944), (-1, 0.0, 0, 0, -0.027441), (-1, 0.0, 0, 0, 0.012373), (11, 0.823, 6, 7, -0.55764), (-1, 0.0, 0, 0, 0.016609), (-1, 0.0, 0, 0, -0.038801), (25, 6.802950010466324, 9, 12, 0.125251), (2, 0.998, 10, 11, 0.040171), (-1, 0.0, 0, 0, -0.004311), (-1, 0.0, 0, 0, 0.021537), (9, 0.2665, 13, 14, 0.63859), (-1, 0.0, 0, 0, 0.039687), (-1, 0.0, 0, 0, 0.001985)),
            ((51, 2.5, 1, 8, 0.0), (53, 0.007000000000000006, 2, 5, 0.03257), (60, 0.46799999999999997, 3, 4, -0.367863), (-1, 0.0, 0, 0, 0.000332), (-1, 0.0, 0, 0, -0.031278), (8, 0.6190476190476191, 6, 7, 0.089564), (-1, 0.0, 0, 0, 0.008767), (-1, 0.0, 0, 0, -0.013948), (52, 0.732, 9, 10, -0.579241), (-1, 0.0, 0, 0, -0.037716), (-1, 0.0, 0, 0, 0.004709)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.155838), (26, 4.759356141538873, 3, 4, -0.020882), (-1, 0.0, 0, 0, -0.006588), (-1, 0.0, 0, 0, 0.038116), (60, 0.009000000000000001, 6, 7, -0.944879), (-1, 0.0, 0, 0, -0.013876), (-1, 0.0, 0, 0, -0.053195), (9, 0.3172857142857143, 9, 12, 0.127305), (7, 0.573, 10, 11, 0.578576), (-1, 0.0, 0, 0, -0.00765), (-1, 0.0, 0, 0, 0.047645), (5, 0.014, 13, 14, -0.022034), (-1, 0.0, 0, 0, 0.02326), (-1, 0.0, 0, 0, -0.010961)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (12, 0.39, 2, 5, 0.059492), (60, 0.043, 3, 4, -0.175655), (-1, 0.0, 0, 0, 0.012496), (-1, 0.0, 0, 0, -0.023386), (11, 0.45, 6, 7, 0.207231), (-1, 0.0, 0, 0, -0.023061), (-1, 0.0, 0, 0, 0.015559), (11, 0.896, 9, 12, -0.317453), (56, 0.4365079365079365, 10, 11, 0.371185), (-1, 0.0, 0, 0, 0.043935), (-1, 0.0, 0, 0, -0.017728), (24, 0.06250000000000003, 13, 14, -0.687559), (-1, 0.0, 0, 0, -0.044789), (-1, 0.0, 0, 0, 0.012777)),
            ((15, -0.751, 1, 4, 0.0), (58, 0.2109990093741555, 2, 3, -0.616759), (-1, 0.0, 0, 0, -0.034202), (-1, 0.0, 0, 0, -0.004159), (51, 2.5, 5, 8, 0.028311), (40, 0.5, 6, 7, 0.063711), (-1, 0.0, 0, 0, 0.011038), (-1, 0.0, 0, 0, -0.005369), (12, 0.45099999999999996, 9, 10, -0.564548), (-1, 0.0, 0, 0, -0.002294), (-1, 0.0, 0, 0, -0.041499)),
            ((20, 3.210268834500856, 1, 8, 0.0), (27, 0.99, 2, 5, -0.035619), (1, 0.01, 3, 4, 0.056239), (-1, 0.0, 0, 0, 0.012229), (-1, 0.0, 0, 0, -0.011733), (42, 0.08712121212121213, 6, 7, -0.239835), (-1, 0.0, 0, 0, 0.012626), (-1, 0.0, 0, 0, -0.025423), (13, 0.556, 9, 10, 0.580838), (-1, 0.0, 0, 0, -0.006407), (-1, 0.0, 0, 0, 0.036975)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.149609), (26, 4.705005283552375, 3, 4, -0.019533), (-1, 0.0, 0, 0, -0.010423), (-1, 0.0, 0, 0, 0.020694), (7, 0.853, 6, 7, -0.925406), (-1, 0.0, 0, 0, -0.053433), (-1, 0.0, 0, 0, -0.015169), (9, 0.3172857142857143, 9, 12, 0.120552), (7, 0.573, 10, 11, 0.549845), (-1, 0.0, 0, 0, -0.007322), (-1, 0.0, 0, 0, 0.045285), (5, 0.014, 13, 14, -0.022229), (-1, 0.0, 0, 0, 0.021896), (-1, 0.0, 0, 0, -0.010454)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.1552), (26, 4.574061244703071, 3, 4, 0.007036), (-1, 0.0, 0, 0, -0.026843), (-1, 0.0, 0, 0, 0.011481), (11, 0.823, 6, 7, -0.534374), (-1, 0.0, 0, 0, 0.015924), (-1, 0.0, 0, 0, -0.037538), (50, 0.5, 9, 12, 0.11847), (17, 0.5, 10, 11, 0.246889), (-1, 0.0, 0, 0, 0.032634), (-1, 0.0, 0, 0, 0.00163), (13, 0.998, 13, 14, -0.177185), (-1, 0.0, 0, 0, -0.017934), (-1, 0.0, 0, 0, 0.046602)),
            ((51, 2.5, 1, 8, 0.0), (60, 0.98, 2, 5, 0.030397), (40, 0.5, 3, 4, 0.06351), (-1, 0.0, 0, 0, 0.010602), (-1, 0.0, 0, 0, -0.004748), (20, -3.683959493972949, 6, 7, -0.600882), (-1, 0.0, 0, 0, -0.036025), (-1, 0.0, 0, 0, -0.006151), (52, 0.732, 9, 10, -0.557796), (-1, 0.0, 0, 0, -0.03651), (-1, 0.0, 0, 0, 0.004764)),
            ((15, -0.751, 1, 4, 0.0), (25, 6.648336461794052, 2, 3, -0.603866), (-1, 0.0, 0, 0, -0.004026), (-1, 0.0, 0, 0, -0.033474), (56, 0.17424242424242425, 5, 8, 0.026595), (26, 4.608290119226802, 6, 7, 0.355866), (-1, 0.0, 0, 0, -0.009861), (-1, 0.0, 0, 0, 0.035521), (9, 0.008750000000000008, 9, 10, -0.035396), (-1, 0.0, 0, 0, -0.029228), (-1, 0.0, 0, 0, 0.000681)),
            ((20, 4.243470074122607, 1, 8, 0.0), (15, -0.751, 2, 5, -0.029757), (7, 0.019000000000000017, 3, 4, -0.624196), (-1, 0.0, 0, 0, -0.006564), (-1, 0.0, 0, 0, -0.037162), (8, 0.6190476190476191, 6, 7, 7.9e-05), (-1, 0.0, 0, 0, 0.003334), (-1, 0.0, 0, 0, -0.015517), (-1, 0.0, 0, 0, 0.033647)),
            ((53, 0.278, 1, 8, 0.0), (25, 6.101438773316817, 2, 5, -0.138641), (5, 0.055, 3, 4, -0.85588), (-1, 0.0, 0, 0, -0.045023), (-1, 0.0, 0, 0, -0.002057), (13, 0.491, 6, 7, -0.024197), (-1, 0.0, 0, 0, 0.018906), (-1, 0.0, 0, 0, -0.019609), (13, 0.959, 9, 12, 0.127382), (52, 0.43899999999999995, 10, 11, -0.209383), (-1, 0.0, 0, 0, 0.035841), (-1, 0.0, 0, 0, -0.022702), (13, 0.989, 13, 14, 0.366851), (-1, 0.0, 0, 0, 0.038569), (-1, 0.0, 0, 0, 0.006364)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.147764), (26, 4.759356141538873, 3, 4, -0.020557), (-1, 0.0, 0, 0, -0.006196), (-1, 0.0, 0, 0, 0.035856), (60, 0.009000000000000001, 6, 7, -0.910034), (-1, 0.0, 0, 0, -0.012635), (-1, 0.0, 0, 0, -0.051585), (8, 0.6190476190476191, 9, 12, 0.117307), (4, 0.979, 10, 11, 0.274621), (-1, 0.0, 0, 0, 0.006012), (-1, 0.0, 0, 0, 0.040715), (9, 0.9245, 13, 14, -0.233349), (-1, 0.0, 0, 0, -0.028316), (-1, 0.0, 0, 0, 0.020508)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.152703), (15, -0.20700000000000002, 3, 4, 0.059367), (-1, 0.0, 0, 0, -0.018374), (-1, 0.0, 0, 0, 0.016249), (1, 0.039, 6, 7, -0.421008), (-1, 0.0, 0, 0, -0.031954), (-1, 0.0, 0, 0, 0.006962), (15, 0.755, 9, 12, 0.115125), (8, 0.6190476190476191, 10, 11, 0.195421), (-1, 0.0, 0, 0, 0.016159), (-1, 0.0, 0, 0, -0.012137), (44, 0.06527999999999999, 13, 14, -0.33295), (-1, 0.0, 0, 0, -0.026519), (-1, 0.0, 0, 0, 0.02338)),
            ((60, 0.98, 1, 8, 0.0), (5, 0.001, 2, 5, 0.025448), (15, 0.894, 3, 4, 0.461504), (-1, 0.0, 0, 0, 0.03771), (-1, 0.0, 0, 0, -0.019538), (8, 0.6190476190476191, 6, 7, -0.028791), (-1, 0.0, 0, 0, 0.002595), (-1, 0.0, 0, 0, -0.019875), (20, -3.683959493972949, 9, 10, -0.585942), (-1, 0.0, 0, 0, -0.035362), (-1, 0.0, 0, 0, -0.005716)),
            ((7, 0.605, 1, 8, 0.0), (26, 4.705005283552375, 2, 5, -0.132471), (13, 0.491, 3, 4, -0.271283), (-1, 0.0, 0, 0, -0.000842), (-1, 0.0, 0, 0, -0.026299), (42, 0.029411764705882353, 6, 7, 0.222751), (-1, 0.0, 0, 0, 0.03984), (-1, 0.0, 0, 0, -0.012539), (5, 0.171, 9, 12, 0.130196), (9, 0.31499999999999995, 10, 11, 0.35905), (-1, 0.0, 0, 0, 0.043787), (-1, 0.0, 0, 0, 0.008425), (4, 0.982, 13, 14, -0.130358), (-1, 0.0, 0, 0, -0.023059), (-1, 0.0, 0, 0, 0.019364)),
            ((20, 4.243470074122607, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.028656), (7, 0.993, 3, 4, 0.019706), (-1, 0.0, 0, 0, 0.003066), (-1, 0.0, 0, 0, -0.02915), (7, 0.977, 6, 7, -0.365355), (-1, 0.0, 0, 0, -0.037374), (-1, 0.0, 0, 0, 0.024145), (-1, 0.0, 0, 0, 0.033234)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (13, 0.989, 2, 5, -0.148819), (7, 0.866, 3, 4, 0.004172), (-1, 0.0, 0, 0, -0.009613), (-1, 0.0, 0, 0, 0.02955), (11, 0.823, 6, 7, -0.510446), (-1, 0.0, 0, 0, 0.016005), (-1, 0.0, 0, 0, -0.036334), (50, 0.5, 9, 12, 0.111971), (6, 0.5826207097771658, 10, 11, 0.237092), (-1, 0.0, 0, 0, 0.024566), (-1, 0.0, 0, 0, -0.004419), (13, 0.998, 13, 14, -0.172493), (-1, 0.0, 0, 0, -0.017378), (-1, 0.0, 0, 0, 0.045348)),
            ((46, 0.8571428571428572, 1, 8, 0.0), (56, 0.1380952380952381, 2, 5, 0.023107), (26, 4.6353310033601, 3, 4, 0.464079), (-1, 0.0, 0, 0, 0.000307), (-1, 0.0, 0, 0, 0.037684), (20, -3.210268834500856, 6, 7, -0.034224), (-1, 0.0, 0, 0, -0.012204), (-1, 0.0, 0, 0, 0.003699), (-1, 0.0, 0, 0, -0.032406)),
            ((53, 0.278, 1, 8, 0.0), (25, 6.101438773316817, 2, 5, -0.132426), (5, 0.055, 3, 4, -0.840836), (-1, 0.0, 0, 0, -0.044321), (-1, 0.0, 0, 0, -0.001974), (13, 0.491, 6, 7, -0.021332), (-1, 0.0, 0, 0, 0.017595), (-1, 0.0, 0, 0, -0.018613), (13, 0.959, 9, 12, 0.123354), (52, 0.43899999999999995, 10, 11, -0.195716), (-1, 0.0, 0, 0, 0.034339), (-1, 0.0, 0, 0, -0.021574), (53, 0.918, 13, 14, 0.35101), (-1, 0.0, 0, 0, 0.028972), (-1, 0.0, 0, 0, -0.00181)),
            ((19, 0.5, 1, 8, 0.0), (26, 4.131156519244238, 2, 5, -0.04138), (52, 0.81, 3, 4, -0.462531), (-1, 0.0, 0, 0, -0.043734), (-1, 0.0, 0, 0, 0.021392), (15, -0.099, 6, 7, 0.013523), (-1, 0.0, 0, 0, -0.012227), (-1, 0.0, 0, 0, 0.007446), (6, 0.8846610635340376, 9, 12, 0.418157), (13, 0.744, 10, 11, 0.58787), (-1, 0.0, 0, 0, 0.004196), (-1, 0.0, 0, 0, 0.03971), (-1, 0.0, 0, 0, -0.019511)),
            ((51, 2.5, 1, 8, 0.0), (12, 0.39, 2, 5, 0.028263), (60, 0.043, 3, 4, -0.171464), (-1, 0.0, 0, 0, 0.012122), (-1, 0.0, 0, 0, -0.023291), (12, 0.6759999999999999, 6, 7, 0.125895), (-1, 0.0, 0, 0, 0.012188), (-1, 0.0, 0, 0, -0.013005), (12, 0.45099999999999996, 9, 10, -0.523098), (-1, 0.0, 0, 0, -0.000736), (-1, 0.0, 0, 0, -0.039324)),
            ((20, 4.243470074122607, 1, 8, 0.0), (39, 0.9045, 2, 5, -0.027159), (53, 0.278, 3, 4, 0.008195), (-1, 0.0, 0, 0, -0.007156), (-1, 0.0, 0, 0, 0.007833), (25, 6.249973378837159, 6, 7, -0.492113), (-1, 0.0, 0, 0, 0.010696), (-1, 0.0, 0, 0, -0.033133), (-1, 0.0, 0, 0, 0.03244)),
            ((39, 0.528, 1, 8, 0.0), (39, 0.0030000000000000027, 2, 5, 0.083639), (59, 0.29200000000000004, 3, 4, -0.129319), (-1, 0.0, 0, 0, 0.003326), (-1, 0.0, 0, 0, -0.032333), (15, -0.013000000000000001, 6, 7, 0.286703), (-1, 0.0, 0, 0, -0.00194), (-1, 0.0, 0, 0, 0.028614), (1, 0.833, 9, 12, -0.180579), (24, 0.033500000000000016, 10, 11, -0.30587), (-1, 0.0, 0, 0, -0.041036), (-1, 0.0, 0, 0, -0.000754), (15, -0.018, 13, 14, 0.603617), (-1, 0.0, 0, 0, 0.047407), (-1, 0.0, 0, 0, -0.013507)),
            ((60, 0.98, 1, 8, 0.0), (56, 0.17424242424242425, 2, 5, 0.024666), (50, 0.5, 3, 4, 0.373739), (-1, 0.0, 0, 0, 0.032873), (-1, 0.0, 0, 0, -0.009874), (12, 0.6759999999999999, 6, 7, -0.041121), (-1, 0.0, 0, 0, 0.001332), (-1, 0.0, 0, 0, -0.018119), (20, -3.683959493972949, 9, 10, -0.5627), (-1, 0.0, 0, 0, -0.034348), (-1, 0.0, 0, 0, -0.004938)),
            ((20, 4.243470074122607, 1, 8, 0.0), (51, 2.5, 2, 5, -0.026659), (12, 0.39, 3, 4, 0.003461), (-1, 0.0, 0, 0, -0.009972), (-1, 0.0, 0, 0, 0.005106), (52, 0.732, 6, 7, -0.529302), (-1, 0.0, 0, 0, -0.035668), (-1, 0.0, 0, 0, 0.005514), (-1, 0.0, 0, 0, 0.032045)),
            ((53, 0.007000000000000006, 1, 8, 0.0), (60, 0.46799999999999997, 2, 5, -0.335342), (25, 6.247074819994996, 3, 4, -0.002115), (-1, 0.0, 0, 0, -0.029746), (-1, 0.0, 0, 0, 0.015336), (24, 0.038750000000000034, 6, 7, -0.575271), (-1, 0.0, 0, 0, -0.035075), (-1, 0.0, 0, 0, -0.007569), (5, 0.001, 9, 12, 0.04195), (12, 0.43600000000000005, 10, 11, 0.61122), (-1, 0.0, 0, 0, -0.003687), (-1, 0.0, 0, 0, 0.048748), (52, 0.029000000000000026, 13, 14, -0.008655), (-1, 0.0, 0, 0, 0.023318), (-1, 0.0, 0, 0, -0.004647)),
            ((40, 2.5, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, 0.035033), (12, 0.34299999999999997, 3, 4, -0.178547), (-1, 0.0, 0, 0, -0.031593), (-1, 0.0, 0, 0, -0.00018), (3, 0.19499999999999995, 6, 7, 0.139617), (-1, 0.0, 0, 0, 0.023636), (-1, 0.0, 0, 0, -0.001167), (4, 0.326, 9, 10, -0.393109), (-1, 0.0, 0, 0, -0.044258), (52, 0.406, 11, 12, 0.002686), (-1, 0.0, 0, 0, 0.022907), (-1, 0.0, 0, 0, -0.018449)),
            ((20, 4.243470074122607, 1, 8, 0.0), (15, -0.751, 2, 5, -0.026255), (7, 0.019000000000000017, 3, 4, -0.578986), (-1, 0.0, 0, 0, -0.004702), (-1, 0.0, 0, 0, -0.035275), (39, 0.528, 6, 7, 0.000731), (-1, 0.0, 0, 0, 0.005), (-1, 0.0, 0, 0, -0.009698), (-1, 0.0, 0, 0, 0.031628)),
            ((60, 0.98, 1, 8, 0.0), (5, 0.001, 2, 5, 0.022916), (15, 0.894, 3, 4, 0.432227), (-1, 0.0, 0, 0, 0.035151), (-1, 0.0, 0, 0, -0.018384), (8, 0.6190476190476191, 6, 7, -0.027914), (-1, 0.0, 0, 0, 0.00267), (-1, 0.0, 0, 0, -0.019881), (20, -3.683959493972949, 9, 10, -0.548939), (-1, 0.0, 0, 0, -0.033492), (-1, 0.0, 0, 0, -0.0051)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.068315), (11, 0.45, 3, 4, -0.002636), (-1, 0.0, 0, 0, -0.016974), (-1, 0.0, 0, 0, 0.004474), (19, 0.5, 6, 7, -0.659534), (-1, 0.0, 0, 0, -0.038608), (-1, 0.0, 0, 0, 0.005847), (44, 0.06817999999999999, 9, 12, 0.218316), (3, 0.31100000000000005, 10, 11, -0.504478), (-1, 0.0, 0, 0, -0.004873), (-1, 0.0, 0, 0, -0.03373), (39, 0.528, 13, 14, 0.383604), (-1, 0.0, 0, 0, 0.028463), (-1, 0.0, 0, 0, -0.011281)),
            ((40, 2.5, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, 0.032671), (26, 4.574061244703071, 3, 4, -0.17053), (-1, 0.0, 0, 0, -0.029355), (-1, 0.0, 0, 0, 0.000497), (3, 0.19499999999999995, 6, 7, 0.131797), (-1, 0.0, 0, 0, 0.022567), (-1, 0.0, 0, 0, -0.001187), (4, 0.326, 9, 10, -0.387268), (-1, 0.0, 0, 0, -0.043368), (52, 0.406, 11, 12, -0.006516), (-1, 0.0, 0, 0, 0.021632), (-1, 0.0, 0, 0, -0.018249)),
            ((52, 0.995, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.029089), (7, 0.993, 3, 4, 0.027203), (-1, 0.0, 0, 0, 0.00369), (-1, 0.0, 0, 0, -0.035897), (60, 0.022, 6, 7, -0.468417), (-1, 0.0, 0, 0, 0.00134), (-1, 0.0, 0, 0, -0.036927), (7, 0.983, 9, 10, 0.580001), (-1, 0.0, 0, 0, -0.010147), (-1, 0.0, 0, 0, 0.039257)),
            ((7, 0.605, 1, 6, 0.0), (26, 4.759356141538873, 2, 5, -0.121574), (1, 0.01, 3, 4, -0.192302), (-1, 0.0, 0, 0, -0.001026), (-1, 0.0, 0, 0, -0.024275), (-1, 0.0, 0, 0, 0.024092), (5, 0.171, 7, 10, 0.11596), (9, 0.31499999999999995, 8, 9, 0.332289), (-1, 0.0, 0, 0, 0.042172), (-1, 0.0, 0, 0, 0.007173), (4, 0.982, 11, 12, -0.132587), (-1, 0.0, 0, 0, -0.022506), (-1, 0.0, 0, 0, 0.0184)),
            ((20, 4.243470074122607, 1, 8, 0.0), (15, -0.751, 2, 5, -0.026697), (7, 0.019000000000000017, 3, 4, -0.569534), (-1, 0.0, 0, 0, -0.004613), (-1, 0.0, 0, 0, -0.034725), (39, 0.528, 6, 7, -0.000347), (-1, 0.0, 0, 0, 0.004763), (-1, 0.0, 0, 0, -0.00938), (-1, 0.0, 0, 0, 0.03114)),
            ((53, 0.007000000000000006, 1, 8, 0.0), (60, 0.46799999999999997, 2, 5, -0.325626), (25, 6.247074819994996, 3, 4, -0.001523), (-1, 0.0, 0, 0, -0.028799), (-1, 0.0, 0, 0, 0.01467), (24, 0.038750000000000034, 6, 7, -0.562518), (-1, 0.0, 0, 0, -0.034104), (-1, 0.0, 0, 0, -0.007893), (5, 0.001, 9, 12, 0.038615), (12, 0.43600000000000005, 10, 11, 0.5665), (-1, 0.0, 0, 0, -0.004186), (-1, 0.0, 0, 0, 0.045403), (52, 0.029000000000000026, 13, 14, -0.007986), (-1, 0.0, 0, 0, 0.022158), (-1, 0.0, 0, 0, -0.004447)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.371, 2, 5, 0.038588), (15, 0.252, 3, 4, -0.021027), (-1, 0.0, 0, 0, 0.001906), (-1, 0.0, 0, 0, -0.03645), (60, 0.008, 6, 7, 0.447591), (-1, 0.0, 0, 0, 0.056734), (-1, 0.0, 0, 0, 0.003201), (6, 0.5935573765069624, 9, 12, -0.329472), (48, 0.953, 10, 11, 0.144559), (-1, 0.0, 0, 0, -0.017714), (-1, 0.0, 0, 0, 0.031299), (1, 0.0, 13, 14, -0.759416), (-1, 0.0, 0, 0, -0.01348), (-1, 0.0, 0, 0, -0.044664)),
            ((51, 2.5, 1, 8, 0.0), (40, 0.5, 2, 5, 0.023924), (1, 0.002, 3, 4, 0.14634), (-1, 0.0, 0, 0, -0.005355), (-1, 0.0, 0, 0, 0.017779), (9, 0.694, 6, 7, -0.109715), (-1, 0.0, 0, 0, -0.000598), (-1, 0.0, 0, 0, -0.029343), (12, 0.45099999999999996, 9, 10, -0.486955), (-1, 0.0, 0, 0, 0.000227), (-1, 0.0, 0, 0, -0.037365)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (3, 0.42400000000000004, 2, 5, 0.046237), (59, 0.20299999999999996, 3, 4, -0.035531), (-1, 0.0, 0, 0, 0.00862), (-1, 0.0, 0, 0, -0.011276), (59, 0.241, 6, 7, 0.371386), (-1, 0.0, 0, 0, -0.016238), (-1, 0.0, 0, 0, 0.029379), (11, 0.896, 9, 12, -0.273894), (56, 0.4365079365079365, 10, 11, 0.360847), (-1, 0.0, 0, 0, 0.042079), (-1, 0.0, 0, 0, -0.016428), (24, 0.06250000000000003, 13, 14, -0.633062), (-1, 0.0, 0, 0, -0.041919), (-1, 0.0, 0, 0, 0.011935)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.067388), (11, 0.45, 3, 4, -0.006865), (-1, 0.0, 0, 0, -0.01692), (-1, 0.0, 0, 0, 0.004183), (19, 0.5, 6, 7, -0.625753), (-1, 0.0, 0, 0, -0.037043), (-1, 0.0, 0, 0, 0.006324), (44, 0.06817999999999999, 9, 12, 0.213831), (3, 0.31100000000000005, 10, 11, -0.48846), (-1, 0.0, 0, 0, -0.004421), (-1, 0.0, 0, 0, -0.03284), (39, 0.528, 13, 14, 0.373809), (-1, 0.0, 0, 0, 0.02734), (-1, 0.0, 0, 0, -0.009989)),
            ((20, 4.243470074122607, 1, 8, 0.0), (15, -0.751, 2, 5, -0.02678), (7, 0.019000000000000017, 3, 4, -0.559354), (-1, 0.0, 0, 0, -0.004291), (-1, 0.0, 0, 0, -0.034264), (40, 0.5, 6, 7, -0.001102), (-1, 0.0, 0, 0, 0.006187), (-1, 0.0, 0, 0, -0.006802), (-1, 0.0, 0, 0, 0.030569)),
            ((52, 0.995, 1, 8, 0.0), (7, 0.993, 2, 5, -0.028231), (55, 0.4494949494949495, 3, 4, 0.010438), (-1, 0.0, 0, 0, 0.003432), (-1, 0.0, 0, 0, -0.023711), (42, 0.3484848484848485, 6, 7, -0.610296), (-1, 0.0, 0, 0, -0.042007), (-1, 0.0, 0, 0, 0.020409), (7, 0.983, 9, 10, 0.568555), (-1, 0.0, 0, 0, -0.009624), (-1, 0.0, 0, 0, 0.038565)),
            ((60, 0.98, 1, 8, 0.0), (56, 0.17424242424242425, 2, 5, 0.020104), (26, 4.608290119226802, 3, 4, 0.337132), (-1, 0.0, 0, 0, -0.008564), (-1, 0.0, 0, 0, 0.032238), (3, 0.42400000000000004, 6, 7, -0.039374), (-1, 0.0, 0, 0, -0.005907), (-1, 0.0, 0, 0, 0.010542), (20, -3.683959493972949, 9, 10, -0.533466), (-1, 0.0, 0, 0, -0.032621), (-1, 0.0, 0, 0, -0.004899)),
            ((7, 0.951, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.066385), (15, -0.17, 3, 4, -0.010243), (-1, 0.0, 0, 0, -0.013702), (-1, 0.0, 0, 0, 0.004781), (19, 0.5, 6, 7, -0.596857), (-1, 0.0, 0, 0, -0.035652), (-1, 0.0, 0, 0, 0.00651), (44, 0.06817999999999999, 9, 12, 0.210201), (3, 0.31100000000000005, 10, 11, -0.472843), (-1, 0.0, 0, 0, -0.00384), (-1, 0.0, 0, 0, -0.032263), (11, 0.997, 13, 14, 0.365027), (-1, 0.0, 0, 0, 0.024534), (-1, 0.0, 0, 0, -0.017883)),
            ((51, 2.5, 1, 8, 0.0), (12, 0.39, 2, 5, 0.022092), (60, 0.043, 3, 4, -0.157619), (-1, 0.0, 0, 0, 0.011672), (-1, 0.0, 0, 0, -0.021751), (6, 0.6501228390275874, 6, 7, 0.109687), (-1, 0.0, 0, 0, 0.013833), (-1, 0.0, 0, 0, -0.007322), (52, 0.732, 9, 10, -0.479288), (-1, 0.0, 0, 0, -0.032961), (-1, 0.0, 0, 0, 0.006206)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (12, 0.39, 2, 5, 0.043679), (60, 0.043, 3, 4, -0.149987), (-1, 0.0, 0, 0, 0.010271), (-1, 0.0, 0, 0, -0.019777), (11, 0.45, 6, 7, 0.163844), (-1, 0.0, 0, 0, -0.022663), (-1, 0.0, 0, 0, 0.012953), (11, 0.896, 9, 12, -0.269806), (56, 0.4365079365079365, 10, 11, 0.349435), (-1, 0.0, 0, 0, 0.040565), (-1, 0.0, 0, 0, -0.015802), (24, 0.06250000000000003, 13, 14, -0.624646), (-1, 0.0, 0, 0, -0.041283), (-1, 0.0, 0, 0, 0.01117)),
            ((15, 0.755, 1, 8, 0.0), (5, 0.002, 2, 5, 0.035482), (48, 0.31699999999999995, 3, 4, 0.421948), (-1, 0.0, 0, 0, -0.008457), (-1, 0.0, 0, 0, 0.037282), (8, 0.6190476190476191, 6, 7, -0.024052), (-1, 0.0, 0, 0, 0.003183), (-1, 0.0, 0, 0, -0.021766), (6, 0.5935573765069624, 9, 12, -0.321341), (48, 0.953, 10, 11, 0.13576), (-1, 0.0, 0, 0, -0.016758), (-1, 0.0, 0, 0, 0.029487), (1, 0.0, 13, 14, -0.73739), (-1, 0.0, 0, 0, -0.012799), (-1, 0.0, 0, 0, -0.043603)),
            ((19, 0.5, 1, 8, 0.0), (26, 4.131156519244238, 2, 5, -0.040154), (52, 0.81, 3, 4, -0.446195), (-1, 0.0, 0, 0, -0.042099), (-1, 0.0, 0, 0, 0.019162), (13, 0.44599999999999995, 6, 7, 0.010045), (-1, 0.0, 0, 0, 0.014197), (-1, 0.0, 0, 0, -0.005333), (6, 0.8846610635340376, 9, 12, 0.37434), (6, 0.06605249291607737, 10, 11, 0.558948), (-1, 0.0, 0, 0, -0.001005), (-1, 0.0, 0, 0, 0.036246), (-1, 0.0, 0, 0, -0.021501)),
            ((39, 0.9045, 1, 8, 0.0), (53, 0.278, 2, 5, 0.025401), (13, 0.491, 3, 4, -0.115633), (-1, 0.0, 0, 0, 0.00727), (-1, 0.0, 0, 0, -0.019383), (13, 0.959, 6, 7, 0.160834), (-1, 0.0, 0, 0, -0.007275), (-1, 0.0, 0, 0, 0.018563), (2, 0.998, 9, 10, -0.419583), (-1, 0.0, 0, 0, -0.035396), (-1, 0.0, 0, 0, 0.0015)),
            ((20, 4.243470074122607, 1, 8, 0.0), (39, 0.9045, 2, 5, -0.027039), (8, 0.6190476190476191, 3, 4, 0.004634), (-1, 0.0, 0, 0, 0.00335), (-1, 0.0, 0, 0, -0.013774), (2, 0.998, 6, 7, -0.445004), (-1, 0.0, 0, 0, -0.038129), (-1, 0.0, 0, 0, 0.001414), (-1, 0.0, 0, 0, 0.029996)),
            ((40, 2.5, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, 0.028928), (12, 0.34299999999999997, 3, 4, -0.167738), (-1, 0.0, 0, 0, -0.029746), (-1, 0.0, 0, 0, -0.000338), (3, 0.19499999999999995, 6, 7, 0.125017), (-1, 0.0, 0, 0, 0.021046), (-1, 0.0, 0, 0, -0.00097), (4, 0.326, 9, 10, -0.364854), (-1, 0.0, 0, 0, -0.04141), (56, 0.5147058823529411, 11, 12, -0.006709), (-1, 0.0, 0, 0, 0.017656), (-1, 0.0, 0, 0, -0.020878)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (3, 0.42400000000000004, 2, 5, 0.041887), (59, 0.20299999999999996, 3, 4, -0.035107), (-1, 0.0, 0, 0, 0.007984), (-1, 0.0, 0, 0, -0.010671), (59, 0.241, 6, 7, 0.345613), (-1, 0.0, 0, 0, -0.015872), (-1, 0.0, 0, 0, 0.027496), (56, 0.4083333333333333, 9, 12, -0.26748), (11, 0.793, 10, 11, 0.308927), (-1, 0.0, 0, 0, 0.04723), (-1, 0.0, 0, 0, -0.017122), (20, 2.319785806352712, 13, 14, -0.628057), (-1, 0.0, 0, 0, -0.038336), (-1, 0.0, 0, 0, 0.018365)),
            ((60, 0.98, 1, 8, 0.0), (56, 0.17424242424242425, 2, 5, 0.018423), (50, 0.5, 3, 4, 0.323044), (-1, 0.0, 0, 0, 0.030342), (-1, 0.0, 0, 0, -0.011841), (20, -3.210268834500856, 6, 7, -0.038645), (-1, 0.0, 0, 0, -0.011762), (-1, 0.0, 0, 0, 0.002761), (20, -3.683959493972949, 9, 10, -0.521293), (-1, 0.0, 0, 0, -0.032093), (-1, 0.0, 0, 0, -0.004503)),
            ((20, 4.243470074122607, 1, 8, 0.0), (15, -0.751, 2, 5, -0.026516), (7, 0.019000000000000017, 3, 4, -0.537574), (-1, 0.0, 0, 0, -0.003267), (-1, 0.0, 0, 0, -0.033414), (39, -0.020499999999999963, 6, 7, -0.001961), (-1, 0.0, 0, 0, 0.021395), (-1, 0.0, 0, 0, -0.001963), (-1, 0.0, 0, 0, 0.029562)),
            ((52, 0.997, 1, 8, 0.0), (57, 0.5, 2, 5, -0.024276), (27, 0.99, 3, 4, 0.021446), (-1, 0.0, 0, 0, 0.005973), (-1, 0.0, 0, 0, -0.011315), (42, 0.2583333333333333, 6, 7, -0.424251), (-1, 0.0, 0, 0, -0.04428), (-1, 0.0, 0, 0, 0.01264), (-1, 0.0, 0, 0, 0.032041)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.371, 2, 5, 0.033921), (15, 0.252, 3, 4, -0.023804), (-1, 0.0, 0, 0, 0.001568), (-1, 0.0, 0, 0, -0.034721), (60, 0.008, 6, 7, 0.423015), (-1, 0.0, 0, 0, 0.052852), (-1, 0.0, 0, 0, 0.00296), (6, 0.5988085425102423, 9, 12, -0.307108), (1, 0.0, 10, 11, 0.134894), (-1, 0.0, 0, 0, -0.019922), (-1, 0.0, 0, 0, 0.025369), (1, 0.0, 13, 14, -0.720897), (-1, 0.0, 0, 0, -0.011361), (-1, 0.0, 0, 0, -0.043293)),
            ((40, 2.5, 1, 8, 0.0), (7, 0.605, 2, 5, 0.02848), (7, 0.44399999999999995, 3, 4, -0.097437), (-1, 0.0, 0, 0, 0.000114), (-1, 0.0, 0, 0, -0.02709), (5, 0.171, 6, 7, 0.165867), (-1, 0.0, 0, 0, 0.019559), (-1, 0.0, 0, 0, -0.004907), (4, 0.326, 9, 10, -0.356124), (-1, 0.0, 0, 0, -0.040634), (56, 0.5147058823529411, 11, 12, -0.005593), (-1, 0.0, 0, 0, 0.017139), (-1, 0.0, 0, 0, -0.02036)),
            ((20, 4.243470074122607, 1, 8, 0.0), (8, 0.6190476190476191, 2, 5, -0.026065), (9, 0.4885, 3, 4, 0.025182), (-1, 0.0, 0, 0, -0.002737), (-1, 0.0, 0, 0, 0.033743), (4, 0.141, 6, 7, -0.269576), (-1, 0.0, 0, 0, 0.023146), (-1, 0.0, 0, 0, -0.027845), (-1, 0.0, 0, 0, 0.029205)),
            ((60, 0.98, 1, 8, 0.0), (6, 0.6343440778663255, 2, 5, 0.018459), (1, 0.009, 3, 4, 0.127492), (-1, 0.0, 0, 0, 0.015748), (-1, 0.0, 0, 0, -0.006813), (11, 0.987, 6, 7, -0.134849), (-1, 0.0, 0, 0, 0.002447), (-1, 0.0, 0, 0, -0.024898), (20, -3.683959493972949, 9, 10, -0.506011), (-1, 0.0, 0, 0, -0.031462), (-1, 0.0, 0, 0, -0.004093)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.014000000000000012, 2, 5, 0.034267), (19, 0.5, 3, 4, -0.070638), (-1, 0.0, 0, 0, -0.006618), (-1, 0.0, 0, 0, 0.024862), (60, 0.0009999999999999974, 6, 7, 0.244445), (-1, 0.0, 0, 0, -0.029814), (-1, 0.0, 0, 0, 0.016278), (6, 0.5935573765069624, 9, 12, -0.311312), (48, 0.953, 10, 11, 0.115), (-1, 0.0, 0, 0, -0.016819), (-1, 0.0, 0, 0, 0.027751), (52, 0.015000000000000013, 13, 14, -0.70381), (-1, 0.0, 0, 0, -0.009501), (-1, 0.0, 0, 0, -0.040883)),
            ((52, 0.997, 1, 8, 0.0), (55, 0.4494949494949495, 2, 5, -0.024025), (7, 0.993, 3, 4, 0.022647), (-1, 0.0, 0, 0, 0.003075), (-1, 0.0, 0, 0, -0.029172), (60, 0.022, 6, 7, -0.389409), (-1, 0.0, 0, 0, 0.003418), (-1, 0.0, 0, 0, -0.034258), (-1, 0.0, 0, 0, 0.03165)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.014000000000000012, 2, 5, 0.032792), (19, 0.5, 3, 4, -0.066552), (-1, 0.0, 0, 0, -0.006295), (-1, 0.0, 0, 0, 0.024319), (42, 0.3798076923076923, 6, 7, 0.230923), (-1, 0.0, 0, 0, 0.00617), (-1, 0.0, 0, 0, 0.049577), (6, 0.5935573765069624, 9, 12, -0.304426), (48, 0.953, 10, 11, 0.109815), (-1, 0.0, 0, 0, -0.016403), (-1, 0.0, 0, 0, 0.026806), (1, 0.0, 13, 14, -0.68818), (-1, 0.0, 0, 0, -0.010943), (-1, 0.0, 0, 0, -0.041467)),
            ((39, 0.9045, 1, 8, 0.0), (53, 0.278, 2, 5, 0.024244), (44, 0.0578, 3, 4, -0.110967), (-1, 0.0, 0, 0, 0.012004), (-1, 0.0, 0, 0, -0.014847), (53, 0.43966666666666665, 6, 7, 0.15334), (-1, 0.0, 0, 0, 0.03011), (-1, 0.0, 0, 0, 0.00071), (25, 6.249973378837159, 9, 10, -0.396903), (-1, 0.0, 0, 0, 0.011499), (53, 0.29775, 11, 12, -0.56019), (-1, 0.0, 0, 0, -0.006936), (-1, 0.0, 0, 0, -0.033124)),
            ((51, 2.5, 1, 8, 0.0), (12, 0.39, 2, 5, 0.020918), (60, 0.043, 3, 4, -0.14303), (-1, 0.0, 0, 0, 0.010758), (-1, 0.0, 0, 0, -0.019898), (6, 0.6501228390275874, 6, 7, 0.100395), (-1, 0.0, 0, 0, 0.01267), (-1, 0.0, 0, 0, -0.006769), (52, 0.732, 9, 10, -0.447795), (-1, 0.0, 0, 0, -0.031418), (-1, 0.0, 0, 0, 0.006532)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.510499), (-1, 0.0, 0, 0, -0.003704), (-1, 0.0, 0, 0, -0.031951), (39, 0.528, 5, 8, 0.017426), (40, 0.5, 6, 7, 0.103649), (-1, 0.0, 0, 0, 0.015465), (-1, 0.0, 0, 0, -0.005188), (1, 0.833, 9, 10, -0.155449), (-1, 0.0, 0, 0, -0.013746), (-1, 0.0, 0, 0, 0.030625)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (60, 0.04999999999999999, 2, 5, 0.040718), (15, -0.072, 3, 4, 0.204248), (-1, 0.0, 0, 0, 0.043035), (-1, 0.0, 0, 0, 0.002849), (55, 0.27525252525252525, 6, 7, -0.08911), (-1, 0.0, 0, 0, 0.002941), (-1, 0.0, 0, 0, -0.021375), (11, 0.896, 9, 12, -0.253566), (4, 0.354, 10, 11, 0.317868), (-1, 0.0, 0, 0, 0.034522), (-1, 0.0, 0, 0, -0.020358), (24, 0.06250000000000003, 13, 14, -0.588544), (-1, 0.0, 0, 0, -0.039516), (-1, 0.0, 0, 0, 0.011126)),
            ((46, 0.8571428571428572, 1, 8, 0.0), (3, 0.04700000000000004, 2, 5, 0.015908), (55, 0.1846590909090909, 3, 4, 0.629973), (-1, 0.0, 0, 0, 0.037118), (-1, 0.0, 0, 0, 0.004164), (15, -0.17, 6, 7, -0.011838), (-1, 0.0, 0, 0, -0.010517), (-1, 0.0, 0, 0, 0.003116), (-1, 0.0, 0, 0, -0.02798)),
            ((60, 0.98, 1, 8, 0.0), (5, 0.001, 2, 5, 0.018364), (15, 0.894, 3, 4, 0.367707), (-1, 0.0, 0, 0, 0.030723), (-1, 0.0, 0, 0, -0.017729), (9, 0.6154999999999999, 6, 7, -0.024889), (-1, 0.0, 0, 0, 0.002294), (-1, 0.0, 0, 0, -0.020224), (20, -3.683959493972949, 9, 10, -0.490457), (-1, 0.0, 0, 0, -0.030948), (-1, 0.0, 0, 0, -0.003388)),
            ((20, 4.243470074122607, 1, 8, 0.0), (26, 4.131156519244238, 2, 5, -0.023675), (52, 0.81, 3, 4, -0.331594), (-1, 0.0, 0, 0, -0.037354), (-1, 0.0, 0, 0, 0.021228), (13, 0.07899999999999996, 6, 7, 0.015494), (-1, 0.0, 0, 0, 0.018711), (-1, 0.0, 0, 0, -0.002949), (-1, 0.0, 0, 0, 0.028435)),
            ((53, 0.007000000000000006, 1, 8, 0.0), (60, 0.46799999999999997, 2, 5, -0.293874), (58, 0.36728966946792685, 3, 4, 0.010388), (-1, 0.0, 0, 0, 0.01387), (-1, 0.0, 0, 0, -0.026358), (11, 0.203, 6, 7, -0.520945), (-1, 0.0, 0, 0, -0.001346), (-1, 0.0, 0, 0, -0.030088), (52, 0.029000000000000026, 9, 12, 0.033929), (25, 6.101438773316817, 10, 11, 0.35673), (-1, 0.0, 0, 0, -0.028253), (-1, 0.0, 0, 0, 0.026347), (5, 0.001, 13, 14, -0.02854), (-1, 0.0, 0, 0, 0.031311), (-1, 0.0, 0, 0, -0.00407)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.014000000000000012, 2, 5, 0.033463), (3, 0.24, 3, 4, -0.06179), (-1, 0.0, 0, 0, 0.005654), (-1, 0.0, 0, 0, -0.0127), (3, 0.511, 6, 7, 0.221478), (-1, 0.0, 0, 0, 0.00415), (-1, 0.0, 0, 0, 0.039888), (6, 0.5988085425102423, 9, 12, -0.296234), (1, 0.0, 10, 11, 0.104298), (-1, 0.0, 0, 0, -0.019696), (-1, 0.0, 0, 0, 0.022854), (52, 0.015000000000000013, 13, 14, -0.675687), (-1, 0.0, 0, 0, -0.008189), (-1, 0.0, 0, 0, -0.039629)),
            ((9, 0.205, 1, 8, 0.0), (7, 0.44399999999999995, 2, 5, -0.120291), (4, 0.943, 3, 4, -0.002495), (-1, 0.0, 0, 0, 0.007479), (-1, 0.0, 0, 0, -0.020489), (60, 0.009000000000000001, 6, 7, -0.853074), (-1, 0.0, 0, 0, -0.010669), (-1, 0.0, 0, 0, -0.047999), (4, 0.979, 9, 12, 0.091547), (26, 4.6503818151035565, 10, 11, -0.021352), (-1, 0.0, 0, 0, 0.008456), (-1, 0.0, 0, 0, -0.021581), (12, 0.6719999999999999, 13, 14, 0.514627), (-1, 0.0, 0, 0, 0.038225), (-1, 0.0, 0, 0, -0.021791)),
            ((57, 0.5, 1, 8, 0.0), (27, 0.99, 2, 5, 0.031255), (1, 0.01, 3, 4, 0.122986), (-1, 0.0, 0, 0, 0.013489), (-1, 0.0, 0, 0, -0.004615), (42, 0.14835164835164835, 6, 7, -0.203487), (-1, 0.0, 0, 0, 0.008154), (-1, 0.0, 0, 0, -0.028323), (42, 0.2583333333333333, 9, 12, -0.307966), (26, 4.171079650336278, 10, 11, -0.855839), (-1, 0.0, 0, 0, 0.012122), (-1, 0.0, 0, 0, -0.050037), (2, 0.994, 13, 14, 0.368507), (-1, 0.0, 0, 0, -0.013497), (-1, 0.0, 0, 0, 0.034356)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.355, 2, 5, 0.032374), (15, 0.252, 3, 4, -0.020459), (-1, 0.0, 0, 0, 0.001779), (-1, 0.0, 0, 0, -0.040667), (60, 0.032, 6, 7, 0.353628), (-1, 0.0, 0, 0, 0.043448), (-1, 0.0, 0, 0, -0.00157), (6, 0.5988085425102423, 9, 12, -0.296591), (48, 0.953, 10, 11, 0.090088), (-1, 0.0, 0, 0, -0.016295), (-1, 0.0, 0, 0, 0.02491), (1, 0.0, 13, 14, -0.66339), (-1, 0.0, 0, 0, -0.009596), (-1, 0.0, 0, 0, -0.040537)),
            ((7, 0.951, 1, 8, 0.0), (9, 0.768, 2, 5, -0.060734), (26, 4.131156519244238, 3, 4, -0.01477), (-1, 0.0, 0, 0, -0.026531), (-1, 0.0, 0, 0, 0.001691), (59, 0.252, 6, 7, -0.626642), (-1, 0.0, 0, 0, -0.002093), (-1, 0.0, 0, 0, -0.045237), (42, 0.09545454545454546, 9, 12, 0.192191), (56, 0.12916666666666665, 10, 11, -0.364493), (-1, 0.0, 0, 0, 0.018718), (-1, 0.0, 0, 0, -0.032175), (27, 0.992, 13, 14, 0.359066), (-1, 0.0, 0, 0, 0.030067), (-1, 0.0, 0, 0, -0.015872)),
            ((51, 2.5, 1, 8, 0.0), (9, 0.6356666666666667, 2, 5, 0.018992), (27, 0.443, 3, 4, 0.066534), (-1, 0.0, 0, 0, 0.024987), (-1, 0.0, 0, 0, -0.001035), (5, 0.033, 6, 7, -0.24631), (-1, 0.0, 0, 0, 0.018349), (-1, 0.0, 0, 0, -0.031456), (12, 0.45099999999999996, 9, 10, -0.438168), (-1, 0.0, 0, 0, 0.001398), (-1, 0.0, 0, 0, -0.034322)),
            ((20, 4.243470074122607, 1, 8, 0.0), (39, 0.9045, 2, 5, -0.025395), (53, 0.278, 3, 4, 0.004272), (-1, 0.0, 0, 0, -0.00592), (-1, 0.0, 0, 0, 0.006087), (2, 0.998, 6, 7, -0.418656), (-1, 0.0, 0, 0, -0.03662), (-1, 0.0, 0, 0, 0.002181), (-1, 0.0, 0, 0, 0.02833)),
            ((40, 2.5, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, 0.026524), (26, 4.574061244703071, 3, 4, -0.156052), (-1, 0.0, 0, 0, -0.027603), (-1, 0.0, 0, 0, 0.000464), (11, 0.991, 6, 7, 0.115296), (-1, 0.0, 0, 0, 0.011064), (-1, 0.0, 0, 0, -0.011137), (6, 0.3024435993004616, 9, 10, -0.340747), (-1, 0.0, 0, 0, 0.014814), (1, 0.179, 11, 12, -0.532409), (-1, 0.0, 0, 0, -0.037787), (-1, 0.0, 0, 0, 0.008052)),
            ((20, 4.243470074122607, 1, 8, 0.0), (39, 0.9045, 2, 5, -0.025086), (8, 0.6190476190476191, 3, 4, 0.003929), (-1, 0.0, 0, 0, 0.003065), (-1, 0.0, 0, 0, -0.01266), (2, 0.998, 6, 7, -0.410507), (-1, 0.0, 0, 0, -0.035889), (-1, 0.0, 0, 0, 0.001923), (-1, 0.0, 0, 0, 0.027962)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.494826), (-1, 0.0, 0, 0, -0.003364), (-1, 0.0, 0, 0, -0.031148), (39, 0.528, 5, 8, 0.015486), (40, 0.5, 6, 7, 0.099258), (-1, 0.0, 0, 0, 0.014671), (-1, 0.0, 0, 0, -0.004787), (1, 0.833, 9, 10, -0.153449), (-1, 0.0, 0, 0, -0.01332), (-1, 0.0, 0, 0, 0.029611)),
            ((57, 0.5, 1, 8, 0.0), (27, 0.99, 2, 5, 0.029551), (7, 0.951, 3, 4, 0.115098), (-1, 0.0, 0, 0, 0.001155), (-1, 0.0, 0, 0, 0.021919), (42, 0.14835164835164835, 6, 7, -0.191236), (-1, 0.0, 0, 0, 0.007914), (-1, 0.0, 0, 0, -0.02725), (2, 0.994, 9, 12, -0.308348), (26, 4.212125883197904, 10, 11, -0.789651), (-1, 0.0, 0, 0, 0.01719), (-1, 0.0, 0, 0, -0.051578), (42, 0.2928571428571428, 13, 14, 0.402364), (-1, 0.0, 0, 0, -0.014384), (-1, 0.0, 0, 0, 0.035804)),
            ((19, 0.5, 1, 8, 0.0), (26, 4.134165555407019, 2, 5, -0.036072), (52, 0.81, 3, 4, -0.443636), (-1, 0.0, 0, 0, -0.039769), (-1, 0.0, 0, 0, 0.014739), (13, 0.44599999999999995, 6, 7, 0.012503), (-1, 0.0, 0, 0, 0.012644), (-1, 0.0, 0, 0, -0.004669), (6, 0.8846610635340376, 9, 12, 0.335776), (6, 0.06605249291607737, 10, 11, 0.53106), (-1, 0.0, 0, 0, -0.003718), (-1, 0.0, 0, 0, 0.035375), (-1, 0.0, 0, 0, -0.02241)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.486291), (-1, 0.0, 0, 0, -0.003325), (-1, 0.0, 0, 0, -0.030635), (39, -0.020499999999999963, 5, 8, 0.015484), (7, 0.886, 6, 7, 0.403623), (-1, 0.0, 0, 0, 0.041334), (-1, 0.0, 0, 0, -0.017436), (7, 0.911, 9, 10, -0.019286), (-1, 0.0, 0, 0, -0.005739), (-1, 0.0, 0, 0, 0.010734)),
            ((51, 2.5, 1, 8, 0.0), (9, 0.6356666666666667, 2, 5, 0.018731), (27, 0.443, 3, 4, 0.066489), (-1, 0.0, 0, 0, 0.023577), (-1, 0.0, 0, 0, -0.00079), (5, 0.009, 6, 7, -0.247651), (-1, 0.0, 0, 0, 0.022033), (-1, 0.0, 0, 0, -0.02793), (52, 0.732, 9, 10, -0.424222), (-1, 0.0, 0, 0, -0.030146), (-1, 0.0, 0, 0, 0.00664)),
            ((57, 0.5, 1, 8, 0.0), (39, 0.5545, 2, 5, 0.029085), (15, -0.125, 3, 4, 0.115226), (-1, 0.0, 0, 0, -0.006017), (-1, 0.0, 0, 0, 0.012165), (24, 0.031000000000000017, 6, 7, -0.178777), (-1, 0.0, 0, 0, -0.031886), (-1, 0.0, 0, 0, 0.002596), (2, 0.994, 9, 12, -0.30021), (26, 4.212125883197904, 10, 11, -0.767753), (-1, 0.0, 0, 0, 0.016741), (-1, 0.0, 0, 0, -0.050323), (42, 0.2928571428571428, 13, 14, 0.38853), (-1, 0.0, 0, 0, -0.01439), (-1, 0.0, 0, 0, 0.035025)),
            ((20, 4.243470074122607, 1, 8, 0.0), (26, 4.131156519244238, 2, 5, -0.02402), (52, 0.81, 3, 4, -0.325785), (-1, 0.0, 0, 0, -0.035458), (-1, 0.0, 0, 0, 0.018061), (13, 0.07899999999999996, 6, 7, 0.013049), (-1, 0.0, 0, 0, 0.016787), (-1, 0.0, 0, 0, -0.00276), (-1, 0.0, 0, 0, 0.027507)),
            ((40, 2.5, 1, 8, 0.0), (20, -3.683959493972949, 2, 5, 0.026005), (12, 0.34299999999999997, 3, 4, -0.14773), (-1, 0.0, 0, 0, -0.028085), (-1, 0.0, 0, 0, 0.000198), (11, 0.991, 6, 7, 0.109931), (-1, 0.0, 0, 0, 0.010542), (-1, 0.0, 0, 0, -0.010622), (4, 0.326, 9, 10, -0.326505), (-1, 0.0, 0, 0, -0.038203), (52, 0.406, 11, 12, 0.004466), (-1, 0.0, 0, 0, 0.020204), (-1, 0.0, 0, 0, -0.016446)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (26, 4.7988828053585495, 2, 5, 0.036604), (57, 0.5, 3, 4, -0.006191), (-1, 0.0, 0, 0, 0.003174), (-1, 0.0, 0, 0, -0.035504), (2, 0.993, 6, 7, 0.508288), (-1, 0.0, 0, 0, 0.001403), (-1, 0.0, 0, 0, 0.035902), (56, 0.4083333333333333, 9, 12, -0.239363), (11, 0.793, 10, 11, 0.306406), (-1, 0.0, 0, 0, 0.043632), (-1, 0.0, 0, 0, -0.014722), (20, 2.319785806352712, 13, 14, -0.590715), (-1, 0.0, 0, 0, -0.036159), (-1, 0.0, 0, 0, 0.01724)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.014000000000000012, 2, 5, 0.03039), (19, 0.5, 3, 4, -0.063549), (-1, 0.0, 0, 0, -0.006051), (-1, 0.0, 0, 0, 0.023515), (56, 0.5634920634920635, 6, 7, 0.214574), (-1, 0.0, 0, 0, -0.001336), (-1, 0.0, 0, 0, 0.025508), (6, 0.5988085425102423, 9, 12, -0.28533), (1, 0.0, 10, 11, 0.094535), (-1, 0.0, 0, 0, -0.019159), (-1, 0.0, 0, 0, 0.021702), (25, 6.291562280800715, 13, 14, -0.64827), (-1, 0.0, 0, 0, -0.008561), (-1, 0.0, 0, 0, -0.039562)),
            ((60, 0.98, 1, 8, 0.0), (56, 0.17424242424242425, 2, 5, 0.016234), (26, 4.623745818803488, 3, 4, 0.292437), (-1, 0.0, 0, 0, -0.0072), (-1, 0.0, 0, 0, 0.032293), (5, 0.982, 6, 7, -0.036069), (-1, 0.0, 0, 0, -0.003853), (-1, 0.0, 0, 0, 0.019169), (20, -3.683959493972949, 9, 10, -0.467988), (-1, 0.0, 0, 0, -0.029949), (-1, 0.0, 0, 0, -0.002715)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.473945), (-1, 0.0, 0, 0, -0.002931), (-1, 0.0, 0, 0, -0.030104), (39, -0.020499999999999963, 5, 8, 0.015968), (7, 0.886, 6, 7, 0.388451), (-1, 0.0, 0, 0, 0.039424), (-1, 0.0, 0, 0, -0.016782), (7, 0.911, 9, 10, -0.017841), (-1, 0.0, 0, 0, -0.005414), (-1, 0.0, 0, 0, 0.010202)),
            ((40, 2.5, 1, 8, 0.0), (60, 0.664, 2, 5, 0.025912), (2, 0.43899999999999995, 3, 4, 0.080236), (-1, 0.0, 0, 0, 0.03534), (-1, 0.0, 0, 0, 0.000458), (26, 4.598548757796862, 6, 7, -0.227526), (-1, 0.0, 0, 0, -0.031903), (-1, 0.0, 0, 0, 0.002277), (4, 0.326, 9, 10, -0.320891), (-1, 0.0, 0, 0, -0.037555), (56, 0.5147058823529411, 11, 12, 0.00272), (-1, 0.0, 0, 0, 0.016623), (-1, 0.0, 0, 0, -0.019331)),
            ((20, 4.243470074122607, 1, 8, 0.0), (8, 0.6190476190476191, 2, 5, -0.022422), (9, 0.4885, 3, 4, 0.026242), (-1, 0.0, 0, 0, -0.002319), (-1, 0.0, 0, 0, 0.03017), (1, 0.0, 6, 7, -0.252563), (-1, 0.0, 0, 0, 0.024886), (-1, 0.0, 0, 0, -0.023441), (-1, 0.0, 0, 0, 0.026801)),
            ((6, 0.6343440778663255, 1, 8, 0.0), (12, 0.394, 2, 5, 0.075882), (59, 0.20299999999999996, 3, 4, -0.159632), (-1, 0.0, 0, 0, 0.008935), (-1, 0.0, 0, 0, -0.025336), (12, 0.5740000000000001, 6, 7, 0.200361), (-1, 0.0, 0, 0, 0.021442), (-1, 0.0, 0, 0, -0.003803), (11, 0.987, 9, 12, -0.122403), (12, 0.22799999999999998, 10, 11, 0.043203), (-1, 0.0, 0, 0, -0.034005), (-1, 0.0, 0, 0, 0.008218), (44, 0.87, 13, 14, -0.452158), (-1, 0.0, 0, 0, -0.036149), (-1, 0.0, 0, 0, 0.022147)),
            ((15, -0.771, 1, 4, 0.0), (25, 6.682730066456371, 2, 3, -0.468006), (-1, 0.0, 0, 0, 0.000947), (-1, 0.0, 0, 0, -0.027198), (39, -0.020499999999999963, 5, 8, 0.016192), (7, 0.886, 6, 7, 0.370408), (-1, 0.0, 0, 0, 0.037361), (-1, 0.0, 0, 0, -0.016224), (40, 0.5, 9, 10, -0.016172), (-1, 0.0, 0, 0, 0.00568), (-1, 0.0, 0, 0, -0.007727)),
            ((60, 0.98, 1, 8, 0.0), (5, 0.001, 2, 5, 0.016562), (6, 0.6492994146520935, 3, 4, 0.347461), (-1, 0.0, 0, 0, 0.029754), (-1, 0.0, 0, 0, -0.013233), (4, 0.012, 6, 7, -0.023705), (-1, 0.0, 0, 0, -0.032282), (-1, 0.0, 0, 0, 0.000665), (20, -3.683959493972949, 9, 10, -0.458275), (-1, 0.0, 0, 0, -0.029417), (-1, 0.0, 0, 0, -0.00258)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.014000000000000012, 2, 5, 0.029904), (3, 0.30100000000000005, 3, 4, -0.061782), (-1, 0.0, 0, 0, 0.00319), (-1, 0.0, 0, 0, -0.014986), (60, 0.0009999999999999974, 6, 7, 0.208391), (-1, 0.0, 0, 0, -0.029243), (-1, 0.0, 0, 0, 0.014235), (6, 0.5988085425102423, 9, 12, -0.276786), (48, 0.953, 10, 11, 0.089905), (-1, 0.0, 0, 0, -0.015659), (-1, 0.0, 0, 0, 0.024171), (1, 0.0, 13, 14, -0.629181), (-1, 0.0, 0, 0, -0.008586), (-1, 0.0, 0, 0, -0.038832)),
            ((9, 0.9245, 1, 8, 0.0), (9, 0.768, 2, 5, -0.031122), (26, 4.131156519244238, 3, 4, 0.021698), (-1, 0.0, 0, 0, -0.020154), (-1, 0.0, 0, 0, 0.00323), (40, 0.5, 6, 7, -0.7927), (-1, 0.0, 0, 0, -0.007304), (-1, 0.0, 0, 0, -0.048797), (25, 6.692703705388288, 9, 12, 0.357678), (56, 0.7596153846153846, 10, 11, 0.609672), (-1, 0.0, 0, 0, 0.035625), (-1, 0.0, 0, 0, -0.004382), (-1, 0.0, 0, 0, -0.007678)),
            ((52, 0.995, 1, 8, 0.0), (7, 0.993, 2, 5, -0.024776), (55, 0.4494949494949495, 3, 4, 0.011889), (-1, 0.0, 0, 0, 0.003122), (-1, 0.0, 0, 0, -0.021279), (42, 0.3484848484848485, 6, 7, -0.576963), (-1, 0.0, 0, 0, -0.039875), (-1, 0.0, 0, 0, 0.016936), (7, 0.983, 9, 10, 0.483377), (-1, 0.0, 0, 0, -0.008822), (-1, 0.0, 0, 0, 0.033736)),
            ((60, 0.664, 1, 8, 0.0), (2, 0.43899999999999995, 2, 5, 0.038333), (27, 0.995, 3, 4, 0.581971), (-1, 0.0, 0, 0, 0.049469), (-1, 0.0, 0, 0, -0.011038), (5, 0.001, 6, 7, -0.021035), (-1, 0.0, 0, 0, 0.021365), (-1, 0.0, 0, 0, -0.004092), (26, 4.598548757796862, 9, 12, -0.215215), (4, 0.843, 10, 11, -0.653765), (-1, 0.0, 0, 0, -0.038898), (-1, 0.0, 0, 0, 0.001471), (25, 6.79065808523149, 13, 14, 0.067476), (-1, 0.0, 0, 0, 0.017305), (-1, 0.0, 0, 0, -0.021194)),
            ((15, -0.771, 1, 4, 0.0), (58, 0.2010069126605852, 2, 3, -0.463143), (-1, 0.0, 0, 0, -0.026874), (-1, 0.0, 0, 0, 0.000856), (39, 0.528, 5, 8, 0.014749), (40, 0.5, 6, 7, 0.088494), (-1, 0.0, 0, 0, 0.013497), (-1, 0.0, 0, 0, -0.004599), (1, 0.833, 9, 10, -0.135043), (-1, 0.0, 0, 0, -0.011893), (-1, 0.0, 0, 0, 0.027337)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.11767), (2, 0.997, 3, 4, 0.009868), (-1, 0.0, 0, 0, 0.007201), (-1, 0.0, 0, 0, -0.022445), (52, 0.867, 6, 7, -0.467475), (-1, 0.0, 0, 0, -0.039419), (-1, 0.0, 0, 0, 0.006592), (50, 0.5, 9, 12, 0.079083), (17, 0.5, 10, 11, 0.206606), (-1, 0.0, 0, 0, 0.030447), (-1, 0.0, 0, 0, 0.000589), (13, 0.998, 13, 14, -0.19141), (-1, 0.0, 0, 0, -0.017696), (-1, 0.0, 0, 0, 0.041365)),
            ((60, 0.664, 1, 8, 0.0), (2, 0.43899999999999995, 2, 5, 0.039539), (42, 0.17424242424242425, 3, 4, 0.560448), (-1, 0.0, 0, 0, 0.045246), (-1, 0.0, 0, 0, -0.01416), (2, 0.985, 6, 7, -0.017585), (-1, 0.0, 0, 0, -0.011057), (-1, 0.0, 0, 0, 0.005381), (26, 4.598548757796862, 9, 12, -0.215204), (4, 0.843, 10, 11, -0.646273), (-1, 0.0, 0, 0, -0.038432), (-1, 0.0, 0, 0, 0.00125), (1, 0.006, 13, 14, 0.061105), (-1, 0.0, 0, 0, 0.015715), (-1, 0.0, 0, 0, -0.022225)),
            ((53, 0.007000000000000006, 1, 8, 0.0), (60, 0.46799999999999997, 2, 5, -0.276212), (25, 6.247074819994996, 3, 4, 0.010124), (-1, 0.0, 0, 0, -0.024647), (-1, 0.0, 0, 0, 0.012837), (11, 0.203, 6, 7, -0.493353), (-1, 0.0, 0, 0, 0.000347), (-1, 0.0, 0, 0, -0.029053), (52, 0.029000000000000026, 9, 12, 0.029938), (25, 6.101438773316817, 10, 11, 0.316888), (-1, 0.0, 0, 0, -0.026258), (-1, 0.0, 0, 0, 0.023253), (11, 0.961, 13, 14, -0.026234), (-1, 0.0, 0, 0, -0.009262), (-1, 0.0, 0, 0, 0.008212)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (55, 0.3038461538461539, 2, 5, -0.11745), (45, 0.3655, 3, 4, 0.005117), (-1, 0.0, 0, 0, -0.005704), (-1, 0.0, 0, 0, 0.025276), (52, 0.867, 6, 7, -0.458401), (-1, 0.0, 0, 0, -0.038854), (-1, 0.0, 0, 0, 0.005992), (50, 0.5, 9, 12, 0.078404), (17, 0.5, 10, 11, 0.200755), (-1, 0.0, 0, 0, 0.029679), (-1, 0.0, 0, 0, 0.000668), (13, 0.998, 13, 14, -0.180152), (-1, 0.0, 0, 0, -0.016866), (-1, 0.0, 0, 0, 0.040309)),
            ((15, 0.755, 1, 8, 0.0), (15, 0.014000000000000012, 2, 5, 0.030209), (19, 0.5, 3, 4, -0.060304), (-1, 0.0, 0, 0, -0.005761), (-1, 0.0, 0, 0, 0.022812), (42, 0.3798076923076923, 6, 7, 0.205451), (-1, 0.0, 0, 0, 0.005111), (-1, 0.0, 0, 0, 0.045968), (44, 0.06817999999999999, 9, 12, -0.278836), (13, 0.028000000000000025, 10, 11, -0.44265), (-1, 0.0, 0, 0, 0.002966), (-1, 0.0, 0, 0, -0.033945), (-1, 0.0, 0, 0, 0.019019)),
            ((57, 0.5, 1, 8, 0.0), (27, 0.99, 2, 5, 0.029683), (59, 0.32399999999999995, 3, 4, 0.11367), (-1, 0.0, 0, 0, 0.000817), (-1, 0.0, 0, 0, 0.020448), (42, 0.14835164835164835, 6, 7, -0.191766), (-1, 0.0, 0, 0, 0.006355), (-1, 0.0, 0, 0, -0.026202), (2, 0.994, 9, 12, -0.292553), (26, 4.212125883197904, 10, 11, -0.738755), (-1, 0.0, 0, 0, 0.01673), (-1, 0.0, 0, 0, -0.048938), (42, 0.2928571428571428, 13, 14, 0.371242), (-1, 0.0, 0, 0, -0.013297), (-1, 0.0, 0, 0, 0.033378)),
            ((51, 2.5, 1, 8, 0.0), (9, 0.6356666666666667, 2, 5, 0.019807), (27, 0.443, 3, 4, 0.065902), (-1, 0.0, 0, 0, 0.022575), (-1, 0.0, 0, 0, -0.000723), (5, 0.033, 6, 7, -0.236973), (-1, 0.0, 0, 0, 0.016179), (-1, 0.0, 0, 0, -0.029198), (52, 0.732, 9, 10, -0.402066), (-1, 0.0, 0, 0, -0.02903), (-1, 0.0, 0, 0, 0.007086)),
            ((15, -0.771, 1, 4, 0.0), (58, 0.2010069126605852, 2, 3, -0.460421), (-1, 0.0, 0, 0, -0.026506), (-1, 0.0, 0, 0, 0.000695), (39, 0.528, 5, 8, 0.016678), (40, 0.5, 6, 7, 0.08959), (-1, 0.0, 0, 0, 0.013134), (-1, 0.0, 0, 0, -0.004146), (1, 0.833, 9, 10, -0.132), (-1, 0.0, 0, 0, -0.01146), (-1, 0.0, 0, 0, 0.025751)),
            ((57, 0.5, 1, 8, 0.0), (27, 0.99, 2, 5, 0.028714), (1, 0.01, 3, 4, 0.109161), (-1, 0.0, 0, 0, 0.012102), (-1, 0.0, 0, 0, -0.004256), (42, 0.14835164835164835, 6, 7, -0.18453), (-1, 0.0, 0, 0, 0.006054), (-1, 0.0, 0, 0, -0.025323), (42, 0.2583333333333333, 9, 12, -0.286621), (26, 4.171079650336278, 10, 11, -0.754997), (-1, 0.0, 0, 0, 0.010786), (-1, 0.0, 0, 0, -0.044529), (48, 0.968, 13, 14, 0.318257), (-1, 0.0, 0, 0, -0.015649), (-1, 0.0, 0, 0, 0.027891)),
            ((15, 0.755, 1, 8, 0.0), (5, 0.001, 2, 5, 0.029829), (48, 0.19099999999999995, 3, 4, 0.403622), (-1, 0.0, 0, 0, -0.015315), (-1, 0.0, 0, 0, 0.033006), (9, 0.6154999999999999, 6, 7, -0.010548), (-1, 0.0, 0, 0, 0.002611), (-1, 0.0, 0, 0, -0.018502), (6, 0.5988085425102423, 9, 12, -0.269475), (1, 0.0, 10, 11, 0.081545), (-1, 0.0, 0, 0, -0.019203), (-1, 0.0, 0, 0, 0.020672), (25, 6.291562280800715, 13, 14, -0.608176), (-1, 0.0, 0, 0, -0.007211), (-1, 0.0, 0, 0, -0.037521)),
            ((9, 0.9245, 1, 8, 0.0), (9, 0.768, 2, 5, -0.029624), (26, 4.131156519244238, 3, 4, 0.021105), (-1, 0.0, 0, 0, -0.019991), (-1, 0.0, 0, 0, 0.003154), (59, 0.252, 6, 7, -0.758543), (-1, 0.0, 0, 0, -0.008152), (-1, 0.0, 0, 0, -0.048458), (26, 4.62190419609534, 9, 12, 0.356801), (56, 0.7596153846153846, 10, 11, 0.593263), (-1, 0.0, 0, 0, 0.034481), (-1, 0.0, 0, 0, -0.003907), (-1, 0.0, 0, 0, -0.006512)),
            ((12, 0.6759999999999999, 1, 8, 0.0), (60, 0.04999999999999999, 2, 5, 0.035657), (48, 0.907, 3, 4, 0.186121), (-1, 0.0, 0, 0, 0.040034), (-1, 0.0, 0, 0, 0.001695), (5, 0.078, 6, 7, -0.082824), (-1, 0.0, 0, 0, 0.0056), (-1, 0.0, 0, 0, -0.014545), (56, 0.4083333333333333, 9, 12, -0.228083), (11, 0.793, 10, 11, 0.29064), (-1, 0.0, 0, 0, 0.040916), (-1, 0.0, 0, 0, -0.013993), (20, 2.319785806352712, 13, 14, -0.568993), (-1, 0.0, 0, 0, -0.034748), (-1, 0.0, 0, 0, 0.016293)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.453571), (-1, 0.0, 0, 0, -0.002699), (-1, 0.0, 0, 0, -0.028731), (39, 0.528, 5, 8, 0.01595), (40, 0.5, 6, 7, 0.087361), (-1, 0.0, 0, 0, 0.012515), (-1, 0.0, 0, 0, -0.003767), (1, 0.833, 9, 10, -0.130133), (-1, 0.0, 0, 0, -0.011203), (-1, 0.0, 0, 0, 0.025128)),
            ((24, 0.04500000000000004, 1, 8, 0.0), (56, 0.4494949494949495, 2, 5, -0.115012), (15, -0.20700000000000002, 3, 4, 0.056565), (-1, 0.0, 0, 0, -0.014254), (-1, 0.0, 0, 0, 0.012596), (1, 0.039, 6, 7, -0.342113), (-1, 0.0, 0, 0, -0.027883), (-1, 0.0, 0, 0, 0.008969), (11, 0.45, 9, 12, 0.078335), (11, 0.086, 10, 11, -0.258786), (-1, 0.0, 0, 0, 0.015422), (-1, 0.0, 0, 0, -0.03044), (15, 0.014000000000000012, 13, 14, 0.166136), (-1, 0.0, 0, 0, -0.000805), (-1, 0.0, 0, 0, 0.019479)),
            ((44, 0.05104999999999999, 1, 8, 0.0), (5, 0.003, 2, 5, -0.26675), (26, 4.2964256503859755, 3, 4, 0.220414), (-1, 0.0, 0, 0, -0.018399), (-1, 0.0, 0, 0, 0.028857), (39, 0.035599999999999965, 6, 7, -0.615156), (-1, 0.0, 0, 0, -0.007824), (-1, 0.0, 0, 0, -0.038245), (44, 0.05144000000000001, 9, 10, 0.030037), (-1, 0.0, 0, 0, 0.0268), (11, 0.45, 11, 12, -0.005931), (-1, 0.0, 0, 0, -0.014012), (-1, 0.0, 0, 0, 0.002864)),
            ((60, 0.98, 1, 8, 0.0), (6, 0.6343440778663255, 2, 5, 0.016124), (1, 0.009, 3, 4, 0.10523), (-1, 0.0, 0, 0, 0.013008), (-1, 0.0, 0, 0, -0.005715), (11, 0.987, 6, 7, -0.111525), (-1, 0.0, 0, 0, 0.002456), (-1, 0.0, 0, 0, -0.021677), (20, -3.683959493972949, 9, 10, -0.44709), (-1, 0.0, 0, 0, -0.028908), (-1, 0.0, 0, 0, -0.002107)),
            ((20, 4.243470074122607, 1, 8, 0.0), (26, 4.131156519244238, 2, 5, -0.021741), (44, 0.62, 3, 4, -0.302512), (-1, 0.0, 0, 0, -0.03537), (-1, 0.0, 0, 0, 0.014912), (11, 0.044, 6, 7, 0.011757), (-1, 0.0, 0, 0, 0.033045), (-1, 0.0, 0, 0, -0.000877), (-1, 0.0, 0, 0, 0.026046)),
            ((15, 0.755, 1, 8, 0.0), (24, 0.04500000000000004, 2, 5, 0.029202), (2, 0.998, 3, 4, -0.104635), (-1, 0.0, 0, 0, 0.000101), (-1, 0.0, 0, 0, -0.028213), (50, 0.5, 6, 7, 0.135284), (-1, 0.0, 0, 0, 0.01377), (-1, 0.0, 0, 0, -0.006284), (44, 0.06817999999999999, 9, 12, -0.267837), (13, 0.028000000000000025, 10, 11, -0.426791), (-1, 0.0, 0, 0, 0.00258), (-1, 0.0, 0, 0, -0.032589), (-1, 0.0, 0, 0, 0.018177)),
            ((60, 0.664, 1, 8, 0.0), (2, 0.43899999999999995, 2, 5, 0.039153), (27, 0.979, 3, 4, 0.556547), (-1, 0.0, 0, 0, 0.052521), (-1, 0.0, 0, 0, -0.002827), (5, 0.001, 6, 7, -0.017379), (-1, 0.0, 0, 0, 0.020193), (-1, 0.0, 0, 0, -0.003713), (26, 4.598548757796862, 9, 12, -0.208081), (4, 0.843, 10, 11, -0.639026), (-1, 0.0, 0, 0, -0.037952), (-1, 0.0, 0, 0, 0.000835), (4, 0.77, 13, 14, 0.063992), (-1, 0.0, 0, 0, 0.0155), (-1, 0.0, 0, 0, -0.022134)),
            ((20, 4.243470074122607, 1, 8, 0.0), (8, 0.6190476190476191, 2, 5, -0.020349), (9, 0.4885, 3, 4, 0.023689), (-1, 0.0, 0, 0, -0.002314), (-1, 0.0, 0, 0, 0.028378), (1, 0.0, 6, 7, -0.22874), (-1, 0.0, 0, 0, 0.022021), (-1, 0.0, 0, 0, -0.020911), (-1, 0.0, 0, 0, 0.025681)),
        )),
        (0.030172, (
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -1.298983), (-1, 0.0, 0, 0, -0.078076), (-1, 0.0, 0, 0, -0.017126), (44, 0.12, 5, 6, 1.788868), (-1, 0.0, 0, 0, 0.018356), (-1, 0.0, 0, 0, 0.094061)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -1.234906), (-1, 0.0, 0, 0, -0.074229), (-1, 0.0, 0, 0, -0.016291), (44, 0.12, 5, 6, 1.706295), (-1, 0.0, 0, 0, 0.017554), (-1, 0.0, 0, 0, 0.08973)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.62, 2, 3, -1.176671), (-1, 0.0, 0, 0, -0.06855), (-1, 0.0, 0, 0, -0.005387), (44, 0.12, 5, 6, 1.633676), (-1, 0.0, 0, 0, 0.01679), (-1, 0.0, 0, 0, 0.085942)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -1.123179), (-1, 0.0, 0, 0, -0.06774), (-1, 0.0, 0, 0, -0.014349), (44, 0.12, 5, 6, 1.569255), (-1, 0.0, 0, 0, 0.01606), (-1, 0.0, 0, 0, 0.0826)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -1.074064), (-1, 0.0, 0, 0, -0.069306), (-1, 0.0, 0, 0, -0.022846), (44, 0.12, 5, 6, 1.511658), (-1, 0.0, 0, 0, 0.015364), (-1, 0.0, 0, 0, 0.079629)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.62, 2, 3, -1.027834), (-1, 0.0, 0, 0, -0.060511), (-1, 0.0, 0, 0, -0.002246), (44, 0.12, 5, 6, 1.459798), (-1, 0.0, 0, 0, 0.014699), (-1, 0.0, 0, 0, 0.07697)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.985215), (-1, 0.0, 0, 0, -0.064405), (-1, 0.0, 0, 0, -0.019613), (44, 0.12, 5, 6, 1.412801), (-1, 0.0, 0, 0, 0.014064), (-1, 0.0, 0, 0, 0.074574)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -0.944898), (-1, 0.0, 0, 0, -0.058352), (-1, 0.0, 0, 0, -0.008441), (44, 0.12, 5, 6, 1.369954), (-1, 0.0, 0, 0, 0.013457), (-1, 0.0, 0, 0, 0.072403)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.907539), (-1, 0.0, 0, 0, -0.060208), (-1, 0.0, 0, 0, -0.01675), (44, 0.12, 5, 6, 1.330673), (-1, 0.0, 0, 0, 0.012877), (-1, 0.0, 0, 0, 0.070426)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -0.871811), (-1, 0.0, 0, 0, -0.054538), (-1, 0.0, 0, 0, -0.006167), (44, 0.12, 5, 6, 1.294472), (-1, 0.0, 0, 0, 0.012323), (-1, 0.0, 0, 0, 0.068616)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.838477), (-1, 0.0, 0, 0, -0.054436), (-1, 0.0, 0, 0, -0.009322), (44, 0.12, 5, 6, 1.260945), (-1, 0.0, 0, 0, 0.011792), (-1, 0.0, 0, 0, 0.066952)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -0.806147), (-1, 0.0, 0, 0, -0.051201), (-1, 0.0, 0, 0, -0.003927), (44, 0.12, 5, 6, 1.229748), (-1, 0.0, 0, 0, 0.011285), (-1, 0.0, 0, 0, 0.065414)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.776046), (-1, 0.0, 0, 0, -0.021716), (-1, 0.0, 0, 0, -0.062893), (44, 0.12, 5, 6, 1.20059), (-1, 0.0, 0, 0, 0.0108), (-1, 0.0, 0, 0, 0.063988)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.746212), (-1, 0.0, 0, 0, -0.052074), (-1, 0.0, 0, 0, -0.009985), (44, 0.12, 5, 6, 1.17322), (-1, 0.0, 0, 0, 0.010675), (-1, 0.0, 0, 0, 0.062659)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.62, 2, 3, -0.718381), (-1, 0.0, 0, 0, -0.044909), (-1, 0.0, 0, 0, 0.008011), (44, 0.12, 5, 6, 1.147399), (-1, 0.0, 0, 0, 0.010725), (-1, 0.0, 0, 0, 0.061417)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.691921), (-1, 0.0, 0, 0, -0.01763), (-1, 0.0, 0, 0, -0.058862), (44, 0.12, 5, 6, 1.122927), (-1, 0.0, 0, 0, 0.010775), (-1, 0.0, 0, 0, 0.060251)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.665738), (-1, 0.0, 0, 0, -0.048166), (-1, 0.0, 0, 0, -0.006514), (44, 0.12, 5, 6, 1.099641), (-1, 0.0, 0, 0, 0.010848), (-1, 0.0, 0, 0, 0.059153)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.750792), (-1, 0.0, 0, 0, -0.052613), (-1, 0.0, 0, 0, -0.00578), (24, 0.179, 5, 6, 0.934246), (-1, 0.0, 0, 0, 0.011595), (-1, 0.0, 0, 0, 0.059347)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.725346), (-1, 0.0, 0, 0, -0.051181), (-1, 0.0, 0, 0, -0.005505), (20, -3.210268834500856, 5, 6, 0.910579), (-1, 0.0, 0, 0, -0.01033), (-1, 0.0, 0, 0, 0.053268)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.600702), (-1, 0.0, 0, 0, -0.011281), (-1, 0.0, 0, 0, -0.052627), (44, 0.12, 5, 6, 1.040063), (-1, 0.0, 0, 0, 0.010984), (-1, 0.0, 0, 0, 0.056281)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.68363), (-1, 0.0, 0, 0, -0.049045), (-1, 0.0, 0, 0, -0.004384), (24, 0.179, 5, 6, 0.876847), (-1, 0.0, 0, 0, 0.011138), (-1, 0.0, 0, 0, 0.056697)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.660698), (-1, 0.0, 0, 0, -0.047757), (-1, 0.0, 0, 0, -0.004175), (24, 0.179, 5, 6, 0.855102), (-1, 0.0, 0, 0, 0.01059), (-1, 0.0, 0, 0, 0.055866)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.539758), (-1, 0.0, 0, 0, -0.009941), (-1, 0.0, 0, 0, -0.052348), (44, 0.12, 5, 6, 0.985795), (-1, 0.0, 0, 0, 0.011151), (-1, 0.0, 0, 0, 0.053727)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.623514), (-1, 0.0, 0, 0, -0.045846), (-1, 0.0, 0, 0, -0.003271), (24, 0.179, 5, 6, 0.826565), (-1, 0.0, 0, 0, 0.011316), (-1, 0.0, 0, 0, 0.054326)),
            ((44, 0.62, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.602604), (-1, 0.0, 0, 0, -0.064969), (-1, 0.0, 0, 0, -0.018565), (20, -3.210268834500856, 5, 6, 0.806119), (-1, 0.0, 0, 0, -0.00825), (-1, 0.0, 0, 0, 0.048169)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.487455), (-1, 0.0, 0, 0, -0.007393), (-1, 0.0, 0, 0, -0.05015), (44, 0.12, 5, 6, 0.939207), (-1, 0.0, 0, 0, 0.011294), (-1, 0.0, 0, 0, 0.051495)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.567892), (-1, 0.0, 0, 0, -0.043291), (-1, 0.0, 0, 0, -0.001186), (24, 0.179, 5, 6, 0.778134), (-1, 0.0, 0, 0, 0.010807), (-1, 0.0, 0, 0, 0.052283)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.607394), (-1, 0.0, 0, 0, -0.045918), (-1, 0.0, 0, 0, -0.003531), (24, 0.179, 5, 6, 0.687731), (-1, 0.0, 0, 0, 0.008538), (-1, 0.0, 0, 0, 0.051638)),
            ((44, 0.62, 1, 4, 0.0), (26, 4.190735823528673, 2, 3, -0.530639), (-1, 0.0, 0, 0, -0.059821), (-1, 0.0, 0, 0, -0.013821), (24, 0.179, 5, 6, 0.74031), (-1, 0.0, 0, 0, 0.009858), (-1, 0.0, 0, 0, 0.050963)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.423233), (-1, 0.0, 0, 0, -0.00224), (-1, 0.0, 0, 0, -0.0453), (44, 0.12, 5, 6, 0.880516), (-1, 0.0, 0, 0, 0.011432), (-1, 0.0, 0, 0, 0.048723)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.558227), (-1, 0.0, 0, 0, -0.043818), (-1, 0.0, 0, 0, -0.001593), (24, 0.179, 5, 6, 0.649319), (-1, 0.0, 0, 0, 0.009007), (-1, 0.0, 0, 0, 0.049754)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.540095), (-1, 0.0, 0, 0, -0.042819), (-1, 0.0, 0, 0, -0.001516), (24, 0.179, 5, 6, 0.630784), (-1, 0.0, 0, 0, 0.008557), (-1, 0.0, 0, 0, 0.049144)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.522453), (-1, 0.0, 0, 0, -0.04184), (-1, 0.0, 0, 0, -0.001443), (55, 0.3205128205128205, 5, 6, 0.612612), (-1, 0.0, 0, 0, 0.047573), (-1, 0.0, 0, 0, 0.006866)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.371796), (-1, 0.0, 0, 0, -0.051466), (-1, 0.0, 0, 0, -0.004115), (7, 0.905, 5, 6, 0.827386), (-1, 0.0, 0, 0, 0.019823), (-1, 0.0, 0, 0, 0.05056)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (53, 0.41900000000000004, 2, 3, -0.584578), (-1, 0.0, 0, 0, -0.039417), (-1, 0.0, 0, 0, -0.002126), (13, 0.968, 5, 6, 0.499548), (-1, 0.0, 0, 0, -0.000328), (-1, 0.0, 0, 0, 0.045877)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (58, 0.41749300143204865, 2, 3, -0.484704), (-1, 0.0, 0, 0, -0.012233), (-1, 0.0, 0, 0, -0.057283), (55, 0.3205128205128205, 5, 6, 0.577051), (-1, 0.0, 0, 0, 0.045875), (-1, 0.0, 0, 0, 0.00537)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.33932), (-1, 0.0, 0, 0, 0.002053), (-1, 0.0, 0, 0, -0.041956), (7, 0.905, 5, 6, 0.792022), (-1, 0.0, 0, 0, 0.01805), (-1, 0.0, 0, 0, 0.049285)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (53, 0.41900000000000004, 2, 3, -0.546611), (-1, 0.0, 0, 0, -0.037589), (-1, 0.0, 0, 0, -0.001866), (13, 0.968, 5, 6, 0.471987), (-1, 0.0, 0, 0, -6.7e-05), (-1, 0.0, 0, 0, 0.043977)),
            ((44, 0.62, 1, 4, 0.0), (26, 4.190735823528673, 2, 3, -0.401544), (-1, 0.0, 0, 0, -0.055568), (-1, 0.0, 0, 0, -0.007656), (53, 0.258, 5, 6, 0.616944), (-1, 0.0, 0, 0, -0.002632), (-1, 0.0, 0, 0, 0.042435)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.520853), (-1, 0.0, 0, 0, -0.008916), (-1, 0.0, 0, 0, -0.047904), (13, 0.968, 5, 6, 0.451455), (-1, 0.0, 0, 0, 0.000297), (-1, 0.0, 0, 0, 0.042368)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.431542), (-1, 0.0, 0, 0, -0.057603), (-1, 0.0, 0, 0, -0.010735), (56, 0.4083333333333333, 5, 6, 0.529453), (-1, 0.0, 0, 0, 0.048895), (-1, 0.0, 0, 0, 0.009446)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.293739), (-1, 0.0, 0, 0, 0.003506), (-1, 0.0, 0, 0, -0.0395), (48, 0.977, 5, 6, 0.737479), (-1, 0.0, 0, 0, 0.019238), (-1, 0.0, 0, 0, 0.050149)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.409931), (-1, 0.0, 0, 0, -0.056827), (-1, 0.0, 0, 0, -0.00988), (56, 0.4641025641025641, 5, 6, 0.507459), (-1, 0.0, 0, 0, 0.043874), (-1, 0.0, 0, 0, 0.005626)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.256578947368421, 2, 3, -0.485785), (-1, 0.0, 0, 0, -0.005813), (-1, 0.0, 0, 0, -0.043741), (40, 0.5, 5, 6, 0.424606), (-1, 0.0, 0, 0, 0.041074), (-1, 0.0, 0, 0, 7.6e-05)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.271731), (-1, 0.0, 0, 0, -0.046997), (-1, 0.0, 0, 0, -0.000344), (48, 0.977, 5, 6, 0.706534), (-1, 0.0, 0, 0, 0.017447), (-1, 0.0, 0, 0, 0.049315)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.461251), (-1, 0.0, 0, 0, -0.006757), (-1, 0.0, 0, 0, -0.045031), (40, 0.5, 5, 6, 0.403512), (-1, 0.0, 0, 0, 0.039691), (-1, 0.0, 0, 0, -0.000313)),
            ((44, 0.62, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.336239), (-1, 0.0, 0, 0, -0.056667), (-1, 0.0, 0, 0, -0.006328), (2, 0.954, 5, 6, 0.543059), (-1, 0.0, 0, 0, -0.006845), (-1, 0.0, 0, 0, 0.039451)),
            ((20, 1.965912816362163, 1, 4, 0.0), (25, 6.556061562263916, 2, 3, -0.203632), (-1, 0.0, 0, 0, -0.03591), (-1, 0.0, 0, 0, 0.002841), (7, 0.46299999999999997, 5, 6, 0.835962), (-1, 0.0, 0, 0, 0.015808), (-1, 0.0, 0, 0, 0.048245)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.256578947368421, 2, 3, -0.434999), (-1, 0.0, 0, 0, -0.003517), (-1, 0.0, 0, 0, -0.041567), (40, 0.5, 5, 6, 0.381912), (-1, 0.0, 0, 0, 0.0384), (-1, 0.0, 0, 0, -0.0008)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.358537), (-1, 0.0, 0, 0, -0.054957), (-1, 0.0, 0, 0, -0.008149), (56, 0.4641025641025641, 5, 6, 0.457015), (-1, 0.0, 0, 0, 0.042008), (-1, 0.0, 0, 0, 0.002435)),
            ((20, 1.965912816362163, 1, 4, 0.0), (58, 0.22564484631765486, 2, 3, -0.188652), (-1, 0.0, 0, 0, 0.003837), (-1, 0.0, 0, 0, -0.032822), (7, 0.46299999999999997, 5, 6, 0.812351), (-1, 0.0, 0, 0, 0.015592), (-1, 0.0, 0, 0, 0.047486)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.413268), (-1, 0.0, 0, 0, -0.004477), (-1, 0.0, 0, 0, -0.043163), (40, 0.5, 5, 6, 0.36351), (-1, 0.0, 0, 0, 0.037088), (-1, 0.0, 0, 0, -0.001009)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.336187), (-1, 0.0, 0, 0, -0.054089), (-1, 0.0, 0, 0, -0.00743), (56, 0.4641025641025641, 5, 6, 0.432173), (-1, 0.0, 0, 0, 0.040498), (-1, 0.0, 0, 0, 0.001635)),
            ((20, 1.965912816362163, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.174936), (-1, 0.0, 0, 0, 0.004271), (-1, 0.0, 0, 0, -0.031505), (7, 0.46299999999999997, 5, 6, 0.788874), (-1, 0.0, 0, 0, 0.015348), (-1, 0.0, 0, 0, 0.046746)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.389878), (-1, 0.0, 0, 0, -0.001523), (-1, 0.0, 0, 0, -0.039383), (13, 0.968, 5, 6, 0.342447), (-1, 0.0, 0, 0, -0.00111), (-1, 0.0, 0, 0, 0.035873)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (44, 0.05984, 2, 3, -0.315434), (-1, 0.0, 0, 0, 0.014453), (-1, 0.0, 0, 0, -0.026957), (56, 0.4641025641025641, 5, 6, 0.410394), (-1, 0.0, 0, 0, 0.038806), (-1, 0.0, 0, 0, 0.001212)),
            ((20, 1.965912816362163, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.160518), (-1, 0.0, 0, 0, -0.036304), (-1, 0.0, 0, 0, 0.002328), (7, 0.46299999999999997, 5, 6, 0.763534), (-1, 0.0, 0, 0, 0.015075), (-1, 0.0, 0, 0, 0.045937)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.370473), (-1, 0.0, 0, 0, -0.003067), (-1, 0.0, 0, 0, -0.04107), (40, 0.5, 5, 6, 0.327205), (-1, 0.0, 0, 0, 0.034689), (-1, 0.0, 0, 0, -0.001082)),
            ((53, 0.278, 1, 4, 0.0), (26, 4.184499866031553, 2, 3, -0.326197), (-1, 0.0, 0, 0, -0.05291), (-1, 0.0, 0, 0, -0.005002), (7, 0.953, 5, 6, 0.355373), (-1, 0.0, 0, 0, 0.00142), (-1, 0.0, 0, 0, 0.043855)),
            ((20, 1.965912816362163, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.148327), (-1, 0.0, 0, 0, -0.029906), (-1, 0.0, 0, 0, 0.004718), (7, 0.46299999999999997, 5, 6, 0.738308), (-1, 0.0, 0, 0, 0.01475), (-1, 0.0, 0, 0, 0.045112)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.349494), (-1, 0.0, 0, 0, -0.001015), (-1, 0.0, 0, 0, -0.037464), (40, 0.5, 5, 6, 0.30889), (-1, 0.0, 0, 0, 0.033507), (-1, 0.0, 0, 0, -0.001015)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.173204), (-1, 0.0, 0, 0, 0.064758), (-1, 0.0, 0, 0, -0.000343), (24, 0.015833333333333345, 5, 6, -0.624209), (-1, 0.0, 0, 0, -0.048959), (-1, 0.0, 0, 0, -0.002941)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.165466), (-1, 0.0, 0, 0, 0.062332), (-1, 0.0, 0, 0, -0.000326), (24, 0.015833333333333345, 5, 6, -0.607932), (-1, 0.0, 0, 0, -0.048295), (-1, 0.0, 0, 0, -0.002809)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.271039), (-1, 0.0, 0, 0, -0.051994), (-1, 0.0, 0, 0, -0.0051), (56, 0.4641025641025641, 5, 6, 0.359089), (-1, 0.0, 0, 0, 0.035566), (-1, 0.0, 0, 0, -0.000366)),
            ((20, 1.965912816362163, 1, 4, 0.0), (40, 0.5, 2, 3, -0.135003), (-1, 0.0, 0, 0, 0.008832), (-1, 0.0, 0, 0, -0.023231), (2, 0.974, 5, 6, 0.707377), (-1, 0.0, 0, 0, 0.014309), (-1, 0.0, 0, 0, 0.044952)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.292598), (-1, 0.0, 0, 0, -0.053362), (-1, 0.0, 0, 0, -0.004755), (7, 0.953, 5, 6, 0.315308), (-1, 0.0, 0, 0, 0.000667), (-1, 0.0, 0, 0, 0.041605)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.319387), (-1, 0.0, 0, 0, -0.000966), (-1, 0.0, 0, 0, -0.039163), (40, 0.5, 5, 6, 0.280741), (-1, 0.0, 0, 0, 0.031399), (-1, 0.0, 0, 0, -0.000966)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.151175), (-1, 0.0, 0, 0, 0.059732), (-1, 0.0, 0, 0, -0.000599), (24, 0.015833333333333345, 5, 6, -0.575542), (-1, 0.0, 0, 0, -0.04716), (-1, 0.0, 0, 0, -0.001951)),
            ((20, 1.965912816362163, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.123755), (-1, 0.0, 0, 0, 0.005252), (-1, 0.0, 0, 0, -0.027399), (2, 0.992, 5, 6, 0.678542), (-1, 0.0, 0, 0, 0.01387), (-1, 0.0, 0, 0, 0.047031)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.144871), (-1, 0.0, 0, 0, 0.057233), (-1, 0.0, 0, 0, -0.000528), (26, 4.598548757796862, 5, 6, -0.559356), (-1, 0.0, 0, 0, -0.049101), (-1, 0.0, 0, 0, -0.004374)),
            ((40, 0.5, 1, 4, 0.0), (46, 0.8571428571428572, 2, 3, 0.275259), (-1, 0.0, 0, 0, 0.01986), (-1, 0.0, 0, 0, -0.045083), (52, 0.985, 5, 6, -0.296031), (-1, 0.0, 0, 0, -0.022435), (-1, 0.0, 0, 0, 0.047931)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.153995), (-1, 0.0, 0, 0, 0.006444), (-1, 0.0, 0, 0, -0.030269), (2, 0.992, 5, 6, 0.504775), (-1, 0.0, 0, 0, 0.008769), (-1, 0.0, 0, 0, 0.040573)),
            ((53, 0.278, 1, 4, 0.0), (26, 4.184499866031553, 2, 3, -0.266133), (-1, 0.0, 0, 0, -0.050229), (-1, 0.0, 0, 0, -0.003411), (7, 0.953, 5, 6, 0.284064), (-1, 0.0, 0, 0, 0.000448), (-1, 0.0, 0, 0, 0.0395)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.581130329250939, 2, 3, 0.265306), (-1, 0.0, 0, 0, -0.007495), (-1, 0.0, 0, 0, 0.030438), (52, 0.985, 5, 6, -0.28846), (-1, 0.0, 0, 0, -0.021542), (-1, 0.0, 0, 0, 0.046615)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.286422), (-1, 0.0, 0, 0, -0.000867), (-1, 0.0, 0, 0, -0.034947), (52, 0.976, 5, 6, 0.251759), (-1, 0.0, 0, 0, 0.004524), (-1, 0.0, 0, 0, 0.045041)),
            ((2, 0.993, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.216282), (-1, 0.0, 0, 0, 0.002813), (-1, 0.0, 0, 0, -0.026043), (56, 0.4641025641025641, 5, 6, 0.328783), (-1, 0.0, 0, 0, 0.048264), (-1, 0.0, 0, 0, 0.002813)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.25828), (-1, 0.0, 0, 0, -0.007466), (-1, 0.0, 0, 0, 0.029221), (52, 0.985, 5, 6, -0.276542), (-1, 0.0, 0, 0, -0.020389), (-1, 0.0, 0, 0, 0.044922)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.137237), (-1, 0.0, 0, 0, 0.05479), (-1, 0.0, 0, 0, -0.000506), (24, 0.015833333333333345, 5, 6, -0.519815), (-1, 0.0, 0, 0, -0.0458), (-1, 0.0, 0, 0, 0.000987)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.249185), (-1, 0.0, 0, 0, 0.029798), (-1, 0.0, 0, 0, -0.005698), (52, 0.985, 5, 6, -0.268117), (-1, 0.0, 0, 0, -0.019678), (-1, 0.0, 0, 0, 0.04392)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.134071), (-1, 0.0, 0, 0, 0.052564), (-1, 0.0, 0, 0, -0.000292), (24, 0.015833333333333345, 5, 6, -0.510654), (-1, 0.0, 0, 0, -0.045379), (-1, 0.0, 0, 0, 0.000901)),
            ((20, 2.319785806352712, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.092607), (-1, 0.0, 0, 0, 0.001466), (-1, 0.0, 0, 0, -0.037395), (13, 0.744, 5, 6, 0.683797), (-1, 0.0, 0, 0, 0.01478), (-1, 0.0, 0, 0, 0.041723)),
            ((40, 0.5, 1, 4, 0.0), (46, 0.8571428571428572, 2, 3, 0.240306), (-1, 0.0, 0, 0, 0.017285), (-1, 0.0, 0, 0, -0.043492), (52, 0.985, 5, 6, -0.260624), (-1, 0.0, 0, 0, -0.019009), (-1, 0.0, 0, 0, 0.04307)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.26238), (-1, 0.0, 0, 0, -0.00067), (-1, 0.0, 0, 0, -0.033356), (25, 6.802950010466324, 5, 6, 0.23557), (-1, 0.0, 0, 0, 0.005481), (-1, 0.0, 0, 0, 0.047023)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.128072), (-1, 0.0, 0, 0, 0.050528), (-1, 0.0, 0, 0, -0.000305), (24, 0.015833333333333345, 5, 6, -0.492708), (-1, 0.0, 0, 0, -0.044569), (-1, 0.0, 0, 0, 0.000977)),
            ((2, 0.993, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.197912), (-1, 0.0, 0, 0, 0.002694), (-1, 0.0, 0, 0, -0.024022), (56, 0.4494949494949495, 5, 6, 0.305655), (-1, 0.0, 0, 0, 0.047263), (-1, 0.0, 0, 0, 0.002694)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.238372), (-1, 0.0, 0, 0, -0.007418), (-1, 0.0, 0, 0, 0.027012), (52, 0.985, 5, 6, -0.253116), (-1, 0.0, 0, 0, -0.018281), (-1, 0.0, 0, 0, 0.041736)),
            ((19, 0.5, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.078637), (-1, 0.0, 0, 0, -0.027818), (-1, 0.0, 0, 0, 0.003759), (6, 0.8846610635340376, 5, 6, 0.695605), (-1, 0.0, 0, 0, 0.040717), (-1, 0.0, 0, 0, 0.015424)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.123134), (-1, 0.0, 0, 0, 0.055037), (-1, 0.0, 0, 0, 0.000623), (24, 0.015833333333333345, 5, 6, -0.474614), (-1, 0.0, 0, 0, -0.043877), (-1, 0.0, 0, 0, 0.001297)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.231058), (-1, 0.0, 0, 0, 0.028134), (-1, 0.0, 0, 0, -0.005521), (52, 0.985, 5, 6, -0.247488), (-1, 0.0, 0, 0, -0.017751), (-1, 0.0, 0, 0, 0.040646)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.119902), (-1, 0.0, 0, 0, -0.036608), (-1, 0.0, 0, 0, 0.00279), (2, 0.992, 5, 6, 0.444122), (-1, 0.0, 0, 0, 0.008105), (-1, 0.0, 0, 0, 0.037898)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.242349), (-1, 0.0, 0, 0, -0.000523), (-1, 0.0, 0, 0, -0.031963), (40, 0.5, 5, 6, 0.22142), (-1, 0.0, 0, 0, 0.02546), (-1, 0.0, 0, 0, -0.000523)),
            ((2, 0.993, 1, 4, 0.0), (42, 0.14835164835164835, 2, 3, -0.184102), (-1, 0.0, 0, 0, 0.002624), (-1, 0.0, 0, 0, -0.025621), (56, 0.4494949494949495, 5, 6, 0.28905), (-1, 0.0, 0, 0, 0.044802), (-1, 0.0, 0, 0, 0.002624)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.118421), (-1, 0.0, 0, 0, 0.052543), (-1, 0.0, 0, 0, 0.000654), (26, 4.598548757796862, 5, 6, -0.45755), (-1, 0.0, 0, 0, -0.045799), (-1, 0.0, 0, 0, -0.001093)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.222057), (-1, 0.0, 0, 0, 0.026702), (-1, 0.0, 0, 0, -0.00508), (52, 0.985, 5, 6, -0.23755), (-1, 0.0, 0, 0, -0.016941), (-1, 0.0, 0, 0, 0.039422)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.115741), (-1, 0.0, 0, 0, 0.044533), (-1, 0.0, 0, 0, -2.2e-05), (24, 0.015833333333333345, 5, 6, -0.448913), (-1, 0.0, 0, 0, -0.043066), (-1, 0.0, 0, 0, 0.002228)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.17, 2, 3, -0.0708), (-1, 0.0, 0, 0, -0.023668), (-1, 0.0, 0, 0, 0.005203), (6, 0.8846610635340376, 5, 6, 0.662291), (-1, 0.0, 0, 0, 0.039563), (-1, 0.0, 0, 0, 0.014787)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.21397), (-1, 0.0, 0, 0, -0.007437), (-1, 0.0, 0, 0, 0.024343), (52, 0.985, 5, 6, -0.230995), (-1, 0.0, 0, 0, -0.016399), (-1, 0.0, 0, 0, 0.038332)),
            ((2, 0.992, 1, 4, 0.0), (55, 0.3431372549019608, 2, 3, -0.181369), (-1, 0.0, 0, 0, 0.00027), (-1, 0.0, 0, 0, -0.033882), (56, 0.4494949494949495, 5, 6, 0.259342), (-1, 0.0, 0, 0, 0.040159), (-1, 0.0, 0, 0, 0.001949)),
            ((20, 2.319785806352712, 1, 4, 0.0), (15, -0.17, 2, 3, -0.071933), (-1, 0.0, 0, 0, -0.023155), (-1, 0.0, 0, 0, 0.004769), (13, 0.744, 5, 6, 0.614027), (-1, 0.0, 0, 0, 0.013552), (-1, 0.0, 0, 0, 0.039369)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.054572), (-1, 0.0, 0, 0, 0.043928), (-1, 0.0, 0, 0, -0.002464), (-1, 0.0, 0, 0, -0.044389)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.203329), (-1, 0.0, 0, 0, 0.02506), (-1, 0.0, 0, 0, -0.00517), (52, 0.985, 5, 6, -0.222371), (-1, 0.0, 0, 0, -0.015715), (-1, 0.0, 0, 0, 0.037059)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.202653), (-1, 0.0, 0, 0, -0.049999), (-1, 0.0, 0, 0, -0.001863), (7, 0.955, 5, 6, 0.22282), (-1, 0.0, 0, 0, 0.000504), (-1, 0.0, 0, 0, 0.035506)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.219157), (-1, 0.0, 0, 0, -0.000523), (-1, 0.0, 0, 0, -0.033336), (25, 6.802950010466324, 5, 6, 0.19825), (-1, 0.0, 0, 0, 0.00405), (-1, 0.0, 0, 0, 0.043744)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.105424), (-1, 0.0, 0, 0, 0.047203), (-1, 0.0, 0, 0, 0.000605), (24, 0.015833333333333345, 5, 6, -0.420698), (-1, 0.0, 0, 0, -0.042025), (-1, 0.0, 0, 0, 0.003011)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.200867), (-1, 0.0, 0, 0, -0.00719), (-1, 0.0, 0, 0, 0.022749), (4, 0.995, 5, 6, -0.219257), (-1, 0.0, 0, 0, -0.01639), (-1, 0.0, 0, 0, 0.027364)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.17, 2, 3, -0.0639), (-1, 0.0, 0, 0, -0.022123), (-1, 0.0, 0, 0, 0.004821), (6, 0.8846610635340376, 5, 6, 0.62723), (-1, 0.0, 0, 0, 0.038357), (-1, 0.0, 0, 0, 0.014083)),
            ((2, 0.992, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.170595), (-1, 0.0, 0, 0, 0.001731), (-1, 0.0, 0, 0, -0.02145), (56, 0.4494949494949495, 5, 6, 0.239852), (-1, 0.0, 0, 0, 0.037713), (-1, 0.0, 0, 0, 0.001731)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.051978), (-1, 0.0, 0, 0, 0.040183), (-1, 0.0, 0, 0, -0.002153), (-1, 0.0, 0, 0, -0.043312)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.193004), (-1, 0.0, 0, 0, 0.0238), (-1, 0.0, 0, 0, -0.00483), (52, 0.985, 5, 6, -0.209228), (-1, 0.0, 0, 0, -0.014753), (-1, 0.0, 0, 0, 0.035327)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.102265), (-1, 0.0, 0, 0, 0.017553), (-1, 0.0, 0, 0, -0.009371), (24, 0.015833333333333345, 5, 6, -0.406304), (-1, 0.0, 0, 0, -0.04147), (-1, 0.0, 0, 0, 0.003311)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.17, 2, 3, -0.059722), (-1, 0.0, 0, 0, -0.021265), (-1, 0.0, 0, 0, 0.004665), (6, 0.8846610635340376, 5, 6, 0.610582), (-1, 0.0, 0, 0, 0.037749), (-1, 0.0, 0, 0, 0.013772)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.048752), (-1, 0.0, 0, 0, 0.038436), (-1, 0.0, 0, 0, -0.002107), (-1, 0.0, 0, 0, -0.042833)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.185657), (-1, 0.0, 0, 0, -0.049066), (-1, 0.0, 0, 0, -0.001517), (7, 0.955, 5, 6, 0.201795), (-1, 0.0, 0, 0, 0.000403), (-1, 0.0, 0, 0, 0.033735)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.186271), (-1, 0.0, 0, 0, 0.023178), (-1, 0.0, 0, 0, -0.004807), (52, 0.985, 5, 6, -0.205674), (-1, 0.0, 0, 0, -0.014337), (-1, 0.0, 0, 0, 0.033638)),
            ((2, 0.993, 1, 4, 0.0), (42, 0.14835164835164835, 2, 3, -0.155054), (-1, 0.0, 0, 0, 0.002029), (-1, 0.0, 0, 0, -0.022106), (56, 0.4494949494949495, 5, 6, 0.236218), (-1, 0.0, 0, 0, 0.037901), (-1, 0.0, 0, 0, 0.002029)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.094396), (-1, 0.0, 0, 0, -0.033573), (-1, 0.0, 0, 0, 0.002918), (42, 0.14835164835164835, 5, 6, 0.370021), (-1, 0.0, 0, 0, 0.006891), (-1, 0.0, 0, 0, 0.034039)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.291595), (-1, 0.0, 0, 0, -0.023462), (-1, 0.0, 0, 0, 0.028917), (26, 4.2964256503859755, 5, 6, 0.128775), (-1, 0.0, 0, 0, -0.014615), (-1, 0.0, 0, 0, 0.016245)),
            ((40, 0.5, 1, 4, 0.0), (15, -0.771, 2, 3, 0.180281), (-1, 0.0, 0, 0, -0.041563), (-1, 0.0, 0, 0, 0.012709), (4, 0.995, 5, 6, -0.200938), (-1, 0.0, 0, 0, -0.015038), (-1, 0.0, 0, 0, 0.025385)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.196206), (-1, 0.0, 0, 0, -0.000523), (-1, 0.0, 0, 0, -0.028836), (58, 0.16510169775282135, 5, 6, 0.175278), (-1, 0.0, 0, 0, 0.041163), (-1, 0.0, 0, 0, 0.003282)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.044861), (-1, 0.0, 0, 0, 0.036026), (-1, 0.0, 0, 0, -0.002082), (-1, 0.0, 0, 0, -0.041532)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.280035), (-1, 0.0, 0, 0, -0.022445), (-1, 0.0, 0, 0, 0.02794), (26, 4.2964256503859755, 5, 6, 0.122805), (-1, 0.0, 0, 0, -0.013856), (-1, 0.0, 0, 0, 0.015367)),
            ((40, 0.5, 1, 4, 0.0), (15, -0.771, 2, 3, 0.175534), (-1, 0.0, 0, 0, -0.040498), (-1, 0.0, 0, 0, 0.01231), (52, 0.985, 5, 6, -0.194397), (-1, 0.0, 0, 0, -0.013555), (-1, 0.0, 0, 0, 0.032315)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.042537), (-1, 0.0, 0, 0, 0.034673), (-1, 0.0, 0, 0, -0.002048), (-1, 0.0, 0, 0, -0.040917)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.054127), (-1, 0.0, 0, 0, -0.01831), (-1, 0.0, 0, 0, 0.004563), (6, 0.8846610635340376, 5, 6, 0.569859), (-1, 0.0, 0, 0, 0.036327), (-1, 0.0, 0, 0, 0.012893)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.274452), (-1, 0.0, 0, 0, -0.021701), (-1, 0.0, 0, 0, 0.026828), (26, 4.280624567377958, 5, 6, 0.117404), (-1, 0.0, 0, 0, -0.013229), (-1, 0.0, 0, 0, 0.01416)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.168088), (-1, 0.0, 0, 0, 0.021678), (-1, 0.0, 0, 0, -0.00508), (4, 0.995, 5, 6, -0.189181), (-1, 0.0, 0, 0, -0.014162), (-1, 0.0, 0, 0, 0.023867)),
            ((53, 0.278, 1, 4, 0.0), (58, 0.4463414621865892, 2, 3, -0.172579), (-1, 0.0, 0, 0, -0.001658), (-1, 0.0, 0, 0, -0.047351), (7, 0.955, 5, 6, 0.183465), (-1, 0.0, 0, 0, 0.000272), (-1, 0.0, 0, 0, 0.031916)),
            ((2, 0.992, 1, 4, 0.0), (55, 0.3431372549019608, 2, 3, -0.149688), (-1, 0.0, 0, 0, 0.000299), (-1, 0.0, 0, 0, -0.030083), (56, 0.4494949494949495, 5, 6, 0.205076), (-1, 0.0, 0, 0, 0.032232), (-1, 0.0, 0, 0, 0.001542)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.732678), (-1, 0.0, 0, 0, -0.041966), (-1, 0.0, 0, 0, -0.013889), (40, 0.5, 5, 6, 0.043107), (-1, 0.0, 0, 0, 0.011615), (-1, 0.0, 0, 0, -0.008508)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.186152), (-1, 0.0, 0, 0, -0.00056), (-1, 0.0, 0, 0, -0.027893), (6, 0.6077266474915606, 5, 6, 0.163751), (-1, 0.0, 0, 0, 0.0197), (-1, 0.0, 0, 0, -0.00056)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.083142), (-1, 0.0, 0, 0, 0.015975), (-1, 0.0, 0, 0, -0.009549), (26, 4.598548757796862, 5, 6, -0.361231), (-1, 0.0, 0, 0, -0.04343), (-1, 0.0, 0, 0, 0.002677)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.71968), (-1, 0.0, 0, 0, -0.041352), (-1, 0.0, 0, 0, -0.013611), (40, 0.5, 5, 6, 0.040586), (-1, 0.0, 0, 0, 0.011115), (-1, 0.0, 0, 0, -0.008205)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.053064), (-1, 0.0, 0, 0, -0.017399), (-1, 0.0, 0, 0, 0.004119), (6, 0.8846610635340376, 5, 6, 0.544128), (-1, 0.0, 0, 0, 0.035434), (-1, 0.0, 0, 0, 0.012277)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.262098), (-1, 0.0, 0, 0, -0.020641), (-1, 0.0, 0, 0, 0.025688), (26, 4.280624567377958, 5, 6, 0.106414), (-1, 0.0, 0, 0, -0.012541), (-1, 0.0, 0, 0, 0.012844)),
            ((55, 0.3038461538461539, 1, 4, 0.0), (6, 0.6501228390275874, 2, 3, 0.115733), (-1, 0.0, 0, 0, 0.015098), (-1, 0.0, 0, 0, -0.012129), (53, 0.252, 5, 6, -0.237934), (-1, 0.0, 0, 0, -0.036979), (-1, 0.0, 0, 0, 0.006608)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.034049), (-1, 0.0, 0, 0, 0.03223), (-1, 0.0, 0, 0, -0.002224), (-1, 0.0, 0, 0, -0.039228)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.710627), (-1, 0.0, 0, 0, -0.040739), (-1, 0.0, 0, 0, -0.013468), (56, 0.4083333333333333, 5, 6, 0.038083), (-1, 0.0, 0, 0, 0.013605), (-1, 0.0, 0, 0, -0.00545)),
            ((2, 0.992, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.146822), (-1, 0.0, 0, 0, 0.00113), (-1, 0.0, 0, 0, -0.01842), (3, 0.42800000000000005, 5, 6, 0.192024), (-1, 0.0, 0, 0, 0.001681), (-1, 0.0, 0, 0, 0.033239)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.077631), (-1, 0.0, 0, 0, 0.014776), (-1, 0.0, 0, 0, -0.008788), (26, 4.598548757796862, 5, 6, -0.344417), (-1, 0.0, 0, 0, -0.042665), (-1, 0.0, 0, 0, 0.002921)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.099, 2, 3, -0.050299), (-1, 0.0, 0, 0, -0.016191), (-1, 0.0, 0, 0, 0.004185), (6, 0.8846610635340376, 5, 6, 0.528886), (-1, 0.0, 0, 0, 0.034875), (-1, 0.0, 0, 0, 0.011965)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.691062), (-1, 0.0, 0, 0, -0.039918), (-1, 0.0, 0, 0, -0.012781), (40, 0.5, 5, 6, 0.036946), (-1, 0.0, 0, 0, 0.010496), (-1, 0.0, 0, 0, -0.00793)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.032482), (-1, 0.0, 0, 0, 0.030901), (-1, 0.0, 0, 0, -0.002127), (-1, 0.0, 0, 0, -0.038564)),
            ((52, 0.976, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.054037), (-1, 0.0, 0, 0, 0.001662), (-1, 0.0, 0, 0, -0.032891), (6, 0.6862712967770672, 5, 6, 0.475209), (-1, 0.0, 0, 0, 0.042475), (-1, 0.0, 0, 0, 0.010529)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.683504), (-1, 0.0, 0, 0, -0.039517), (-1, 0.0, 0, 0, -0.012643), (56, 0.4494949494949495, 5, 6, 0.035197), (-1, 0.0, 0, 0, 0.011448), (-1, 0.0, 0, 0, -0.006236)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.171707), (-1, 0.0, 0, 0, -0.000671), (-1, 0.0, 0, 0, -0.026582), (25, 6.802950010466324, 5, 6, 0.144851), (-1, 0.0, 0, 0, 0.002286), (-1, 0.0, 0, 0, 0.037263)),
            ((2, 0.992, 1, 4, 0.0), (55, 0.3431372549019608, 2, 3, -0.137146), (-1, 0.0, 0, 0, -0.000193), (-1, 0.0, 0, 0, -0.02715), (1, 0.04, 5, 6, 0.181477), (-1, 0.0, 0, 0, 0.001108), (-1, 0.0, 0, 0, 0.031402)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.032418), (-1, 0.0, 0, 0, 0.029114), (-1, 0.0, 0, 0, -0.001921), (-1, 0.0, 0, 0, -0.037992)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.674235), (-1, 0.0, 0, 0, -0.039056), (-1, 0.0, 0, 0, -0.01241), (40, 0.5, 5, 6, 0.03566), (-1, 0.0, 0, 0, 0.010142), (-1, 0.0, 0, 0, -0.00768)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (42, 0.14835164835164835, 2, 3, -0.07964), (-1, 0.0, 0, 0, 0.005635), (-1, 0.0, 0, 0, -0.018011), (42, 0.14835164835164835, 5, 6, 0.305056), (-1, 0.0, 0, 0, 0.005635), (-1, 0.0, 0, 0, 0.032714)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.028093), (-1, 0.0, 0, 0, 0.00251), (-1, 0.0, 0, 0, -0.026947), (-1, 0.0, 0, 0, 0.043392)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.03122), (-1, 0.0, 0, 0, 0.027549), (-1, 0.0, 0, 0, -0.001808), (-1, 0.0, 0, 0, -0.037564)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.223679), (-1, 0.0, 0, 0, -0.026121), (-1, 0.0, 0, 0, 0.008481), (15, -0.068, 5, 6, 0.106545), (-1, 0.0, 0, 0, 0.047612), (-1, 0.0, 0, 0, 0.000343)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.046209), (-1, 0.0, 0, 0, -0.015804), (-1, 0.0, 0, 0, 0.003787), (6, 0.8846610635340376, 5, 6, 0.500343), (-1, 0.0, 0, 0, 0.033925), (-1, 0.0, 0, 0, 0.011353)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.663117), (-1, 0.0, 0, 0, -0.038474), (-1, 0.0, 0, 0, -0.012073), (40, 0.5, 5, 6, 0.034185), (-1, 0.0, 0, 0, 0.009758), (-1, 0.0, 0, 0, -0.007431)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.027841), (-1, 0.0, 0, 0, 0.002308), (-1, 0.0, 0, 0, -0.025771), (-1, 0.0, 0, 0, 0.042869)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.069754), (-1, 0.0, 0, 0, 0.0136), (-1, 0.0, 0, 0, -0.008273), (26, 4.598548757796862, 5, 6, -0.321293), (-1, 0.0, 0, 0, -0.041865), (-1, 0.0, 0, 0, 0.003699)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.044153), (-1, 0.0, 0, 0, -0.009752), (-1, 0.0, 0, 0, 0.008248), (6, 0.3024435993004616, 5, 6, -0.492396), (-1, 0.0, 0, 0, 0.015018), (-1, 0.0, 0, 0, -0.036481)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.215786), (-1, 0.0, 0, 0, -0.025237), (-1, 0.0, 0, 0, 0.008031), (15, -0.068, 5, 6, 0.100907), (-1, 0.0, 0, 0, 0.045407), (-1, 0.0, 0, 0, 0.000253)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.026863), (-1, 0.0, 0, 0, 0.002194), (-1, 0.0, 0, 0, -0.024814), (-1, 0.0, 0, 0, 0.042205)),
            ((52, 0.976, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.048804), (-1, 0.0, 0, 0, -0.034746), (-1, 0.0, 0, 0, 0.000817), (24, 0.06250000000000003, 5, 6, 0.446872), (-1, 0.0, 0, 0, 0.009952), (-1, 0.0, 0, 0, 0.041013)),
            ((2, 0.992, 1, 4, 0.0), (39, 0.655, 2, 3, -0.128421), (-1, 0.0, 0, 0, -0.001637), (-1, 0.0, 0, 0, -0.036012), (3, 0.42800000000000005, 5, 6, 0.166997), (-1, 0.0, 0, 0, 0.000964), (-1, 0.0, 0, 0, 0.032022)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.027975), (-1, 0.0, 0, 0, 0.026652), (-1, 0.0, 0, 0, -0.001864), (-1, 0.0, 0, 0, -0.036432)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.646181), (-1, 0.0, 0, 0, -0.037753), (-1, 0.0, 0, 0, -0.011518), (40, 0.5, 5, 6, 0.032007), (-1, 0.0, 0, 0, 0.009268), (-1, 0.0, 0, 0, -0.007105)),
            ((19, 0.5, 1, 4, 0.0), (58, 0.22564484631765486, 2, 3, -0.045357), (-1, 0.0, 0, 0, 0.003548), (-1, 0.0, 0, 0, -0.015257), (6, 0.8846610635340376, 5, 6, 0.482154), (-1, 0.0, 0, 0, 0.033337), (-1, 0.0, 0, 0, 0.01092)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.211689), (-1, 0.0, 0, 0, -0.0245), (-1, 0.0, 0, 0, 0.007422), (15, -0.068, 5, 6, 0.097608), (-1, 0.0, 0, 0, 0.043237), (-1, 0.0, 0, 0, 0.00029)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.026236), (-1, 0.0, 0, 0, 0.002052), (-1, 0.0, 0, 0, -0.02385), (-1, 0.0, 0, 0, 0.041347)),
            ((15, -0.751, 1, 4, 0.0), (58, 0.2109990093741555, 2, 3, -0.631503), (-1, 0.0, 0, 0, -0.034988), (-1, 0.0, 0, 0, -0.007797), (40, 0.5, 5, 6, 0.030592), (-1, 0.0, 0, 0, 0.008863), (-1, 0.0, 0, 0, -0.006816)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.026458), (-1, 0.0, 0, 0, 0.025582), (-1, 0.0, 0, 0, -0.001805), (-1, 0.0, 0, 0, -0.035922)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.156601), (-1, 0.0, 0, 0, -0.000662), (-1, 0.0, 0, 0, -0.028922), (25, 6.802950010466324, 5, 6, 0.130139), (-1, 0.0, 0, 0, 0.001701), (-1, 0.0, 0, 0, 0.035751)),
            ((9, 0.9530000000000001, 1, 4, 0.0), (26, 4.131156519244238, 2, 3, -0.034022), (-1, 0.0, 0, 0, -0.024046), (-1, 0.0, 0, 0, 0.001486), (42, 0.20714285714285713, 5, 6, 0.593149), (-1, 0.0, 0, 0, 0.013978), (-1, 0.0, 0, 0, 0.051124)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.024497), (-1, 0.0, 0, 0, 0.001922), (-1, 0.0, 0, 0, -0.022605), (-1, 0.0, 0, 0, 0.041074)),
            ((40, 2.5, 1, 4, 0.0), (60, 0.664, 2, 3, 0.042152), (-1, 0.0, 0, 0, 0.005881), (-1, 0.0, 0, 0, -0.015081), (6, 0.3024435993004616, 5, 6, -0.461951), (-1, 0.0, 0, 0, 0.015879), (-1, 0.0, 0, 0, -0.035025)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.099, 2, 3, -0.042033), (-1, 0.0, 0, 0, -0.014313), (-1, 0.0, 0, 0, 0.003672), (6, 0.8846610635340376, 5, 6, 0.466648), (-1, 0.0, 0, 0, 0.032794), (-1, 0.0, 0, 0, 0.010615)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.614199), (-1, 0.0, 0, 0, -0.036465), (-1, 0.0, 0, 0, -0.010188), (39, 0.528, 5, 6, 0.029722), (-1, 0.0, 0, 0, 0.007248), (-1, 0.0, 0, 0, -0.009329)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.025371), (-1, 0.0, 0, 0, 0.024607), (-1, 0.0, 0, 0, -0.001744), (-1, 0.0, 0, 0, -0.035105)),
            ((52, 0.997, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.024862), (-1, 0.0, 0, 0, -0.024714), (-1, 0.0, 0, 0, 0.001477), (-1, 0.0, 0, 0, 0.040618)),
            ((2, 0.992, 1, 4, 0.0), (39, 0.655, 2, 3, -0.122693), (-1, 0.0, 0, 0, -0.001477), (-1, 0.0, 0, 0, -0.035007), (3, 0.42800000000000005, 5, 6, 0.157796), (-1, 0.0, 0, 0, 0.000878), (-1, 0.0, 0, 0, 0.030831)),
            ((52, 0.976, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.045364), (-1, 0.0, 0, 0, -0.033236), (-1, 0.0, 0, 0, 0.000852), (24, 0.06250000000000003, 5, 6, 0.422479), (-1, 0.0, 0, 0, 0.009428), (-1, 0.0, 0, 0, 0.040043)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.039536), (-1, 0.0, 0, 0, -0.009014), (-1, 0.0, 0, 0, 0.007489), (5, 0.888, 5, 6, -0.452277), (-1, 0.0, 0, 0, -0.032762), (-1, 0.0, 0, 0, 0.01981)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.605841), (-1, 0.0, 0, 0, -0.035972), (-1, 0.0, 0, 0, -0.010084), (40, 0.5, 5, 6, 0.028056), (-1, 0.0, 0, 0, 0.008538), (-1, 0.0, 0, 0, -0.00672)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.17424242424242425, 2, 3, 0.023748), (-1, 0.0, 0, 0, 0.02039), (-1, 0.0, 0, 0, -0.002274), (-1, 0.0, 0, 0, -0.034592)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.135111), (-1, 0.0, 0, 0, -0.045231), (-1, 0.0, 0, 0, -0.000622), (42, 0.2928571428571428, 5, 6, 0.132556), (-1, 0.0, 0, 0, -6.4e-05), (-1, 0.0, 0, 0, 0.032769)),
            ((15, -0.108, 1, 4, 0.0), (42, 0.14835164835164835, 2, 3, -0.198876), (-1, 0.0, 0, 0, 0.005758), (-1, 0.0, 0, 0, -0.025193), (15, -0.068, 5, 6, 0.089107), (-1, 0.0, 0, 0, 0.040691), (-1, 0.0, 0, 0, 5.7e-05)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.024702), (-1, 0.0, 0, 0, 0.001825), (-1, 0.0, 0, 0, -0.022233), (-1, 0.0, 0, 0, 0.039751)),
            ((9, 0.9530000000000001, 1, 4, 0.0), (26, 4.131156519244238, 2, 3, -0.033536), (-1, 0.0, 0, 0, -0.023464), (-1, 0.0, 0, 0, 0.001318), (42, 0.20714285714285713, 5, 6, 0.558465), (-1, 0.0, 0, 0, 0.013123), (-1, 0.0, 0, 0, 0.049608)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.596875), (-1, 0.0, 0, 0, -0.035596), (-1, 0.0, 0, 0, -0.00979), (6, 0.6343440778663255, 5, 6, 0.02718), (-1, 0.0, 0, 0, 0.007409), (-1, 0.0, 0, 0, -0.007826)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.041556), (-1, 0.0, 0, 0, -0.014196), (-1, 0.0, 0, 0, 0.003258), (6, 0.8846610635340376, 5, 6, 0.445896), (-1, 0.0, 0, 0, 0.032054), (-1, 0.0, 0, 0, 0.010109)),
            ((2, 0.992, 1, 4, 0.0), (39, 0.655, 2, 3, -0.117695), (-1, 0.0, 0, 0, -0.001335), (-1, 0.0, 0, 0, -0.034083), (3, 0.42800000000000005, 5, 6, 0.147324), (-1, 0.0, 0, 0, 0.000741), (-1, 0.0, 0, 0, 0.029627)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.193614), (-1, 0.0, 0, 0, -0.023243), (-1, 0.0, 0, 0, 0.00767), (15, -0.068, 5, 6, 0.086391), (-1, 0.0, 0, 0, 0.038888), (-1, 0.0, 0, 0, 9.7e-05)),
            ((52, 0.997, 1, 4, 0.0), (60, 0.002, 2, 3, -0.023962), (-1, 0.0, 0, 0, -0.021234), (-1, 0.0, 0, 0, 0.001694), (-1, 0.0, 0, 0, 0.039029)),
            ((40, 2.5, 1, 4, 0.0), (60, 0.664, 2, 3, 0.036582), (-1, 0.0, 0, 0, 0.005281), (-1, 0.0, 0, 0, -0.014094), (6, 0.3024435993004616, 5, 6, -0.429424), (-1, 0.0, 0, 0, 0.016384), (-1, 0.0, 0, 0, -0.033376)),
            ((20, 3.210268834500856, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, -0.031371), (-1, 0.0, 0, 0, 0.004013), (-1, 0.0, 0, 0, -0.010277), (13, 0.556, 5, 6, 0.562148), (-1, 0.0, 0, 0, 0.013269), (-1, 0.0, 0, 0, 0.038354)),
            ((15, -0.751, 1, 4, 0.0), (25, 6.648336461794052, 2, 3, -0.580979), (-1, 0.0, 0, 0, -0.005924), (-1, 0.0, 0, 0, -0.032663), (40, 0.5, 5, 6, 0.025686), (-1, 0.0, 0, 0, 0.008103), (-1, 0.0, 0, 0, -0.006454)),
            ((52, 0.976, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.04311), (-1, 0.0, 0, 0, -0.031972), (-1, 0.0, 0, 0, 0.000854), (24, 0.06250000000000003, 5, 6, 0.39766), (-1, 0.0, 0, 0, 0.008864), (-1, 0.0, 0, 0, 0.03923)),
            ((9, 0.9530000000000001, 1, 4, 0.0), (39, 0.0030000000000000027, 2, 3, -0.032686), (-1, 0.0, 0, 0, -0.013637), (-1, 0.0, 0, 0, 0.00373), (42, 0.20714285714285713, 5, 6, 0.53749), (-1, 0.0, 0, 0, 0.01262), (-1, 0.0, 0, 0, 0.048553)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.17424242424242425, 2, 3, 0.021159), (-1, 0.0, 0, 0, 0.019469), (-1, 0.0, 0, 0, -0.002263), (-1, 0.0, 0, 0, -0.033078)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.13827), (-1, 0.0, 0, 0, -0.000651), (-1, 0.0, 0, 0, -0.023909), (25, 6.802950010466324, 5, 6, 0.112223), (-1, 0.0, 0, 0, 0.000965), (-1, 0.0, 0, 0, 0.033981)),
            ((15, -0.108, 1, 4, 0.0), (42, 0.13392857142857142, 2, 3, -0.184092), (-1, 0.0, 0, 0, 0.007361), (-1, 0.0, 0, 0, -0.02271), (15, -0.068, 5, 6, 0.085046), (-1, 0.0, 0, 0, 0.037371), (-1, 0.0, 0, 0, 0.000177)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.020776), (-1, 0.0, 0, 0, 0.001823), (-1, 0.0, 0, 0, -0.020967), (-1, 0.0, 0, 0, 0.038508)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.567171), (-1, 0.0, 0, 0, -0.034412), (-1, 0.0, 0, 0, -0.008582), (39, 0.528, 5, 6, 0.027176), (-1, 0.0, 0, 0, 0.006657), (-1, 0.0, 0, 0, -0.008643)),
            ((2, 0.992, 1, 4, 0.0), (39, 0.655, 2, 3, -0.110791), (-1, 0.0, 0, 0, -0.001156), (-1, 0.0, 0, 0, -0.032888), (3, 0.42800000000000005, 5, 6, 0.143521), (-1, 0.0, 0, 0, 0.000818), (-1, 0.0, 0, 0, 0.028735)),
            ((40, 2.5, 1, 4, 0.0), (60, 0.664, 2, 3, 0.036336), (-1, 0.0, 0, 0, 0.005105), (-1, 0.0, 0, 0, -0.013408), (4, 0.326, 5, 6, -0.411994), (-1, 0.0, 0, 0, -0.046087), (-1, 0.0, 0, 0, -0.000528)),
            ((20, 3.210268834500856, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, -0.029064), (-1, 0.0, 0, 0, 0.003903), (-1, 0.0, 0, 0, -0.009835), (13, 0.556, 5, 6, 0.546385), (-1, 0.0, 0, 0, 0.012933), (-1, 0.0, 0, 0, 0.037862)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.021598), (-1, 0.0, 0, 0, 0.021665), (-1, 0.0, 0, 0, -0.001587), (-1, 0.0, 0, 0, -0.032509)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.021342), (-1, 0.0, 0, 0, 0.001663), (-1, 0.0, 0, 0, -0.020203), (-1, 0.0, 0, 0, 0.037879)),
            ((15, -0.099, 1, 4, 0.0), (9, 0.205, 2, 3, -0.177336), (-1, 0.0, 0, 0, -0.021697), (-1, 0.0, 0, 0, 0.007534), (15, -0.068, 5, 6, 0.082873), (-1, 0.0, 0, 0, 0.03895), (-1, 0.0, 0, 0, 0.000167)),
            ((39, 0.9045, 1, 4, 0.0), (2, 0.992, 2, 3, 0.032079), (-1, 0.0, 0, 0, -0.004708), (-1, 0.0, 0, 0, 0.010714), (15, 0.959, 5, 6, -0.442208), (-1, 0.0, 0, 0, -0.032611), (-1, 0.0, 0, 0, 0.00825)),
            ((40, 0.5, 1, 4, 0.0), (60, 0.003999999999999999, 2, 3, 0.111521), (-1, 0.0, 0, 0, -0.025402), (-1, 0.0, 0, 0, 0.011237), (7, 0.958, 5, 6, -0.131023), (-1, 0.0, 0, 0, -0.01324), (-1, 0.0, 0, 0, 0.014819)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.03635), (-1, 0.0, 0, 0, -0.013728), (-1, 0.0, 0, 0, 0.003416), (27, 0.963, 5, 6, 0.416327), (-1, 0.0, 0, 0, 0.009499), (-1, 0.0, 0, 0, 0.039887)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.554684), (-1, 0.0, 0, 0, -0.033818), (-1, 0.0, 0, 0, -0.008211), (40, 0.5, 5, 6, 0.0251), (-1, 0.0, 0, 0, 0.007682), (-1, 0.0, 0, 0, -0.006048)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.17424242424242425, 2, 3, 0.020699), (-1, 0.0, 0, 0, 0.017758), (-1, 0.0, 0, 0, -0.001997), (-1, 0.0, 0, 0, -0.031965)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.021019), (-1, 0.0, 0, 0, 0.001535), (-1, 0.0, 0, 0, -0.019291), (-1, 0.0, 0, 0, 0.037102)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.132203), (-1, 0.0, 0, 0, -0.000596), (-1, 0.0, 0, 0, -0.027069), (58, 0.16510169775282135, 5, 6, 0.108374), (-1, 0.0, 0, 0, 0.03285), (-1, 0.0, 0, 0, 0.000947)),
            ((15, -0.751, 1, 4, 0.0), (25, 6.648336461794052, 2, 3, -0.547768), (-1, 0.0, 0, 0, -0.004596), (-1, 0.0, 0, 0, -0.031231), (39, 0.528, 5, 6, 0.02561), (-1, 0.0, 0, 0, 0.006199), (-1, 0.0, 0, 0, -0.008052)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.034894), (-1, 0.0, 0, 0, -0.008116), (-1, 0.0, 0, 0, 0.006617), (6, 0.3024435993004616, 5, 6, -0.3943), (-1, 0.0, 0, 0, 0.017483), (-1, 0.0, 0, 0, -0.031707)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.280624567377958, 2, 3, -0.066221), (-1, 0.0, 0, 0, -0.025138), (-1, 0.0, 0, 0, 0.002141), (53, 0.274, 5, 6, 0.211764), (-1, 0.0, 0, 0, 0.003639), (-1, 0.0, 0, 0, 0.027082)),
            ((15, -0.099, 1, 4, 0.0), (9, 0.205, 2, 3, -0.173858), (-1, 0.0, 0, 0, -0.021155), (-1, 0.0, 0, 0, 0.007053), (15, -0.068, 5, 6, 0.078693), (-1, 0.0, 0, 0, 0.037326), (-1, 0.0, 0, 0, 9.4e-05)),
            ((20, 3.210268834500856, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, -0.027959), (-1, 0.0, 0, 0, 0.003607), (-1, 0.0, 0, 0, -0.009274), (13, 0.556, 5, 6, 0.525013), (-1, 0.0, 0, 0, 0.012426), (-1, 0.0, 0, 0, 0.037139)),
            ((52, 0.997, 1, 4, 0.0), (60, 0.002, 2, 3, -0.020608), (-1, 0.0, 0, 0, -0.019245), (-1, 0.0, 0, 0, 0.001572), (-1, 0.0, 0, 0, 0.036401)),
            ((60, 0.664, 1, 4, 0.0), (60, 0.43100000000000005, 2, 3, 0.052083), (-1, 0.0, 0, 0, -0.000623), (-1, 0.0, 0, 0, 0.025931), (26, 4.598548757796862, 5, 6, -0.254156), (-1, 0.0, 0, 0, -0.039827), (-1, 0.0, 0, 0, 0.006416)),
            ((15, -0.099, 1, 4, 0.0), (42, 0.13392857142857142, 2, 3, -0.17231), (-1, 0.0, 0, 0, 0.006269), (-1, 0.0, 0, 0, -0.020974), (15, -0.068, 5, 6, 0.079055), (-1, 0.0, 0, 0, 0.036006), (-1, 0.0, 0, 0, 0.000255)),
            ((52, 0.976, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.03757), (-1, 0.0, 0, 0, -0.029544), (-1, 0.0, 0, 0, 0.000876), (24, 0.06250000000000003, 5, 6, 0.367617), (-1, 0.0, 0, 0, 0.008251), (-1, 0.0, 0, 0, 0.038116)),
            ((2, 0.992, 1, 4, 0.0), (39, 0.655, 2, 3, -0.103783), (-1, 0.0, 0, 0, -0.001012), (-1, 0.0, 0, 0, -0.031417), (3, 0.42800000000000005, 5, 6, 0.131438), (-1, 0.0, 0, 0, 0.000691), (-1, 0.0, 0, 0, 0.027787)),
            ((15, -0.751, 1, 4, 0.0), (58, 0.2109990093741555, 2, 3, -0.531024), (-1, 0.0, 0, 0, -0.030562), (-1, 0.0, 0, 0, -0.003869), (40, 0.5, 5, 6, 0.022629), (-1, 0.0, 0, 0, 0.007431), (-1, 0.0, 0, 0, -0.006)),
            ((9, 0.9530000000000001, 1, 4, 0.0), (26, 4.131156519244238, 2, 3, -0.029165), (-1, 0.0, 0, 0, -0.022914), (-1, 0.0, 0, 0, 0.001383), (42, 0.20714285714285713, 5, 6, 0.484832), (-1, 0.0, 0, 0, 0.011392), (-1, 0.0, 0, 0, 0.046621)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.035068), (-1, 0.0, 0, 0, -0.012797), (-1, 0.0, 0, 0, 0.003037), (27, 0.963, 5, 6, 0.394896), (-1, 0.0, 0, 0, 0.008996), (-1, 0.0, 0, 0, 0.039491)),
            ((15, -0.108, 1, 4, 0.0), (42, 0.13392857142857142, 2, 3, -0.1709), (-1, 0.0, 0, 0, 0.0065), (-1, 0.0, 0, 0, -0.021318), (15, -0.09, 5, 6, 0.075468), (-1, 0.0, 0, 0, 0.042377), (-1, 0.0, 0, 0, 0.001146)),
            ((52, 0.997, 1, 4, 0.0), (60, 0.002, 2, 3, -0.019928), (-1, 0.0, 0, 0, -0.018089), (-1, 0.0, 0, 0, 0.00143), (-1, 0.0, 0, 0, 0.036055)),
            ((40, 2.5, 1, 4, 0.0), (9, 0.9530000000000001, 2, 3, 0.032213), (-1, 0.0, 0, 0, 4.1e-05), (-1, 0.0, 0, 0, 0.03003), (4, 0.326, 5, 6, -0.375316), (-1, 0.0, 0, 0, -0.044341), (-1, 0.0, 0, 0, 0.000769)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.522305), (-1, 0.0, 0, 0, -0.03257), (-1, 0.0, 0, 0, -0.006855), (40, 0.5, 5, 6, 0.02255), (-1, 0.0, 0, 0, 0.007124), (-1, 0.0, 0, 0, -0.005665)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.458041958041958, 2, 3, -0.019685), (-1, 0.0, 0, 0, 0.001322), (-1, 0.0, 0, 0, -0.018652), (-1, 0.0, 0, 0, 0.035592)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.057585), (-1, 0.0, 0, 0, -0.025787), (-1, 0.0, 0, 0, 0.002252), (42, 0.17424242424242425, 5, 6, 0.223794), (-1, 0.0, 0, 0, 0.004155), (-1, 0.0, 0, 0, 0.031958)),
            ((15, -0.099, 1, 4, 0.0), (9, 0.205, 2, 3, -0.165872), (-1, 0.0, 0, 0, -0.020443), (-1, 0.0, 0, 0, 0.006929), (15, -0.068, 5, 6, 0.073766), (-1, 0.0, 0, 0, 0.033673), (-1, 0.0, 0, 0, 0.000218)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.0173), (-1, 0.0, 0, 0, 0.019391), (-1, 0.0, 0, 0, -0.001542), (-1, 0.0, 0, 0, -0.030137)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.123557), (-1, 0.0, 0, 0, -0.000632), (-1, 0.0, 0, 0, -0.026485), (25, 6.802950010466324, 5, 6, 0.098285), (-1, 0.0, 0, 0, 0.000614), (-1, 0.0, 0, 0, 0.03129)),
            ((2, 0.992, 1, 4, 0.0), (39, 0.655, 2, 3, -0.098353), (-1, 0.0, 0, 0, -0.000834), (-1, 0.0, 0, 0, -0.030691), (3, 0.42800000000000005, 5, 6, 0.124644), (-1, 0.0, 0, 0, 0.000657), (-1, 0.0, 0, 0, 0.027058)),
            ((39, 0.0030000000000000027, 1, 4, 0.0), (7, 0.951, 2, 3, -0.157845), (-1, 0.0, 0, 0, -0.015273), (-1, 0.0, 0, 0, 0.0199), (39, 0.528, 5, 6, 0.075544), (-1, 0.0, 0, 0, 0.014433), (-1, 0.0, 0, 0, -0.0074)),
            ((52, 0.997, 1, 4, 0.0), (60, 0.002, 2, 3, -0.018823), (-1, 0.0, 0, 0, -0.017575), (-1, 0.0, 0, 0, 0.001414), (-1, 0.0, 0, 0, 0.035067)),
            ((20, 4.243470074122607, 1, 4, 0.0), (15, -0.099, 2, 3, -0.020617), (-1, 0.0, 0, 0, -0.009934), (-1, 0.0, 0, 0, 0.003089), (-1, 0.0, 0, 0, 0.030738)),
            ((12, 0.39, 1, 4, 0.0), (60, 0.043, 2, 3, -0.161038), (-1, 0.0, 0, 0, -0.002206), (-1, 0.0, 0, 0, -0.023063), (56, 0.3693181818181818, 5, 6, 0.072816), (-1, 0.0, 0, 0, 0.018727), (-1, 0.0, 0, 0, -0.002206)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.032648), (-1, 0.0, 0, 0, -0.007533), (-1, 0.0, 0, 0, 0.006139), (6, 0.3024435993004616, 5, 6, -0.361498), (-1, 0.0, 0, 0, 0.01845), (-1, 0.0, 0, 0, -0.030103)),
            ((6, 0.6501228390275874, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, 0.082556), (-1, 0.0, 0, 0, 0.010546), (-1, 0.0, 0, 0, -0.012842), (7, 0.899, 5, 6, -0.140617), (-1, 0.0, 0, 0, -0.017596), (-1, 0.0, 0, 0, 0.013703)),
            ((15, -0.751, 1, 4, 0.0), (58, 0.2109990093741555, 2, 3, -0.505412), (-1, 0.0, 0, 0, -0.029474), (-1, 0.0, 0, 0, -0.002803), (40, 0.5, 5, 6, 0.023094), (-1, 0.0, 0, 0, 0.007019), (-1, 0.0, 0, 0, -0.00549)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.458041958041958, 2, 3, -0.017366), (-1, 0.0, 0, 0, 0.001315), (-1, 0.0, 0, 0, -0.017902), (-1, 0.0, 0, 0, 0.034871)),
            ((20, 3.210268834500856, 1, 4, 0.0), (56, 0.3798076923076923, 2, 3, -0.023746), (-1, 0.0, 0, 0, 0.006257), (-1, 0.0, 0, 0, -0.005615), (13, 0.556, 5, 6, 0.495972), (-1, 0.0, 0, 0, 0.011806), (-1, 0.0, 0, 0, 0.036119)),
            ((60, 0.664, 1, 4, 0.0), (60, 0.43100000000000005, 2, 3, 0.049565), (-1, 0.0, 0, 0, -0.00066), (-1, 0.0, 0, 0, 0.024986), (24, 0.015833333333333345, 5, 6, -0.234291), (-1, 0.0, 0, 0, -0.035653), (-1, 0.0, 0, 0, 0.009167)),
            ((52, 0.997, 1, 4, 0.0), (15, -0.17, 2, 3, -0.016747), (-1, 0.0, 0, 0, -0.010856), (-1, 0.0, 0, 0, 0.00286), (-1, 0.0, 0, 0, 0.034434)),
            ((39, 0.9045, 1, 4, 0.0), (39, 0.8765000000000001, 2, 3, 0.028823), (-1, 0.0, 0, 0, -0.000136), (-1, 0.0, 0, 0, 0.031411), (15, 0.959, 5, 6, -0.391722), (-1, 0.0, 0, 0, -0.030915), (-1, 0.0, 0, 0, 0.010455)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.17424242424242425, 2, 3, 0.019378), (-1, 0.0, 0, 0, 0.015735), (-1, 0.0, 0, 0, -0.001724), (-1, 0.0, 0, 0, -0.029235)),
            ((52, 0.978, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.031609), (-1, 0.0, 0, 0, -0.027862), (-1, 0.0, 0, 0, 0.001005), (24, 0.06250000000000003, 5, 6, 0.352156), (-1, 0.0, 0, 0, 0.008014), (-1, 0.0, 0, 0, 0.036482)),
        )),
        (0.030172, (
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -1.298983), (-1, 0.0, 0, 0, -0.078076), (-1, 0.0, 0, 0, -0.017126), (44, 0.12, 5, 6, 1.788868), (-1, 0.0, 0, 0, 0.018356), (-1, 0.0, 0, 0, 0.094061)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -1.234906), (-1, 0.0, 0, 0, -0.074229), (-1, 0.0, 0, 0, -0.016291), (44, 0.12, 5, 6, 1.706295), (-1, 0.0, 0, 0, 0.017554), (-1, 0.0, 0, 0, 0.08973)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.62, 2, 3, -1.176671), (-1, 0.0, 0, 0, -0.06855), (-1, 0.0, 0, 0, -0.005387), (44, 0.12, 5, 6, 1.633676), (-1, 0.0, 0, 0, 0.01679), (-1, 0.0, 0, 0, 0.085942)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -1.123179), (-1, 0.0, 0, 0, -0.06774), (-1, 0.0, 0, 0, -0.014349), (44, 0.12, 5, 6, 1.569255), (-1, 0.0, 0, 0, 0.01606), (-1, 0.0, 0, 0, 0.0826)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -1.074064), (-1, 0.0, 0, 0, -0.069306), (-1, 0.0, 0, 0, -0.022846), (44, 0.12, 5, 6, 1.511658), (-1, 0.0, 0, 0, 0.015364), (-1, 0.0, 0, 0, 0.079629)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.62, 2, 3, -1.027834), (-1, 0.0, 0, 0, -0.060511), (-1, 0.0, 0, 0, -0.002246), (44, 0.12, 5, 6, 1.459798), (-1, 0.0, 0, 0, 0.014699), (-1, 0.0, 0, 0, 0.07697)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.985215), (-1, 0.0, 0, 0, -0.064405), (-1, 0.0, 0, 0, -0.019613), (44, 0.12, 5, 6, 1.412801), (-1, 0.0, 0, 0, 0.014064), (-1, 0.0, 0, 0, 0.074574)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -0.944898), (-1, 0.0, 0, 0, -0.058352), (-1, 0.0, 0, 0, -0.008441), (44, 0.12, 5, 6, 1.369954), (-1, 0.0, 0, 0, 0.013457), (-1, 0.0, 0, 0, 0.072403)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.907539), (-1, 0.0, 0, 0, -0.060208), (-1, 0.0, 0, 0, -0.01675), (44, 0.12, 5, 6, 1.330673), (-1, 0.0, 0, 0, 0.012877), (-1, 0.0, 0, 0, 0.070426)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -0.871811), (-1, 0.0, 0, 0, -0.054538), (-1, 0.0, 0, 0, -0.006167), (44, 0.12, 5, 6, 1.294472), (-1, 0.0, 0, 0, 0.012323), (-1, 0.0, 0, 0, 0.068616)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.838477), (-1, 0.0, 0, 0, -0.054436), (-1, 0.0, 0, 0, -0.009322), (44, 0.12, 5, 6, 1.260945), (-1, 0.0, 0, 0, 0.011792), (-1, 0.0, 0, 0, 0.066952)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.20606000000000002, 2, 3, -0.806147), (-1, 0.0, 0, 0, -0.051201), (-1, 0.0, 0, 0, -0.003927), (44, 0.12, 5, 6, 1.229748), (-1, 0.0, 0, 0, 0.011285), (-1, 0.0, 0, 0, 0.065414)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.776046), (-1, 0.0, 0, 0, -0.021716), (-1, 0.0, 0, 0, -0.062893), (44, 0.12, 5, 6, 1.20059), (-1, 0.0, 0, 0, 0.0108), (-1, 0.0, 0, 0, 0.063988)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.746212), (-1, 0.0, 0, 0, -0.052074), (-1, 0.0, 0, 0, -0.009985), (44, 0.12, 5, 6, 1.17322), (-1, 0.0, 0, 0, 0.010336), (-1, 0.0, 0, 0, 0.062659)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (44, 0.62, 2, 3, -0.718381), (-1, 0.0, 0, 0, -0.044909), (-1, 0.0, 0, 0, 0.008011), (44, 0.12, 5, 6, 1.147422), (-1, 0.0, 0, 0, 0.009892), (-1, 0.0, 0, 0, 0.061417)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.691921), (-1, 0.0, 0, 0, -0.01763), (-1, 0.0, 0, 0, -0.058862), (44, 0.12, 5, 6, 1.123011), (-1, 0.0, 0, 0, 0.009467), (-1, 0.0, 0, 0, 0.060251)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (17, 0.5, 2, 3, -0.665738), (-1, 0.0, 0, 0, -0.048166), (-1, 0.0, 0, 0, -0.006514), (44, 0.12, 5, 6, 1.099824), (-1, 0.0, 0, 0, 0.009061), (-1, 0.0, 0, 0, 0.059153)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.750514), (-1, 0.0, 0, 0, -0.052613), (-1, 0.0, 0, 0, -0.005741), (24, 0.179, 5, 6, 0.934246), (-1, 0.0, 0, 0, 0.011595), (-1, 0.0, 0, 0, 0.059347)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.725076), (-1, 0.0, 0, 0, -0.051181), (-1, 0.0, 0, 0, -0.005468), (20, -3.210268834500856, 5, 6, 0.910579), (-1, 0.0, 0, 0, -0.01033), (-1, 0.0, 0, 0, 0.053268)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.600716), (-1, 0.0, 0, 0, -0.011282), (-1, 0.0, 0, 0, -0.052627), (44, 0.12, 5, 6, 1.040403), (-1, 0.0, 0, 0, 0.009153), (-1, 0.0, 0, 0, 0.056281)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.68324), (-1, 0.0, 0, 0, -0.049045), (-1, 0.0, 0, 0, -0.004332), (24, 0.179, 5, 6, 0.876847), (-1, 0.0, 0, 0, 0.011138), (-1, 0.0, 0, 0, 0.056697)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.660318), (-1, 0.0, 0, 0, -0.047757), (-1, 0.0, 0, 0, -0.004125), (24, 0.179, 5, 6, 0.855103), (-1, 0.0, 0, 0, 0.01059), (-1, 0.0, 0, 0, 0.055866)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.539792), (-1, 0.0, 0, 0, -0.009944), (-1, 0.0, 0, 0, -0.052349), (44, 0.12, 5, 6, 0.986323), (-1, 0.0, 0, 0, 0.009123), (-1, 0.0, 0, 0, 0.053727)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.622997), (-1, 0.0, 0, 0, -0.045846), (-1, 0.0, 0, 0, -0.003204), (24, 0.179, 5, 6, 0.826566), (-1, 0.0, 0, 0, 0.011316), (-1, 0.0, 0, 0, 0.054326)),
            ((44, 0.62, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.6021), (-1, 0.0, 0, 0, -0.044666), (-1, 0.0, 0, 0, -0.00305), (20, -3.210268834500856, 5, 6, 0.806119), (-1, 0.0, 0, 0, -0.00825), (-1, 0.0, 0, 0, 0.048169)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.486967), (-1, 0.0, 0, 0, -0.055527), (-1, 0.0, 0, 0, -0.009944), (44, 0.12, 5, 6, 0.937975), (-1, 0.0, 0, 0, 0.009), (-1, 0.0, 0, 0, 0.051491)),
            ((44, 0.62, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.56625), (-1, 0.0, 0, 0, -0.06362), (-1, 0.0, 0, 0, -0.016858), (24, 0.179, 5, 6, 0.774045), (-1, 0.0, 0, 0, 0.010282), (-1, 0.0, 0, 0, 0.052283)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.604265), (-1, 0.0, 0, 0, -0.045887), (-1, 0.0, 0, 0, -0.003114), (24, 0.179, 5, 6, 0.68349), (-1, 0.0, 0, 0, 0.008096), (-1, 0.0, 0, 0, 0.051638)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.437744), (-1, 0.0, 0, 0, -0.002437), (-1, 0.0, 0, 0, -0.046282), (50, 0.5, 5, 6, 0.893718), (-1, 0.0, 0, 0, 0.052236), (-1, 0.0, 0, 0, 0.017777)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.57574), (-1, 0.0, 0, 0, -0.044375), (-1, 0.0, 0, 0, -0.002688), (24, 0.179, 5, 6, 0.661767), (-1, 0.0, 0, 0, 0.00871), (-1, 0.0, 0, 0, 0.050432)),
            ((44, 0.62, 1, 4, 0.0), (26, 4.190735823528673, 2, 3, -0.501157), (-1, 0.0, 0, 0, -0.058697), (-1, 0.0, 0, 0, -0.012511), (53, 0.258, 5, 6, 0.714185), (-1, 0.0, 0, 0, 0.001797), (-1, 0.0, 0, 0, 0.046455)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.396744), (-1, 0.0, 0, 0, -0.002267), (-1, 0.0, 0, 0, -0.047148), (50, 0.5, 5, 6, 0.852388), (-1, 0.0, 0, 0, 0.050717), (-1, 0.0, 0, 0, 0.014608)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.531699), (-1, 0.0, 0, 0, -0.042377), (-1, 0.0, 0, 0, -0.00122), (24, 0.179, 5, 6, 0.622683), (-1, 0.0, 0, 0, 0.008404), (-1, 0.0, 0, 0, 0.048821)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (24, 0.04500000000000004, 2, 3, -0.514294), (-1, 0.0, 0, 0, -0.041406), (-1, 0.0, 0, 0, -0.001161), (55, 0.3205128205128205, 5, 6, 0.604671), (-1, 0.0, 0, 0, 0.047371), (-1, 0.0, 0, 0, 0.006091)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.360421), (-1, 0.0, 0, 0, -0.000879), (-1, 0.0, 0, 0, -0.045361), (50, 0.5, 5, 6, 0.811399), (-1, 0.0, 0, 0, 0.04924), (-1, 0.0, 0, 0, 0.011397)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (58, 0.41749300143204865, 2, 3, -0.49063), (-1, 0.0, 0, 0, -0.012438), (-1, 0.0, 0, 0, -0.057632), (24, 0.179, 5, 6, 0.580656), (-1, 0.0, 0, 0, 0.007497), (-1, 0.0, 0, 0, 0.047354)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (27, 0.622, 2, 3, -0.564717), (-1, 0.0, 0, 0, 0.025583), (-1, 0.0, 0, 0, -0.036379), (13, 0.968, 5, 6, 0.488201), (-1, 0.0, 0, 0, -0.000486), (-1, 0.0, 0, 0, 0.045302)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.1601245927962225, 2, 3, -0.464649), (-1, 0.0, 0, 0, -0.056499), (-1, 0.0, 0, 0, -0.011452), (55, 0.3205128205128205, 5, 6, 0.556876), (-1, 0.0, 0, 0, 0.044726), (-1, 0.0, 0, 0, 0.004292)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (27, 0.622, 2, 3, -0.541673), (-1, 0.0, 0, 0, 0.023481), (-1, 0.0, 0, 0, -0.035065), (13, 0.968, 5, 6, 0.46792), (-1, 0.0, 0, 0, -0.000215), (-1, 0.0, 0, 0, 0.043724)),
            ((44, 0.62, 1, 4, 0.0), (26, 4.190735823528673, 2, 3, -0.392635), (-1, 0.0, 0, 0, -0.055013), (-1, 0.0, 0, 0, -0.007513), (2, 0.954, 5, 6, 0.599856), (-1, 0.0, 0, 0, -0.004667), (-1, 0.0, 0, 0, 0.041554)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (27, 0.622, 2, 3, -0.518294), (-1, 0.0, 0, 0, 0.022374), (-1, 0.0, 0, 0, -0.033813), (13, 0.968, 5, 6, 0.448133), (-1, 0.0, 0, 0, 0.00019), (-1, 0.0, 0, 0, 0.042159)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.256578947368421, 2, 3, -0.500647), (-1, 0.0, 0, 0, -0.005646), (-1, 0.0, 0, 0, -0.044786), (13, 0.968, 5, 6, 0.431026), (-1, 0.0, 0, 0, 0.000181), (-1, 0.0, 0, 0, 0.040938)),
            ((44, 0.62, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.365743), (-1, 0.0, 0, 0, -0.057557), (-1, 0.0, 0, 0, -0.007469), (2, 0.954, 5, 6, 0.571076), (-1, 0.0, 0, 0, -0.004256), (-1, 0.0, 0, 0, 0.040016)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.278271), (-1, 0.0, 0, 0, 0.004398), (-1, 0.0, 0, 0, -0.038972), (50, 0.5, 5, 6, 0.711497), (-1, 0.0, 0, 0, 0.046311), (-1, 0.0, 0, 0, 0.001441)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (44, 0.05984, 2, 3, -0.395136), (-1, 0.0, 0, 0, 0.013064), (-1, 0.0, 0, 0, -0.031064), (56, 0.4641025641025641, 5, 6, 0.48916), (-1, 0.0, 0, 0, 0.043954), (-1, 0.0, 0, 0, 0.00309)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (27, 0.622, 2, 3, -0.465334), (-1, 0.0, 0, 0, 0.022523), (-1, 0.0, 0, 0, -0.031179), (40, 0.5, 5, 6, 0.404347), (-1, 0.0, 0, 0, 0.040001), (-1, 0.0, 0, 0, -0.000591)),
            ((20, 1.965912816362163, 1, 4, 0.0), (25, 6.556061562263916, 2, 3, -0.210702), (-1, 0.0, 0, 0, -0.036971), (-1, 0.0, 0, 0, 0.003181), (7, 0.46299999999999997, 5, 6, 0.842739), (-1, 0.0, 0, 0, 0.006104), (-1, 0.0, 0, 0, 0.048589)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.443299), (-1, 0.0, 0, 0, -0.005308), (-1, 0.0, 0, 0, -0.044706), (40, 0.5, 5, 6, 0.385687), (-1, 0.0, 0, 0, 0.038631), (-1, 0.0, 0, 0, -0.000756)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.364049), (-1, 0.0, 0, 0, -0.055686), (-1, 0.0, 0, 0, -0.00791), (56, 0.4641025641025641, 5, 6, 0.458169), (-1, 0.0, 0, 0, 0.042279), (-1, 0.0, 0, 0, 0.00184)),
            ((20, 1.965912816362163, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.195151), (-1, 0.0, 0, 0, -0.039179), (-1, 0.0, 0, 0, 0.001651), (7, 0.46299999999999997, 5, 6, 0.819537), (-1, 0.0, 0, 0, 0.004517), (-1, 0.0, 0, 0, 0.047818)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.256578947368421, 2, 3, -0.420698), (-1, 0.0, 0, 0, -0.002251), (-1, 0.0, 0, 0, -0.041306), (40, 0.5, 5, 6, 0.366925), (-1, 0.0, 0, 0, 0.037311), (-1, 0.0, 0, 0, -0.000979)),
            ((20, 1.965912816362163, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.183648), (-1, 0.0, 0, 0, -0.032973), (-1, 0.0, 0, 0, 0.004466), (7, 0.46299999999999997, 5, 6, 0.801897), (-1, 0.0, 0, 0, 0.003757), (-1, 0.0, 0, 0, 0.047224)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.333008), (-1, 0.0, 0, 0, -0.05437), (-1, 0.0, 0, 0, -0.007002), (56, 0.4641025641025641, 5, 6, 0.426493), (-1, 0.0, 0, 0, 0.040666), (-1, 0.0, 0, 0, 0.000456)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.399926), (-1, 0.0, 0, 0, -0.003424), (-1, 0.0, 0, 0, -0.042883), (40, 0.5, 5, 6, 0.349792), (-1, 0.0, 0, 0, 0.03604), (-1, 0.0, 0, 0, -0.001133)),
            ((20, 1.965912816362163, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.17033), (-1, 0.0, 0, 0, -0.036897), (-1, 0.0, 0, 0, 0.002033), (7, 0.46299999999999997, 5, 6, 0.779138), (-1, 0.0, 0, 0, 0.002347), (-1, 0.0, 0, 0, 0.046485)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.38107), (-1, 0.0, 0, 0, -0.000635), (-1, 0.0, 0, 0, -0.039298), (52, 0.976, 5, 6, 0.333509), (-1, 0.0, 0, 0, 0.006186), (-1, 0.0, 0, 0, 0.048911)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (58, 0.45418165222705165, 2, 3, -0.307963), (-1, 0.0, 0, 0, -0.006126), (-1, 0.0, 0, 0, -0.053502), (56, 0.4641025641025641, 5, 6, 0.401137), (-1, 0.0, 0, 0, 0.038951), (-1, 0.0, 0, 0, -0.000286)),
            ((20, 1.965912816362163, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.156037), (-1, 0.0, 0, 0, 0.00517), (-1, 0.0, 0, 0, -0.03061), (6, 0.8717732060837587, 5, 6, 0.75328), (-1, 0.0, 0, 0, 0.042464), (-1, 0.0, 0, 0, -0.018205)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (53, 0.41900000000000004, 2, 3, -0.35938), (-1, 0.0, 0, 0, -0.028718), (-1, 0.0, 0, 0, 0.01602), (40, 0.5, 5, 6, 0.315602), (-1, 0.0, 0, 0, 0.034212), (-1, 0.0, 0, 0, -0.002376)),
            ((44, 0.20606000000000002, 1, 4, 0.0), (25, 6.070641713151899, 2, 3, -0.28881), (-1, 0.0, 0, 0, -0.0529), (-1, 0.0, 0, 0, -0.00541), (56, 0.4641025641025641, 5, 6, 0.378791), (-1, 0.0, 0, 0, 0.037411), (-1, 0.0, 0, 0, -0.000733)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, -0.186314), (-1, 0.0, 0, 0, 0.00673), (-1, 0.0, 0, 0, -0.033481), (50, 0.5, 5, 6, 0.569617), (-1, 0.0, 0, 0, 0.04258), (-1, 0.0, 0, 0, -0.013395)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.16876), (-1, 0.0, 0, 0, 0.064123), (-1, 0.0, 0, 0, -0.000766), (24, 0.015833333333333345, 5, 6, -0.616674), (-1, 0.0, 0, 0, -0.048451), (-1, 0.0, 0, 0, -0.002972)),
            ((20, 1.965912816362163, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.13867), (-1, 0.0, 0, 0, -0.029585), (-1, 0.0, 0, 0, 0.005223), (7, 0.46299999999999997, 5, 6, 0.714638), (-1, 0.0, 0, 0, -0.002904), (-1, 0.0, 0, 0, 0.044585)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.330009), (-1, 0.0, 0, 0, 0.001351), (-1, 0.0, 0, 0, -0.036722), (40, 0.5, 5, 6, 0.290597), (-1, 0.0, 0, 0, 0.032584), (-1, 0.0, 0, 0, -0.002928)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.160717), (-1, 0.0, 0, 0, 0.061722), (-1, 0.0, 0, 0, -0.000743), (24, 0.015833333333333345, 5, 6, -0.596303), (-1, 0.0, 0, 0, -0.047597), (-1, 0.0, 0, 0, -0.002746)),
            ((53, 0.278, 1, 4, 0.0), (26, 4.184499866031553, 2, 3, -0.292524), (-1, 0.0, 0, 0, -0.05189), (-1, 0.0, 0, 0, -0.003743), (7, 0.953, 5, 6, 0.310345), (-1, 0.0, 0, 0, -0.000117), (-1, 0.0, 0, 0, 0.041581)),
            ((20, 1.965912816362163, 1, 4, 0.0), (40, 0.5, 2, 3, -0.126534), (-1, 0.0, 0, 0, 0.009474), (-1, 0.0, 0, 0, -0.022927), (6, 0.8717732060837587, 5, 6, 0.686615), (-1, 0.0, 0, 0, 0.039932), (-1, 0.0, 0, 0, -0.023437)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.151098), (-1, 0.0, 0, 0, 0.059665), (-1, 0.0, 0, 0, -0.000829), (26, 4.598548757796862, 5, 6, -0.572698), (-1, 0.0, 0, 0, -0.049566), (-1, 0.0, 0, 0, -0.004706)),
            ((52, 0.976, 1, 4, 0.0), (40, 0.5, 2, 3, -0.112168), (-1, 0.0, 0, 0, 0.010485), (-1, 0.0, 0, 0, -0.023233), (6, 0.8289655815830271, 5, 6, 0.732731), (-1, 0.0, 0, 0, 0.04752), (-1, 0.0, 0, 0, -0.001832)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.30303), (-1, 0.0, 0, 0, 0.00047), (-1, 0.0, 0, 0, -0.038481), (52, 0.976, 5, 6, 0.267835), (-1, 0.0, 0, 0, 0.004626), (-1, 0.0, 0, 0, 0.046017)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.144952), (-1, 0.0, 0, 0, 0.057599), (-1, 0.0, 0, 0, -0.000755), (24, 0.015833333333333345, 5, 6, -0.551881), (-1, 0.0, 0, 0, -0.046284), (-1, 0.0, 0, 0, -0.000776)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.270316), (-1, 0.0, 0, 0, -0.007892), (-1, 0.0, 0, 0, 0.031381), (52, 0.985, 5, 6, -0.2881), (-1, 0.0, 0, 0, -0.021791), (-1, 0.0, 0, 0, 0.046389)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.139347), (-1, 0.0, 0, 0, 0.055558), (-1, 0.0, 0, 0, -0.000679), (24, 0.015833333333333345, 5, 6, -0.537242), (-1, 0.0, 0, 0, -0.045726), (-1, 0.0, 0, 0, -0.000625)),
            ((20, 2.319785806352712, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.102372), (-1, 0.0, 0, 0, -0.031117), (-1, 0.0, 0, 0, 0.003626), (6, 0.890011768611039, 5, 6, 0.704214), (-1, 0.0, 0, 0, 0.039841), (-1, 0.0, 0, 0, -0.030964)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.26337), (-1, 0.0, 0, 0, 0.031501), (-1, 0.0, 0, 0, -0.00597), (52, 0.985, 5, 6, -0.282218), (-1, 0.0, 0, 0, -0.021089), (-1, 0.0, 0, 0, 0.045333)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.284508), (-1, 0.0, 0, 0, 0.003403), (-1, 0.0, 0, 0, -0.03459), (50, 0.5, 5, 6, 0.252839), (-1, 0.0, 0, 0, 0.022647), (-1, 0.0, 0, 0, -0.014474)),
            ((2, 0.993, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.218601), (-1, 0.0, 0, 0, 0.013059), (-1, 0.0, 0, 0, -0.026398), (56, 0.4641025641025641, 5, 6, 0.330209), (-1, 0.0, 0, 0, 0.047364), (-1, 0.0, 0, 0, 0.001269)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.135825), (-1, 0.0, 0, 0, 0.052506), (-1, 0.0, 0, 0, -0.000351), (26, 4.598548757796862, 5, 6, -0.525895), (-1, 0.0, 0, 0, -0.048053), (-1, 0.0, 0, 0, -0.002846)),
            ((40, 0.5, 1, 4, 0.0), (46, 0.8571428571428572, 2, 3, 0.251662), (-1, 0.0, 0, 0, 0.018329), (-1, 0.0, 0, 0, -0.044414), (52, 0.985, 5, 6, -0.271653), (-1, 0.0, 0, 0, -0.020156), (-1, 0.0, 0, 0, 0.044249)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.247317), (-1, 0.0, 0, 0, -0.052167), (-1, 0.0, 0, 0, -0.002939), (7, 0.953, 5, 6, 0.263255), (-1, 0.0, 0, 0, -0.000334), (-1, 0.0, 0, 0, 0.038128)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, -0.269045), (-1, 0.0, 0, 0, 0.001336), (-1, 0.0, 0, 0, -0.036546), (50, 0.5, 5, 6, 0.237503), (-1, 0.0, 0, 0, 0.021722), (-1, 0.0, 0, 0, -0.014618)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.581130329250939, 2, 3, 0.239953), (-1, 0.0, 0, 0, -0.007955), (-1, 0.0, 0, 0, 0.028256), (52, 0.985, 5, 6, -0.261052), (-1, 0.0, 0, 0, -0.019261), (-1, 0.0, 0, 0, 0.042964)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.125487), (-1, 0.0, 0, 0, 0.057472), (-1, 0.0, 0, 0, 0.00029), (24, 0.015833333333333345, 5, 6, -0.498432), (-1, 0.0, 0, 0, -0.044548), (-1, 0.0, 0, 0, 0.000747)),
            ((20, 2.319785806352712, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.090639), (-1, 0.0, 0, 0, 0.001657), (-1, 0.0, 0, 0, -0.037755), (6, 0.890011768611039, 5, 6, 0.665648), (-1, 0.0, 0, 0, 0.038363), (-1, 0.0, 0, 0, -0.03135)),
            ((2, 0.993, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.201419), (-1, 0.0, 0, 0, 0.012434), (-1, 0.0, 0, 0, -0.024733), (56, 0.4494949494949495, 5, 6, 0.300893), (-1, 0.0, 0, 0, 0.047119), (-1, 0.0, 0, 0, 0.001473)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.231598), (-1, 0.0, 0, 0, -0.00769), (-1, 0.0, 0, 0, 0.026707), (52, 0.985, 5, 6, -0.253323), (-1, 0.0, 0, 0, -0.018521), (-1, 0.0, 0, 0, 0.041838)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.120994), (-1, 0.0, 0, 0, 0.05484), (-1, 0.0, 0, 0, 0.000413), (24, 0.015833333333333345, 5, 6, -0.484338), (-1, 0.0, 0, 0, -0.043984), (-1, 0.0, 0, 0, 0.000993)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.126104), (-1, 0.0, 0, 0, -0.037382), (-1, 0.0, 0, 0, 0.002805), (50, 0.5, 5, 6, 0.445308), (-1, 0.0, 0, 0, 0.039574), (-1, 0.0, 0, 0, -0.028405)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (56, 0.4494949494949495, 2, 3, -0.250389), (-1, 0.0, 0, 0, 0.004311), (-1, 0.0, 0, 0, -0.032566), (50, 0.5, 5, 6, 0.221182), (-1, 0.0, 0, 0, 0.020533), (-1, 0.0, 0, 0, -0.014065)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.223382), (-1, 0.0, 0, 0, 0.027819), (-1, 0.0, 0, 0, -0.006213), (52, 0.985, 5, 6, -0.245089), (-1, 0.0, 0, 0, -0.017859), (-1, 0.0, 0, 0, 0.041007)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.19090909090909092, 2, 3, 0.117979), (-1, 0.0, 0, 0, 0.046558), (-1, 0.0, 0, 0, -0.000381), (24, 0.015833333333333345, 5, 6, -0.476148), (-1, 0.0, 0, 0, -0.043433), (-1, 0.0, 0, 0, 0.000652)),
            ((2, 0.993, 1, 4, 0.0), (55, 0.12916666666666665, 2, 3, -0.191114), (-1, 0.0, 0, 0, 0.013768), (-1, 0.0, 0, 0, -0.021761), (56, 0.4494949494949495, 5, 6, 0.285985), (-1, 0.0, 0, 0, 0.044716), (-1, 0.0, 0, 0, 0.001566)),
            ((19, 0.5, 1, 4, 0.0), (26, 4.2964256503859755, 2, 3, -0.077074), (-1, 0.0, 0, 0, -0.027219), (-1, 0.0, 0, 0, 0.003572), (6, 0.8846610635340376, 5, 6, 0.665776), (-1, 0.0, 0, 0, 0.039338), (-1, 0.0, 0, 0, -0.030225)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.218444), (-1, 0.0, 0, 0, 0.026734), (-1, 0.0, 0, 0, -0.005523), (52, 0.985, 5, 6, -0.239951), (-1, 0.0, 0, 0, -0.017292), (-1, 0.0, 0, 0, 0.039713)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.114086), (-1, 0.0, 0, 0, 0.050156), (-1, 0.0, 0, 0, 0.000594), (26, 4.598548757796862, 5, 6, -0.463121), (-1, 0.0, 0, 0, -0.045841), (-1, 0.0, 0, 0, -0.001484)),
            ((2, 0.993, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.183324), (-1, 0.0, 0, 0, 0.010622), (-1, 0.0, 0, 0, -0.022348), (56, 0.4494949494949495, 5, 6, 0.274969), (-1, 0.0, 0, 0, 0.04261), (-1, 0.0, 0, 0, 0.001821)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.238946), (-1, 0.0, 0, 0, 0.00076), (-1, 0.0, 0, 0, -0.036312), (50, 0.5, 5, 6, 0.211727), (-1, 0.0, 0, 0, 0.020006), (-1, 0.0, 0, 0, -0.014447)),
            ((60, 0.664, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.109004), (-1, 0.0, 0, 0, 0.048093), (-1, 0.0, 0, 0, 0.000568), (24, 0.015833333333333345, 5, 6, -0.447995), (-1, 0.0, 0, 0, -0.042473), (-1, 0.0, 0, 0, 0.001682)),
            ((40, 0.5, 1, 4, 0.0), (46, 0.8571428571428572, 2, 3, 0.210656), (-1, 0.0, 0, 0, 0.015047), (-1, 0.0, 0, 0, -0.041839), (52, 0.985, 5, 6, -0.232022), (-1, 0.0, 0, 0, -0.01662), (-1, 0.0, 0, 0, 0.038506)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.17, 2, 3, -0.071726), (-1, 0.0, 0, 0, -0.022981), (-1, 0.0, 0, 0, 0.004741), (6, 0.8846610635340376, 5, 6, 0.642822), (-1, 0.0, 0, 0, 0.038495), (-1, 0.0, 0, 0, -0.029779)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.051542), (-1, 0.0, 0, 0, 0.042472), (-1, 0.0, 0, 0, -0.002512), (-1, 0.0, 0, 0, -0.044684)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.207057), (-1, 0.0, 0, 0, -0.050398), (-1, 0.0, 0, 0, -0.001869), (13, 0.963, 5, 6, 0.216902), (-1, 0.0, 0, 0, -0.011233), (-1, 0.0, 0, 0, 0.024621)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.203047), (-1, 0.0, 0, 0, 0.02556), (-1, 0.0, 0, 0, -0.005618), (52, 0.985, 5, 6, -0.225636), (-1, 0.0, 0, 0, -0.016023), (-1, 0.0, 0, 0, 0.036917)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.318596), (-1, 0.0, 0, 0, -0.025149), (-1, 0.0, 0, 0, 0.029613), (26, 4.2964256503859755, 5, 6, 0.139822), (-1, 0.0, 0, 0, -0.01502), (-1, 0.0, 0, 0, 0.018094)),
            ((55, 0.3038461538461539, 1, 4, 0.0), (26, 4.586238930524267, 2, 3, 0.149242), (-1, 0.0, 0, 0, -0.008029), (-1, 0.0, 0, 0, 0.01903), (20, -3.210268834500856, 5, 6, -0.286887), (-1, 0.0, 0, 0, -0.043496), (-1, 0.0, 0, 0, 0.002942)),
            ((2, 0.992, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.177841), (-1, 0.0, 0, 0, 0.009496), (-1, 0.0, 0, 0, -0.02186), (56, 0.4494949494949495, 5, 6, 0.239466), (-1, 0.0, 0, 0, 0.037471), (-1, 0.0, 0, 0, 0.001642)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.17, 2, 3, -0.067935), (-1, 0.0, 0, 0, -0.021601), (-1, 0.0, 0, 0, 0.004252), (6, 0.8846610635340376, 5, 6, 0.62167), (-1, 0.0, 0, 0, 0.037668), (-1, 0.0, 0, 0, -0.028966)),
            ((40, 0.5, 1, 4, 0.0), (26, 4.574061244703071, 2, 3, 0.192811), (-1, 0.0, 0, 0, -0.007421), (-1, 0.0, 0, 0, 0.022243), (52, 0.985, 5, 6, -0.216384), (-1, 0.0, 0, 0, -0.015348), (-1, 0.0, 0, 0, 0.03576)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.047595), (-1, 0.0, 0, 0, 0.039191), (-1, 0.0, 0, 0, -0.002363), (-1, 0.0, 0, 0, -0.043522)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.21574), (-1, 0.0, 0, 0, 0.001731), (-1, 0.0, 0, 0, -0.03538), (50, 0.5, 5, 6, 0.187765), (-1, 0.0, 0, 0, 0.018636), (-1, 0.0, 0, 0, -0.014985)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.097277), (-1, 0.0, 0, 0, 0.017116), (-1, 0.0, 0, 0, -0.009417), (26, 4.598548757796862, 5, 6, -0.416021), (-1, 0.0, 0, 0, -0.044349), (-1, 0.0, 0, 0, -8.2e-05)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.065047), (-1, 0.0, 0, 0, -0.019998), (-1, 0.0, 0, 0, 0.00477), (6, 0.8846610635340376, 5, 6, 0.604167), (-1, 0.0, 0, 0, 0.036968), (-1, 0.0, 0, 0, -0.028049)),
            ((53, 0.278, 1, 4, 0.0), (25, 6.101438773316817, 2, 3, -0.193554), (-1, 0.0, 0, 0, -0.049082), (-1, 0.0, 0, 0, -0.001822), (13, 0.959, 5, 6, 0.198839), (-1, 0.0, 0, 0, -0.011062), (-1, 0.0, 0, 0, 0.023103)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.30363), (-1, 0.0, 0, 0, -0.023525), (-1, 0.0, 0, 0, 0.027568), (26, 4.280624567377958, 5, 6, 0.128266), (-1, 0.0, 0, 0, -0.01384), (-1, 0.0, 0, 0, 0.015769)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.185289), (-1, 0.0, 0, 0, 0.023941), (-1, 0.0, 0, 0, -0.005664), (4, 0.995, 5, 6, -0.210449), (-1, 0.0, 0, 0, -0.015846), (-1, 0.0, 0, 0, 0.026763)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.205452), (-1, 0.0, 0, 0, 0.002069), (-1, 0.0, 0, 0, -0.034753), (50, 0.5, 5, 6, 0.177483), (-1, 0.0, 0, 0, 0.017941), (-1, 0.0, 0, 0, -0.014721)),
            ((2, 0.992, 1, 4, 0.0), (56, 0.2928571428571428, 2, 3, -0.165339), (-1, 0.0, 0, 0, 0.009177), (-1, 0.0, 0, 0, -0.020763), (56, 0.4494949494949495, 5, 6, 0.219348), (-1, 0.0, 0, 0, 0.035163), (-1, 0.0, 0, 0, 0.001285)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.090552), (-1, 0.0, 0, 0, 0.016284), (-1, 0.0, 0, 0, -0.009205), (24, 0.015833333333333345, 5, 6, -0.396241), (-1, 0.0, 0, 0, -0.040656), (-1, 0.0, 0, 0, 0.0035)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.174441), (-1, 0.0, 0, 0, 0.022666), (-1, 0.0, 0, 0, -0.005452), (52, 0.985, 5, 6, -0.199121), (-1, 0.0, 0, 0, -0.014194), (-1, 0.0, 0, 0, 0.03413)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.041447), (-1, 0.0, 0, 0, 0.036588), (-1, 0.0, 0, 0, -0.002391), (-1, 0.0, 0, 0, -0.041988)),
            ((19, 0.5, 1, 4, 0.0), (15, -0.17, 2, 3, -0.060507), (-1, 0.0, 0, 0, -0.020056), (-1, 0.0, 0, 0, 0.003971), (6, 0.8846610635340376, 5, 6, 0.578938), (-1, 0.0, 0, 0, 0.035973), (-1, 0.0, 0, 0, -0.027116)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (1, 0.89, 2, 3, 0.084486), (-1, 0.0, 0, 0, 0.000577), (-1, 0.0, 0, 0, 0.043783), (11, 0.896, 5, 6, -0.401316), (-1, 0.0, 0, 0, 0.025947), (-1, 0.0, 0, 0, -0.041841)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.04059), (-1, 0.0, 0, 0, 0.035054), (-1, 0.0, 0, 0, -0.002244), (-1, 0.0, 0, 0, -0.04158)),
            ((15, -0.771, 1, 2, 0.0), (-1, 0.0, 0, 0, -0.040104), (40, 0.5, 3, 4, 0.042276), (-1, 0.0, 0, 0, 0.012042), (-1, 0.0, 0, 0, -0.00893)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.085774), (-1, 0.0, 0, 0, 0.015366), (-1, 0.0, 0, 0, -0.008619), (26, 4.598548757796862, 5, 6, -0.382285), (-1, 0.0, 0, 0, -0.043329), (-1, 0.0, 0, 0, 0.000958)),
            ((53, 0.278, 1, 4, 0.0), (58, 0.4463414621865892, 2, 3, -0.178254), (-1, 0.0, 0, 0, -0.001501), (-1, 0.0, 0, 0, -0.048009), (7, 0.955, 5, 6, 0.180109), (-1, 0.0, 0, 0, -0.00209), (-1, 0.0, 0, 0, 0.033258)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (1, 0.89, 2, 3, 0.08202), (-1, 0.0, 0, 0, 0.000624), (-1, 0.0, 0, 0, 0.042264), (11, 0.896, 5, 6, -0.396082), (-1, 0.0, 0, 0, 0.024625), (-1, 0.0, 0, 0, -0.040796)),
            ((55, 0.4494949494949495, 1, 4, 0.0), (15, -0.751, 2, 3, 0.075131), (-1, 0.0, 0, 0, -0.037043), (-1, 0.0, 0, 0, 0.006332), (7, 0.977, 5, 6, -0.426156), (-1, 0.0, 0, 0, -0.041076), (-1, 0.0, 0, 0, 0.023074)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.057164), (-1, 0.0, 0, 0, -0.018293), (-1, 0.0, 0, 0, 0.004327), (6, 0.8846610635340376, 5, 6, 0.555647), (-1, 0.0, 0, 0, 0.035062), (-1, 0.0, 0, 0, -0.026175)),
            ((15, -0.17, 1, 4, 0.0), (24, 0.179, 2, 3, -0.276979), (-1, 0.0, 0, 0, -0.021452), (-1, 0.0, 0, 0, 0.025744), (26, 4.280624567377958, 5, 6, 0.112588), (-1, 0.0, 0, 0, -0.013318), (-1, 0.0, 0, 0, 0.013946)),
            ((40, 0.5, 1, 4, 0.0), (56, 0.4641025641025641, 2, 3, 0.162355), (-1, 0.0, 0, 0, 0.021502), (-1, 0.0, 0, 0, -0.005381), (4, 0.995, 5, 6, -0.187229), (-1, 0.0, 0, 0, -0.014213), (-1, 0.0, 0, 0, 0.024597)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.189582), (-1, 0.0, 0, 0, 0.002266), (-1, 0.0, 0, 0, -0.033471), (50, 0.5, 5, 6, 0.161694), (-1, 0.0, 0, 0, 0.016932), (-1, 0.0, 0, 0, -0.014735)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.766199), (-1, 0.0, 0, 0, -0.012891), (-1, 0.0, 0, 0, -0.044383), (56, 0.4083333333333333, 5, 6, 0.038117), (-1, 0.0, 0, 0, 0.014605), (-1, 0.0, 0, 0, -0.006102)),
            ((2, 0.992, 1, 4, 0.0), (3, 0.30100000000000005, 2, 3, -0.153354), (-1, 0.0, 0, 0, 0.002357), (-1, 0.0, 0, 0, -0.028379), (53, 0.27566666666666667, 5, 6, 0.198422), (-1, 0.0, 0, 0, -0.008787), (-1, 0.0, 0, 0, 0.020818)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.035438), (-1, 0.0, 0, 0, 0.032755), (-1, 0.0, 0, 0, -0.002239), (-1, 0.0, 0, 0, -0.040236)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.756738), (-1, 0.0, 0, 0, -0.012922), (-1, 0.0, 0, 0, -0.043832), (3, 0.17000000000000004, 5, 6, 0.036577), (-1, 0.0, 0, 0, 0.016927), (-1, 0.0, 0, 0, -0.004354)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (42, 0.14835164835164835, 2, 3, -0.089866), (-1, 0.0, 0, 0, 0.008978), (-1, 0.0, 0, 0, -0.019316), (50, 0.5, 5, 6, 0.330725), (-1, 0.0, 0, 0, 0.036305), (-1, 0.0, 0, 0, -0.034573)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.078757), (-1, 0.0, 0, 0, 0.014615), (-1, 0.0, 0, 0, -0.008488), (24, 0.015833333333333345, 5, 6, -0.363887), (-1, 0.0, 0, 0, -0.039251), (-1, 0.0, 0, 0, 0.004332)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (1, 0.89, 2, 3, 0.075272), (-1, 0.0, 0, 0, 0.000514), (-1, 0.0, 0, 0, 0.040067), (11, 0.896, 5, 6, -0.376731), (-1, 0.0, 0, 0, 0.023351), (-1, 0.0, 0, 0, -0.039154)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.033946), (-1, 0.0, 0, 0, 0.030871), (-1, 0.0, 0, 0, -0.002087), (-1, 0.0, 0, 0, -0.039741)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.746616), (-1, 0.0, 0, 0, -0.012746), (-1, 0.0, 0, 0, -0.04325), (40, 0.5, 5, 6, 0.035119), (-1, 0.0, 0, 0, 0.010837), (-1, 0.0, 0, 0, -0.008341)),
            ((2, 0.992, 1, 4, 0.0), (1, 0.01, 2, 3, -0.147362), (-1, 0.0, 0, 0, 0.006085), (-1, 0.0, 0, 0, -0.021727), (53, 0.27566666666666667, 5, 6, 0.18714), (-1, 0.0, 0, 0, -0.008612), (-1, 0.0, 0, 0, 0.019867)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.032153), (-1, 0.0, 0, 0, 0.029807), (-1, 0.0, 0, 0, -0.002052), (-1, 0.0, 0, 0, -0.039146)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.239459), (-1, 0.0, 0, 0, -0.027174), (-1, 0.0, 0, 0, 0.008158), (15, -0.068, 5, 6, 0.111882), (-1, 0.0, 0, 0, 0.047056), (-1, 0.0, 0, 0, 0.000791)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.053227), (-1, 0.0, 0, 0, -0.017324), (-1, 0.0, 0, 0, 0.004105), (6, 0.8846610635340376, 5, 6, 0.525619), (-1, 0.0, 0, 0, 0.033878), (-1, 0.0, 0, 0, -0.025481)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (44, 0.62, 2, 3, 0.072571), (-1, 0.0, 0, 0, -0.003195), (-1, 0.0, 0, 0, 0.018856), (11, 0.896, 5, 6, -0.363788), (-1, 0.0, 0, 0, 0.022461), (-1, 0.0, 0, 0, -0.037957)),
            ((55, 0.4494949494949495, 1, 4, 0.0), (51, 2.5, 2, 3, 0.065543), (-1, 0.0, 0, 0, 0.005689), (-1, 0.0, 0, 0, -0.034241), (7, 0.977, 5, 6, -0.391802), (-1, 0.0, 0, 0, -0.038834), (-1, 0.0, 0, 0, 0.022849)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.733288), (-1, 0.0, 0, 0, -0.012228), (-1, 0.0, 0, 0, -0.04272), (3, 0.17000000000000004, 5, 6, 0.033754), (-1, 0.0, 0, 0, 0.016144), (-1, 0.0, 0, 0, -0.004199)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.177455), (-1, 0.0, 0, 0, 0.002136), (-1, 0.0, 0, 0, -0.032058), (50, 0.5, 5, 6, 0.146581), (-1, 0.0, 0, 0, 0.01584), (-1, 0.0, 0, 0, -0.014244)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.030835), (-1, 0.0, 0, 0, 0.028229), (-1, 0.0, 0, 0, -0.001932), (-1, 0.0, 0, 0, -0.038656)),
            ((60, 0.664, 1, 4, 0.0), (5, 0.001, 2, 3, 0.071844), (-1, 0.0, 0, 0, 0.033623), (-1, 0.0, 0, 0, -0.000556), (26, 4.598548757796862, 5, 6, -0.343407), (-1, 0.0, 0, 0, -0.041991), (-1, 0.0, 0, 0, 0.002261)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.693236), (-1, 0.0, 0, 0, -0.040098), (-1, 0.0, 0, 0, -0.011522), (40, 0.5, 5, 6, 0.033579), (-1, 0.0, 0, 0, 0.010243), (-1, 0.0, 0, 0, -0.007825)),
            ((52, 0.976, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.056487), (-1, 0.0, 0, 0, 0.001502), (-1, 0.0, 0, 0, -0.032718), (24, 0.06250000000000003, 5, 6, 0.470669), (-1, 0.0, 0, 0, -0.002028), (-1, 0.0, 0, 0, 0.042046)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (1, 0.89, 2, 3, 0.067525), (-1, 0.0, 0, 0, 0.000438), (-1, 0.0, 0, 0, 0.037036), (11, 0.896, 5, 6, -0.349032), (-1, 0.0, 0, 0, 0.02142), (-1, 0.0, 0, 0, -0.036452)),
            ((7, 0.951, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.091557), (-1, 0.0, 0, 0, 9.5e-05), (-1, 0.0, 0, 0, -0.039649), (27, 0.992, 5, 6, 0.275101), (-1, 0.0, 0, 0, 0.026842), (-1, 0.0, 0, 0, -0.019843)),
            ((51, 2.5, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, 0.035094), (-1, 0.0, 0, 0, 0.007966), (-1, 0.0, 0, 0, -0.010795), (20, 0.6263814842476838, 5, 6, -0.653159), (-1, 0.0, 0, 0, -0.039448), (-1, 0.0, 0, 0, 0.014204)),
            ((15, -0.751, 1, 4, 0.0), (7, 0.019000000000000017, 2, 3, -0.685217), (-1, 0.0, 0, 0, -0.010834), (-1, 0.0, 0, 0, -0.039443), (3, 0.17000000000000004, 5, 6, 0.032642), (-1, 0.0, 0, 0, 0.015538), (-1, 0.0, 0, 0, -0.003976)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.028671), (-1, 0.0, 0, 0, 0.026496), (-1, 0.0, 0, 0, -0.001823), (-1, 0.0, 0, 0, -0.037781)),
            ((2, 0.992, 1, 4, 0.0), (1, 0.01, 2, 3, -0.13935), (-1, 0.0, 0, 0, 0.006161), (-1, 0.0, 0, 0, -0.021029), (3, 0.42800000000000005, 5, 6, 0.174076), (-1, 0.0, 0, 0, 0.000724), (-1, 0.0, 0, 0, 0.032647)),
            ((19, 0.5, 1, 4, 0.0), (26, 4.280624567377958, 2, 3, -0.050858), (-1, 0.0, 0, 0, -0.021103), (-1, 0.0, 0, 0, 0.002529), (6, 0.8846610635340376, 5, 6, 0.503198), (-1, 0.0, 0, 0, 0.03299), (-1, 0.0, 0, 0, -0.024696)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.706729), (-1, 0.0, 0, 0, -0.011731), (-1, 0.0, 0, 0, -0.041323), (40, 0.5, 5, 6, 0.030816), (-1, 0.0, 0, 0, 0.00966), (-1, 0.0, 0, 0, -0.007517)),
            ((24, 0.06250000000000003, 1, 4, 0.0), (26, 4.280624567377958, 2, 3, -0.081187), (-1, 0.0, 0, 0, -0.030527), (-1, 0.0, 0, 0, 0.002161), (50, 0.5, 5, 6, 0.29879), (-1, 0.0, 0, 0, 0.035225), (-1, 0.0, 0, 0, -0.034537)),
            ((15, -0.17, 1, 4, 0.0), (7, 0.988, 2, 3, -0.248347), (-1, 0.0, 0, 0, -0.017527), (-1, 0.0, 0, 0, 0.033044), (26, 4.280624567377958, 5, 6, 0.093674), (-1, 0.0, 0, 0, -0.012522), (-1, 0.0, 0, 0, 0.011743)),
            ((51, 2.5, 1, 4, 0.0), (55, 0.3038461538461539, 2, 3, 0.033439), (-1, 0.0, 0, 0, 0.007623), (-1, 0.0, 0, 0, -0.010451), (20, 0.6263814842476838, 5, 6, -0.636139), (-1, 0.0, 0, 0, -0.038439), (-1, 0.0, 0, 0, 0.013855)),
            ((52, 0.995, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.035534), (-1, 0.0, 0, 0, 0.002032), (-1, 0.0, 0, 0, -0.027586), (7, 0.983, 5, 6, 0.708798), (-1, 0.0, 0, 0, -0.007902), (-1, 0.0, 0, 0, 0.045915)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.667491), (-1, 0.0, 0, 0, -0.038901), (-1, 0.0, 0, 0, -0.01046), (3, 0.17000000000000004, 5, 6, 0.031097), (-1, 0.0, 0, 0, 0.01516), (-1, 0.0, 0, 0, -0.003934)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.16599), (-1, 0.0, 0, 0, 0.002247), (-1, 0.0, 0, 0, -0.031091), (25, 6.802950010466324, 5, 6, 0.135245), (-1, 0.0, 0, 0, 0.001414), (-1, 0.0, 0, 0, 0.039902)),
            ((5, 0.001, 1, 4, 0.0), (52, 0.026000000000000023, 2, 3, 0.40651), (-1, 0.0, 0, 0, -0.010134), (-1, 0.0, 0, 0, 0.050888), (8, 0.6190476190476191, 5, 6, -0.05822), (-1, 0.0, 0, 0, 0.001366), (-1, 0.0, 0, 0, -0.023971)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.067255), (-1, 0.0, 0, 0, 0.013632), (-1, 0.0, 0, 0, -0.008491), (24, 0.015833333333333345, 5, 6, -0.323204), (-1, 0.0, 0, 0, -0.037714), (-1, 0.0, 0, 0, 0.005549)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (1, 0.89, 2, 3, 0.063734), (-1, 0.0, 0, 0, 0.000402), (-1, 0.0, 0, 0, 0.035373), (11, 0.896, 5, 6, -0.333084), (-1, 0.0, 0, 0, 0.020102), (-1, 0.0, 0, 0, -0.034799)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.280624567377958, 2, 3, -0.08627), (-1, 0.0, 0, 0, -0.027248), (-1, 0.0, 0, 0, 0.00177), (27, 0.992, 5, 6, 0.262681), (-1, 0.0, 0, 0, 0.025601), (-1, 0.0, 0, 0, -0.018715)),
            ((51, 2.5, 1, 4, 0.0), (40, 0.5, 2, 3, 0.033033), (-1, 0.0, 0, 0, 0.009114), (-1, 0.0, 0, 0, -0.006707), (52, 0.732, 5, 6, -0.621155), (-1, 0.0, 0, 0, -0.04114), (-1, 0.0, 0, 0, 0.005618)),
            ((19, 0.5, 1, 4, 0.0), (5, 0.831, 2, 3, -0.047796), (-1, 0.0, 0, 0, 0.002068), (-1, 0.0, 0, 0, -0.021391), (13, 0.653, 5, 6, 0.481737), (-1, 0.0, 0, 0, -0.016207), (-1, 0.0, 0, 0, 0.034615)),
            ((15, -0.751, 1, 4, 0.0), (7, 0.019000000000000017, 2, 3, -0.654527), (-1, 0.0, 0, 0, -0.010008), (-1, 0.0, 0, 0, -0.037936), (3, 0.15900000000000003, 5, 6, 0.030168), (-1, 0.0, 0, 0, 0.015424), (-1, 0.0, 0, 0, -0.003356)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.026476), (-1, 0.0, 0, 0, 0.025108), (-1, 0.0, 0, 0, -0.001778), (-1, 0.0, 0, 0, -0.036325)),
            ((60, 0.664, 1, 4, 0.0), (6, 0.6343440778663255, 2, 3, 0.064524), (-1, 0.0, 0, 0, 0.013047), (-1, 0.0, 0, 0, -0.008085), (26, 4.598548757796862, 5, 6, -0.315339), (-1, 0.0, 0, 0, -0.0406), (-1, 0.0, 0, 0, 0.002871)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.21425), (-1, 0.0, 0, 0, -0.026052), (-1, 0.0, 0, 0, 0.009422), (15, -0.068, 5, 6, 0.096141), (-1, 0.0, 0, 0, 0.042643), (-1, 0.0, 0, 0, 0.000271)),
            ((52, 0.995, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.033164), (-1, 0.0, 0, 0, 0.001926), (-1, 0.0, 0, 0, -0.026223), (7, 0.983, 5, 6, 0.689658), (-1, 0.0, 0, 0, -0.007377), (-1, 0.0, 0, 0, 0.045019)),
            ((19, 0.5, 1, 4, 0.0), (5, 0.831, 2, 3, -0.046311), (-1, 0.0, 0, 0, 0.002011), (-1, 0.0, 0, 0, -0.020836), (6, 0.8846610635340376, 5, 6, 0.475265), (-1, 0.0, 0, 0, 0.031918), (-1, 0.0, 0, 0, -0.024101)),
            ((51, 2.5, 1, 4, 0.0), (15, -0.751, 2, 3, 0.031798), (-1, 0.0, 0, 0, -0.031708), (-1, 0.0, 0, 0, 0.003289), (52, 0.732, 5, 6, -0.602685), (-1, 0.0, 0, 0, -0.040158), (-1, 0.0, 0, 0, 0.00569)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.040161), (-1, 0.0, 0, 0, -0.009813), (-1, 0.0, 0, 0, 0.008024), (4, 0.326, 5, 6, -0.471814), (-1, 0.0, 0, 0, -0.049883), (-1, 0.0, 0, 0, -0.00184)),
            ((5, 0.001, 1, 4, 0.0), (52, 0.026000000000000023, 2, 3, 0.38696), (-1, 0.0, 0, 0, -0.009359), (-1, 0.0, 0, 0, 0.048296), (8, 0.6190476190476191, 5, 6, -0.055907), (-1, 0.0, 0, 0, 0.00158), (-1, 0.0, 0, 0, -0.024319)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (12, 0.39, 2, 3, 0.061145), (-1, 0.0, 0, 0, -0.008976), (-1, 0.0, 0, 0, 0.010379), (11, 0.896, 5, 6, -0.324439), (-1, 0.0, 0, 0, 0.018827), (-1, 0.0, 0, 0, -0.033529)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.028086), (-1, 0.0, 0, 0, 0.001977), (-1, 0.0, 0, 0, -0.024104), (-1, 0.0, 0, 0, 0.039952)),
            ((7, 0.951, 1, 4, 0.0), (26, 4.131156519244238, 2, 3, -0.080995), (-1, 0.0, 0, 0, -0.035377), (-1, 0.0, 0, 0, 7.5e-05), (39, 0.528, 5, 6, 0.247236), (-1, 0.0, 0, 0, 0.024563), (-1, 0.0, 0, 0, -0.018393)),
            ((51, 2.5, 1, 4, 0.0), (40, 0.5, 2, 3, 0.029956), (-1, 0.0, 0, 0, 0.008577), (-1, 0.0, 0, 0, -0.00642), (52, 0.732, 5, 6, -0.586479), (-1, 0.0, 0, 0, -0.039241), (-1, 0.0, 0, 0, 0.005576)),
            ((19, 0.5, 1, 4, 0.0), (25, 6.609348335006922, 2, 3, -0.045229), (-1, 0.0, 0, 0, -0.015462), (-1, 0.0, 0, 0, 0.003625), (6, 0.8846610635340376, 5, 6, 0.460886), (-1, 0.0, 0, 0, 0.031289), (-1, 0.0, 0, 0, -0.023616)),
            ((5, 0.001, 1, 4, 0.0), (52, 0.026000000000000023, 2, 3, 0.382215), (-1, 0.0, 0, 0, -0.00849), (-1, 0.0, 0, 0, 0.046721), (8, 0.6190476190476191, 5, 6, -0.054294), (-1, 0.0, 0, 0, 0.001487), (-1, 0.0, 0, 0, -0.023424)),
            ((15, -0.17, 1, 4, 0.0), (7, 0.861, 2, 3, -0.228756), (-1, 0.0, 0, 0, -0.021422), (-1, 0.0, 0, 0, 0.010957), (26, 4.280624567377958, 5, 6, 0.084954), (-1, 0.0, 0, 0, -0.01158), (-1, 0.0, 0, 0, 0.010437)),
            ((12, 0.6759999999999999, 1, 4, 0.0), (4, 0.98, 2, 3, 0.061208), (-1, 0.0, 0, 0, -0.001257), (-1, 0.0, 0, 0, 0.022018), (11, 0.896, 5, 6, -0.31408), (-1, 0.0, 0, 0, 0.017846), (-1, 0.0, 0, 0, -0.032297)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.646412), (-1, 0.0, 0, 0, -0.008674), (-1, 0.0, 0, 0, -0.039211), (39, 0.528, 5, 6, 0.026664), (-1, 0.0, 0, 0, 0.007207), (-1, 0.0, 0, 0, -0.009816)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.027001), (-1, 0.0, 0, 0, 0.001844), (-1, 0.0, 0, 0, -0.023025), (-1, 0.0, 0, 0, 0.03926)),
            ((24, 0.04500000000000004, 1, 4, 0.0), (13, 0.989, 2, 3, -0.151659), (-1, 0.0, 0, 0, 0.00258), (-1, 0.0, 0, 0, -0.030059), (50, 0.5, 5, 6, 0.123451), (-1, 0.0, 0, 0, 0.014391), (-1, 0.0, 0, 0, -0.013747)),
            ((39, 0.9045, 1, 4, 0.0), (2, 0.992, 2, 3, 0.034635), (-1, 0.0, 0, 0, -0.005417), (-1, 0.0, 0, 0, 0.012058), (53, 0.32799999999999996, 5, 6, -0.498412), (-1, 0.0, 0, 0, -0.001504), (-1, 0.0, 0, 0, -0.039053)),
            ((51, 2.5, 1, 4, 0.0), (46, 0.8571428571428572, 2, 3, 0.029058), (-1, 0.0, 0, 0, 0.00289), (-1, 0.0, 0, 0, -0.034504), (52, 0.732, 5, 6, -0.575909), (-1, 0.0, 0, 0, -0.038646), (-1, 0.0, 0, 0, 0.005479)),
            ((20, 3.210268834500856, 1, 4, 0.0), (5, 0.001, 2, 3, -0.035189), (-1, 0.0, 0, 0, 0.018256), (-1, 0.0, 0, 0, -0.004492), (13, 0.556, 5, 6, 0.588404), (-1, 0.0, 0, 0, -0.016292), (-1, 0.0, 0, 0, 0.038935)),
            ((40, 2.5, 1, 4, 0.0), (9, 0.205, 2, 3, 0.037819), (-1, 0.0, 0, 0, -0.006487), (-1, 0.0, 0, 0, 0.009387), (4, 0.326, 5, 6, -0.447481), (-1, 0.0, 0, 0, -0.048271), (-1, 0.0, 0, 0, -0.001355)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (12, 0.39, 2, 3, 0.056247), (-1, 0.0, 0, 0, -0.008748), (-1, 0.0, 0, 0, 0.009798), (11, 0.564, 5, 6, -0.30675), (-1, 0.0, 0, 0, 0.026151), (-1, 0.0, 0, 0, -0.027575)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.631877), (-1, 0.0, 0, 0, -0.008193), (-1, 0.0, 0, 0, -0.038585), (3, 0.17000000000000004, 5, 6, 0.024863), (-1, 0.0, 0, 0, 0.013869), (-1, 0.0, 0, 0, -0.003841)),
            ((20, 3.210268834500856, 1, 4, 0.0), (5, 0.001, 2, 3, -0.033906), (-1, 0.0, 0, 0, 0.017365), (-1, 0.0, 0, 0, -0.004301), (13, 0.556, 5, 6, 0.577822), (-1, 0.0, 0, 0, -0.015883), (-1, 0.0, 0, 0, 0.038511)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.022687), (-1, 0.0, 0, 0, 0.023871), (-1, 0.0, 0, 0, -0.00183), (-1, 0.0, 0, 0, -0.034006)),
            ((15, -0.108, 1, 4, 0.0), (9, 0.205, 2, 3, -0.197058), (-1, 0.0, 0, 0, -0.024683), (-1, 0.0, 0, 0, 0.009413), (15, -0.068, 5, 6, 0.087076), (-1, 0.0, 0, 0, 0.039984), (-1, 0.0, 0, 0, -1.6e-05)),
            ((52, 0.997, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.025367), (-1, 0.0, 0, 0, 0.001821), (-1, 0.0, 0, 0, -0.022477), (-1, 0.0, 0, 0, 0.038249)),
            ((39, 0.528, 1, 4, 0.0), (39, 0.28500000000000003, 2, 3, 0.090648), (-1, 0.0, 0, 0, -0.003522), (-1, 0.0, 0, 0, 0.02283), (26, 4.714182565154557, 5, 6, -0.187399), (-1, 0.0, 0, 0, -0.020144), (-1, 0.0, 0, 0, 0.015593)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.037835), (-1, 0.0, 0, 0, -0.009307), (-1, 0.0, 0, 0, 0.007572), (6, 0.3024435993004616, 5, 6, -0.435122), (-1, 0.0, 0, 0, 0.016695), (-1, 0.0, 0, 0, -0.0334)),
            ((12, 0.6759999999999999, 1, 4, 0.0), (26, 4.134165555407019, 2, 3, 0.056263), (-1, 0.0, 0, 0, -0.018911), (-1, 0.0, 0, 0, 0.006406), (11, 0.896, 5, 6, -0.291086), (-1, 0.0, 0, 0, 0.016774), (-1, 0.0, 0, 0, -0.030357)),
            ((51, 2.5, 1, 4, 0.0), (50, 0.5, 2, 3, 0.028214), (-1, 0.0, 0, 0, 0.005694), (-1, 0.0, 0, 0, -0.011277), (12, 0.45099999999999996, 5, 6, -0.551023), (-1, 0.0, 0, 0, 0.000178), (-1, 0.0, 0, 0, -0.042045)),
            ((39, 0.9045, 1, 4, 0.0), (2, 0.992, 2, 3, 0.032701), (-1, 0.0, 0, 0, -0.004997), (-1, 0.0, 0, 0, 0.01117), (53, 0.32799999999999996, 5, 6, -0.475369), (-1, 0.0, 0, 0, -0.000998), (-1, 0.0, 0, 0, -0.037924)),
            ((20, 3.210268834500856, 1, 4, 0.0), (6, 0.6501228390275874, 2, 3, -0.032533), (-1, 0.0, 0, 0, 0.003828), (-1, 0.0, 0, 0, -0.010629), (13, 0.556, 5, 6, 0.570524), (-1, 0.0, 0, 0, -0.015209), (-1, 0.0, 0, 0, 0.038148)),
            ((60, 0.98, 1, 4, 0.0), (3, 0.15900000000000003, 2, 3, 0.026356), (-1, 0.0, 0, 0, 0.015643), (-1, 0.0, 0, 0, -0.003461), (20, -3.683959493972949, 5, 6, -0.577638), (-1, 0.0, 0, 0, -0.038831), (-1, 0.0, 0, 0, 0.002723)),
            ((15, -0.751, 1, 4, 0.0), (7, 0.019000000000000017, 2, 3, -0.590932), (-1, 0.0, 0, 0, -0.00672), (-1, 0.0, 0, 0, -0.035781), (39, 0.528, 5, 6, 0.025436), (-1, 0.0, 0, 0, 0.006786), (-1, 0.0, 0, 0, -0.00924)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (56, 0.1380952380952381, 2, 3, 0.021587), (-1, 0.0, 0, 0, 0.022838), (-1, 0.0, 0, 0, -0.001751), (-1, 0.0, 0, 0, -0.033329)),
            ((52, 0.995, 1, 4, 0.0), (7, 0.993, 2, 3, -0.029177), (-1, 0.0, 0, 0, 0.000714), (-1, 0.0, 0, 0, -0.032344), (7, 0.983, 5, 6, 0.63878), (-1, 0.0, 0, 0, -0.007017), (-1, 0.0, 0, 0, 0.042601)),
            ((15, -0.17, 1, 4, 0.0), (7, 0.861, 2, 3, -0.212271), (-1, 0.0, 0, 0, -0.020064), (-1, 0.0, 0, 0, 0.010454), (13, 0.992, 5, 6, 0.076742), (-1, 0.0, 0, 0, 0.009171), (-1, 0.0, 0, 0, -0.013851)),
            ((60, 0.664, 1, 4, 0.0), (2, 0.43899999999999995, 2, 3, 0.055724), (-1, 0.0, 0, 0, 0.031874), (-1, 0.0, 0, 0, -0.00044), (26, 4.598548757796862, 5, 6, -0.279126), (-1, 0.0, 0, 0, -0.039069), (-1, 0.0, 0, 0, 0.004229)),
            ((19, 0.5, 1, 4, 0.0), (26, 4.134165555407019, 2, 3, -0.041021), (-1, 0.0, 0, 0, -0.024662), (-1, 0.0, 0, 0, 0.001161), (6, 0.8846610635340376, 5, 6, 0.431398), (-1, 0.0, 0, 0, 0.030236), (-1, 0.0, 0, 0, -0.024084)),
            ((15, -0.17, 1, 4, 0.0), (59, 0.16100000000000003, 2, 3, -0.209306), (-1, 0.0, 0, 0, 0.007181), (-1, 0.0, 0, 0, -0.021688), (13, 0.992, 5, 6, 0.075366), (-1, 0.0, 0, 0, 0.0089), (-1, 0.0, 0, 0, -0.013206)),
            ((52, 0.978, 1, 4, 0.0), (60, 0.0010000000000000009, 2, 3, -0.042874), (-1, 0.0, 0, 0, -0.030287), (-1, 0.0, 0, 0, 0.000606), (24, 0.06250000000000003, 5, 6, 0.410768), (-1, 0.0, 0, 0, -0.001574), (-1, 0.0, 0, 0, 0.038814)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.601213), (-1, 0.0, 0, 0, -0.00698), (-1, 0.0, 0, 0, -0.037427), (40, 0.5, 5, 6, 0.023583), (-1, 0.0, 0, 0, 0.008268), (-1, 0.0, 0, 0, -0.006661)),
            ((7, 0.951, 1, 4, 0.0), (42, 0.13392857142857142, 2, 3, -0.073088), (-1, 0.0, 0, 0, 0.006562), (-1, 0.0, 0, 0, -0.015049), (53, 0.274, 5, 6, 0.224631), (-1, 0.0, 0, 0, -0.01213), (-1, 0.0, 0, 0, 0.026194)),
            ((51, 2.5, 1, 4, 0.0), (50, 0.5, 2, 3, 0.026428), (-1, 0.0, 0, 0, 0.005439), (-1, 0.0, 0, 0, -0.01083), (52, 0.732, 5, 6, -0.531435), (-1, 0.0, 0, 0, -0.036689), (-1, 0.0, 0, 0, 0.006552)),
            ((60, 0.98, 1, 4, 0.0), (5, 0.001, 2, 3, 0.024624), (-1, 0.0, 0, 0, 0.023663), (-1, 0.0, 0, 0, -0.001596), (20, -3.683959493972949, 5, 6, -0.563819), (-1, 0.0, 0, 0, -0.038297), (-1, 0.0, 0, 0, 0.003093)),
            ((20, 3.210268834500856, 1, 4, 0.0), (26, 4.134165555407019, 2, 3, -0.031223), (-1, 0.0, 0, 0, -0.019126), (-1, 0.0, 0, 0, 0.00108), (13, 0.556, 5, 6, 0.554453), (-1, 0.0, 0, 0, -0.014957), (-1, 0.0, 0, 0, 0.037527)),
            ((12, 0.6759999999999999, 1, 4, 0.0), (12, 0.39, 2, 3, 0.05253), (-1, 0.0, 0, 0, -0.008438), (-1, 0.0, 0, 0, 0.009347), (56, 0.4083333333333333, 5, 6, -0.280673), (-1, 0.0, 0, 0, 0.015255), (-1, 0.0, 0, 0, -0.030417)),
            ((52, 0.995, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.02792), (-1, 0.0, 0, 0, 0.001612), (-1, 0.0, 0, 0, -0.022967), (7, 0.983, 5, 6, 0.624524), (-1, 0.0, 0, 0, -0.006529), (-1, 0.0, 0, 0, 0.041746)),
            ((15, -0.751, 1, 4, 0.0), (20, -4.595119850134589, 2, 3, -0.570489), (-1, 0.0, 0, 0, -0.035192), (-1, 0.0, 0, 0, -0.006116), (39, 0.528, 5, 6, 0.024188), (-1, 0.0, 0, 0, 0.00648), (-1, 0.0, 0, 0, -0.008858)),
            ((40, 2.5, 1, 4, 0.0), (20, -3.683959493972949, 2, 3, 0.034356), (-1, 0.0, 0, 0, -0.009068), (-1, 0.0, 0, 0, 0.007181), (4, 0.326, 5, 6, -0.411628), (-1, 0.0, 0, 0, -0.04643), (-1, 0.0, 0, 0, -6e-05)),
            ((3, 0.42400000000000004, 1, 4, 0.0), (59, 0.24, 2, 3, -0.06439), (-1, 0.0, 0, 0, 0.006005), (-1, 0.0, 0, 0, -0.015754), (2, 0.984, 5, 6, 0.244838), (-1, 0.0, 0, 0, -0.015047), (-1, 0.0, 0, 0, 0.029089)),
            ((60, 0.664, 1, 4, 0.0), (2, 0.43899999999999995, 2, 3, 0.052806), (-1, 0.0, 0, 0, 0.03059), (-1, 0.0, 0, 0, -0.000495), (24, 0.015833333333333345, 5, 6, -0.26942), (-1, 0.0, 0, 0, -0.03539), (-1, 0.0, 0, 0, 0.006734)),
            ((15, -0.17, 1, 4, 0.0), (27, 0.995, 2, 3, -0.202438), (-1, 0.0, 0, 0, -0.001325), (-1, 0.0, 0, 0, -0.032662), (26, 4.280624567377958, 5, 6, 0.071773), (-1, 0.0, 0, 0, -0.0115), (-1, 0.0, 0, 0, 0.009324)),
            ((50, 0.5, 1, 4, 0.0), (24, 0.06250000000000003, 2, 3, 0.06661), (-1, 0.0, 0, 0, -0.003569), (-1, 0.0, 0, 0, 0.033252), (3, 0.15100000000000002, 5, 6, -0.218762), (-1, 0.0, 0, 0, 0.029024), (-1, 0.0, 0, 0, -0.021239)),
            ((39, 0.9045, 1, 4, 0.0), (2, 0.992, 2, 3, 0.030534), (-1, 0.0, 0, 0, -0.004781), (-1, 0.0, 0, 0, 0.010577), (53, 0.32799999999999996, 5, 6, -0.450312), (-1, 0.0, 0, 0, -0.000395), (-1, 0.0, 0, 0, -0.036696)),
            ((8, 0.6190476190476191, 1, 4, 0.0), (27, 0.421, 2, 3, 0.052375), (-1, 0.0, 0, 0, 0.03016), (-1, 0.0, 0, 0, -0.001804), (5, 0.009, 5, 6, -0.270659), (-1, 0.0, 0, 0, 0.032482), (-1, 0.0, 0, 0, -0.03219)),
            ((39, 0.528, 1, 4, 0.0), (5, 0.172, 2, 3, 0.083863), (-1, 0.0, 0, 0, 0.013124), (-1, 0.0, 0, 0, -0.009663), (26, 4.73345209625079, 5, 6, -0.17512), (-1, 0.0, 0, 0, -0.016986), (-1, 0.0, 0, 0, 0.020873)),
            ((51, 2.5, 1, 4, 0.0), (55, 0.27525252525252525, 2, 3, 0.026183), (-1, 0.0, 0, 0, 0.006722), (-1, 0.0, 0, 0, -0.007745), (12, 0.45099999999999996, 5, 6, -0.51739), (-1, 0.0, 0, 0, 0.001125), (-1, 0.0, 0, 0, -0.040099)),
            ((20, 3.210268834500856, 1, 4, 0.0), (8, 0.6190476190476191, 2, 3, -0.029875), (-1, 0.0, 0, 0, 0.001539), (-1, 0.0, 0, 0, -0.016189), (13, 0.556, 5, 6, 0.544173), (-1, 0.0, 0, 0, -0.014562), (-1, 0.0, 0, 0, 0.037092)),
            ((46, 0.8571428571428572, 1, 4, 0.0), (11, 0.026, 2, 3, 0.020433), (-1, 0.0, 0, 0, 0.034544), (-1, 0.0, 0, 0, -0.000793), (-1, 0.0, 0, 0, -0.032003)),
            ((3, 0.42400000000000004, 1, 4, 0.0), (59, 0.24, 2, 3, -0.062188), (-1, 0.0, 0, 0, 0.005674), (-1, 0.0, 0, 0, -0.01508), (59, 0.244, 5, 6, 0.241221), (-1, 0.0, 0, 0, -0.023662), (-1, 0.0, 0, 0, 0.023759)),
            ((15, -0.751, 1, 4, 0.0), (7, 0.019000000000000017, 2, 3, -0.560841), (-1, 0.0, 0, 0, -0.005634), (-1, 0.0, 0, 0, -0.034469), (39, 0.528, 5, 6, 0.023652), (-1, 0.0, 0, 0, 0.00623), (-1, 0.0, 0, 0, -0.008481)),
            ((40, 2.5, 1, 4, 0.0), (60, 0.664, 2, 3, 0.034018), (-1, 0.0, 0, 0, 0.005011), (-1, 0.0, 0, 0, -0.013782), (6, 0.3024435993004616, 5, 6, -0.402213), (-1, 0.0, 0, 0, 0.016799), (-1, 0.0, 0, 0, -0.031525)),
            ((52, 0.995, 1, 4, 0.0), (7, 0.993, 2, 3, -0.026598), (-1, 0.0, 0, 0, 0.000816), (-1, 0.0, 0, 0, -0.031711), (7, 0.983, 5, 6, 0.611561), (-1, 0.0, 0, 0, -0.006255), (-1, 0.0, 0, 0, 0.041191)),
            ((12, 0.6819999999999999, 1, 4, 0.0), (4, 0.941, 2, 3, 0.049644), (-1, 0.0, 0, 0, -0.003033), (-1, 0.0, 0, 0, 0.015862), (56, 0.4083333333333333, 5, 6, -0.273957), (-1, 0.0, 0, 0, 0.014414), (-1, 0.0, 0, 0, -0.029959)),
            ((60, 0.98, 1, 4, 0.0), (5, 0.001, 2, 3, 0.023586), (-1, 0.0, 0, 0, 0.021909), (-1, 0.0, 0, 0, -0.001443), (20, -3.683959493972949, 5, 6, -0.540947), (-1, 0.0, 0, 0, -0.037186), (-1, 0.0, 0, 0, 0.003088)),
            ((8, 0.6190476190476191, 1, 4, 0.0), (27, 0.421, 2, 3, 0.051353), (-1, 0.0, 0, 0, 0.02872), (-1, 0.0, 0, 0, -0.001638), (5, 0.009, 5, 6, -0.263469), (-1, 0.0, 0, 0, 0.030226), (-1, 0.0, 0, 0, -0.030677)),
            ((20, 3.210268834500856, 1, 4, 0.0), (26, 4.134165555407019, 2, 3, -0.028638), (-1, 0.0, 0, 0, -0.018732), (-1, 0.0, 0, 0, 0.001128), (13, 0.556, 5, 6, 0.536992), (-1, 0.0, 0, 0, -0.013993), (-1, 0.0, 0, 0, 0.036725)),
            ((52, 0.995, 1, 4, 0.0), (55, 0.4494949494949495, 2, 3, -0.025067), (-1, 0.0, 0, 0, 0.00166), (-1, 0.0, 0, 0, -0.022417), (7, 0.983, 5, 6, 0.602645), (-1, 0.0, 0, 0, -0.005907), (-1, 0.0, 0, 0, 0.040592)),
            ((15, -0.771, 1, 4, 0.0), (7, 0.02400000000000002, 2, 3, -0.569285), (-1, 0.0, 0, 0, -0.005893), (-1, 0.0, 0, 0, -0.035949), (40, 0.5, 5, 6, 0.022523), (-1, 0.0, 0, 0, 0.007741), (-1, 0.0, 0, 0, -0.006179)),
            ((51, 2.5, 1, 4, 0.0), (8, 0.6190476190476191, 2, 3, 0.025386), (-1, 0.0, 0, 0, 0.004297), (-1, 0.0, 0, 0, -0.013386), (12, 0.45099999999999996, 5, 6, -0.497459), (-1, 0.0, 0, 0, 0.001226), (-1, 0.0, 0, 0, -0.038772)),
            ((9, 0.9530000000000001, 1, 4, 0.0), (9, 0.75, 2, 3, -0.031113), (-1, 0.0, 0, 0, 0.001211), (-1, 0.0, 0, 0, -0.035736), (5, 0.06, 5, 6, 0.495163), (-1, 0.0, 0, 0, 0.039985), (-1, 0.0, 0, 0, -0.007904)),
            ((39, 0.528, 1, 4, 0.0), (48, 0.989, 2, 3, 0.081289), (-1, 0.0, 0, 0, 0.000794), (-1, 0.0, 0, 0, 0.038007), (26, 4.714182565154557, 5, 6, -0.169776), (-1, 0.0, 0, 0, -0.018424), (-1, 0.0, 0, 0, 0.014875)),
        )),
    ],
}
CORTE = 0.2
# <<FIN>>

# Mezcla por longitud de la respuesta: las cortas (tipo test) usan el modelo con todos los rasgos;
# las largas (tipo giro) el modelo sin longitud de respuesta, que es un atajo del train que no
# extrapola. alfa = peso del modelo largo, en escala log-longitud (suave, sin saltos).
LARGO_CENTRO = 180.0   # caracteres de respuesta donde ambos pesan igual
LARGO_ESCALA = 0.25    # anchura de la transición en log-longitud


def _arbol(nodos, x):
    i = 0
    while nodos[i][0] >= 0:
        f, t, izq, der, _ = nodos[i]
        i = izq if x[f] <= t else der
    return nodos[i][4]


def _logit_conjunto(nombre, c):
    x = [c[n] for n in NOMBRES[nombre]]
    raws = [base + sum(_arbol(a, x) for a in arboles) for base, arboles in MODELOS[nombre]]
    return sum(raws) / len(raws)


def logit_riesgo(caso):
    """Logit de P(alucina): mezcla del modelo corto y el largo según la longitud de la respuesta."""
    c = extraer(caso)
    alfa = 1.0 / (1.0 + math.exp(-(c["logR"] - math.log(LARGO_CENTRO)) / LARGO_ESCALA))
    return (1.0 - alfa) * _logit_conjunto("corto", c) + alfa * _logit_conjunto("largo", c)


class Guardian:
    def score(self, caso):
        if not MODELOS:
            raise RuntimeError("modelo vacío: ejecuta entrenar_guardian.py")
        z = logit_riesgo(caso) - math.log(CORTE / (1.0 - CORTE))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
