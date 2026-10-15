"""Bot de El Guardián v2: combina las señales precalculadas (LLM juez, NLI) con
comprobaciones lingüísticas propias, genéricas y sin depender del id ni del train.

Solo librería estándar. Cómo decide (todo es una suma de "log-odds"):
  z = sesgo + Σ peso_i · gravedad_tipo_i · señal_i        (pesos en PESOS_*)
  score = sigmoide(z + ln(COSTE_FN))                      (corte 0.5 del organizador)
El `+ ln(4)` mueve el corte a P(alucina) = 1/(1+4) = 0.2, el corte óptimo cuando
dejar pasar una alucinación cuesta 4 veces una falsa alarma.

PATRONES encontrados analizando el train (casos H, los representativos del test):
  1. El LLM juez acierta casi siempre cuando alarma, pero deja pasar ~60 % de las
     alucinaciones. Un "respaldada" con confianza baja (<= 0.92) es sospechoso.
  2. Las alucinaciones son ediciones mínimas de una respuesta fiel. Cuatro huellas
     lingüísticas aparecen una y otra vez:
       a) Endurecer la modalidad: la fuente dice "podría / prevé / estudia / si se
          aprueba" y la respuesta lo afirma en futuro ("bajará", "destinará").
          -> om_modal, om_prev, om_cond y, sobre todo, fut_nuevo.
       b) Perder una restricción: se omite "solo / salvo / siempre / la mayoría de /
          algunos / el primer" -> sobregeneralización. -> om_cond, om_limit.
       c) Añadir causa, finalidad o consecuencia que ninguna fuente da ("gracias a",
          "debido a", "impulsado por"). -> ad_causa.
       d) Cambiar una cifra por otra que no sale de las fuentes. -> cifra_sin_respaldo.
  3. Las respuestas fieles reescriben más (paráfrasis, aritmética, combinan fuentes),
     las alucinadas copian casi literal con un cambio puntual: por eso "palabras_nuevas"
     pesa NEGATIVO. Es un rasgo del dataset: si el giro cambia de terreno, ponedlo a 0.
  4. Con pocas fuentes el LLM y el NLI son casi perfectos; con 7-9 el LLM pierde
     recall. Por eso hay dos juegos de pesos (SIMPLE / COMPLEJO), elegidos por
     N_FUENTES_SIMPLE (el id no llega al bot; el nº de fuentes es el proxy de "caso H").
"""
import itertools
import math
import re
import unicodedata

# ======================= PARÁMETROS AJUSTABLES =======================
COSTE_FN = 4.0            # una alucinación escapada cuesta 4 falsas alarmas -> corte en P=0.2
N_FUENTES_SIMPLE = 3      # <= este nº de fuentes se usa PESOS_SIMPLE, si no PESOS_COMPLEJO

# Gravedad por tipo de error (de peor a menos peor): contradicción > distorsión >
# sobregeneralización = no respaldado. Multiplica el peso de las señales que
# detectan ese tipo. Subid un valor para ser más estricto con ese tipo de error.
GRAVEDAD_TIPO = {"contradiccion": 1.3, "distorsion": 1.15,
                 "sobregeneralizacion": 1.0, "no_respaldado": 1.0}

# Qué tipo de error detecta cada señal (la gravedad de arriba se aplica por esta tabla).
TIPO_DE_SENAL = {
    "llm_contradicha": "contradiccion", "nli_contradiccion": "contradiccion",
    "cifra_sin_respaldo": "distorsion", "cifra_otra_fuente": "distorsion",
    "om_modal": "distorsion", "om_prev": "distorsion", "fut_nuevo": "distorsion",
    "om_cond": "sobregeneralizacion", "om_limit": "sobregeneralizacion",
    "cuantificador_nuevo": "sobregeneralizacion",
    "llm_no_respaldada": "no_respaldado", "nli_fragmento": "no_respaldado",
    "ad_causa": "no_respaldado",
}

# Umbrales de confianza del LLM cuando dice "respaldada" (más duda = más sospecha).
CONF_LLM_BAJA = 0.92      # <= : duda alta
CONF_LLM_MEDIA = 0.97     # < (y > BAJA): duda media

# PESOS en log-odds. Positivo = empuja hacia "alucina". "sesgo" es el punto de partida.
# --- casos con pocas fuentes: el LLM y el NLI son fiables, se les cree casi todo ---
PESOS_SIMPLE = {
    "sesgo": -4.4,
    "llm_contradicha": 3.3, "llm_no_respaldada": 4.0, "llm_nulo": 1.0,
    "llm_duda_alta": 1.9, "llm_duda_media": 0.2,
    "nli_fragmento": 2.5, "nli_todas": 3.8, "nli_contradiccion": 0.5,
    "cifra_sin_respaldo": 0.5, "cifra_otra_fuente": 0.3, "cuantificador_nuevo": 0.5,
    "palabras_nuevas": -1.2, "palabras_nuevas_citada": -1.0,
    "om_modal": 0.3, "om_prev": 0.4, "om_cond": 0.5, "om_limit": 0.3,
    "ad_causa": 0.8, "fut_nuevo": 2.0,
}
# --- casos con muchas fuentes (tipo H / test): el LLM se deja pasar mucho, así que
#     el peso se reparte con el NLI por fragmentos, las huellas lingüísticas y la
#     duda del LLM. Pesos de una regresión logística sobre los H validada por
#     escenarios (CV repetida ~0.30 de puntuación; el bot de ejemplo da -0.44). ---
PESOS_COMPLEJO = {
    "sesgo": -1.4,
    "llm_contradicha": 3.2, "llm_no_respaldada": 3.0, "llm_nulo": 3.6,
    "llm_duda_alta": 1.8, "llm_duda_media": 0.9,
    "nli_fragmento": 0.8, "nli_todas": 0.0, "nli_contradiccion": 0.0,
    "cifra_sin_respaldo": 0.45, "cifra_otra_fuente": 0.0, "cuantificador_nuevo": 0.0,
    "palabras_nuevas": -3.0, "palabras_nuevas_citada": 0.0,
    "om_modal": 0.2, "om_prev": 0.4, "om_cond": 0.5, "om_limit": 0.3,
    "ad_causa": 0.8, "fut_nuevo": 2.5,
}
# =====================================================================

# --- Léxicos (prefijos sobre palabras sin tildes; los de >=4 letras casan por prefijo) ---
MODAL = ("podri", "podra", "podre", "podr", "pueden", "puede", "pudier", "deben", "debe", "debier")
PREVISION = ("prev", "estim", "estudi", "propon", "propus", "plane", "pretend", "aspir", "espera",
             "probabl", "posible", "supuest", "presunt", "alegad", "sugier", "recomend", "baraj")
CONDICION = ("si", "aunque", "salvo", "excepto", "siempre", "solo", "unicamente", "exclusivamente",
             "cuando", "antes", "hasta", "minimo", "maximo", "mientras", "sujet", "condicion")
LIMITE = ("ningun", "mayori", "parte", "algun", "vari", "cerca", "aproxim", "alrededor", "casi",
          "unos", "unas", "primer", "segund", "tercer", "ultim", "principal", "menos")
CAUSA = ("gracias", "debid", "porque", "causa", "motivo", "consecuen", "impuls", "permit", "provoc",
         "origin", "gener", "favorec", "propici", "result", "conllev", "razon", "asi")
TOTAL = ("todo", "toda", "todos", "todas", "cualqu", "total", "nunca", "siempre", "absolut",
         "completa", "entera", "ambos", "ambas", "cada")
FUTURO_IRREGULAR = {"sera", "seran", "habra", "habran", "tendra", "tendran", "podra", "podran",
                    "hara", "haran", "dira", "diran", "saldra", "saldran", "vendra", "vendran",
                    "pondra", "pondran", "querra", "sabra", "cabra", "valdra"}
CUANTIFICADORES = {"todos", "todas", "todo", "toda", "cualquier", "cualquiera", "siempre",
                   "nunca", "ningun", "ninguna", "ninguno", "cada", "totalidad", "mayoria",
                   "generalmente", "habitualmente", "entero", "entera", "exclusivamente",
                   "unicamente", "ambos", "ambas"}
NUMEROS_EN_LETRA = {"cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
                    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
                    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
                    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
                    "cien": 100, "mil": 1000}
PALABRAS_VACIAS = set("de la el en y a los las del un una que se por con para al lo su sus es "
                      "son fue como mas o no ha han sido este esta estos estas entre sobre "
                      "tras hasta desde sin ser le les ya".split())


def _norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _tokens(s):
    return re.findall(r"[a-z0-9]+", _norm(s))


def _palabras_solo(s):
    return re.findall(r"[a-z]+", _norm(s))


def _frases(texto):
    return [p for p in re.split(r"(?<=[.;])\s+", texto) if p.strip()] or [texto]


def _cifras(s):
    """Cifras del texto (1.200 -> 1200, 3,5 -> 3.5) y números escritos en letra."""
    s, out = _norm(s), set()
    for m in re.finditer(r"\d+(?:[.,]\d+)*", s):
        t = m.group()
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        try:
            out.add(float(t.replace(",", ".")))
        except ValueError:
            pass
    out.update(float(NUMEROS_EN_LETRA[w]) for w in re.findall(r"[a-z]+", s) if w in NUMEROS_EN_LETRA)
    return out


def _derivable(n, pool):
    """¿Sale `n` de operar (suma, resta, producto, cociente, %, doble/mitad) con cifras de las fuentes?"""
    pool = sorted(pool)[:60]
    if any(abs(n - a * k) < 1e-6 for a in pool for k in (2, 3, 0.5)):
        return True
    for a, b in itertools.combinations(pool, 2):
        cand = [a + b, abs(a - b), a * b]
        for x, y in ((a, b), (b, a)):
            if y:
                cand += [x / y, 100 * x / y, 100 * abs(a - b) / y]
        if any(abs(n - v) <= max(0.051, 0.006 * abs(n)) for v in cand):
            return True
    return False


def _raices(s):
    return {w[:5] for w in _tokens(s) if w not in PALABRAS_VACIAS and len(w) > 2 and not w.isdigit()}


def _casan(palabras, prefijos):
    """Prefijos del léxico que aparecen en la lista de palabras."""
    return {p for p in prefijos
            if any(t == p or (len(p) >= 4 and t.startswith(p)) for t in palabras)}


def _tramo_de_fuente(respuesta, fuentes):
    """Frases de las fuentes de las que sale la respuesta (solape de raíces); si ninguna
    llega al umbral, la más parecida. Sirve para ver qué matices se OMITEN al reescribir."""
    r = _raices(respuesta)
    cand = [(s, _raices(s)) for f in fuentes for s in _frases(f)]
    pts = [(len(r & st), len(r & st) / max(1, len(r)), s) for s, st in cand]
    sel = [s for n, fr, s in pts if n >= 3 and fr >= 0.25]
    return sel or ([max(pts, key=lambda t: t[0])[2]] if pts else [])


def _futuros(texto):
    """Formas de futuro de indicativo (con tilde: bajará, se cerrarán...) y futuros irregulares."""
    pal = re.findall(r"[a-zñáéíóú]+", texto.lower())
    return sum(1 for t in pal if re.fullmatch(r"[a-zñáéíóú]{3,}(ará|erá|irá|arán|erán|irán)", t)
               or _norm(t) in FUTURO_IRREGULAR)


def _senales_texto(respuesta, fuentes):
    """Huellas lingüísticas de alucinación (ver PATRONES): modalidad endurecida,
    restricciones omitidas y causas añadidas."""
    rt, todas = _palabras_solo(respuesta), [t for f in fuentes for t in _palabras_solo(f)]
    tramo = _tramo_de_fuente(respuesta, fuentes)
    st = [t for s in tramo for t in _palabras_solo(s)]
    omitida = lambda lex: float(min(len(_casan(st, lex) - _casan(rt, lex)), 2))   # en la fuente, no en la respuesta
    anadida = lambda lex: float(min(len(_casan(rt, lex) - _casan(todas, lex)), 2))  # en la respuesta, en ninguna fuente
    return {
        "om_modal": omitida(MODAL), "om_prev": omitida(PREVISION),
        "om_cond": omitida(CONDICION), "om_limit": omitida(LIMITE),
        "ad_causa": anadida(CAUSA),
        # futuro que no existe en ninguna fuente: la respuesta afirma lo que la fuente deja en condicional
        "fut_nuevo": float(_futuros(respuesta) > 0 and _futuros(" ".join(fuentes)) == 0),
    }


def _senales_oracion(o, fuentes, cifras_fuentes, palabras_fuentes, cuant_fuentes):
    """Señales de UNA oración (LLM, NLI, cifras, palabras nuevas)."""
    r = o["respaldo_por_fuente"]
    llm = o.get("llm") or {}
    ver, conf = llm.get("veredicto"), llm.get("confianza") or 0.0
    k = llm.get("fuente")
    if k is None or not 0 <= k < len(fuentes):          # sin cita: usar la fuente que más respalda el NLI
        k = max(range(len(r)), key=r.__getitem__) if r else 0
    citada = fuentes[k] if fuentes else ""
    texto = o["texto"]
    palabras = _raices(texto)
    cuant = {w for w in _tokens(texto) if w in CUANTIFICADORES}
    # Fuentes "de referencia" para las cifras: la que cita el LLM + las 2 que más respalda el NLI.
    top = {k, *sorted(range(len(r)), key=r.__getitem__, reverse=True)[:2]}
    cifras_top = set().union([_cifras(fuentes[i]) for i in top]) if fuentes else set()
    cifras = _cifras(texto)
    inexplicadas = [n for n in cifras if n not in cifras_top and not _derivable(n, cifras_top)]
    inventadas = [n for n in inexplicadas if n not in cifras_fuentes]
    respaldada = ver == "respaldada"
    return {
        "llm_contradicha": conf if ver == "contradicha" else 0.0,
        "llm_no_respaldada": conf if ver == "no_respaldada" else 0.0,
        "llm_nulo": float(ver is None),
        "llm_duda_alta": float(respaldada and conf <= CONF_LLM_BAJA),
        "llm_duda_media": float(respaldada and CONF_LLM_BAJA < conf < CONF_LLM_MEDIA),
        "nli_todas": 1.0 - o["respaldo_todas"],
        "nli_contradiccion": max(max(o["contradiccion_por_fuente"], default=0.0), o["contradiccion_todas"]),
        "cifra_sin_respaldo": float(min(len(inventadas), 2)),
        "cifra_otra_fuente": float(min(len(inexplicadas) - len(inventadas), 2)),
        "cuantificador_nuevo": float(min(len(cuant - cuant_fuentes), 2)),
        "palabras_nuevas": len(palabras - palabras_fuentes) / max(1, len(palabras)),
        "palabras_nuevas_citada": len(palabras - _raices(citada)) / max(1, len(palabras)),
    }


def extraer_senales(caso):
    """Señales del caso. Una sola afirmación falsa basta, así que se toma la PEOR
    oración (máximo) en cada señal; el NLI de fragmentos localiza la parte dudosa."""
    fuentes, sen = caso["fuentes"], caso["senales"]
    cifras_f = set().union(*[_cifras(f) for f in fuentes]) if fuentes else set()
    palabras_f = set().union(*[_raices(f) for f in fuentes]) if fuentes else set()
    cuant_f = {w for f in fuentes for w in _tokens(f) if w in CUANTIFICADORES}
    por_oracion = [_senales_oracion(o, fuentes, cifras_f, palabras_f, cuant_f) for o in sen["oraciones"]]
    out = {k: max(s[k] for s in por_oracion) for k in por_oracion[0]} if por_oracion else {}
    # Debilidad del fragmento peor respaldado por UNA fuente (donde el NLI de oración se diluye).
    out["nli_fragmento"] = max((1.0 - max(f["respaldo_por_fuente"], default=0.0) for f in sen["fragmentos"]),
                               default=out.get("nli_todas", 0.0))
    out.update(_senales_texto(caso["respuesta"], fuentes))
    return out


def log_odds(caso):
    """z: log-odds de alucinación según los pesos del régimen (simple / complejo)."""
    pesos = PESOS_SIMPLE if len(caso["fuentes"]) <= N_FUENTES_SIMPLE else PESOS_COMPLEJO
    z = pesos["sesgo"]
    for nombre, valor in extraer_senales(caso).items():
        z += pesos[nombre] * GRAVEDAD_TIPO.get(TIPO_DE_SENAL.get(nombre), 1.0) * valor
    return z


class Guardian:
    def score(self, caso):
        z = log_odds(caso) + math.log(COSTE_FN)      # corte 0.5 <=> P(alucina) = 1/(1+COSTE_FN)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
