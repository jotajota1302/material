"""El Guardián — bot del equipo (versión 1).

Solo usa la librería estándar de Python: no instala ni ejecuta ningún modelo.

Idea general
------------
1. Para cada caso calculamos un conjunto de CARACTERÍSTICAS (números entre 0 y
   algo) que miden sospechas concretas:
     - qué dicen las señales precalculadas (juez LLM y NLI), cogiendo siempre
       la PEOR oración / fragmento, nunca la media;
     - si las cifras de la respuesta salen de las fuentes (directamente o
       haciendo cuentas: sumas, restas, porcentajes, proporciones);
     - si una cifra correcta se atribuye a otra entidad;
     - si la respuesta pierde matices de la fuente («solo», «prevé», «si se
       aprueba», «de los 14»...) o añade alcance («cualquier», «todos») o
       causas («gracias a», «porque»...);
     - cuántas palabras de la respuesta no aparecen en ninguna fuente.
2. Combinamos esas características con una regresión logística cuyos pesos se
   han aprendido con train.jsonl (ver entrenar.py, que los escribe abajo).
3. Reescalamos la probabilidad para que el corte óptimo (un escape cuesta 4
   veces más que una falsa alarma) caiga justo en 0,5, que es donde corta el
   organizador.
"""
import math
import re
import unicodedata
from itertools import combinations

# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def normalizar(texto):
    """Minúsculas y sin tildes, para comparar palabras sin que molesten acentos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")


# Palabras vacías: no aportan contenido y no las usamos para comparar.
VACIAS = set("""
a al algo ante antes asi aun bajo cada como con contra cual cuando de del desde
donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso
esos esta estaba estan estas este esto estos fue fueron ha han hasta hay la las
le les lo los mas me mientras muy ni no nos o otra otras otro otros para pero
por que quien se sea segun ser si sido sin sobre son su sus tambien tan tanto
te tiene tienen todo tras tu un una uno unos unas y ya yo año años
""".split())


def palabras(texto):
    """Lista de palabras normalizadas (solo letras)."""
    return re.findall(r"[a-zñ]+", normalizar(texto))


def raiz(palabra):
    """'Raíz' muy simple: los 5 primeros caracteres. Así 'plantó' y 'plantaron'
    o 'destinatario' y 'destinatarios' cuentan como la misma palabra."""
    return palabra[:5]


def raices_contenido(texto):
    """Conjunto de raíces de las palabras con contenido (sin palabras vacías)."""
    return {raiz(p) for p in palabras(texto) if p not in VACIAS and len(p) > 2}


def nombres_propios(texto):
    """Palabras en mayúscula que NO van al principio de frase: nombres de
    empresas, pueblos, organismos... Sirven para saber de QUIÉN se habla."""
    nombres = set()
    for frase in re.split(r"(?<=[.!?;:])\s+", texto):
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", frase)
        for t in tokens[1:]:
            if t[0].isupper() and len(t) > 2:
                nombres.add(normalizar(t))
    return nombres


# ---------------------------------------------------------------------------
# Cifras: extraerlas y comprobar si salen de las fuentes
# ---------------------------------------------------------------------------

UNIDADES = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}

# Una cifra escrita con dígitos al estilo español: 1.380 / 2,6 / 18.400 / 45
RE_CIFRA = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
# Una palabra-número (seis, treinta y dos...)
RE_PALABRA_NUM = re.compile(
    r"\b(" + "|".join(sorted(UNIDADES, key=len, reverse=True)) +
    r")(?:\s+y\s+(uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))?\b")


def _valor(cifra):
    return float(cifra.replace(".", "").replace(",", "."))


def extraer_numeros(texto):
    """Devuelve una lista de cifras del texto. Cada cifra es un dict con:
       valores: posibles valores (p. ej. 5,3 millones -> {5.3, 5300000})
       pct:     si va seguida de % o «por ciento»
       anio:    si parece un año (1900-2100)
       aprox:   si es una proporción «X de cada Y» (comparación aproximada)
    """
    t = normalizar(texto)
    encontrados = []  # (posicion_inicio, posicion_fin, valor)
    for m in RE_CIFRA.finditer(t):
        encontrados.append((m.start(), m.end(), _valor(m.group())))
    for m in RE_PALABRA_NUM.finditer(t):
        v = UNIDADES[m.group(1)]
        if m.group(2):
            v += UNIDADES.get(m.group(2), 0) if m.group(2) != "uno" else 1
        encontrados.append((m.start(), m.end(), float(v)))
    encontrados.sort()

    numeros = []
    usados = set()
    for i, (ini, fin, v) in enumerate(encontrados):
        if i in usados:
            continue
        resto = t[fin:fin + 25]
        # «seis de cada diez» -> 60 % aproximado
        m = re.match(r"\s+de\s+cada\s+", resto)
        if m and i + 1 < len(encontrados) and encontrados[i + 1][0] == fin + m.end():
            total = encontrados[i + 1][2]
            usados.add(i + 1)
            if total:
                numeros.append({"valores": {100.0 * v / total}, "pct": True,
                                "anio": False, "aprox": True, "ini": ini,
                                "fin": encontrados[i + 1][1]})
            continue
        valores = {v}
        m = re.match(r"\s*(mil\s+)?millones", resto)
        if m:
            valores.add(v * (1e9 if m.group(1) else 1e6))
        elif re.match(r"\s*mil\b", resto):
            valores.add(v * 1e3)
        if re.match(r"\s*centimos", resto):
            valores.add(v / 100.0)
        pct = bool(re.match(r"\s*(%|por\s*ciento)", resto))
        anio = v.is_integer() and 1900 <= v <= 2100 and not pct
        numeros.append({"valores": valores, "pct": pct, "anio": anio, "aprox": False,
                        "ini": ini, "fin": fin})
    return numeros


# Palabras que anuncian que la respuesta HACE UNA CUENTA con sus cifras. Si la
# respuesta dice «suman 1.980» o «cuatro menos que en 2024», ese número tiene
# que salir de ESA operación concreta, no de cualquier combinación.
RE_CUENTA_SUMA = re.compile(r"\b(suman|sumaron|suma|sumo|sumaba|sumando|en total|entre (los|las) (dos|tres|cuatro)"
                            r"|entre ambos|entre ambas|juntos|juntas|en conjunto)\b")
RE_CUENTA_DIF_DESPUES = re.compile(r"^\s*((%\s*|por ciento\s*|puntos?\s*|[a-z]+\s+)?(mas|menos)\b"
                                   r"|puntos?\b)")  # «25 puntos» = diferencia de porcentajes
RE_CUENTA_DIF_ANTES = re.compile(r"(diferencia|aumento|subida|incremento|rebaja|descenso|caida|"
                                 r"ganancia|perdida|ahorro|crecio|subio|bajo|aumento|cayo|mejoro|"
                                 r"empeoro|gano|perdio|redujo|anadio)\s+(en\s+|de\s+)?(unos?\s+|el\s+|un\s+)?$")
RE_CUENTA_CALCULO = re.compile(r"(es decir|lo que supone|lo que significa|lo que equivale|equivale a|"
                               r"o sea|lo que representa)[\s,]*(a\s+)?(unos?\s+|unas\s+|el\s+|la\s+|un\s+|una\s+)?$")


def tipo_de_cuenta(texto_norm, num):
    """'suma', 'diferencia', 'calculo' o None según las palabras alrededor de la cifra."""
    antes = texto_norm[max(0, num["ini"] - 40):num["ini"]]
    despues = texto_norm[num["fin"]:num["fin"] + 30]
    if RE_CUENTA_CALCULO.search(antes) or re.match(r"\s*veces\b", despues):
        return "calculo"  # «es decir, 11.600» / «seis veces más»: producto o proporción
    if RE_CUENTA_DIF_DESPUES.match(despues) or RE_CUENTA_DIF_ANTES.search(antes):
        return "diferencia"
    if RE_CUENTA_SUMA.search(antes[-25:] + " " + despues[:25]):
        return "suma"
    return None


def cuenta_cuadra(tipo, num, anclas, operandos):
    """¿Sale la cifra de una operación del tipo anunciado?
    anclas:    las otras cifras de la respuesta (si las hay, la cuenta debe usar una)
    operandos: cifras de las fuentes en las que se apoya la respuesta"""
    # Las anclas siempre entran (antes un recorte a 30 cifras tiraba las grandes)
    vals = sorted(set(anclas) | set(sorted(operandos)[:60]))
    candidatos = set()
    if tipo == "suma":
        for a, b in combinations(vals, 2):
            candidatos.update({a + b, a * b})  # «en total» también puede ser un producto
        for a, b, c in combinations(vals[:30], 3):
            candidatos.add(a + b + c)
    else:
        bases = anclas if anclas else vals
        for a in bases:
            for b in vals:
                if a == b:
                    continue
                candidatos.add(abs(a - b))
                if tipo == "calculo":
                    candidatos.update({a * b / 100.0, a * b})
                    if b:
                        candidatos.update({100.0 * a / b, a / b})
                    if a:
                        candidatos.update({100.0 * b / a, b / a})
    return any(_cerca(v, c) for v in num["valores"] for c in candidatos)


def _cerca(a, b, tolerancia_rel=0.01):
    """¿Son a y b el mismo número salvo redondeo?"""
    return abs(a - b) <= max(tolerancia_rel * abs(b), 0.051 if not float(a).is_integer() else 0.5)


def valores_derivados(valores_fuente):
    """Todo lo que se puede calcular con dos (o tres) cifras de las fuentes:
    sumas, restas, porcentajes de una cantidad, proporciones..."""
    vals = sorted(valores_fuente)[:60]  # límite por si hay muchísimas cifras
    derivados = set()
    for a, b in combinations(vals, 2):
        derivados.update({a + b, abs(a - b), a * b / 100.0, a * b})
        if b:
            derivados.update({100.0 * a / b, a / b})
        if a:
            derivados.update({100.0 * b / a, b / a})
    for a, b, c in combinations(vals[:30], 3):
        derivados.add(a + b + c)
    return derivados


def clasificar_numero(num, valores_fuente, derivados):
    """'directo' si la cifra está en las fuentes, 'derivado' si sale de hacer
    cuentas con ellas y 'inexplicado' si no sale de ningún sitio."""
    tol = 0.12 if num["aprox"] else 0.01
    for v in num["valores"]:
        if any(_cerca(v, f, tol) for f in valores_fuente):
            return "directo"
    for v in num["valores"]:
        if any(_cerca(v, d, tol) for d in derivados):
            return "derivado"
    return "inexplicado"


# Afirmaciones de proporción: «el doble», «cuadruplica», «seis veces», «un tercio»...
PROPORCIONES = [
    (r"\bdoble\b|\bduplica", 2.0), (r"\btriple\b|\btriplica", 3.0),
    (r"\bcuadruplica|\bcuadruple\b", 4.0), (r"\bquintuplica", 5.0),
    (r"\bla mitad\b", 0.5), (r"\bun tercio\b|\btercera parte\b", 1 / 3.0),
    (r"\bdos tercios\b", 2 / 3.0), (r"\bun cuarto\b|\bcuarta parte\b", 0.25),
    (r"\btres cuartos\b", 0.75),
]


def proporciones_falsas(texto_resp, numeros_resp, valores_fuente):
    """Cuenta afirmaciones de proporción que NO cuadran con ningún par de cifras
    (al menos una de las dos debe aparecer en la respuesta)."""
    t = normalizar(texto_resp)
    afirmaciones = []  # (k, es_minimo): es_minimo si dice «más de(l) ...»

    def es_minimo(inicio):
        return bool(re.search(r"(mas|algo mas) del?\s*(un|una|la|el)?\s*$", t[max(0, inicio - 14):inicio]))

    for patron, k in PROPORCIONES:
        m = re.search(patron, t)
        if m:
            afirmaciones.append((k, es_minimo(m.start())))
    for m in re.finditer(r"(\d+(?:,\d+)?|" + "|".join(UNIDADES) + r")\s+veces\b", t):
        palabra = m.group(1)
        k = UNIDADES.get(palabra) or _valor(palabra)
        afirmaciones.append((float(k), False))
    if not afirmaciones:
        return 0
    de_resp = {v for n in numeros_resp for v in n["valores"] if v}
    todos = {v for v in de_resp | set(valores_fuente) if v}
    falsas = 0
    for k, minimo in afirmaciones:
        ok = False
        for a in (de_resp or todos):
            for b in todos:
                if a == b:
                    continue
                # Probamos a/b y b/a; k < 1 son fracciones (un tercio), k > 1 múltiplos
                for r in (a / b, b / a):
                    if minimo and k <= r <= k * 1.6:
                        ok = True
                    elif abs(r - k) <= 0.12 * k:
                        ok = True
            if ok:
                break
        if not ok:
            falsas += 1
    return falsas


# ---------------------------------------------------------------------------
# Matices, alcance y causas
# ---------------------------------------------------------------------------

# Matices que restringen o suavizan una afirmación. Si la fuente los tiene y la
# respuesta no, es típico de «sobregeneralización» o de «endurecer un verbo».
MATICES = {
    "restrictivo": [r"\bsolo\b", r"\bunicamente\b", r"\bexclusivamente\b", r"\bsalvo\b",
                    r"\bexcepto\b", r"\bsiempre que\b", r"\ba condicion\b", r"\bse limita",
                    r"\bsolamente\b"],
    "condicional": [r"\bsi se\b", r"\bsi lo\b", r"\bsi la\b", r"\bsi el\b", r"\bsi los\b",
                    r"\bsi las\b", r"\bpreve\b", r"\bprevista?\b", r"\bprevisto\b",
                    r"\bestudia\b", r"\bplantea\b", r"\bpropone\b", r"\bpropuesta\b",
                    r"\bpodria", r"\bposible", r"\bpendiente", r"\bprovisional",
                    r"\bestima\b", r"\bestimo\b", r"\bespera\b", r"\bpretende\b",
                    r"\bproyecto\b", r"\bplanea\b", r"\bdepend", r"\bcondicionad",
                    r"\bsujet[oa]s? a\b", r"\ba expensas\b", r"\bse confirm"],
    "aproximado": [r"\bmas de\b", r"\bmenos de\b", r"\bal menos\b", r"\bcasi\b",
                   r"\bcerca de\b", r"\bunos\b", r"\bunas\b", r"\balrededor de\b",
                   r"\baproximadamente\b", r"\bhasta\b", r"\bcomo maximo\b",
                   r"\bcomo minimo\b", r"\bmaximo\b"],
    "parcial": [r"\balgunos\b", r"\balgunas\b", r"\bvarios\b", r"\bvarias\b",
                r"\bparte de\b", r"\bmayoria\b", r"\bde los \d", r"\bde las \d",
                r"\bentre \d", r"\bde ell[oa]s\b"],
}

# Palabras de alcance universal: si la respuesta las añade, puede estar ampliando.
# (Versión 2: quitamos «general», «toda la», «todo el», que salían en muchas
# respuestas fieles y solo metían ruido.)
UNIVERSALES = [r"\btodos\b", r"\btodas\b", r"\bcualquier", r"\bsiempre\b", r"\bnunca\b",
               r"\bningun", r"\bnadie\b", r"\btotalidad\b", r"\bintegramente\b",
               r"\bcompletamente\b", r"\bdel pais\b", r"\bde espana\b", r"\bsin excepcion"]

# Conectores de causa / consecuencia: si la respuesta los añade, puede estar
# inventándose un porqué. (Versión 2: fuera «para», que salía casi siempre.)
CAUSALES = [r"\bporque\b", r"\bya que\b", r"\bpuesto que\b", r"\bdebido a\b",
            r"\bgracias a\b", r"\bpor su\b", r"\bpor sus\b", r"\blo que permit",
            r"\bpermitio\b", r"\bpermitira\b", r"\bpor lo que\b", r"\bde ahi\b",
            r"\ba raiz de\b", r"\bprovoc", r"\bcausad", r"\bcomo consecuencia\b",
            r"\bcomo resultado\b", r"\bfruto de\b", r"\bpor ello\b", r"\bpor eso\b",
            r"\basi que\b", r"\bcon lo que\b", r"\bpara que\b", r"\bpara (ampliar|reducir|mejorar|evitar|financiar|cubrir)\b"]


def contar(patrones, texto_norm):
    return sum(1 for p in patrones if re.search(p, texto_norm))


def perdidos(patrones, fuente_norm, resp_norm):
    """Matices que están en la fuente pero no en la respuesta."""
    return sum(1 for p in patrones if re.search(p, fuente_norm) and not re.search(p, resp_norm))


def anadidos(patrones, fuentes_norm, resp_norm):
    """Expresiones que están en la respuesta pero en ninguna fuente."""
    return sum(1 for p in patrones if re.search(p, resp_norm) and not re.search(p, fuentes_norm))


def tramo_cubierto(fuente, raices_resp):
    """Trozo de la fuente que la respuesta 'usa': desde la primera hasta la
    última palabra de la fuente que también sale en la respuesta. Devuelve
    (texto_del_tramo, nº de palabras con contenido del tramo que la respuesta
    se ha dejado)."""
    tokens = re.findall(r"[a-zñ0-9.,%]+", normalizar(fuente))
    pos = [i for i, tok in enumerate(tokens)
           if (tok.isalpha() and tok not in VACIAS and raiz(tok) in raices_resp)]
    if len(pos) < 2:
        return "", 0
    tramo = tokens[min(pos):max(pos) + 1]
    huecos = sum(1 for tok in tramo
                 if tok.isalpha() and tok not in VACIAS and len(tok) > 2
                 and raiz(tok) not in raices_resp)
    return " ".join(tramo), huecos


# ---------------------------------------------------------------------------
# Características de un caso
# ---------------------------------------------------------------------------

def caracteristicas(caso):
    """Calcula todas las características (dict nombre -> número) de un caso."""
    fuentes = caso.get("fuentes") or []
    respuesta = caso.get("respuesta") or ""
    pregunta = caso.get("pregunta") or ""
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    fragmentos = senales.get("fragmentos") or []
    n_fuentes = len(fuentes)
    f = {}

    # --- 1. Señales del juez LLM (la PEOR oración manda) ---
    veredictos = [((o.get("llm") or {}).get("veredicto")) for o in oraciones]
    confianzas = [((o.get("llm") or {}).get("confianza") or 0.0) for o in oraciones]
    f["llm_no_respaldada"] = float(any(v == "no_respaldada" for v in veredictos))
    f["llm_contradicha"] = float(any(v == "contradicha" for v in veredictos))
    f["llm_sin_veredicto"] = float(any(v is None for v in veredictos))
    conf_resp = [c for v, c in zip(veredictos, confianzas) if v == "respaldada"]
    f["llm_conf_min_respaldada"] = min(conf_resp) if conf_resp else 0.0

    # --- 2. Señales del NLI ---
    def mins(lista, campo):
        vals = [x.get(campo, 0.0) for x in lista]
        return min(vals) if vals else 0.0

    def maxs(lista, campo):
        vals = [x.get(campo, 0.0) for x in lista]
        return max(vals) if vals else 0.0

    f["nli_resp_todas_min_orac"] = mins(oraciones, "respaldo_todas")
    f["nli_resp_todas_min_frag"] = mins(fragmentos, "respaldo_todas") if fragmentos else f["nli_resp_todas_min_orac"]
    f["nli_contr_todas_max_orac"] = maxs(oraciones, "contradiccion_todas")
    f["nli_contr_todas_max_frag"] = maxs(fragmentos, "contradiccion_todas")
    # La mejor fuente individual para la peor oración (lo que hacía el bot de ejemplo)
    f["nli_mejor_fuente_min_orac"] = min(
        [max(o.get("respaldo_por_fuente") or [0.0]) for o in oraciones] or [0.0])
    f["nli_contr_fuente_max_frag"] = max(
        [max(x.get("contradiccion_por_fuente") or [0.0]) for x in fragmentos] or [0.0])

    # Respaldo NLI de la fuente que cita el LLM: si el LLM dice «respaldada por
    # la fuente 3» y el NLI dice que la fuente 3 no la respalda, desconfiamos.
    citadas = set()
    resp_citada = []
    for o in oraciones:
        llm = o.get("llm") or {}
        i = llm.get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            citadas.add(i)
            rpf = o.get("respaldo_por_fuente") or []
            if i < len(rpf):
                resp_citada.append(rpf[i])
        sim = o.get("similitud_por_fuente") or []
        if sim:  # la fuente más parecida a cada oración también es «relevante»
            citadas.add(max(range(len(sim)), key=lambda k: sim[k]))
    f["nli_resp_fuente_citada_min"] = min(resp_citada) if resp_citada else 0.0

    # --- 3. Cifras ---
    texto_fuentes = " ".join(fuentes) + " " + pregunta
    nums_fuente = extraer_numeros(texto_fuentes)
    valores_fuente = {v for n in nums_fuente for v in n["valores"]}
    nums_resp = extraer_numeros(respuesta)
    nums_por_fuente = [extraer_numeros(x) for x in fuentes]

    def esta_en(n, lista_nums):
        return any(_cerca(v, w) for v in n["valores"] for m in lista_nums for w in m["valores"])

    # Cifras de la respuesta que están literalmente en las fuentes
    directos = [n for n in nums_resp if esta_en(n, nums_fuente)]

    # Operandos «locales»: las cifras directas de la respuesta y las de las
    # fuentes en las que se apoya (las que contienen esas cifras y las citadas
    # o más parecidas). Las cuentas fieles casi siempre salen de ahí. Si solo
    # cuadran combinando cifras sueltas de todo el texto, es casualidad.
    # No usamos años (2026 - 2025 = 1 explicaría cualquier «1»).
    fuentes_base = set(citadas) | {i for i, ns in enumerate(nums_por_fuente)
                                   if any(esta_en(d, ns) for d in directos)}

    def operandos_de(lista_nums):
        ops = set()
        for n in lista_nums:
            if n["anio"]:
                continue
            ops.update(n["valores"])
            if n["pct"]:  # «el 70 %... el resto» -> también vale el 30 %
                ops.update(100.0 - v for v in n["valores"] if v < 100)
        return ops

    ops_locales = operandos_de(directos) | {100.0}
    for i in fuentes_base:
        ops_locales |= operandos_de(nums_por_fuente[i])
    ops_globales = operandos_de(nums_fuente)
    der_locales = valores_derivados(ops_locales) if nums_resp else set()
    der_globales = valores_derivados(ops_globales) if nums_resp else set()
    resp_norm = normalizar(respuesta)

    clases = []
    for n in nums_resp:
        # Otras cifras de la respuesta («anclas»): una cuenta fiel suele usarlas
        anclas = operandos_de([d for d in directos if d is not n])
        tipo = tipo_de_cuenta(resp_norm, n)
        if tipo and not n["anio"]:
            # La respuesta anuncia una cuenta: comprobamos ESA operación
            if (cuenta_cuadra(tipo, n, anclas, ops_locales) or
                    # «11 pozos menos que la otra ONG»: la otra cifra puede estar
                    # en una fuente que no es la principal
                    (tipo == "diferencia" and not anclas and cuenta_cuadra(tipo, n, set(), ops_globales))):
                clases.append("cuenta_correcta")
            elif any(esta_en(n, nums_por_fuente[i]) for i in fuentes_base):
                # «pagarán un 9 % más»: no es una cuenta, el 9 % está tal cual
                # en la fuente en la que se apoya la respuesta
                clases.append("directo")
            else:
                clases.append("cuenta_falsa")
        elif esta_en(n, nums_fuente):
            clases.append("directo")
        elif anclas and cuenta_cuadra("calculo", n, anclas, ops_locales):
            clases.append("derivado_anclado")  # cuenta con cifras de la propia respuesta
        elif clasificar_numero(n, set(), der_locales) == "derivado":
            clases.append("derivado_local")
        elif clasificar_numero(n, set(), der_globales) == "derivado":
            clases.append("derivado_casual")  # solo cuadra por casualidad: sospechoso
        else:
            clases.append("inexplicado")
    sospechosos = [n for n, c in zip(nums_resp, clases)
                   if c in ("inexplicado", "derivado_casual", "cuenta_falsa")]
    f["num_inexplicados"] = float(sum(1 for c in clases if c == "inexplicado"))
    f["num_derivado_casual"] = float(sum(1 for c in clases if c == "derivado_casual"))
    f["num_sospechosos_grandes"] = float(sum(1 for n in sospechosos if max(n["valores"]) > 10))
    f["num_derivado_local"] = float(sum(1 for c in clases if c == "derivado_local"))
    f["num_derivado_anclado"] = float(sum(1 for c in clases if c == "derivado_anclado"))
    f["num_cuenta_falsa"] = float(sum(1 for c in clases if c == "cuenta_falsa"))
    f["num_cuenta_correcta"] = float(sum(1 for c in clases if c == "cuenta_correcta"))
    f["num_hay_cifras"] = float(len(nums_resp) > 0)
    f["num_proporcion_falsa"] = float(proporciones_falsas(respuesta, nums_resp, valores_fuente))

    # --- 4. ¿De quién es la cifra? (atribución cruzada) ---
    # La respuesta nombra a alguien (un nombre propio) y la cifra solo aparece
    # en fuentes que nombran a OTROS y a ese no. Ej.: «Valdemarina tiene un
    # 7,9 % de extranjeros» cuando el 7,9 % es de «Algaida Baja».
    # Antes también saltaba en comparaciones fieles entre dos entidades; ahora
    # basta con que una de las fuentes de la cifra comparta un nombre con la respuesta.
    nombres_resp = nombres_propios(respuesta)
    nombres_fuentes = [nombres_propios(x) for x in fuentes]
    ajenas = 0
    if nombres_resp:
        for n, c in zip(nums_resp, clases):
            if c != "directo" or n["anio"]:
                continue
            donde = [i for i, ns in enumerate(nums_por_fuente) if esta_en(n, ns)]
            if donde and all(nombres_fuentes[i] and not (nombres_fuentes[i] & nombres_resp)
                             for i in donde):
                ajenas += 1
    f["num_entidad_ajena"] = float(ajenas)
    # Nombres propios de la respuesta que no salen en ninguna fuente ni en la pregunta
    todos_nombres = set().union(*nombres_fuentes) if nombres_fuentes else set()
    conocidos = todos_nombres | {normalizar(p) for p in re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", pregunta)}
    f["nombre_inventado"] = float(len(nombres_resp - conocidos))

    # --- 5. Matices perdidos, alcance ampliado y causas añadidas ---
    fuentes_norm = normalizar(" ".join(fuentes))
    # Texto de las fuentes en las que se apoya la respuesta: un «todos» o un
    # «porque» cuenta como añadido si no está en ESAS fuentes
    base_norm = normalizar(" ".join(fuentes[i] for i in sorted(fuentes_base)))
    raices_resp = raices_contenido(respuesta) | {
        tok for tok in re.findall(r"\d[\d.,]*", resp_norm)}
    perdidos_tramo = {k: 0 for k in MATICES}
    perdidos_fuente = {k: 0 for k in MATICES}
    huecos = []
    for i in citadas:
        fn = normalizar(fuentes[i])
        tramo, h = tramo_cubierto(fuentes[i], raices_resp)
        huecos.append(h)
        for k, pats in MATICES.items():
            perdidos_fuente[k] += perdidos(pats, fn, resp_norm)
            if tramo:
                perdidos_tramo[k] += perdidos(pats, tramo, resp_norm)
    for k in MATICES:
        f["matiz_perdido_tramo_" + k] = float(perdidos_tramo[k])
        f["matiz_perdido_fuente_" + k] = float(perdidos_fuente[k])
    f["huecos_tramo_max"] = float(max(huecos) if huecos else 0)
    f["anadido_universal"] = float(anadidos(UNIVERSALES, base_norm, resp_norm))
    f["anadido_causal"] = float(anadidos(CAUSALES, base_norm, resp_norm))
    # Los mismos, pero comparando con TODAS las fuentes (más estricto: si
    # ninguna fuente dice «porque», el porqué es inventado seguro)
    f["anadido_causal_todas"] = float(anadidos(CAUSALES, fuentes_norm, resp_norm))

    # --- 6. Palabras nuevas: con contenido y que no salen en fuentes ni pregunta ---
    raices_fuentes = raices_contenido(" ".join(fuentes) + " " + pregunta)
    nuevas = raices_contenido(respuesta) - raices_fuentes
    f["palabras_nuevas"] = float(len(nuevas))
    f["palabras_nuevas_frac"] = len(nuevas) / max(1, len(raices_contenido(respuesta)))

    # --- 7. Cruces útiles: el LLM da el visto bueno PERO hay otra sospecha ---
    llm_ok = 1.0 - max(f["llm_no_respaldada"], f["llm_contradicha"])
    f["llm_ok_y_num_sospechoso"] = llm_ok * min(1.0, f["num_sospechosos_grandes"])
    f["llm_ok_y_nli_bajo"] = llm_ok * float(f["nli_resp_fuente_citada_min"] < 0.3)
    # El LLM dice «contradicha» pero no hay nada más raro: todas las cifras
    # cuadran y el NLI no ve contradicción. Aquí el LLM suele equivocarse.
    cifras_ok = float(f["num_sospechosos_grandes"] == 0 and f["num_entidad_ajena"] == 0)
    f["llm_contra_sin_apoyo"] = f["llm_contradicha"] * cifras_ok * (1.0 - f["nli_contr_todas_max_frag"])
    f["n_oraciones"] = float(len(oraciones))

    # --- 8. Las comprobaciones propias, en el régimen «el LLM dice que está bien» ---
    # Un modelo lineal da a cada sospecha el mismo peso siempre. Pero si el LLM
    # ya ha dicho «contradicha», una cifra rara añade poco; si ha dicho
    # «respaldada», esa cifra rara es justo lo que el LLM no ha visto. Duplicando
    # estas características multiplicadas por llm_ok, el modelo aprende un peso
    # distinto para cada situación.
    for nombre in list(f):
        if nombre.startswith(("num_", "matiz_", "anadido_", "huecos_", "nombre_")):
            f["llmok_" + nombre] = llm_ok * f[nombre]
    return f


# ---------------------------------------------------------------------------
# Modelo: regresión logística (pesos generados por entrenar.py)
# ---------------------------------------------------------------------------

# >>> PESOS (generado automáticamente por entrenar.py; no editar a mano)
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_sin_veredicto", "llm_conf_min_respaldada", "nli_resp_todas_min_orac", "nli_resp_todas_min_frag", "nli_contr_todas_max_orac", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "num_proporcion_falsa", "nombre_inventado", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_inexplicados", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_num_proporcion_falsa", "llmok_nombre_inventado", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_tramo_aproximado", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas"]
MEDIAS = [0.114821, 0.278266, 0.002929, 0.655952, 0.371689, 0.336231, 0.350083, 0.400602, 0.333558, 0.618182, 0.288552, 0.082015, 0.033392, 0.11365, 0.038664, 0.041593, 0.031634, 0.089631, 0.95372, 0.015231, 0.02519, 0.027534, 0.04628, 0.032806, 0.066198, 0.026362, 0.048623, 0.020504, 0.040422, 0.894552, 0.043937, 0.03925, 0.03925, 0.210679, 0.044523, 0.346807, 0.065709, 1.251904, 0.011716, 0.021675, 0.045694, 0.032806, 0.031634, 0.023433, 0.075571, 0.575864, 0.008787, 0.007616, 0.018746, 0.03222, 0.025776, 0.049209, 0.019332, 0.033392, 0.01406, 0.026362, 0.612771, 0.014646, 0.021675, 0.021675]
ESCALAS = [0.318806, 0.448145, 0.054042, 0.454544, 0.418673, 0.409101, 0.415509, 0.420962, 0.38974, 0.404686, 0.403628, 0.276515, 0.179658, 0.321056, 0.198778, 0.202571, 0.175025, 0.31305, 0.210091, 0.122472, 0.16401, 0.170643, 0.220963, 0.190831, 0.291974, 0.170827, 0.225711, 0.145791, 0.202808, 1.308576, 0.204954, 0.197183, 0.197183, 0.19087, 0.206253, 0.475954, 0.220545, 0.819962, 0.11292, 0.145622, 0.214358, 0.184589, 0.178341, 0.151274, 0.293706, 0.494211, 0.093328, 0.093431, 0.144008, 0.18939, 0.172622, 0.258274, 0.145951, 0.189187, 0.117737, 0.163825, 1.159521, 0.120129, 0.14959, 0.14959]
PESOS = [0.456213, 1.09674, 0.199239, -0.630703, -0.089509, -0.225308, -0.084017, -0.218414, -0.715089, -0.004995, -0.306505, -0.045244, -0.278434, -0.143115, -0.962031, -0.599192, -0.140884, -1.066344, 0.230963, 0.205952, 0.191273, -0.128953, 0.174432, -0.133417, 0.267304, 0.046893, 0.168144, 0.200957, -0.085004, -0.05774, -0.176952, -0.048444, -0.048444, -0.56363, 0.571417, -0.575724, -0.315883, -0.210423, -0.06068, 0.191103, -0.132318, 0.720436, 0.302032, 0.241425, 0.371454, -0.78027, -0.116035, 0.014756, 0.018727, 0.159346, 0.326078, 0.046396, -0.141396, -0.081718, 0.03177, 0.075255, 0.159542, 0.007774, 0.06936, 0.06936]
SESGO = 0.861564
UMBRAL = 0.15
# <<< PESOS


def probabilidad(feats):
    """Probabilidad de alucinación según la regresión logística."""
    z = SESGO
    for nombre, media, escala, peso in zip(NOMBRES, MEDIAS, ESCALAS, PESOS):
        z += peso * (feats.get(nombre, 0.0) - media) / escala
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def reescalar(p, umbral):
    """Lleva el corte elegido (umbral) a 0,5 manteniendo el orden de los casos:
    [0, umbral] -> [0, 0,5] y [umbral, 1] -> [0,5, 1]."""
    if p < umbral:
        return 0.5 * p / umbral
    return 0.5 + 0.5 * (p - umbral) / (1.0 - umbral)


class Guardian:
    def score(self, caso):
        try:
            p = probabilidad(caracteristicas(caso))
        except Exception:
            # Ante un caso raro, mejor dar la alarma (un escape cuesta 4 veces más)
            return 1.0
        return reescalar(p, UMBRAL)
