"""El Guardián — bot de joseca (versión 3).

v3: mismas características que v2, pero el modelo pasa de regresión logística
a gradient boosting (árboles de decisión), embebido en este fichero como datos.
En la validación por familias: H +0,36 -> +0,42, AUROC 0,877 -> 0,907.

--- Notas de la versión 2 ---

Cambios respecto a joseca_v1 (los comentarios con «v2» marcan cada uno):
  - cuentas anunciadas («9 % más», «X menos que»): si la cifra está tal cual
    en la fuente, es un dato y no una cuenta; y las diferencias sin otra cifra
    en la respuesta solo se buscan en fuentes que nombran a alguien de ella;
  - atribución cruzada rehecha: entidad principal frente a secundarias, y solo
    cuentan las fuentes que hablan de OTRA entidad comparable («la vecina...»);
  - nombres propios al principio de frase («Lusel retiene...»);
  - plazos («nueve meses después») y horas («10:05») comprobados;
  - recuentos limitados a 2, para que las respuestas largas (el giro) no
    disparen el score solo por tener más oraciones;
  - entrenar.py excluye características redundantes o rarísimas (ablación).

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
import collections
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

# Frases que anuncian que una fuente habla de OTRA entidad comparable (v2)
RE_OTRA_ENTIDAD = re.compile(r"\b(vecin[oa]s?|competidor[ae]?s?|rival(es)?|por su parte|en cambio|"
                             r"mientras que|a diferencia|otr[oa]s? (empresa|entidad|ciudad|municipio|"
                             r"comarca|asociacion|ong|fundacion|cooperativa|banco|ayuntamiento|"
                             r"hospital|universidad|operadora|region|provincia))\b")
# Una hora: 10:05, 8:30
RE_HORA = re.compile(r"\b\d{1,2}:\d{2}\b")
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
    # v2: las horas («10:05») no son dos cifras sueltas; las tapamos con espacios
    # (misma longitud, para no mover las posiciones) y se comprueban aparte.
    t = RE_HORA.sub(lambda m: " " * len(m.group()), t)
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


# ---------------------------------------------------------------------------
# Fechas y plazos (v2)
# ---------------------------------------------------------------------------

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
         "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
         "noviembre": 11, "diciembre": 12}
RE_FECHA = re.compile(r"(?:(\d{1,2}) de )?(" + "|".join(MESES) + r")(?: de(?:l)? (\d{4}))?")
RE_PLAZO = re.compile(r"\b(\d+|" + "|".join(UNIDADES) + r"|un|una)\s+(dias|semanas|meses|anos)\s+"
                      r"(despues|antes|mas tarde|mas)\b")


def _fechas(texto_norm):
    """Fechas del texto como (año, mes, día); año o día pueden faltar (None)."""
    fechas = []
    for m in RE_FECHA.finditer(texto_norm):
        dia = int(m.group(1)) if m.group(1) else None
        anio = int(m.group(3)) if m.group(3) else None
        fechas.append((anio, MESES[m.group(2)], dia))
    return fechas


def comprobar_plazos(respuesta, texto_fuentes):
    """Cuenta plazos («N meses después») que (no) salen de restar dos fechas.
    Devuelve (falsos, correctos)."""
    r = normalizar(respuesta)
    s = normalizar(texto_fuentes)
    falsos = correctos = 0
    for m in RE_PLAZO.finditer(r):
        palabra, unidad = m.group(1), m.group(2)
        n = 1 if palabra in ("un", "una") else (UNIDADES.get(palabra) or int(palabra))
        # Fechas disponibles: las de las fuentes y las que cita la propia respuesta.
        # Si la respuesta cita fechas, el plazo debe contarse DESDE una de ellas
        # («el 30 de junio de 2027, y nueve meses después»); si no, entre dos
        # fechas cualesquiera de las fuentes.
        f_fuentes, f_resp = _fechas(s), _fechas(r)
        a_fuentes = {int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", s)}
        a_resp = {int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", r)}
        diferencias = set()
        if unidad == "anos":
            origen = a_resp or a_fuentes
            diferencias = {abs(a - b) for a in origen for b in a_fuentes | a_resp}
        elif unidad == "meses":
            # Completamos el año que falte con el de otra fecha («junio» -> «junio de 2027»)
            con_anio = [a for a, _, _ in f_resp + f_fuentes if a]
            defecto = con_anio[0] if con_anio else 2000
            m_fuentes = {(a or defecto) * 12 + mes for a, mes, _ in f_fuentes}
            m_resp = {(a or defecto) * 12 + mes for a, mes, _ in f_resp}
            origen = m_resp or m_fuentes
            diferencias = {abs(x - y) for x in origen for y in m_fuentes | m_resp}
        else:
            # Días o semanas: si no podemos calcularlo, no opinamos
            continue
        if not diferencias - {0}:
            continue
        if n in diferencias:
            correctos += 1
        else:
            falsos += 1
    return falsos, correctos


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
    # Cifras de las fuentes que nombran a alguien que también nombra la respuesta
    nombres_fuentes = [nombres_propios(x) for x in fuentes]
    # nombres_propios() ignora la primera palabra de cada frase (siempre va en
    # mayúscula). Pero «Lusel retiene...» empieza por un nombre: lo recuperamos
    # si esa palabra aparece como nombre propio en las fuentes.
    nombres_conocidos = set().union(*nombres_fuentes) if nombres_fuentes else set()
    nombres_resp = nombres_propios(respuesta) | {
        normalizar(m) for m in re.findall(r"(?:^|[.;:!?]\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)", respuesta)
        if normalizar(m) in nombres_conocidos}
    fuentes_nombradas = set(fuentes_base) | {i for i, nf in enumerate(nombres_fuentes) if nf & nombres_resp}
    ops_nombradas = set(ops_locales)
    for i in fuentes_nombradas:
        ops_nombradas |= operandos_de(nums_por_fuente[i])
    der_locales = valores_derivados(ops_locales) if nums_resp else set()
    der_globales = valores_derivados(ops_globales) if nums_resp else set()
    resp_norm = normalizar(respuesta)

    clases = []
    for n in nums_resp:
        # Otras cifras de la respuesta («anclas»): una cuenta fiel suele usarlas
        anclas = operandos_de([d for d in directos if d is not n])
        tipo = tipo_de_cuenta(resp_norm, n)
        if tipo and not n["anio"] and any(esta_en(n, nums_por_fuente[i]) for i in fuentes_base):
            # «pagarán un 9 % más»: no es una cuenta, el 9 % está tal cual en la
            # fuente en la que se apoya la respuesta. (v2: esto se mira ANTES de
            # buscar cuentas; si no, alguna resta casual lo «explicaba».)
            clases.append("directo")
        elif tipo and not n["anio"]:
            # La respuesta anuncia una cuenta: comprobamos ESA operación
            if (cuenta_cuadra(tipo, n, anclas, ops_locales) or
                    # «11 pozos menos que la otra ONG»: la otra cifra puede estar en
                    # otra fuente, pero (v2) solo en una que nombre a alguien de la
                    # respuesta, no en cualquiera
                    (tipo == "diferencia" and not anclas and cuenta_cuadra(tipo, n, set(), ops_nombradas))):
                clases.append("cuenta_correcta")
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
    # v2: distinguimos la entidad PRINCIPAL (la de la pregunta o la que sale en
    # muchas fuentes: «Lusel», «Valdemarina») de las SECUNDARIAS (las que solo
    # aparecen en alguna fuente: «Banca Ordial», «Algaida Baja», «Fuenvieja»).
    # Si una cifra solo está en fuentes que hablan de una entidad secundaria y
    # la respuesta no nombra a esa entidad, la cifra se está atribuyendo a otro.
    # Ej.: «Valdemarina tiene un 7,9 % de extranjeros», cuando el 7,9 % es de Algaida Baja.
    nombres_pregunta = {normalizar(p) for p in re.findall(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+", pregunta)}
    frecuencia = collections.Counter(n for nf in nombres_fuentes for n in nf)
    principales = nombres_pregunta | {n for n, k in frecuencia.items() if k >= 3}
    secundarios = [nf - principales for nf in nombres_fuentes]
    # Solo cuentan las fuentes que hablan de OTRA entidad comparable: lo dicen
    # («la comarca vecina», «competidora», «por su parte») o repiten el tipo de
    # entidad de la pregunta con otro nombre («Banco de Alimentos de Cuevalta»).
    # Una «Universidad de Lambrera» que firma un estudio no es otra entidad.
    otra_entidad = [bool(secundarios[i]) and (bool(RE_OTRA_ENTIDAD.search(normalizar(x)))
                                              or bool(nombres_fuentes[i] & nombres_pregunta))
                    for i, x in enumerate(fuentes)]
    ajenas = ajenas_fuerte = 0
    for n, c in zip(nums_resp, clases):
        if c != "directo" or n["anio"]:
            continue
        donde = [i for i, ns in enumerate(nums_por_fuente) if esta_en(n, ns)]
        if donde and all(otra_entidad[i] and not (secundarios[i] & nombres_resp) for i in donde):
            ajenas += 1
            # Más grave si además la respuesta nombra explícitamente a la principal
            if nombres_resp & principales:
                ajenas_fuerte += 1
    f["entidad_ajena"] = float(min(ajenas, 2))
    f["entidad_ajena_fuerte"] = float(min(ajenas_fuerte, 2))
    # Nombres propios de la respuesta que no salen en ninguna fuente ni en la pregunta
    todos_nombres = set().union(*nombres_fuentes) if nombres_fuentes else set()
    conocidos = todos_nombres | {normalizar(p) for p in re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", pregunta)}
    f["nombre_inventado"] = float(len(nombres_resp - conocidos))

    # --- 4b. Fechas, plazos y horas (v2) ---
    # «nueve meses después», «tres años antes»: el plazo tiene que salir de
    # restar dos fechas de las fuentes. «a las 10:05»: la hora tiene que estar.
    falsos, correctos = comprobar_plazos(respuesta, " ".join(fuentes) + " " + pregunta)
    f["fecha_plazo_falso"] = float(min(falsos, 2))
    f["fecha_plazo_correcto"] = float(min(correctos, 2))
    horas_fuentes = set(RE_HORA.findall(" ".join(fuentes)))
    f["hora_inventada"] = float(any(h not in horas_fuentes for h in RE_HORA.findall(respuesta)))

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
    # v2: los recuentos se limitan a 2. Con respuestas largas (el giro) hay más
    # oraciones y más fuentes citadas, y la suma crecería solo por la longitud.
    for k in MATICES:
        f["matiz_perdido_tramo_" + k] = float(min(perdidos_tramo[k], 2))
        f["matiz_perdido_fuente_" + k] = float(min(perdidos_fuente[k], 2))
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
    cifras_ok = float(f["num_sospechosos_grandes"] == 0 and f["entidad_ajena"] == 0
                      and f["fecha_plazo_falso"] == 0)
    f["llm_contra_sin_apoyo"] = f["llm_contradicha"] * cifras_ok * (1.0 - f["nli_contr_todas_max_frag"])
    f["n_oraciones"] = float(len(oraciones))

    # --- 8. Las comprobaciones propias, en el régimen «el LLM dice que está bien» ---
    # Un modelo lineal da a cada sospecha el mismo peso siempre. Pero si el LLM
    # ya ha dicho «contradicha», una cifra rara añade poco; si ha dicho
    # «respaldada», esa cifra rara es justo lo que el LLM no ha visto. Duplicando
    # estas características multiplicadas por llm_ok, el modelo aprende un peso
    # distinto para cada situación.
    for nombre in list(f):
        if nombre.startswith(("num_", "matiz_", "anadido_", "huecos_", "nombre_", "entidad_",
                              "fecha_", "hora_")):
            f["llmok_" + nombre] = llm_ok * f[nombre]
    return f


# ---------------------------------------------------------------------------
# Modelo: gradient boosting (árboles generados por entrenar.py)
# ---------------------------------------------------------------------------
# v3: en vez de una regresión logística (una suma de pesos), usamos «gradient
# boosting»: muchos árboles de decisión pequeños (3 niveles de preguntas tipo
# «¿llm_contradicha > 0,5?», «¿num_cuenta_correcta > 0,5?») que se suman, y
# cada uno corrige los errores de los anteriores. Así el modelo capta
# combinaciones («el LLM dice OK, PERO se ha perdido un "si se aprueba"») que
# una suma de pesos no puede representar.
#
# Los árboles se entrenan con scikit-learn en entrenar.py y se escriben aquí
# como listas de números, así que este fichero NO necesita scikit-learn:
# recorrerlos es Python puro.
#
# Cada árbol es (variable, umbral, hijo_izq, hijo_der, valor) por nodo; en las
# hojas hijo_izq = -1 y «valor» es lo que el árbol suma.

# >>> MODELO (generado automáticamente por entrenar.py; no editar a mano)
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas"]
INICIAL = -0.035456
TASA = 0.03
ARBOLES = [[[2,3,36,-2,-2,4,-2,-2,25,10,-2,-2,34,-2,-2],[0.935,0.0475,0.5,-2.0,-2.0,0.2845,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003123,0.414307,0.470507,1.998345,1.099628,0.284888,1.604819,0.622315,-0.180073,-0.229503,-1.075064,0.575633,0.246568,1.723938,-0.022807]],[[2,36,3,-2,-2,5,-2,-2,25,0,-2,-2,34,-2,-2],[0.935,0.5,0.0325,-2.0,-2.0,0.023,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002144,0.392471,0.429736,1.935119,1.295863,0.22132,1.876183,0.37438,-0.179774,-0.226511,-0.982608,2.070053,0.248609,1.98387,0.079975]],[[2,45,3,-2,-2,5,-2,-2,25,45,-2,-2,34,-2,-2],[0.925,0.5,0.0325,-2.0,-2.0,0.023,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.183333,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004804,0.378497,0.432915,1.879301,1.446332,0.178807,1.774794,0.251579,-0.177927,-0.224793,0.58765,-1.070632,0.251357,1.588907,0.033748]],[[2,45,3,-2,-2,5,-2,-2,25,0,-2,-2,34,-2,-2],[0.935,0.5,0.0325,-2.0,-2.0,0.0235,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.01515,0.363198,0.403706,1.821913,1.238751,0.186584,1.752476,0.357282,-0.18464,-0.226141,-0.978926,1.911674,0.207534,1.701358,0.030788]],[[2,45,3,-2,-2,5,-2,-2,2,34,-2,-2,0,-2,-2],[0.935,0.5,0.0325,-2.0,-2.0,0.0235,-2.0,-2.0,0.96,0.261364,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.012032,0.35095,0.400511,1.776073,1.25883,0.145871,1.64852,0.141269,-0.174518,-0.03262,0.22587,-1.008253,-0.261423,-1.160517,1.857008]],[[2,7,3,-2,-2,36,-2,-2,25,0,-2,-2,34,-2,-2],[0.935,0.0075,0.636,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.154762,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00545,0.355431,0.418103,1.740652,0.956535,0.252989,1.256792,0.002126,-0.173013,-0.204586,-0.90294,1.756185,0.1876,1.81423,-0.385168]],[[2,45,3,-2,-2,7,-2,-2,2,34,-2,-2,0,-2,-2],[0.935,0.5,0.3485,-2.0,-2.0,0.0025,-2.0,-2.0,0.96,0.035714,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006944,0.34121,0.386845,1.68804,0.999488,0.145537,1.395442,0.158317,-0.151078,-0.011491,0.703129,-0.373147,-0.234229,-1.064666,1.839059]],[[45,5,3,-2,-2,34,-2,-2,2,14,-2,-2,35,-2,-2],[0.5,0.555,0.0065,-2.0,-2.0,0.236111,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.014763,0.325843,0.380603,1.710586,1.33581,-0.001999,0.704847,-1.517175,-0.170898,-0.005156,0.163225,-1.48497,-0.292524,-1.312769,0.240933]],[[2,36,3,-2,-2,5,-2,-2,25,10,-2,-2,34,-2,-2],[0.925,0.5,0.253,-2.0,-2.0,0.023,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003328,0.31608,0.354402,1.589422,0.938377,0.112605,1.669938,-0.172962,-0.140998,-0.177819,-0.860225,0.602893,0.208373,1.506135,-0.148124]],[[2,36,3,-2,-2,7,-2,-2,5,45,-2,-2,10,-2,-2],[0.935,0.5,0.0325,-2.0,-2.0,0.0075,-2.0,-2.0,0.0515,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001114,0.319252,0.350507,1.633198,1.126789,0.158017,1.249315,-0.329907,-0.143081,0.064373,1.805266,-0.066903,-0.202453,-0.946307,0.53149]],[[45,59,5,-2,-2,-2,2,14,-2,-2,25,-2,-2],[0.5,0.5,0.529,-2.0,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.00063,0.292326,0.334311,1.506505,0.651253,-0.73463,-0.137232,0.004626,0.202128,-1.41674,-0.247494,-1.154617,0.706727]],[[2,5,34,-2,-2,-2,45,38,-2,-2,34,-2,-2],[0.935,0.669,0.449495,-2.0,-2.0,-2.0,0.5,1.5,-2.0,-2.0,0.253571,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-1.2e-05,0.302456,0.326925,1.42542,0.574499,0.376869,-0.138847,0.154077,-0.30695,1.805171,-0.171827,-0.417313,-1.34109]],[[2,45,3,-2,-2,7,-2,-2,25,10,-2,-2,34,-2,-2],[0.935,0.5,0.3365,-2.0,-2.0,0.0025,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001709,0.283448,0.317974,1.471902,0.747491,0.141168,1.346616,0.160478,-0.128844,-0.165275,-0.836995,0.798594,0.246912,1.505534,0.17714]],[[2,3,45,-2,-2,4,-2,-2,24,10,-2,-2,-2],[0.935,0.0325,0.5,-2.0,-2.0,0.195,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.004833,0.266861,0.318487,1.496119,0.58211,0.181157,1.217207,0.320154,-0.132135,-0.160171,-0.789465,0.542574,1.236247]],[[2,5,37,-2,-2,34,-2,-2,2,55,-2,-2,5,-2,-2],[0.935,0.023,0.8,-2.0,-2.0,0.242647,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.0045,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001409,0.272573,0.334713,1.518487,1.053757,0.222984,1.143252,0.321914,-0.125809,0.012271,-0.12678,1.210043,-0.207115,1.208242,-0.971668]],[[2,3,36,-2,-2,4,-2,-2,25,10,-2,-2,34,-2,-2],[0.935,0.0475,0.5,-2.0,-2.0,0.4435,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.154762,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006913,0.2616,0.309727,1.452004,0.914615,0.17653,1.073684,0.008325,-0.129542,-0.164247,-0.814439,0.683906,0.226773,1.556758,0.033923]],[[45,5,31,-2,-2,34,-2,-2,2,44,-2,-2,55,-2,-2],[0.5,0.6655,0.5,-2.0,-2.0,0.236111,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004811,0.250304,0.299842,1.398014,0.543705,-0.037143,0.407757,-1.463439,-0.122812,0.004999,0.202933,-1.533248,-0.227559,-1.091594,0.434726]],[[45,5,38,-2,-2,59,-2,-2,2,44,-2,-2,34,-2,-2],[0.5,0.6655,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.303846,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003756,0.253986,0.299039,1.287764,1.782849,0.009378,0.697739,-1.03761,-0.113526,0.023398,0.240071,-1.27453,-0.220908,-0.732735,-1.474358]],[[2,5,34,-2,-2,1,-2,-2,25,10,-2,-2,34,-2,-2],[0.935,0.1475,0.449495,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.183333,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001481,0.235803,0.277384,1.322369,0.558062,0.136758,-0.023411,1.206985,-0.109466,-0.140536,-0.727217,0.63753,0.194968,1.37728,-0.136108]],[[45,5,38,-2,-2,-2,55,2,-2,-2,34,-2,-2],[0.5,0.963,1.5,-2.0,-2.0,-2.0,0.5,0.96,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002505,0.240052,0.267554,1.153504,1.732668,-0.678717,-0.104725,-0.13959,-0.121475,-0.942388,0.199807,1.386542,-0.039551]],[[45,59,38,-2,-2,-2,2,53,-2,-2,10,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004765,0.227582,0.26498,1.157683,1.744439,-0.509454,-0.10145,0.024521,-0.073513,1.775737,-0.202049,-0.974287,0.409578]],[[2,7,3,-2,-2,34,-2,-2,24,10,-2,-2,-2],[0.935,0.0075,0.0125,-2.0,-2.0,0.236111,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003633,0.228295,0.27464,1.248818,1.416399,0.156499,0.974227,-0.101583,-0.101757,-0.122343,-0.647288,0.750116,1.037393]],[[45,5,38,-2,-2,59,-2,-2,55,2,-2,-2,34,-2,-2],[0.5,0.663,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.96,-2.0,-2.0,0.154762,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003627,0.207322,0.253131,1.13875,1.696851,-0.054601,0.419058,-1.040964,-0.104672,-0.138376,-0.088311,-0.989279,0.225056,1.481926,0.053535]],[[2,7,5,-2,-2,36,-2,-2,23,25,-2,-2,34,-2,-2],[0.935,0.0015,0.116,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.171569,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003815,0.210728,0.278204,1.335721,1.495351,0.175982,0.994383,0.171411,-0.095715,-0.122194,-0.646046,0.673438,0.228544,1.763717,-0.258895]],[[2,7,5,-2,-2,36,-2,-2,24,10,-2,-2,-2],[0.935,0.0015,0.1175,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.007677,0.204314,0.269737,1.331382,1.462191,0.17128,0.991825,0.100827,-0.10416,-0.123041,-0.658678,0.594011,0.999273]],[[45,59,38,-2,-2,-2,2,14,-2,-2,25,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001622,0.214163,0.240686,1.110664,1.707384,-0.438465,-0.09231,0.008516,0.196308,-1.305077,-0.176452,-0.894797,0.395031]],[[45,5,31,-2,-2,34,-2,-2,25,40,-2,-2,34,-2,-2],[0.5,0.6655,0.5,-2.0,-2.0,0.212862,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004264,0.183554,0.221625,1.200892,0.060421,-0.025374,0.391609,-1.286742,-0.092211,-0.126826,-0.709503,0.436366,0.191628,1.410447,0.157396]],[[2,14,11,-2,-2,-2,0,1,-2,-2,5,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.0025,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.006412,0.119061,0.144671,0.764672,-1.209086,-1.200259,-0.135765,-0.15746,-0.773328,1.876537,0.353581,1.53667,1.568857]],[[45,5,38,-2,-2,34,-2,-2,2,12,-2,-2,25,-2,-2],[0.5,0.6655,1.5,-2.0,-2.0,0.236111,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005509,0.187972,0.223633,1.087089,1.590207,-0.018412,0.483781,-1.446269,-0.081287,0.020369,0.204173,-1.29799,-0.166794,-0.88722,0.566475]],[[2,7,59,-2,-2,36,-2,-2,25,10,-2,-2,6,-2,-2],[0.935,0.0075,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.67,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006482,0.176997,0.221351,1.126626,1.462637,0.111526,0.830401,-0.594338,-0.09388,-0.117241,-0.643155,0.497288,0.161871,1.29125,0.151487]],[[2,3,36,-2,-2,6,-2,-2,3,6,-2,-2,34,-2,-2],[0.935,0.344,0.5,-2.0,-2.0,0.9465,-2.0,-2.0,0.0115,0.9025,-2.0,-2.0,0.261364,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004006,0.182169,0.204127,1.218857,0.484659,0.082394,0.006823,1.49586,-0.080084,0.061422,0.99612,-0.268811,-0.135144,-0.356044,-1.099961]],[[2,3,36,-2,-2,4,-2,-2,34,30,-2,-2,3,-2,-2],[0.935,0.0755,0.5,-2.0,-2.0,0.442,-2.0,-2.0,0.113248,1.5,-2.0,-2.0,0.114,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007462,0.169295,0.198283,1.227091,0.380471,0.100285,0.93589,-0.654218,-0.088248,0.023458,-0.302838,1.048067,-0.147875,-0.223447,-1.081822]],[[45,59,38,-2,-2,-2,44,2,-2,-2,34,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.975,-2.0,-2.0,0.309524,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.006605,0.165593,0.19074,0.940284,1.655289,-0.747873,-0.08701,-0.05911,0.114381,-0.672754,-0.304088,-1.53942,-1.151758]],[[2,44,11,-2,-2,-2,5,4,-2,-2,10,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.0065,0.5815,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.016619,0.113025,0.133157,0.747679,-1.092326,-1.315895,-0.110678,0.236752,0.641326,2.115604,-0.135127,-0.739383,0.668889]],[[2,7,3,-2,-2,1,-2,-2,25,23,-2,-2,34,-2,-2],[0.935,0.0075,0.6865,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005805,0.159717,0.207651,1.252013,0.472228,0.086419,-0.087868,0.930973,-0.063711,-0.094084,-0.5352,0.678704,0.220132,1.514331,0.310011]],[[45,5,38,-2,-2,-2,23,25,-2,-2,-2],[0.5,0.9655,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.002243,0.166527,0.184829,0.941568,1.534806,-0.501303,-0.073997,-0.093988,-0.539591,0.602878,0.853032]],[[45,5,38,-2,-2,2,-2,-2,2,53,-2,-2,5,-2,-2],[0.5,0.6655,1.5,-2.0,-2.0,0.985,-2.0,-2.0,0.975,0.5,-2.0,-2.0,0.6755,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001387,0.149583,0.184066,0.983432,1.476608,-0.064491,0.184724,-1.027816,-0.066893,0.009834,-0.10597,1.461283,-0.148161,-0.308347,-1.097131]],[[45,5,38,-2,-2,30,-2,-2,55,40,-2,-2,34,-2,-2],[0.5,0.6655,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005497,0.134248,0.171195,0.92541,1.455411,-0.079731,0.822784,-0.794664,-0.056633,-0.084059,-0.536683,0.755579,0.189488,1.522513,0.157979]],[[2,14,12,-2,-2,-2,0,10,-2,-2,34,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.381048,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.005559,0.085278,0.109178,0.646796,-1.226177,-1.32333,-0.098501,-0.115371,-0.657707,0.807446,0.28014,1.422557,1.184964]],[[2,14,11,-2,-2,-2,1,0,-2,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.005275,0.095287,0.114937,0.671847,-1.173813,-1.046238,-0.115863,-0.12876,-0.710111,1.321837,1.750986]],[[2,14,11,-2,-2,4,-2,-2,5,36,-2,-2,34,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,0.5575,-2.0,-2.0,0.245,0.5,-2.0,-2.0,0.114379,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006097,0.086662,0.108988,0.649441,-1.001001,-0.237773,-1.71511,-0.419731,-0.106297,0.011391,1.472029,-0.221195,-0.16855,-0.377949,-1.134739]],[[45,5,38,-2,-2,6,-2,-2,2,14,-2,-2,34,-2,-2],[0.5,0.819,1.5,-2.0,-2.0,0.5575,-2.0,-2.0,0.975,0.5,-2.0,-2.0,0.114379,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004843,0.137698,0.159305,0.893191,1.410169,-0.033753,0.634705,-1.005646,-0.073131,-0.002156,0.125322,-1.277507,-0.148574,-0.290956,-1.043087]],[[2,7,59,-2,-2,36,-2,-2,25,10,-2,-2,34,-2,-2],[0.935,0.0075,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006447,0.13827,0.167703,1.021074,1.292878,0.091708,0.776457,-0.52673,-0.074782,-0.096304,-0.576332,0.56853,0.130958,1.327816,-0.297131]],[[2,4,2,-2,-2,6,-2,-2,23,3,-2,-2,34,-2,-2],[0.935,0.0165,0.4,-2.0,-2.0,0.9485,-2.0,-2.0,0.5,0.0055,-2.0,-2.0,0.171569,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001876,0.139318,0.208316,1.20082,1.260619,0.122771,0.406967,1.054673,-0.06527,-0.085664,0.236208,-0.596216,0.176726,1.623368,-0.475432]],[[45,5,38,-2,-2,34,-2,-2,23,25,-2,-2,-2],[0.5,0.663,1.5,-2.0,-2.0,0.236111,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.006113,0.122108,0.152348,0.889959,1.486736,-0.043319,0.393265,-1.643916,-0.047839,-0.068804,-0.436694,0.620752,0.949845]],[[45,5,38,-2,-2,34,-2,-2,23,13,-2,-2,-2],[0.5,0.663,1.5,-2.0,-2.0,0.212862,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.011454,0.120045,0.143221,0.856305,1.345153,-0.030112,0.436494,-1.102286,-0.072093,-0.091325,-0.533455,0.730831,0.787312]],[[45,3,38,-2,-2,7,-2,-2,34,53,-2,-2,2,-2,-2],[0.5,0.253,1.5,-2.0,-2.0,0.784,-2.0,-2.0,0.343137,0.5,-2.0,-2.0,0.925,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.008108,0.124008,0.156114,0.963049,1.380223,-0.012754,0.515751,-0.954075,-0.045867,-0.011341,-0.16481,1.240471,-0.19634,0.638205,-1.153865]],[[45,59,38,-2,-2,-2,44,2,-2,-2,34,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.96,-2.0,-2.0,0.309524,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.005058,0.120935,0.146207,0.890431,1.351598,-0.685998,-0.06388,-0.040207,0.178441,-0.5242,-0.263851,-1.416512,-1.262144]],[[5,3,44,-2,-2,34,-2,-2,2,34,-2,-2,5,-2,-2],[0.6755,0.0385,0.5,-2.0,-2.0,0.280788,-2.0,-2.0,0.995,0.027778,-2.0,-2.0,0.9615,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.009627,0.057087,0.123243,0.781642,-0.682315,-0.02913,0.114137,-0.906832,-0.098099,-0.038095,0.831436,-0.583999,-0.224414,-1.017712,-1.335852]],[[2,5,45,-2,-2,1,-2,-2,34,2,-2,-2,3,-2,-2],[0.935,0.023,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.146429,0.985,-2.0,-2.0,0.1115,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008155,0.108673,0.154079,1.074947,1.189895,0.071383,-0.024919,0.879608,-0.061869,0.009197,0.454105,-0.565526,-0.119313,-0.176785,-0.965813]],[[2,11,14,-2,-2,-2,5,3,-2,-2,25,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.0685,0.071,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002129,0.069051,0.087706,0.593356,-1.225055,-1.048286,-0.085145,0.085841,1.221645,-0.687976,-0.125293,-0.77982,0.493094]],[[2,7,5,-2,-2,6,-2,-2,25,10,-2,-2,34,-2,-2],[0.935,0.0015,0.0565,-2.0,-2.0,0.696,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001987,0.115543,0.156649,1.085748,1.281003,0.094457,0.989707,0.358022,-0.05994,-0.084618,-0.524976,0.417309,0.212971,1.489989,0.176441]],[[45,3,38,-2,-2,30,-2,-2,53,54,-2,-2,-2],[0.5,0.0545,1.5,-2.0,-2.0,2.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002025,0.102046,0.139595,0.937494,1.340916,0.011311,0.411748,-1.13896,-0.050985,-0.071043,-0.438271,1.114004,0.942801]],[[2,5,7,-2,-2,5,-2,-2,10,34,-2,-2,6,-2,-2],[0.935,0.1475,0.0415,-2.0,-2.0,0.2335,-2.0,-2.0,0.5,0.196154,-2.0,-2.0,0.719,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001769,0.107449,0.129143,0.892405,1.374688,0.060199,-0.576027,0.586322,-0.046974,-0.066441,0.032529,-0.746686,0.147783,-0.233466,0.982131]],[[5,3,12,-2,-2,2,-2,-2,34,2,-2,-2,1,-2,-2],[0.0685,0.071,0.5,-2.0,-2.0,0.925,-2.0,-2.0,0.129167,0.985,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00946,0.10258,0.158723,1.116789,-1.187007,-0.042209,0.762509,-0.574987,-0.044279,0.044316,0.65012,-0.471715,-0.116291,-0.727656,0.924339]],[[2,3,2,-2,-2,4,-2,-2,10,23,-2,-2,6,-2,-2],[0.935,0.0475,0.825,-2.0,-2.0,0.341,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.719,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002032,0.100668,0.127675,0.974142,1.024904,0.048831,0.813212,-0.470363,-0.043765,-0.066915,-0.434818,0.641297,0.207944,-0.066853,1.35303]],[[14,5,3,-2,-2,2,-2,-2,6,2,-2,-2,-2],[0.5,0.664,0.0925,-2.0,-2.0,0.975,-2.0,-2.0,0.986,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001957,0.018401,0.059919,0.619219,-0.123633,-0.073928,0.273956,-0.873215,-0.208791,-0.254762,-1.541405,-1.268644,-0.490388]],[[45,59,38,-2,-2,-2,53,25,-2,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.002199,0.091831,0.111212,0.739652,1.315832,-0.414075,-0.040264,-0.061332,-0.442129,0.708908,1.017675]],[[2,42,11,-2,-2,-2,1,5,-2,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.0045,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000662,0.065408,0.080162,0.557634,-1.026547,-1.217321,-0.083077,-0.09284,1.113,-0.580216,1.586908]],[[1,5,3,-2,-2,34,-2,-2,38,34,-2,-2,-2],[0.5,0.047,0.071,-2.0,-2.0,0.129167,-2.0,-2.0,1.5,0.211111,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001979,-0.028537,0.059757,0.658926,-0.259023,-0.060797,0.045072,-0.581832,0.126417,0.114166,1.086659,0.555102,1.571443]],[[2,5,2,-2,-2,-2,23,10,-2,-2,34,-2,-2],[0.935,0.669,0.835,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.179144,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003319,0.102061,0.114453,0.842103,0.978369,-0.079945,-0.0447,-0.062596,-0.433566,0.589789,0.141005,1.369074,-0.645087]],[[2,12,11,-2,-2,-2,5,3,-2,-2,34,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.261,0.0875,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004213,0.056453,0.070806,0.503134,-1.043961,-1.076822,-0.066817,0.035091,0.638102,-0.42171,-0.121515,-0.371183,-0.931572]],[[2,4,3,-2,-2,3,-2,-2,24,10,-2,-2,-2],[0.935,0.1765,0.0205,-2.0,-2.0,0.0655,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003628,0.089053,0.140499,0.770639,1.077383,0.061151,0.84327,-0.38657,-0.038972,-0.056001,-0.403138,0.595263,1.181007]],[[2,14,11,-2,-2,-2,5,3,-2,-2,30,-2,-2],[0.975,0.5,0.5,-2.0,-2.0,-2.0,0.261,0.0875,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.01025,0.057627,0.075066,0.542014,-1.100832,-1.042857,-0.071559,0.042367,0.733448,-0.568633,-0.134454,-1.230251,-0.533278]],[[14,5,12,-2,-2,2,-2,-2,34,2,-2,-2,-2],[0.5,0.826,0.5,-2.0,-2.0,0.975,-2.0,-2.0,0.309524,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00208,0.016726,0.054644,0.404733,-0.864286,-0.086858,0.205998,-0.85688,-0.213892,-0.258147,-1.589146,-1.285675,-0.809643]],[[14,12,5,-2,-2,30,-2,-2,7,-2,5,-2,-2],[0.5,0.5,0.8135,-2.0,-2.0,0.5,-2.0,-2.0,0.011,-2.0,0.3675,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.007694,0.008768,0.023488,0.340792,-0.431626,-0.249496,-1.733226,-0.910527,-0.203784,-0.940278,-0.239599,-1.266683,-1.564288]],[[14,12,2,-2,-2,59,-2,-2,34,2,-2,-2,-2],[0.5,0.5,0.975,-2.0,-2.0,0.5,-2.0,-2.0,0.309524,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003082,0.019466,0.033028,0.451057,-0.243751,-0.198998,-1.599776,-0.564342,-0.200783,-0.242979,-1.53513,-1.269746,-0.817994]],[[5,12,3,-2,-2,-2,34,10,-2,-2,4,-2,-2],[0.0415,0.5,0.2225,-2.0,-2.0,-2.0,0.394444,0.5,-2.0,-2.0,0.905,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003827,0.073797,0.092272,0.85638,-0.227663,-1.524904,-0.040244,-0.016814,-0.206409,0.670428,-0.184644,-0.987461,-1.592553]],[[14,11,34,-2,-2,7,-2,-2,3,-2,2,-2,-2],[0.5,0.5,0.475294,-2.0,-2.0,0.0395,-2.0,-2.0,0.014,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000741,0.015375,0.028873,0.242154,-0.987286,-0.21196,-0.5611,-1.642933,-0.183373,-0.775363,-0.213682,-1.448495,-1.208327]],[[2,9,2,-2,-2,-2,23,25,-2,-2,34,-2,-2],[0.935,0.5,0.825,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.160256,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002592,0.077669,0.090691,0.639646,0.961111,-0.477989,-0.033496,-0.051566,-0.384503,0.660086,0.167127,1.547779,-0.271552]],[[45,38,2,-2,-2,37,-2,-2,25,53,-2,-2,34,-2,-2],[0.5,1.5,0.96,-2.0,-2.0,0.0005,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.154762,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.013437,0.069714,0.052279,0.604957,-0.756824,0.248014,1.212331,1.543859,-0.053647,-0.074752,-0.483305,0.514464,0.149442,1.187363,0.22421]],[[34,14,12,-2,-2,7,-2,-2,5,-2,2,-2,-2],[0.458042,0.5,0.5,-2.0,-2.0,0.011,-2.0,-2.0,0.0125,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001218,0.016289,0.031384,0.270299,-1.027205,-0.189628,-0.845857,-1.345324,-0.180802,0.191778,-0.222804,-1.485593,-1.170045]],[[5,12,34,-2,-2,-2,34,10,-2,-2,6,-2,-2],[0.0475,0.5,0.474713,-2.0,-2.0,-2.0,0.343137,0.5,-2.0,-2.0,0.965,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000596,0.077732,0.089993,0.706323,-0.553878,-1.262978,-0.039288,-0.016771,-0.204171,0.737256,-0.15377,-1.215765,-0.199882]],[[2,12,34,-2,-2,-2,34,30,-2,-2,1,-2,-2],[0.96,0.5,0.468627,-2.0,-2.0,-2.0,0.11678,1.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003753,0.046114,0.059136,0.437427,-0.776749,-1.019771,-0.055558,0.044005,-0.251008,1.469742,-0.105392,-0.724689,1.516456]],[[2,12,14,-2,-2,-2,5,7,-2,-2,25,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.0065,0.0025,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004884,0.048862,0.063226,0.480243,-0.952824,-0.968248,-0.057017,0.160021,-0.058021,1.757389,-0.074533,-0.553293,0.544235]],[[5,12,3,-2,-2,-2,2,11,-2,-2,30,-2,-2],[0.0475,0.5,0.087,-2.0,-2.0,-2.0,0.96,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003839,0.071122,0.091079,0.879523,-0.030374,-1.548857,-0.041038,0.014903,0.191021,-1.19376,-0.08808,-1.000117,-0.285248]],[[34,55,2,-2,-2,7,-2,-2,5,3,-2,-2,1,-2,-2],[0.27682,0.5,0.975,-2.0,-2.0,0.034,-2.0,-2.0,0.0235,0.0295,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007605,0.038061,0.022159,0.394848,-0.301959,0.218447,0.901386,1.264041,-0.068284,0.062344,0.833258,-0.709498,-0.1208,-0.834328,0.96369]],[[2,12,34,-2,-2,4,-2,-2,1,5,-2,-2,-2],[0.975,0.5,0.03125,-2.0,-2.0,0.745,-2.0,-2.0,0.5,0.6755,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.008993,0.029417,0.044685,0.907631,0.007326,-0.20313,-1.487228,-0.245794,-0.072558,-0.081527,-0.128175,-0.852357,1.546111]],[[2,11,12,-2,-2,-2,10,1,-2,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.013589,0.054034,0.066472,0.489927,-0.755005,-0.950918,-0.040133,-0.053869,-0.393252,1.480568,0.696144]],[[14,11,2,-2,-2,6,-2,-2,3,-2,2,-2,-2],[0.5,0.5,0.975,-2.0,-2.0,0.8865,-2.0,-2.0,0.012,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.008316,0.021583,0.034533,0.435029,-0.172492,-0.201432,-1.890096,-0.166662,-0.154736,-0.308679,-0.203506,-1.454024,-1.215092]],[[14,2,9,-2,-2,34,-2,-2,3,-2,2,-2,-2],[0.5,0.935,0.5,-2.0,-2.0,0.113248,-2.0,-2.0,0.012,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000748,0.014279,0.075926,0.737296,-0.241244,-0.016996,0.324474,-0.333879,-0.170506,-0.672171,-0.196066,-1.436462,-1.208594]],[[11,14,5,-2,-2,34,-2,-2,3,-2,-2],[0.5,0.5,0.6845,-2.0,-2.0,0.309524,-2.0,-2.0,0.095,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001438,0.011796,0.026393,0.336789,-0.202958,-0.164197,-1.350859,-0.470409,-0.180114,-0.301377,-1.788546]],[[1,53,5,-2,-2,34,-2,-2,2,37,-2,-2,5,-2,-2],[0.5,0.5,0.0325,-2.0,-2.0,0.171569,-2.0,-2.0,0.475,0.5445,-2.0,-2.0,0.0205,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000453,-0.021555,-0.03472,0.311685,-0.34894,0.152234,1.51114,-0.438816,0.092414,0.078197,1.072129,0.246568,0.29009,1.337285,1.599142]],[[2,36,59,-2,-2,34,-2,-2,5,6,-2,-2,30,-2,-2],[0.975,0.5,1.5,-2.0,-2.0,0.275253,-2.0,-2.0,0.261,0.6275,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005097,0.028737,0.078123,0.530908,1.167909,-0.030934,0.088767,-0.802479,-0.061338,0.020768,0.940288,-0.243608,-0.104243,-0.939625,-0.263246]],[[34,59,45,-2,-2,4,-2,-2,2,10,-2,-2,1,-2,-2],[0.129167,1.5,0.5,-2.0,-2.0,0.0765,-2.0,-2.0,0.935,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004863,0.062573,0.013593,0.774601,-0.208639,0.21624,1.454603,0.653931,-0.034833,0.05715,0.677723,-0.658224,-0.071029,-0.444808,1.476317]],[[12,5,3,-2,-2,34,-2,-2,4,-2,-2],[0.5,0.0455,0.2455,-2.0,-2.0,0.253571,-2.0,-2.0,0.737,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001458,0.009032,0.069871,0.811855,-0.38634,-0.021388,0.060857,-0.681487,-0.181893,-1.404191,-0.348124]],[[2,11,14,-2,-2,-2,10,1,-2,-2,-2],[0.96,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000245,0.042637,0.058026,0.454421,-1.012451,-1.164001,-0.056924,-0.070923,-0.510756,1.391944,0.57959]],[[14,11,45,-2,-2,5,-2,-2,6,2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.1425,-2.0,-2.0,0.9885,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.006592,0.006289,0.017726,0.644208,-0.06442,-0.17202,-0.138658,-1.564174,-0.159934,-0.186628,-1.492853,-1.190911,-0.583324]],[[2,7,5,-2,-2,6,-2,-2,25,23,-2,-2,6,-2,-2],[0.935,0.0015,0.0565,-2.0,-2.0,0.1185,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.675,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008471,0.056169,0.100092,0.985569,1.220604,0.033832,1.122048,0.144514,-0.038208,-0.053954,-0.407896,0.588838,0.135823,1.277049,0.149953]],[[43,1,28,-2,-2,2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.475,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004116,-0.004886,-0.025186,-0.202277,1.257522,0.077063,0.730552,1.410177,1.089174]],[[34,10,14,-2,-2,5,-2,-2,5,3,-2,-2,2,-2,-2],[0.458042,0.5,0.5,-2.0,-2.0,0.2065,-2.0,-2.0,0.034,0.0055,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000804,0.010631,-0.004127,0.039846,-1.085708,0.125955,0.313996,1.097948,-0.134062,0.009991,0.900774,-0.79283,-0.187864,-1.467461,-1.136547]],[[14,5,12,-2,-2,2,-2,-2,34,-2,-2],[0.5,0.6755,0.5,-2.0,-2.0,0.985,-2.0,-2.0,0.309524,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001593,0.010487,0.03738,0.318451,-0.807216,-0.053683,0.187608,-0.938194,-0.163651,-1.366858,-0.977676]],[[40,45,31,-2,-2,59,-2,-2,7,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.0245,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.0085,-0.003053,0.055226,0.650054,-0.894588,-0.034296,-0.473403,0.273046,0.182299,1.130906,0.293475]],[[34,59,2,-2,-2,4,-2,-2,5,12,-2,-2,1,-2,-2],[0.130769,1.5,0.985,-2.0,-2.0,0.066,-2.0,-2.0,0.0685,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003375,0.047896,0.008453,0.297156,-0.541204,0.17342,1.520601,0.272663,-0.040366,0.037684,0.358358,-1.416707,-0.090299,-0.651305,0.662514]],[[14,28,3,-2,-2,-2,7,-2,-2],[0.5,0.5,0.1145,-2.0,-2.0,-2.0,0.0215,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.007883,0.019529,0.012571,0.336439,-0.182701,1.412471,-0.128172,-0.722988,-1.277191]],[[1,34,10,-2,-2,5,-2,-2,38,3,-2,-2,-2],[0.5,0.377155,0.5,-2.0,-2.0,0.0065,-2.0,-2.0,1.5,0.4815,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.005304,-0.014381,0.004379,-0.058664,0.698814,-0.113062,0.739538,-1.024145,0.088588,0.079707,0.951689,1.05493,1.419296]],[[3,5,31,-2,-2,6,-2,-2,53,34,-2,-2,-2],[0.0265,0.686,0.5,-2.0,-2.0,0.8665,-2.0,-2.0,0.5,0.129167,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000326,0.042744,0.070259,0.593617,-0.711728,-0.124035,0.003067,-1.334039,-0.032748,-0.04627,0.009488,-0.536744,0.95624]],[[12,14,2,-2,-2,34,-2,-2,34,-2,-2],[0.5,0.5,0.975,-2.0,-2.0,0.309524,-2.0,-2.0,0.261364,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004382,0.006041,0.018491,0.332154,-0.250467,-0.139095,-1.320394,-0.543036,-0.188655,-1.525628,-0.208636]],[[14,34,53,-2,-2,7,-2,-2,34,2,-2,-2,-2],[0.5,0.20943,0.5,-2.0,-2.0,0.0045,-2.0,-2.0,0.309524,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000377,0.011106,0.040706,0.152797,1.30544,-0.034788,0.337099,-0.542193,-0.140352,-0.169407,-1.417475,-1.167789,-0.889421]],[[34,10,53,-2,-2,5,-2,-2,4,5,-2,-2,-2],[0.468627,0.5,0.5,-2.0,-2.0,0.2045,-2.0,-2.0,0.8895,0.0375,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001698,0.011495,-0.002452,-0.095483,0.737586,0.124703,-0.01266,1.266147,-0.130581,-0.098878,0.711583,-1.15436,-1.643406]],[[34,14,5,-2,-2,59,-2,-2,5,3,-2,-2,2,-2,-2],[0.458042,0.5,0.6545,-2.0,-2.0,0.5,-2.0,-2.0,0.033,0.0045,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00507,0.005057,0.01549,0.247717,-0.21739,-0.137368,-1.374828,-0.667633,-0.12479,-0.00333,0.798964,-0.831494,-0.172989,-1.537124,-1.146489]],[[3,5,31,-2,-2,6,-2,-2,11,23,-2,-2,-2],[0.1145,0.553,0.5,-2.0,-2.0,0.8665,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000193,0.039146,0.070494,0.615419,-0.903462,-0.10664,-0.014883,-1.150397,-0.042723,-0.030676,-0.28417,0.683512,-1.858236]],[[12,3,5,-2,-2,53,-2,-2,34,-2,-2],[0.5,0.1145,0.5525,-2.0,-2.0,0.5,-2.0,-2.0,0.292857,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00138,0.008274,0.045664,0.570571,-0.522971,-0.034379,-0.346321,1.065466,-0.18602,-1.356694,-0.41667]],[[25,36,2,-2,-2,6,-2,-2,34,3,-2,-2,-2],[0.5,0.5,0.985,-2.0,-2.0,0.02,-2.0,-2.0,0.223611,0.0205,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001368,-0.014103,0.020852,0.508156,-0.436347,-0.059219,1.433729,-0.390822,0.134401,0.181971,0.976058,1.412234,-0.059704]],[[14,5,12,-2,-2,2,-2,-2,34,2,-2,-2,-2],[0.5,0.9835,0.5,-2.0,-2.0,0.985,-2.0,-2.0,0.309524,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.007466,0.003907,0.01814,0.188887,-0.888225,-0.099224,-0.09361,-1.109128,-0.134374,-0.162893,-1.367203,-1.157114,-0.905408]],[[12,34,58,-2,-2,3,-2,-2,4,-2,-2],[0.5,0.275253,0.5,-2.0,-2.0,0.0125,-2.0,-2.0,0.9085,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005208,0.014623,0.040748,0.206894,1.21903,-0.051571,0.162953,-0.715601,-0.156901,-1.310794,-0.037451]],[[34,2,7,-2,-2,5,-2,-2,2,5,-2,-2,0,-2,-2],[0.130769,0.985,0.84,-2.0,-2.0,0.9855,-2.0,-2.0,0.935,0.453,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003413,0.040651,0.075285,0.349553,1.636726,-0.041741,-0.022553,-1.229016,-0.034088,0.050519,0.785955,-0.495366,-0.065109,-0.44885,1.014606]],[[34,40,55,-2,-2,9,-2,-2,3,-2,4,-2,-2],[0.458042,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.0015,-2.0,0.8855,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.005473,0.014776,0.004498,-0.03749,0.73409,0.17129,1.292734,0.213505,-0.109704,0.829957,-0.13445,-0.904538,-1.373927]],[[14,34,2,-2,-2,36,-2,-2,34,2,-2,-2,-2],[0.5,0.27682,0.985,-2.0,-2.0,0.5,-2.0,-2.0,0.309524,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003149,0.014017,0.036571,0.392744,-0.21754,-0.043714,0.042833,-0.657206,-0.137651,-0.159589,-1.364907,-1.154208,-0.991767]],[[54,1,4,-2,-2,2,-2,-2,-2],[0.5,0.5,0.0585,-2.0,-2.0,0.475,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003747,-0.010621,-0.028303,0.073639,-0.351177,0.057967,0.599716,1.465777,0.82234]],[[3,5,59,-2,-2,7,-2,-2,23,10,-2,-2,-2],[0.1145,0.685,2.5,-2.0,-2.0,0.1385,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.008058,0.025892,0.05334,0.297469,1.180231,-0.109002,-1.593527,-0.323252,-0.045393,-0.064523,-0.549758,0.615425,0.857582]],[[11,2,31,-2,-2,10,-2,-2,3,-2,-2],[0.5,0.935,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.095,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00396,0.004534,0.050041,0.571708,-0.888703,-0.017895,-0.211654,0.59411,-0.149114,-0.366172,-1.879806]],[[1,34,30,-2,-2,11,-2,-2,2,37,-2,-2,-2],[0.5,0.129167,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.475,0.4455,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.011668,-0.027784,0.019512,-0.182984,0.63115,-0.057126,-0.289142,-1.491635,0.059241,0.047786,1.034729,-0.192635,1.386826]],[[46,43,11,-2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.1335,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.007623,0.000209,-0.007212,0.014777,-0.987821,0.960824,-0.143712,-0.305213,-1.272994]],[[3,5,59,-2,-2,6,-2,-2,53,7,-2,-2,-2],[0.0925,0.549,2.5,-2.0,-2.0,0.982,-2.0,-2.0,0.5,0.235,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000716,0.032228,0.057235,0.353125,1.119388,-0.078658,-0.01431,-1.387335,-0.037123,-0.050663,-0.602558,0.026183,0.745023]],[[34,59,6,-2,-2,4,-2,-2,7,2,-2,-2,4,-2,-2],[0.130769,1.5,0.3095,-2.0,-2.0,0.0765,-2.0,-2.0,0.0045,0.985,-2.0,-2.0,0.5255,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004807,0.044284,0.007688,-0.564821,0.24711,0.152606,1.488347,0.087887,-0.022294,0.032086,-0.012558,1.358551,-0.055608,-0.137851,-0.809376]],[[54,3,2,-2,-2,14,-2,-2,-2],[0.5,0.0015,0.985,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.009001,0.001182,0.07906,1.041228,-0.170266,-0.014952,-0.031305,-1.263192,1.133783]],[[34,59,2,-2,-2,4,-2,-2,5,3,-2,-2,1,-2,-2],[0.130769,1.5,0.985,-2.0,-2.0,0.0765,-2.0,-2.0,0.1375,0.072,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007278,0.046881,0.016285,0.394504,-0.516299,0.150105,1.628841,0.202165,-0.020453,0.029023,0.432379,-0.27805,-0.067008,-0.539666,0.796336]],[[25,1,40,-2,-2,2,-2,-2,6,4,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.475,-2.0,-2.0,0.97,0.0945,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000315,-0.009226,-0.026372,-0.248917,0.408874,0.062019,0.770483,1.302691,0.095011,0.152546,1.222623,0.515217,-0.624199]],[[45,38,2,-2,-2,37,-2,-2,38,59,-2,-2,3,-2,-2],[0.5,1.5,0.985,-2.0,-2.0,0.023,-2.0,-2.0,2.5,2.5,-2.0,-2.0,0.013,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005959,0.040983,0.027352,0.461896,-0.917637,0.171821,1.038444,1.490565,-0.027983,-0.019902,-0.24928,0.470542,-0.320206,-2.19147,-0.961627]],[[12,34,43,-2,-2,4,-2,-2,34,-2,-2],[0.5,0.458042,0.5,-2.0,-2.0,0.905,-2.0,-2.0,0.261364,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001393,0.00907,0.018988,0.090036,0.750901,-0.11059,-0.627346,-1.548716,-0.139594,-1.299688,-0.112355]],[[5,34,2,-2,-2,36,-2,-2,4,2,-2,-2,2,-2,-2],[0.6755,0.27682,0.995,-2.0,-2.0,0.5,-2.0,-2.0,0.439,0.975,-2.0,-2.0,0.985,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007315,0.030258,0.05682,0.314376,1.223273,-0.030976,0.694834,-0.64547,-0.044698,-0.011893,0.677271,-0.611034,-0.15721,-1.22565,-0.602131]],[[36,6,2,-2,-2,3,-2,-2,6,-2,38,-2,-2],[0.5,0.9785,0.985,-2.0,-2.0,0.2245,-2.0,-2.0,0.0285,-2.0,2.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004548,0.022743,-0.004232,0.26488,-0.552038,0.094295,0.445329,2.270443,-0.038493,1.291091,-0.053838,-0.256995,-1.852218]],[[53,57,45,-2,-2,-2,4,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.171,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.002411,-0.005316,-0.011284,0.406215,-0.230373,1.010174,0.13768,1.437921,-0.276248]],[[12,54,28,-2,-2,-2,4,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9075,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.005436,0.002147,-0.004587,-0.072344,1.023682,0.809801,-0.155081,-1.443834,-0.074193]],[[34,14,12,-2,-2,7,-2,-2,5,4,-2,-2,1,-2,-2],[0.33908,0.5,0.5,-2.0,-2.0,0.011,-2.0,-2.0,0.0335,0.144,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007034,0.007477,0.016936,0.168876,-0.735211,-0.116597,-0.453281,-1.325191,-0.079511,0.040829,0.934805,-0.338114,-0.130115,-1.218061,0.668889]],[[60,11,2,-2,-2,7,-2,-2,-2],[0.5,0.5,0.935,-2.0,-2.0,0.0115,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000453,0.005112,0.013196,0.657364,-0.063474,-0.118192,0.02538,-1.479882,-1.294314]],[[12,34,2,-2,-2,2,-2,-2,34,-2,-2],[0.5,0.027778,0.985,-2.0,-2.0,0.925,-2.0,-2.0,0.267857,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001548,0.00608,0.057752,0.744909,-0.303714,-0.016335,0.460405,-0.270115,-0.136025,-1.29926,-0.119418]],[[14,3,34,-2,-2,-2,7,-2,2,-2,-2],[0.5,0.9965,0.33908,-2.0,-2.0,-2.0,0.011,-2.0,0.975,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.004133,0.011964,0.015754,0.188836,-0.381161,-1.448654,-0.097127,-0.477125,-0.130978,-1.343685,-1.126672]],[[57,2,7,-2,-2,38,-2,-2,-2],[0.5,0.975,0.455,-2.0,-2.0,1.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.002489,-0.002874,0.01959,0.010606,0.723734,-0.041275,-0.455008,0.765006,0.94274]],[[34,42,4,-2,-2,-2,1,4,-2,-2,38,-2,-2],[0.27682,0.5,0.0375,-2.0,-2.0,-2.0,0.5,0.9525,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001867,0.023136,0.031098,0.550948,0.050475,-0.735981,-0.050895,-0.068402,-0.392924,-1.349685,0.095661,0.950142,1.387379]],[[11,2,9,-2,-2,10,-2,-2,7,-2,-2],[0.5,0.935,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.0115,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001177,0.008969,0.05619,0.827211,-0.863081,-0.012055,-0.167737,0.63021,-0.11014,0.206199,-1.566712]],[[25,1,35,-2,-2,38,-2,-2,6,54,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.9695,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000808,-0.009874,-0.026631,-0.270434,0.418567,0.056711,0.75027,1.324283,0.107599,0.14592,0.673907,1.303198,-0.236683]],[[43,3,6,-2,-2,23,-2,-2,-2],[0.5,0.0645,0.1205,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00485,-0.001123,0.029195,1.098638,0.107828,-0.032102,-0.336054,0.929009,0.735075]],[[34,10,53,-2,-2,5,-2,-2,5,-2,4,-2,-2],[0.458042,0.5,0.5,-2.0,-2.0,0.208,-2.0,-2.0,0.0055,-2.0,0.5165,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003206,0.010249,-0.000193,-0.063732,0.552105,0.092078,0.094849,1.045533,-0.085878,0.66794,-0.109292,-0.734666,-1.409321]],[[12,34,53,-2,-2,5,-2,-2,7,-2,-2],[0.5,0.269563,0.5,-2.0,-2.0,0.0325,-2.0,-2.0,0.017,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00276,0.004406,0.025655,0.10152,0.990623,-0.048105,0.192376,-0.67635,-0.130899,-1.292529,-0.439429]],[[5,59,3,-2,-2,-2,34,59,-2,-2,1,-2,-2],[0.0685,2.5,0.071,-2.0,-2.0,-2.0,0.129167,1.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.004667,0.041304,0.025224,0.570973,-0.473251,0.950787,-0.032201,0.011724,-0.200063,0.622136,-0.066457,-0.581837,1.02158]],[[3,34,2,-2,-2,36,-2,-2,14,28,-2,-2,2,-2,-2],[0.0015,0.11678,0.45,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000478,0.057782,0.127284,1.059923,1.282735,0.020213,0.962273,-0.594412,-0.012241,-0.002206,-0.056935,0.960324,-0.124839,-1.349444,-1.12816]],[[3,5,31,-2,-2,7,-2,-2,36,6,-2,-2,6,-2,-2],[0.0925,0.5845,0.5,-2.0,-2.0,0.1305,-2.0,-2.0,0.5,0.9835,-2.0,-2.0,0.039,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004145,0.02678,0.051785,0.503208,-0.78531,-0.091528,-1.294873,-0.35507,-0.039119,-0.001113,-0.245914,1.619085,-0.089417,1.090569,-0.614056]],[[43,53,3,-2,-2,6,-2,-2,-2],[0.5,0.5,0.1145,-2.0,-2.0,0.699,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003322,-0.002474,-0.008688,0.161997,-0.291162,0.099728,1.04069,-0.058065,0.689037]],[[55,1,4,-2,-2,2,-2,-2,34,-2,-2],[0.5,0.5,0.0355,-2.0,-2.0,0.475,-2.0,-2.0,0.129167,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004808,-0.012624,-0.026436,0.127425,-0.339322,0.04165,0.540696,1.318009,0.105605,1.283353,0.107672]],[[11,36,2,-2,-2,6,-2,-2,3,-2,-2],[0.5,0.5,0.985,-2.0,-2.0,0.0095,-2.0,-2.0,0.0915,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000825,0.008036,0.033597,0.69818,-0.335794,-0.026842,1.844472,-0.226699,-0.118701,-0.244231,-1.825006]],[[57,1,4,-2,-2,38,-2,-2,-2],[0.5,0.5,0.3515,-2.0,-2.0,1.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001672,-0.006481,-0.020366,0.033589,-0.41427,0.050571,0.723208,1.317208,0.758281]],[[2,32,53,-2,-2,-2,30,38,-2,-2,34,-2,-2],[0.975,0.5,0.5,-2.0,-2.0,-2.0,1.5,1.5,-2.0,-2.0,0.11678,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003796,0.018636,0.01001,0.012769,0.81355,1.524863,-0.04144,-0.069712,-0.71855,0.94127,0.021297,1.095522,-0.344025]],[[14,53,5,-2,-2,-2,34,2,-2,-2,-2],[0.5,0.5,0.9835,-2.0,-2.0,-2.0,0.309524,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.005895,0.001654,-0.004652,0.048182,-0.692342,0.593853,-0.09556,-0.137694,-1.421235,-1.127146,-0.25865]],[[34,12,35,-2,-2,30,-2,-2,5,-2,37,-2,-2],[0.396429,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.0045,-2.0,0.0005,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000826,0.012986,0.020512,0.092874,0.635472,-0.105077,-1.548193,-0.02709,-0.069106,0.979409,-0.093245,-0.945264,1.000479]],[[12,3,5,-2,-2,53,-2,-2,4,-2,-2],[0.5,0.1145,0.5495,-2.0,-2.0,0.5,-2.0,-2.0,0.8585,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002297,0.005232,0.032003,0.448874,-0.383083,-0.026526,-0.319333,0.865559,-0.144849,-1.536339,0.011301]],[[2,59,12,-2,-2,4,-2,-2,10,0,-2,-2,-2],[0.96,2.5,0.5,-2.0,-2.0,0.113,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00198,0.027186,0.013502,0.193648,-1.097721,0.143365,1.409596,0.305496,-0.03164,-0.041341,-0.381956,1.098678,0.524663]],[[12,34,53,-2,-2,5,-2,-2,34,-2,-2],[0.5,0.449495,0.5,-2.0,-2.0,0.0055,-2.0,-2.0,0.267857,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001346,0.006491,0.015514,0.0547,0.748611,-0.094507,1.013966,-1.110356,-0.139955,-1.38759,-0.220972]],[[31,12,11,-2,-2,2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.96,-2.0,-2.0,0.0065,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.005628,-0.000431,0.006609,0.108318,-0.746294,-0.11722,-1.4005,-0.133125,-0.176449,1.127087,-1.509816]],[[4,4,59,-2,-2,-2,1,5,-2,-2,3,-2,-2],[0.0585,0.0515,2.5,-2.0,-2.0,-2.0,0.5,0.075,-2.0,-2.0,0.253,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003125,0.031994,0.021208,-0.0125,0.982338,1.465797,-0.025184,-0.050011,0.009972,-0.506069,0.044751,1.068053,-0.431394]],[[14,11,5,-2,-2,7,-2,-2,34,2,-2,-2,-2],[0.5,0.5,0.9895,-2.0,-2.0,0.015,-2.0,-2.0,0.354167,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000299,0.007889,0.015478,0.193253,-0.541386,-0.100775,0.131688,-1.508925,-0.095348,-0.129543,-1.338237,-1.120832,0.010752]],[[28,3,6,-2,-2,14,-2,-2,-2],[0.5,0.0015,0.3365,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001913,-0.002724,0.048552,1.345798,0.227834,-0.013488,-0.037259,-1.235337,0.882622]],[[28,46,43,-2,-2,30,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003623,-0.001469,0.004965,-0.018042,0.850355,-0.121813,-0.251698,-1.060354,0.908314]],[[3,6,34,-2,-2,5,-2,-2,53,5,-2,-2,-2],[0.1145,0.6295,0.279221,-2.0,-2.0,0.549,-2.0,-2.0,0.5,0.2095,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000125,0.028109,0.092699,1.076536,0.005597,0.00148,0.242677,-0.712306,-0.033005,-0.04641,-0.732769,-0.11486,0.790665]],[[5,14,60,-2,-2,34,-2,-2,2,-2,30,-2,-2],[0.9835,0.5,0.5,-2.0,-2.0,0.309524,-2.0,-2.0,0.985,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.007853,0.018436,0.0273,0.23729,-0.996777,-0.089651,-1.263266,-0.117054,-0.072594,-0.010106,-0.108923,-0.622383,-1.201186]],[[11,12,5,-2,-2,2,-2,-2,5,-2,-2],[0.5,0.5,0.685,-2.0,-2.0,0.96,-2.0,-2.0,0.1435,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.007462,-0.000722,0.005919,0.198406,-0.280301,-0.108829,-1.394763,-0.052661,-0.128483,-0.098511,-1.526949]],[[3,6,2,-2,-2,4,-2,-2,55,28,-2,-2,7,-2,-2],[0.0015,0.928,0.96,-2.0,-2.0,0.9,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.027,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008274,0.050512,0.11906,0.667855,1.389939,0.002529,-0.968057,0.754579,-0.0205,-0.028238,-0.253175,0.705201,0.09678,-0.055561,1.09123]],[[34,36,2,-2,-2,6,-2,-2,5,-2,7,-2,-2],[0.452273,0.5,0.995,-2.0,-2.0,0.025,-2.0,-2.0,0.0055,-2.0,0.0075,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.009978,0.018105,0.042983,0.669005,-0.446684,-0.014278,1.589093,-0.17755,-0.079559,0.958095,-0.103236,-0.343838,-1.15736]],[[34,40,25,-2,-2,5,-2,-2,2,-2,-2],[0.585714,0.5,0.5,-2.0,-2.0,0.1795,-2.0,-2.0,0.975,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.010042,0.013799,0.006896,-0.0214,0.671454,0.108705,-0.190891,0.93188,-0.131361,-1.235598,-1.153077]],[[46,28,43,-2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.0695,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.000771,0.007926,0.003304,-0.018369,0.609042,1.118468,-0.12207,0.0231,-1.09765]],[[34,40,55,-2,-2,5,-2,-2,5,3,-2,-2,2,-2,-2],[0.458042,0.5,0.5,-2.0,-2.0,0.3365,-2.0,-2.0,0.034,0.007,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002306,0.008549,0.001163,-0.048309,0.638348,0.110187,-0.021732,1.21818,-0.076636,0.032631,0.85149,-0.538975,-0.113252,-1.284847,-1.002666]],[[46,12,54,-2,-2,34,-2,-2,4,-2,34,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.261364,-2.0,-2.0,0.0515,-2.0,0.190909,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.00211,0.00446,0.010172,0.034769,0.84685,-0.109819,-1.446419,0.35667,-0.107334,0.198109,-0.166776,-1.366575,-0.547674]],[[60,3,7,-2,-2,14,-2,-2,-2],[0.5,0.0015,0.8875,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001266,0.001774,0.053893,0.739748,-0.659708,-0.009143,-0.003369,-1.203524,-1.182546]],[[54,4,23,-2,-2,7,-2,-2,-2],[0.5,0.1435,0.5,-2.0,-2.0,0.0075,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.006195,-0.011192,0.015773,-0.003968,1.10857,-0.034215,0.228746,-0.666107,0.856036]],[[4,59,3,-2,-2,-2,5,12,-2,-2,36,-2,-2],[0.0345,2.5,0.847,-2.0,-2.0,-2.0,0.0285,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000642,0.041512,0.018178,0.540834,-0.291082,1.082523,-0.019952,0.035537,0.4981,-1.169342,-0.04141,-0.094716,-0.482434]],[[55,36,6,-2,-2,6,-2,-2,6,-2,-2],[0.5,0.5,0.9775,-2.0,-2.0,0.017,-2.0,-2.0,0.675,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.005696,-0.013292,0.009403,-0.190842,0.917749,-0.045211,1.659477,-0.355661,0.092781,1.182988,0.10385]],[[2,38,59,-2,-2,-2,0,1,-2,-2,3,-2,-2],[0.96,1.5,2.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.0015,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000496,0.019196,0.026946,0.100689,1.141577,-1.219759,-0.027504,-0.033693,-0.297946,1.270931,0.12951,1.143706,0.998483]],[[57,3,5,-2,-2,5,-2,-2,-2],[0.5,0.1145,0.5845,-2.0,-2.0,0.235,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000684,-0.004679,0.019294,0.319928,-0.424323,-0.032592,-0.658085,0.005312,0.99846]],[[2,12,7,-2,-2,-2,10,0,-2,-2,-2],[0.96,0.5,0.187,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.004674,0.018192,0.024289,0.076006,0.564764,-0.806936,-0.035718,-0.046637,-0.459598,1.185961,0.641861]],[[34,7,3,-2,-2,10,-2,-2,7,2,-2,-2,2,-2,-2],[0.452273,0.0645,0.821,-2.0,-2.0,0.5,-2.0,-2.0,0.0075,0.925,-2.0,-2.0,0.975,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005448,0.013507,-0.008764,0.061696,-0.676988,0.045342,0.22803,1.273156,-0.084741,-0.010425,1.096884,-0.590372,-0.135186,-1.535112,-0.817539]],[[45,3,6,-2,-2,-2,38,55,-2,-2,3,-2,-2],[0.5,0.9385,0.1755,-2.0,-2.0,-2.0,2.5,0.5,-2.0,-2.0,0.0065,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.009961,0.030763,0.03988,1.275886,0.327941,-1.217982,-0.029307,-0.022177,-0.221575,0.488867,-0.272699,-2.13902,-0.739832]],[[34,59,2,-2,-2,4,-2,-2,5,7,-2,-2,4,-2,-2],[0.129167,1.5,0.985,-2.0,-2.0,0.0765,-2.0,-2.0,0.0065,0.0025,-2.0,-2.0,0.442,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001414,0.031074,-0.007877,0.191938,-0.697744,0.148825,1.429028,0.228483,-0.019446,0.056022,0.120004,1.496011,-0.032254,-0.04813,-0.54682]],[[60,11,14,-2,-2,4,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.7265,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005059,0.007947,0.013136,0.156631,-0.885365,-0.092986,-1.709872,-0.006296,-0.978194]],[[3,4,34,-2,-2,7,-2,-2,-2],[0.9965,0.0765,0.279221,-2.0,-2.0,0.0075,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000975,0.00385,0.029782,0.399378,-0.355561,-0.01349,0.189581,-0.321252,-1.420381]],[[58,43,1,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.001696,-0.003679,-0.008547,-0.152175,0.693433,0.569071,0.7322]],[[23,5,4,-2,-2,3,-2,-2,3,-2,-2],[0.5,0.9895,0.0075,-2.0,-2.0,0.9945,-2.0,-2.0,0.033,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003994,-0.001989,0.005409,0.522169,-0.038305,-0.081534,-0.898641,0.170749,0.076474,-0.285439,1.068928]],[[36,2,59,-2,-2,6,-2,-2,6,-2,38,-2,-2],[0.5,0.985,0.5,-2.0,-2.0,0.9955,-2.0,-2.0,0.0295,-2.0,2.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.002901,0.017389,0.047395,0.308859,0.805853,-0.047128,-0.635227,1.584359,-0.029056,1.38465,-0.044255,-0.218286,-1.919765]],[[60,3,6,-2,-2,3,-2,-2,-2],[0.5,0.3485,0.0515,-2.0,-2.0,0.5095,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.008298,-0.004421,0.012143,1.39606,-0.00967,-0.035235,-1.080736,-0.132249,-1.383994]],[[34,59,7,-2,-2,4,-2,-2,5,3,-2,-2,5,-2,-2],[0.113248,1.5,0.6665,-2.0,-2.0,0.0765,-2.0,-2.0,0.1345,0.2845,-2.0,-2.0,0.2335,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004857,0.041972,0.011184,0.338354,-0.492461,0.163734,1.444534,0.391265,-0.015447,0.018582,0.32617,-0.433287,-0.048171,-1.192363,-0.232918]],[[5,6,36,-2,-2,7,-2,-2,15,-2,2,-2,-2],[0.9065,0.026,0.5,-2.0,-2.0,0.235,-2.0,-2.0,0.5,-2.0,0.985,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001229,0.013062,0.126245,0.799041,1.312275,0.005528,-0.064746,0.491274,-0.047535,-1.322096,-0.029392,0.262404,-0.78124]],[[11,12,5,-2,-2,6,-2,-2,5,-2,-2],[0.5,0.5,0.0695,-2.0,-2.0,0.9485,-2.0,-2.0,0.1435,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.009217,-0.003603,0.002252,0.310678,-0.132775,-0.099204,-1.40264,0.089959,-0.112072,-0.002668,-1.503178]],[[43,36,2,-2,-2,3,-2,-2,-2],[0.5,0.5,0.985,-2.0,-2.0,0.097,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003083,-0.001553,0.020439,0.47578,-0.335331,-0.031758,0.02746,-0.463331,0.577719]],[[60,34,4,-2,-2,2,-2,-2,-2],[0.5,0.565714,0.0575,-2.0,-2.0,0.96,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000603,0.00437,0.008056,0.296871,-0.077997,-0.128909,-1.422734,-1.092927,-1.166386]],[[28,4,4,-2,-2,7,-2,-2,-2],[0.5,0.3515,0.246,-2.0,-2.0,0.0075,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.007881,0.003694,0.023242,0.075792,1.129703,-0.023555,0.286093,-0.583759,0.999557]],[[43,2,4,-2,-2,34,-2,-2,-2],[0.5,0.96,0.06,-2.0,-2.0,0.120048,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005881,0.000684,0.022869,0.478111,0.008301,-0.027672,0.192128,-0.525424,0.66653]],[[5,11,36,-2,-2,7,-2,-2,5,3,-2,-2,-2],[0.9925,0.5,0.5,-2.0,-2.0,0.008,-2.0,-2.0,0.9965,0.8655,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002265,0.00821,0.014425,0.405097,-0.067335,-0.087933,0.1602,-1.531425,-0.08036,-0.049479,-0.744191,0.208675,-1.144707]],[[34,59,6,-2,-2,7,-2,-2,3,5,-2,-2,29,-2,-2],[0.064583,1.5,0.793,-2.0,-2.0,0.106,-2.0,-2.0,0.097,0.5495,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.008202,0.047665,0.023207,-0.20376,0.63467,0.154133,0.189957,1.637191,-0.008885,0.019828,0.379796,-0.643971,-0.041639,-0.383806,0.588744]],[[4,59,5,-2,-2,34,-2,-2,4,3,-2,-2,4,-2,-2],[0.0345,1.5,0.856,-2.0,-2.0,0.174242,-2.0,-2.0,0.0465,0.0175,-2.0,-2.0,0.0585,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002179,0.03879,0.015083,0.407972,-0.486834,0.11778,1.186862,-0.183196,-0.014176,-0.151239,-2.37291,-0.314516,-0.004574,0.86949,-0.08848]],[[36,11,2,-2,-2,-2,38,23,-2,-2,-2],[0.5,0.5,0.995,-2.0,-2.0,-2.0,2.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.002201,0.018578,0.027217,0.512521,-0.474498,-1.850577,-0.029229,-0.022305,-0.215201,0.633638,-1.810581]],[[34,43,53,-2,-2,-2,5,-2,4,-2,-2],[0.458042,0.5,0.5,-2.0,-2.0,-2.0,0.0045,-2.0,0.5165,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.006744,0.013009,0.00852,0.016992,0.598956,0.599357,-0.067376,0.786507,-0.088183,-0.668488,-1.379182]],[[14,5,11,-2,-2,2,-2,-2,34,59,-2,-2,-2],[0.5,0.9895,0.5,-2.0,-2.0,0.985,-2.0,-2.0,0.309524,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00196,0.004606,0.012252,0.145084,-0.578357,-0.06102,0.015763,-0.942427,-0.081944,-0.104375,-1.254741,-1.111056,-0.752988]],[[55,46,7,-2,-2,59,-2,-2,5,-2,-2],[0.5,0.5,0.0035,-2.0,-2.0,0.5,-2.0,-2.0,0.1755,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004192,-0.01218,-0.003514,0.281356,-0.154531,-0.167701,-0.618752,-1.397375,0.104473,1.04863,0.377624]],[[40,59,55,-2,-2,-2,6,-2,6,-2,-2],[0.5,4.5,0.5,-2.0,-2.0,-2.0,0.719,-2.0,0.9665,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.001459,-0.005593,-0.001046,-0.073353,0.620646,-0.994079,0.083989,-0.527424,0.196783,1.342643,0.456381]],[[5,59,3,-2,-2,-2,36,2,-2,-2,2,-2,-2],[0.0495,2.5,0.7255,-2.0,-2.0,-2.0,0.5,0.995,-2.0,-2.0,0.995,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.000585,0.033748,0.018889,0.374248,-1.130397,0.85015,-0.016581,0.009409,0.327232,-0.775164,-0.05555,-0.479482,0.754416]],[[36,2,34,-2,-2,38,-2,-2,6,-2,34,-2,-2],[0.5,0.995,0.396429,-2.0,-2.0,1.5,-2.0,-2.0,0.0095,-2.0,0.275253,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.005169,0.022303,0.040014,0.5878,-0.573104,-0.039667,-0.903271,1.107998,-0.042203,1.65083,-0.052209,-0.178269,-0.822845]],[[36,2,3,-2,-2,38,-2,-2,6,-2,38,-2,-2],[0.5,0.995,0.9935,-2.0,-2.0,1.5,-2.0,-2.0,0.0095,-2.0,2.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.001976,0.018584,0.036274,0.482244,-1.141682,-0.049227,-1.061079,1.145658,-0.027812,1.710711,-0.038342,-0.188076,-1.897546]],[[30,59,6,-2,-2,4,-2,-2,-2],[4.5,2.5,0.3255,-2.0,-2.0,0.0505,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.008099,0.012274,0.005073,-0.347393,0.166389,0.075627,1.270924,-0.057731,-0.765537]],[[34,53,43,-2,-2,-2,5,3,-2,-2,37,-2,-2],[0.396429,0.5,0.5,-2.0,-2.0,-2.0,0.0065,0.0005,-2.0,-2.0,0.0005,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.005163,0.014689,0.008385,0.030254,0.678042,0.637303,-0.050081,0.100252,0.380909,1.273849,-0.076901,-0.862099,1.249127]],[[34,53,55,-2,-2,-2,36,34,-2,-2,34,-2,-2],[0.27682,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.452273,-2.0,-2.0,0.297059,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.007626,0.020411,0.013522,0.053369,0.668614,0.601678,-0.024979,0.009584,0.525411,-0.804006,-0.07001,-1.319722,-0.242922]],[[5,3,4,-2,-2,3,-2,-2,4,2,-2,-2,-2],[0.6755,0.832,0.0305,-2.0,-2.0,0.9715,-2.0,-2.0,0.8935,0.975,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002737,0.015193,0.029384,0.848877,0.126023,-0.087711,-1.164674,0.033302,-0.044344,-0.027459,0.222766,-0.675905,-1.341712]],[[34,2,5,-2,-2,34,-2,-2,1,4,-2,-2,38,-2,-2],[0.130769,0.985,0.1615,-2.0,-2.0,0.102632,-2.0,-2.0,0.5,0.0585,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004531,0.028241,0.050598,0.046307,0.752632,-0.023287,-0.470673,0.586216,-0.027587,-0.039336,0.007124,-0.496039,0.04316,0.574884,1.3474]],[[46,43,34,-2,-2,-2,4,-2,-2],[0.5,0.5,0.223611,-2.0,-2.0,-2.0,0.0715,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.000893,0.005894,0.001255,0.145641,-0.267948,0.673163,-0.077896,0.115606,-0.895959]],[[57,3,5,-2,-2,23,-2,-2,-2],[0.5,0.1145,0.5495,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002298,-0.005905,0.015293,0.30946,-0.418001,-0.030395,-0.367246,0.740639,0.662564]],[[14,3,4,-2,-2,-2,34,-2,4,-2,-2],[0.5,0.9965,0.3415,-2.0,-2.0,-2.0,0.133929,-2.0,0.6245,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.000675,0.006575,0.009258,0.205975,-0.13898,-1.426236,-0.074142,-1.264815,-0.050912,-1.072202,-0.577603]],[[39,35,3,-2,-2,-2,-2],[0.5,0.5,0.9965,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.001837,0.002475,-0.001619,0.010477,-1.228044,0.532652,-0.608768]],[[46,12,3,-2,-2,5,-2,-2,4,-2,-2],[0.5,0.5,0.3485,-2.0,-2.0,0.045,-2.0,-2.0,0.067,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004053,0.002593,0.008257,0.256688,-0.210041,-0.101962,-1.349255,-0.409408,-0.11559,0.164614,-1.227245]],[[36,2,4,-2,-2,38,-2,-2,38,59,-2,-2,-2],[0.5,0.995,0.439,-2.0,-2.0,1.5,-2.0,-2.0,2.5,2.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00269,0.018902,0.035384,0.622032,0.031881,-0.042781,-0.893217,0.980001,-0.03007,-0.022882,-0.255363,0.390322,-1.681659]],[[28,5,6,-2,-2,4,-2,-2,-2],[0.5,0.6755,0.5685,-2.0,-2.0,0.353,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003922,-9.9e-05,0.01527,0.573167,-0.048904,-0.037134,-0.06745,-1.100317,0.828376]],[[46,4,4,-2,-2,7,-2,-2,4,-2,-2],[0.5,0.3515,0.246,-2.0,-2.0,0.0075,-2.0,-2.0,0.0985,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004101,0.001625,0.019459,0.052454,1.073454,-0.023502,0.335446,-0.671416,-0.112658,-0.231281,-1.092616]],[[11,36,2,-2,-2,6,-2,-2,4,-2,-2],[0.5,0.5,0.995,-2.0,-2.0,0.0285,-2.0,-2.0,0.8735,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.005874,-0.000681,0.017553,0.385492,-0.521202,-0.02572,1.372724,-0.257456,-0.086901,-1.613128,0.140029]],[[4,4,4,-2,-2,6,-2,-2,7,6,-2,-2,1,-2,-2],[0.3515,0.246,0.1945,-2.0,-2.0,0.8755,-2.0,-2.0,0.0055,0.555,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008179,0.008575,-0.003095,0.034674,-1.070842,0.143411,1.837168,0.550431,-0.031271,0.025443,1.507673,0.068216,-0.072152,-0.725738,0.168601]],[[34,4,6,-2,-2,3,-2,-2,36,5,-2,-2,34,-2,-2],[0.27682,0.061,0.9905,-2.0,-2.0,0.832,-2.0,-2.0,0.5,0.592,-2.0,-2.0,0.297059,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001608,0.011965,0.049534,0.234093,1.343332,-0.011025,-0.006366,-1.539567,-0.034837,0.001432,0.703025,-0.531109,-0.082156,-1.242731,-0.387904]],[[59,17,46,-2,-2,-2,4,34,-2,-2,3,-2,-2],[2.5,0.5,0.5,-2.0,-2.0,-2.0,0.051,0.154762,-2.0,-2.0,0.0195,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004988,-0.002404,-0.005757,-0.017914,-0.95854,1.404059,0.060381,0.173758,1.407283,0.625588,-0.020126,-0.824012,0.301369]],[[3,5,5,-2,-2,7,-2,-2,53,5,-2,-2,-2],[0.0925,0.553,0.2915,-2.0,-2.0,0.1305,-2.0,-2.0,0.5,0.0065,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.010317,0.009292,0.025609,0.140825,1.234415,-0.060127,-1.306837,-0.166051,-0.031855,-0.041702,-1.139599,-0.279109,0.618331]],[[11,40,2,-2,-2,5,-2,-2,4,-2,-2],[0.5,0.5,0.925,-2.0,-2.0,0.41,-2.0,-2.0,0.864,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001121,0.005536,-0.001277,0.53648,-0.157295,0.095314,-0.024418,1.246515,-0.112985,-1.673895,-0.361663]],[[11,28,14,-2,-2,-2,4,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9105,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.005438,0.010971,0.006823,0.103939,-0.914212,0.832802,-0.080915,-1.549726,0.492781]],[[34,4,6,-2,-2,36,-2,-2,3,2,-2,-2,5,-2,-2],[0.475294,0.0345,0.9875,-2.0,-2.0,0.5,-2.0,-2.0,0.032,0.425,-2.0,-2.0,0.2415,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001722,0.007189,0.033579,0.10649,1.231449,-0.003848,0.265746,-0.234554,-0.067552,-0.010464,1.104302,-0.473303,-0.095408,-1.292204,-0.984249]],[[31,5,3,-2,-2,3,-2,-2,1,-2,-2],[0.5,0.6755,0.8675,-2.0,-2.0,0.9165,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004962,0.009101,0.022697,0.312914,-0.462851,-0.023201,-0.417666,0.316401,-0.12448,-1.238652,1.268481]],[[28,23,3,-2,-2,34,-2,-2,-2],[0.5,0.5,0.2485,-2.0,-2.0,0.236111,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004521,0.00072,-0.00536,0.157764,-0.338188,0.066105,0.828643,-0.978765,0.87145]],[[3,5,2,-2,-2,7,-2,-2,53,7,-2,-2,-2],[0.1145,0.5495,0.975,-2.0,-2.0,0.5595,-2.0,-2.0,0.5,0.0135,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003456,0.020769,0.037594,0.22136,0.707867,-0.050396,-1.199266,-0.055003,-0.016729,-0.027217,-0.681723,0.02549,0.642574]],[[14,11,3,-2,-2,4,-2,-2,7,-2,2,-2,-2],[0.5,0.5,0.9965,-2.0,-2.0,0.8745,-2.0,-2.0,0.011,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000621,0.006761,0.011671,0.118402,-1.247381,-0.073123,-1.630667,0.218911,-0.067804,-0.347891,-0.09549,-1.377105,-1.079713]],[[3,4,59,-2,-2,4,-2,-2,34,-2,-2],[0.9955,0.0345,2.5,-2.0,-2.0,0.0465,-2.0,-2.0,0.236111,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005108,0.008446,0.042207,0.179317,1.016444,-0.006069,-1.216533,0.03528,-0.125098,-1.21677,-0.986214]],[[36,2,30,-2,-2,6,-2,-2,39,6,-2,-2,-2],[0.5,0.985,3.5,-2.0,-2.0,0.9955,-2.0,-2.0,0.5,0.0285,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002393,0.016978,0.039199,0.360254,1.701085,-0.029981,-0.549666,1.599042,-0.027625,-0.014932,1.039472,-0.156057,-1.129997]],[[30,34,23,-2,-2,5,-2,-2,-2],[4.5,0.129167,0.5,-2.0,-2.0,0.1375,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002742,0.000292,0.024448,0.115154,1.008282,-0.01686,0.102559,-0.376116,-0.645416]],[[43,36,2,-2,-2,6,-2,-2,-2],[0.5,0.5,0.995,-2.0,-2.0,0.01,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00445,-0.00021,0.019798,0.406986,-0.411764,-0.028589,1.564247,-0.252794,0.611151]],[[46,43,36,-2,-2,-2,59,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.001478,0.002724,-0.001399,0.167021,-0.167508,0.576139,-0.078561,0.325942,-0.906055]],[[34,38,59,-2,-2,45,-2,-2,5,-2,5,-2,-2],[0.458042,2.5,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.0185,-2.0,0.0775,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002544,0.008724,0.011812,0.028124,0.463915,-0.092306,0.939,-1.631349,-0.077871,0.103657,-0.101515,-1.477668,-0.973377]],[[3,4,6,-2,-2,4,-2,-2,-2],[0.9955,0.0345,0.9875,-2.0,-2.0,0.0435,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001565,0.004374,0.040296,0.215535,0.991436,-0.011176,-0.992591,-0.040407,-1.222396]],[[28,34,59,-2,-2,7,-2,-2,-2],[0.5,0.027778,1.5,-2.0,-2.0,0.0045,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000872,-0.0031,0.03158,0.064253,0.918,-0.018112,0.168044,-0.366748,0.977815]],[[60,34,3,-2,-2,2,-2,-2,-2],[0.5,0.550505,0.1205,-2.0,-2.0,0.975,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001321,0.004584,0.007792,0.218932,-0.080389,-0.09557,-1.32821,-1.11185,-1.077928]],[[6,7,34,-2,-2,5,-2,-2,7,5,-2,-2,5,-2,-2],[0.9535,0.0665,0.160256,-2.0,-2.0,0.3105,-2.0,-2.0,0.3775,0.6065,-2.0,-2.0,0.9795,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002543,-0.018554,-0.045338,-0.8072,-4.2e-05,0.006103,0.83776,-0.20357,0.023283,0.006357,0.233435,-1.004976,0.102196,1.502168,-0.652414]],[[60,59,12,-2,-2,34,-2,-2,-2],[0.5,2.5,0.5,-2.0,-2.0,0.253571,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.006537,0.009633,0.002541,0.067404,-0.651806,0.067255,0.714858,-0.549236,-0.987155]],[[3,6,2,-2,-2,2,-2,-2,14,34,-2,-2,2,-2,-2],[0.0015,0.9275,0.96,-2.0,-2.0,0.96,-2.0,-2.0,0.5,0.422177,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004359,0.043846,0.097027,0.80823,1.179419,0.009845,1.207474,-0.89047,-0.004064,0.00225,0.066651,-0.562167,-0.076682,-1.180279,-1.107336]],[[3,5,31,-2,-2,7,-2,-2,5,2,-2,-2,2,-2,-2],[0.1135,0.549,0.5,-2.0,-2.0,0.5595,-2.0,-2.0,0.2135,0.925,-2.0,-2.0,0.975,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001509,0.019792,0.034169,0.415843,-0.895144,-0.043895,-1.116113,0.026118,-0.020284,-0.081883,0.471576,-0.918203,0.006796,0.345002,-0.272128]],[[1,38,59,-2,-2,59,-2,-2,2,3,-2,-2,5,-2,-2],[0.5,2.5,2.5,-2.0,-2.0,1.5,-2.0,-2.0,0.475,0.2645,-2.0,-2.0,0.0205,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003751,-0.004098,0.000946,-0.075408,0.412141,-0.169019,-0.358749,-1.596875,0.03954,0.031368,0.704464,0.899034,0.149274,1.134038,1.438986]],[[40,53,3,-2,-2,-2,7,-2,-2],[0.5,0.5,0.2455,-2.0,-2.0,-2.0,0.0245,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.004278,-0.001443,-0.005663,0.125638,-0.29924,0.421895,0.079147,0.824561,-0.114184]],[[40,30,59,-2,-2,-2,6,-2,-2],[0.5,4.5,1.5,-2.0,-2.0,-2.0,0.719,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.003904,-0.002363,0.00196,-0.118694,0.376353,-0.787457,0.095221,-0.205453,0.759079]],[[42,11,14,-2,-2,7,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.0165,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001007,0.005213,0.009986,0.139564,-0.872025,-0.069542,0.136084,-1.413076,-0.792725]],[[14,11,5,-2,-2,7,-2,-2,34,-2,-2],[0.5,0.5,0.9835,-2.0,-2.0,0.0115,-2.0,-2.0,0.309524,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001582,0.003402,0.00835,0.127541,-0.418203,-0.068352,0.12565,-1.403918,-0.072609,-1.224978,-0.659858]],[[34,53,4,-2,-2,-2,36,5,-2,-2,2,-2,-2],[0.27682,0.5,0.9105,-2.0,-2.0,-2.0,0.5,0.898,-2.0,-2.0,0.985,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004516,0.016712,0.01081,-0.03734,0.678166,0.601262,-0.025757,0.001014,0.476114,-0.754247,-0.062139,-0.633558,0.570649]],[[5,3,4,-2,-2,7,-2,-2,5,2,-2,-2,2,-2,-2],[0.6845,0.832,0.0305,-2.0,-2.0,0.0065,-2.0,-2.0,0.9355,0.985,-2.0,-2.0,0.985,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000255,0.015466,0.026775,0.802622,0.126515,-0.075309,-1.080705,0.171196,-0.034006,-0.092732,-0.842846,-0.5647,0.002436,0.493287,-0.448132]],[[28,3,5,-2,-2,5,-2,-2,-2],[0.5,0.1145,0.5365,-2.0,-2.0,0.0065,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.004713,-0.00877,0.009721,0.268577,-0.506108,-0.029687,-1.040435,-0.195908,0.786831]],[[4,4,59,-2,-2,-2,6,6,-2,-2,30,-2,-2],[0.0585,0.0495,2.5,-2.0,-2.0,-2.0,0.244,0.12,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.00294,0.021888,0.008232,-0.102728,0.793805,1.610473,-0.018027,-0.108742,-0.349301,-1.593216,-0.009168,-0.000353,-0.866981]],[[4,4,7,-2,-2,-2,4,3,-2,-2,4,-2,-2],[0.0345,0.0285,0.855,-2.0,-2.0,-2.0,0.0465,0.0175,-2.0,-2.0,0.0585,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.006521,0.024658,0.013134,0.306016,-0.340486,1.056427,-0.020199,-0.13781,-2.292357,-0.237503,-0.012117,0.990878,-0.179782]],[[30,3,34,-2,-2,7,-2,-2,-2],[4.5,0.9955,0.458042,-2.0,-2.0,0.9375,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00242,0.006464,0.008963,0.115104,-0.516988,-0.102845,-1.314071,-0.793856,-0.800224]],[[34,53,34,-2,-2,-2,7,6,-2,-2,2,-2,-2],[0.396429,0.5,0.295207,-2.0,-2.0,-2.0,0.0025,0.91,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002537,0.011092,0.004782,-0.035269,0.75093,0.641434,-0.047395,0.017649,0.90024,-0.499037,-0.073118,-1.128476,-0.605132]],[[36,6,2,-2,-2,3,-2,-2,6,7,-2,-2,6,-2,-2],[0.5,0.9795,0.995,-2.0,-2.0,0.7535,-2.0,-2.0,0.0285,0.0125,-2.0,-2.0,0.2405,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002214,0.019504,-0.000535,0.152977,-0.599482,0.066879,0.583847,1.95554,-0.019364,0.144096,0.481006,1.431799,-0.030261,-0.906136,-0.110549]],[[27,6,6,-2,-2,7,-2,-2,34,-2,-2],[0.5,0.9535,0.63,-2.0,-2.0,0.229,-2.0,-2.0,0.223611,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004379,-0.008274,-0.02314,0.044006,-0.567548,0.015598,-0.061178,0.854657,0.076284,0.951326,-0.959715]],[[58,43,36,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.007291,-0.003345,-0.007533,0.117438,-0.225674,0.428507,-0.69905]],[[25,5,6,-2,-2,6,-2,-2,6,3,-2,-2,-2],[0.5,0.0205,0.9305,-2.0,-2.0,0.735,-2.0,-2.0,0.9655,0.029,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-4.7e-05,-0.005642,0.027462,0.623168,-0.031687,-0.015126,-0.384191,0.068555,0.055956,0.093242,0.687216,0.967788,-0.408247]],[[59,45,38,-2,-2,40,-2,-2,38,34,-2,-2,-2],[1.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.348485,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.007811,-0.002002,0.02579,0.283186,1.061125,-0.020765,-0.269292,0.387653,0.037964,0.055059,0.486226,-0.73867,-1.557469]],[[12,4,6,-2,-2,4,-2,-2,4,-2,-2],[0.5,0.0345,0.9875,-2.0,-2.0,0.0455,-2.0,-2.0,0.9315,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000905,0.003558,0.032415,0.160863,0.948815,-0.010124,-0.906533,-0.026945,-0.078699,-1.466441,1.192814]],[[57,60,3,-2,-2,-2,-2],[0.5,0.5,0.0015,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.001237,-0.002485,0.000125,0.588157,-0.076123,-1.162144,0.793234]],[[39,43,59,-2,-2,-2,-2],[0.5,0.5,4.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.006192,-0.002355,-0.006757,-0.025317,-0.927536,0.558409,-0.531748]],[[34,58,59,-2,-2,-2,34,36,-2,-2,7,-2,-2],[0.275253,0.5,2.5,-2.0,-2.0,-2.0,0.295207,0.5,-2.0,-2.0,0.0005,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003024,0.010994,0.00619,-0.024705,0.410829,0.681255,-0.039661,-0.132566,0.460792,-1.544493,-0.025084,0.869723,-0.394715]],[[54,60,23,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.001414,-0.002822,-0.000306,-0.056359,0.547065,-0.921483,0.777728]],[[46,3,59,-2,-2,34,-2,-2,4,-2,-2],[0.5,0.0105,0.5,-2.0,-2.0,0.33908,-2.0,-2.0,0.0715,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000368,0.004784,-0.020896,0.204918,-0.660983,0.017589,0.259303,-0.794571,-0.085954,0.014849,-0.964738]],[[55,36,6,-2,-2,4,-2,-2,5,-2,-2],[0.5,0.5,0.9785,-2.0,-2.0,0.833,-2.0,-2.0,0.1755,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00384,-0.009511,0.008833,-0.115189,0.696989,-0.033975,-0.35929,0.12394,0.070971,0.985376,0.111637]],[[6,3,2,-2,-2,2,-2,-2,36,11,-2,-2,6,-2,-2],[0.0285,0.469,0.975,-2.0,-2.0,0.975,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9955,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004235,0.051995,0.129156,1.181736,1.769107,-0.01401,0.87204,-1.116332,-0.009849,0.008244,0.185421,-1.718046,-0.031853,-0.084765,-0.835113]],[[2,38,7,-2,-2,4,-2,-2,38,10,-2,-2,3,-2,-2],[0.975,1.5,0.165,-2.0,-2.0,0.853,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.1285,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004854,0.009113,0.016384,-0.031826,0.706809,-0.158102,-1.281947,-0.157519,-0.028593,-0.040693,-0.518419,0.414488,0.058981,0.136227,1.347468]],[[25,4,4,-2,-2,5,-2,-2,6,6,-2,-2,-2],[0.5,0.845,0.3325,-2.0,-2.0,0.4975,-2.0,-2.0,0.97,0.7875,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001928,-0.007744,-0.019565,-0.012216,-0.664298,0.02082,0.601488,-0.859976,0.055655,0.079109,0.5828,0.906099,-0.210579]],[[3,2,38,-2,-2,5,-2,-2,7,-2,-2],[0.9955,0.975,1.5,-2.0,-2.0,0.261,-2.0,-2.0,0.9355,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002672,0.005177,0.017298,0.210912,-0.98287,-0.014761,0.202099,-0.391824,-0.10045,-0.962151,-0.970351]],[[2,38,7,-2,-2,-2,38,30,-2,-2,5,-2,-2],[0.975,1.5,0.187,-2.0,-2.0,-2.0,1.5,1.5,-2.0,-2.0,0.685,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.005923,0.019424,0.027692,0.123782,0.657107,-0.968827,-0.016138,-0.026868,-0.465941,0.160436,0.062621,1.051864,-0.631234]],[[14,11,6,-2,-2,36,-2,-2,2,-2,-2],[0.5,0.5,0.9975,-2.0,-2.0,0.5,-2.0,-2.0,0.96,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001735,0.003862,0.009226,0.119916,-0.491894,-0.08023,-1.780605,-0.18831,-0.072474,-1.133203,-1.09665]],[[11,2,5,-2,-2,7,-2,-2,36,-2,-2],[0.5,0.975,0.2895,-2.0,-2.0,0.0015,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004208,0.001062,0.014981,-0.053469,0.443392,-0.022194,-0.866647,-0.121618,-0.087215,-1.749372,-0.387661]],[[30,5,7,-2,-2,6,-2,-2,-2],[4.5,0.9835,0.235,-2.0,-2.0,0.421,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003056,0.001113,0.007269,-0.044682,0.397424,-0.048692,-0.879093,-0.113335,-0.868058]],[[35,11,14,-2,-2,3,-2,-2,6,-2,7,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.024,-2.0,-2.0,0.719,-2.0,0.02,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.007424,0.000191,0.00569,0.111661,-1.072354,-0.093887,-1.017683,-0.741991,0.10294,-0.317165,0.175565,1.166004,0.4997]],[[38,55,59,-2,-2,5,-2,-2,59,1,-2,-2,-2],[2.5,0.5,1.5,-2.0,-2.0,0.1755,-2.0,-2.0,1.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.006086,0.0092,0.00376,-0.061998,0.28508,0.079963,1.176391,0.14788,-0.07929,0.046588,-0.186952,1.255125,-1.599823]],[[52,6,45,-2,-2,6,-2,-2,-2],[0.5,0.305,0.5,-2.0,-2.0,0.381,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002312,-0.006047,-0.034719,0.5411,-0.625374,0.003354,1.325972,-0.023677,0.738555]],[[46,43,36,-2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.1335,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.001459,0.005396,0.00036,0.199006,-0.1701,0.791838,-0.062471,0.174563,-1.132998]],[[3,6,2,-2,-2,4,-2,-2,4,53,-2,-2,7,-2,-2],[0.0015,0.9325,0.96,-2.0,-2.0,0.849,-2.0,-2.0,0.3515,0.5,-2.0,-2.0,0.0045,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000213,0.036353,0.075732,0.710704,1.080048,0.011807,-0.890429,1.067194,-0.007558,0.007881,-0.011069,0.724213,-0.031472,0.293929,-0.608853]],[[53,14,59,-2,-2,5,-2,-2,6,-2,-2],[0.5,0.5,3.5,-2.0,-2.0,0.359,-2.0,-2.0,0.642,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001302,-0.003885,0.00078,0.048587,-0.470834,-0.06789,-0.871545,-1.341809,0.086195,1.170608,-0.163374]],[[60,12,34,-2,-2,4,-2,-2,-2],[0.5,0.5,0.03125,-2.0,-2.0,0.907,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001394,0.001393,0.005369,0.323054,-0.05616,-0.072267,-1.470324,1.269399,-1.031635]],[[46,26,43,-2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.062,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.002614,0.006885,0.010179,0.055509,0.702815,-1.206153,-0.073706,0.165339,-0.898162]],[[3,4,4,-2,-2,4,-2,-2,-2],[0.9965,0.0585,0.0465,-2.0,-2.0,0.234,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000181,0.002098,0.023585,0.103629,1.067032,-0.011619,-0.508891,0.026812,-1.395412]],[[4,4,4,-2,-2,5,-2,-2,6,36,-2,-2,6,-2,-2],[0.0585,0.0455,0.0375,-2.0,-2.0,0.144,-2.0,-2.0,0.244,0.5,-2.0,-2.0,0.547,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000979,0.0204,0.005832,0.140197,-0.989893,0.170036,0.139222,1.639957,-0.014603,-0.109099,-0.689507,-1.418533,-0.005846,0.60369,-0.14711]],[[34,23,4,-2,-2,3,-2,-2,34,36,-2,-2,3,-2,-2],[0.27682,0.5,0.9105,-2.0,-2.0,0.061,-2.0,-2.0,0.295207,0.5,-2.0,-2.0,0.5295,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000846,0.011637,0.005031,-0.075938,0.577996,0.07457,-0.32977,0.927031,-0.024925,-0.12191,0.330916,-1.260958,-0.010132,0.107948,-0.78907]],[[25,30,3,-2,-2,-2,4,6,-2,-2,-2],[0.5,4.5,0.9955,-2.0,-2.0,-2.0,0.6575,0.262,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.005765,0.000489,0.003554,0.056644,-0.937757,-0.673095,0.056735,0.089153,0.570192,0.978538,-0.596947]],[[5,12,3,-2,-2,-2,5,-2,4,-2,-2],[0.0755,0.5,0.832,-2.0,-2.0,-2.0,0.0925,-2.0,0.0895,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[-0.003993,0.018157,0.026578,0.403945,-0.527206,-0.983041,-0.01721,-1.514134,-0.010807,0.208204,-0.335367]],[[31,54,6,-2,-2,-2,1,-2,-2],[0.5,0.5,0.9575,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.000484,0.002457,-0.001299,-0.132132,0.200179,0.734321,-0.109004,-1.22871,1.396275]],[[39,5,7,-2,-2,7,-2,-2,-2],[0.5,0.9065,0.8955,-2.0,-2.0,0.9355,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003543,0.000399,0.011172,0.082533,1.816081,-0.040456,-1.012425,-0.219938,-0.500574]],[[34,5,7,-2,-2,2,-2,-2,5,-2,1,-2,-2],[0.423963,0.9835,0.235,-2.0,-2.0,0.985,-2.0,-2.0,0.0045,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001614,0.00717,0.012468,-0.009478,0.455277,-0.032811,0.211551,-0.813765,-0.048701,0.887057,-0.068046,-0.967467,0.830955]],[[34,40,6,-2,-2,6,-2,-2,2,-2,-2],[0.585714,0.5,0.9905,-2.0,-2.0,0.719,-2.0,-2.0,0.975,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005031,0.007477,0.003404,-0.045282,0.380867,0.054589,-0.565061,0.680627,-0.090044,-1.169495,-1.116351]],[[12,46,3,-2,-2,59,-2,-2,5,-2,-2],[0.5,0.5,0.9955,-2.0,-2.0,0.5,-2.0,-2.0,0.04,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002544,0.002116,0.006461,0.081528,-1.224731,-0.073248,0.213549,-0.819813,-0.082581,-1.489944,-0.293783]],[[28,46,30,-2,-2,4,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.0715,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004463,0.000222,0.004543,-0.164965,0.18843,-0.073893,0.072529,-0.909948,0.935509]],[[43,7,9,-2,-2,4,-2,-2,-2],[0.5,0.0065,0.5,-2.0,-2.0,0.3515,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.009789,0.005962,0.026822,0.17824,1.333578,-0.007024,0.202692,-0.597618,0.596706]],[[59,6,13,-2,-2,3,-2,-2,4,-2,-2],[3.5,0.9535,0.5,-2.0,-2.0,0.0495,-2.0,-2.0,0.0455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.007653,0.012735,-0.000414,-0.042495,0.528398,0.034422,-0.027061,0.75509,-0.070495,0.028651,-0.8647]],[[60,34,6,-2,-2,34,-2,-2,-2],[0.5,0.029412,0.735,-2.0,-2.0,0.104356,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00536,0.008125,0.033085,-0.068347,0.666499,-0.002002,-0.662432,0.035226,-1.083795]],[[57,5,6,-2,-2,4,-2,-2,-2],[0.5,0.6755,0.0095,-2.0,-2.0,0.9165,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002582,-0.005284,0.004104,1.353616,-0.011969,-0.028319,-0.136996,-1.255394,0.529745]],[[5,3,59,-2,-2,3,-2,-2,3,6,-2,-2,6,-2,-2],[0.6845,0.832,2.5,-2.0,-2.0,0.9665,-2.0,-2.0,0.0375,0.981,-2.0,-2.0,0.9825,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001562,0.009419,0.022548,0.133004,0.883641,-0.083875,-1.096388,-0.150399,-0.026186,-0.129747,-0.752025,-1.424455,0.002848,-0.190362,0.914901]],[[12,4,4,-2,-2,7,-2,-2,4,-2,-2],[0.5,0.3515,0.2835,-2.0,-2.0,0.0075,-2.0,-2.0,0.8585,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000638,0.002911,0.016913,0.069971,1.405051,-0.017715,0.231352,-0.502766,-0.065102,-1.476205,0.610403]],[[4,36,2,-2,-2,3,-2,-2,5,34,-2,-2,-2],[0.9985,0.5,0.985,-2.0,-2.0,0.0075,-2.0,-2.0,0.0335,0.129167,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000296,-0.002992,0.01221,0.383565,-0.310338,-0.021789,-0.503141,-0.003182,0.084236,0.015456,1.044321,-0.039688,2.568279]],[[59,1,4,-2,-2,2,-2,-2,34,3,-2,-2,38,-2,-2],[1.5,0.5,0.914,-2.0,-2.0,0.475,-2.0,-2.0,0.129167,0.0165,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003892,-0.00708,-0.020333,-0.069143,-0.737339,0.031201,0.675934,1.267823,0.037963,0.104786,0.117385,0.958905,-0.008074,0.107599,-1.139904]],[[57,34,59,-2,-2,4,-2,-2,-2],[0.5,0.064583,1.5,-2.0,-2.0,0.091,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001144,-0.001802,0.022753,-0.008852,0.845803,-0.012666,0.122999,-0.280899,0.605289]],[[2,37,2,-2,-2,37,-2,-2,7,4,-2,-2,5,-2,-2],[0.935,0.2815,0.835,-2.0,-2.0,0.808,-2.0,-2.0,0.0665,0.659,-2.0,-2.0,0.3555,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002015,0.027183,0.03952,0.752476,0.534418,-0.022448,-1.487509,0.922003,-0.009758,-0.029549,-0.398971,0.208162,0.009948,0.612377,-0.100292]],[[31,10,27,-2,-2,5,-2,-2,37,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4285,-2.0,-2.0,0.0015,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000577,0.002643,-0.003268,-0.077756,0.738746,0.047337,0.049596,0.81349,-0.087878,-1.151003,1.306517]],[[5,12,3,-2,-2,-2,5,2,-2,-2,2,-2,-2],[0.0475,0.5,0.0025,-2.0,-2.0,-2.0,0.0995,0.96,-2.0,-2.0,0.975,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003597,0.018846,0.030211,-0.100897,0.494727,-1.332543,-0.014438,-0.07632,-0.940945,0.536586,-0.005154,0.156723,-0.34395]],[[15,2,-2,-2,30,3,-2,-2,6,-2,-2],[0.5,0.96,-2.0,-2.0,0.5,0.0265,-2.0,-2.0,0.9975,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[0.002742,-0.064791,0.03428,-1.163109,0.007055,-0.014682,0.315571,-0.458843,0.023976,0.277013,-0.781469]],[[6,17,3,-2,-2,-2,45,6,-2,-2,7,-2,-2],[0.9965,0.5,0.9965,-2.0,-2.0,-2.0,0.5,0.9975,-2.0,-2.0,0.0025,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002736,0.009422,0.006343,0.078514,-1.3503,1.175965,-0.049531,0.021428,-0.408483,1.106883,-0.108149,-1.196193,-0.650634]]]
UMBRAL = 0.16
# <<< MODELO


def _recorrer(arbol, x):
    """Baja por el árbol respondiendo a sus preguntas hasta llegar a una hoja."""
    variable, umbral, izq, der, valor = arbol
    nodo = 0
    while izq[nodo] != -1:
        nodo = izq[nodo] if x[variable[nodo]] <= umbral[nodo] else der[nodo]
    return valor[nodo]


def probabilidad(feats):
    """Probabilidad de alucinación: suma de todos los árboles -> sigmoide."""
    x = [float(feats.get(n, 0.0)) for n in NOMBRES]
    z = INICIAL + TASA * sum(_recorrer(a, x) for a in ARBOLES)
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
