"""El Guardián — bot de joseca (versión 4).

v4 (propuesta A): verificador de cuentas DENTRO de la frase
(verificar_cuentas_frase): recalcula la operación concreta que hace la
respuesta con sus propias cifras («de A a B, un X %», «A por B ... fue de X»,
«A y B sumaron X», «X más que [otra entidad]», «perdió A, un X % de su
población») en vez de aceptar cualquier combinación casual de cifras.
Además: «el 30 % de 18.400» entre fuentes distintas, y comparación en float32
para que el bot dé exactamente lo mismo que scikit-learn.
En la validación por familias: H +0,42 -> +0,51, AUROC 0,907 -> 0,931.
Después, dos ideas generales (programadas por nosotros, tras ver qué
señales aprovechaban otros equipos): «matiz perdido por oración» (la oración
no conserva ninguna cautela y su fuente sí) y el reparto del respaldo entre
fragmentos (segundo peor, fracción de fragmentos flojos). La cobertura se
probó y se descartó (empeoraba). Resultado: H +0,55, AUROC 0,941.

--- Notas de la versión 3 ---

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
import struct
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


# ---------------------------------------------------------------------------
# Verificador de cuentas DENTRO de la frase (v4)
# ---------------------------------------------------------------------------
# Hasta v3, para saber si una cifra derivada era correcta buscábamos CUALQUIER
# combinación de cifras de las fuentes que la diera. Con tantas cifras, casi
# siempre aparecía alguna por casualidad, y así se escapaban cuentas falsas
# como «de 212 a 258 millones, un 12 %» (en realidad un 21,7 %).
# Aquí reconocemos la forma concreta de la cuenta que hace la frase y
# recalculamos ESA operación con las cifras que la propia frase menciona:
#   A. cambio:        «de A a B, un X %» / «X puntos» / «X más»
#   B. producto:      «A habitantes y B euros por habitante ... fue de X»
#   C. suma nombrada: «el fútbol sala y el baloncesto sumaron X»
#   D. comparación:   «X más/menos que [otra entidad / otro año]»
#   E. parte del total: «perdió A, un X % de su población»
# Cada cifra comprobada queda marcada como «ok» o «mal».

MESES_RE = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
RE_DIA_MES = re.compile(r"\s+de\s+(" + MESES_RE + r")\b")
RE_NUMERO_ORDEN = re.compile(r"(n\.?\s*[o\u00ba\u00b0]\.?|numero)\s*$")
RE_SUMA_NOMBRADA = re.compile(r"\b(suman|sumaron|suma|sumo|sumaban|sumarian|entre (los|las) (dos|tres)|en total)\b")
RE_VERBO_CAMBIO = re.compile(r"\b(aumento|aumentaron|crecio|crecieron|subio|subieron|bajo|bajaron|cayo|"
                             r"cayeron|redujo|mejoro|empeoro|gano|perdio|un|una)\s*([a-z]+\s+)?$")
RE_DESDE = re.compile(r"\b(de|del|desde)\s*$")
RE_HASTA = re.compile(r"\b(a|al|hasta)\s+(los\s+|las\s+|el\s+|la\s+)?$")
RE_TASA = re.compile(r"\bpor (habitante|persona|vecino|alumno|escolar|metro|kilo|litro|hora|dia|mes|"
                     r"ano|unidad|plaza|cama|hectarea|trabajador|empleado|socio|cliente|vivienda|"
                     r"hogar|familia|tripulante|deportista|participante|beneficiario|usuario)|"
                     r"\bal (mes|ano|dia)\b|\bmensual|\banual")
RE_RESULTADO = re.compile(r"(fue de|es de|sera de|seran|asciende a|ascendio a|supone|suponen|"
                          r"pagara|pagaran|paga|pagan|cuesta|costara|costo|ingresara|ingresa|"
                          r"recibira|recibe|total de|en total|es decir|lo que da|resulta|"
                          r"alcanza|alcanzo|suma|suman)\s+(unos?\s+|unas\s+|el\s+|los\s+|de\s+)?$")
RE_COMPARA = re.compile(r"^\s*(%\s*|por ciento\s*|puntos?\s*)?([a-z]+\s+){0,2}?(mas|menos|mayor|menor|"
                        r"superior|inferior)\s+(que|a|al)\s+")
RE_PARTE = re.compile(r"^[\s,]*(lo que supone\s+|es decir,?\s+)?(un|el)\s*$")
RE_PERDIDA = re.compile(r"\b(perdio|perdieron|pierde|bajo|bajaron|cayo|cayeron|redujo|redujeron|"
                        r"disminuyo|descendio|menos)\b")
RE_GANANCIA = re.compile(r"\b(gano|ganaron|gana|subio|subieron|crecio|crecieron|aumento|aumentaron)\b")
GENERICAS = {"total", "entre", "ambos", "ambas", "sumar", "suman", "sumar"}


def _valores_no_anio(nums):
    return [v for n in nums if not n["anio"] for v in n["valores"]]


def _casi(real, dicho, texto_antes):
    """¿Cuadra la cifra «dicha» con la calculada («real»)? Respeta «más de»,
    «casi», «unos»... que la frase ponga delante."""
    if real is None:
        return False
    if re.search(r"(mas del?|por encima del?|superior al?)\s*(un\s+|el\s+)?$", texto_antes):
        return dicho * 0.98 <= real <= dicho * 1.6 + 0.5
    if re.search(r"(menos del?|por debajo del?|inferior al?)\s*(un\s+|el\s+)?$", texto_antes):
        return dicho * 0.4 - 0.5 <= real <= dicho * 1.02
    if re.search(r"(casi|cerca de|alrededor de|unos|unas|aproximadamente)\s*(un\s+|el\s+)?$", texto_antes):
        return abs(real - dicho) <= max(0.1 * abs(dicho), 0.6)
    return abs(real - dicho) <= max(0.03 * abs(dicho), 0.6 if dicho >= 3 else 0.06)


def _fuentes_que_hablan_de(texto_norm, fuentes_norm):
    """Fuentes que más palabras (o años) comparten con un trozo de texto."""
    raices = raices_contenido(texto_norm) - GENERICAS
    anios = set(re.findall(r"\b(?:19|20)\d\d\b", texto_norm))
    puntos = []
    for i, fn in enumerate(fuentes_norm):
        p = len(raices & raices_contenido(fn)) + 2 * sum(1 for a in anios if a in fn)
        puntos.append(p)
    mejor = max(puntos) if puntos else 0
    return [i for i, p in enumerate(puntos) if p == mejor and p > 0]


def verificar_cuentas_frase(resp_norm, nums_resp, fuentes, nums_por_fuente):
    """Devuelve {índice de la cifra en nums_resp: ("ok"|"mal", tipo_de_cuenta)}."""
    fuentes_norm = [normalizar(x) for x in fuentes]
    todas_fuente = _valores_no_anio([n for ns in nums_por_fuente for n in ns])
    resultado = {}
    # Trocear la respuesta en frases, respetando posiciones
    cortes = [0] + [m.end() for m in re.finditer(r"[.;](\s|$)", resp_norm)] + [len(resp_norm)]
    for ini_f, fin_f in zip(cortes, cortes[1:]):
        en_frase = [k for k, n in enumerate(nums_resp) if ini_f <= n["ini"] < fin_f]
        # No cuentan los años, los días de una fecha («31 de julio») ni «el n.º 2»
        utiles = [k for k in en_frase if not nums_resp[k]["anio"]
                  and not RE_DIA_MES.match(resp_norm[nums_resp[k]["fin"]:nums_resp[k]["fin"] + 15])
                  and not RE_NUMERO_ORDEN.search(resp_norm[max(0, nums_resp[k]["ini"] - 8):nums_resp[k]["ini"]])]

        def copiada(k):
            """La cifra está tal cual en una fuente que habla de lo mismo (mismo
            tipo, % o no, y al menos 2 palabras en común con la frase): es un
            dato copiado, no una cuenta de la respuesta. No la juzgamos como «mal»."""
            raices_frase = raices_contenido(resp_norm[ini_f:fin_f])
            for i, ns in enumerate(nums_por_fuente):
                for n in ns:
                    if n["pct"] == nums_resp[k]["pct"] and any(
                            _cerca(v, w, 0.001) for v in n["valores"] for w in nums_resp[k]["valores"]):
                        if len(raices_frase & raices_contenido(fuentes_norm[i])) >= 2:
                            return True
            return False

        def anotar(k, ok, tipo):
            if ok:
                resultado[k] = ("ok", tipo)
            elif not copiada(k):
                resultado[k] = ("mal", tipo)

        def antes(k, ancho=30):
            return resp_norm[max(ini_f, nums_resp[k]["ini"] - ancho):nums_resp[k]["ini"]]

        def despues(k, ancho=40):
            return resp_norm[nums_resp[k]["fin"]:min(fin_f, nums_resp[k]["fin"] + ancho)]

        def base(k):
            return min(nums_resp[k]["valores"])  # el valor «tal cual» (sin millones)

        # --- A. Cambio «de A a B» + cifra que lo resume ---
        for x, a in enumerate(utiles):
            if not RE_DESDE.search(antes(a, 12)):
                continue
            siguientes = [k for k in utiles if nums_resp[k]["ini"] > nums_resp[a]["fin"]]
            if not siguientes:
                continue
            b = siguientes[0]
            if not RE_HASTA.search(resp_norm[nums_resp[a]["fin"]:nums_resp[b]["ini"]]):
                continue
            va, vb = base(a), base(b)
            for c in utiles:
                if c in (a, b) or c in resultado:
                    continue
                # Pegada detrás («de 6,4 a 6,8 millones, un 8 % más») o delante
                # («aumentó un 12 % en 2025, de 212 a 258»), sin saltar a otra cláusula
                if nums_resp[c]["ini"] > nums_resp[b]["fin"]:
                    hueco = resp_norm[nums_resp[b]["fin"]:nums_resp[c]["ini"]]
                    pegada = (len(hueco) <= 45 and ";" not in hueco
                              and re.search(r",\s*([a-z]+\s+){0,6}$", hueco))
                else:
                    hueco = resp_norm[nums_resp[c]["fin"]:nums_resp[a]["ini"]]
                    pegada = (nums_resp[c]["fin"] <= nums_resp[a]["ini"] and len(hueco) <= 35
                              and ";" not in hueco and RE_VERBO_CAMBIO.search(antes(c, 25)))
                if not pegada:
                    continue
                tras = despues(c, 25)
                dicho = base(c)
                if re.match(r"\s*puntos?\b", tras):
                    real = abs(vb - va)
                elif nums_resp[c]["pct"]:
                    reales = [abs(vb - va) / va * 100 if va else None]
                    if nums_resp[a]["pct"] and nums_resp[b]["pct"]:
                        reales.append(abs(vb - va))  # «subió 15 puntos» dicho como «un 15 %»
                    anotar(c, any(_casi(r, dicho, antes(c)) for r in reales), "cambio")
                    continue
                elif re.match(r"\s*([a-z]+\s+){0,2}(mas|menos)\b", tras):
                    real = abs(vb - va)
                elif re.match(r"\s*veces\b", tras):
                    real = max(va, vb) / min(va, vb) if min(va, vb) else None
                else:
                    continue
                anotar(c, _casi(real, dicho, antes(c)), "cambio")

        # --- B. Producto con una tasa («por habitante», «al mes») ---
        frase = resp_norm[ini_f:fin_f]
        if RE_TASA.search(frase) and len(utiles) >= 3:
            r = utiles[-1]
            if r not in resultado and RE_RESULTADO.search(antes(r, 35)):
                factores = [1.0]
                if re.search(r"\bal mes\b|\bmensual", frase) and re.search(r"\bal ano\b|\banual", frase):
                    factores += [12.0, 1 / 12.0]
                ops = [nums_resp[k]["valores"] for k in utiles[:-1]]
                ok = False
                for tam in (2, 3):
                    for combo in combinations(ops, tam):
                        prods = [1.0]
                        for vals in combo:
                            prods = [p * v for p in prods for v in vals]
                        if any(_cerca(rv, p * fct, 0.02) for p in prods for fct in factores
                               for rv in nums_resp[r]["valores"]):
                            ok = True
                anotar(r, ok, "producto")

        # --- C. Suma de cosas con nombre: «A y B sumaron X» ---
        m = RE_SUMA_NOMBRADA.search(frase)
        if m:
            pos_marca = ini_f + m.start()
            # (el «dos» de «entre los dos» no es el resultado de la suma)
            tras_marca = [k for k in utiles if ini_f + m.end() <= nums_resp[k]["ini"] <= ini_f + m.end() + 25
                          and not re.search(r"entre (los|las|ambos)\s*$", antes(k, 15))]
            if tras_marca and tras_marca[0] not in resultado:
                x = tras_marca[0]
                sujeto = resp_norm[ini_f:pos_marca]
                # Partes unidas por «y»; si hay «y», las comas también separan
                # («marroquí, rumana y colombiana»). Sin «y» no hay suma de cosas
                # («el verso, un endecasílabo, suma once sílabas»).
                trozos = re.split(r"\s(?:y|e)\s", sujeto)
                if len(trozos) >= 2:
                    trozos = re.split(r",", trozos[0])[-3:] + trozos[1:]
                partes = [p for p in trozos if raices_contenido(p) - GENERICAS]
                if len(partes) >= 2:
                    candidatos = []
                    for p in partes:
                        idx = _fuentes_que_hablan_de(p, fuentes_norm)
                        # Si la parte nombra una cifra en la propia respuesta, también vale
                        vals = set(_valores_no_anio([n for k2, n in enumerate(nums_resp)
                                                     if k2 in utiles and n["ini"] < pos_marca
                                                     and p.strip() and p.strip()[:8] in resp_norm]))
                        for i in idx:
                            # Solo las cifras que están junto a las palabras de esa parte
                            for n in nums_por_fuente[i]:
                                if n["anio"]:
                                    continue
                                cerca_txt = fuentes_norm[i][max(0, n["ini"] - 60):n["fin"] + 20]
                                if raices_contenido(p) & raices_contenido(cerca_txt):
                                    vals.update(n["valores"])
                        candidatos.append(sorted(vals)[:25])
                    if all(candidatos) and len(candidatos) <= 4:
                        sumas = [0.0]
                        for vals in candidatos:
                            sumas = [s + v for s in sumas for v in vals][:20000]
                        ok = any(_cerca(xv, s, 0.01) for s in sumas for xv in nums_resp[x]["valores"])
                        anotar(x, ok, "suma")

        # --- D. Comparación: «X más/menos que [otra entidad / otro año]» ---
        for c in utiles:
            if c in resultado:
                continue
            tras = resp_norm[nums_resp[c]["fin"]:fin_f]
            m = RE_COMPARA.match(tras)
            if not m:
                continue
            objetivo = re.split(r"[,;]|\sy\s", tras[m.end():m.end() + 60])[0]
            anclas = [k for k in utiles if k != c and nums_resp[k]["ini"] < nums_resp[c]["ini"]]
            idx = set(_fuentes_que_hablan_de(objetivo, fuentes_norm))
            idx |= {i for i, ns in enumerate(nums_por_fuente) for k in anclas
                    if any(_cerca(v, w, 0.001) for n in ns for v in n["valores"] for w in nums_resp[k]["valores"])}
            if not anclas or not idx:
                continue
            otras = _valores_no_anio([n for i in idx for n in nums_por_fuente[i]])
            # La otra cifra puede venir en la propia frase («... que las 4 horas de Lusel»)
            fin_obj = nums_resp[c]["fin"] + m.end() + len(objetivo)
            otras += _valores_no_anio([nums_resp[k] for k in utiles
                                       if nums_resp[c]["fin"] < nums_resp[k]["ini"] < fin_obj])
            dicho = base(c)
            pares = [(av, o) for k in anclas for av in nums_resp[k]["valores"] for o in otras if o]
            if re.match(r"\s*veces\b", tras):
                # «seis veces más que»: proporción, no resta
                ok = any(_casi(max(av, o) / min(av, o), dicho, antes(c)) for av, o in pares if min(av, o))
            elif nums_resp[c]["pct"] and not any(nums_resp[k]["pct"] for k in anclas):
                # «9.800 préstamos, un 15 % más que en 2024»: cambio relativo
                ok = any(_casi(abs(av - o) / o * 100, dicho, antes(c)) for av, o in pares)
            else:
                ok = any(_casi(abs(av - o), dicho, antes(c)) for av, o in pares)
            anotar(c, ok, "comparacion")

        # --- E. Parte del total: «perdió 2.410 habitantes, un 18 % de su población» ---
        for c in utiles:
            if c in resultado or not nums_resp[c]["pct"]:
                continue
            previas = [k for k in utiles if nums_resp[k]["fin"] < nums_resp[c]["ini"]
                       and not nums_resp[k]["pct"]]
            if not previas:
                continue
            x = previas[-1]
            entre = resp_norm[nums_resp[x]["fin"]:nums_resp[c]["ini"]]
            if not (re.search(r",\s*(lo que supone\s+|es decir,?\s+)?(un|el)\s*$", entre)
                    and re.match(r"\s*(%|por ciento)\s+(de|del)\b", resp_norm[nums_resp[c]["fin"]:fin_f])):
                continue
            vx = base(x) if base(x) > 1000 or not any(v > 1000 for v in nums_resp[x]["valores"]) else max(nums_resp[x]["valores"])
            # La base del porcentaje es la cifra de PARTIDA: si perdió X, la
            # base B cumple que B - X también está en las fuentes (si ganó, B + X)
            if RE_PERDIDA.search(frase):
                bases = [b for b in todas_fuente if any(_cerca(b - vx, o) for o in todas_fuente)]
            elif RE_GANANCIA.search(frase):
                bases = [b for b in todas_fuente if any(_cerca(b + vx, o) for o in todas_fuente)]
            else:
                bases = [b for b in todas_fuente if b > vx]  # «119.040 TEU, el 64 % del total»
            if not bases:
                continue
            dicho = base(c)
            anotar(c, any(_casi(100.0 * vx / b, dicho, antes(c)) for b in bases if b), "parte")
    return resultado


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

    # --- 2b. Reparto del respaldo entre fragmentos (v4) ---
    # Hasta ahora solo mirábamos el PEOR fragmento. Pero importa también
    # cuántos fragmentos flojean y cómo está el segundo peor: una respuesta con
    # un único fragmento flojo suele ser una paráfrasis; con varios, una invención.
    # Para cada fragmento tomamos el mejor respaldo, sea de una fuente sola o de
    # todas juntas (así no penalizamos las respuestas que combinan fuentes).
    resp_frag = sorted(max([x.get("respaldo_todas", 0.0)] + list(x.get("respaldo_por_fuente") or []))
                       for x in fragmentos) or [0.0]
    f["frag_respaldo_min"] = resp_frag[0]
    f["frag_respaldo_2o"] = resp_frag[1] if len(resp_frag) > 1 else resp_frag[0]
    # Fracciones (no recuentos), para que las respuestas largas del giro no suban solas
    f["frag_frac_bajo_05"] = sum(1 for v in resp_frag if v < 0.5) / len(resp_frag)
    f["frag_frac_bajo_02"] = sum(1 for v in resp_frag if v < 0.2) / len(resp_frag)
    f["frag_similitud_min"] = min([max(x.get("similitud_por_fuente") or [0.0]) for x in fragmentos] or [0.0])

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

    # v4: primero el verificador de cuentas dentro de la frase. Si reconoce la
    # cuenta, su veredicto manda sobre las comprobaciones genéricas de abajo.
    verificadas = verificar_cuentas_frase(resp_norm, nums_resp, fuentes, nums_por_fuente)
    # v4: «el 30 % de 18.400» o «el resto (30 %) de 18.400»: porcentaje de una
    # cantidad, aunque estén en fuentes distintas
    pct_fuentes = [v for n in nums_fuente if n["pct"] for v in n["valores"] if v < 100]
    totales = [v for n in nums_fuente if not n["pct"] and not n["anio"] for v in n["valores"]]
    partes_de_total = {p * t / 100.0 for p in pct_fuentes + [100.0 - q for q in pct_fuentes] for t in totales}

    clases = []
    for k, n in enumerate(nums_resp):
        # Otras cifras de la respuesta («anclas»): una cuenta fiel suele usarlas
        anclas = operandos_de([d for d in directos if d is not n])
        tipo = tipo_de_cuenta(resp_norm, n)
        if k in verificadas:
            clases.append("cuenta_correcta" if verificadas[k][0] == "ok" else "cuenta_falsa")
        elif tipo and not n["anio"] and any(esta_en(n, nums_por_fuente[i]) for i in fuentes_base):
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
        elif not n["pct"] and any(_cerca(v, x, 0.005) for v in n["valores"] for x in partes_de_total):
            clases.append("derivado_local")  # v4: porcentaje de un total («5.520 pinos»)
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
    # v4: veredictos del verificador por separado, para que el modelo pueda
    # fiarse más de ellos que de las comprobaciones genéricas
    f["verif_cuenta_mal"] = float(min(2, sum(1 for e, _ in verificadas.values() if e == "mal")))
    f["verif_cuenta_ok"] = float(min(2, sum(1 for e, _ in verificadas.values() if e == "ok")))
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

    # v4: cobertura. (a) Qué parte de la fuente citada por el LLM se queda
    # fuera de la respuesta: si la respuesta usa una fuente larga y se deja la
    # mitad, puede estar quitando un matiz. (b) Qué parte de la respuesta no
    # sale de las fuentes en las que se apoya: si dice cosas que solo están en
    # otras fuentes, puede estar mezclando datos de entidades distintas.
    usadas = set(citadas)
    for x in fragmentos:
        rpf = x.get("respaldo_por_fuente") or []
        if rpf:
            usadas.add(max(range(len(rpf)), key=lambda k: rpf[k]))
    usadas = {i for i in usadas if 0 <= i < n_fuentes}
    raices_r = raices_contenido(respuesta)
    fuera = []
    for o in oraciones:
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            rf = raices_contenido(fuentes[i])
            fuera.append(len(rf - raices_r) / max(1, len(rf)))
    f["cobertura_citada_fuera"] = max(fuera) if fuera else 0.5
    raices_usadas = raices_contenido(" ".join(fuentes[i] for i in usadas) + " " + pregunta)
    f["resp_fuera_de_usadas"] = len(raices_r - raices_usadas) / max(1, len(raices_r))

    # v4: matiz perdido «por oración»: la oración no tiene NINGÚN matiz y la
    # fuente en la que se apoya sí. Más tolerante con las paráfrasis que contar
    # matiz a matiz: si la oración conserva alguna cautela, no salta.
    todos_matices = [p for pats in MATICES.values() for p in pats]
    pierde = 0.0
    for o in oraciones:
        on = normalizar(o.get("texto") or "")
        if any(re.search(p, on) for p in todos_matices):
            continue
        fuentes_o = set()
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            fuentes_o.add(i)
        sim = o.get("similitud_por_fuente") or []
        if sim:
            fuentes_o.add(max(range(len(sim)), key=lambda k: sim[k]))
        if any(any(re.search(p, normalizar(fuentes[j])) for p in todos_matices) for j in fuentes_o if j < n_fuentes):
            pierde = 1.0
    f["matiz_oracion_perdido"] = pierde
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
        if nombre.startswith(("num_", "verif_", "matiz_", "anadido_", "huecos_", "nombre_", "entidad_",
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
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas"]
INICIAL = -0.035456
TASA = 0.05
ARBOLES = [[[2,3,44,45,-2,-2,-2,4,3,-2,-2,5,-2,-2,38,15,8,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,0.7999999821186066,-2.0,-2.0,-2.0,0.28450000286102295,0.8585000038146973,-2.0,-2.0,0.06400000117719173,-2.0,-2.0,0.5,0.5,0.07600000128149986,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.2361111119389534,0.9639999866485596,-2.0,-2.0,0.3920000046491623,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.003123,0.414307,0.470507,0.499429,2.029206,1.672342,1.099628,0.284888,0.401079,1.958398,1.035778,0.15553,1.464484,0.294805,-0.180073,-0.253411,-0.303999,0.227305,-1.426177,0.164366,-0.113343,1.717811,0.161875,0.361804,1.8685,0.455349,-0.21841,0.285542,-1.459743]],[[2,8,53,3,-2,-2,-2,7,8,-2,-2,12,-2,-2,8,4,12,-2,-2,53,-2,-2,38,18,-2,-2,42,-2,-2],[0.9350000023841858,0.17599999904632568,0.5,0.03350000083446503,-2.0,-2.0,-2.0,0.001500000071246177,0.8425000011920929,-2.0,-2.0,0.801499992609024,-2.0,-2.0,0.07600000128149986,0.08900000154972076,0.4000000059604645,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.00217,0.384225,0.463661,0.480351,1.938536,1.6406,1.32736,0.301958,0.460039,1.953784,1.606365,0.247804,0.774852,1.970837,-0.175893,0.192759,0.415825,2.011573,1.449406,0.049516,1.833115,-0.225823,-0.237859,-0.292026,-1.3107,0.461629,0.066632,0.941949,-1.538252]],[[2,53,3,45,-2,-2,42,-2,-2,5,-2,9,-2,-2,8,4,7,-2,-2,1,-2,-2,38,15,-2,-2,42,-2,-2],[0.9249999821186066,0.5,0.032500000670552254,0.7800000011920929,-2.0,-2.0,0.24038461595773697,-2.0,-2.0,0.023000000044703484,-2.0,0.9480000138282776,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.019999999552965164,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.17995689809322357,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.004673,0.363377,0.417046,0.451309,1.848674,1.544747,0.349238,1.60673,0.548731,0.166435,1.746028,0.048465,-0.489781,0.829253,-0.17073,0.160555,0.324096,0.724495,1.677876,-0.084755,-0.706168,1.933868,-0.234678,-0.294087,-1.310075,0.084223,0.070384,1.39837,-1.01035]],[[2,53,3,8,-2,-2,42,-2,-2,5,-2,9,-2,-2,8,4,4,-2,-2,5,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,0.17149999737739563,-2.0,-2.0,0.24038461595773697,-2.0,-2.0,0.02350000012665987,-2.0,0.9784999787807465,-2.0,-2.0,0.09149999916553497,0.8700000047683716,0.08900000154972076,-2.0,-2.0,0.07499999925494194,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.014661,0.340953,0.379876,0.427106,1.777785,1.553792,0.289824,1.379231,0.269365,0.17125,1.69057,0.074854,-0.338006,0.940018,-0.173967,0.131209,0.321959,1.696396,0.661882,-0.125742,0.037559,-1.500037,-0.235032,-0.286574,-1.309857,0.347232,0.039693,0.997485,-1.13477]],[[2,53,3,0,-2,-2,45,-2,-2,9,-2,-2,8,4,7,-2,-2,9,-2,-2,2,15,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,0.5,-2.0,-2.0,0.5565000176429749,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.09149999916553497,0.8700000047683716,0.0035000001080334187,-2.0,-2.0,0.8824999928474426,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.010987,0.322549,0.369591,0.390775,1.686956,1.383502,0.22752,0.170249,1.717899,0.127894,-0.259473,1.113384,-0.16029,0.104414,0.28639,0.415662,1.488701,-0.12013,-1.168778,0.394934,-0.212199,-0.072699,-0.51639,1.253787,-0.292202,-0.79194,-1.454801]],[[2,7,3,53,-2,-2,-2,44,3,-2,-2,-2,8,4,6,-2,-2,53,-2,-2,38,18,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.636000007390976,0.5,-2.0,-2.0,-2.0,0.5,0.107999999076128,-2.0,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.3439999967813492,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.17995689809322357,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.00524,0.320448,0.379449,0.39077,1.672722,1.324922,0.831656,0.224007,0.280651,1.547362,0.624028,-0.085495,-0.156463,0.13736,0.277757,1.562538,0.602508,-0.069801,1.565893,-0.654251,-0.209967,-0.251952,-1.167147,0.703083,0.027563,1.377575,-1.049161]],[[2,53,3,42,-2,-2,45,-2,-2,7,-2,12,-2,-2,8,44,4,-2,-2,4,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,0.4833333343267441,-2.0,-2.0,0.718999981880188,-2.0,-2.0,0.0025000000605359674,-2.0,0.47699999809265137,-2.0,-2.0,0.1314999982714653,0.5,0.9160000085830688,-2.0,-2.0,0.060499999672174454,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2613636404275894,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.007961,0.301428,0.343259,0.366835,1.582555,0.875034,0.22192,0.322023,1.628688,0.122063,1.29053,0.01981,1.09351,-0.411873,-0.130774,0.12959,0.397659,1.768194,1.119255,0.017973,1.385063,-0.338275,-0.185457,-0.238035,-1.119309,0.461612,0.073325,0.997807,-1.077742]],[[53,5,39,3,-2,-2,-2,42,-2,-2,2,52,70,-2,-2,5,-2,-2,51,65,-2,-2,-2],[0.5,0.5550000071525574,0.5,0.11449999734759331,-2.0,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.1525000035762787,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[-0.013723,0.2834,0.329929,0.341699,1.540658,1.12173,0.722995,0.004838,0.697123,-1.399349,-0.149926,0.001093,0.073233,-0.000585,1.512934,-0.398302,-1.880191,-1.470091,-0.260748,-0.288743,-1.331892,0.336415,0.581925]],[[53,69,5,2,-2,-2,1,-2,-2,-2,2,19,38,-2,-2,8,-2,-2,51,32,-2,-2,-2],[0.5,0.5,0.4529999941587448,0.4749999940395355,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.726500004529953,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.00296,0.262327,0.310368,0.331769,1.408424,1.714462,0.185794,0.255476,1.279384,-0.732051,-0.123107,0.012274,0.068176,-0.017361,1.335592,-0.336164,-1.100292,-1.692159,-0.223521,-0.256014,-1.215634,0.449478,0.757549]],[[2,7,3,12,-2,-2,-2,44,4,-2,-2,-2,38,15,8,-2,-2,9,-2,-2,42,12,-2,-2,12,-2,-2],[0.9350000023841858,0.007500000298023224,0.6884999871253967,0.48349998891353607,-2.0,-2.0,-2.0,0.5,0.27400000393390656,-2.0,-2.0,-2.0,0.5,0.5,0.1314999982714653,-2.0,-2.0,0.9814999997615814,-2.0,-2.0,0.2567765712738037,0.5664999783039093,-2.0,-2.0,0.4270000010728836,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.002076,0.268479,0.314104,0.324555,1.303092,1.524264,0.768395,0.192863,0.248676,1.548067,0.715748,-0.36162,-0.118671,-0.160384,-0.195674,0.072953,-1.029251,0.160852,0.127174,1.462395,0.123018,0.278807,0.679138,1.777451,-0.230504,-0.109766,-1.663336]],[[53,69,5,46,-2,-2,12,-2,-2,-2,54,32,2,-2,-2,10,-2,-2,-2],[0.5,0.5,0.5289999842643738,1.5,-2.0,-2.0,0.6539999842643738,-2.0,-2.0,-2.0,0.5,0.5,0.9600000083446503,-2.0,-2.0,0.2222222276031971,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.001994,0.237885,0.273754,0.294775,1.294585,1.657646,0.125409,1.050962,0.088066,-0.71035,-0.113847,-0.138151,-0.177025,-0.269467,-1.115459,0.213848,0.322975,1.521305,1.479795]],[[2,5,69,2,-2,-2,-2,-2,38,15,53,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,0.75,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.2361111119389534,0.9460000097751617,-2.0,-2.0,0.390500009059906,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.001648,0.245604,0.266945,0.252481,1.278247,-0.214151,1.623127,0.262313,-0.115139,-0.164368,-0.195358,0.448789,-0.995459,0.103588,-0.32676,1.416502,0.102514,0.26287,1.524696,0.110787,-0.231943,-0.087711,-1.559607]],[[2,8,7,8,-2,-2,-2,9,3,-2,-2,1,-2,-2,32,15,46,-2,-2,9,-2,-2,42,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,0.048499999567866325,-2.0,-2.0,-2.0,0.9634999930858612,0.33650000393390656,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,1.5,-2.0,-2.0,0.9524999856948853,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[0.0021,0.224172,0.278624,0.285378,1.419868,1.213054,0.963083,0.165508,0.069919,0.734688,-0.490818,0.228734,0.736552,1.308959,-0.100805,-0.133055,-0.164965,-0.858971,0.354028,0.167329,-0.095798,1.390412,0.231827,1.601132,0.341612]],[[2,8,2,3,-2,-2,-2,44,4,-2,-2,-2,31,15,0,-2,-2,9,-2,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,0.010499999858438969,-2.0,-2.0,-2.0,0.5,0.19250000268220901,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.003225,0.205604,0.271309,0.262054,1.377176,1.083563,1.49819,0.134846,0.179548,1.256,0.502089,-0.038946,-0.101071,-0.125238,-0.151528,-0.743984,1.484217,0.127615,-0.057363,1.32626,1.185566]],[[2,5,45,3,-2,-2,-2,42,6,-2,-2,4,-2,-2,38,15,45,-2,-2,9,-2,-2,42,-2,70,-2,-2],[0.9350000023841858,0.023000000044703484,0.7999999821186066,0.07200000062584877,-2.0,-2.0,-2.0,0.24264705926179886,0.8695000112056732,-2.0,-2.0,0.016500000841915607,-2.0,-2.0,0.5,0.5,0.0005000000237487257,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.11324786394834518,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,25,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,26,-1,-1],[-0.000999,0.208083,0.260044,0.270389,1.339611,1.520555,0.806113,0.166617,0.205646,0.620431,1.215393,0.038262,1.522699,-0.575908,-0.099091,-0.139601,-0.168352,-0.80915,1.859676,0.095771,-0.260574,1.282742,0.088783,1.353541,-0.055589,1.660852,-0.429609]],[[2,3,5,8,-2,-2,1,-2,-2,4,3,-2,-2,5,-2,-2,38,15,0,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.04749999940395355,0.05350000038743019,0.17149999737739563,-2.0,-2.0,0.5,-2.0,-2.0,0.44350001215934753,0.7790000140666962,-2.0,-2.0,0.0885000005364418,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.9554999768733978,-2.0,-2.0,0.2899159789085388,0.9864999949932098,-2.0,-2.0,0.443000003695488,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.005529,0.195705,0.235501,0.213402,1.268187,-0.091114,0.286029,1.518043,1.360133,0.12536,0.200968,1.185761,0.48136,-0.042376,0.929507,-0.798296,-0.097432,-0.143168,-0.170189,-0.841509,1.363233,0.119794,-0.365166,1.192858,0.122213,0.259518,1.458216,0.097727,-0.268576,-0.710184,-1.550651]],[[53,5,3,46,-2,-2,37,-2,-2,-2,31,21,52,-2,-2,-2,-2],[0.5,0.965499997138977,0.012500000186264515,1.5,-2.0,-2.0,2.5,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,-1],[-0.004054,0.183146,0.204783,0.248128,1.274225,1.57058,0.136274,0.850637,-0.343534,-0.840336,-0.090642,-0.11476,-0.134985,-0.473966,-1.54091,1.209603,1.103791]],[[2,19,16,17,-2,-2,-2,7,-2,-2,0,1,18,-2,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.11600000038743019,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,18,-1,-1],[0.003763,0.112257,0.14628,0.175383,0.953277,-1.247256,-1.149386,-0.28779,-0.847961,-1.802175,-0.140013,-0.158211,-0.170229,-0.906305,0.51246,1.722939,0.30396,1.389661,1.584567]],[[2,5,42,3,-2,-2,-2,1,4,-2,-2,7,-2,-2,31,21,1,-2,-2,-2,-2],[0.9350000023841858,0.14750000089406967,0.44949495792388916,0.09750000014901161,-2.0,-2.0,-2.0,0.5,0.47050000727176666,-2.0,-2.0,0.25200000405311584,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[0.001662,0.170183,0.201756,0.211485,1.117172,1.288677,0.357605,0.094979,-0.020338,0.479425,-0.735507,0.203349,1.349216,0.475216,-0.078129,-0.097022,-0.114193,-0.584137,1.686356,1.307231,1.164986]],[[53,5,46,42,-2,-2,8,-2,-2,-2,65,54,52,-2,-2,-2,42,-2,-2],[0.5,0.9629999995231628,1.5,0.24264705926179886,-2.0,-2.0,0.0860000029206276,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,18,-1,-1],[0.00264,0.168549,0.188541,0.174628,1.118929,0.586724,0.309803,1.349474,1.790892,-0.571963,-0.072252,-0.100571,-0.120989,-0.432653,-1.408139,1.178435,0.175095,1.330101,0.002282]],[[2,19,16,17,-2,-2,-2,12,-2,-2,46,70,43,-2,-2,-2,53,8,-2,-2,3,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5439999997615814,-2.0,-2.0,1.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.05650000087916851,-2.0,-2.0,0.06650000065565109,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.005548,0.102223,0.131169,0.15797,0.89447,-0.988574,-1.128103,-0.256319,-0.875258,-1.657151,-0.124042,-0.154568,-0.180856,-1.008628,0.207457,0.247082,0.092369,0.28926,1.31571,1.666585,-0.130305,-1.508325,0.191537]],[[2,7,3,3,-2,-2,42,-2,-2,8,3,-2,-2,-2,42,37,2,-2,-2,4,-2,-2,8,5,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.012500000186264515,0.006500000134110451,-2.0,-2.0,0.30384616553783417,-2.0,-2.0,0.9359999895095825,0.032500000670552254,-2.0,-2.0,-2.0,0.12916667014360428,1.5,0.9750000238418579,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.0755000002682209,0.0494999997317791,-2.0,-2.0,0.33908046782016754,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.002879,0.159795,0.194118,0.174793,1.186755,0.291685,0.232588,1.323331,0.857661,0.106625,0.060107,0.965973,-0.267217,1.364123,-0.070732,0.01996,-0.068477,0.349074,-0.788366,0.207931,2.066604,-0.087734,-0.122359,0.078796,0.862698,-0.827449,-0.162857,-0.566751,-1.304742]],[[2,19,16,17,-2,-2,-2,6,-2,-2,8,5,6,-2,-2,-2,42,2,-2,-2,46,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9275000095367432,-2.0,-2.0,0.13349999487400055,0.07250000163912773,0.9845000207424164,-2.0,-2.0,-2.0,0.12916667014360428,0.9850000143051147,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[-0.00271,0.080399,0.110768,0.133801,0.80801,-1.079785,-1.225656,-0.290938,-1.183881,-1.64904,-0.111666,0.074506,0.188784,1.261493,0.181542,-0.591319,-0.1482,-0.051902,0.685929,-0.627866,-0.223833,-1.221385,0.285365]],[[2,7,53,45,-2,-2,-2,12,9,-2,-2,42,-2,-2,38,8,4,-2,-2,42,-2,-2,42,6,-2,-2,-2],[0.9350000023841858,0.001500000071246177,0.5,0.15450000017881393,-2.0,-2.0,-2.0,0.7975000143051147,0.9670000076293945,-2.0,-2.0,0.18333333730697632,-2.0,-2.0,0.5,0.05649999901652336,0.878000020980835,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.2899159789085388,0.9864999949932098,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[0.005263,0.142305,0.191672,0.183568,1.200779,1.265786,1.158826,0.116883,0.088701,0.131307,0.802297,0.206553,1.099205,1.360571,-0.060657,-0.095093,0.097022,1.080327,-0.44835,-0.120454,-0.251417,-0.840291,0.114967,0.209943,1.313208,-0.166287,-0.895158]],[[2,8,2,1,-2,-2,-2,8,-2,8,-2,-2,8,4,5,-2,-2,5,-2,-2,42,30,-2,-2,46,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,0.5,-2.0,-2.0,-2.0,0.2565000057220459,-2.0,0.9440000057220459,-2.0,-2.0,0.0755000002682209,0.9280000030994415,0.032500000670552254,-2.0,-2.0,0.015000000596046448,-2.0,-2.0,0.2604166716337204,0.5,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.005949,0.135717,0.173122,0.163765,0.808113,1.216342,1.277107,0.098032,-0.180401,0.13033,0.504222,1.098456,-0.070425,0.089119,0.166013,1.212299,0.131614,-0.077849,0.292703,-0.744653,-0.096753,-0.046999,-0.359428,1.092542,-0.194434,-1.056233,0.224002]],[[53,69,46,3,-2,-2,8,-2,-2,-2,64,21,38,-2,-2,-2,-2],[0.5,0.5,1.5,0.9379999935626984,-2.0,-2.0,0.07250000163912773,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,-1],[0.002652,0.139065,0.159403,0.14582,0.989054,-0.051341,0.27418,1.317624,1.681623,-0.505022,-0.057636,-0.076244,-0.092683,-0.595136,0.326947,1.122343,1.068233]],[[2,8,4,44,-2,-2,4,-2,-2,8,-2,5,-2,-2,32,54,1,-2,-2,-2,12,-2,-2],[0.9350000023841858,0.17599999904632568,0.9455000162124634,0.5,-2.0,-2.0,0.9705000221729279,-2.0,-2.0,0.2565000057220459,-2.0,0.6689999997615814,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.408500000834465,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,22,-1,-1],[-0.003585,0.115613,0.156598,0.179331,1.20431,1.24687,0.129783,0.01267,1.191187,0.073925,-0.424647,0.100327,0.739759,-0.021218,-0.058041,-0.082734,-0.098173,-0.54564,1.546176,1.109823,0.15776,1.52782,0.389922]],[[2,19,17,16,-2,-2,-2,7,-2,-2,18,0,42,-2,-2,4,-2,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,0.11437908560037613,-2.0,-2.0,0.11600000038743019,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[0.006511,0.08312,0.111522,0.127283,0.792138,-0.820527,-0.96469,-0.243122,-0.7023,-1.659381,-0.090179,-0.105877,-0.121478,-0.108637,-1.017521,0.227178,1.264906,1.410422,0.861351]],[[2,19,16,17,-2,-2,-2,8,-2,-2,8,5,8,-2,-2,-2,42,37,-2,-2,46,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.7374999821186066,-2.0,-2.0,0.13199999928474426,0.032500000670552254,0.018499999307096004,-2.0,-2.0,-2.0,0.11437908560037613,1.5,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.006805,0.074417,0.101027,0.122495,0.791841,-1.135522,-0.972656,-0.23483,-0.495494,-1.592521,-0.088287,0.083255,0.240277,0.826637,1.740687,-0.400854,-0.118559,-0.036594,-0.652348,1.181264,-0.168226,-1.002612,0.145307]],[[2,19,16,17,-2,-2,-2,7,-2,-2,8,5,3,-2,-2,-2,42,2,-2,-2,3,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.13199999928474426,0.07250000163912773,0.0025000000605359674,-2.0,-2.0,-2.0,0.11437908560037613,0.9850000143051147,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[-0.00614,0.063993,0.09201,0.108909,0.717543,-0.903091,-0.722883,-0.238794,-0.912458,-1.546854,-0.096961,0.055047,0.146977,0.354333,1.347067,-0.537979,-0.127561,-0.072431,0.519544,-0.755395,-0.15858,-0.092178,-1.012559]],[[2,8,37,7,-2,-2,3,-2,-2,1,-2,42,-2,-2,21,32,1,-2,-2,42,-2,-2,-2],[0.9350000023841858,0.7449999749660492,0.5,0.035499999299645424,-2.0,-2.0,0.01899999938905239,-2.0,-2.0,0.5,-2.0,0.18333333730697632,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.001389,0.112018,0.13225,0.10826,0.989791,-0.167323,0.161246,1.07592,1.290497,0.047352,-0.265566,0.153631,1.232486,0.366495,-0.050826,-0.063114,-0.0825,-0.480263,1.538586,0.127004,1.418489,-0.101754,1.213196]],[[2,4,4,8,-2,-2,3,-2,-2,8,8,-2,-2,4,-2,-2,42,37,2,-2,-2,4,-2,-2,8,4,-2,-2,32,-2,-2],[0.9350000023841858,0.948500007390976,0.44200000166893005,0.9440000057220459,-2.0,-2.0,0.07499999925494194,-2.0,-2.0,0.04899999871850014,0.02350000012665987,-2.0,-2.0,0.9819999933242798,-2.0,-2.0,0.11324786394834518,1.5,0.9750000238418579,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.09699999913573265,0.8684999942779541,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.004244,0.100605,0.074602,0.113241,0.610828,1.049963,-0.007529,0.713324,-0.788401,0.147078,0.128774,1.15911,0.877213,0.169498,1.318623,1.181975,-0.052165,0.039182,-0.031963,0.351519,-0.580658,0.207587,1.7392,0.209993,-0.100924,0.036972,0.843965,-0.525873,-0.132702,-0.827382,0.236851]],[[19,2,17,16,-2,-2,-2,5,8,-2,-2,3,-2,-2,6,-2,55,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,0.5489999949932098,0.593999981880188,-2.0,-2.0,0.06200000084936619,-2.0,-2.0,0.7094999849796295,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[-0.003723,0.019432,0.072599,0.088931,0.634031,-0.702177,-1.075107,-0.066946,0.037762,0.572898,-0.480682,-0.146518,-1.383414,-0.665694,-0.21699,-1.111598,-0.243984,-1.507609,-1.263213]],[[19,2,16,17,-2,-2,-2,46,15,-2,-2,42,-2,-2,6,5,-2,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.20256410539150238,-2.0,-2.0,0.9804999828338623,0.20099999755620956,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,-1],[0.014596,0.035286,0.089583,0.10459,0.706772,-0.623575,-0.772266,-0.053913,-0.081585,-0.609938,0.83239,0.147506,1.61932,0.496128,-0.203933,-0.165693,-0.720923,-1.280529,-1.377596]],[[11,1,38,18,-2,-2,42,-2,-2,3,46,-2,-2,-2,65,5,6,-2,-2,2,-2,-2,-2],[0.20714285969734192,0.5,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.3269999921321869,3.5,-2.0,-2.0,-2.0,0.5,0.02850000001490116,0.9975000023841858,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.00168,-0.044366,-0.066085,-0.09427,-0.629454,0.639141,0.077821,0.899395,-0.72171,0.128441,0.172467,1.153851,1.565758,0.188822,0.099217,0.082921,0.137246,1.104102,0.377558,0.02296,1.004759,-0.220982,1.381808]],[[53,69,39,46,-2,-2,1,-2,-2,-2,42,38,69,-2,-2,12,-2,-2,2,-2,43,-2,-2],[0.5,0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.17914438992738724,0.5,2.5,-2.0,-2.0,0.5769999921321869,-2.0,-2.0,0.9249999821186066,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,22,-1,-1],[0.002821,0.09683,0.111764,0.121162,0.901692,1.220918,-0.010963,-1.09675,1.290164,-0.22989,-0.040806,0.014405,-0.023224,-0.315481,0.955067,0.203752,0.608175,1.465485,-0.095344,0.506858,-0.112343,-0.818544,0.213214]],[[2,19,16,50,-2,-2,-2,7,-2,-2,8,6,7,-2,-2,5,-2,-2,42,4,-2,-2,12,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.010999999940395355,-2.0,-2.0,0.09299999848008156,0.8015000224113464,0.7419999837875366,-2.0,-2.0,0.06299999915063381,-2.0,-2.0,0.11437908560037613,0.006500000134110451,-2.0,-2.0,0.26649999618530273,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[4.8e-05,0.05037,0.071144,0.084675,0.596021,-0.841944,-0.788378,-0.17138,-0.559664,-1.472848,-0.066431,0.051714,0.220975,1.3219,1.016948,-0.050523,0.334301,-1.089341,-0.089425,-0.020148,-1.188881,0.114928,-0.128936,0.036678,-0.937397]],[[19,2,44,42,-2,-2,42,-2,-2,5,8,-2,-2,4,-2,-2,7,-2,2,-2,-2],[0.5,0.9750000238418579,0.5,0.5657142996788025,-2.0,-2.0,0.02777777798473835,-2.0,-2.0,0.5489999949932098,0.7365000247955322,-2.0,-2.0,0.05350000038743019,-2.0,-2.0,0.007500000298023224,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[0.003945,0.020432,0.060293,0.107598,0.861252,-1.321004,-0.003755,0.577762,-0.228319,-0.045505,0.029302,0.56556,-0.65976,-0.104779,-0.325365,-1.066817,-0.176313,-0.83101,-0.200924,-1.391675,-1.226957]],[[19,8,39,6,-2,-2,-2,42,2,-2,-2,6,-2,-2,6,-2,2,-2,-2],[0.5,0.08900000154972076,0.5,0.5525000095367432,-2.0,-2.0,-2.0,0.33908046782016754,0.9750000238418579,-2.0,-2.0,0.15699999779462814,-2.0,-2.0,0.9275000095367432,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[0.004102,0.021132,0.11702,0.126628,1.193769,0.796041,-0.206718,-0.012552,0.012568,0.338989,-0.328593,-0.145499,-1.416228,-0.73114,-0.176771,-0.986896,-0.210779,-1.360073,-1.257289]],[[2,19,16,17,-2,-2,-2,8,-2,-2,8,6,12,-2,-2,-2,4,10,-2,-2,53,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,0.04649999924004078,0.9165000021457672,0.2814999967813492,-2.0,-2.0,-2.0,0.14400000125169754,0.2916666716337204,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.00574,0.055348,0.073534,0.086427,0.623379,-0.718423,-0.812276,-0.147194,-0.103975,-1.430521,-0.061023,0.109516,0.200985,0.966422,1.46266,-0.06353,-0.079637,-0.027409,-0.375461,1.177129,-0.145993,0.265382,-1.037496]],[[8,9,44,46,-2,-2,6,-2,-2,44,13,-2,-2,-2,42,30,15,-2,-2,-2,1,43,-2,-2,-2],[0.09149999916553497,0.945499986410141,0.5,1.5,-2.0,-2.0,0.9679999947547913,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.22649572789669037,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.003519,0.090732,0.057078,0.106052,0.970817,1.169971,-0.00747,0.564623,-0.96247,0.146221,0.107993,1.074154,0.197271,1.251389,-0.027246,0.010005,-0.013251,-0.192826,0.811828,1.136621,-0.100306,-0.115266,-0.894738,0.129066,0.418477]],[[53,39,3,8,-2,-2,-2,1,-2,-2,38,54,4,-2,-2,-2,42,-2,2,-2,-2],[0.5,0.5,0.9385000169277191,0.9434999823570251,-2.0,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,0.05849999934434891,-2.0,-2.0,-2.0,0.11324786394834518,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,-1,19,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,18,-1,20,-1,-1],[-0.002513,0.074144,0.086099,0.094881,0.755609,1.258087,-0.310624,-0.082351,-1.30328,1.296905,-0.039237,-0.061525,-0.070989,-0.063862,-0.686182,0.74934,0.097262,1.427267,0.001534,0.655334,-1.139692]],[[2,37,7,42,-2,-2,8,-2,-2,3,3,-2,-2,-2,21,32,30,-2,-2,42,-2,-2,-2],[0.9350000023841858,0.5,0.001500000071246177,0.30384616553783417,-2.0,-2.0,0.15150000154972076,-2.0,-2.0,0.625,0.26750001311302185,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.007677,0.072717,0.04705,0.107431,1.058215,1.158279,0.006977,0.707665,-0.406989,0.099395,0.112145,0.8871,1.316158,0.108337,-0.045639,-0.054255,-0.069516,-0.525834,0.502812,0.085754,1.136142,-0.150755,0.875145]],[[2,9,8,42,-2,-2,12,-2,-2,3,12,-2,-2,3,-2,-2,54,31,30,-2,-2,-2,-2],[0.9350000023841858,0.9675000011920929,0.153499998152256,0.3693181872367859,-2.0,-2.0,0.801499992609024,-2.0,-2.0,0.18849999457597733,0.5205000042915344,-2.0,-2.0,0.8585000038146973,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[-0.003765,0.073121,0.046089,0.094708,1.072007,1.259631,-0.023044,-0.589141,0.90251,0.106852,0.089418,0.603088,1.164611,0.153574,1.305307,0.605583,-0.038285,-0.04855,-0.061959,-0.477241,0.542366,0.864552,1.085434]],[[8,4,69,6,-2,-2,-2,44,46,-2,-2,9,-2,-2,42,17,19,-2,-2,-2,10,53,-2,-2,-2],[0.09149999916553497,0.8684999942779541,0.5,0.9839999973773956,-2.0,-2.0,-2.0,0.5,1.5,-2.0,-2.0,0.7894999980926514,-2.0,-2.0,0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,0.4166666716337204,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.004458,0.082295,0.126434,0.093781,0.971743,0.224686,1.145122,0.040987,0.083514,0.909857,1.104311,-0.050908,-0.781094,0.348714,-0.021838,-0.001366,0.009589,0.1575,-0.997978,-1.414246,-0.133592,-0.117754,0.092215,-1.05902,-1.309187]],[[53,39,3,46,-2,-2,4,-2,-2,45,-2,-2,38,54,4,-2,-2,-2,42,-2,2,-2,-2],[0.5,0.5,0.25300000607967377,3.5,-2.0,-2.0,0.2810000032186508,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0,0.5,0.5,0.009499999694526196,-2.0,-2.0,-2.0,0.1819852963089943,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,22,-1,-1],[-0.008281,0.059459,0.071504,0.087758,0.851755,1.267083,0.007388,0.417789,-0.91182,-0.092505,-1.282145,1.137625,-0.039518,-0.058352,-0.068115,0.055781,-0.541968,0.685365,0.07112,0.945177,-0.04438,0.184333,-1.017393]],[[8,5,69,53,-2,-2,-2,-2,42,63,15,-2,-2,-2,53,-2,3,-2,-2],[0.09149999916553497,0.37699998915195465,1.5,0.5,-2.0,-2.0,-2.0,-2.0,0.343137264251709,0.5,0.5,-2.0,-2.0,-2.0,0.5,-2.0,0.04450000077486038,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,15,-1,17,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,16,-1,18,-1,-1],[0.006612,0.07453,0.083539,0.071974,1.098236,0.12927,0.89739,-0.379204,-0.017141,0.00363,-0.009692,-0.180604,0.946511,0.924358,-0.119518,-0.025434,-0.138707,-1.482156,-0.981262]],[[2,9,53,7,-2,-2,-2,9,37,-2,-2,8,-2,-2,38,5,4,-2,-2,7,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.9440000057220459,0.5,0.051500000059604645,-2.0,-2.0,-2.0,0.9884999990463257,0.5,-2.0,-2.0,0.9740000069141388,-2.0,-2.0,0.5,0.9745000004768372,0.07649999856948853,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.22474747896194458,0.8619999885559082,-2.0,-2.0,0.5190000236034393,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.006182,0.063471,0.036044,0.057536,0.964001,-0.415444,-0.564359,0.088106,0.121763,0.696484,1.249783,0.069697,0.49024,1.128011,-0.037742,-0.055673,-0.037829,0.06936,-0.446338,-0.137095,-1.222932,-0.591177,0.049758,0.140308,1.261608,-0.127202,-0.100555,0.13802,-1.547908]],[[19,5,17,16,-2,-2,7,-2,-2,2,42,-2,-2,8,-2,-2,7,-2,2,-2,-2],[0.5,0.6994999945163727,0.5,0.5,-2.0,-2.0,0.005499999970197678,-2.0,-2.0,0.9950000047683716,0.02777777798473835,-2.0,-2.0,0.9945000112056732,-2.0,-2.0,0.010999999940395355,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[0.007113,0.020756,0.047756,0.058911,0.521342,-0.598426,-0.154432,-1.429056,-0.578357,-0.041633,0.00314,1.201894,-0.447451,-0.137554,-1.138498,-0.427358,-0.127086,-0.67061,-0.157269,-1.390212,-1.152668]],[[19,2,3,8,-2,-2,4,-2,-2,42,2,-2,-2,11,-2,-2,6,5,-2,-2,42,-2,-2],[0.5,0.9350000023841858,0.09349999949336052,0.24249999970197678,-2.0,-2.0,0.3255000114440918,-2.0,-2.0,0.02777777798473835,0.9850000143051147,-2.0,-2.0,0.36666667461395264,-2.0,-2.0,0.9275000095367432,0.2919999957084656,-2.0,-2.0,0.18333333730697632,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.005307,0.007498,0.0639,0.079706,0.804043,1.151971,0.019865,0.695358,-0.672899,-0.020418,0.060423,0.974971,-0.479448,-0.050706,-0.457911,0.243648,-0.125554,-0.090604,-0.482321,-1.165138,-0.1554,-0.870743,-1.270761]],[[19,5,16,17,-2,-2,4,-2,-2,12,12,-2,-2,2,-2,-2,4,6,7,-2,-2,-2,-2],[0.5,0.9794999957084656,0.5,0.5,-2.0,-2.0,0.9004999995231628,-2.0,-2.0,0.5264999866485596,0.4259999990463257,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.9695000052452087,0.9810000061988831,0.010999999940395355,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[0.002915,0.016481,0.032092,0.044581,0.377564,-0.647632,-0.116275,-1.283196,0.149867,-0.091559,-0.151534,-0.91368,-1.493405,-0.040964,0.314999,-0.802043,-0.12148,-0.138606,-0.106995,-0.653057,-1.231082,-1.296044,-0.528738]],[[2,12,5,9,-2,-2,-2,8,3,-2,-2,-2,31,54,9,-2,-2,-2,-2],[0.9350000023841858,0.804999977350235,0.6689999997615814,0.9675000011920929,-2.0,-2.0,-2.0,0.08449999988079071,0.0025000000605359674,-2.0,-2.0,-2.0,0.5,0.5,0.945499986410141,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,17,16,-1,-1,-1,-1],[-0.003887,0.055171,0.044627,0.05332,0.173673,0.827812,-0.348013,0.099896,0.069977,1.066423,1.105511,1.165706,-0.033009,-0.046562,-0.057634,-0.748833,-0.201115,1.081412,1.063143]],[[1,42,15,63,-2,-2,12,-2,-2,8,-2,8,-2,-2,46,45,3,-2,-2,4,-2,-2,-2],[0.5,0.4580419659614563,0.5,0.5,-2.0,-2.0,0.5175000131130219,-2.0,-2.0,0.032500000670552254,-2.0,0.20600000023841858,-2.0,-2.0,1.5,0.2814999967813492,0.04150000028312206,-2.0,-2.0,0.19200000166893005,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.002495,-0.020285,-0.008746,-0.020473,-0.226205,0.727171,0.103311,1.07705,0.068134,-0.133844,0.458082,-0.164617,-1.623098,-1.151384,0.070406,0.060466,0.077097,1.079814,1.081013,0.024601,1.026467,-0.548688,1.420313]],[[19,2,7,17,-2,-2,8,-2,-2,8,7,-2,-2,42,-2,-2,9,-2,12,-2,2,-2,-2],[0.5,0.9750000238418579,0.23799999803304672,0.5,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.05250000022351742,0.003999999957159162,-2.0,-2.0,0.11437908560037613,-2.0,-2.0,0.5785000026226044,-2.0,0.4529999941587448,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,22,-1,-1],[0.004922,0.017699,0.048237,0.026154,0.29092,-1.428311,0.143092,1.125964,-0.278576,-0.030378,0.097066,1.280881,0.25871,-0.047746,0.104357,-0.705091,-0.107955,-0.412016,-0.122532,-0.548631,-0.146176,-1.329141,-1.146308]],[[8,4,2,42,-2,-2,7,-2,-2,7,37,-2,-2,-2,65,1,21,-2,-2,3,-2,-2,-2],[0.0885000005364418,0.9159999787807465,0.9350000023841858,0.3693181872367859,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.04800000041723251,0.5,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.34849999845027924,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.00635,0.070369,0.113358,0.075631,1.008349,1.1436,0.156501,1.385495,0.819122,0.023623,0.0423,0.071928,1.034369,-0.863262,-0.015805,-0.025422,-0.040643,-0.336028,0.967569,0.078493,1.163861,0.082405,0.885146]],[[19,16,2,3,-2,-2,15,-2,-2,9,-2,-2,6,12,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.4975000023841858,-2.0,-2.0,0.5,-2.0,-2.0,0.9424999952316284,-2.0,-2.0,0.9810000061988831,0.5380000174045563,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,-1],[0.001079,0.011782,0.021606,0.071113,0.733262,1.012104,-0.004094,-0.12226,0.620358,-0.11668,-1.465331,0.511465,-0.110411,-0.089087,-0.475687,-1.191552,-1.161862]],[[10,15,30,44,-2,-2,-2,-2,17,4,8,-2,-2,7,-2,-2,-2],[0.2678571492433548,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.16099999845027924,0.2590000033378601,-2.0,-2.0,0.02949999924749136,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.000943,-0.030846,-0.04173,-0.056026,-0.170539,-0.642222,0.568814,0.629803,0.037871,0.049958,0.106881,0.704451,1.569853,0.0222,0.630303,-0.486251,-1.015166]],[[19,10,30,44,-2,-2,3,-2,-2,17,9,-2,-2,-2,9,-2,2,-2,-2],[0.5,0.2678571492433548,0.5,0.5,-2.0,-2.0,0.3514999970793724,-2.0,-2.0,0.5,0.9255000054836273,-2.0,-2.0,-2.0,0.652999997138977,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,-1],[0.000645,0.011181,-0.018803,-0.035865,0.00946,-0.524174,0.133598,0.30398,1.184076,0.049437,0.061595,0.257486,0.89717,-1.021598,-0.106366,-0.377778,-0.12097,-1.199214,-1.130923]],[[2,16,46,69,-2,-2,-2,-2,18,1,0,-2,-2,-2,-2],[0.9600000083446503,0.5,1.5,2.5,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,-1],[0.002283,0.037274,0.048898,0.056702,0.354255,1.120658,-1.020653,-0.835474,-0.042972,-0.054508,-0.061279,-0.54435,1.10819,1.444098,0.88188]],[[1,42,4,8,-2,-2,5,-2,-2,8,-2,2,-2,-2,46,3,45,-2,-2,-2,-2],[0.5,0.4580419659614563,0.341499999165535,0.41850000619888306,-2.0,-2.0,0.6545000076293945,-2.0,-2.0,0.06300000101327896,-2.0,0.9600000083446503,-2.0,-2.0,1.5,0.6604999899864197,0.4455000013113022,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[-0.00044,-0.016783,-0.007276,0.017343,0.707316,-0.071445,-0.051479,-0.085723,-1.039835,-0.122042,0.320495,-0.156019,-1.431622,-1.173947,0.066205,0.056992,0.052403,1.039096,0.12691,1.127479,1.445286]],[[5,8,5,8,-2,-2,2,-2,-2,3,-2,-2,42,37,6,-2,-2,4,-2,-2,1,4,-2,-2,10,-2,-2],[0.02850000001490116,0.925000011920929,0.016500000841915607,0.6304999887943268,-2.0,-2.0,0.8250000178813934,-2.0,-2.0,0.9695000052452087,-2.0,-2.0,0.12916667014360428,0.5,0.4584999978542328,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.05849999934434891,-2.0,-2.0,0.3095238208770752,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.003466,0.055727,0.071941,0.053318,0.701611,-0.894844,0.147973,0.663807,1.320397,-0.073656,-1.1079,0.789586,-0.015558,0.026191,-0.045729,-0.988595,0.058761,0.075082,1.069475,0.091318,-0.049539,-0.066482,-0.11005,-0.710891,0.104197,1.25293,1.073123]],[[2,16,17,7,-2,-2,-2,-2,51,46,70,-2,-2,5,-2,-2,-2],[0.9600000083446503,0.5,0.5,0.15399999916553497,-2.0,-2.0,-2.0,-2.0,0.5,1.5,0.5,-2.0,-2.0,0.6850000023841858,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.001685,0.027668,0.038874,0.047399,0.238524,0.842855,-0.789732,-0.914672,-0.033643,-0.045321,-0.06156,-0.631117,0.230766,0.060867,0.984783,-0.804706,0.782387]],[[54,31,2,39,-2,-2,38,-2,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.000498,-0.006684,-0.014577,0.035672,0.472404,-1.209219,-0.040253,-0.410626,0.285946,1.001106,1.253193]],[[54,2,42,7,-2,-2,8,-2,-2,5,8,-2,-2,8,-2,-2,-2],[0.5,0.9750000238418579,0.44949495792388916,0.9354999959468842,-2.0,-2.0,0.060499999672174454,-2.0,-2.0,0.4545000046491623,0.593999981880188,-2.0,-2.0,0.9679999947547913,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.005217,-0.001073,0.024032,0.035119,0.185498,1.269712,-0.128415,0.403795,-1.246268,-0.044275,0.000256,0.444301,-0.829971,-0.075757,-1.14475,-0.150194,1.262784]],[[19,5,7,17,-2,-2,6,-2,-2,2,6,-2,-2,42,-2,-2,12,-2,2,-2,-2],[0.5,0.8260000050067902,0.22999999672174454,0.5,-2.0,-2.0,0.948500007390976,-2.0,-2.0,0.9750000238418579,0.3870000094175339,-2.0,-2.0,0.11437908560037613,-2.0,-2.0,0.4529999941587448,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[-0.001174,0.009743,0.028353,0.015033,0.191724,-1.037285,0.112122,0.375688,1.777969,-0.040652,0.048128,-0.538498,0.844519,-0.076405,-0.111131,-1.143294,-0.101356,-0.610716,-0.121457,-1.309137,-1.113373]],[[42,19,17,42,-2,-2,37,-2,-2,8,-2,-2,8,-2,2,8,-2,-2,37,-2,-2],[0.4580419659614563,0.5,0.5,0.02777777798473835,-2.0,-2.0,0.5,-2.0,-2.0,0.6675000190734863,-2.0,-2.0,0.032500000670552254,-2.0,0.9600000083446503,0.3569999933242798,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,15,16,-1,-1,19,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,18,17,-1,-1,20,-1,-1],[-0.003947,0.004377,0.014103,0.020917,0.529889,0.008182,-0.102704,-1.332454,-0.173772,-0.094989,-0.607119,-1.253064,-0.107395,0.638273,-0.135543,-0.215773,-1.752178,-1.197607,-0.07842,-1.165659,-0.476238]],[[19,42,69,2,-2,-2,4,-2,-2,2,42,-2,-2,1,-2,-2,6,42,-2,-2,-2],[0.5,0.13076923787593842,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.9350000023841858,0.1547619104385376,-2.0,-2.0,0.5,-2.0,-2.0,0.9275000095367432,0.22500000149011612,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.001667,0.012192,0.052333,0.011136,0.418198,-0.681529,0.17763,1.586728,0.296323,-0.015674,0.042217,-0.531709,0.724818,-0.040136,-0.322758,1.393562,-0.105562,-0.081625,-1.175789,-0.903119,-1.19473]],[[10,15,63,53,-2,-2,-2,-2,9,4,6,-2,-2,7,-2,-2,53,42,-2,-2,9,-2,-2],[0.2678571492433548,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.9845000207424164,0.16099999845027924,0.715499997138977,-2.0,-2.0,0.02949999924749136,-2.0,-2.0,0.5,0.40833333134651184,-2.0,-2.0,0.9945000112056732,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.003328,-0.030953,-0.039368,-0.048374,0.130715,-0.482596,0.468893,0.491044,0.032783,0.006452,0.092795,0.458395,1.316912,-0.032351,0.137365,-0.862571,0.107009,0.032399,0.956699,-0.918483,0.204681,1.837131,0.714576]],[[42,69,12,-2,2,-2,-2,-2,2,69,3,-2,-2,-2,15,6,-2,-2,12,-2,-2],[0.02777777798473835,2.5,0.36649999022483826,-2.0,0.9950000047683716,-2.0,-2.0,-2.0,0.9350000023841858,0.5,0.07499999925494194,-2.0,-2.0,-2.0,0.5,0.5590000152587891,-2.0,-2.0,0.5380000174045563,-2.0,-2.0],[1,2,3,-1,5,-1,-1,-1,9,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[8,7,4,-1,6,-1,-1,-1,14,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[-0.000169,0.048121,0.029371,-0.747446,0.045321,0.662933,-0.483507,1.440793,-0.020499,0.033334,0.018747,0.621083,-0.302061,1.170205,-0.04149,-0.054184,-0.023475,-0.615971,0.068906,0.9781,-0.400246]],[[70,44,2,42,-2,-2,46,-2,-2,4,8,-2,-2,9,-2,-2,42,9,12,-2,-2,-2,-2],[0.5,0.5,0.9950000047683716,0.1339285746216774,-2.0,-2.0,1.5,-2.0,-2.0,0.05849999934434891,0.9835000038146973,-2.0,-2.0,0.7764999866485596,-2.0,-2.0,0.25357143580913544,0.9914999902248383,0.5244999825954437,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[0.0001,-0.009982,0.017794,0.042318,1.003524,0.055884,-0.068705,-1.154105,0.957747,-0.051049,0.014295,0.501849,-0.835584,-0.0813,-1.072207,-0.353769,0.074871,0.158377,0.110813,0.464843,1.249488,1.310944,-1.126173]],[[1,32,9,4,-2,-2,42,-2,-2,12,-2,-2,46,6,8,-2,-2,3,-2,-2,8,-2,-2],[0.5,0.5,0.9514999985694885,0.007500000298023224,-2.0,-2.0,0.4365079402923584,-2.0,-2.0,0.4989999979734421,-2.0,-2.0,1.5,0.948500007390976,0.17599999904632568,-2.0,-2.0,0.0819999985396862,-2.0,-2.0,0.04449999891221523,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,22,-1,-1],[-0.013236,-0.027125,-0.038463,-0.086902,0.912319,-0.773997,-0.011323,0.029581,-1.292033,0.077195,0.970909,0.244776,0.043399,0.034084,0.004008,1.087238,-0.445861,0.053811,1.055762,1.151721,0.180009,1.115065,1.454044]],[[42,19,17,16,-2,-2,12,-2,-2,5,-2,9,-2,-2,5,-2,8,-2,3,-2,-2],[0.4580419659614563,0.5,0.5,0.5,-2.0,-2.0,0.4960000216960907,-2.0,-2.0,0.09849999845027924,-2.0,0.9199999868869781,-2.0,-2.0,0.01850000023841858,-2.0,0.20600000023841858,-2.0,0.6790000200271606,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[0.003163,0.011556,0.0203,0.026593,0.275223,-0.541443,-0.088366,0.138926,-1.292605,-0.086555,-1.232953,-0.068942,-0.346561,-1.199023,-0.098211,0.236001,-0.130551,-1.549954,-0.096488,-0.625854,-1.386834]],[[11,8,44,5,-2,-2,-2,15,63,-2,-2,9,-2,-2,17,42,10,-2,-2,-2,-2],[0.20714285969734192,0.2565000057220459,0.5,0.022499999962747097,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9814999997615814,-2.0,-2.0,0.5,0.539393961429596,0.5833333432674408,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[-0.000274,-0.019838,-0.124546,-0.051137,1.142234,-1.147986,-1.512944,-0.012278,-0.021885,-0.280767,0.563826,0.101568,0.265961,1.073346,0.042015,0.050081,0.055529,0.839423,0.218923,-0.911614,-0.7339]],[[42,69,2,69,-2,-2,7,-2,-2,4,-2,-2,2,39,45,-2,-2,-2,9,1,-2,-2,10,-2,-2],[0.02777777798473835,1.5,0.9850000143051147,0.5,-2.0,-2.0,0.5690000057220459,-2.0,-2.0,0.050999999046325684,-2.0,-2.0,0.9350000023841858,0.5,0.5365000069141388,-2.0,-2.0,-2.0,0.9884999990463257,0.5,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[0.000665,0.039246,0.010545,0.034879,0.147614,1.091327,-0.055046,0.304427,-0.896885,0.164026,1.345216,0.581426,-0.015485,0.027605,0.036392,0.595351,-0.411663,-1.037963,-0.034259,-0.0642,-0.555745,1.369254,0.00042,-0.169308,0.955219]],[[51,19,2,46,-2,-2,46,-2,-2,6,-2,2,-2,-2,-2],[0.5,0.5,0.9750000238418579,1.5,-2.0,-2.0,1.5,-2.0,-2.0,0.7094999849796295,-2.0,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.002227,-0.003963,0.00512,0.028005,0.304222,-0.768866,-0.036485,-0.482507,0.73403,-0.089337,-0.67339,-0.112263,-1.229691,-1.118708,0.965753]],[[16,2,17,46,-2,-2,-2,4,6,-2,-2,1,-2,-2,6,-2,-2],[0.5,0.9600000083446503,0.5,1.5,-2.0,-2.0,-2.0,0.08950000256299973,0.9914999902248383,-2.0,-2.0,0.5,-2.0,-2.0,0.9160000085830688,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.000678,0.006067,0.030643,0.03898,0.416692,-0.833897,-0.805167,-0.024944,0.01177,-0.080741,1.079562,-0.062401,-0.645919,1.349726,-0.099869,-1.203458,-0.053838]],[[65,48,53,39,-2,-2,4,-2,-2,12,-2,-2,6,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.08950000256299973,-2.0,-2.0,0.5375000238418579,-2.0,-2.0,0.4455000013113022,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.003159,-0.004404,-0.011744,0.025445,0.475107,-0.801873,-0.033556,0.006938,-0.543415,0.080421,0.96796,-0.274361,0.10461,0.553299,0.929653]],[[42,2,5,11,-2,-2,3,-2,-2,7,-2,4,-2,-2,44,5,3,-2,-2,12,-2,-2,5,7,-2,-2,-2],[0.029411764815449715,0.9850000143051147,0.21199999749660492,0.4166666716337204,-2.0,-2.0,0.061000000685453415,-2.0,-2.0,0.5690000057220459,-2.0,0.031000000424683094,-2.0,-2.0,0.5,0.8980000019073486,0.8760000169277191,-2.0,-2.0,0.4490000009536743,-2.0,-2.0,0.6425000131130219,0.005499999970197678,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[-0.007839,0.030739,0.066102,0.014815,-0.479648,1.123809,0.130578,0.538992,1.209575,-0.048081,0.156247,-0.081024,-1.175688,-0.271134,-0.024712,0.004897,0.03267,0.250995,1.763349,-0.043375,-1.157973,-0.095429,-0.059074,-0.042192,-0.02723,-0.643529,-1.367916]],[[54,38,16,44,-2,-2,9,-2,-2,42,6,-2,-2,3,-2,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.8730000257492065,-2.0,-2.0,0.22474747896194458,0.9839999973773956,-2.0,-2.0,0.003000000142492354,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.006322,0.000818,-0.009807,-0.002309,0.238024,-0.248919,-0.118511,-1.437226,-0.394299,0.053304,0.101702,1.090715,-0.405452,-0.059331,0.519021,-0.86042,1.21896]],[[54,19,42,69,-2,-2,66,-2,-2,6,-2,22,-2,-2,-2],[0.5,0.5,0.11324786394834518,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7094999849796295,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.007848,0.002974,0.011161,0.047566,0.221057,1.503071,-0.010157,-0.017642,-1.594728,-0.077625,-0.635164,-0.095961,-1.310898,-0.877327,1.140633]],[[64,6,3,12,-2,-2,3,-2,-2,5,7,-2,-2,-2,-2],[0.5,0.9575000107288361,0.005499999970197678,0.24249999970197678,-2.0,-2.0,0.009499999694526196,-2.0,-2.0,0.9794999957084656,0.23499999940395355,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[-0.000417,-0.005471,-0.024371,0.031477,1.610187,0.130118,-0.039848,-1.131073,-0.262676,0.026427,0.038301,0.155386,1.197039,-1.121355,0.755789]],[[16,19,5,42,-2,-2,-2,8,8,-2,-2,-2,3,-2,-2],[0.5,0.5,0.9964999854564667,0.02777777798473835,-2.0,-2.0,-2.0,0.6540000140666962,0.2420000061392784,-2.0,-2.0,-2.0,0.0949999988079071,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,14,-1,-1],[0.000725,0.006256,0.013672,0.017954,0.512273,0.018421,-1.238686,-0.066428,-0.029262,-0.868061,0.317175,-1.208532,-0.092996,-0.163979,-1.473719]],[[8,4,3,7,-2,-2,2,-2,-2,12,53,-2,-2,12,-2,-2,42,63,15,-2,-2,-2,12,4,-2,-2,10,-2,-2],[0.09149999916553497,0.8700000047683716,0.005499999970197678,0.02500000037252903,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.47450000047683716,0.5,-2.0,-2.0,0.5760000050067902,-2.0,-2.0,0.343137264251709,0.5,0.5,-2.0,-2.0,-2.0,0.3554999977350235,0.07000000029802322,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[-0.001576,0.044106,0.075371,0.042971,0.26052,1.160424,0.122005,1.064041,1.043204,0.014655,-0.055101,0.682627,-1.204045,0.049417,1.430969,0.128942,-0.017404,-0.005132,-0.016407,-0.245237,0.8556,0.82311,-0.078848,-0.005182,-0.860741,0.654573,-0.100811,-0.94223,-1.774472]],[[54,5,69,3,-2,-2,-2,4,6,-2,-2,4,-2,-2,-2],[0.5,0.02850000001490116,2.5,0.8495000004768372,-2.0,-2.0,-2.0,0.16099999845027924,0.9905000030994415,-2.0,-2.0,0.99549999833107,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[-0.003548,-0.007936,0.032779,0.019589,0.44007,-1.349713,0.847581,-0.023035,0.00389,-0.102634,1.102633,-0.049648,-0.525124,0.530051,0.941751]],[[42,69,38,12,-2,-2,-2,5,2,-2,-2,-2,2,15,2,-2,-2,-2,12,-2,20,-2,-2],[0.12916667014360428,1.5,0.5,0.3490000069141388,-2.0,-2.0,-2.0,0.851500004529953,0.9750000238418579,-2.0,-2.0,-2.0,0.9350000023841858,0.5,0.8250000178813934,-2.0,-2.0,-2.0,0.18150000274181366,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,-1,19,-1,21,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,17,16,-1,-1,-1,20,-1,22,-1,-1],[0.001261,0.031125,-0.000901,-0.021713,-1.408262,-0.046207,1.21889,0.131602,0.070079,0.813496,-0.089928,1.769565,-0.019281,0.028344,0.041887,0.387923,0.887814,-0.779051,-0.038021,0.996367,-0.045319,-1.248086,-0.303214]],[[32,8,11,53,-2,-2,9,-2,-2,16,42,-2,-2,-2,12,-2,12,-2,-2],[0.5,0.09149999916553497,0.36666667461395264,0.5,-2.0,-2.0,0.7730000019073486,-2.0,-2.0,0.5,0.2604166716337204,-2.0,-2.0,-2.0,0.408500000834465,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,-1],[-0.002036,-0.009506,0.032766,-0.01958,1.006298,-0.842266,0.047455,0.081728,0.916465,-0.023468,-0.017683,0.001091,-0.592579,-1.031888,0.069662,1.47499,0.027872,-0.502178,1.098113]],[[2,7,3,7,-2,-2,5,-2,-2,9,-2,42,-2,-2,51,12,4,-2,-2,9,-2,-2,-2],[0.9600000083446503,0.15399999916553497,0.7254999876022339,0.006500000134110451,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.9370000064373016,-2.0,0.3541666716337204,-2.0,-2.0,0.5,0.2565000057220459,0.029000000096857548,-2.0,-2.0,0.20850000530481339,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.001449,0.022455,0.004887,0.020311,0.598722,-0.24211,-0.078505,-1.274112,-0.382464,0.097283,0.097889,0.142347,1.234904,-0.580719,-0.026878,-0.035541,0.080054,-0.573592,1.399916,-0.046269,0.54592,-0.511915,0.720677]],[[54,1,4,37,-2,-2,46,-2,-2,2,45,-2,-2,-2,-2],[0.5,0.5,0.035499999299645424,1.5,-2.0,-2.0,2.5,-2.0,-2.0,0.4749999940395355,0.4455000013113022,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[-0.007299,-0.011096,-0.022127,0.016732,-0.081055,0.654027,-0.045137,-0.3185,-1.321288,0.034039,0.028095,1.000262,-0.116142,1.301951,0.871831]],[[17,2,41,3,-2,-2,-2,32,42,-2,-2,10,-2,-2,4,-2,-2],[0.5,0.9350000023841858,0.5,0.06600000336766243,-2.0,-2.0,-2.0,0.5,0.13076923787593842,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.5965000092983246,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.005935,-0.000332,0.030007,0.023063,0.773962,-0.040925,1.21038,-0.014702,-0.023664,0.068955,-0.390479,0.077693,0.146279,1.355147,-0.106125,-1.420462,-0.444359]],[[19,5,5,70,-2,-2,-2,2,42,-2,-2,7,-2,-2,3,-2,2,-2,-2],[0.5,0.9715000092983246,0.9535000026226044,0.5,-2.0,-2.0,-2.0,0.9850000143051147,0.36666667461395264,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,0.007500000298023224,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[-0.000677,0.006296,0.015634,0.009934,0.006682,0.581896,1.157136,-0.047571,0.021879,0.417063,-1.197938,-0.081394,-0.842508,-1.166497,-0.070625,-0.358175,-0.085172,-1.241646,-1.132726]],[[39,42,19,35,-2,-2,4,-2,-2,42,2,-2,-2,-2,12,-2,-2],[0.5,0.5657142996788025,0.5,0.5,-2.0,-2.0,0.9535000026226044,-2.0,-2.0,0.7321428656578064,0.9750000238418579,-2.0,-2.0,-2.0,0.4544999897480011,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.00105,0.002821,0.006295,0.012352,0.077697,0.923078,-0.064273,-1.120657,-0.190108,-0.129592,-0.088568,-1.286627,-1.068986,-1.846105,-0.114247,0.426567,-1.347748]],[[71,11,4,4,-2,-2,5,-2,-2,10,44,-2,-2,9,-2,-2,-2],[0.5,0.36666667461395264,0.351500004529953,0.24399999529123306,-2.0,-2.0,0.06200000084936619,-2.0,-2.0,0.8333333432674408,0.5,-2.0,-2.0,0.0625,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.000595,0.00236,-0.012067,0.005965,-0.039459,1.0355,-0.045402,0.324208,-0.730898,0.043221,0.075801,0.614442,1.044793,0.004897,1.05033,-0.747049,-0.928324]],[[12,6,9,11,-2,-2,-2,3,-2,-2,66,70,17,-2,-2,42,-2,-2,-2],[0.2580000013113022,0.792000025510788,0.9880000054836273,0.4166666716337204,-2.0,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.25357143580913544,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[0.003774,0.078313,0.17633,0.075873,0.196734,1.351383,2.5145,-0.008075,1.19924,-0.808631,-0.001538,0.002259,-0.005034,0.004494,-1.072701,0.065983,0.886949,-0.565415,-0.728321]],[[16,42,69,6,-2,-2,4,-2,-2,69,12,-2,-2,69,-2,-2,4,-2,-2],[0.5,0.13076923787593842,1.5,0.3095000088214874,-2.0,-2.0,0.054500000551342964,-2.0,-2.0,1.5,0.22949999570846558,-2.0,-2.0,2.5,-2.0,-2.0,0.8720000088214874,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.00433,0.002322,0.02878,0.004327,-0.716917,0.300203,0.107791,1.288146,0.021849,-0.017491,-0.000809,1.226956,-0.09184,-0.067538,-0.914414,-0.124891,-0.095384,-1.316243,-0.268973]],[[35,44,2,8,-2,-2,6,-2,-2,6,-2,11,-2,-2,-2],[0.5,0.5,0.9850000143051147,0.9794999957084656,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.022499999962747097,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.005081,0.001399,0.018698,0.040627,0.299615,1.049439,-0.024566,-0.589589,1.863278,-0.02153,1.416422,-0.030553,-0.396452,0.262124,0.981039]],[[42,19,3,3,-2,-2,-2,6,-2,-2,5,9,3,-2,-2,-2,45,15,-2,-2,-2],[0.33974359929561615,0.5,0.99549999833107,0.992499977350235,-2.0,-2.0,-2.0,0.9874999821186066,-2.0,-2.0,0.006500000134110451,0.9469999969005585,0.0005000000237487257,-2.0,-2.0,-2.0,0.0005000000237487257,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.003928,0.013943,0.020705,0.024104,0.177853,1.351743,-1.077231,-0.062962,-0.691076,-1.095957,-0.04564,0.072039,0.098475,0.632804,1.427932,-0.066155,-0.06662,-0.078329,-0.909915,-0.980485,0.949672]],[[8,3,7,8,-2,-2,16,-2,-2,-2,5,-2,5,-2,56,-2,-2],[0.5940000116825104,0.34300000965595245,0.014500000048428774,0.28450000286102295,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.00849999999627471,-2.0,0.021499999798834324,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,13,-1,15,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,14,-1,16,-1,-1],[0.001426,0.022607,0.016457,0.043652,0.362979,1.032432,-0.023024,-0.068677,-1.328809,1.44402,-0.019473,-1.154226,-0.012631,0.918306,-0.020352,-0.102788,-1.136644]],[[38,17,11,53,-2,-2,9,-2,-2,4,-2,-2,42,6,9,-2,-2,-2,12,-2,4,-2,-2],[0.5,0.5,0.36666667461395264,0.5,-2.0,-2.0,0.6554999947547913,-2.0,-2.0,0.6245000064373016,-2.0,-2.0,0.2361111119389534,0.9855000078678131,0.9605000019073486,-2.0,-2.0,-2.0,0.3804999887943268,-2.0,0.044999999925494194,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,22,-1,-1],[-0.003644,-0.013534,-0.007528,-0.0209,0.211162,-0.282374,0.03303,-0.289623,0.991909,-0.114846,-1.46007,-0.657454,0.043503,0.084712,0.111398,0.989273,1.138772,-0.48145,-0.048772,0.674926,-0.103326,-0.550018,-1.096417]],[[19,42,17,63,-2,-2,9,-2,-2,5,9,-2,-2,1,-2,-2,42,2,-2,-2,-2],[0.5,0.33908046782016754,0.5,0.5,-2.0,-2.0,0.9650000035762787,-2.0,-2.0,0.006500000134110451,0.8330000042915344,-2.0,-2.0,0.5,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.002528,0.009403,0.017787,0.022738,0.164266,0.615154,-0.067712,-1.255528,0.041027,-0.034021,0.081679,1.267064,0.655184,-0.053437,-0.747905,1.270738,-0.062863,-0.079743,-1.144967,-1.068678,-0.69246]],[[5,3,44,2,-2,-2,6,-2,-2,8,-2,-2,12,6,8,-2,-2,-2,2,-2,-2],[0.9794999957084656,0.9895000159740448,0.5,0.9950000047683716,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.7409999966621399,0.9699999988079071,0.9774999916553497,-2.0,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,-1],[0.000724,0.00853,0.004818,0.027353,0.590734,-0.491091,-0.018835,1.291374,-0.221236,0.119299,1.30271,-0.464173,-0.051629,-0.082809,-0.062722,-1.244624,-0.397103,-1.342125,0.005767,0.313233,-1.091488]],[[19,42,5,6,-2,-2,4,-2,-2,5,-2,8,-2,-2,6,-2,2,-2,-2],[0.5,0.4580419659614563,0.6545000076293945,0.024000000208616257,-2.0,-2.0,0.8224999904632568,-2.0,-2.0,0.019499999471008778,-2.0,0.210999995470047,-2.0,-2.0,0.7094999849796295,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,-1],[-0.000779,0.004958,0.010531,0.022181,1.595489,0.122559,-0.015816,0.040106,-1.331622,-0.06767,0.411515,-0.097527,-1.261924,-0.997051,-0.060209,-0.439094,-0.080182,-1.081617,-1.141705]],[[10,63,44,6,-2,-2,5,-2,-2,-2,4,8,8,-2,-2,-2,9,4,-2,-2,44,-2,-2],[0.2678571492433548,0.5,0.5,0.7740000188350677,-2.0,-2.0,0.7580000162124634,-2.0,-2.0,-2.0,0.16899999976158142,0.2590000033378601,0.1679999977350235,-2.0,-2.0,-2.0,0.9465000033378601,0.8574999868869781,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[-0.000745,-0.018134,-0.025279,-0.000854,-0.374875,0.440484,-0.056714,-0.370749,-1.416129,0.50725,0.022283,0.060562,0.042724,0.854572,-0.345928,1.155138,0.004367,-0.027749,-0.992343,0.2885,0.050554,0.23636,0.805685]],[[8,8,3,6,-2,-2,-2,-2,37,42,63,-2,-2,12,-2,-2,-2],[0.0925000011920929,0.08249999955296516,0.048499999567866325,0.7590000033378601,-2.0,-2.0,-2.0,-2.0,4.5,0.22649572789669037,0.5,-2.0,-2.0,0.6024999916553497,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.000376,0.033312,0.026777,0.020337,0.817777,0.01329,0.859721,1.370223,-0.012179,-0.00752,0.008301,-0.019542,0.841429,-0.037791,-0.093963,-1.060985,-0.844349]],[[44,2,4,6,-2,-2,7,-2,-2,5,11,-2,-2,8,-2,-2,6,-2,15,63,-2,-2,3,-2,-2],[0.5,0.9950000047683716,0.4389999955892563,0.9824999868869781,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5449999868869781,0.2678571492433548,-2.0,-2.0,0.9945000112056732,-2.0,-2.0,0.016999999526888132,-2.0,0.5,0.5,-2.0,-2.0,0.030500000342726707,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,24,-1,-1],[-0.001103,0.017233,0.032236,0.060659,0.42869,1.815952,-0.002916,1.008603,-0.438373,-0.042347,0.083707,1.186278,0.979938,-0.070047,-1.067172,0.112638,-0.024362,1.225698,-0.032861,-0.044372,-0.454723,0.497502,0.051888,1.098518,-0.361972]],[[12,8,5,42,-2,-2,-2,-2,8,9,4,-2,-2,7,-2,-2,3,69,-2,-2,69,-2,-2],[0.2580000013113022,0.38449999690055847,0.13350000232458115,0.2678571492433548,-2.0,-2.0,-2.0,-2.0,0.6304999887943268,0.7695000171661377,0.060499999672174454,-2.0,-2.0,0.887499988079071,-2.0,-2.0,0.007500000298023224,0.5,-2.0,-2.0,2.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.004028,0.072895,0.008279,0.068738,1.456838,0.325641,-0.924073,1.482392,-0.009513,0.013345,-0.019905,1.053725,-0.583939,0.043144,0.570442,-0.847516,-0.031892,-0.121503,-0.599215,-1.259977,-0.021751,-0.339908,0.468175]],[[42,17,8,8,-2,-2,30,-2,-2,42,-2,-2,-2],[0.7321428656578064,0.5,0.6324999928474426,0.3410000056028366,-2.0,-2.0,0.5,-2.0,-2.0,0.3229166716337204,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.004028,0.005566,0.009799,0.027917,0.17157,0.775404,-0.008241,-0.173772,0.648776,-0.073389,-1.007996,0.198745,-1.778404]],[[5,44,2,7,-2,-2,46,-2,-2,12,9,-2,-2,-2,2,-2,42,-2,12,-2,-2],[0.9895000159740448,0.5,0.9950000047683716,0.9505000114440918,-2.0,-2.0,1.5,-2.0,-2.0,0.8770000040531158,0.3854999989271164,-2.0,-2.0,-2.0,0.9750000238418579,-2.0,0.19090909510850906,-2.0,0.5705000162124634,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,20,-1,-1],[-0.001494,0.004295,0.024845,0.040319,0.361445,1.471923,-0.039416,-1.051875,0.979342,-0.016857,-0.026302,-0.769011,-0.122865,1.074229,-0.059002,0.106892,-0.081586,-1.060676,-0.058644,-1.143619,-1.059129]],[[71,43,10,5,-2,-2,4,-2,-2,8,-2,-2,-2],[0.5,0.5,0.2361111119389534,0.00849999999627471,-2.0,-2.0,0.5660000145435333,-2.0,-2.0,0.45000000298023224,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.001301,0.0038,-0.001116,-0.01874,-0.897313,-0.106394,0.022254,0.60188,-0.006714,0.072368,-0.198883,1.228782,-0.749029]],[[19,2,7,46,-2,-2,42,-2,-2,10,42,-2,-2,7,-2,-2,42,2,-2,-2,-2],[0.5,0.9750000238418579,0.2775000035762787,1.5,-2.0,-2.0,0.44949495792388916,-2.0,-2.0,0.3541666716337204,0.11437908560037613,-2.0,-2.0,0.027500000782310963,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.003383,0.010146,0.026735,0.010619,0.171958,-0.851872,0.108276,1.031434,-1.411746,-0.017475,-0.034327,0.09629,-0.772287,0.044568,1.096871,-0.111868,-0.06389,-0.077971,-1.117261,-1.087281,-0.833746]],[[69,42,12,6,-2,-2,69,-2,-2,-2,-2],[4.5,0.7321428656578064,0.2580000013113022,0.792000025510788,-2.0,-2.0,2.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.004124,-0.001038,0.000262,0.063732,1.430939,-0.161363,-0.0044,-0.111863,0.437433,-1.82702,-0.710712]],[[44,16,2,7,-2,-2,6,-2,-2,-2,5,69,5,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.9750000238418579,0.9284999966621399,-2.0,-2.0,0.99549999833107,-2.0,-2.0,-2.0,0.6425000131130219,2.5,0.382999986410141,-2.0,-2.0,0.17914438992738724,-2.0,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,-1],[-0.005912,0.008783,0.014291,0.04119,0.459618,0.933426,-0.0242,-0.481852,1.002722,-1.283755,-0.025282,-0.014065,-0.029587,-0.365876,0.633214,0.066314,1.094853,-0.460199,-0.114745,-1.419955,0.24277]],[[42,16,12,42,-2,-2,42,-2,-2,12,-2,-2,10,7,-2,-2,-2],[0.5857143104076385,0.5,0.21249999850988388,0.3166666775941849,-2.0,-2.0,0.18465909361839294,-2.0,-2.0,0.5900000035762787,-2.0,-2.0,0.4166666716337204,0.9745000004768372,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,-1],[-0.000198,0.002425,0.006625,0.114282,1.74846,0.358852,0.003203,0.197358,-0.189671,-0.06403,-0.064825,-1.389466,-0.098878,-0.064932,-0.793309,-1.619199,-1.608491]],[[42,2,7,8,-2,-2,-2,9,9,-2,-2,4,-2,-2,42,-2,5,16,-2,-2,2,-2,-2],[0.02777777798473835,0.9850000143051147,0.5754999816417694,0.9824999868869781,-2.0,-2.0,-2.0,0.9934999942779541,0.9799999892711639,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.10435571894049644,-2.0,0.9745000004768372,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[-0.006687,0.022421,0.046767,0.026796,0.50945,-0.719935,1.387152,-0.032872,-0.071434,-0.668168,-1.295736,0.019573,-0.979423,0.595434,-0.018529,-0.998454,-0.012385,-0.002926,0.030778,-1.346494,-0.075552,-1.019184,-1.116193]],[[66,70,16,6,-2,-2,42,-2,-2,42,32,-2,-2,-2,69,-2,-2],[0.5,0.5,0.5,0.4584999978542328,-2.0,-2.0,0.18333333730697632,-2.0,-2.0,0.2613636404275894,0.5,-2.0,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.004209,-0.001055,-0.00686,-0.001552,-0.366424,0.114292,-0.079885,-1.240903,-0.263164,0.050344,0.098713,1.296866,0.520038,-0.697445,-0.083721,-0.058964,-1.377779]],[[16,4,4,37,-2,-2,-2,69,12,-2,-2,5,-2,-2,4,-2,-2],[0.5,0.05849999934434891,0.05049999989569187,1.5,-2.0,-2.0,-2.0,1.5,0.2070000022649765,-2.0,-2.0,0.6899999976158142,-2.0,-2.0,0.8650000095367432,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.001196,0.003316,0.026777,0.017125,-0.08543,0.660941,1.295298,-0.01217,0.004751,2.254601,-0.00243,-0.064412,-0.183664,-1.091802,-0.074607,-1.268345,-0.028443]],[[4,6,12,4,-2,-2,12,-2,-2,5,10,-2,-2,42,-2,-2,7,69,10,-2,-2,3,-2,-2,12,4,-2,-2,-2],[0.351500004529953,0.4580000042915344,0.4884999990463257,0.02350000012665987,-2.0,-2.0,0.6270000040531158,-2.0,-2.0,0.2030000016093254,0.22500000149011612,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.007500000298023224,0.5,0.21111111342906952,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.8289999961853027,0.5220000147819519,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,-1],[0.003616,0.018859,-0.010865,0.053198,-0.270163,1.400091,-0.031079,-1.093998,-0.063308,0.040406,-0.015626,-0.490814,0.357248,0.078796,1.049232,0.293369,-0.017926,0.016117,0.047331,1.232268,0.632863,-0.058736,-1.476925,0.121518,-0.049865,-0.067602,-1.461367,-0.561782,0.941036]],[[64,54,9,5,-2,-2,63,-2,-2,-2,-2],[0.5,0.5,0.945499986410141,0.8040000200271606,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.008252,0.004211,0.000795,-0.024261,-0.109658,-1.092553,0.016763,0.079022,1.011069,0.891688,1.039029]],[[42,69,6,38,-2,-2,5,-2,-2,4,-2,-2,2,46,7,-2,-2,-2,1,0,-2,-2,-2],[0.11324786394834518,1.5,0.9905000030994415,0.5,-2.0,-2.0,0.07249999791383743,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.9750000238418579,1.5,0.17000000178813934,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,-1],[0.001765,0.024886,0.006069,-0.009198,-0.297236,1.159137,0.065735,-0.329706,1.305499,0.096154,1.312469,0.159859,-0.011056,0.007026,0.015702,0.003765,0.597758,-1.070884,-0.041098,-0.04713,-0.70315,0.915031,1.388299]],[[71,42,52,5,-2,-2,6,-2,-2,5,-2,-2,-2],[0.5,0.612500011920929,0.5,0.6489999890327454,-2.0,-2.0,0.9810000061988831,-2.0,-2.0,0.7070000171661377,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.000377,0.002569,0.004641,0.009342,0.206486,-0.140195,-0.044664,-0.62757,-1.016017,-0.106037,-1.283398,-1.66775,-0.871504]],[[11,44,3,8,-2,-2,6,-2,-2,3,9,-2,-2,-2,9,4,7,-2,-2,4,-2,-2,44,13,-2,-2,12,-2,-2],[0.36666667461395264,0.5,0.9835000038146973,0.9794999957084656,-2.0,-2.0,0.148499995470047,-2.0,-2.0,0.9895000159740448,0.9934999942779541,-2.0,-2.0,-2.0,0.7730000019073486,0.09699999913573265,0.004500000039115548,-2.0,-2.0,0.6344999969005585,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5164999961853027,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.001296,-0.010688,0.007441,0.018651,-0.011755,0.757303,-0.076368,-0.719707,-1.320442,-0.034286,-0.041985,-0.157895,-0.843335,0.652704,0.025981,-0.014614,0.066602,1.514685,0.414578,-0.046529,-1.638682,-0.278845,0.067047,0.018398,0.656038,-1.380307,0.128754,0.688666,1.415114]],[[6,70,6,44,-2,-2,44,-2,-2,42,9,-2,-2,-2,12,4,-2,69,-2,-2,12,3,-2,-2,2,-2,-2],[0.9575000107288361,0.5,0.02850000001490116,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.2613636404275894,0.9799999892711639,-2.0,-2.0,-2.0,0.5024999976158142,0.017500000074505806,-2.0,0.5,-2.0,-2.0,0.6270000040531158,0.21699999272823334,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.001949,-0.01475,-0.024502,0.031394,-0.199301,1.362879,-0.034864,-0.168085,-0.506819,0.051753,0.090702,0.58003,1.006582,-0.83163,0.019409,-0.013409,0.76552,-0.027259,0.127347,-0.799986,0.059711,0.127781,0.841953,2.080023,0.008397,0.915459,-0.911892]],[[5,6,3,12,-2,-2,-2,3,69,-2,-2,6,-2,-2,4,2,3,-2,-2,4,-2,-2,-2],[0.6545000076293945,0.014500000048428774,0.2435000017285347,0.601500004529953,-2.0,-2.0,-2.0,0.7045000195503235,2.5,-2.0,-2.0,0.13350000232458115,-2.0,-2.0,0.9510000050067902,0.9950000047683716,0.8039999902248383,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.00518,0.01848,0.10497,0.030798,1.189058,0.06637,1.679401,0.013621,0.024246,0.200413,0.7678,-0.045082,-1.556373,-0.202617,-0.024557,-0.013302,0.010962,-0.257392,0.542423,-0.058653,-1.182149,-0.686041,-1.315585]],[[3,44,6,2,-2,-2,3,-2,-2,11,8,-2,-2,9,-2,-2,7,-2,-2],[0.99549999833107,0.5,0.9784999787807465,0.9850000143051147,-2.0,-2.0,0.22449999302625656,-2.0,-2.0,0.4166666716337204,0.2824999988079071,-2.0,-2.0,0.9784999787807465,-2.0,-2.0,0.9560000002384186,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.001866,0.000967,0.015194,0.000372,0.243988,-0.52419,0.052819,0.099329,1.841103,-0.016301,-0.036892,-1.127968,-0.170906,0.03604,-0.058232,1.124434,-0.125338,-1.472003,-1.261049]],[[67,3,4,4,-2,-2,4,-2,-2,-2,-2],[0.5,0.9964999854564667,0.05849999934434891,0.051500000059604645,-2.0,-2.0,0.06750000268220901,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002509,-0.000382,0.001367,0.020394,0.117066,1.5804,-0.010315,-1.433957,-0.085726,-1.471129,0.784743]],[[38,44,6,3,-2,-2,3,-2,-2,6,-2,6,-2,-2,46,42,8,-2,-2,3,-2,-2,-2],[0.5,0.5,0.9794999957084656,0.25300000607967377,-2.0,-2.0,0.22449999302625656,-2.0,-2.0,0.02850000001490116,-2.0,0.24050000309944153,-2.0,-2.0,2.5,0.2361111119389534,0.04350000061094761,-2.0,-2.0,0.012500000186264515,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.004155,-0.012466,0.003523,-0.015255,0.143299,-0.509457,0.050447,0.211568,1.703913,-0.032781,0.901194,-0.039356,-1.123736,-0.235051,0.034841,0.047032,0.081764,0.605289,1.011879,-0.028869,0.694562,-0.805123,-1.053059]],[[42,51,46,63,-2,-2,69,-2,-2,-2,5,3,-2,-2,45,2,-2,-2,-2],[0.33908046782016754,0.5,2.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,-2.0,0.006500000134110451,0.003999999957159162,-2.0,-2.0,0.0005000000237487257,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,-1,15,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,14,13,-1,-1,18,17,-1,-1,-1],[-0.00447,0.003639,-0.001321,0.002116,-0.03314,0.664386,-0.113116,0.499482,-1.634366,0.901341,-0.044971,0.055023,0.481032,1.204895,-0.062592,-0.072693,-1.07683,-0.883175,0.796538]],[[71,42,12,6,-2,-2,12,-2,-2,-2,-2],[0.5,0.7321428656578064,0.2580000013113022,0.792000025510788,-2.0,-2.0,0.3110000044107437,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-7.8e-05,0.002427,0.003636,0.06326,1.14164,0.177389,-0.000837,-0.854408,0.049401,-1.695669,-0.919897]],[[44,6,2,69,-2,-2,5,-2,-2,3,2,-2,-2,-2,6,-2,38,6,-2,-2,42,-2,-2],[0.5,0.9794999957084656,0.9850000143051147,0.5,-2.0,-2.0,0.5455000102519989,-2.0,-2.0,0.22449999302625656,0.9750000238418579,-2.0,-2.0,-2.0,0.02850000001490116,-2.0,0.5,0.2410000041127205,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[0.000142,0.013687,-0.002143,0.01953,-0.076551,0.675494,-0.037058,0.603219,-0.872341,0.050148,0.015565,1.053427,-0.681054,1.444437,-0.017699,0.830484,-0.023496,-0.037717,-1.14808,-0.230076,0.041796,0.761661,-0.66741]],[[10,7,6,6,-2,-2,5,-2,-2,2,42,-2,-2,6,-2,-2,9,32,12,-2,-2,-2,8,5,-2,-2,9,-2,-2],[0.2678571492433548,0.048499999567866325,0.24449999630451202,0.04950000159442425,-2.0,-2.0,0.6570000052452087,-2.0,-2.0,0.9850000143051147,0.44949495792388916,-2.0,-2.0,0.9810000061988831,-2.0,-2.0,0.9845000207424164,0.5,0.94200000166893,-2.0,-2.0,-2.0,0.06149999983608723,0.032500000670552254,-2.0,-2.0,0.9945000112056732,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.004593,-0.008282,-0.036359,-0.13328,-0.133361,-1.59938,-0.022189,-0.073949,-1.150195,0.012987,0.051054,0.559333,-1.014098,-0.023516,-0.746238,0.775381,0.021308,0.005788,-0.005487,-0.151773,1.595928,1.131957,0.063696,0.011466,1.088235,-0.603588,0.111407,1.434314,0.668361]],[[3,4,6,6,-2,-2,48,-2,-2,4,3,-2,-2,6,-2,-2,42,-2,-2],[0.99549999833107,0.341499999165535,0.2509999945759773,0.1524999961256981,-2.0,-2.0,0.5,-2.0,-2.0,0.5189999938011169,0.08249999955296516,-2.0,-2.0,0.4285000115633011,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.002652,0.0049,0.017848,-0.013233,0.049583,-1.107578,0.032592,0.244596,0.931983,-0.012551,-0.113044,-0.533038,-1.497208,0.001515,1.107266,-0.135717,-0.089142,-1.168258,-0.833277]],[[42,51,72,19,-2,-2,-2,-2,5,12,9,-2,-2,-2,5,-2,2,-2,-2],[0.343137264251709,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.019499999471008778,0.5595000088214874,0.8145000040531158,-2.0,-2.0,-2.0,0.05250000022351742,-2.0,0.875,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,15,-1,17,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,16,-1,18,-1,-1],[0.002304,0.009588,0.005241,0.001939,0.080367,-1.073274,0.845115,0.864558,-0.031044,0.036176,0.072492,1.304402,0.920302,-0.592543,-0.050714,-1.18224,-0.03382,1.259896,-0.802249]],[[69,16,2,39,-2,-2,42,-2,-2,9,-2,-2,-2],[4.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.2695630192756653,-2.0,-2.0,0.9595000147819519,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.001344,0.003767,0.007383,0.033056,0.756224,-0.831217,-0.004295,0.08886,-0.40753,-0.045131,-0.884267,0.609569,-0.80341]],[[11,44,6,3,-2,-2,3,-2,-2,5,8,-2,-2,42,-2,-2,7,53,6,-2,-2,8,-2,-2,12,9,-2,-2,7,-2,-2],[0.4166666716337204,0.5,0.9779999852180481,0.3034999966621399,-2.0,-2.0,0.21699999272823334,-2.0,-2.0,0.6834999918937683,0.24199999868869781,-2.0,-2.0,0.15142857283353806,-2.0,-2.0,0.004500000039115548,0.5,0.8179999887943268,-2.0,-2.0,0.04450000077486038,-2.0,-2.0,0.7240000069141388,0.35100001096725464,-2.0,-2.0,0.035499999299645424,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[0.000183,-0.011049,0.006858,-0.00947,0.310449,-0.420738,0.060595,-0.045025,1.569909,-0.033387,-0.019091,-0.857465,-0.017419,-0.13593,-1.418013,-1.162959,0.031436,0.061778,0.022046,1.050731,1.027116,0.125072,0.618281,1.245873,0.00011,-0.019415,-1.067829,0.194657,0.043531,0.97048,0.774648]],[[17,42,51,46,-2,-2,-2,8,53,-2,-2,8,-2,-2,42,-2,-2],[0.5,0.33908046782016754,0.5,2.5,-2.0,-2.0,-2.0,0.06300000101327896,0.5,-2.0,-2.0,0.14550000429153442,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[0.002317,0.00607,0.012774,0.007224,0.112964,-0.735843,1.057001,-0.025456,0.045987,0.440357,0.846432,-0.048158,-1.26869,-0.650893,-0.076568,-1.092448,-0.006481]],[[48,5,56,19,-2,-2,4,-2,-2,9,-2,6,-2,-2,7,-2,-2],[0.5,0.992499977350235,0.5,0.5,-2.0,-2.0,0.632999986410141,-2.0,-2.0,0.9945000112056732,-2.0,0.26900000870227814,-2.0,-2.0,0.015000000596046448,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,-1],[0.003711,-0.000421,0.00423,0.008022,0.145104,-0.727423,-0.065665,-0.016968,-1.259452,-0.05621,0.074614,-0.07486,-0.489764,-1.056616,0.061188,0.986182,0.021295]],[[3,44,3,7,-2,-2,2,-2,-2,6,-2,6,-2,-2,4,2,-2,-2,-2],[0.9905000030994415,0.5,0.8440000116825104,0.7400000095367432,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,0.02850000001490116,-2.0,0.24050000309944153,-2.0,-2.0,0.0025000000605359674,0.9650000035762787,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,-1],[-0.002869,0.000367,0.015215,0.005981,0.449067,-0.24926,0.060961,1.15468,-1.008629,-0.019195,1.047139,-0.026353,-1.022788,-0.145262,-0.069337,-0.105937,-1.288533,-0.894374,-0.06339]],[[5,69,6,7,-2,-2,44,-2,-2,42,-2,-2,4,4,9,-2,-2,3,-2,-2,9,-2,-2],[0.625,2.5,0.009499999694526196,0.035499999299645424,-2.0,-2.0,0.5,-2.0,-2.0,0.2182539701461792,-2.0,-2.0,0.828499972820282,0.004500000039115548,0.9975000023841858,-2.0,-2.0,0.8765000104904175,-2.0,-2.0,0.9619999825954437,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,22,-1,-1],[-0.003478,0.006111,-0.000218,0.091551,0.742824,1.720352,-0.005308,0.370872,-0.237154,0.069432,0.888433,-0.251365,-0.025578,-0.009881,-0.06448,-0.757095,-1.425166,0.006694,-0.238075,0.692739,-0.123102,-0.886993,-1.38505]],[[38,3,9,44,-2,-2,2,-2,-2,6,69,-2,-2,3,-2,-2,35,6,8,-2,-2,8,-2,-2,-2],[0.5,0.001500000071246177,0.9745000004768372,0.5,-2.0,-2.0,0.9350000023841858,-2.0,-2.0,0.7400000095367432,2.5,-2.0,-2.0,0.9200000166893005,-2.0,-2.0,0.5,0.6420000195503235,0.1274999976158142,-2.0,-2.0,0.6324999928474426,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[-0.0009,-0.007001,0.026976,-0.003296,0.610305,-0.49742,0.074718,1.067938,1.108108,-0.013459,-0.033503,-0.460552,0.565731,0.001993,-0.115073,0.550029,0.02938,0.01719,0.056111,-0.235703,1.080336,-0.015075,0.530472,-0.61306,1.0403]],[[8,9,69,16,-2,-2,5,-2,-2,18,6,-2,-2,-2,12,9,5,-2,-2,6,-2,-2,7,-2,42,-2,-2],[0.9034999907016754,0.8619999885559082,1.5,0.5,-2.0,-2.0,0.10949999839067459,-2.0,-2.0,0.5,0.9985000193119049,-2.0,-2.0,-2.0,0.7384999990463257,0.9905000030994415,0.48649999499320984,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,0.015500000212341547,-2.0,0.03125,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,26,-1,-1],[-0.003213,0.00753,-0.013019,0.002053,0.193628,-0.959391,-0.08228,-1.294864,-0.185292,0.023255,0.016392,0.269425,-1.0621,1.221349,-0.024944,-0.008387,-0.070711,0.13592,-1.15064,0.015601,-0.078208,0.654292,-0.066232,-1.319638,-0.034163,0.184451,-0.892099]],[[51,8,4,8,-2,-2,42,-2,-2,3,12,-2,-2,5,-2,-2,8,-2,-2],[0.5,0.17599999904632568,0.8669999837875366,0.10700000077486038,-2.0,-2.0,0.2198067605495453,-2.0,-2.0,0.5065000057220459,0.8314999938011169,-2.0,-2.0,0.239499993622303,-2.0,-2.0,0.3425000011920929,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.000985,-0.002676,0.020214,0.038516,0.478594,1.090949,0.00147,0.582409,-0.728382,-0.013022,-0.032118,-0.438653,0.745893,0.008789,-0.356637,0.339675,0.068715,-0.151912,1.320337]],[[17,42,9,9,-2,-2,12,-2,-2,66,4,-2,-2,-2,37,-2,-2],[0.5,0.02777777798473835,0.9964999854564667,0.992499977350235,-2.0,-2.0,0.5540000200271606,-2.0,-2.0,0.5,0.05849999934434891,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.003562,-0.000194,0.024304,0.036045,0.267382,0.822531,-0.038315,0.081028,-0.958453,-0.010515,-0.005921,0.190346,-0.238828,-1.143761,-0.067817,-1.110759,-0.232588]],[[51,16,44,2,-2,-2,6,-2,-2,6,-2,-2,8,-2,-2],[0.5,0.5,0.5,0.9850000143051147,-2.0,-2.0,0.009499999694526196,-2.0,-2.0,0.875,-2.0,-2.0,0.35750000178813934,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.001736,-0.001809,0.002646,0.014895,0.496424,-0.27869,-0.01565,1.339587,-0.202004,-0.071797,-1.401743,-0.336334,0.063569,0.125038,0.834771]],[[9,9,53,12,-2,-2,-2,48,32,-2,-2,-2,2,12,48,-2,-2,69,-2,-2,10,3,-2,-2,-2],[0.945499986410141,0.05849999934434891,0.5,0.6464999914169312,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.9600000083446503,0.6534999907016754,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4166666716337204,0.8735000193119049,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[0.000533,-0.016975,0.037254,0.020683,1.0643,1.028593,1.111749,-0.029791,-0.021066,-0.404866,0.784209,-0.813582,0.012169,0.034775,0.011223,-0.011773,1.048984,0.100021,0.386896,1.030007,-0.016145,-0.027573,-0.748078,0.098518,0.545815]],[[2,40,69,6,-2,-2,12,-2,-2,-2,37,38,0,-2,-2,-2,10,42,-2,-2,-2],[0.9750000238418579,0.5,2.5,0.7894999980926514,-2.0,-2.0,0.56700000166893,-2.0,-2.0,-2.0,1.5,0.5,0.5,-2.0,-2.0,-2.0,0.2678571492433548,0.11678004637360573,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[-0.002744,0.008614,0.003214,-0.003886,-0.317085,0.188946,0.056812,-0.103524,1.32003,1.636611,-0.021804,-0.039022,-0.048233,-0.777327,1.101155,0.452411,0.016405,-0.0119,0.550478,-0.950853,0.934308]],[[12,8,5,42,-2,-2,-2,-2,12,42,4,-2,-2,6,-2,-2,63,8,-2,-2,-2],[0.2580000013113022,0.6084999740123749,0.1510000005364418,0.25833334028720856,-2.0,-2.0,-2.0,-2.0,0.3489999920129776,0.24038461595773697,0.3464999943971634,-2.0,-2.0,0.9645000100135803,-2.0,-2.0,0.5,0.848499983549118,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[-0.003385,0.04617,0.00712,0.075137,1.350905,0.53854,-0.972108,1.324872,-0.007059,-0.055607,-0.104684,-0.620434,-1.545292,0.012717,-0.534463,0.850231,-0.000628,-0.00501,0.20955,-0.347518,0.679041]],[[42,54,17,19,-2,-2,7,-2,-2,-2,15,2,8,-2,-2,5,-2,-2,-2],[0.33908046782016754,0.5,0.5,0.5,-2.0,-2.0,0.01699999999254942,-2.0,-2.0,-2.0,0.5,0.9249999821186066,0.036000000312924385,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[0.001117,0.0081,0.004539,0.008895,0.158552,-0.849313,-0.06697,-1.312686,-0.349783,1.178797,-0.031743,-0.020215,0.043966,0.6548,0.874022,-0.038662,1.501582,-0.906458,-0.987315]],[[17,10,63,41,-2,-2,-2,9,12,-2,-2,42,-2,-2,4,-2,-2],[0.5,0.2678571492433548,0.5,0.5,-2.0,-2.0,-2.0,0.45350000262260437,0.7354999780654907,-2.0,-2.0,0.3452381044626236,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.003053,0.000278,-0.013682,-0.019318,-0.256102,0.531342,0.42397,0.018439,-0.008887,-0.54214,0.68421,0.037457,0.713627,-0.192489,-0.07264,-1.42065,-0.220054]],[[2,69,17,5,-2,-2,-2,6,-2,-2,4,3,8,-2,-2,6,-2,-2,1,6,-2,-2,-2],[0.9600000083446503,2.5,0.5,0.2135000005364418,-2.0,-2.0,-2.0,0.9290000200271606,-2.0,-2.0,0.07649999856948853,0.00849999999627471,0.010499999858438969,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,0.5,0.910999983549118,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.001198,0.016075,0.007751,0.013194,-0.015367,0.45094,-1.069856,0.086749,1.092999,-0.081153,-0.018644,0.001302,0.083102,-0.282191,0.934874,-0.017193,-0.478855,0.550494,-0.036242,-0.041818,-0.158515,-0.888954,1.200422]],[[12,1,5,69,-2,-2,4,-2,-2,46,11,-2,-2,-2,8,37,2,-2,-2,3,-2,-2,-2],[0.8680000007152557,0.5,0.02850000001490116,2.5,-2.0,-2.0,0.8039999902248383,-2.0,-2.0,3.5,0.45000000298023224,-2.0,-2.0,-2.0,0.9914999902248383,0.5,0.9249999821186066,-2.0,-2.0,0.051500000059604645,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.00093,-0.003253,-0.01038,0.020838,0.048658,0.963468,-0.019054,-0.08591,-0.65128,0.025559,0.020753,0.701624,0.418928,1.229078,0.036828,0.018758,-0.037604,1.065836,-0.959205,0.06855,0.557581,1.13464,1.077772]],[[39,9,69,9,-2,-2,3,-2,-2,48,63,-2,-2,-2,5,-2,-2],[0.5,0.945499986410141,1.5,0.9165000021457672,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.007999999914318323,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.000826,0.00233,-0.014518,0.00089,0.198246,-0.946733,-0.083011,-1.117193,-0.590204,0.012799,0.005933,-0.004984,0.782528,0.947065,-0.104587,0.936175,-1.124302]],[[5,3,51,4,-2,-2,-2,7,6,-2,-2,-2,3,6,-2,1,-2,-2,6,37,-2,-2,3,-2,-2],[0.6044999957084656,0.6725000143051147,0.5,0.06350000202655792,-2.0,-2.0,-2.0,0.10949999839067459,0.4880000054836273,-2.0,-2.0,-2.0,0.7330000102519989,0.12350000441074371,-2.0,0.5,-2.0,-2.0,0.698500007390976,1.5,-2.0,-2.0,0.968500018119812,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,15,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.000393,0.008519,0.018221,0.013375,0.574427,0.022437,0.785504,-0.041486,-0.062208,-0.091111,-0.908078,0.59969,-0.018577,-0.049031,0.656418,-0.065429,-0.88308,0.303893,0.018136,-0.029419,-0.871588,0.308362,0.09211,1.294018,0.089552]],[[5,19,9,42,-2,-2,4,-2,-2,12,-2,2,-2,-2,-2],[0.9964999854564667,0.5,0.9975000023841858,0.02777777798473835,-2.0,-2.0,0.007999999914318323,-2.0,-2.0,0.46299999952316284,-2.0,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.002362,0.004763,0.00956,0.00539,0.320443,-0.041937,0.054759,1.110371,0.214368,-0.041971,-0.344159,-0.054907,-1.15263,-1.051596,-1.126952]],[[35,42,12,2,-2,-2,12,-2,-2,-2,-2],[0.5,0.7321428656578064,0.2150000035762787,0.9850000143051147,-2.0,-2.0,0.3894999921321869,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002667,0.00031,0.001416,0.074963,1.380151,0.216143,-0.001172,-0.405619,0.088782,-1.664553,0.762339]],[[19,3,3,8,-2,-2,9,-2,-2,3,-2,3,-2,-2,42,69,-2,-2,-2],[0.5,0.9504999816417694,0.9205000102519989,0.992499977350235,-2.0,-2.0,0.9839999973773956,-2.0,-2.0,0.9665000140666962,-2.0,0.9745000004768372,-2.0,-2.0,0.3166666775941849,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,-1],[0.002494,0.006482,0.011222,0.008066,0.133571,-0.642099,0.099425,-0.541407,1.290716,-0.026111,-1.259287,-0.006736,0.821373,-0.246082,-0.03952,-0.05278,-1.055053,-1.063103,-0.417117]],[[16,8,35,66,-2,-2,-2,12,7,-2,-2,44,-2,-2,12,-2,-2],[0.5,0.9905000030994415,0.5,0.5,-2.0,-2.0,-2.0,0.6994999945163727,0.1080000028014183,-2.0,-2.0,0.5,-2.0,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.000325,0.003087,0.007808,0.005407,0.101246,-0.579697,0.866638,-0.033805,-0.08487,-1.131106,-0.984972,0.032222,-0.07103,0.955948,-0.052372,-0.265154,-1.255271]],[[71,3,4,4,-2,-2,8,-2,-2,3,-2,3,-2,-2,-2],[0.5,0.9504999816417694,0.007500000298023224,0.004500000039115548,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.9665000140666962,-2.0,0.9805000126361847,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.003204,0.005165,0.010597,0.059768,0.256477,2.172479,0.006382,0.125893,-0.890935,-0.028401,-1.341019,-0.008238,0.613466,-0.423589,-0.756409]],[[32,33,43,37,-2,-2,6,-2,-2,-2,12,-2,3,-2,-2],[0.5,0.5,0.5,4.5,-2.0,-2.0,0.7190000116825104,-2.0,-2.0,-2.0,0.4440000057220459,-2.0,0.01699999999254942,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,13,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,14,-1,-1],[-0.003815,-0.007868,-0.005913,-0.009292,-0.091671,-0.914767,0.037569,-0.303723,0.569343,-0.948061,0.036358,0.814061,0.015482,-0.659133,0.622392]],[[39,38,3,12,-2,-2,6,-2,-2,6,42,-2,-2,-2,1,-2,-2],[0.5,0.5,0.001500000071246177,0.2590000033378601,-2.0,-2.0,0.9594999849796295,-2.0,-2.0,0.9869999885559082,0.2361111119389534,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.004696,-0.002239,-0.008187,0.026443,1.44728,0.214766,-0.015165,-0.325421,0.106776,0.026259,0.045724,0.894574,-0.104916,-0.622153,-0.074607,-1.063008,1.331151]],[[16,42,44,2,-2,-2,5,-2,-2,2,-2,-2,4,-2,-2],[0.5,0.612500011920929,0.5,0.9950000047683716,-2.0,-2.0,0.6159999966621399,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.8745000064373016,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.005791,0.008783,0.010476,0.021681,0.503406,-0.371389,-0.005148,0.072331,-1.275198,-0.084485,-1.553012,-1.169683,-0.043914,-1.298025,0.244409]],[[51,19,42,12,-2,-2,10,-2,-2,42,2,-2,-2,-2,-2],[0.5,0.5,0.5857143104076385,0.2580000013113022,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[0.005284,0.002508,0.006622,0.008454,0.669683,0.053691,-0.076049,-1.069785,-1.488483,-0.038997,-0.053869,-1.135358,-1.045698,-0.412988,0.485212]],[[35,4,4,42,-2,-2,42,-2,-2,4,6,-2,-2,4,-2,-2,-2],[0.5,0.007500000298023224,0.004500000039115548,0.1547619104385376,-2.0,-2.0,0.19090909510850906,-2.0,-2.0,0.012500000186264515,0.21299999952316284,-2.0,-2.0,0.9715000092983246,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.00128,-0.000906,0.025966,-0.000402,-0.576491,0.54363,0.107483,1.338559,-0.12199,-0.006203,-0.065863,-0.332206,-1.063057,-0.002099,-0.107601,0.435355,0.842393]],[[11,44,2,3,-2,-2,46,-2,-2,3,7,-2,-2,5,-2,-2,7,8,3,-2,-2,9,-2,-2,3,37,-2,-2,-2],[0.4166666716337204,0.5,0.9950000047683716,0.8440000116825104,-2.0,-2.0,1.5,-2.0,-2.0,0.007500000298023224,0.010499999858438969,-2.0,-2.0,0.006500000134110451,-2.0,-2.0,0.005499999970197678,0.022499999962747097,0.0025000000605359674,-2.0,-2.0,0.9839999973773956,-2.0,-2.0,0.0625,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,-1],[0.000565,-0.007083,0.008829,0.021316,0.05147,0.728822,-0.026759,-0.855411,0.875493,-0.027856,-0.10459,-1.101636,-0.399892,-0.010386,-0.721012,-0.04073,0.022384,0.043559,0.014413,0.076306,1.154727,0.084079,0.662774,1.282202,-0.00362,0.008437,-0.294368,0.66635,-0.591952]],[[17,42,42,63,-2,-2,53,-2,-2,15,2,-2,-2,-2,42,-2,-2],[0.5,0.3452381044626236,0.2952069789171219,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.9249999821186066,-2.0,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.000717,0.002421,0.008253,0.003705,-0.012228,0.579528,0.059095,-0.247261,1.126153,-0.02581,-0.012172,0.888076,-0.574474,-1.071048,-0.064989,-1.173997,0.021651]],[[9,30,16,2,-2,-2,-2,-2,48,63,69,-2,-2,-2,-2],[0.9465000033378601,0.5,0.5,0.9249999821186066,-2.0,-2.0,-2.0,-2.0,0.5,0.5,3.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,-1],[-0.000514,-0.015501,-0.009297,-0.002629,0.532841,-0.251873,-0.801015,-0.896334,0.009514,0.002255,-0.003989,0.021763,-0.732424,0.750832,0.880033]],[[64,5,4,12,-2,-2,7,-2,-2,6,42,-2,-2,12,-2,-2,-2],[0.5,0.9835000038146973,0.16849999874830246,0.46049998700618744,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.26900000870227814,0.21111111342906952,-2.0,-2.0,0.5604999959468842,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.001709,-0.004364,0.001157,0.015817,0.591167,-0.003035,-0.010446,0.158373,-0.383996,-0.047928,-0.004127,0.880187,-1.029018,-0.072139,-0.633102,-1.275975,0.851209]],[[4,4,69,3,-2,-2,-2,-2,5,4,8,-2,-2,3,-2,-2,12,-2,12,-2,-2],[0.0345000009983778,0.02850000001490116,2.5,0.992499977350235,-2.0,-2.0,-2.0,-2.0,0.02850000001490116,0.4385000020265579,0.05199999921023846,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.2580000013113022,-2.0,0.3879999965429306,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[-0.000487,0.020784,0.013055,0.00252,-0.102086,0.588202,0.713122,1.098821,-0.010231,0.019369,-0.027912,0.756627,-0.866015,0.042149,0.337046,1.203117,-0.021678,0.483629,-0.026167,-0.952902,-0.179769]],[[5,7,6,3,-2,-2,6,-2,-2,2,42,-2,-2,42,-2,-2,3,-2,42,-2,3,-2,-2],[0.9895000159740448,0.9335000216960907,0.014500000048428774,0.2540000006556511,-2.0,-2.0,0.9575000107288361,-2.0,-2.0,0.9950000047683716,0.19090909510850906,-2.0,-2.0,0.0555555559694767,-2.0,-2.0,0.278999999165535,-2.0,0.21111111342906952,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,22,-1,-1],[-0.003637,0.00076,-0.003691,0.068085,0.491429,1.239309,-0.007549,-0.260865,0.157823,0.045167,0.095631,1.211688,-0.413406,-0.02338,-1.168606,0.371731,-0.051233,-1.098175,-0.03602,-0.224864,-0.058089,-1.087319,-1.334653]],[[54,12,6,2,-2,-2,3,-2,-2,12,2,-2,-2,65,-2,-2,-2],[0.5,0.2580000013113022,0.792000025510788,0.9850000143051147,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0,0.3279999941587448,0.9249999821186066,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.00132,-0.001256,0.04473,0.095528,1.568959,-0.358746,-0.001231,0.731854,-0.711728,-0.004889,-0.048032,1.107091,-1.017601,-0.000647,-0.05535,0.598175,0.940309]],[[6,6,37,8,-2,-2,-2,12,-2,12,-2,-2,35,44,20,-2,-2,6,-2,-2,-2],[0.24700000137090683,0.11749999970197678,1.5,0.6879999935626984,-2.0,-2.0,-2.0,0.5585000216960907,-2.0,0.7874999940395355,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.38099999725818634,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[5.3e-05,-0.023779,-0.003421,-0.016016,0.357367,-0.71164,0.698399,-0.086923,-1.58581,-0.033392,-1.036021,0.032018,0.006551,0.003404,0.018685,-0.731075,0.406041,-0.013213,0.710582,-0.188019,0.922451]],[[1,39,38,12,-2,-2,34,-2,-2,-2,2,4,7,-2,-2,6,-2,-2,-2],[0.5,0.5,0.5,0.2575000077486038,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.4749999940395355,0.6780000030994415,0.24800000339746475,-2.0,-2.0,0.8664999902248383,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.001528,-0.007741,-0.004705,-0.010249,0.451591,-0.160324,0.027365,0.667745,-0.633291,-0.836549,0.026517,0.019986,0.031753,1.079753,0.194984,0.013147,-0.611389,1.037089,1.325586]],[[51,7,7,3,-2,-2,38,-2,-2,42,4,-2,-2,46,-2,-2,-2],[0.5,0.05049999989569187,0.006500000134110451,0.8095000088214874,-2.0,-2.0,0.5,-2.0,-2.0,0.33974359929561615,0.8064999878406525,-2.0,-2.0,1.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.004088,0.000879,-0.010794,0.004301,0.195439,-0.410601,-0.04001,-0.680974,0.458543,0.016177,0.02627,0.440656,-0.531462,-0.029442,-0.988001,0.460541,0.575852]],[[3,17,9,5,-2,-2,2,-2,-2,4,-2,-2,6,5,-2,15,-2,-2,42,-2,44,-2,-2],[0.24550000578165054,0.5,0.9914999902248383,0.6545000076293945,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.8869999945163727,-2.0,-2.0,0.9835000038146973,0.00849999999627471,-2.0,0.5,-2.0,-2.0,0.1339285746216774,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,16,-1,-1,19,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,15,-1,17,-1,-1,20,-1,22,-1,-1],[-0.002811,0.006866,0.011137,0.001627,0.201968,-0.781725,0.038383,0.81011,-0.728774,-0.059562,-1.327748,-0.129998,-0.018211,-0.028309,-0.801689,-0.023232,-0.386491,0.454087,0.033906,0.871991,-0.024823,0.46972,-1.023213]],[[37,5,7,12,-2,-2,69,-2,-2,3,-2,-2,-2],[4.5,0.992499977350235,0.9535000026226044,0.5304999947547913,-2.0,-2.0,1.5,-2.0,-2.0,0.9769999980926514,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.003075,0.005125,0.008019,0.003783,0.196734,-0.100366,0.046314,-0.054332,1.322717,-0.033701,-0.882972,0.218414,-0.464125]],[[2,46,5,69,-2,-2,6,-2,-2,-2,4,9,-2,4,-2,-2,1,42,-2,-2,-2],[0.9750000238418579,1.5,0.8950000107288361,2.5,-2.0,-2.0,0.3700000047683716,-2.0,-2.0,-2.0,0.05849999934434891,0.824999988079071,-2.0,0.019499999471008778,-2.0,-2.0,0.5,0.029411764815449715,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,18,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,20,19,-1,-1,-1],[0.003795,0.012495,0.016664,0.008065,0.022025,0.614961,0.089428,-0.323167,1.360368,-0.638289,-0.010519,0.009824,1.332736,-0.003809,-0.488256,0.548012,-0.027462,-0.033024,0.174349,-0.734815,1.363201]],[[64,6,12,9,-2,-2,3,-2,-2,6,-2,12,-2,-2,-2],[0.5,0.9575000107288361,0.8350000083446503,0.9774999916553497,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.9675000011920929,-2.0,0.5160000026226044,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[-0.001276,-0.003755,-0.013877,-0.023418,-0.644763,-0.05537,0.024381,0.165901,1.209118,0.012201,0.935055,0.005136,-0.249505,0.498989,0.724439]],[[51,69,19,9,-2,-2,6,-2,-2,6,-2,-2,-2],[0.5,3.5,0.5,0.9914999902248383,-2.0,-2.0,0.9275000095367432,-2.0,-2.0,0.9290000200271606,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.002457,-7.9e-05,0.002963,0.006833,-0.022675,0.255844,-0.032983,-0.319191,-0.995132,-0.044966,0.05226,-0.80526,0.491795]],[[17,42,19,16,-2,-2,9,-2,-2,-2,9,-2,-2],[0.5,0.7321428656578064,0.5,0.5,-2.0,-2.0,0.8125,-2.0,-2.0,-2.0,0.9050000011920929,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[0.001428,0.003798,0.00469,0.008032,0.138482,-0.436662,-0.027524,4.9e-05,-1.098393,-1.673602,-0.050243,-1.19211,-0.165848]],[[39,19,2,46,-2,-2,10,-2,-2,4,6,-2,-2,-2,45,-2,-2],[0.5,0.5,0.9750000238418579,1.5,-2.0,-2.0,0.3541666716337204,-2.0,-2.0,0.9664999842643738,0.8925000131130219,-2.0,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[0.000505,0.002574,0.006544,0.01648,0.270617,-0.951289,-0.00999,-0.293913,0.464095,-0.031735,-0.041071,-1.036873,-1.143112,0.137093,-0.060779,-0.898209,1.36308]],[[71,5,5,3,-2,-2,43,-2,-2,7,69,-2,-2,2,-2,-2,-2],[0.5,0.6845000088214874,0.23349999636411667,0.34700000286102295,-2.0,-2.0,0.5,-2.0,-2.0,0.8244999945163727,0.5,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.001206,0.000521,0.008217,-0.001645,0.136139,-0.416913,0.03765,0.367933,0.94924,-0.017814,-0.067547,-0.2361,-1.281676,-0.002735,0.28317,-1.055658,-0.953569]],[[63,8,21,7,-2,-2,-2,7,12,-2,-2,5,-2,-2,-2],[0.5,0.87950000166893,0.5,0.005499999970197678,-2.0,-2.0,-2.0,0.029000000096857548,0.7414999902248383,-2.0,-2.0,0.31450000405311584,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[0.003526,0.000797,0.008319,0.005167,0.381701,-0.122399,0.94139,-0.013707,-0.062486,-0.444104,-1.031865,0.009959,0.913333,-0.024195,0.536713]],[[5,2,3,3,-2,-2,8,-2,-2,4,-2,8,-2,-2,4,3,2,-2,-2,12,-2,-2,-2],[0.6775000095367432,0.9950000047683716,0.7254999876022339,0.5615000128746033,-2.0,-2.0,0.9724999964237213,-2.0,-2.0,0.020999999716877937,-2.0,0.04649999924004078,-2.0,-2.0,0.8935000002384186,0.9905000030994415,0.9750000238418579,-2.0,-2.0,0.7709999978542328,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.00222,0.010683,0.004555,0.011343,0.099343,1.030448,-0.036214,-0.941699,0.075461,0.063061,1.525848,0.017603,1.239912,-0.072007,-0.018021,-0.009405,-0.000504,0.383649,-0.37234,-0.070005,-0.970854,-1.737456,-1.155477]],[[17,64,5,69,-2,-2,12,-2,-2,-2,37,-2,-2],[0.5,0.5,0.02949999924749136,2.5,-2.0,-2.0,0.3994999974966049,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[-0.003615,-0.00105,-0.003389,0.015927,0.093429,0.956849,-0.010717,-0.474454,-0.014706,0.760667,-0.052066,-1.07595,-0.203586]],[[9,53,20,-2,42,-2,-2,4,12,-2,-2,4,-2,-2,63,8,8,-2,-2,9,-2,-2,-2],[0.9494999945163727,0.5,0.5,-2.0,0.14835165441036224,-2.0,-2.0,0.2785000056028366,0.5339999794960022,-2.0,-2.0,0.5715000033378601,-2.0,-2.0,0.5,0.40949998795986176,0.2685000002384186,-2.0,-2.0,0.9605000019073486,-2.0,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,7,4,-1,6,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.000127,-0.014516,0.013033,-0.694778,0.025493,0.086768,1.081296,-0.033911,-0.012531,0.370978,-0.650038,-0.051289,-1.254243,-0.435256,0.009361,0.004262,0.039636,0.417104,1.327062,-0.00906,1.121039,-0.171719,0.882135]],[[71,42,4,6,-2,-2,4,-2,-2,2,-2,-2,-2],[0.5,0.5857143104076385,0.057499999180436134,0.9945000112056732,-2.0,-2.0,0.06949999928474426,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.00192,0.003595,0.005325,0.021087,0.170996,1.220035,-0.003559,-1.14546,-0.013023,-0.070781,-1.49735,-1.206843,-0.797721]],[[14,3,3,69,-2,-2,-2,4,6,-2,-2,-2,-2],[0.5,0.00849999999627471,0.006500000134110451,0.5,-2.0,-2.0,-2.0,0.9715000092983246,0.9975000023841858,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,-1],[0.003068,0.005312,-0.010387,-0.001942,0.380814,-0.338555,-1.233457,0.012377,0.008742,0.146445,-0.712026,1.356529,-0.41374]],[[7,6,42,-2,6,-2,-2,6,-2,43,-2,-2,3,6,53,-2,-2,1,-2,-2,6,42,-2,-2,9,-2,-2],[0.014500000048428774,0.24000000208616257,0.1339285746216774,-2.0,0.029500000178813934,-2.0,-2.0,0.3674999922513962,-2.0,0.5,-2.0,-2.0,0.5824999809265137,0.33650000393390656,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.3265000134706497,0.41428571939468384,-2.0,-2.0,0.9914999902248383,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,4,-1,6,-1,-1,9,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.000294,0.011091,-0.036739,-1.601518,0.007036,1.51942,-0.560188,0.019241,1.191033,0.012715,0.082205,0.862325,-0.008713,-0.024462,0.025118,1.271887,-0.144527,-0.038294,-0.654838,0.522916,0.015857,-0.026166,-0.234071,-1.383862,0.051365,1.327483,-0.01544]],[[5,16,42,69,-2,-2,69,-2,-2,69,-2,-2,-2],[0.9964999854564667,0.5,0.029411764815449715,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.000847,0.00276,0.005419,0.022029,0.176886,0.64435,-0.001718,0.222102,-0.237987,-0.039052,-1.229075,0.216142,-1.117768]],[[17,6,4,3,-2,-2,9,-2,-2,19,42,-2,-2,8,-2,-2,4,-2,-2],[0.5,0.24400000274181366,0.035499999299645424,0.3825000077486038,-2.0,-2.0,0.9504999816417694,-2.0,-2.0,0.5,0.029411764815449715,-2.0,-2.0,0.8070000112056732,-2.0,-2.0,0.7450000047683716,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.005255,0.008246,-0.015272,0.001466,0.562456,-0.304816,-0.060912,-0.505094,-1.52542,0.014727,0.019837,0.572358,0.119417,-0.025012,-0.261437,-1.123839,-0.055267,-1.323171,0.284409]],[[4,4,37,3,-2,-2,42,-2,-2,-2,4,4,-2,3,-2,-2,4,-2,9,-2,-2],[0.0345000009983778,0.02850000001490116,2.5,0.9895000159740448,-2.0,-2.0,0.17424242943525314,-2.0,-2.0,-2.0,0.04649999924004078,0.03749999962747097,-2.0,0.25299999862909317,-2.0,-2.0,0.05849999934434891,-2.0,0.7739999890327454,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,-1,19,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,18,-1,20,-1,-1],[0.002502,0.018527,0.012361,0.004804,-0.076535,0.44703,0.050957,0.876481,-0.357328,0.833234,-0.004657,-0.06004,0.061669,-0.096799,-1.094249,-0.359041,-0.000777,0.704949,-0.004189,-0.438031,0.079746]],[[7,7,8,17,-2,-2,12,-2,-2,3,12,-2,-2,3,-2,-2,2,8,8,-2,-2,3,-2,-2,5,12,-2,-2,8,-2,-2],[0.06650000065565109,0.007500000298023224,0.925000011920929,0.5,-2.0,-2.0,0.7425000071525574,-2.0,-2.0,0.0494999997317791,0.5779999792575836,-2.0,-2.0,0.13700000196695328,-2.0,-2.0,0.9850000143051147,0.5434999763965607,0.08750000223517418,-2.0,-2.0,0.40199999511241913,-2.0,-2.0,0.9615000188350677,0.6365000009536743,-2.0,-2.0,0.9945000112056732,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[9.6e-05,-0.009243,0.003543,0.013038,0.289833,-1.078107,-0.039146,0.016613,-1.019597,-0.038457,-0.005176,-0.357201,0.762758,-0.07251,-1.412688,-0.456037,0.013816,0.033515,-0.00938,0.693683,-0.564693,0.054698,1.120246,0.320429,-0.008529,0.016553,0.758285,-0.896507,-0.039772,-1.1151,-0.02766]],[[5,43,63,10,-2,-2,-2,5,-2,-2,-2],[0.9964999854564667,0.5,0.5,0.2361111119389534,-2.0,-2.0,-2.0,0.2045000046491623,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,-1],[0.003409,0.005201,0.002282,-0.000788,-0.136982,0.214005,0.54858,0.042888,-0.115499,0.673687,-0.873968]],[[19,5,3,3,-2,-2,7,-2,-2,12,69,-2,-2,-2,6,42,-2,-2,12,-2,-2],[0.5,0.9895000159740448,0.9895000159740448,0.9525000154972076,-2.0,-2.0,0.3034999966621399,-2.0,-2.0,0.8680000007152557,0.5,-2.0,-2.0,-2.0,0.926499992609024,0.20714285969734192,-2.0,-2.0,0.46299999952316284,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[0.002408,0.005854,0.009284,0.006754,0.137104,-0.531037,0.070347,0.984599,-0.2216,-0.023237,-0.038171,-0.009848,-0.843558,0.846229,-0.028721,-0.012952,-1.037227,0.07228,-0.042163,-0.748263,-1.114053]],[[65,56,4,3,-2,-2,4,-2,-2,4,-2,-2,12,-2,-2],[0.5,0.5,0.03749999962747097,0.99549999833107,-2.0,-2.0,0.972000002861023,-2.0,-2.0,0.5519999861717224,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[-0.000881,-0.004471,-0.001111,0.015967,0.304031,-1.14069,-0.008675,-0.24787,0.461548,-0.066712,-0.54647,-1.198006,0.047946,0.960457,0.381312]],[[16,43,2,44,-2,-2,7,-2,-2,6,-2,-2,6,-2,-2],[0.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.06650000065565109,-2.0,-2.0,0.7529999911785126,-2.0,-2.0,0.9535000026226044,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.001272,0.004487,0.001146,0.017292,0.29167,0.727118,-0.007445,-0.353535,0.193344,0.042714,-0.425869,0.739493,-0.050896,-1.150906,-0.013947]],[[3,44,6,37,-2,-2,6,-2,-2,5,37,-2,-2,42,-2,-2,5,-2,-2],[0.9934999942779541,0.5,0.4584999978542328,1.5,-2.0,-2.0,0.5685000121593475,-2.0,-2.0,0.6159999966621399,2.5,-2.0,-2.0,0.18333333730697632,-2.0,-2.0,0.940500020980835,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.001648,0.001014,0.012371,-0.010007,-0.572709,0.613641,0.02409,1.621457,0.355771,-0.013351,-0.005329,-0.195561,0.571262,-0.088099,-1.30605,-0.608338,-0.066676,-1.205044,-0.7919]],[[44,2,42,46,-2,-2,3,-2,-2,46,8,-2,-2,7,-2,-2,6,-2,4,69,-2,-2,3,-2,-2],[0.5,0.9950000047683716,0.44949495792388916,1.5,-2.0,-2.0,0.013999999966472387,-2.0,-2.0,1.5,0.99549999833107,-2.0,-2.0,0.7515000104904175,-2.0,-2.0,0.009499999694526196,-2.0,0.9619999825954437,3.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,24,-1,-1],[-0.000673,0.009327,0.01651,0.021398,0.459871,-0.607732,-0.04299,1.065436,-1.178379,-0.015805,-0.030412,-1.040173,0.48917,0.043998,1.128353,0.524497,-0.014154,1.094343,-0.018799,-0.026008,-0.226865,-0.784864,0.022845,-0.347076,1.394068]],[[16,2,46,7,-2,-2,-2,7,8,-2,-2,5,-2,-2,9,-2,-2],[0.5,0.9750000238418579,1.5,0.9354999959468842,-2.0,-2.0,-2.0,0.0025000000605359674,0.03749999962747097,-2.0,-2.0,0.5525000095367432,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.000626,0.002059,0.010326,0.015036,0.136168,0.935205,-0.732627,-0.012276,-0.061946,0.678777,-1.115565,-0.005026,0.335273,-0.426695,-0.041651,-0.940749,-0.236982]],[[37,73,14,6,-2,-2,-2,-2,-2],[4.5,0.5,0.5,0.3254999965429306,-2.0,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,-1],[0.003247,0.005348,0.003325,0.00559,-0.229036,0.168732,-0.415874,0.83989,-0.602861]],[[42,51,4,6,-2,-2,45,-2,-2,-2,-2],[0.7321428656578064,0.5,0.19449999928474426,0.99549999833107,-2.0,-2.0,0.5445000231266022,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002661,0.003414,0.001339,0.011263,0.077894,0.86865,-0.008628,-0.087589,-1.284219,0.437375,-1.614338]],[[19,3,3,69,-2,-2,-2,6,42,-2,-2,6,-2,-2,4,6,-2,-2,-2],[0.5,0.007500000298023224,0.005499999970197678,0.5,-2.0,-2.0,-2.0,0.9794999957084656,0.3484848588705063,-2.0,-2.0,0.9975000023841858,-2.0,-2.0,0.9679999947547913,0.8925000131130219,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,-1],[0.005702,0.008907,-0.008386,0.003884,0.529979,-0.301048,-1.22218,0.016854,0.004895,0.150324,-0.692821,0.058319,0.898346,-0.158025,-0.025479,-0.035139,-0.449604,-1.13215,0.261456]]]
UMBRAL = 0.18
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
    # scikit-learn compara en precisión simple (float32); hacemos lo mismo para
    # que el bot dé exactamente las mismas probabilidades que en el entrenamiento
    x = [struct.unpack("f", struct.pack("f", float(feats.get(n, 0.0))))[0] for n in NOMBRES]
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
