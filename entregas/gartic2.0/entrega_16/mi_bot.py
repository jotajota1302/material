"""El Guardián — bot de joseca (versión 12).

v12 = v4 + características nuevas para los fallos de la v4, aceptadas solo si
mejoraban H y respuestas de varias oraciones con DOS semillas del modelo
(validación por familias): llm.

--- Notas de la versión 4 ---

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

def _v12_llm():
    """v12: características nuevas «llm» (espacio de nombres propio)."""
    import re, unicodedata

    def _n(t):
        t = unicodedata.normalize("NFKD", t or "").lower()
        return "".join(c for c in t if not unicodedata.combining(c))

    def _toks(t):
        return set(re.findall(r"\w{4,}", _n(t)))

    def _nums(t):
        return set(re.findall(r"\d+(?:[.,]\d+)?", t or ""))

    def nuevas(caso):
        fu = caso.get("fuentes") or []
        fn = [_n(f) for f in fu]
        ft = [_toks(f) for f in fu]
        ors = caso["senales"]["oraciones"]
        malas = 0; dudosas = 0; cob_min = 1.0; resp_malas = 0.0
        for o in ors:
            v = (o.get("llm") or {}).get("veredicto", "")
            if v not in ("contradicha", "no_respaldada"):
                continue
            malas += 1
            tx = o.get("texto") or ""
            tk = _toks(tx); nu = _nums(tx)
            cob = max((len(tk & t) / max(1, len(tk)) for t in ft), default=0.0)
            numok = all(any(x in f for f in fn) for x in nu)
            rp = max(o.get("respaldo_por_fuente") or [0.0])
            cob_min = min(cob_min, cob)
            resp_malas = max(resp_malas, rp)
            if cob >= 0.8 and numok and rp >= 0.5:
                dudosas += 1
        return {"llm_malas_resp_max": resp_malas if malas else -1.0,
                "llm_malas_cob_min": cob_min if malas else 1.0}

    return nuevas


nuevas_llm = _v12_llm()


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
    f.update(nuevas_llm(caso))  # v12
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
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas", "llm_malas_resp_max", "llm_malas_cob_min"]
INICIAL = -0.035456
TASA = 0.05
ARBOLES = [[[74,2,19,70,-2,-2,4,-2,-2,32,43,-2,-2,-2,3,45,3,-2,-2,-2,4,3,-2,-2,7,-2,-2],[-0.5,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.37549999356269836,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.07499999925494194,0.7999999821186066,0.03350000083446503,-2.0,-2.0,-2.0,0.28450000286102295,0.8075000047683716,-2.0,-2.0,0.05199999921023846,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,26,-1,-1],[0.003123,-0.168097,0.015381,0.08349,-0.044107,1.471813,-0.454372,-1.965165,-1.631727,-0.310171,-0.349398,-1.535728,0.171429,0.58109,0.453308,0.496534,0.502928,2.029761,1.750288,1.678837,0.300125,0.438863,2.036092,1.399529,0.169241,1.390728,-0.328287]],[[2,8,53,3,-2,-2,-2,7,8,-2,-2,12,-2,-2,8,4,42,-2,-2,74,-2,-2,38,51,-2,-2,42,-2,-2],[0.9350000023841858,0.17599999904632568,0.5,0.03350000083446503,-2.0,-2.0,-2.0,0.001500000071246177,0.8425000011920929,-2.0,-2.0,0.801499992609024,-2.0,-2.0,0.07600000128149986,0.08900000154972076,0.24038461595773697,-2.0,-2.0,-0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.002386,0.38759,0.465744,0.480568,1.938913,1.650848,1.395588,0.306651,0.461002,1.955734,1.614531,0.253775,0.805823,1.964644,-0.177145,0.195157,0.41393,1.859267,1.242614,0.05467,-0.200175,1.93676,-0.239725,-0.29492,-1.324647,0.483868,0.070542,0.970052,-1.555576]],[[74,32,2,52,-2,-2,48,-2,-2,10,-2,-2,3,45,8,-2,-2,-2,4,75,-2,-2,7,-2,-2],[-0.5,0.5,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.032500000670552254,0.7800000011920929,0.48900000751018524,-2.0,-2.0,-2.0,0.44350001215934753,0.4305555522441864,-2.0,-2.0,0.007500000298023224,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[0.004794,-0.160197,-0.21168,-0.066362,-0.007238,-1.687316,-0.323575,-1.423848,-0.02422,0.278964,0.399668,1.765433,0.421307,0.451607,0.458609,1.857259,1.684388,1.549422,0.354401,0.44691,1.35014,1.892893,0.253302,1.88712,0.617037]],[[74,2,52,70,-2,-2,-2,32,43,-2,-2,-2,3,8,45,-2,-2,37,-2,-2,4,0,-2,-2,42,-2,-2],[-0.5,0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.032500000670552254,0.17149999737739563,0.7039999961853027,-2.0,-2.0,0.5,-2.0,-2.0,0.28450000286102295,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.014819,-0.165309,0.003266,0.054745,-0.086062,1.162264,-1.468465,-0.2947,-0.320614,-1.411927,0.133841,0.27177,0.384744,0.428821,0.437768,1.776483,1.802525,0.388826,1.23085,1.810412,0.293155,0.396269,1.817024,1.337377,0.192685,1.208559,-0.666777]],[[74,2,52,70,-2,-2,8,-2,-2,18,38,-2,-2,-2,3,0,45,-2,-2,3,-2,-2,45,-2,-2],[-0.5,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.6970000267028809,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.34849999845027924,0.5,0.7999999821186066,-2.0,-2.0,0.032500000670552254,-2.0,-2.0,0.5565000176429749,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[-0.010984,-0.154558,-0.003672,0.054491,-0.084365,1.324358,-0.353395,-1.092682,-1.724857,-0.275844,-0.304259,-1.381111,-0.191025,0.583157,0.372974,0.392272,0.412827,1.728124,1.467487,0.349028,1.680156,0.622582,0.232671,0.20081,1.683557]],[[2,7,3,42,-2,-2,-2,44,3,-2,-2,-2,8,4,6,-2,-2,74,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.636000007390976,0.0416666679084301,-2.0,-2.0,-2.0,0.5,0.25300000607967377,-2.0,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.3439999967813492,-2.0,-2.0,-0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.17995689809322357,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.004806,0.329309,0.385941,0.397182,1.476653,1.713969,0.863548,0.236739,0.283524,1.491825,0.52227,0.13774,-0.159942,0.150183,0.301985,1.612357,0.76676,-0.073806,-0.631519,1.474472,-0.216415,-0.259181,-1.233999,0.462377,0.025538,1.517259,-1.187257]],[[74,2,19,42,-2,-2,8,-2,-2,11,42,-2,-2,-2,3,42,3,-2,-2,-2,45,-2,-2],[-0.5,0.9750000238418579,0.5,0.2752525359392166,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,0.4166666716337204,0.12916667014360428,-2.0,-2.0,-2.0,0.34849999845027924,0.4833333343267441,0.11649999767541885,-2.0,-2.0,-2.0,0.718999981880188,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,22,-1,-1],[0.007168,-0.128354,-0.006223,0.047496,0.492722,-0.591879,-0.335987,-1.102275,-1.654263,-0.256177,-0.285081,-0.790126,-1.533394,0.138114,0.347382,0.369383,0.375133,1.572954,1.69906,1.095364,0.217119,0.251181,1.603648]],[[74,2,52,70,-2,-2,5,-2,-2,51,65,-2,-2,-2,5,3,40,-2,-2,4,-2,-2,-2],[-0.5,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.16450000554323196,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.4529999941587448,0.11449999734759331,0.5,-2.0,-2.0,0.44350001215934753,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.013266,-0.139408,0.011082,0.069584,-0.017425,1.356378,-0.341481,-1.640705,-1.290817,-0.253376,-0.279797,-1.289964,0.263596,0.586085,0.317468,0.339202,0.351775,1.5514,1.185785,0.274765,1.559162,0.471175,0.609909]],[[74,70,21,2,-2,-2,-2,42,6,-2,-2,-2,3,8,3,-2,-2,9,-2,-2,4,3,-2,-2,-2],[-0.5,0.5,0.5,0.9249999821186066,-2.0,-2.0,-2.0,0.22474747896194458,0.9169999957084656,-2.0,-2.0,-2.0,0.25300000607967377,0.17599999904632568,0.041999999433755875,-2.0,-2.0,0.4845000058412552,-2.0,-2.0,0.3739999979734421,0.8075000047683716,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[0.003321,-0.116941,-0.169924,-0.194209,0.447069,-0.943068,1.448157,0.15199,0.308609,1.695813,0.30368,-0.882139,0.307579,0.326263,0.341098,1.50458,1.629337,0.290189,0.612401,1.400877,0.206208,0.310841,1.601494,1.022719,-0.048415]],[[2,7,3,12,-2,-2,-2,44,75,-2,-2,-2,8,4,2,-2,-2,12,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.6884999871253967,0.48349998891353607,-2.0,-2.0,-2.0,0.5,0.7888889014720917,-2.0,-2.0,-2.0,0.0885000005364418,0.8745000064373016,0.9600000083446503,-2.0,-2.0,0.4845000058412552,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.002369,0.275657,0.318625,0.328319,1.323991,1.5394,0.82616,0.204444,0.251366,0.727384,1.603331,-0.134884,-0.121497,0.123076,0.243637,1.527898,0.612753,-0.041256,-0.866878,0.414301,-0.168293,-0.211544,-1.053065,0.500451,0.090126,1.21437,-1.134111]],[[74,64,54,2,-2,-2,-2,-2,5,6,4,-2,-2,3,-2,-2,-2],[-0.5,0.5,0.5,0.9750000238418579,-2.0,-2.0,-2.0,-2.0,0.6629999876022339,0.9755000174045563,0.17650000005960464,-2.0,-2.0,0.05199999921023846,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.001091,-0.107297,-0.132855,-0.158796,-0.284319,-1.085765,1.514396,1.399674,0.272028,0.286905,0.259562,1.430675,0.912603,0.319498,1.45922,1.570817,0.375433]],[[2,5,69,2,-2,-2,-2,-2,38,74,43,-2,-2,4,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,0.75,-2.0,-2.0,-2.0,-2.0,0.5,-0.5,0.5,-2.0,-2.0,0.032999999821186066,-2.0,-2.0,0.2361111119389534,0.9460000097751617,-2.0,-2.0,0.390500009059906,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.001152,0.251032,0.271573,0.253978,1.27523,-0.014501,1.722728,0.31358,-0.116907,-0.167989,-0.188919,-0.99175,0.504718,0.317105,1.227401,1.519033,0.108939,0.283873,1.606596,0.195842,-0.255925,-0.135189,-1.677497]],[[74,32,15,2,-2,-2,9,-2,-2,11,-2,-2,3,42,10,-2,-2,-2,45,-2,-2],[-0.5,0.5,0.5,0.9350000023841858,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.25757576525211334,-2.0,-2.0,0.33650000393390656,0.4833333343267441,0.8333333432674408,-2.0,-2.0,-2.0,0.715499997138977,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,-1],[0.002224,-0.094866,-0.12896,-0.167412,0.539924,-0.850177,0.181893,0.036544,1.651003,0.234614,0.396167,1.675924,0.252798,0.27076,0.276848,1.389926,1.231522,0.745684,0.137622,-0.08414,1.471761]],[[2,8,2,3,-2,-2,-2,7,8,-2,-2,8,-2,-2,8,4,37,-2,-2,37,-2,-2,42,38,-2,-2,48,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,0.010499999858438969,-2.0,-2.0,-2.0,0.001500000071246177,0.7580000162124634,-2.0,-2.0,0.9740000069141388,-2.0,-2.0,0.09149999916553497,0.21649999916553497,0.5,-2.0,-2.0,2.5,-2.0,-2.0,0.18465909361839294,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.002642,0.213402,0.276757,0.261756,1.37856,1.092722,1.636481,0.145174,0.27657,1.449366,1.072646,0.10403,0.279563,1.206467,-0.103869,0.120336,0.269831,0.671283,1.600228,-0.002647,-0.381986,0.978036,-0.14674,-0.07236,-0.577915,1.166058,-0.225148,-1.19214,0.515514]],[[2,5,74,-2,45,-2,-2,42,6,-2,-2,4,-2,-2,38,15,74,-2,-2,9,-2,-2,42,-2,4,-2,-2],[0.9350000023841858,0.023000000044703484,-0.5,-2.0,0.7999999821186066,-2.0,-2.0,0.24264705926179886,0.8695000112056732,-2.0,-2.0,0.016500000841915607,-2.0,-2.0,0.5,0.5,-0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.11324786394834518,-2.0,0.20099999755620956,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,25,-1,-1],[14,7,4,-1,6,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,26,-1,-1],[-0.000131,0.215881,0.267913,1.617264,0.246357,1.339335,0.800496,0.174357,0.210858,0.642503,1.24148,0.054317,1.573091,-0.504305,-0.101473,-0.142166,-0.173988,-0.868935,1.298538,0.118351,-0.301288,1.509246,0.08725,1.404516,-0.065245,0.20834,-0.920822]],[[2,3,5,8,-2,-2,74,-2,-2,4,3,-2,-2,5,-2,-2,38,15,74,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.04749999940395355,0.055000001564621925,0.17149999737739563,-2.0,-2.0,0.07149999961256981,-2.0,-2.0,0.44350001215934753,0.7790000140666962,-2.0,-2.0,0.0885000005364418,-2.0,-2.0,0.5,0.5,-0.5,-2.0,-2.0,0.9554999768733978,-2.0,-2.0,0.2567765712738037,0.9639999866485596,-2.0,-2.0,0.3920000046491623,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.005512,0.201014,0.239731,0.220024,1.28997,-0.004649,0.285105,1.592327,1.360723,0.132576,0.208647,1.193899,0.56889,-0.036187,0.963034,-0.772041,-0.099832,-0.146189,-0.176256,-0.898908,1.286672,0.146423,-0.398279,1.374984,0.122794,0.287306,1.560072,0.497103,-0.226795,0.114623,-1.562233]],[[74,8,4,5,-2,-2,12,-2,-2,21,38,-2,-2,-2,3,46,7,-2,-2,8,-2,-2,6,12,-2,-2,6,-2,-2],[-0.5,0.09149999916553497,0.8700000047683716,0.03799999877810478,-2.0,-2.0,0.48999999463558197,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.032500000670552254,1.5,0.02650000061839819,-2.0,-2.0,0.020999999716877937,-2.0,-2.0,0.6960000097751617,0.570499986410141,-2.0,-2.0,0.9410000145435333,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.004477,-0.08363,0.130135,0.283135,1.613604,0.748516,-0.064847,-1.172191,0.339306,-0.121872,-0.13812,-0.800061,0.141318,1.101757,0.207228,0.240828,0.236033,1.298206,1.348797,0.273355,1.360011,1.446965,0.12772,0.241104,0.456247,1.441046,0.046425,-0.755431,0.901207]],[[2,9,7,74,-2,-2,12,-2,-2,7,4,-2,-2,-2,8,4,8,-2,-2,12,-2,-2,42,38,-2,-2,42,-2,-2],[0.9350000023841858,0.99549999833107,0.001500000071246177,0.0005000000237487257,-2.0,-2.0,0.7975000143051147,-2.0,-2.0,0.012500000186264515,0.1655000001192093,-2.0,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.016000000294297934,-2.0,-2.0,0.4830000102519989,-2.0,-2.0,0.11324786394834518,0.5,-2.0,-2.0,0.33908046782016754,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.003585,0.187397,0.16914,0.240591,1.388179,1.299119,0.133295,0.400778,1.383053,0.269504,0.23918,1.350325,1.242555,1.481903,-0.082683,0.126395,0.269109,0.751999,1.615142,-0.056725,-1.00944,0.321787,-0.123278,-0.00466,-0.332516,1.575077,-0.189893,-0.675164,-1.347757]],[[74,2,52,49,-2,-2,7,-2,-2,15,10,-2,-2,-2,5,42,37,-2,-2,-2,8,-2,-2],[-0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,0.010999999940395355,-2.0,-2.0,0.5,0.4166666716337204,-2.0,-2.0,-2.0,0.4529999941587448,0.4833333343267441,0.5,-2.0,-2.0,-2.0,0.718999981880188,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,22,-1,-1],[0.001947,-0.070841,0.009648,0.05446,0.36243,-1.108432,-0.261929,-0.997722,-1.458992,-0.160004,-0.175617,-1.011322,-0.293343,0.208442,0.190471,0.206788,0.212912,1.116542,1.301139,0.530916,0.071675,1.20219,-0.19164]],[[2,8,4,75,-2,-2,4,-2,-2,9,12,-2,-2,4,-2,-2,32,21,74,-2,-2,-2,42,-2,-2],[0.9350000023841858,0.17499999701976776,0.9465000033378601,0.9375,-2.0,-2.0,0.965499997138977,-2.0,-2.0,0.9465000033378601,0.801499992609024,-2.0,-2.0,0.28450000286102295,-2.0,-2.0,0.5,0.5,-0.5,-2.0,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,24,-1,-1],[0.002269,0.169921,0.205127,0.228866,1.265314,1.400355,0.178791,0.110277,1.270228,0.13205,0.026521,-0.451544,1.370193,0.188526,1.178829,0.684503,-0.071659,-0.098967,-0.117333,-0.643116,1.245457,1.23294,0.176793,1.377452,-0.036438]],[[74,2,49,52,-2,-2,-2,42,37,-2,-2,15,-2,-2,37,8,45,-2,-2,7,-2,-2,6,-2,-2],[-0.5,0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.11437908560037613,1.5,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.8264999985694885,0.4520000070333481,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.9415000081062317,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[0.005084,-0.062506,0.033879,0.065949,0.450775,-1.087052,-1.235494,-0.140384,-0.047222,-0.569286,0.590878,-0.197212,-1.108393,-0.026547,0.180921,0.19563,0.187933,1.188577,0.764692,0.250589,1.198874,1.406185,0.086644,-0.216782,1.327838]],[[2,4,3,8,-2,-2,53,-2,-2,3,42,-2,-2,7,-2,-2,15,31,74,-2,-2,-2,9,-2,-2],[0.9350000023841858,0.17650000005960464,0.02049999963492155,0.04400000162422657,-2.0,-2.0,0.5,-2.0,-2.0,0.032500000670552254,0.21111111342906952,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,-0.5,-2.0,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,24,-1,-1],[0.003203,0.157557,0.218392,0.169444,1.304576,0.410483,0.264414,1.233083,1.469115,0.126795,0.17437,1.255086,0.695258,0.027699,1.231621,-0.341042,-0.069206,-0.091674,-0.110748,-0.632364,1.209816,0.930161,0.162891,-0.156814,1.364265]],[[74,65,2,9,-2,-2,42,-2,-2,42,-2,-2,3,75,3,-2,-2,10,-2,-2,4,7,-2,-2,3,-2,-2],[-0.5,0.5,0.9600000083446503,0.926499992609024,-2.0,-2.0,0.11437908560037613,-2.0,-2.0,0.1547619104385376,-2.0,-2.0,0.07499999925494194,0.5227272808551788,0.006500000134110451,-2.0,-2.0,0.3095238208770752,-2.0,-2.0,0.28450000286102295,0.018499999307096004,-2.0,-2.0,0.23600000143051147,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.002951,-0.06232,-0.087739,-0.004395,-0.699911,0.261659,-0.148597,-0.367468,-1.014561,0.184707,1.41094,0.102526,0.149417,0.171979,0.144964,1.240794,0.171254,0.187207,1.284222,1.22307,0.064496,0.186249,0.666372,1.360298,-0.112872,0.054559,-0.946565]],[[2,69,3,8,-2,-2,12,-2,-2,-2,38,8,4,-2,-2,42,-2,-2,42,6,-2,-2,-2],[0.9350000023841858,1.5,0.016500000841915607,0.44200000166893005,-2.0,-2.0,0.8314999938011169,-2.0,-2.0,-2.0,0.5,0.05649999901652336,0.878000020980835,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.2899159789085388,0.9864999949932098,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.005715,0.140813,0.130483,0.162517,1.035623,1.38174,0.083768,0.27826,1.333616,1.219923,-0.05927,-0.093948,0.105921,1.162334,-0.407532,-0.120332,-0.245096,-0.846487,0.117587,0.21703,1.357964,-0.157049,-0.966294]],[[2,9,5,3,-2,-2,75,-2,-2,75,3,-2,-2,6,-2,-2,8,4,3,-2,-2,9,-2,-2,42,30,-2,-2,12,-2,-2],[0.9350000023841858,0.9410000145435333,0.11300000175833702,0.012500000186264515,-2.0,-2.0,0.7321428656578064,-2.0,-2.0,0.5227272808551788,0.032500000670552254,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.0755000002682209,0.9280000030994415,0.02049999963492155,-2.0,-2.0,0.7620000243186951,-2.0,-2.0,0.2604166716337204,0.5,-2.0,-2.0,0.5879999995231628,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.005519,0.133068,0.098828,0.154055,0.819382,1.302611,-0.026107,0.898846,-0.687285,0.167309,0.088718,1.1381,-0.273086,0.201745,1.267621,0.727204,-0.068593,0.096021,0.171683,0.615726,1.627623,-0.068273,-0.662003,0.359789,-0.095757,-0.045532,-0.35668,1.166445,-0.194361,-0.844397,-1.378634]],[[74,54,64,52,-2,-2,-2,-2,9,6,6,-2,-2,3,-2,-2,-2],[-0.5,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.9975000023841858,0.948500007390976,0.6960000097751617,-2.0,-2.0,0.05850000120699406,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.00254,-0.051824,-0.067334,-0.087253,-0.307597,-1.312463,1.069976,1.213434,0.148872,0.154583,0.131515,1.074719,0.383196,0.178813,1.205528,1.387065,0.33919]],[[74,65,21,63,-2,-2,-2,42,-2,-2,8,3,3,-2,-2,-2,42,9,-2,-2,4,-2,-2],[-0.5,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.17599999904632568,0.048499999567866325,0.021499999798834324,-2.0,-2.0,-2.0,0.2152777761220932,0.968500018119812,-2.0,-2.0,0.19250000268220901,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[-0.003115,-0.0522,-0.077231,-0.094661,-0.566159,0.516223,1.197546,0.157961,1.350542,0.211634,0.125275,0.154647,0.15197,1.185637,0.687834,1.280677,0.083575,0.129296,0.340266,1.271322,-0.035643,0.855217,-0.867167]],[[2,19,17,16,-2,-2,-2,8,-2,-2,74,18,42,-2,-2,-2,8,4,-2,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,-0.5,0.5,0.11437908560037613,-2.0,-2.0,-2.0,0.03849999979138374,0.7495000064373016,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.007352,0.08008,0.106943,0.125117,0.788982,-0.98909,-1.183895,-0.228479,-0.349293,-1.627134,-0.08444,-0.102667,-0.120592,-0.095889,-1.060123,0.889622,0.191355,0.167717,1.194072,1.22846,1.424717]],[[2,19,16,17,-2,-2,-2,7,-2,-2,8,5,8,-2,-2,-2,42,69,-2,-2,5,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.13199999928474426,0.032500000670552254,0.019999999552965164,-2.0,-2.0,-2.0,0.11437908560037613,1.5,-2.0,-2.0,0.6540000140666962,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.006495,0.070899,0.096183,0.11954,0.789006,-1.329978,-1.093704,-0.222938,-0.578921,-1.557654,-0.084085,0.08481,0.229737,0.847748,1.897455,-0.345333,-0.11389,-0.035411,-0.654186,1.248695,-0.161443,-0.753951,-1.142284]],[[2,7,69,8,-2,-2,-2,44,42,-2,-2,-2,8,4,3,-2,-2,9,-2,-2,38,51,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.5,0.6410000026226044,-2.0,-2.0,-2.0,0.5,0.2361111119389534,-2.0,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.016000000294297934,-2.0,-2.0,0.8824999928474426,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.17995689809322357,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.005968,0.107225,0.137511,0.122981,1.060491,0.216898,1.345,0.06252,0.09851,0.827933,-0.350056,-0.461076,-0.059886,0.088435,0.164769,0.600501,1.58953,-0.023131,-0.849555,0.528626,-0.089894,-0.114427,-0.730561,0.387023,0.046136,1.002063,-0.581192]],[[74,54,4,8,-2,-2,38,-2,-2,-2,8,74,5,-2,-2,-2,-2],[-0.5,0.5,0.05849999934434891,0.5185000002384186,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.9054999947547913,0.6729999780654907,0.09149999916553497,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,-1],[0.001895,-0.045718,-0.060594,0.009635,1.012192,-0.246423,-0.1174,-0.769726,0.083567,1.278462,0.121179,0.132666,0.127676,1.124849,0.683563,1.344575,0.202546]],[[2,9,8,42,-2,-2,6,-2,-2,42,3,-2,-2,-2,42,37,2,-2,-2,4,-2,-2,74,51,-2,-2,8,-2,-2],[0.9350000023841858,0.9440000057220459,0.17499999701976776,0.3693181872367859,-2.0,-2.0,0.6800000071525574,-2.0,-2.0,0.44949495792388916,0.37700000405311584,-2.0,-2.0,-2.0,0.11324786394834518,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,-0.5,0.5,-2.0,-2.0,0.053999999538064,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.003152,0.101042,0.066656,0.12196,0.99432,1.293409,-0.023387,0.583848,-0.701609,0.135297,0.146629,0.90032,1.194917,-0.594781,-0.050773,0.041996,-0.032626,0.230586,-0.708661,0.218633,1.885359,0.220141,-0.100292,-0.116494,-0.743614,0.273537,0.168483,1.059951,1.419631]],[[19,2,39,5,-2,-2,-2,11,42,-2,-2,7,-2,-2,6,-2,55,-2,-2],[0.5,0.9350000023841858,0.5,0.45749999582767487,-2.0,-2.0,-2.0,0.4166666716337204,0.06458333507180214,-2.0,-2.0,0.005499999970197678,-2.0,-2.0,0.7094999849796295,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[-0.003443,0.019279,0.111651,0.119096,0.82859,1.089419,-0.279943,-0.026358,-0.058496,0.181145,-0.55035,0.121012,1.26626,0.165803,-0.212727,-1.089735,-0.237998,-1.542634,-1.228564]],[[19,2,16,17,-2,-2,-2,5,8,-2,-2,2,-2,-2,6,5,-2,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,0.2619999945163727,0.5959999859333038,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,0.9804999828338623,0.17499999701976776,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,-1],[0.015327,0.03566,0.08844,0.105952,0.742506,-0.91373,-0.982196,-0.051046,0.046856,0.765103,-0.46478,-0.104435,-0.343677,-0.990361,-0.199437,-0.164169,-0.683196,-1.266958,-1.370336]],[[11,42,37,2,-2,-2,4,-2,-2,1,70,-2,-2,6,-2,-2,65,5,6,-2,-2,2,-2,-2,-2],[0.20714285969734192,0.13076923787593842,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9430000185966492,-2.0,-2.0,0.5,0.02850000001490116,0.9975000023841858,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[0.002358,-0.041927,0.024607,-0.040785,0.120673,-0.951641,0.150392,1.828053,-0.169133,-0.09015,-0.10992,-0.780349,0.079676,0.129317,0.21129,1.259542,0.096164,0.079081,0.130368,1.134209,0.228971,0.022474,0.9935,-0.205651,1.400487]],[[2,37,8,4,-2,-2,10,-2,-2,12,-2,53,-2,-2,42,38,37,-2,-2,-2,8,4,-2,-2,15,-2,-2],[0.9350000023841858,0.5,0.15150000154972076,0.9465000033378601,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.3869999945163727,-2.0,0.5,-2.0,-2.0,0.13076923787593842,0.5,2.5,-2.0,-2.0,-2.0,0.1314999982714653,0.9259999990463257,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,26,-1,-1],[0.003423,0.092732,0.05869,0.102575,1.186879,0.678117,-0.002207,0.446218,-0.937238,0.129931,0.256017,0.150619,1.062231,1.221454,-0.03683,0.037537,-0.002249,-0.219277,1.407315,1.218302,-0.084015,0.055026,0.617492,-0.254367,-0.122268,-0.835831,0.118406]],[[2,19,16,50,-2,-2,-2,7,-2,-2,74,70,4,-2,-2,42,-2,-2,8,42,-2,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.010999999940395355,-2.0,-2.0,-0.5,0.5,0.08950000256299973,-2.0,-2.0,0.20256410539150238,-2.0,-2.0,0.0409999992698431,0.377976194024086,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.001726,0.051479,0.072714,0.088723,0.640024,-1.055781,-0.963904,-0.175195,-0.535682,-1.5257,-0.064002,-0.07745,-0.094488,-0.326892,-0.891316,0.054115,0.824984,-1.049884,0.14421,0.115959,1.156062,0.710856,1.359496]],[[19,2,17,16,-2,-2,12,-2,-2,5,8,-2,-2,42,-2,-2,8,-2,8,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,0.5564999878406525,-2.0,-2.0,0.6780000030994415,0.7365000247955322,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,0.6555000245571136,-2.0,0.9309999942779541,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[0.00461,0.020995,0.059146,0.075724,0.580754,-0.965024,-0.19256,-0.635719,-1.585962,-0.042112,0.025303,0.520569,-0.506217,-0.103598,-0.245304,-1.060862,-0.174533,-0.811489,-0.208403,-1.468666,-1.289143]],[[19,8,39,44,-2,-2,-2,42,17,-2,-2,75,-2,-2,6,-2,42,-2,-2],[0.5,0.08900000154972076,0.5,0.5,-2.0,-2.0,-2.0,0.33908046782016754,0.5,-2.0,-2.0,0.7857142984867096,-2.0,-2.0,0.6404999792575836,-2.0,0.3166666775941849,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[0.004409,0.021143,0.112468,0.121031,0.979511,0.898808,-0.140781,-0.010938,0.01432,0.168017,-1.287899,-0.144611,0.185121,-1.028311,-0.173325,-0.919918,-0.197445,-1.394167,-0.964209]],[[2,19,16,17,-2,-2,-2,8,-2,-2,8,6,75,-2,-2,-2,4,10,-2,-2,53,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,0.04649999924004078,0.9769999980926514,0.6904762089252472,-2.0,-2.0,-2.0,0.14400000125169754,0.2916666716337204,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.00595,0.057378,0.075708,0.090159,0.656852,-0.882336,-0.950329,-0.146779,-0.108502,-1.453589,-0.063261,0.101863,0.175574,0.921486,1.23616,-0.294569,-0.081284,-0.030861,-0.384118,1.074303,-0.145347,0.1184,-1.033845]],[[2,19,16,17,-2,-2,-2,8,-2,-2,74,42,4,-2,-2,43,-2,-2,8,4,-2,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,-0.5,0.11437908560037613,0.3630000054836273,-2.0,-2.0,0.5,-2.0,-2.0,0.053999999538064,0.7495000064373016,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.004918,0.047423,0.062462,0.076127,0.576296,-0.855532,-0.825576,-0.130424,0.021891,-1.414441,-0.05438,-0.067706,0.006636,0.299594,-0.905272,-0.109616,-0.914517,0.167786,0.148082,0.113949,1.115481,1.161836,1.388485]],[[53,39,3,8,-2,-2,-2,1,-2,-2,70,21,4,-2,-2,-2,42,8,-2,-2,-2],[0.5,0.5,0.9385000169277191,0.9434999823570251,-2.0,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,0.05849999934434891,-2.0,-2.0,-2.0,0.22474747896194458,0.6379999816417694,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[-0.00253,0.068788,0.081703,0.089796,0.742813,1.388223,-0.299422,-0.100277,-1.452654,1.168767,-0.036697,-0.059093,-0.069474,-0.086217,-0.651936,0.819535,0.10046,0.185644,1.529261,0.571557,-0.395159]],[[2,37,7,1,-2,-2,53,-2,-2,12,3,-2,-2,3,-2,-2,21,32,30,-2,-2,42,-2,-2,-2],[0.9350000023841858,0.5,0.001500000071246177,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.6550000011920929,0.02049999963492155,-2.0,-2.0,0.09750000014901161,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[-0.006367,0.072975,0.046967,0.103175,1.072382,1.109105,0.009664,0.418729,-0.559916,0.100007,0.132877,0.979347,1.177122,0.063629,1.162643,-0.330344,-0.043832,-0.053384,-0.06886,-0.527452,0.529665,0.088595,1.14614,-0.13324,0.948263]],[[2,9,8,74,-2,-2,12,-2,-2,53,42,-2,-2,-2,21,31,30,-2,-2,-2,-2],[0.9350000023841858,0.9675000011920929,0.153499998152256,0.0005000000237487257,-2.0,-2.0,0.801499992609024,-2.0,-2.0,0.5,0.2928571552038193,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[-0.003204,0.072359,0.044232,0.08964,1.077746,1.096101,-0.020337,-0.528564,0.838013,0.107456,0.084806,1.053911,0.34334,0.954058,-0.037131,-0.048079,-0.061787,-0.483093,0.56595,0.88094,1.111828]],[[8,4,69,4,-2,-2,-2,42,9,-2,-2,44,-2,-2,42,17,19,-2,-2,-2,74,3,-2,-2,-2],[0.09149999916553497,0.8485000133514404,0.5,0.001500000071246177,-2.0,-2.0,-2.0,0.2198067605495453,0.9784999787807465,-2.0,-2.0,0.5,-2.0,-2.0,0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,-0.5,0.04749999940395355,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.004993,0.083398,0.12774,0.089385,1.362435,0.769873,1.188817,0.043394,0.087078,0.423497,1.556702,-0.039304,0.725901,-0.871535,-0.021495,-0.000629,0.010842,0.161588,-0.976862,-1.424447,-0.1354,-0.149329,-1.481847,-0.990607,0.011281]],[[19,16,2,7,-2,-2,8,-2,-2,9,-2,-2,55,6,-2,-2,-2],[0.5,0.5,0.9750000238418579,0.23849999904632568,-2.0,-2.0,0.08900000154972076,-2.0,-2.0,0.5600000023841858,-2.0,-2.0,0.5,0.7094999849796295,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,-1],[-0.00775,0.006018,0.018862,0.055605,0.216713,0.961218,-0.04101,0.848688,-0.485069,-0.166026,-1.511708,-0.680725,-0.148932,-0.179823,-1.178304,-1.472521,-0.907815]],[[8,5,69,9,-2,-2,-2,-2,42,15,63,-2,-2,3,-2,-2,74,2,-2,-2,-2],[0.09149999916553497,0.37699998915195465,1.5,0.992499977350235,-2.0,-2.0,-2.0,-2.0,0.343137264251709,0.5,0.5,-2.0,-2.0,0.05900000035762787,-2.0,-2.0,-0.49899999995250255,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.006633,0.07445,0.082711,0.070178,0.51956,1.141498,0.933812,-0.301367,-0.017084,0.004293,-0.013563,-0.202147,1.046058,0.190498,1.513298,0.583532,-0.122445,-0.140916,-1.220678,-1.039243,0.187762]],[[74,54,70,4,-2,-2,42,-2,-2,-2,12,3,9,-2,-2,3,-2,-2,7,12,-2,-2,-2],[-0.5,0.5,0.5,0.351500004529953,-2.0,-2.0,0.2899159789085388,-2.0,-2.0,-2.0,0.6055000126361847,0.048499999567866325,0.9945000112056732,-2.0,-2.0,0.4194999933242798,-2.0,-2.0,0.04900000058114529,0.6315000057220459,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[-0.005556,-0.036388,-0.04529,-0.065364,-0.235129,-0.743137,0.050665,0.739309,-1.473275,0.9124,0.074484,0.095919,0.085415,1.050636,0.82619,0.135776,1.238037,0.840774,0.051648,0.074518,-0.108043,1.071131,-0.202468]],[[48,10,30,2,-2,-2,4,-2,-2,17,52,-2,-2,-2,42,6,-2,-2,-2],[0.5,0.2678571492433548,0.5,0.9249999821186066,-2.0,-2.0,0.13650000095367432,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.3541666716337204,0.9404999911785126,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,-1],[0.007077,-0.004135,-0.04164,-0.059694,0.270654,-0.544518,0.120555,1.294916,0.029824,0.046364,0.062986,0.640095,-1.212212,-1.330877,0.159259,0.223361,1.41563,0.829145,-0.009112]],[[74,42,9,4,-2,-2,8,-2,-2,2,-2,6,-2,-2,3,7,7,-2,-2,8,-2,-2,6,4,-2,-2,-2],[-0.5,0.33908046782016754,0.9494999945163727,0.029999999329447746,-2.0,-2.0,0.4479999989271164,-2.0,-2.0,0.9249999821186066,-2.0,0.3669999986886978,-2.0,-2.0,0.25300000607967377,0.08249999955296516,0.013500000350177288,-2.0,-2.0,0.0859999991953373,-2.0,-2.0,0.945000022649765,0.19049999862909317,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[-0.005082,-0.029556,-0.009466,-0.079951,0.595919,-0.781348,0.027718,1.077663,-0.043552,-0.117001,0.649253,-0.136626,-1.465268,-0.79011,0.059723,0.071151,0.063484,0.993904,0.016477,0.118547,1.098662,1.196632,-0.000463,-0.052826,0.310989,-0.927427,1.202683]],[[19,5,16,17,-2,-2,4,-2,-2,12,12,-2,-2,6,-2,-2,4,6,7,-2,-2,-2,-2],[0.5,0.9794999957084656,0.5,0.5,-2.0,-2.0,0.9004999995231628,-2.0,-2.0,0.5264999866485596,0.4259999990463257,-2.0,-2.0,0.45500001311302185,-2.0,-2.0,0.9664999842643738,0.9810000061988831,0.010999999940395355,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[0.003064,0.016053,0.031984,0.0449,0.3866,-0.700819,-0.12146,-1.279276,0.073206,-0.094196,-0.153525,-0.891932,-1.499041,-0.044146,-0.853483,0.189581,-0.116045,-0.133713,-0.101408,-0.565277,-1.22279,-1.30888,-0.486082]],[[2,74,3,3,-2,-2,3,-2,-2,3,75,-2,-2,-2,31,21,9,-2,-2,-2,-2],[0.9350000023841858,0.09149999916553497,0.10700000077486038,0.03849999979138374,-2.0,-2.0,0.7630000114440918,-2.0,-2.0,0.107999999076128,0.5277777910232544,-2.0,-2.0,-2.0,0.5,0.5,0.945499986410141,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[-0.002755,0.056707,0.068635,0.051707,0.829425,-0.71909,0.123218,1.253662,0.246311,0.021183,0.067522,-0.185966,1.109929,-0.313858,-0.032075,-0.046145,-0.057736,-0.769574,-0.20388,1.097431,1.090993]],[[19,2,16,17,-2,-2,-2,74,42,-2,-2,8,-2,-2,9,-2,12,-2,6,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,-0.5,0.11437908560037613,-2.0,-2.0,0.027500000782310963,-2.0,-2.0,0.5785000026226044,-2.0,0.4529999941587448,-2.0,0.8925000131130219,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[-0.002009,0.009702,0.037275,0.048457,0.427954,-0.79159,-0.746754,-0.036327,-0.049107,0.079839,-0.677053,0.114164,1.10387,1.299646,-0.103558,-0.338146,-0.118132,-0.675331,-0.142579,-1.146619,-1.293075]],[[19,2,42,7,-2,-2,2,-2,-2,5,8,-2,-2,6,-2,-2,7,42,-2,-2,2,-2,-2],[0.5,0.9750000238418579,0.19090909510850906,0.2880000025033951,-2.0,-2.0,0.9350000023841858,-2.0,-2.0,0.6755000054836273,0.05250000022351742,-2.0,-2.0,0.9914999902248383,-2.0,-2.0,0.007500000298023224,0.27142858505249023,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.005358,0.017342,0.046473,0.086998,0.482394,1.28168,-0.003143,0.688148,-0.318928,-0.02852,0.025064,1.030469,-0.082214,-0.070484,-0.667883,0.544952,-0.100511,-0.059711,-0.202698,-1.055127,-0.12706,-1.249486,-1.134889]],[[8,4,74,7,-2,-2,4,-2,-2,7,15,-2,-2,-2,65,54,53,-2,-2,-2,-2],[0.0885000005364418,0.9185000061988831,-0.5,0.0035000001080334187,-2.0,-2.0,0.6394999921321869,-2.0,-2.0,0.04800000041723251,0.5,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[0.007058,0.069671,0.110183,0.157642,1.348316,0.835902,0.065792,1.06785,1.105564,0.025413,0.04448,0.148859,1.235662,-0.903805,-0.014611,-0.024454,-0.032566,0.309758,-0.376179,1.038357,0.893776]],[[19,16,2,3,-2,-2,15,-2,-2,9,-2,-2,6,42,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.4975000023841858,-2.0,-2.0,0.5,-2.0,-2.0,0.9424999952316284,-2.0,-2.0,0.9810000061988831,0.18333333730697632,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,-1],[0.000936,0.011072,0.021087,0.07079,0.72761,1.026354,-0.004713,-0.141383,0.681689,-0.119897,-1.44568,0.426084,-0.104647,-0.083203,-1.188054,-0.640433,-1.171228]],[[10,15,30,44,-2,-2,-2,-2,17,7,8,-2,-2,42,-2,-2,-2],[0.2678571492433548,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.02949999924749136,0.28450000286102295,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.000266,-0.028987,-0.041341,-0.056662,-0.193579,-0.630993,0.641993,0.711117,0.037015,0.049643,0.078873,0.608443,1.31388,-0.008563,0.809234,-0.682763,-1.066499]],[[38,74,18,32,-2,-2,-2,3,74,-2,-2,8,-2,-2,42,6,9,-2,-2,-2,12,2,-2,-2,-2],[0.5,-0.5,0.5,0.5,-2.0,-2.0,-2.0,0.07499999925494194,0.15650000423192978,-2.0,-2.0,0.2645000070333481,-2.0,-2.0,0.22474747896194458,0.987500011920929,0.9839999973773956,-2.0,-2.0,-2.0,0.4919999986886978,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.000886,-0.01373,-0.036484,-0.047067,-0.378127,0.608108,0.536313,0.045761,0.059019,0.942052,0.996759,0.003291,-1.178731,0.39764,0.07199,0.119341,0.155833,1.078857,1.298536,-0.218118,-0.048255,0.041128,0.997587,-0.421452,-1.109828]],[[2,16,46,69,-2,-2,-2,-2,18,74,42,-2,-2,8,-2,-2,-2],[0.9600000083446503,0.5,1.5,2.5,-2.0,-2.0,-2.0,-2.0,0.5,-0.5,0.11437908560037613,-2.0,-2.0,0.05650000087916851,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.002487,0.036998,0.048831,0.056707,0.352279,1.102074,-1.07264,-0.86415,-0.042147,-0.053188,-0.063621,-0.11475,-0.812958,0.100496,0.900043,1.295379,0.835335]],[[19,5,7,3,-2,-2,9,-2,-2,4,2,-2,-2,-2,7,-2,2,-2,-2],[0.5,0.6994999945163727,0.22999999672174454,0.0625,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,0.882999986410141,0.9950000047683716,-2.0,-2.0,-2.0,0.007500000298023224,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,-1],[9.6e-05,0.009686,0.029364,0.016651,0.456333,-0.259919,0.118877,0.395413,1.610099,-0.035476,-0.01552,0.226722,-0.913889,-1.507625,-0.097829,-0.506776,-0.129253,-1.29304,-1.124616]],[[2,5,69,3,-2,-2,-2,-2,29,42,2,-2,-2,8,-2,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,0.7019999921321869,-2.0,-2.0,-2.0,-2.0,0.5,0.02777777798473835,0.9850000143051147,-2.0,-2.0,0.0885000005364418,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.003297,0.047817,0.057679,0.047886,0.744999,-0.507202,1.022817,-0.31539,-0.018354,-0.026728,0.02364,0.592615,-0.411329,-0.047663,0.254763,-0.479094,0.845691]],[[48,74,70,4,-2,-2,42,-2,-2,39,74,-2,-2,45,-2,-2,9,-2,-2],[0.5,-0.5,0.5,0.351500004529953,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,0.5,0.6959999799728394,-2.0,-2.0,0.006000000284984708,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.001692,-0.006401,-0.026632,-0.044934,-0.132535,-0.725015,0.056255,0.885532,-0.994521,0.040148,0.045642,0.566575,1.173051,-0.037564,-1.493669,1.133512,0.10125,-0.06309,1.135873]],[[54,64,2,39,-2,-2,38,-2,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-6.3e-05,-0.006609,-0.013531,0.035992,0.487544,-1.155223,-0.039788,-0.412222,0.297799,1.004403,1.283864]],[[54,42,2,7,-2,-2,8,-2,-2,5,75,-2,-2,6,-2,-2,-2],[0.5,0.44949495792388916,0.9750000238418579,0.9354999959468842,-2.0,-2.0,0.5929999947547913,-2.0,-2.0,0.019499999471008778,0.5833333432674408,-2.0,-2.0,0.9745000004768372,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.005811,-0.000706,0.009062,0.034007,0.178057,1.242161,-0.035943,0.142789,-0.553995,-0.107753,0.013832,1.112587,-0.384547,-0.137044,-1.028836,-1.585843,1.27373]],[[19,42,17,2,-2,-2,37,-2,-2,5,-2,2,-2,-2,12,-2,2,-2,-2],[0.5,0.4580419659614563,0.5,0.9750000238418579,-2.0,-2.0,0.5,-2.0,-2.0,0.019499999471008778,-2.0,0.9750000238418579,-2.0,-2.0,0.4529999941587448,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,-1],[-5.3e-05,0.010202,0.018542,0.026089,0.414639,-0.115084,-0.11316,-1.511803,-0.014222,-0.101268,0.518495,-0.140054,-1.275356,-1.018647,-0.094152,-0.573805,-0.115044,-1.285197,-1.113083]],[[42,15,30,8,-2,-2,8,-2,-2,8,9,-2,-2,-2,8,-2,2,7,-2,-2,2,-2,-2],[0.4580419659614563,0.5,0.5,0.1314999982714653,-2.0,-2.0,0.8334999978542328,-2.0,-2.0,0.4479999989271164,0.940500020980835,-2.0,-2.0,-2.0,0.032500000670552254,-2.0,0.9600000083446503,0.007500000298023224,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[-0.00337,0.005013,-0.006179,-0.016731,0.50885,-0.317079,0.116712,-0.334743,1.559812,0.10081,0.031516,-0.360641,1.147877,1.390417,-0.107547,0.692094,-0.134594,-0.212272,-0.964351,-1.642777,-0.079287,-1.135175,-0.554073]],[[19,42,17,70,-2,-2,9,-2,-2,8,9,-2,-2,74,-2,-2,42,69,-2,-2,-2],[0.5,0.33908046782016754,0.5,0.5,-2.0,-2.0,0.9650000035762787,-2.0,-2.0,0.032500000670552254,0.8145000040531158,-2.0,-2.0,0.050999999046325684,-2.0,-2.0,0.3166666775941849,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.001582,0.01202,0.026338,0.035309,0.185466,0.749188,-0.126968,-1.41244,-0.318431,-0.063793,0.069528,1.24656,0.102178,-0.106327,-1.110267,1.195991,-0.10475,-0.121141,-1.271764,-1.134516,-1.027727]],[[10,15,63,53,-2,-2,-2,-2,17,7,39,-2,-2,9,-2,-2,-2],[0.2678571492433548,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.02949999924749136,0.5,-2.0,-2.0,0.34300000965595245,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.00355,-0.030673,-0.03929,-0.048485,0.108688,-0.477382,0.495538,0.492652,0.031904,0.042729,0.07103,0.826849,-1.031785,-0.015499,-1.018983,0.388985,-1.281251]],[[42,69,2,5,-2,-2,7,-2,-2,-2,2,2,7,-2,-2,6,-2,-2,15,6,-2,-2,12,-2,-2],[0.02777777798473835,2.5,0.9950000047683716,0.9204999804496765,-2.0,-2.0,0.7954999804496765,-2.0,-2.0,-2.0,0.9350000023841858,0.8250000178813934,0.001500000071246177,-2.0,-2.0,0.46150000393390656,-2.0,-2.0,0.5,0.5590000152587891,-2.0,-2.0,0.5380000174045563,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.000694,0.048316,0.029003,0.05329,0.193809,1.755259,-0.067217,1.090335,-1.173135,1.480195,-0.019355,0.032961,0.010469,0.995282,-0.24359,0.120958,0.437954,1.204704,-0.039754,-0.052419,-0.031861,-0.612513,0.070391,1.005632,-0.425062]],[[70,74,4,6,-2,-2,12,-2,-2,37,42,-2,-2,6,-2,-2,42,9,9,-2,-2,-2,-2],[0.5,-0.5,0.351500004529953,0.7080000042915344,-2.0,-2.0,0.42499999701976776,-2.0,-2.0,2.5,0.4833333343267441,-2.0,-2.0,0.9395000040531158,-2.0,-2.0,0.25357143580913544,0.9914999902248383,0.9554999768733978,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[5e-06,-0.009395,-0.030588,-0.007536,-0.430109,0.357793,-0.070537,-1.027903,-0.2415,0.033448,0.043239,0.787917,-0.682861,-0.025719,-1.142797,1.089099,0.069725,0.152379,0.102016,1.284037,0.370996,1.317133,-1.165162]],[[1,4,12,42,-2,-2,30,-2,-2,5,46,-2,-2,12,-2,-2,46,45,3,-2,-2,45,-2,-2,74,-2,-2],[0.5,0.3310000002384186,0.48649999499320984,0.4365079402923584,-2.0,-2.0,0.5,-2.0,-2.0,0.018000000156462193,1.5,-2.0,-2.0,0.7975000143051147,-2.0,-2.0,1.5,0.2814999967813492,0.044999999925494194,-2.0,-2.0,0.80799999833107,-2.0,-2.0,0.036000000312924385,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,26,-1,-1],[-0.013052,-0.02633,-0.002315,0.053365,0.593477,-0.947399,-0.03508,-0.404468,0.802671,-0.072123,0.047235,0.707927,-0.914762,-0.094917,-0.846017,0.482556,0.041089,0.034893,0.051519,0.989933,1.048942,0.000565,-0.753277,0.940824,0.131962,1.101347,1.269171]],[[42,19,17,16,-2,-2,12,-2,-2,9,-2,6,-2,-2,5,-2,8,-2,3,-2,-2],[0.4580419659614563,0.5,0.5,0.5,-2.0,-2.0,0.4960000216960907,-2.0,-2.0,0.906000018119812,-2.0,0.9375,-2.0,-2.0,0.012500000186264515,-2.0,0.20600000023841858,-2.0,0.6790000200271606,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[0.003458,0.011715,0.020384,0.027473,0.292164,-0.608552,-0.102026,0.044577,-1.385905,-0.08556,-0.496375,-0.106725,-1.099643,-1.287675,-0.096271,0.314063,-0.120844,-1.45118,-0.092624,-0.588176,-1.36745]],[[11,63,51,44,-2,-2,-2,6,-2,-2,17,43,65,-2,-2,-2,-2],[0.20714285969734192,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.6420000195503235,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,-1],[-0.000413,-0.019732,-0.027922,-0.035847,-0.065902,-0.53964,0.734623,0.071178,1.121772,-0.264156,0.04135,0.0498,0.040941,0.375756,1.264897,0.825861,-0.806958]],[[42,69,38,6,-2,-2,-2,4,-2,6,-2,-2,5,12,69,-2,-2,2,-2,-2,4,42,-2,-2,1,-2,-2],[0.13076923787593842,1.5,0.5,0.3930000066757202,-2.0,-2.0,-2.0,0.0364999994635582,-2.0,0.9309999942779541,-2.0,-2.0,0.02949999924749136,0.5370000004768372,0.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.3790000081062317,0.3452381044626236,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.001524,0.03101,-0.000197,-0.020914,-0.865731,0.08409,1.253787,0.137074,1.772233,0.019134,0.84317,-0.800514,-0.018979,0.034033,0.08156,0.667413,1.084363,-0.024843,0.564902,-0.719294,-0.041715,-0.013117,0.145943,-0.963758,-0.086527,-0.912074,1.042505]],[[51,2,17,46,-2,-2,-2,38,12,-2,-2,6,-2,-2,-2],[0.5,0.9600000083446503,0.5,2.5,-2.0,-2.0,-2.0,0.5,0.24650000035762787,-2.0,-2.0,0.973499983549118,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[0.002552,-0.003599,0.01974,0.028468,0.305046,-1.466672,-1.012184,-0.03649,-0.051905,0.73662,-0.616855,0.059465,0.871956,-0.715962,0.963654]],[[16,5,17,19,-2,-2,5,-2,-2,44,2,-2,-2,-2,6,-2,-2],[0.5,0.6845000088214874,0.5,0.5,-2.0,-2.0,0.03999999910593033,-2.0,-2.0,0.5,0.9950000047683716,-2.0,-2.0,-2.0,0.9160000085830688,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.000815,0.006522,0.025071,0.03325,0.40406,-0.671795,-0.10687,-1.34104,-0.563506,-0.03546,-0.010327,0.298378,-0.94557,-1.535818,-0.108709,-1.246853,-0.167142]],[[64,43,2,39,-2,-2,4,-2,-2,12,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.08950000256299973,-2.0,-2.0,0.5259999930858612,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.00291,-0.002933,-0.010057,0.026279,0.454444,-0.958943,-0.027682,0.038027,-0.514849,0.081614,1.02586,-0.272638,1.090857]],[[42,2,5,11,-2,-2,3,-2,-2,7,-2,4,-2,-2,2,4,37,-2,-2,3,-2,-2,74,46,-2,-2,0,-2,-2],[0.029411764815449715,0.9850000143051147,0.21199999749660492,0.4166666716337204,-2.0,-2.0,0.061000000685453415,-2.0,-2.0,0.5690000057220459,-2.0,0.031000000424683094,-2.0,-2.0,0.9350000023841858,0.19299999624490738,1.5,-2.0,-2.0,0.09750000014901161,-2.0,-2.0,-0.5,2.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.00679,0.030961,0.069324,0.01761,-0.433662,1.110885,0.134336,0.521812,1.264966,-0.054545,0.121967,-0.088405,-1.173995,-0.340058,-0.023301,0.022561,0.063619,0.594879,1.256054,-0.007467,0.339726,-0.822425,-0.042542,-0.050066,-0.343733,-1.345552,0.084145,1.235376,0.635203]],[[54,16,2,17,-2,-2,12,-2,-2,9,-2,-2,-2],[0.5,0.5,0.9600000083446503,0.5,-2.0,-2.0,0.2575000077486038,-2.0,-2.0,0.9424999952316284,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.006753,0.001459,0.007514,0.028585,0.339753,-0.921174,-0.020415,1.096235,-0.30281,-0.093235,-1.097766,0.201385,1.193191]],[[54,19,42,69,-2,-2,66,-2,-2,55,6,-2,-2,-2,-2],[0.5,0.5,0.11324786394834518,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.7094999849796295,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[0.008684,0.003926,0.012229,0.048964,0.226908,1.575565,-0.009283,-0.012641,-1.582042,-0.077817,-0.101624,-1.092874,-1.204675,-0.643775,1.11782]],[[42,69,6,38,-2,-2,5,-2,-2,4,-2,-2,2,39,45,-2,-2,-2,66,5,-2,-2,-2],[0.11324786394834518,1.5,0.972000002861023,0.5,-2.0,-2.0,0.20400000363588333,-2.0,-2.0,0.0364999994635582,-2.0,-2.0,0.9350000023841858,0.5,0.5305000245571136,-2.0,-2.0,-2.0,0.5,0.02850000001490116,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,-1],[-0.000638,0.02783,0.001662,-0.025788,-0.443025,1.224785,0.067054,0.243542,1.282188,0.119951,1.613548,0.224901,-0.016807,0.026791,0.03526,0.628965,-0.455024,-0.841752,-0.036579,-0.029442,0.230249,-0.389374,-1.247615]],[[16,19,5,42,-2,-2,-2,8,8,-2,-2,-2,44,-2,-2],[0.5,0.5,0.9964999854564667,0.02777777798473835,-2.0,-2.0,-2.0,0.7374999821186066,0.40150000154972076,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,14,-1,-1],[0.0011,0.006951,0.014391,0.01888,0.529893,0.024141,-1.295752,-0.065971,-0.032197,-0.929508,0.195801,-1.250938,-0.098032,-1.537989,-0.491188]],[[8,4,69,8,-2,-2,-2,7,12,-2,-2,-2,62,6,6,-2,-2,48,-2,-2,-2],[0.09149999916553497,0.8700000047683716,0.5,0.022499999962747097,-2.0,-2.0,-2.0,0.02650000061839819,0.47450000047683716,-2.0,-2.0,-2.0,0.5,0.2524999976158142,0.12000000104308128,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.001769,0.042855,0.07496,0.045464,0.542528,0.938905,1.028656,0.012613,0.036308,-0.209697,0.942735,-0.671283,-0.01723,-0.022869,-0.078338,-0.497694,-1.343333,-0.006147,-0.164867,0.896722,0.752767]],[[54,5,69,3,-2,-2,-2,4,6,-2,-2,44,-2,-2,-2],[0.5,0.02850000001490116,2.5,0.8495000004768372,-2.0,-2.0,-2.0,0.07200000062584877,0.9905000030994415,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[-0.003362,-0.007619,0.03079,0.016941,0.429378,-1.345315,0.872527,-0.021863,0.010281,-0.064364,1.478012,-0.045727,-0.131293,-0.638532,0.917955]],[[42,69,38,12,-2,-2,-2,5,2,-2,-2,-2,12,42,-2,3,-2,-2,2,15,-2,-2,20,-2,-2],[0.12916667014360428,1.5,0.5,0.3490000069141388,-2.0,-2.0,-2.0,0.851500004529953,0.9750000238418579,-2.0,-2.0,-2.0,0.21849999576807022,0.3166666775941849,-2.0,0.001500000071246177,-2.0,-2.0,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,15,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.001271,0.031406,-0.001133,-0.020299,-1.361621,-0.03042,1.22462,0.133492,0.072515,0.802084,-0.041757,1.799963,-0.019456,0.093961,1.836593,0.001215,1.073362,-0.59932,-0.025691,0.020277,0.460172,-0.81767,-0.043484,-1.261874,-0.302737]],[[32,8,11,74,-2,-2,9,-2,-2,17,42,-2,-2,-2,12,-2,12,-2,-2],[0.5,0.09149999916553497,0.36666667461395264,-0.5,-2.0,-2.0,0.976500004529953,-2.0,-2.0,0.5,0.33908046782016754,-2.0,-2.0,-2.0,0.408500000834465,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,-1],[-0.002718,-0.010098,0.033668,-0.017662,-0.725922,1.062367,0.048072,0.212727,1.056387,-0.024553,-0.018905,-0.055486,-0.896147,-1.082577,0.068113,1.490108,0.025307,-0.539428,1.122515]],[[2,7,3,7,-2,-2,5,-2,-2,42,8,-2,-2,-2,18,9,-2,12,-2,-2,-2],[0.9600000083446503,0.23349999636411667,0.7254999876022339,0.006500000134110451,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.39444445073604584,0.9465000033378601,-2.0,-2.0,-2.0,0.5,0.1719999983906746,-2.0,0.1860000044107437,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,-1],[0.001493,0.021832,0.003892,0.019999,0.574774,-0.1759,-0.081609,-1.247614,-0.422404,0.109359,0.137604,0.538822,1.404459,-0.537727,-0.025935,-0.034082,0.859128,-0.04043,1.340589,-0.473258,0.68526]],[[16,54,19,2,-2,-2,42,-2,-2,-2,4,-2,-2],[0.5,0.5,0.5,0.9350000023841858,-2.0,-2.0,0.3095238208770752,-2.0,-2.0,-2.0,0.9105000197887421,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[-0.007131,-0.001397,-0.005246,0.002086,0.491557,-0.108668,-0.073798,-1.136273,-0.933499,0.900071,-0.095505,-1.428862,0.104311]],[[17,42,21,19,-2,-2,-2,5,9,-2,-2,2,-2,-2,4,-2,-2],[0.5,0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,0.006500000134110451,0.8160000145435333,-2.0,-2.0,0.875,-2.0,-2.0,0.5965000092983246,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.005444,0.000147,0.009421,0.005311,0.10845,-0.899775,1.214928,-0.045041,0.075184,1.36347,0.485689,-0.067403,0.788724,-1.06615,-0.105416,-1.445139,-0.406264]],[[19,5,5,70,-2,-2,-2,2,42,-2,-2,5,-2,-2,9,-2,2,-2,-2],[0.5,0.9715000092983246,0.9535000026226044,0.5,-2.0,-2.0,-2.0,0.9850000143051147,0.36666667461395264,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,0.6680000126361847,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[-5.2e-05,0.006508,0.016227,0.010064,0.01438,0.567323,1.27917,-0.049557,0.02187,0.402527,-1.258196,-0.084343,-0.860794,-1.179651,-0.065871,-0.281784,-0.077636,-1.229967,-1.074445]],[[39,6,70,12,-2,-2,42,-2,-2,12,12,-2,-2,7,-2,-2,12,-2,-2],[0.5,0.948500007390976,0.5,0.17599999904632568,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.31049999594688416,0.25950001180171967,-2.0,-2.0,0.22299999743700027,-2.0,-2.0,0.4544999897480011,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.001345,0.002469,-0.014244,-0.025939,1.381926,-0.280267,0.066774,1.140972,-0.945743,0.028676,-0.0632,0.259598,-1.325967,0.043932,0.270364,1.14962,-0.112892,0.42726,-1.391209]],[[5,6,3,7,-2,-2,-2,30,3,-2,-2,-2,4,2,7,-2,-2,6,-2,-2,-2],[0.6755000054836273,0.02850000001490116,0.13700000196695328,0.05300000123679638,-2.0,-2.0,-2.0,0.5,0.7045000195503235,-2.0,-2.0,-2.0,0.8935000002384186,0.9850000143051147,0.8709999918937683,-2.0,-2.0,0.976500004529953,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.001035,0.011819,0.110749,0.049454,1.19025,0.050074,1.742499,0.003985,-0.003004,0.122798,-0.668598,1.017883,-0.032066,-0.016441,0.044404,-0.300263,1.005761,-0.070166,-1.033416,0.014216,-1.389478]],[[71,11,4,37,-2,-2,4,-2,-2,44,8,-2,-2,9,-2,-2,-2],[0.5,0.36666667461395264,0.351500004529953,1.5,-2.0,-2.0,0.5189999938011169,-2.0,-2.0,0.5,0.12949999794363976,-2.0,-2.0,0.6504999995231628,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.003901,0.006607,-0.004597,0.016245,-0.0867,0.503892,-0.040342,-1.283997,-0.214789,0.042977,0.01484,0.624069,-0.8146,0.088472,0.049377,1.235423,-0.92033]],[[16,42,69,6,-2,-2,4,-2,-2,69,12,-2,-2,5,-2,-2,4,-2,-2],[0.5,0.13076923787593842,1.5,0.3095000088214874,-2.0,-2.0,0.054500000551342964,-2.0,-2.0,1.5,0.2044999971985817,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.8720000088214874,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.004582,0.002377,0.031205,0.006613,-0.680907,0.322305,0.110669,1.328997,0.007691,-0.019212,-0.002976,1.445267,-0.108639,-0.067921,0.26778,-0.722057,-0.099834,-1.327619,-0.359172]],[[35,42,69,5,-2,-2,2,-2,-2,44,5,-2,-2,11,-2,-2,-2],[0.5,0.03125,0.5,0.49650000035762787,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.9905000030994415,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.005396,0.001568,0.030804,-0.010068,0.256152,-0.695772,0.100265,1.180264,0.075709,-0.011118,0.010947,0.329612,-0.582118,-0.036949,-0.496764,0.266818,1.007758]],[[42,19,3,9,-2,-2,-2,6,-2,-2,8,3,9,-2,-2,-2,45,8,-2,-2,-2],[0.33974359929561615,0.5,0.99549999833107,0.945499986410141,-2.0,-2.0,-2.0,0.9874999821186066,-2.0,-2.0,0.0409999992698431,0.007500000298023224,0.6840000152587891,-2.0,-2.0,-2.0,0.0024999999441206455,0.20600000023841858,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.004227,0.014225,0.020751,0.02424,-0.085947,0.380584,-1.107018,-0.060007,-0.655551,-1.119852,-0.045258,0.036988,0.072793,1.46535,0.611042,-0.465884,-0.069285,-0.079645,-1.449727,-0.906398,0.843176]],[[10,12,5,69,-2,-2,2,-2,-2,5,69,-2,-2,63,-2,-2,4,8,11,-2,-2,-2,7,8,-2,-2,9,-2,-2],[0.2678571492433548,0.3994999974966049,0.053999999538064,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.00849999999627471,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.16099999845027924,0.2590000033378601,0.4166666716337204,-2.0,-2.0,-2.0,0.007500000298023224,0.06149999983608723,-2.0,-2.0,0.9479999840259552,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.001082,-0.016546,-0.07043,0.064734,1.273585,-0.456844,-0.105069,-1.179603,-0.733745,-0.00017,-0.134596,-0.114008,-1.318701,0.008509,-0.018601,0.954088,0.024097,0.069644,0.045475,-0.349951,0.989293,1.309973,0.00379,0.046215,0.250272,1.045299,-0.047909,-0.779411,0.192138]],[[17,2,46,7,-2,-2,-2,5,6,-2,-2,37,-2,-2,42,-2,-2],[0.5,0.9750000238418579,1.5,0.8264999985694885,-2.0,-2.0,-2.0,0.2619999945163727,0.010499999858438969,-2.0,-2.0,0.5,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.004046,0.000988,0.01652,0.02558,0.150466,1.009044,-1.21471,-0.024915,0.019053,2.118893,0.017794,-0.048697,-1.164627,-0.263569,-0.101394,-1.247654,-0.221231]],[[19,42,17,4,-2,-2,9,-2,-2,5,12,-2,-2,74,-2,-2,42,2,-2,-2,-2],[0.5,0.33908046782016754,0.5,0.0345000009983778,-2.0,-2.0,0.9650000035762787,-2.0,-2.0,0.030499999411404133,0.5690000057220459,-2.0,-2.0,0.050999999046325684,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.002906,0.009737,0.01785,0.022948,0.430476,0.099222,-0.070203,-1.268616,-0.029432,-0.032282,0.033112,0.873908,-0.637103,-0.059423,-0.870822,1.224823,-0.062074,-0.078358,-1.159919,-1.064241,-0.752776]],[[5,3,44,2,-2,-2,6,-2,-2,8,-2,-2,12,6,8,-2,-2,-2,2,-2,-2],[0.9794999957084656,0.9895000159740448,0.5,0.9950000047683716,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.7409999966621399,0.9699999988079071,0.9774999916553497,-2.0,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,-1],[0.000453,0.008473,0.004791,0.027155,0.607156,-0.550473,-0.018683,1.311537,-0.223067,0.118327,1.29808,-0.443505,-0.053329,-0.084829,-0.062675,-1.259208,-0.405417,-1.352715,0.004656,0.300229,-1.103296]],[[42,19,9,4,-2,-2,8,-2,-2,6,-2,69,-2,-2,5,12,-2,-2,8,-2,3,-2,-2],[0.4580419659614563,0.5,0.945499986410141,0.02949999924749136,-2.0,-2.0,0.40949998795986176,-2.0,-2.0,0.7094999849796295,-2.0,0.5,-2.0,-2.0,0.019499999471008778,0.39650000631809235,-2.0,-2.0,0.210999995470047,-2.0,0.6389999985694885,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,19,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,20,-1,22,-1,-1],[-0.00091,0.004232,0.009925,-0.011545,0.834505,-0.340972,0.023913,0.995343,0.042524,-0.060031,-0.269856,-0.084674,-1.233281,-1.108,-0.06169,0.014688,1.100773,-0.166051,-0.084641,-1.313619,-0.057524,-0.680286,-1.278109]],[[64,5,6,3,-2,-2,3,-2,-2,3,6,-2,-2,9,-2,-2,-2],[0.5,0.6545000076293945,0.015500000212341547,0.1055000014603138,-2.0,-2.0,0.7260000109672546,-2.0,-2.0,0.10449999943375587,0.8664999902248383,-2.0,-2.0,0.9494999945163727,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.001009,-0.005152,0.007258,0.131038,0.847,1.928175,0.000378,0.137877,-0.58254,-0.033795,-0.112758,-0.368593,-1.431865,-0.008299,-1.001278,0.126753,0.867417]],[[8,8,3,6,-2,-2,-2,-2,37,42,63,-2,-2,12,-2,-2,-2],[0.0925000011920929,0.08249999955296516,0.048499999567866325,0.7590000033378601,-2.0,-2.0,-2.0,-2.0,4.5,0.22649572789669037,0.5,-2.0,-2.0,0.5394999980926514,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-1.6e-05,0.032774,0.026351,0.019872,0.830456,0.02307,0.872855,1.395934,-0.011504,-0.007093,0.00919,-0.007408,0.818626,-0.038247,0.035928,-0.882177,-0.829601]],[[44,2,4,6,-2,-2,4,-2,-2,6,5,-2,-2,-2,6,-2,4,11,-2,-2,3,-2,-2],[0.5,0.9950000047683716,0.4389999955892563,0.9824999868869781,-2.0,-2.0,0.5725000202655792,-2.0,-2.0,0.992499977350235,0.5449999868869781,-2.0,-2.0,-2.0,0.019999999552965164,-2.0,0.9715000092983246,0.4166666716337204,-2.0,-2.0,0.005499999970197678,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[-0.00098,0.018037,0.032574,0.061278,0.441516,1.909651,-0.002926,-1.17364,0.149519,-0.039692,-0.054301,1.099849,-1.096659,1.133472,-0.025103,1.178554,-0.03378,-0.044868,-0.477842,0.158662,0.049181,-0.181691,1.288753]],[[5,19,17,10,-2,-2,37,-2,-2,6,-2,42,-2,-2,8,-2,8,-2,12,-2,-2],[0.9835000038146973,0.5,0.5,0.2678571492433548,-2.0,-2.0,0.5,-2.0,-2.0,0.7094999849796295,-2.0,0.3166666775941849,-2.0,-2.0,0.9799999892711639,-2.0,0.9895000159740448,-2.0,0.8680000007152557,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[-0.003705,0.00348,0.010379,0.015253,-0.024758,0.400314,-0.06764,-1.292099,0.084275,-0.050746,-0.467664,-0.065318,-1.119963,-0.591317,-0.059435,-1.36554,-0.037595,0.433688,-0.06598,-0.976165,0.213923]],[[17,42,70,6,-2,-2,6,-2,-2,3,7,-2,-2,12,-2,-2,42,-2,-2],[0.5,0.2752525359392166,0.5,0.9514999985694885,-2.0,-2.0,0.9354999959468842,-2.0,-2.0,0.3059999942779541,0.006500000134110451,-2.0,-2.0,0.8564999997615814,-2.0,-2.0,0.3062500059604645,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.003825,0.00812,0.020454,0.011916,-0.124802,0.499611,0.074054,1.079833,-0.071711,-0.023335,-0.000819,0.463329,-0.385906,-0.062942,-1.066433,0.378733,-0.076912,-1.025607,0.067768]],[[5,44,2,7,-2,-2,46,-2,-2,6,-2,6,-2,-2,2,-2,42,-2,12,-2,-2],[0.9895000159740448,0.5,0.9950000047683716,0.9505000114440918,-2.0,-2.0,1.5,-2.0,-2.0,0.017999999690800905,-2.0,0.24050000309944153,-2.0,-2.0,0.9750000238418579,-2.0,0.19090909510850906,-2.0,0.5705000162124634,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[-0.002362,0.003537,0.024548,0.040057,0.377982,1.51009,-0.039859,-1.058138,0.988034,-0.018089,1.276759,-0.026752,-0.937079,-0.128369,-0.060959,0.023138,-0.080433,-1.093205,-0.050914,-1.145978,-1.048893]],[[48,65,2,8,-2,-2,7,-2,-2,9,-2,-2,6,-2,-2],[0.5,0.5,0.9249999821186066,0.9359999895095825,-2.0,-2.0,0.06599999964237213,-2.0,-2.0,0.9244999885559082,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.001077,-0.003662,-0.008321,0.01707,0.172699,0.80465,-0.021428,-0.37887,-0.012635,0.061075,1.097512,0.125343,0.067013,-0.565159,0.848502]],[[19,2,7,46,-2,-2,6,-2,-2,37,38,-2,-2,42,-2,-2,42,2,-2,-2,-2],[0.5,0.9750000238418579,0.2395000010728836,1.5,-2.0,-2.0,0.4520000070333481,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.11678004637360573,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.003134,0.009492,0.024509,0.007602,0.142108,-0.950508,0.108163,-0.009191,1.358134,-0.01551,-0.035813,-0.560069,0.683742,0.023264,0.924792,-0.332123,-0.060116,-0.072445,-1.113885,-1.077521,-0.926242]],[[69,42,12,6,-2,-2,69,-2,-2,-2,-2],[4.5,0.7321428656578064,0.2580000013113022,0.792000025510788,-2.0,-2.0,2.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.003156,-0.000213,0.001103,0.066676,1.671309,-0.171878,-0.003714,-0.106628,0.444224,-1.752383,-0.687907]],[[44,16,2,8,-2,-2,6,-2,-2,-2,5,69,5,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.9850000143051147,0.9444999992847443,-2.0,-2.0,0.99549999833107,-2.0,-2.0,-2.0,0.6425000131130219,2.5,0.382999986410141,-2.0,-2.0,0.17914438992738724,-2.0,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,-1],[-0.005978,0.008564,0.014005,0.036599,0.28409,0.79077,-0.029705,-0.652677,1.049026,-1.34348,-0.025146,-0.012901,-0.027633,-0.35124,0.666115,0.06339,1.10583,-0.443831,-0.12281,-1.468386,0.220357]],[[16,42,63,4,-2,-2,-2,8,69,-2,-2,53,-2,-2,9,-2,-2],[0.5,0.22361110895872116,0.5,0.05300000123679638,-2.0,-2.0,-2.0,0.8939999938011169,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9584999978542328,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.000192,0.00415,0.01834,0.010739,0.332337,-0.063562,0.842923,-0.020846,-0.000866,0.224211,-0.667805,-0.072134,-1.285438,-0.816174,-0.068297,-1.050006,0.372737]],[[16,5,44,3,-2,-2,4,-2,-2,42,-2,2,-2,-2,4,-2,-2],[0.5,0.9835000038146973,0.5,0.08049999922513962,-2.0,-2.0,0.030499999411404133,-2.0,-2.0,0.04545454680919647,-2.0,0.9850000143051147,-2.0,-2.0,0.8734999895095825,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,-1],[-0.006814,-0.00213,0.00531,0.028217,0.025718,0.669662,-0.019421,0.215053,-0.299781,-0.057341,-0.013249,-0.079818,-1.196035,-1.112043,-0.081881,-1.307718,-0.244403]],[[56,33,51,16,-2,-2,-2,-2,6,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.9695000052452087,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,-1,-1],[8,7,6,5,-1,-1,-1,-1,10,-1,-1],[-0.004647,-0.000818,0.002063,-0.002008,0.044133,-0.878843,0.68506,-1.091165,-0.073135,0.106898,-1.063781]],[[16,4,4,37,-2,-2,-2,69,2,-2,-2,5,-2,-2,4,-2,-2],[0.5,0.05849999934434891,0.05049999989569187,1.5,-2.0,-2.0,-2.0,1.5,0.9750000238418579,-2.0,-2.0,0.6525000035762787,-2.0,-2.0,0.8650000095367432,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.001667,0.003075,0.025912,0.015963,-0.096106,0.654763,1.383743,-0.012,0.004377,0.25939,-0.5156,-0.06256,-0.146735,-1.079903,-0.078816,-1.288182,-0.145782]],[[42,4,4,4,-2,-2,-2,44,3,-2,-2,8,-2,-2,5,12,-2,-2,3,4,-2,-2,-2],[0.42217741906642914,0.05849999934434891,0.04649999924004078,0.03749999962747097,-2.0,-2.0,-2.0,0.5,0.7985000014305115,-2.0,-2.0,0.025500000454485416,-2.0,-2.0,0.019499999471008778,0.5595000088214874,-2.0,-2.0,0.9860000014305115,0.9050000011920929,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,19,20,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,22,21,-1,-1,-1],[0.003449,0.009424,0.035081,0.023592,0.34534,-0.771797,1.150783,-0.006055,0.020727,0.160749,1.854564,-0.039075,-1.188735,-0.212808,-0.046775,0.037737,1.108522,-0.484807,-0.075571,-0.062938,-0.650373,-1.297261,-1.72465]],[[64,54,9,74,-2,-2,63,-2,-2,-2,-2],[0.5,0.5,0.945499986410141,-0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.008426,0.004452,0.001024,-0.024472,-0.550408,0.422952,0.017272,0.08838,0.982538,0.903303,1.0523]],[[42,4,3,-2,6,-2,-2,4,69,-2,-2,12,-2,-2,66,5,5,-2,-2,1,-2,-2,-2],[0.11324786394834518,0.006500000134110451,0.9505000114440918,-2.0,0.1510000005364418,-2.0,-2.0,0.03350000083446503,0.5,-2.0,-2.0,0.3709999918937683,-2.0,-2.0,0.5,0.02850000001490116,0.019499999471008778,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,7,4,-1,6,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.001551,0.023961,-0.054642,-0.050043,-0.101206,-0.620107,-1.023237,0.038155,0.117958,0.231139,1.497264,0.016731,-0.765359,0.382594,-0.010875,-0.006531,0.027544,0.13172,1.203642,-0.020334,-0.276316,0.514857,-1.068145]],[[71,42,4,4,-2,-2,6,-2,-2,12,-2,-2,-2],[0.5,0.612500011920929,0.05849999934434891,0.0494999997317791,-2.0,-2.0,0.26999999582767487,-2.0,-2.0,0.4650000035762787,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.000115,0.002061,0.004219,0.021708,0.114194,1.720389,-0.007108,-0.986878,0.002236,-0.111093,-1.461798,-1.653519,-0.805343]],[[11,63,44,3,-2,-2,9,-2,-2,-2,9,4,9,-2,-2,4,-2,-2,44,13,-2,-2,12,-2,-2],[0.36666667461395264,0.5,0.5,0.9835000038146973,-2.0,-2.0,0.9934999942779541,-2.0,-2.0,-2.0,0.7730000019073486,0.09699999913573265,0.029999999329447746,-2.0,-2.0,0.6344999969005585,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5164999961853027,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.001003,-0.010988,-0.015914,0.004159,0.187147,-1.045942,-0.042338,-0.145691,-0.791752,0.434099,0.027996,-0.014029,0.069962,1.098128,0.947862,-0.047035,-1.681845,-0.271483,0.070508,0.016878,0.62902,-1.236875,0.138533,0.819709,1.436791]],[[6,64,6,70,-2,-2,7,-2,-2,-2,12,4,-2,69,-2,-2,12,3,-2,-2,2,-2,-2],[0.9575000107288361,0.5,0.9165000021457672,0.5,-2.0,-2.0,0.03749999962747097,-2.0,-2.0,-2.0,0.5024999976158142,0.017500000074505806,-2.0,1.5,-2.0,-2.0,0.6270000040531158,0.21699999272823334,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.001781,-0.014189,-0.020683,-0.011615,-0.214359,0.561459,-0.075507,-1.326526,-0.087767,1.026966,0.018922,-0.012173,0.800916,-0.026323,-0.040488,-0.998793,0.057109,0.126411,0.857439,2.067716,0.004866,0.843875,-0.889609]],[[48,5,42,70,-2,-2,42,-2,-2,3,6,-2,-2,6,-2,-2,8,-2,-2],[0.5,0.5494999885559082,0.2768199294805527,0.5,-2.0,-2.0,0.2952069789171219,-2.0,-2.0,0.6765000224113464,0.9835000038146973,-2.0,-2.0,0.3265000134706497,-2.0,-2.0,0.45000000298023224,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.004759,-0.000949,0.013481,0.028255,0.194546,1.060385,-0.020674,-1.072493,-0.055343,-0.028097,-0.057735,-0.385601,-1.212122,0.005651,-0.628183,0.603533,0.077967,-0.046952,1.057257]],[[3,42,11,44,-2,-2,9,-2,-2,-2,7,-2,-2],[0.99549999833107,0.7321428656578064,0.36666667461395264,0.5,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,-2.0,0.9560000002384186,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[-0.001632,0.001274,0.002484,-0.006975,0.18488,-0.298665,0.027643,0.134058,1.085003,-1.672837,-0.12827,-1.447105,-1.311722]],[[67,12,8,5,-2,-2,-2,12,42,-2,-2,6,-2,-2,-2],[0.5,0.2604999989271164,0.6084999740123749,0.15850000083446503,-2.0,-2.0,-2.0,0.3110000044107437,0.17424242943525314,-2.0,-2.0,0.9575000107288361,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[0.002633,-0.00045,0.055594,0.011765,1.20606,-0.969968,1.420885,-0.004754,-0.071321,-1.184867,-0.31892,0.00043,-0.154164,0.303872,0.821807]],[[17,38,16,6,-2,-2,42,-2,-2,46,42,-2,-2,-2,9,-2,-2],[0.5,0.5,0.5,0.7054999768733978,-2.0,-2.0,0.18333333730697632,-2.0,-2.0,2.5,0.2613636404275894,-2.0,-2.0,-2.0,0.93299999833107,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.003366,0.000303,-0.006952,-0.002162,-0.281651,0.144378,-0.067437,-1.188896,-0.211956,0.032871,0.045433,0.94601,-0.501129,-1.100885,-0.080673,-1.226952,-0.557405]],[[42,9,9,69,-2,-2,-2,5,4,-2,-2,3,-2,-2,5,42,-2,6,-2,-2,4,42,-2,-2,-2],[0.02777777798473835,0.9945000112056732,0.9914999902248383,0.5,-2.0,-2.0,-2.0,0.15850000083446503,0.3474999889731407,-2.0,-2.0,0.9595000147819519,-2.0,-2.0,0.5494999885559082,0.10435571894049644,-2.0,0.02850000001490116,-2.0,-2.0,0.8935000002384186,0.12916667014360428,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,18,-1,-1,21,22,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,17,-1,19,-1,-1,24,23,-1,-1,-1],[-0.004204,0.022811,0.038563,0.018888,-0.13809,0.915294,1.2482,-0.023993,-0.108369,-1.287236,1.065849,0.020028,-0.332157,0.87427,-0.015399,-0.000947,-0.886956,0.005372,1.145884,-0.011396,-0.047367,-0.031382,0.530802,-0.562444,-1.224198]],[[71,12,8,5,-2,-2,-2,12,53,-2,-2,6,-2,-2,-2],[0.5,0.2580000013113022,0.6084999740123749,0.15850000083446503,-2.0,-2.0,-2.0,0.3110000044107437,0.5,-2.0,-2.0,0.9755000174045563,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[-8.9e-05,0.002464,0.056884,0.016081,1.074918,-0.988034,1.33666,-0.001684,-0.064125,0.391468,-1.064261,0.003186,-0.112041,0.432271,-0.95921]],[[42,44,2,3,-2,-2,46,-2,-2,6,-2,38,-2,-2,2,-2,-2],[0.612500011920929,0.5,0.9950000047683716,0.804999977350235,-2.0,-2.0,1.5,-2.0,-2.0,0.02850000001490116,-2.0,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,-1],[0.000403,0.002376,0.015772,0.028969,0.183946,0.937544,-0.031375,-0.856388,0.904595,-0.015044,0.991517,-0.021754,-0.295412,0.359738,-0.106107,-1.628417,-1.271472]],[[3,15,44,2,-2,-2,6,-2,-2,9,3,-2,-2,-2,-2],[0.9964999854564667,0.5,0.5,0.9950000047683716,-2.0,-2.0,0.010999999940395355,-2.0,-2.0,0.992499977350235,0.04050000011920929,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[0.004689,0.006813,0.001597,0.015493,0.448883,-0.602921,-0.016082,1.594564,-0.202525,0.047503,0.019422,0.719447,-0.331123,1.226395,-1.406047]],[[12,8,5,42,-2,-2,-2,-2,67,17,2,-2,-2,4,-2,-2,-2],[0.2580000013113022,0.6704999804496765,0.15850000083446503,0.25833334028720856,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.9750000238418579,-2.0,-2.0,0.6590000092983246,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.002525,0.058708,0.016246,0.086396,1.373239,0.763399,-0.850363,1.585881,-0.001546,-0.004384,-0.000651,0.130809,-0.261712,-0.082324,-1.436304,0.028564,0.657128]],[[42,70,50,16,-2,-2,-2,9,-2,-2,8,9,3,-2,-2,-2,1,42,-2,-2,-2],[0.2695630192756653,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9819999933242798,-2.0,-2.0,0.07050000131130219,0.99549999833107,0.013500000350177288,-2.0,-2.0,-2.0,0.5,0.2970588356256485,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.002405,0.012974,0.004454,0.009361,0.158112,-0.776092,-0.960541,0.075635,0.475014,1.044203,-0.02255,0.026696,0.009725,0.577897,-1.112559,1.372278,-0.039073,-0.047305,-1.012461,-0.500206,0.958032]],[[69,39,15,30,-2,-2,8,-2,-2,5,-2,-2,-2],[4.5,0.5,0.5,0.5,-2.0,-2.0,0.45000000298023224,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.000959,0.003466,0.005922,0.001082,-0.036971,0.602331,0.043581,-0.003561,0.859853,-0.070557,1.387743,-1.042934,-0.83481]],[[8,5,8,9,-2,-2,-2,12,-2,-2,63,12,-2,7,-2,-2,-2],[0.6324999928474426,0.6044999957084656,0.5694999992847443,0.973499983549118,-2.0,-2.0,-2.0,0.3955000042915344,-2.0,-2.0,0.5,0.26100000739097595,-2.0,0.23499999940395355,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,-1,14,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,13,-1,15,-1,-1,-1],[0.000116,0.019392,0.026046,0.021294,0.095286,0.680515,1.041731,-0.052969,-1.089163,-0.018836,-0.02005,-0.028335,1.051141,-0.034505,-0.586973,-0.075554,0.689982]],[[17,66,42,51,-2,-2,8,-2,-2,4,-2,-2,42,-2,-2],[0.5,0.5,0.33908046782016754,0.5,-2.0,-2.0,0.06300000101327896,-2.0,-2.0,0.042000001296401024,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.002721,0.006087,0.009208,0.01659,0.110252,1.229153,-0.02661,0.67908,-0.782151,-0.074438,0.090437,-1.094522,-0.06803,-1.013701,0.008044]],[[5,43,56,19,-2,-2,4,-2,-2,7,-2,-2,-2],[0.9964999854564667,0.5,0.5,0.5,-2.0,-2.0,0.632999986410141,-2.0,-2.0,0.024500000290572643,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.004238,0.007195,0.003083,0.006538,0.124181,-0.726519,-0.063537,-0.041136,-1.239727,0.062561,0.825046,-0.028015,-1.070143]],[[3,44,8,5,-2,-2,2,-2,-2,6,-2,6,-2,-2,4,2,-2,-2,-2],[0.9905000030994415,0.5,0.9794999957084656,0.9715000092983246,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.02850000001490116,-2.0,0.24050000309944153,-2.0,-2.0,0.0025000000605359674,0.9650000035762787,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,-1],[-0.00275,0.000648,0.014283,0.003896,0.207803,-1.005805,0.059724,1.17978,-0.117467,-0.017315,1.130581,-0.025034,-1.035169,-0.128501,-0.072567,-0.110726,-1.293383,-0.935693,-0.069291]],[[39,5,69,12,-2,-2,42,-2,-2,4,2,-2,-2,9,-2,-2,5,-2,-2],[0.5,0.6064999997615814,2.5,0.9124999940395355,-2.0,-2.0,0.22401433438062668,-2.0,-2.0,0.8219999969005585,0.9750000238418579,-2.0,-2.0,0.9614999890327454,-2.0,-2.0,0.02350000012665987,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.004112,-0.001232,0.008849,0.002273,-0.030426,1.199824,0.072105,0.839647,-0.548007,-0.023466,-0.006683,0.342282,-0.359438,-0.114072,-0.870685,-1.391324,-0.081901,0.393429,-1.096279]],[[19,33,38,6,-2,-2,42,-2,-2,-2,9,-2,2,-2,-2],[0.5,0.5,0.5,0.7400000095367432,-2.0,-2.0,0.2899159789085388,-2.0,-2.0,-2.0,0.6209999918937683,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,13,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,14,-1,-1],[-0.001033,0.003591,0.005722,-0.000952,-0.218081,0.13878,0.039332,0.79617,-0.500094,-0.770307,-0.04454,-0.211253,-0.053101,-1.099522,-1.043933]],[[8,9,69,16,-2,-2,5,-2,-2,18,6,-2,-2,-2,12,9,5,-2,-2,6,-2,-2,7,-2,42,-2,-2],[0.9034999907016754,0.8619999885559082,1.5,0.5,-2.0,-2.0,0.10949999839067459,-2.0,-2.0,0.5,0.9985000193119049,-2.0,-2.0,-2.0,0.7384999990463257,0.9905000030994415,0.48649999499320984,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,0.015500000212341547,-2.0,0.03125,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,26,-1,-1],[-0.003484,0.008562,-0.012869,0.002702,0.210668,-0.963796,-0.084423,-1.324376,-0.191191,0.024961,0.017562,0.285958,-1.016619,1.305746,-0.027849,-0.010412,-0.076573,0.109379,-1.172363,0.015052,-0.107637,0.717296,-0.071329,-1.343521,-0.037568,0.100446,-0.922758]],[[48,10,3,69,-2,-2,69,-2,-2,17,4,-2,-2,-2,6,-2,42,-2,-2],[0.5,0.2678571492433548,0.007500000298023224,1.5,-2.0,-2.0,1.5,-2.0,-2.0,0.5,0.5660000145435333,-2.0,-2.0,-2.0,0.7529999911785126,-2.0,0.18333333730697632,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,-1],[0.001111,-0.003436,-0.017936,-0.094604,-0.580003,-1.107753,-0.006652,-0.282106,0.379241,0.01654,0.02176,0.647074,-0.011679,-0.771004,0.055122,-0.408695,0.107312,0.378453,1.026372]],[[42,69,6,74,-2,-2,12,-2,-2,7,5,-2,-2,2,-2,-2,66,4,42,-2,-2,4,-2,-2,-2],[0.02777777798473835,0.5,0.6735000014305115,-0.49899999995250255,-2.0,-2.0,0.5460000038146973,-2.0,-2.0,0.10349999740719795,0.08449999988079071,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.06000000052154064,0.2613636404275894,-2.0,-2.0,0.0794999971985817,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[-0.003501,0.019778,-0.000631,-0.055081,-1.072164,0.588976,0.031398,-0.447448,1.071401,0.052611,-0.017895,0.398809,-0.790357,0.10928,1.344967,0.578439,-0.013409,-0.009079,0.013436,0.480119,-0.434367,-0.022792,-1.288638,-0.213668,-1.145079]],[[51,16,44,2,-2,-2,6,-2,-2,6,-2,-2,8,-2,-2],[0.5,0.5,0.5,0.9850000143051147,-2.0,-2.0,0.009499999694526196,-2.0,-2.0,0.875,-2.0,-2.0,0.35750000178813934,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.001865,-0.001692,0.002827,0.014826,0.514185,-0.319968,-0.015098,1.423831,-0.202375,-0.072681,-1.390798,-0.350072,0.063918,0.019009,0.927001]],[[9,9,53,12,-2,-2,-2,43,32,-2,-2,-2,2,12,48,-2,-2,69,-2,-2,10,3,-2,-2,-2],[0.945499986410141,0.05849999934434891,0.5,0.6464999914169312,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.9600000083446503,0.6534999907016754,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4166666716337204,0.8735000193119049,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[0.000796,-0.01821,0.038323,0.020973,1.049887,1.030697,1.171121,-0.03157,-0.022064,-0.42377,0.750605,-0.847848,0.013429,0.034267,0.011528,-0.008033,1.050343,0.097262,0.352443,1.026618,-0.012672,-0.025175,-0.745332,0.144433,0.676023]],[[2,40,12,-2,12,-2,-2,-2,37,38,74,-2,-2,-2,10,7,-2,-2,-2],[0.9750000238418579,0.5,0.21249999850988388,-2.0,0.8409999907016754,-2.0,-2.0,-2.0,1.5,0.5,-0.5,-2.0,-2.0,-2.0,0.2678571492433548,0.945499986410141,-2.0,-2.0,-2.0],[1,2,3,-1,5,-1,-1,-1,9,10,11,-1,-1,-1,15,16,-1,-1,-1],[8,7,4,-1,6,-1,-1,-1,14,13,12,-1,-1,-1,18,17,-1,-1,-1],[-0.002326,0.008404,0.003169,1.355481,8e-05,-0.07875,0.70262,1.579676,-0.020333,-0.036708,-0.045709,-0.773808,1.12226,0.432546,0.016002,-0.010715,-1.101986,0.533937,0.947097]],[[12,8,5,42,-2,-2,-2,-2,12,42,4,-2,-2,6,-2,-2,63,8,-2,-2,-2],[0.2580000013113022,0.6084999740123749,0.1510000005364418,0.25833334028720856,-2.0,-2.0,-2.0,-2.0,0.3489999920129776,0.24038461595773697,0.3464999943971634,-2.0,-2.0,0.9645000100135803,-2.0,-2.0,0.5,0.6304999887943268,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[-0.003496,0.043864,0.003244,0.063066,1.32485,0.435268,-0.944214,1.323284,-0.007007,-0.052083,-0.101117,-0.678623,-1.492341,0.01618,-0.483804,0.945756,-0.001036,-0.005128,0.274356,-0.30691,0.619791]],[[42,54,17,19,-2,-2,7,-2,-2,-2,15,2,75,-2,-2,74,-2,-2,-2],[0.33908046782016754,0.5,0.5,0.5,-2.0,-2.0,0.01699999999254942,-2.0,-2.0,-2.0,0.5,0.9249999821186066,0.563492089509964,-2.0,-2.0,-0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[0.001486,0.008841,0.005259,0.009353,0.160494,-0.817752,-0.061937,-1.229439,-0.293128,1.175536,-0.033124,-0.021387,0.037635,0.271804,1.111738,-0.03835,-1.020861,0.91135,-1.031993]],[[10,63,44,6,-2,-2,42,-2,-2,-2,9,4,2,-2,-2,74,-2,-2,8,5,-2,-2,53,-2,-2],[0.2678571492433548,0.5,0.5,0.4584999978542328,-2.0,-2.0,0.15192308276891708,-2.0,-2.0,-2.0,0.9835000038146973,0.061000000685453415,0.9850000143051147,-2.0,-2.0,-0.5,-2.0,-2.0,0.06149999983608723,0.018499999307096004,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.0034,-0.016524,-0.02191,-0.005836,-0.484826,0.238657,-0.042851,-0.69997,-0.145392,0.398759,0.013234,-0.004837,0.044354,0.341473,1.445059,-0.024755,-0.571599,0.321877,0.063424,0.010059,1.266481,-0.658714,0.124079,0.910573,1.377165]],[[2,69,17,5,-2,-2,-2,6,-2,-2,4,4,12,-2,-2,-2,74,6,-2,-2,7,-2,-2],[0.9600000083446503,2.5,0.5,0.2135000005364418,-2.0,-2.0,-2.0,0.9290000200271606,-2.0,-2.0,0.07649999856948853,0.051500000059604645,0.466499999165535,-2.0,-2.0,-2.0,-0.5,0.910999983549118,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.00144,0.015309,0.005916,0.011582,-0.045835,0.444083,-1.068492,0.095059,1.153946,-0.09259,-0.017059,0.003955,-0.012074,0.463932,-0.598178,1.015395,-0.035599,-0.042726,-0.169279,-0.926198,0.074401,0.465532,1.20841]],[[42,63,17,9,-2,-2,-2,-2,30,8,10,-2,-2,12,-2,-2,-2],[0.19615384936332703,0.5,0.5,0.9985000193119049,-2.0,-2.0,-2.0,-2.0,0.5,0.87950000166893,0.7083333432674408,-2.0,-2.0,0.8585000038146973,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.000739,0.012494,0.005137,0.009512,0.073335,0.969191,-0.971904,0.855198,-0.01251,-0.007307,0.004046,0.202486,-0.58607,-0.032795,-0.6427,0.692499,-0.984128]],[[39,9,69,9,-2,-2,3,-2,-2,43,9,-2,-2,-2,5,-2,-2],[0.5,0.945499986410141,1.5,0.9165000021457672,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.9594999849796295,-2.0,-2.0,-2.0,0.02350000012665987,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.000507,0.00309,-0.01408,0.001261,0.203043,-0.969813,-0.082276,-1.052028,-0.626814,0.013758,0.006843,1.003395,0.020333,0.952877,-0.118737,-0.176245,-1.301669]],[[5,3,39,69,-2,-2,1,-2,-2,7,6,-2,-2,-2,3,6,-2,6,-2,-2,6,37,-2,-2,5,-2,-2],[0.6545000076293945,0.6725000143051147,0.5,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.10949999839067459,0.8529999852180481,-2.0,-2.0,-2.0,0.7330000102519989,0.12350000441074371,-2.0,0.9874999821186066,-2.0,-2.0,0.7364999949932098,1.5,-2.0,-2.0,0.9839999973773956,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.000426,0.0084,0.017136,0.021612,0.245986,0.747705,-0.075476,-1.160477,1.11949,-0.036592,-0.057543,-0.227708,-0.931491,0.668379,-0.0198,-0.054045,0.645582,-0.072447,-0.698441,-1.131755,0.019131,-0.030133,-0.855434,0.277476,0.102212,1.429707,-0.004126]],[[5,9,19,8,-2,-2,12,-2,-2,5,12,-2,-2,-2,42,-2,6,12,-2,-2,-2],[0.9895000159740448,0.9975000023841858,0.5,0.9855000078678131,-2.0,-2.0,0.46299999952316284,-2.0,-2.0,0.8959999978542328,0.6679999828338623,-2.0,-2.0,-2.0,0.04545454680919647,-2.0,0.26900000870227814,0.7784999907016754,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,-1],[0.002516,0.007383,0.002298,0.007271,0.147229,-0.400983,-0.042477,-0.295138,-1.126515,0.07142,0.027091,0.797467,-0.515063,1.61679,-0.04267,0.215606,-0.065451,-0.017531,-1.164779,0.449685,-1.259352]],[[35,42,12,46,-2,-2,12,-2,-2,-2,-2],[0.5,0.7321428656578064,0.20499999821186066,1.5,-2.0,-2.0,0.3894999921321869,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002667,0.000276,0.001388,0.077247,1.50824,-0.201229,-0.000832,-0.401322,0.093528,-1.603461,0.767471]],[[19,6,8,44,-2,-2,6,-2,-2,6,7,-2,-2,6,-2,-2,42,2,-2,-2,-2],[0.5,0.24400000274181366,0.05700000002980232,0.5,-2.0,-2.0,0.12000000104308128,-2.0,-2.0,0.5685000121593475,0.004500000039115548,-2.0,-2.0,0.7090000212192535,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.002347,0.00635,-0.01763,0.036762,0.798574,1.189017,-0.036435,-0.211813,-0.972762,0.013507,0.06146,1.303219,0.171646,0.005557,-0.553571,0.145819,-0.039824,-0.052695,-1.061468,-1.063628,-0.481338]],[[16,8,35,66,-2,-2,-2,12,5,-2,-2,44,-2,-2,12,-2,-2],[0.5,0.9905000030994415,0.5,0.5,-2.0,-2.0,-2.0,0.6994999945163727,0.34299999475479126,-2.0,-2.0,0.5,-2.0,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[9e-05,0.003783,0.008591,0.006021,0.110272,-0.59959,0.92697,-0.033788,-0.081852,-1.027416,-1.041323,0.028358,-0.089414,0.894861,-0.056252,-0.317898,-1.25795]],[[5,5,8,9,-2,-2,42,-2,-2,42,-2,-2,8,42,-2,-2,8,-2,69,-2,-2],[0.9835000038146973,0.9535000026226044,0.9835000038146973,0.9665000140666962,-2.0,-2.0,0.21111111342906952,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.9884999990463257,0.17142857611179352,-2.0,-2.0,0.992499977350235,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,17,-1,19,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,18,-1,20,-1,-1],[0.003395,0.008366,0.004345,0.009371,-0.078742,0.326258,-0.047655,-0.673765,0.31379,0.068296,1.072224,-0.158325,-0.034395,-0.071431,-1.180145,-1.033899,-0.00801,0.918354,-0.04952,0.170777,-1.003425]],[[32,1,33,6,-2,-2,-2,6,-2,10,-2,-2,10,-2,8,-2,-2],[0.5,0.5,0.5,0.9964999854564667,-2.0,-2.0,-2.0,0.14300000295042992,-2.0,0.45000000298023224,-2.0,-2.0,0.2916666716337204,-2.0,0.0494999997317791,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,-1,15,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,14,-1,16,-1,-1],[-0.004415,-0.008041,-0.014248,-0.011693,-0.072288,-0.734155,-1.135515,0.016507,-0.663688,0.022081,0.933171,0.304213,0.031521,-0.098217,0.060631,0.733343,1.232298]],[[39,5,3,3,-2,-2,3,-2,-2,4,2,-2,-2,-2,5,6,-2,-2,-2],[0.5,0.6545000076293945,0.68299999833107,0.5590000152587891,-2.0,-2.0,0.9665000140666962,-2.0,-2.0,0.882999986410141,0.9750000238418579,-2.0,-2.0,-2.0,0.044999999925494194,0.9235000014305115,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,-1],[-0.004733,-0.001953,0.006517,0.015145,0.150599,1.437947,-0.036646,-0.757807,0.171571,-0.021342,-0.009895,0.286709,-0.511867,-1.273159,-0.083802,0.002174,1.890647,-0.668811,-1.292839]],[[16,42,7,5,-2,-2,2,-2,-2,2,-2,-2,6,-2,-2],[0.5,0.612500011920929,0.06650000065565109,0.5489999949932098,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.8865000009536743,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.006395,0.009519,0.011294,-0.000469,0.08888,-0.897113,0.028367,0.511061,0.008801,-0.088253,-1.657499,-1.141833,-0.045502,-1.1212,0.342652]],[[51,19,42,12,-2,-2,10,-2,-2,42,2,-2,-2,-2,-2],[0.5,0.5,0.5857143104076385,0.2580000013113022,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[0.005434,0.00262,0.006645,0.00851,0.628098,0.05955,-0.077521,-1.174683,-1.487156,-0.037989,-0.052372,-1.132861,-1.037869,-0.446645,0.488156]],[[35,56,4,4,-2,-2,3,-2,-2,5,-2,-2,-2],[0.5,0.5,0.9715000092983246,0.35899999737739563,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.2264999970793724,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.001375,-0.00089,0.002024,-0.003515,0.088377,-0.374151,0.031007,0.271774,1.529473,-0.053772,-0.156533,-1.181245,0.853164]],[[11,16,7,38,-2,-2,6,-2,-2,-2,7,8,3,-2,-2,9,-2,-2,3,69,-2,-2,-2],[0.4166666716337204,0.5,0.03800000064074993,0.5,-2.0,-2.0,0.9975000023841858,-2.0,-2.0,-2.0,0.005499999970197678,0.022499999962747097,0.0025000000605359674,-2.0,-2.0,0.9839999973773956,-2.0,-2.0,0.0625,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.001056,-0.006886,-0.002247,-0.023629,-0.36189,0.396327,0.015994,0.259743,-0.736729,-1.351182,0.023711,0.044545,0.013108,0.067915,1.138229,0.08825,0.668382,1.356964,-0.001876,0.010897,-0.194208,0.686529,-0.576547]],[[17,42,42,63,-2,-2,53,-2,-2,15,2,-2,-2,-2,12,-2,-2],[0.5,0.3452381044626236,0.2952069789171219,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.9249999821186066,-2.0,-2.0,-2.0,0.4960000216960907,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.00095,0.00207,0.008242,0.003763,-0.006684,0.531436,0.058312,-0.441399,1.191915,-0.02781,-0.013736,0.815679,-0.628788,-1.107952,-0.062798,-0.171316,-1.134333]],[[6,39,70,15,-2,-2,6,-2,-2,5,-2,-2,4,-2,10,4,-2,-2,8,-2,-2],[0.9905000030994415,0.5,0.5,0.5,-2.0,-2.0,0.8205000162124634,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.05250000022351742,-2.0,0.2678571492433548,0.9724999964237213,-2.0,-2.0,0.21299999952316284,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,15,16,-1,-1,19,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,18,17,-1,-1,20,-1,-1],[-0.000849,-0.007669,-0.005258,-0.010986,-0.227659,0.327044,0.032542,0.951505,-0.400975,-0.089039,0.289972,-0.92596,0.026432,1.141576,0.00818,-0.028954,-0.728998,0.670714,0.037049,0.413512,1.198892]],[[5,69,6,39,-2,-2,7,-2,-2,6,-2,4,-2,-2,38,3,-2,6,-2,-2,-2],[0.9835000038146973,2.5,0.9794999957084656,0.5,-2.0,-2.0,0.26999999582767487,-2.0,-2.0,0.4294999986886978,-2.0,0.05049999989569187,-2.0,-2.0,0.5,0.23099999874830246,-2.0,0.26900000870227814,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,17,-1,19,-1,-1,-1],[-0.001204,0.003553,-0.001061,-0.011901,-0.111874,-1.18372,0.022918,0.142294,0.922225,0.037238,1.080238,0.005934,0.744648,-0.494354,-0.038146,-0.053816,-0.204537,-0.069527,-1.038036,-1.1242,0.281284]],[[4,69,3,8,-2,-2,3,-2,-2,-2,5,4,8,-2,-2,3,-2,-2,5,6,-2,-2,5,-2,-2],[0.03749999962747097,2.5,0.4204999953508377,0.7504999935626984,-2.0,-2.0,0.992499977350235,-2.0,-2.0,-2.0,0.02850000001490116,0.4385000020265579,0.05199999921023846,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.20650000125169754,0.3854999989271164,-2.0,-2.0,0.6044999957084656,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.001052,0.019754,0.009123,0.034797,0.332134,0.99553,-0.013009,-0.44699,0.606856,0.741835,-0.011218,0.016981,-0.035501,0.644531,-0.899546,0.040121,0.304979,1.168166,-0.022014,-0.050239,-1.497194,-0.40172,-0.005767,0.387009,-0.325753]],[[5,7,4,2,-2,-2,3,-2,-2,16,3,-2,-2,-2,6,12,-2,-2,69,-2,-2],[0.9895000159740448,0.06650000065565109,0.9554999768733978,0.9350000023841858,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.5,0.2135000005364418,-2.0,-2.0,-2.0,0.28700000047683716,0.6050000190734863,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[-0.00309,0.001445,-0.009218,-0.020786,0.127227,-0.384668,0.02839,0.227927,1.365625,0.019784,0.026171,0.025047,0.500196,-1.291709,-0.052167,-0.019371,-1.368927,0.289954,-0.065511,-0.85316,-1.126897]],[[54,11,4,12,-2,-2,4,-2,-2,7,8,-2,-2,12,-2,-2,-2],[0.5,0.36666667461395264,0.351500004529953,0.20100000500679016,-2.0,-2.0,0.5189999938011169,-2.0,-2.0,0.005499999970197678,0.051500000059604645,-2.0,-2.0,0.7084999978542328,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.001101,-0.001461,-0.009029,0.005189,1.878627,0.020413,-0.0343,-1.183205,-0.284745,0.020822,0.041289,0.518162,1.049733,-0.004066,-0.456993,0.702801,0.942656]],[[12,2,-2,-2,6,70,12,-2,-2,42,-2,-2,5,7,-2,-2,-2],[0.17549999803304672,0.9850000143051147,-2.0,-2.0,0.9575000107288361,0.5,0.8680000007152557,-2.0,-2.0,0.2613636404275894,-2.0,-2.0,0.9794999957084656,0.3544999957084656,-2.0,-2.0,-2.0],[1,2,-1,-1,5,6,7,-1,-1,10,-1,-1,13,14,-1,-1,-1],[4,3,-1,-1,12,9,8,-1,-1,11,-1,-1,16,15,-1,-1,-1],[-7.9e-05,0.0949,1.560654,0.030997,-0.001966,-0.01307,-0.022452,-0.38232,0.328481,0.04348,0.862872,-0.741951,0.015961,0.022422,0.126664,0.884882,-0.975079]],[[1,39,64,9,-2,-2,-2,-2,46,8,37,-2,-2,6,-2,-2,-2],[0.5,0.5,0.5,0.9524999856948853,-2.0,-2.0,-2.0,-2.0,1.5,0.26000000536441803,1.5,-2.0,-2.0,0.734499990940094,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.001603,-0.00805,-0.004941,-0.008511,-0.356578,0.023632,0.656447,-0.910965,0.027498,0.021979,0.016291,1.04575,-0.054225,0.033658,0.190991,1.049877,1.246672]],[[51,7,7,8,-2,-2,38,-2,-2,42,4,-2,-2,74,-2,-2,-2],[0.5,0.05049999989569187,0.006500000134110451,0.9239999949932098,-2.0,-2.0,0.5,-2.0,-2.0,0.33974359929561615,0.8064999878406525,-2.0,-2.0,-0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.003712,0.000494,-0.011281,0.003523,0.234307,-0.376274,-0.039936,-0.668856,0.406197,0.015927,0.026056,0.439261,-0.534207,-0.029854,-1.023049,0.595849,0.580727]],[[74,46,9,9,-2,-2,10,-2,-2,3,-2,-2,3,8,8,-2,-2,4,-2,-2,-2],[-0.5,2.5,0.945499986410141,0.057499999180436134,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.006500000134110451,-2.0,-2.0,0.4450000077486038,0.221500001847744,0.17249999940395355,-2.0,-2.0,0.9409999847412109,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.003013,-0.010576,-0.007634,-0.029508,0.841465,-0.467434,0.003702,-0.060819,0.694054,-0.11621,-1.422643,0.015234,0.016735,0.021577,0.012083,0.835968,-0.639546,0.04804,1.088419,0.637421,-0.298647]],[[9,9,7,2,-2,-2,-2,8,-2,-2,42,42,5,-2,-2,6,-2,-2,5,42,-2,-2,11,-2,-2],[0.38199999928474426,0.20250000059604645,0.04900000058114529,0.9600000083446503,-2.0,-2.0,-2.0,0.22899999469518661,-2.0,-2.0,0.33908046782016754,0.2952069789171219,0.99549999833107,-2.0,-2.0,0.987500011920929,-2.0,-2.0,0.006500000134110451,0.44155845046043396,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.00248,-0.022093,0.0042,0.018402,0.178192,0.959131,-1.004869,-0.091201,-1.075836,-0.635843,0.00712,0.014334,0.007891,0.133487,-0.975975,0.074649,1.218911,-0.735869,-0.024894,0.043274,0.642809,1.172152,-0.03423,-0.433052,-1.366817]],[[16,2,53,75,-2,-2,7,-2,-2,5,2,-2,-2,2,-2,-2,4,-2,-2],[0.5,0.9350000023841858,0.5,0.3452381044626236,-2.0,-2.0,0.009499999694526196,-2.0,-2.0,0.2045000046491623,0.9950000047683716,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.7975000143051147,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.003429,0.005931,0.02335,0.011838,-0.833802,0.546289,0.070329,0.400121,1.137497,-0.002123,-0.020167,-0.33388,0.672392,0.010129,0.478645,-0.111094,-0.040073,-1.116726,0.038421]],[[64,6,12,10,-2,-2,69,-2,-2,6,-2,12,-2,-2,-2],[0.5,0.9575000107288361,0.715499997138977,0.875,-2.0,-2.0,1.5,-2.0,-2.0,0.9675000011920929,-2.0,0.5160000026226044,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[-0.000999,-0.003556,-0.014398,-0.02748,-0.251168,-1.159959,0.009002,-0.019495,0.741135,0.013535,0.899097,0.006803,-0.226005,0.514697,0.750708]],[[51,69,72,19,-2,-2,-2,6,-2,-2,-2],[0.5,3.5,0.5,0.5,-2.0,-2.0,-2.0,0.9290000200271606,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,-1],[0.002197,-0.000554,0.002807,0.000768,0.065026,-0.951196,0.792208,-0.050139,0.061723,-0.906577,0.521381]],[[17,42,19,74,-2,-2,9,-2,-2,-2,9,-2,-2],[0.5,0.7321428656578064,0.5,0.6265000104904175,-2.0,-2.0,0.8125,-2.0,-2.0,-2.0,0.9050000011920929,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[0.002014,0.00436,0.005214,0.008503,0.083209,1.160907,-0.026494,-0.048217,-1.094251,-1.572503,-0.049106,-1.1301,-0.228452]],[[46,69,12,8,-2,-2,6,-2,-2,5,-2,7,-2,-2,69,1,-2,-2,-2],[2.5,2.5,0.2575000077486038,0.40449999272823334,-2.0,-2.0,0.4584999978542328,-2.0,-2.0,0.08449999988079071,-2.0,0.1599999964237213,-2.0,-2.0,1.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,-1],[0.000796,0.002858,-0.001627,0.056718,0.087144,1.199688,-0.005218,-0.352875,0.050076,0.036926,0.91215,0.002818,-1.074807,0.489901,-0.060852,0.018749,-0.679072,1.15675,-1.389575]],[[71,5,5,3,-2,-2,48,-2,-2,7,69,-2,-2,2,-2,-2,-2],[0.5,0.6845000088214874,0.23349999636411667,0.34700000286102295,-2.0,-2.0,0.5,-2.0,-2.0,0.8244999945163727,0.5,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.002013,-0.00032,0.007796,-0.002572,0.134627,-0.453894,0.038738,0.397023,0.947757,-0.019656,-0.070708,-0.209478,-1.316699,-0.004177,0.302944,-1.074753,-0.969666]],[[42,9,12,-2,6,-2,-2,8,-2,-2,9,42,-2,6,-2,-2,-2],[0.02777777798473835,0.9964999854564667,0.3824999928474426,-2.0,0.4740000069141388,-2.0,-2.0,0.9514999985694885,-2.0,-2.0,0.9985000193119049,0.10555555671453476,-2.0,0.6299999952316284,-2.0,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,-1,-1,11,12,-1,14,-1,-1,-1],[10,7,4,-1,6,-1,-1,9,-1,-1,16,13,-1,15,-1,-1,-1],[0.003134,0.020255,0.029798,-0.446972,0.039632,-0.119936,0.840049,-0.036088,0.439922,-0.812839,-0.004066,-0.006929,-0.813485,-0.004232,0.194948,-0.180946,0.677806]],[[5,2,3,3,-2,-2,8,-2,-2,4,-2,3,-2,-2,4,6,37,-2,-2,2,-2,-2,-2],[0.6775000095367432,0.9950000047683716,0.7254999876022339,0.5615000128746033,-2.0,-2.0,0.9724999964237213,-2.0,-2.0,0.020999999716877937,-2.0,0.0025000000605359674,-2.0,-2.0,0.8935000002384186,0.4270000010728836,1.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.001967,0.011247,0.005233,0.012068,0.108139,1.101081,-0.035822,-0.987143,0.096407,0.062661,1.577213,0.015661,1.160228,-0.048272,-0.020231,-0.012008,-0.040934,-1.007303,0.19032,0.008077,0.473048,-0.313173,-1.183068]],[[17,19,64,5,-2,-2,-2,6,-2,55,-2,-2,6,-2,-2],[0.5,0.5,0.5,0.3085000067949295,-2.0,-2.0,-2.0,0.7094999849796295,-2.0,0.5,-2.0,-2.0,0.948500007390976,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,14,-1,-1],[-0.003187,-0.000554,0.003358,0.00088,0.148693,-0.168449,0.779991,-0.035896,-0.445113,-0.048845,-1.061457,-1.067601,-0.052931,-1.205807,-0.001386]],[[9,74,4,-2,5,-2,-2,37,10,-2,-2,-2,63,8,8,-2,-2,9,-2,-2,-2],[0.9494999945163727,-0.5,0.00849999999627471,-2.0,0.09950000047683716,-2.0,-2.0,2.5,0.22500000149011612,-2.0,-2.0,-2.0,0.5,0.40949998795986176,0.2685000002384186,-2.0,-2.0,0.9605000019073486,-2.0,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,7,4,-1,6,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[0.000129,-0.014203,-0.033534,0.513423,-0.040011,-0.667964,-0.366149,0.018156,0.027716,1.109745,0.757857,-0.604478,0.009579,0.004814,0.040097,0.417843,1.455522,-0.008473,1.048346,-0.158667,0.82696]],[[71,42,4,6,-2,-2,5,-2,-2,2,-2,-2,-2],[0.5,0.5857143104076385,0.057499999180436134,0.9945000112056732,-2.0,-2.0,0.6545000076293945,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.001795,0.003396,0.005054,0.02042,0.14964,1.359235,-0.003606,0.087626,-0.381177,-0.067931,-1.51941,-1.135718,-0.788002]],[[35,14,32,3,-2,-2,5,-2,-2,-2,-2],[0.5,0.5,0.5,0.00849999999627471,-2.0,-2.0,0.01699999999254942,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002962,0.001235,0.0034,0.00018,-0.262058,0.083896,0.034095,-0.148972,0.646993,-0.403325,0.765559]],[[42,69,12,-2,6,-2,-2,-2,42,-2,7,3,-2,-2,48,-2,-2],[0.02777777798473835,1.5,0.40549999475479126,-2.0,0.5575000047683716,-2.0,-2.0,-2.0,0.08712121471762657,-2.0,0.019499999471008778,0.9665000140666962,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,5,-1,-1,-1,9,-1,11,12,-1,-1,15,-1,-1],[8,7,4,-1,6,-1,-1,-1,10,-1,14,13,-1,-1,16,-1,-1],[-0.000189,0.015409,0.00511,-0.902114,0.015397,-0.366487,0.544619,0.57439,-0.006677,-0.82556,-0.003685,0.010176,0.0591,0.907175,-0.018013,-0.170939,-0.638659]],[[8,51,16,63,-2,-2,4,-2,-2,2,-2,-2,42,-2,-2],[0.9964999854564667,0.5,0.5,0.5,-2.0,-2.0,0.863999992609024,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.24264705926179886,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.000724,0.002879,0.000301,0.004093,0.015825,0.488194,-0.05788,-1.302102,-0.212871,0.053653,0.072272,0.774314,-0.063099,-1.270095,-0.819004]],[[17,42,6,7,-2,-2,74,-2,-2,42,-2,5,-2,-2,4,-2,-2],[0.5,0.029411764815449715,0.9905000030994415,0.048499999567866325,-2.0,-2.0,-0.5,-2.0,-2.0,0.08712121471762657,-2.0,0.02949999924749136,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,-1],[0.005646,0.008751,0.02888,0.017022,-0.174591,0.522972,0.074654,1.213252,1.020714,0.000242,-0.758912,0.00246,0.414386,-0.077394,-0.057179,-1.304662,0.167106]],[[4,4,8,3,-2,-2,8,-2,-2,-2,4,12,-2,-2,4,-2,9,-2,-2],[0.0345000009983778,0.02850000001490116,0.39500001072883606,0.02049999963492155,-2.0,-2.0,0.9724999964237213,-2.0,-2.0,-2.0,0.04649999924004078,0.5705000162124634,-2.0,-2.0,0.05849999934434891,-2.0,0.7739999890327454,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,-1,15,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,14,13,-1,-1,16,-1,18,-1,-1],[0.002323,0.016993,0.009516,0.039759,0.615556,1.214082,-0.001696,-0.639869,0.185112,0.966551,-0.00423,-0.057032,-1.05979,-0.162037,-0.000531,0.683079,-0.003814,-0.451896,0.092508]],[[6,6,3,53,-2,-2,7,-2,-2,12,-2,-2,51,3,3,-2,-2,5,-2,-2,-2],[0.24400000274181366,0.11899999901652336,0.68299999833107,0.5,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.5640000104904175,-2.0,-2.0,0.5,0.5570000112056732,0.4035000056028366,-2.0,-2.0,0.8194999992847443,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.000388,-0.023136,-0.003489,0.01946,1.164329,-0.070536,-0.033804,-1.277473,-0.272295,-0.094628,-1.482465,-0.637549,0.005774,0.002148,-0.007041,-0.023393,-0.812999,0.03004,-0.011779,0.742726,0.676432]],[[5,48,63,8,-2,-2,-2,5,-2,-2,-2],[0.9964999854564667,0.5,0.5,0.46400000154972076,-2.0,-2.0,-2.0,0.2045000046491623,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,-1],[0.003197,0.005089,0.002114,-0.000937,0.235753,-0.147945,0.528033,0.043509,-0.138243,0.680636,-0.933853]],[[19,5,3,3,-2,-2,7,-2,-2,12,69,-2,-2,-2,6,42,-2,-2,12,-2,-2],[0.5,0.9895000159740448,0.9895000159740448,0.9525000154972076,-2.0,-2.0,0.3034999966621399,-2.0,-2.0,0.8680000007152557,0.5,-2.0,-2.0,-2.0,0.9340000152587891,0.20714285969734192,-2.0,-2.0,0.46299999952316284,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[0.001928,0.005342,0.008899,0.006596,0.143033,-0.576698,0.064471,0.919834,-0.211402,-0.024824,-0.039328,0.033264,-0.866015,0.799123,-0.028908,-0.013931,-1.042956,0.050294,-0.042454,-0.746824,-1.114832]],[[65,56,4,6,-2,-2,4,-2,-2,7,-2,-2,12,-2,-2],[0.5,0.5,0.05650000087916851,0.9945000112056732,-2.0,-2.0,0.06750000268220901,-2.0,-2.0,0.009500000160187483,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[-0.000617,-0.004268,-0.001004,0.014413,0.10242,1.104471,-0.009736,-1.362834,-0.110733,-0.064739,-0.40553,-1.2112,0.04905,1.011451,0.35608]],[[16,43,44,2,-2,-2,2,-2,-2,6,-2,-2,6,-2,-2],[0.5,0.5,0.5,0.9950000047683716,-2.0,-2.0,0.9350000023841858,-2.0,-2.0,0.7529999911785126,-2.0,-2.0,0.9535000026226044,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.001059,0.004251,0.000522,0.009776,0.362808,-0.700307,-0.013924,0.657172,-0.259543,0.046913,-0.318917,0.731106,-0.050732,-1.16468,0.026949]],[[3,44,2,3,-2,-2,46,-2,-2,5,37,-2,-2,-2,5,-2,-2],[0.9934999942779541,0.5,0.9950000047683716,0.8760000169277191,-2.0,-2.0,1.5,-2.0,-2.0,0.6414999961853027,2.5,-2.0,-2.0,-2.0,0.940500020980835,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.001891,0.00089,0.012098,0.021135,0.203457,0.852894,-0.022618,-1.053171,0.995283,-0.013285,-0.004962,-0.192266,0.58026,-1.068895,-0.069843,-1.164079,-0.909041]],[[44,2,46,42,-2,-2,-2,46,8,-2,-2,7,-2,-2,6,-2,4,69,-2,-2,3,-2,-2],[0.5,0.9950000047683716,1.5,0.44949495792388916,-2.0,-2.0,-2.0,1.5,0.99549999833107,-2.0,-2.0,0.7515000104904175,-2.0,-2.0,0.009499999694526196,-2.0,0.9619999825954437,3.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[-0.000446,0.009747,0.017944,0.022078,0.49628,-0.803986,-0.741975,-0.018934,-0.032333,-1.061217,0.607309,0.035925,1.109665,0.537975,-0.014187,1.23157,-0.019477,-0.02671,-0.224423,-0.801407,0.02231,-0.299472,1.292186]],[[16,54,17,2,-2,-2,42,-2,-2,-2,9,-2,-2],[0.5,0.5,0.5,0.9750000238418579,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[-0.000842,0.001819,-5.1e-05,0.002609,0.161066,-0.19882,-0.045375,-1.136195,0.5073,0.709443,-0.041489,-0.937335,-0.242679]],[[37,73,42,12,-2,-2,7,-2,-2,-2,-2],[4.5,0.5,0.06458333507180214,0.36649999022483826,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.003048,0.005246,0.003159,0.019625,-0.512866,0.372867,-0.004307,0.210267,-0.218852,0.847047,-0.601275]],[[51,5,6,44,-2,-2,8,-2,-2,2,8,-2,-2,8,-2,-2,-2],[0.5,0.9535000026226044,0.014500000048428774,0.5,-2.0,-2.0,0.17599999904632568,-2.0,-2.0,0.9850000143051147,0.9905000030994415,-2.0,-2.0,0.9934999942779541,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.002984,0.000909,-0.003784,0.059555,0.351706,1.056527,-0.00755,0.19723,-0.225879,0.021569,0.084277,1.037119,0.357622,-0.015757,-0.720626,0.695908,0.434138]],[[42,12,-2,12,-2,63,-2,-2,30,12,75,-2,-2,8,-2,-2,-2],[0.18465909361839294,0.2580000013113022,-2.0,0.3330000042915344,-2.0,0.5,-2.0,-2.0,0.5,0.17549999803304672,0.20000000298023224,-2.0,-2.0,0.39249999821186066,-2.0,-2.0,-2.0],[1,2,-1,4,-1,6,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,3,-1,5,-1,7,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[0.00608,0.015721,0.996106,0.012167,-0.693224,0.0177,0.150635,0.87646,-0.005257,-0.000819,0.073585,-1.782686,1.534516,-0.003549,0.148856,-0.231051,-0.79852]]]
UMBRAL = 0.20
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
