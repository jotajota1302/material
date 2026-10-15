"""El Guardián — fernando_v2. Solo librería estándar (no lee ficheros ni ejecuta modelos).

Base: el bot joseca_v4 de nuestro equipo, sin cambios en sus 74 características,
en su modelo (gradient boosting de 200 árboles, profundidad 4, embebido como
datos) ni en su corte (UMBRAL = 0,18, reescalado para que el organizador corte
en 0,5). Único cambio validado: el VERIFICADOR DE CUENTAS AMPLIADO (variante A1,
tomado de joseca_v5) dentro de verificar_cuentas_frase / comprobar_plazos /
extraer_numeros:
  - diferencias con verbo («superó en 35 millones al del viñedo»: se rehace la resta);
  - proporciones con palabra («triplica», «el doble», «la mitad», «más de un tercio»);
  - «350 usos más al día» no es una comparación con otra cifra;
  - fecha + plazo = fecha («acta en septiembre de 2026 y 30 meses de obra»);
  - «1.900 personas» no es un año (los años no llevan punto de miles).
Con esas reglas cambian los valores de verif_cuenta_mal / verif_cuenta_ok /
fecha_plazo_* y el modelo se reentrena con el train del kit (lab/construir_final.py).

Validación (lab/, CV agrupada por documento sobre el 80 % de los casos H, 5 x 3):
v4 +0,512 -> fernando_v2 +0,578 (gana en 11 de 15 particiones); en un 20 % de
familias H reservado y usado una sola vez: v4 +0,507 -> +0,582.

Cómo puntúa:
1. caracteristicas(caso): señales del juez LLM y del NLI tomando la PEOR oración
   o fragmento; comprobación de cifras y cuentas (literal, derivada, cuenta
   rehecha en la frase), plazos, atribución a otra entidad, matices perdidos,
   alcance y causas añadidas, palabras nuevas.
2. probabilidad(): suma de los árboles embebidos (bloque MODELO) y sigmoide,
   en float32 para coincidir con scikit-learn.
3. reescalar(p, UMBRAL): lleva el corte elegido (0,18, coste 4:1) a 0,5.
Si algo falla, devuelve 1.0 (un escape cuesta 4 veces más que una falsa alarma).
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
        # v6: «1.900 personas» no es un año: los años no llevan punto de miles
        anio = v.is_integer() and 1900 <= v <= 2100 and not pct and "." not in t[ini:fin]
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
    # v5: «acta prevista para septiembre de 2026 y un plazo de 30 meses, la obra
    # terminaría en marzo de 2029»: la respuesta trae dos fechas y un plazo;
    # el plazo tiene que ser exactamente la distancia entre esas dos fechas.
    f_resp = [(a, mes) for a, mes, _ in _fechas(r) if a]
    if len(f_resp) >= 2:
        meses_resp = {a * 12 + mes for a, mes in f_resp}
        distancias = {abs(x - y) for x in meses_resp for y in meses_resp} - {0}
        for m in RE_PLAZO_DURACION.finditer(r):
            palabra, unidad = m.group(1), m.group(2)
            n = UNIDADES.get(palabra) or int(palabra)
            n_meses = n * 12 if unidad == "anos" else n
            if n_meses in distancias:
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
RE_PLAZO_DURACION = re.compile(r"\b(?:plazo de|en un plazo de|durante|tras)\s+(\d+|" + "|".join(UNIDADES) +
                               r")\s+(meses|anos)\b")
RE_UNIDAD_TIEMPO = re.compile(r"\s*(dia|mes|ano|semana|hora|trimestre|curso)\b")
RE_VERBO_DIFERENCIA = re.compile(r"\b(supero|supera|superan|superaron|superaba|redujo|reduce|reducen|recorto|"
                                 r"rebajo|aumento|crecio|subio|bajo|amplio|incremento|disminuyo|excede|"
                                 r"excedio|adelanta|aventaja|aventajo)\s+en\s+(unos?\s+|unas\s+)?$")
PROPORCION_VALOR = {"duplica": 2.0, "doble": 2.0, "triplica": 3.0, "triple": 3.0, "cuadruplica": 4.0,
                    "cuadruple": 4.0, "quintuplica": 5.0, "mitad": 2.0, "tercio": 3.0, "cuarta parte": 4.0}
RE_PROPORCION = re.compile(r"\b(duplica|doble|triplica|triple|cuadruplica|cuadruple|quintuplica|mitad|tercio|cuarta parte)\b")
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
            if RE_UNIDAD_TIEMPO.match(objetivo):
                # v5: «350 usos más AL DÍA» no compara con nadie: es «X más» suelto.
                # Si cuadra con alguna cifra de las fuentes, bien; si no, no opinamos.
                anclas = [k for k in utiles if k != c]
                if anclas and any(_casi(abs(av - o), base(c), antes(c)) for k in anclas
                                  for av in nums_resp[k]["valores"] for o in todas_fuente):
                    resultado[c] = ("ok", "comparacion")
                continue
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

        # --- F. Diferencia con verbo (v5): «superó en 35 millones al del viñedo»,
        # «redujo en 70 el número de afectados respecto a la primera propuesta» ---
        for c in utiles:
            if c in resultado or not RE_VERBO_DIFERENCIA.search(antes(c, 25)):
                continue
            anclas = [k for k in utiles if k != c]
            m_verbo = RE_VERBO_DIFERENCIA.search(antes(c, 25))
            sujeto = resp_norm[ini_f:nums_resp[c]["ini"] - len(m_verbo.group())]
            objetivo = re.split(r"[;]", resp_norm[nums_resp[c]["fin"]:fin_f])[0][:80]
            idx = set(_fuentes_que_hablan_de(sujeto, fuentes_norm)) | set(_fuentes_que_hablan_de(objetivo, fuentes_norm))
            if not idx:
                continue
            vals = sorted(set(_valores_no_anio([n for i in idx for n in nums_por_fuente[i]])))[:40]
            dicho = base(c)
            # v6: probamos tres formas de hacer la resta antes de decir «mal»:
            #   1) una cifra de la frase contra las fuentes de las que habla;
            #   2) dos cifras de esas fuentes entre sí («superó en 17 puntos a la
            #      de los ciclos ordinarios»: 71 - 54, sin cifras en la frase);
            #   3) una cifra de la frase contra cualquier fuente («superó en
            #      1.500 personas al programa X, que atendió a 1.900»: la otra
            #      cifra, 3.400, es la del programa de la pregunta).
            pares = [(av, o) for k in anclas for av in nums_resp[k]["valores"] for o in vals]
            pares += list(combinations(vals, 2))
            pares += [(av, o) for k in anclas for av in nums_resp[k]["valores"] for o in todas_fuente]
            if not pares:
                continue
            if nums_resp[c]["pct"] and not any(nums_resp[k]["pct"] for k in anclas):
                ok = any(_casi(abs(a - b) / min(a, b) * 100, dicho, antes(c)) or _casi(abs(a - b), dicho, antes(c))
                         for a, b in pares if min(a, b))
            else:
                ok = any(_casi(abs(a - b), dicho, antes(c)) for a, b in pares)
            anotar(c, ok, "diferencia")

        # --- G. Proporciones con palabra (v5): «triplica», «el doble», «la mitad» ---
        mp = RE_PROPORCION.search(frase)
        if mp and ("prop", ini_f) not in resultado:
            k_dicho = PROPORCION_VALOR[mp.group(1)]
            # Las dos cifras de la comparación: las de la frase; si solo hay una,
            # la otra en la fuente de la entidad con la que se compara
            vals_frase = [base(k) for k in utiles]
            pares = list(combinations(vals_frase, 2))
            if len(vals_frase) == 1 and nums_resp[utiles[0]]["pct"]:
                # v6: «algo más de un tercio de las compras, el 37 %»
                pares = [(100.0, vals_frase[0])]
            elif len(vals_frase) == 1:
                objetivo = resp_norm[ini_f + mp.end():fin_f][:60]
                idx = _fuentes_que_hablan_de(objetivo, fuentes_norm)
                pares = [(vals_frase[0], o) for o in _valores_no_anio([n for i in idx for n in nums_por_fuente[i]])]
            previo = frase[:mp.start()]
            minimo = bool(re.search(r"mas (del|de la|de)\s*(un|una|la|el)?\s*$", previo))
            maximo = bool(re.search(r"menos (del|de la|de)\s*(un|una|la|el)?\s*$", previo))
            # v6: con fracciones («más de un tercio») la proporción grande/pequeña
            # tiene que ser MENOR que 3; con múltiplos («más del doble»), MAYOR que 2
            fraccion = mp.group(1) in ("mitad", "tercio", "cuarta parte")
            if fraccion:
                minimo, maximo = maximo, minimo
            if pares:
                ok = False
                for a, b in pares:
                    if min(a, b) <= 0:
                        continue
                    r = max(a, b) / min(a, b)
                    if minimo and k_dicho * 0.98 <= r <= k_dicho * 1.6:
                        ok = True
                    elif maximo and k_dicho / 1.6 <= r <= k_dicho * 1.02:
                        ok = True
                    elif abs(r - k_dicho) <= 0.1 * k_dicho:
                        ok = True
                resultado[("prop", ini_f)] = ("ok" if ok else "mal", "proporcion")

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

# >>> MODELO (generado por fernando_v2/lab/construir_final.py; no editar a mano)
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas"]
INICIAL = -0.035456
TASA = 0.05
ARBOLES = [[[2,3,44,45,-2,-2,-2,4,3,-2,-2,5,-2,-2,38,15,8,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,0.7999999821186066,-2.0,-2.0,-2.0,0.28450000286102295,0.8585000038146973,-2.0,-2.0,0.06400000117719173,-2.0,-2.0,0.5,0.5,0.07600000128149986,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.2361111119389534,0.9639999866485596,-2.0,-2.0,0.3920000046491623,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.003123,0.414307,0.470507,0.499429,2.029206,1.672342,1.099628,0.284888,0.401079,1.958398,1.035778,0.15553,1.464484,0.294805,-0.180073,-0.253411,-0.30764,0.227305,-1.441883,0.211833,0.071839,1.775141,0.161875,0.361804,1.8685,0.455349,-0.21841,0.285542,-1.459743]],[[2,8,53,3,-2,-2,-2,7,8,-2,-2,12,-2,-2,8,4,12,-2,-2,53,-2,-2,21,38,-2,-2,-2],[0.9350000023841858,0.17599999904632568,0.5,0.03350000083446503,-2.0,-2.0,-2.0,0.001500000071246177,0.8425000011920929,-2.0,-2.0,0.801499992609024,-2.0,-2.0,0.07600000128149986,0.08900000154972076,0.4000000059604645,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[0.002195,0.384225,0.463661,0.480351,1.938536,1.6406,1.32736,0.301958,0.460039,1.953784,1.606365,0.247804,0.774852,1.970837,-0.175857,0.192494,0.415648,2.011573,1.448233,0.049194,1.832888,-0.227384,-0.237772,-0.267926,-1.317132,0.237141,1.460158]],[[2,53,3,45,-2,-2,42,-2,-2,5,-2,9,-2,-2,8,4,7,-2,-2,1,-2,-2,38,15,-2,-2,42,-2,-2],[0.9249999821186066,0.5,0.032500000670552254,0.7800000011920929,-2.0,-2.0,0.24038461595773697,-2.0,-2.0,0.023000000044703484,-2.0,0.9480000138282776,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.019999999552965164,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.17995689809322357,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.004833,0.363377,0.417046,0.451309,1.848674,1.544747,0.349238,1.60673,0.548731,0.166435,1.746028,0.048465,-0.489781,0.829253,-0.170491,0.16072,0.324427,0.724036,1.680461,-0.084842,-0.706477,1.93344,-0.234424,-0.293815,-1.315894,0.356114,0.070543,1.431823,-1.046897]],[[2,53,3,8,-2,-2,42,-2,-2,5,-2,9,-2,-2,8,4,4,-2,-2,5,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,0.17149999737739563,-2.0,-2.0,0.24038461595773697,-2.0,-2.0,0.02350000012665987,-2.0,0.9784999787807465,-2.0,-2.0,0.09149999916553497,0.8700000047683716,0.08900000154972076,-2.0,-2.0,0.07499999925494194,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.014411,0.340953,0.379876,0.427106,1.777785,1.553792,0.289824,1.379231,0.269365,0.17125,1.69057,0.074854,-0.338006,0.940018,-0.173604,0.131526,0.322732,1.700296,0.663759,-0.12604,0.036178,-1.50086,-0.234661,-0.286208,-1.313892,0.621378,0.040093,1.029553,-1.17824]],[[2,53,3,0,-2,-2,45,-2,-2,9,-2,-2,21,38,0,-2,-2,42,-2,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,0.5,-2.0,-2.0,0.5565000176429749,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.2567765712738037,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.011113,0.322549,0.369591,0.390775,1.686956,1.383502,0.22752,0.170249,1.717899,0.127894,-0.259473,1.113384,-0.160473,-0.187911,-0.246246,-1.070613,1.671077,0.102503,1.087486,-1.119299,1.422799]],[[2,7,3,53,-2,-2,-2,44,3,-2,-2,-2,8,4,6,-2,-2,53,-2,-2,54,38,-2,-2,-2],[0.9350000023841858,0.007500000298023224,0.636000007390976,0.5,-2.0,-2.0,-2.0,0.5,0.107999999076128,-2.0,-2.0,-2.0,0.0885000005364418,0.8700000047683716,0.3439999967813492,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[-0.005445,0.320448,0.379449,0.39077,1.672722,1.324922,0.831656,0.224007,0.280651,1.547362,0.624028,-0.085495,-0.156763,0.147336,0.293375,1.612583,0.670423,-0.068148,1.56318,-0.647203,-0.212139,-0.24167,-1.183192,0.018217,1.388757]],[[2,53,3,42,-2,-2,45,-2,-2,7,-2,12,-2,-2,8,44,4,-2,-2,4,-2,-2,38,21,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,0.4833333343267441,-2.0,-2.0,0.718999981880188,-2.0,-2.0,0.0025000000605359674,-2.0,0.47699999809265137,-2.0,-2.0,0.1314999982714653,0.5,0.9160000085830688,-2.0,-2.0,0.060499999672174454,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.007582,0.301428,0.343259,0.366835,1.582555,0.875034,0.22192,0.322023,1.628688,0.122063,1.29053,0.01981,1.09351,-0.411873,-0.131331,0.135148,0.403934,1.795475,1.117288,0.023234,1.426355,-0.325853,-0.187299,-0.237853,-1.097816,1.193229,0.061522,0.859039,-1.468353]],[[53,5,39,3,-2,-2,-2,42,-2,-2,2,19,38,-2,-2,5,-2,-2,18,65,-2,-2,-2],[0.5,0.5550000071525574,0.5,0.11449999734759331,-2.0,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.1525000035762787,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[-0.013744,0.283379,0.330209,0.341764,1.540875,1.120075,0.735169,0.003012,0.692687,-1.411251,-0.149947,0.004482,0.083781,0.052642,1.495251,-0.402356,-1.883346,-1.519965,-0.263271,-0.293521,-1.349778,0.298711,0.755344]],[[53,69,5,2,-2,-2,1,-2,-2,-2,2,19,70,-2,-2,7,-2,-2,51,32,-2,-2,-2],[0.5,0.5,0.4529999941587448,0.4749999940395355,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.003215,0.262386,0.310388,0.331816,1.408415,1.712176,0.185666,0.258059,1.275313,-0.72861,-0.122757,0.016323,0.078978,0.040902,1.313536,-0.342518,-1.079046,-1.720761,-0.225915,-0.260719,-1.235009,0.424766,0.916593]],[[2,7,3,12,-2,-2,-2,44,4,-2,-2,-2,15,32,5,-2,-2,42,-2,-2,12,-2,-2],[0.9350000023841858,0.007500000298023224,0.6884999871253967,0.48349998891353607,-2.0,-2.0,-2.0,0.5,0.27400000393390656,-2.0,-2.0,-2.0,0.5,0.5,0.024500000290572643,-2.0,-2.0,0.16025641560554504,-2.0,-2.0,0.5259999930858612,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,22,-1,-1],[0.002694,0.268573,0.313948,0.324423,1.302524,1.523884,0.766706,0.193372,0.248621,1.547662,0.715697,-0.348118,-0.117815,-0.148853,-0.185807,0.182865,-0.965874,0.205459,1.62689,-0.025867,0.204342,1.407227,-0.113422]],[[53,69,5,46,-2,-2,12,-2,-2,-2,21,65,2,-2,-2,10,-2,-2,-2],[0.5,0.5,0.5289999842643738,1.5,-2.0,-2.0,0.6539999842643738,-2.0,-2.0,-2.0,0.5,0.5,0.9600000083446503,-2.0,-2.0,0.2222222276031971,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.001394,0.238317,0.273983,0.294794,1.295137,1.649079,0.127122,1.051756,0.098285,-0.6951,-0.113169,-0.14079,-0.179627,-0.267537,-1.135128,0.207144,0.261869,1.519189,1.400805]],[[2,5,69,2,-2,-2,-2,-2,38,21,8,-2,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,0.75,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.05250000022351742,-2.0,-2.0,-2.0,0.2361111119389534,0.9460000097751617,-2.0,-2.0,0.390500009059906,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[-0.000895,0.2457,0.266862,0.252468,1.278265,-0.215038,1.621045,0.26921,-0.114085,-0.163025,-0.186471,0.361782,-0.962376,1.300281,0.102295,0.275849,1.565088,0.161666,-0.259687,-0.170915,-1.647287]],[[2,8,7,8,-2,-2,-2,9,3,-2,-2,1,-2,-2,32,15,38,-2,-2,9,-2,-2,42,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,0.048499999567866325,-2.0,-2.0,-2.0,0.9634999930858612,0.33650000393390656,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.9804999828338623,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[0.002234,0.224098,0.27845,0.285102,1.418686,1.212457,0.967533,0.165541,0.069991,0.735126,-0.490588,0.228742,0.73739,1.308962,-0.100573,-0.131856,-0.16749,-0.890496,0.243674,0.224301,0.332721,1.746995,0.222083,1.565828,0.282585]],[[2,8,2,3,-2,-2,-2,44,4,-2,-2,-2,8,4,37,-2,-2,46,-2,-2,2,19,-2,-2,42,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,0.010499999858438969,-2.0,-2.0,-2.0,0.5,0.19250000268220901,-2.0,-2.0,-2.0,0.09149999916553497,0.21649999916553497,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.9750000238418579,0.5,-2.0,-2.0,0.11437908560037613,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.003553,0.205477,0.271043,0.26205,1.377222,1.08345,1.491054,0.134868,0.179476,1.255547,0.501975,-0.037657,-0.101493,0.11377,0.254972,0.595607,1.526307,-0.002392,-0.397492,1.210248,-0.142654,-0.050627,-0.004762,-1.545664,-0.215815,-0.616157,-1.206735]],[[2,5,45,3,-2,-2,-2,42,6,-2,-2,4,-2,-2,21,38,1,-2,-2,42,-2,-2,-2],[0.9350000023841858,0.023000000044703484,0.7999999821186066,0.07200000062584877,-2.0,-2.0,-2.0,0.24264705926179886,0.8695000112056732,-2.0,-2.0,0.016500000841915607,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.000809,0.208006,0.25994,0.270273,1.339464,1.518853,0.806136,0.166561,0.205697,0.617748,1.218126,0.037854,1.514764,-0.572603,-0.098776,-0.117658,-0.161901,-0.772964,1.883621,0.078676,0.941899,-0.835969,1.199012]],[[2,3,5,8,-2,-2,1,-2,-2,4,3,-2,-2,5,-2,-2,38,15,0,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.9350000023841858,0.04749999940395355,0.05350000038743019,0.17149999737739563,-2.0,-2.0,0.5,-2.0,-2.0,0.44350001215934753,0.7790000140666962,-2.0,-2.0,0.0885000005364418,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.9774999916553497,-2.0,-2.0,0.2899159789085388,0.9864999949932098,-2.0,-2.0,0.443000003695488,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.005409,0.195569,0.235254,0.213114,1.26711,-0.092298,0.285875,1.516743,1.36008,0.125419,0.200411,1.185252,0.476788,-0.04095,0.928721,-0.787314,-0.097196,-0.141729,-0.174733,-0.864824,1.397481,0.204907,0.072887,1.560192,0.116675,0.259639,1.454314,0.090365,-0.290224,-0.719913,-1.623276]],[[53,5,46,39,-2,-2,8,-2,-2,6,-2,-2,64,15,38,-2,-2,7,-2,-2,-2],[0.5,0.6654999852180481,1.5,0.5,-2.0,-2.0,0.0494999997317791,-2.0,-2.0,0.7284999787807465,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.04650000110268593,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.004234,0.18555,0.221895,0.206101,1.161498,0.022016,0.362489,1.450374,2.099315,-0.025339,0.883508,-0.910003,-0.092016,-0.116507,-0.149133,-0.843336,0.17824,0.162117,1.428044,-0.194794,1.11271]],[[2,19,16,17,-2,-2,-2,8,-2,-2,0,1,42,-2,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,0.5,0.5,0.1180555559694767,-2.0,-2.0,-2.0,0.11600000038743019,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,18,-1,-1],[0.003774,0.112417,0.1522,0.180718,0.98317,-1.255121,-1.075527,-0.310036,-0.893437,-1.823583,-0.140198,-0.159264,-0.171467,-0.320111,-1.079769,1.749986,0.324927,1.446002,1.603407]],[[2,5,42,5,-2,-2,-2,1,4,-2,-2,7,-2,-2,31,21,1,-2,-2,-2,-2],[0.9350000023841858,0.14750000089406967,0.44949495792388916,0.08400000259280205,-2.0,-2.0,-2.0,0.5,0.47050000727176666,-2.0,-2.0,0.25200000405311584,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[0.001719,0.169885,0.20114,0.210905,1.108183,1.404439,0.351247,0.095435,-0.020854,0.458986,-0.715473,0.204719,1.352935,0.481064,-0.077904,-0.096969,-0.116332,-0.595239,1.686665,1.263473,1.172967]],[[53,5,46,42,-2,-2,8,-2,-2,-2,52,2,42,-2,-2,10,-2,-2,8,-2,8,-2,-2],[0.5,0.9629999995231628,1.5,0.24264705926179886,-2.0,-2.0,0.0860000029206276,-2.0,-2.0,-2.0,0.5,0.9750000238418579,0.19090909510850906,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.398499995470047,-2.0,0.7010000050067902,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,22,-1,-1],[0.001845,0.170454,0.190646,0.175154,1.122902,0.586472,0.325673,1.385852,1.833712,-0.582451,-0.074266,-0.038987,0.054984,0.729628,-0.297959,-0.132028,-0.816159,0.083093,-0.285308,-1.560925,-0.265887,-0.996149,-1.478443]],[[2,19,16,17,-2,-2,-2,7,-2,-2,46,70,48,-2,-2,-2,53,8,-2,-2,3,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,1.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.05650000087916851,-2.0,-2.0,0.06650000065565109,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.003307,0.098235,0.131949,0.155245,0.882333,-1.038341,-1.031831,-0.27705,-0.938312,-1.691878,-0.123941,-0.155636,-0.181061,-1.022165,0.226843,0.206924,0.100756,0.302674,1.345442,1.69837,-0.127604,-1.505246,0.202937]],[[2,7,3,3,-2,-2,42,-2,-2,8,3,-2,-2,-2,15,31,0,-2,-2,-2,9,-2,-2],[0.9350000023841858,0.007500000298023224,0.012500000186264515,0.006500000134110451,-2.0,-2.0,0.30384616553783417,-2.0,-2.0,0.9359999895095825,0.032500000670552254,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,22,-1,-1],[0.002851,0.158875,0.193621,0.175434,1.186611,0.296683,0.229826,1.316504,0.849679,0.105048,0.059132,0.960219,-0.275097,1.355101,-0.070341,-0.093679,-0.112661,-0.610833,1.27125,0.917785,0.181757,0.049435,1.310092]],[[53,5,39,2,-2,-2,45,-2,-2,42,-2,-2,31,2,52,-2,-2,8,-2,-2,-2],[0.5,0.6629999876022339,0.5,0.9350000023841858,-2.0,-2.0,0.0020000000367872417,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.5,0.9750000238418579,0.5,-2.0,-2.0,0.06349999830126762,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.001626,0.136116,0.167095,0.179026,1.008185,1.375352,0.005281,-0.812699,1.346777,-0.041017,0.480106,-0.721907,-0.067605,-0.086568,-0.013824,0.154179,-1.332223,-0.161289,0.203965,-0.94642,1.246387]],[[2,7,8,3,-2,-2,-2,12,9,-2,-2,42,-2,-2,38,8,4,-2,-2,42,-2,-2,42,6,-2,-2,-2],[0.9350000023841858,0.001500000071246177,0.7580000162124634,0.02049999963492155,-2.0,-2.0,-2.0,0.7975000143051147,0.9440000057220459,-2.0,-2.0,0.18333333730697632,-2.0,-2.0,0.5,0.1314999982714653,0.8720000088214874,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.2899159789085388,0.9864999949932098,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[0.00408,0.140384,0.190829,0.197129,1.234489,1.319738,0.772969,0.114407,0.08594,0.064294,0.765306,0.204985,1.088193,1.366187,-0.061484,-0.094445,0.061226,0.979878,-0.60986,-0.126392,-0.216623,-0.938258,0.106614,0.207627,1.300898,-0.192285,-0.975207]],[[2,8,2,1,-2,-2,-2,8,-2,8,-2,-2,21,31,1,-2,-2,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,0.5,-2.0,-2.0,-2.0,0.2565000057220459,-2.0,0.9440000057220459,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,18,17,16,-1,-1,-1,-1],[-0.006987,0.133341,0.171704,0.162353,0.79734,1.214183,1.274321,0.09469,-0.230889,0.128238,0.49206,1.097142,-0.070853,-0.085528,-0.101851,-0.55226,1.598244,0.921406,1.100195]],[[53,69,46,3,-2,-2,8,-2,-2,-2,15,64,30,-2,-2,-2,7,-2,-2],[0.5,0.5,1.5,0.9379999935626984,-2.0,-2.0,0.08650000020861626,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.031000000424683094,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,18,-1,-1],[0.002342,0.139108,0.158702,0.14424,0.986599,-0.090647,0.280902,1.34579,1.607313,-0.46348,-0.058101,-0.083038,-0.103686,-0.633228,0.714179,0.975402,0.163874,1.231535,0.056218]],[[19,2,39,5,-2,-2,-2,10,42,-2,-2,4,-2,-2,12,42,-2,-2,-2],[0.5,0.9350000023841858,0.5,0.2134999930858612,-2.0,-2.0,-2.0,0.2678571492433548,0.12916667014360428,-2.0,-2.0,0.5160000026226044,-2.0,-2.0,0.6444999873638153,0.2321428582072258,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,-1],[-0.003603,0.024607,0.145627,0.152979,0.898803,1.145227,-0.014857,-0.034738,-0.08659,0.033313,-0.790572,0.087135,0.940124,-0.181476,-0.242997,-0.215482,-0.982268,-1.38619,-1.61567]],[[2,19,17,16,-2,-2,-2,8,-2,-2,0,42,2,-2,-2,1,-2,-2,4,9,-2,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,0.5,0.11437908560037613,0.9850000143051147,-2.0,-2.0,0.5,-2.0,-2.0,0.11600000038743019,0.9165000021457672,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.005728,0.08199,0.114112,0.131543,0.806196,-0.753385,-1.086978,-0.250517,-0.499645,-1.674039,-0.090525,-0.105252,-0.014299,0.805719,-0.464739,-0.155859,-0.929859,1.521329,0.241643,0.22765,1.320115,1.279069,1.396386]],[[2,19,16,17,-2,-2,-2,8,-2,-2,8,5,8,-2,-2,-2,42,37,-2,-2,46,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.7374999821186066,-2.0,-2.0,0.13199999928474426,0.032500000670552254,0.019999999552965164,-2.0,-2.0,-2.0,0.11437908560037613,1.5,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.005797,0.072779,0.102435,0.123674,0.809715,-1.217665,-0.997643,-0.239039,-0.572605,-1.616949,-0.088407,0.089002,0.251365,0.877979,1.778867,-0.406545,-0.119715,-0.03168,-0.651804,1.314975,-0.173058,-1.032287,0.149665]],[[2,19,17,16,-2,-2,-2,5,-2,-2,8,5,3,-2,-2,-2,42,2,-2,-2,3,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.22549999505281448,-2.0,-2.0,0.13199999928474426,0.07250000163912773,0.0025000000605359674,-2.0,-2.0,-2.0,0.11437908560037613,0.9850000143051147,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[-0.006556,0.064032,0.0952,0.110657,0.755056,-0.758751,-0.99694,-0.241452,-0.981312,-1.647031,-0.097965,0.064401,0.156628,0.407002,1.393072,-0.515573,-0.13065,-0.071261,0.521072,-0.760907,-0.164067,-0.074898,-1.049641]],[[2,3,44,7,-2,-2,9,-2,-2,4,7,-2,-2,-2,15,32,1,-2,-2,10,-2,-2,7,-2,-2],[0.9350000023841858,0.3439999967813492,0.5,0.17750000208616257,-2.0,-2.0,0.9650000035762787,-2.0,-2.0,0.1889999955892563,0.017500000074505806,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.04650000110268593,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[0.0016,0.110919,0.127995,0.140666,1.036327,1.319155,0.059552,-0.235732,0.789316,0.033327,0.104242,0.186542,1.122711,-0.697038,-0.049998,-0.068219,-0.089735,-0.528986,1.509564,0.137881,-0.096295,1.552027,0.157503,1.1562,0.026492]],[[2,8,2,3,-2,-2,-2,8,-2,9,-2,-2,42,37,2,-2,-2,4,-2,-2,8,4,-2,-2,32,-2,-2],[0.9350000023841858,0.17599999904632568,0.8250000178813934,0.010499999858438969,-2.0,-2.0,-2.0,0.2565000057220459,-2.0,0.968500018119812,-2.0,-2.0,0.11324786394834518,1.5,0.9750000238418579,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.09699999913573265,0.8684999942779541,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.005041,0.099101,0.135239,0.126241,1.157615,0.721592,1.222638,0.060687,-0.739051,0.091796,0.20114,0.825785,-0.052639,0.042815,-0.03376,0.342679,-0.597342,0.224075,1.882753,0.280224,-0.10359,0.039468,0.871224,-0.533592,-0.136559,-0.8458,0.158291]],[[19,8,17,42,-2,-2,-2,2,42,-2,-2,42,-2,-2,55,69,-2,-2,3,-2,-2],[0.5,0.17599999904632568,0.5,0.5099999904632568,-2.0,-2.0,-2.0,0.9750000238418579,0.39444445073604584,-2.0,-2.0,0.06458333507180214,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.31300000846385956,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,19,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,20,-1,-1],[-0.003131,0.021644,0.11879,0.135161,0.920785,-0.634763,-1.003933,-0.025363,0.038888,0.347986,-0.953799,-0.102437,-0.175133,-0.886341,-0.211424,-0.248657,-1.58429,-1.262957,-0.175019,-1.250518,-0.915102]],[[19,2,16,17,-2,-2,-2,46,11,-2,-2,9,-2,-2,6,-2,42,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,1.5,0.4166666716337204,-2.0,-2.0,0.9805000126361847,-2.0,-2.0,0.6944999992847443,-2.0,0.3166666775941849,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,-1],[0.013847,0.036586,0.090559,0.105839,0.729502,-0.75324,-0.84123,-0.052255,-0.08072,-0.606934,0.535236,0.152487,0.533449,1.975194,-0.203178,-1.062369,-0.230056,-1.456043,-0.986695]],[[11,42,37,2,-2,-2,4,-2,-2,1,43,-2,-2,46,-2,-2,4,4,7,-2,-2,-2,5,4,-2,-2,1,-2,-2],[0.20714285969734192,0.13076923787593842,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.5,-2.0,-2.0,3.5,-2.0,-2.0,0.05849999934434891,0.03749999962747097,0.08999999985098839,-2.0,-2.0,-2.0,0.02850000001490116,0.8485000133514404,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.002517,-0.042713,0.028745,-0.038025,0.124045,-0.932763,0.157178,1.822051,-0.112487,-0.094505,-0.116788,-0.769415,0.199798,0.152862,0.786742,1.568763,0.098324,0.165456,0.147716,0.95983,1.19114,1.497284,0.071454,0.131679,1.355299,0.811818,0.0117,-0.209655,1.173662]],[[53,69,46,39,-2,-2,8,-2,-2,-2,21,42,70,-2,-2,2,-2,-2,-2],[0.5,0.5,1.5,0.5,-2.0,-2.0,0.0860000029206276,-2.0,-2.0,-2.0,0.5,0.1558704450726509,0.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[0.001929,0.095869,0.110644,0.09923,0.895924,-0.478046,0.207261,1.159643,1.445052,-0.227627,-0.041665,-0.056902,0.006749,-0.199288,1.037052,-0.109972,0.619668,-0.767329,1.017402]],[[19,2,16,50,-2,-2,-2,5,8,-2,-2,3,-2,-2,12,2,8,-2,-2,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,0.4545000046491623,0.593999981880188,-2.0,-2.0,0.06799999997019768,-2.0,-2.0,0.6444999873638153,0.9600000083446503,0.6665000021457672,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[0.000964,0.020295,0.069442,0.083239,0.584771,-0.71988,-0.845182,-0.059077,0.029596,0.590341,-0.583472,-0.123046,-1.243582,-0.578737,-0.176187,-0.153132,-0.120475,-0.312961,-1.312777,-1.294424,-1.483888]],[[19,2,17,16,-2,-2,7,-2,-2,15,46,-2,-2,-2,12,42,-2,-2,-2],[0.5,0.9750000238418579,0.5,0.5,-2.0,-2.0,0.007000000216066837,-2.0,-2.0,0.5,1.5,-2.0,-2.0,-2.0,0.6444999873638153,0.25357143580913544,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,-1],[0.003645,0.021977,0.061835,0.077208,0.577918,-0.855279,-0.168896,-1.425683,-0.521778,-0.043914,-0.06272,-0.557977,0.593317,0.959527,-0.174082,-0.15257,-0.875564,-1.327975,-1.464283]],[[8,6,53,3,-2,-2,7,-2,-2,2,3,-2,-2,37,-2,-2,42,15,38,-2,-2,9,-2,-2,7,3,-2,-2,8,-2,-2],[0.08900000154972076,0.80799999833107,0.5,0.012500000186264515,-2.0,-2.0,0.01699999999254942,-2.0,-2.0,0.9600000083446503,0.009499999694526196,-2.0,-2.0,1.5,-2.0,-2.0,0.33908046782016754,0.5,0.5,-2.0,-2.0,0.9390000104904175,-2.0,-2.0,0.0025000000605359674,0.18299999833106995,-2.0,-2.0,0.34449999034404755,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[0.0041,0.098915,0.145632,0.113725,1.168147,0.212509,0.194083,1.30957,0.726413,0.068193,0.104802,1.180153,0.05337,-0.056968,-0.761628,0.413453,-0.02779,-0.003501,-0.019918,-0.305763,0.587273,0.214111,0.317558,1.470629,-0.149467,-0.050279,0.205997,-1.073664,-0.169756,-1.435748,-0.989822]],[[2,22,16,19,-2,-2,-2,8,-2,-2,8,6,12,-2,-2,-2,4,6,-2,-2,53,-2,-2],[0.9600000083446503,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.7384999990463257,-2.0,-2.0,0.04649999924004078,0.9165000021457672,0.2760000079870224,-2.0,-2.0,-2.0,0.14400000125169754,0.9914999902248383,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.00603,0.055851,0.075611,0.087277,0.630282,-0.99464,-0.779869,-0.206542,-0.793505,-1.48657,-0.06102,0.108801,0.203101,0.893077,1.429955,-0.090133,-0.079555,-0.026374,-0.341093,1.279262,-0.147123,0.165058,-1.040023]],[[8,9,44,46,-2,-2,6,-2,-2,44,13,-2,-2,-2,42,30,15,-2,-2,-2,43,1,-2,-2,-2],[0.09149999916553497,0.945499986410141,0.5,1.5,-2.0,-2.0,0.9679999947547913,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.22649572789669037,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.003095,0.08977,0.056254,0.106816,0.966126,1.172703,-0.010386,0.540764,-0.97547,0.145033,0.106532,1.065063,0.18705,1.263516,-0.02748,0.012323,-0.011807,-0.186899,0.97385,1.181067,-0.105545,-0.126405,-0.959857,0.372528,0.344554]],[[53,39,3,8,-2,-2,-2,1,-2,-2,38,54,4,-2,-2,-2,42,-2,2,-2,-2],[0.5,0.5,0.9385000169277191,0.9434999823570251,-2.0,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,0.05849999934434891,-2.0,-2.0,-2.0,0.11324786394834518,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,-1,19,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,18,-1,20,-1,-1],[-0.002548,0.075138,0.087051,0.095342,0.755499,1.290552,-0.259956,-0.080811,-1.261495,1.235557,-0.039766,-0.06152,-0.074841,-0.082797,-0.721032,0.910313,0.093463,1.439012,-0.008504,0.605999,-1.166665]],[[2,37,53,45,-2,-2,-2,3,3,-2,-2,-2,21,32,30,-2,-2,42,-2,-2,-2],[0.9350000023841858,0.5,0.5,0.5305000245571136,-2.0,-2.0,-2.0,0.625,0.26750001311302185,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.0072,0.074219,0.045043,0.069555,0.877597,-0.439246,-0.368345,0.104545,0.117374,0.930941,1.322016,0.137487,-0.045646,-0.05559,-0.070891,-0.539614,0.469907,0.083869,1.155248,-0.219117,0.878497]],[[2,9,8,42,-2,-2,12,-2,-2,37,12,-2,-2,-2,31,21,30,-2,-2,-2,-2],[0.9350000023841858,0.9675000011920929,0.153499998152256,0.3693181872367859,-2.0,-2.0,0.801499992609024,-2.0,-2.0,2.5,0.5430000126361847,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[-0.005063,0.071824,0.045962,0.094769,1.071139,1.256897,-0.023437,-0.60333,0.889933,0.104095,0.091589,0.61917,1.076882,1.275007,-0.039584,-0.051379,-0.062097,-0.484443,0.538922,0.804274,0.816761]],[[19,17,11,16,-2,-2,7,-2,-2,8,-2,-2,7,42,-2,-2,2,-2,-2],[0.5,0.5,0.36666667461395264,0.5,-2.0,-2.0,0.005499999970197678,-2.0,-2.0,0.16449999809265137,-2.0,-2.0,0.010999999940395355,0.27142858505249023,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,17,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,18,-1,-1],[0.004619,0.019804,0.030881,0.00404,0.106213,-1.433171,0.108455,1.377538,0.320619,-0.189883,-0.53875,-1.484053,-0.136798,-0.075886,-0.18489,-1.01966,-0.176212,-1.397967,-1.176493]],[[19,16,2,39,-2,-2,38,-2,-2,9,-2,-2,55,6,-2,-2,6,-2,-2],[0.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5600000023841858,-2.0,-2.0,0.5,0.7139999866485596,-2.0,-2.0,0.9244999885559082,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,17,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,18,-1,-1],[-0.008073,0.00793,0.020759,0.08736,0.919063,-0.598651,-0.01334,-0.20656,0.498271,-0.175755,-1.591958,-0.742377,-0.154274,-0.198245,-1.297406,-1.434086,-0.119759,-0.715341,-1.176116]],[[8,5,4,3,-2,-2,7,-2,-2,-2,42,63,15,-2,-2,-2,45,9,-2,-2,-2],[0.09149999916553497,0.37699998915195465,0.8684999942779541,0.005499999970197678,-2.0,-2.0,0.02650000061839819,-2.0,-2.0,-2.0,0.3060200661420822,0.5,0.5,-2.0,-2.0,-2.0,0.0024999999441206455,0.9895000159740448,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.006058,0.07613,0.085122,0.116858,0.75207,1.214204,0.052682,0.698297,-0.556779,-0.35582,-0.018448,0.007092,-0.008549,-0.175218,1.125526,1.057761,-0.109513,-0.128109,-0.687059,-1.301859,1.123194]],[[53,69,12,2,-2,-2,7,-2,-2,-2,54,70,4,-2,-2,42,-2,-2,-2],[0.5,0.5,0.6030000150203705,0.44999998807907104,-2.0,-2.0,0.09549999982118607,-2.0,-2.0,-2.0,0.5,0.5,0.07649999856948853,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.005828,0.063817,0.082441,0.11811,1.033621,1.231528,0.045228,0.831113,-0.421865,-0.802577,-0.038343,-0.049568,-0.070276,-0.150672,-0.700203,0.053862,0.825549,-0.885072,0.904457]],[[19,10,63,2,-2,-2,-2,17,4,-2,-2,-2,8,8,-2,-2,-2],[0.5,0.2678571492433548,0.5,0.9350000023841858,-2.0,-2.0,-2.0,0.5,0.16099999845027924,-2.0,-2.0,-2.0,0.9090000092983246,0.4324999898672104,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,16,15,-1,-1,-1],[0.005843,0.020103,-0.016298,-0.030377,0.565844,-0.354203,0.695619,0.064992,0.082843,1.12357,0.388415,-1.315337,-0.123822,-0.098949,-1.144216,-0.53379,-1.336805]],[[19,2,9,3,-2,-2,69,-2,-2,42,2,-2,-2,11,-2,-2,9,-2,6,-2,2,-2,-2],[0.5,0.9350000023841858,0.9675000011920929,0.25300000607967377,-2.0,-2.0,0.5,-2.0,-2.0,0.02777777798473835,0.9850000143051147,-2.0,-2.0,0.36666667461395264,-2.0,-2.0,0.5775000154972076,-2.0,0.6944999992847443,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,22,-1,-1],[-0.004917,0.008914,0.067426,0.046079,0.704246,-0.392499,0.092127,0.858391,1.026524,-0.020086,0.067518,1.038608,-0.456662,-0.053416,-0.48277,0.289239,-0.121738,-0.367101,-0.139809,-0.815633,-0.166775,-1.357016,-1.168328]],[[19,5,16,17,-2,-2,4,-2,-2,12,12,-2,-2,2,-2,-2,4,6,-2,7,-2,-2,-2],[0.5,0.9794999957084656,0.5,0.5,-2.0,-2.0,0.9004999995231628,-2.0,-2.0,0.5264999866485596,0.4259999990463257,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.9664999842643738,0.6944999992847443,-2.0,0.0035000001080334187,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,-1],[0.002139,0.016679,0.033601,0.046206,0.398947,-0.717738,-0.131202,-1.537245,0.084651,-0.099333,-0.161579,-0.947136,-1.504812,-0.046572,0.362585,-0.905865,-0.117969,-0.131921,-0.853669,-0.16323,-1.21289,-1.311277,-0.570755]],[[2,12,9,3,-2,-2,37,-2,-2,8,8,-2,-2,-2,31,21,9,-2,-2,-2,-2],[0.9350000023841858,0.804999977350235,0.9675000011920929,0.04700000025331974,-2.0,-2.0,2.5,-2.0,-2.0,0.23749999701976776,0.014000000432133675,-2.0,-2.0,-2.0,0.5,0.5,0.945499986410141,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[-0.003767,0.055951,0.045563,0.011877,0.711341,-0.641883,0.072054,0.581475,1.266657,0.100015,0.076017,1.067498,1.105455,1.182547,-0.033214,-0.046794,-0.057439,-0.756244,-0.20075,0.833582,1.034886]],[[19,2,42,17,-2,-2,3,-2,-2,46,37,-2,-2,5,-2,-2,9,-2,12,-2,2,-2,-2],[0.5,0.9750000238418579,0.4686274528503418,0.5,-2.0,-2.0,0.021000000648200512,-2.0,-2.0,1.5,1.5,-2.0,-2.0,0.6850000023841858,-2.0,-2.0,0.5775000154972076,-2.0,0.4529999941587448,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,22,-1,-1],[-0.00271,0.010248,0.039973,0.049505,0.441619,-0.801969,-0.138815,0.32595,-1.45757,-0.039126,-0.058654,-0.686158,0.074557,0.084633,1.180823,-0.76239,-0.105796,-0.405232,-0.120696,-0.680254,-0.144231,-1.33188,-1.141039]],[[19,2,44,42,-2,-2,42,-2,-2,15,5,-2,-2,-2,12,6,2,-2,-2,-2,-2],[0.5,0.9750000238418579,0.5,0.5657142996788025,-2.0,-2.0,0.2752525359392166,-2.0,-2.0,0.5,0.010499999858438969,-2.0,-2.0,-2.0,0.6444999873638153,0.9945000112056732,0.9600000083446503,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[0.003223,0.015883,0.04585,0.08586,0.892594,-1.492897,-0.013388,0.146204,-0.655175,-0.031455,-0.044431,1.095679,-0.461932,0.774672,-0.101165,-0.084219,-0.065734,-0.300297,-1.191637,-1.20454,-1.303428]],[[8,4,2,42,-2,-2,7,-2,-2,7,37,-2,-2,-2,42,65,15,-2,-2,-2,9,53,-2,-2,-2],[0.0885000005364418,0.9185000061988831,0.9350000023841858,0.3693181872367859,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.04800000041723251,0.5,-2.0,-2.0,-2.0,0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,0.9975000023841858,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.006952,0.069734,0.112676,0.073109,1.003207,1.146968,0.158113,1.376051,0.869014,0.022822,0.042223,0.066146,1.043623,-0.894531,-0.014775,0.005421,-0.006703,-0.144812,0.784335,1.072765,-0.107867,-0.093047,0.333057,-0.935326,-1.639992]],[[1,48,4,6,-2,-2,12,-2,-2,9,-2,-2,46,6,4,-2,-2,8,-2,-2,4,-2,-2],[0.5,0.5,0.08950000256299973,0.9945000112056732,-2.0,-2.0,0.7339999973773956,-2.0,-2.0,0.9814999997615814,-2.0,-2.0,1.5,0.8664999902248383,0.44350001215934753,-2.0,-2.0,0.1614999994635582,-2.0,-2.0,0.9634999930858612,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,22,-1,-1],[0.000725,-0.016252,-0.029269,0.017626,0.039359,1.722783,-0.076314,-0.661834,0.071105,0.119187,0.08948,1.314793,0.071188,0.061889,0.029305,1.009583,-0.404118,0.076121,1.079344,1.080078,0.216498,1.431519,1.295321]],[[42,69,2,5,-2,-2,5,-2,-2,4,-2,6,-2,-2,8,4,2,-2,-2,44,-2,-2,1,15,-2,-2,37,-2,-2],[0.13076923787593842,1.5,0.9850000143051147,0.2135000005364418,-2.0,-2.0,0.6389999985694885,-2.0,-2.0,0.07649999856948853,-2.0,0.9309999942779541,-2.0,-2.0,0.0625,0.9235000014305115,0.9850000143051147,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.002199,0.040243,0.004009,0.036274,0.030811,0.690785,-0.092074,0.011895,-1.083215,0.156581,1.574105,0.043036,1.015645,-0.688555,-0.032833,0.043983,0.084754,0.497995,1.367934,-0.026107,0.655146,-0.742986,-0.057919,-0.072557,-0.661873,0.285042,0.074021,-0.209567,1.231146]],[[19,38,2,3,-2,-2,10,-2,-2,42,9,-2,-2,70,-2,-2,9,-2,2,12,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.04749999940395355,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,0.2899159789085388,0.9914999902248383,-2.0,-2.0,0.5,-2.0,-2.0,0.5775000154972076,-2.0,0.9600000083446503,0.5474999845027924,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,-1],[0.000722,0.012353,-0.003224,0.049697,0.925559,0.138209,-0.029842,-0.376075,0.167905,0.084954,0.124253,0.681546,1.219796,-0.087808,0.849012,-1.013566,-0.104816,-0.245258,-0.123291,-0.149332,-1.162614,-1.339067,-1.12155]],[[2,22,16,46,-2,-2,-2,-2,51,1,0,-2,-2,-2,-2],[0.9600000083446503,0.5,0.5,1.5,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,-1],[0.001564,0.03801,0.050215,0.061239,0.552792,-0.90663,-0.792394,-0.885198,-0.045573,-0.056075,-0.062328,-0.556247,1.078193,1.369954,0.739464]],[[19,2,44,42,-2,-2,37,-2,-2,5,8,-2,-2,8,-2,-2,8,2,-2,-2,37,-2,-2],[0.5,0.9750000238418579,0.5,0.44949495792388916,-2.0,-2.0,2.5,-2.0,-2.0,0.5489999949932098,0.5370000004768372,-2.0,-2.0,0.9535000026226044,-2.0,-2.0,0.6675000190734863,0.9600000083446503,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.000252,0.01025,0.034491,0.069849,0.831881,-0.888338,-0.016295,-0.251609,0.598967,-0.031232,0.031926,0.697344,-0.418536,-0.075785,-1.144514,-0.202909,-0.095929,-0.058315,0.010337,-1.203496,-0.124833,-1.143426,-1.290403]],[[2,22,42,50,-2,-2,-2,-2,4,9,-2,46,-2,-2,1,5,-2,-2,-2],[0.9600000083446503,0.5,0.5470085740089417,0.5,-2.0,-2.0,-2.0,-2.0,0.07649999856948853,0.7924999892711639,-2.0,1.5,-2.0,-2.0,0.5,0.006500000134110451,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,-1,12,-1,-1,15,16,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,11,-1,13,-1,-1,18,17,-1,-1,-1],[0.003519,0.031066,0.040964,0.046396,0.451419,-0.852179,-1.47683,-0.857873,-0.033711,0.009819,1.925576,-0.016139,-0.287514,1.455836,-0.069838,-0.080459,0.876383,-0.746363,1.35803]],[[16,2,17,7,-2,-2,-2,18,12,-2,-2,-2,9,-2,-2],[0.5,0.9600000083446503,0.5,0.15399999916553497,-2.0,-2.0,-2.0,0.5,0.2575000077486038,-2.0,-2.0,-2.0,0.5600000023841858,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,14,-1,-1],[-0.000416,0.006752,0.034545,0.044109,0.217843,0.828553,-0.918627,-0.031185,-0.042963,0.586598,-0.460873,0.809169,-0.121035,-1.293726,-0.481145]],[[54,31,2,39,-2,-2,42,-2,-2,-2,-2],[0.5,0.5,0.9350000023841858,0.5,-2.0,-2.0,0.14642857760190964,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.001367,-0.007603,-0.015584,0.033052,0.456085,-1.053672,-0.040379,0.026703,-0.558135,0.983042,1.051265]],[[54,42,2,7,-2,-2,46,-2,-2,5,8,-2,-2,5,-2,-2,-2],[0.5,0.33908046782016754,0.9750000238418579,0.9354999959468842,-2.0,-2.0,1.5,-2.0,-2.0,0.02350000012665987,0.2214999943971634,-2.0,-2.0,0.05250000022351742,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.003155,-0.003975,0.009896,0.035102,0.177269,1.332983,-0.039003,-0.418394,0.520671,-0.074269,0.035718,0.812424,-0.89553,-0.109836,-1.268426,-0.963145,1.043718]],[[19,5,16,17,-2,-2,9,-2,-2,8,42,-2,-2,2,-2,-2,12,-2,2,-2,-2],[0.5,0.9715000092983246,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.9805000126361847,0.2792207896709442,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.4529999941587448,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[-0.001032,0.008857,0.022037,0.030955,0.31662,-0.695042,-0.098578,-1.320668,0.149661,-0.061589,-0.146059,-1.493374,-0.80436,-0.02368,0.559604,-0.951405,-0.085571,-0.462627,-0.105148,-1.270302,-1.096646]],[[42,69,5,5,-2,-2,6,-2,-2,9,2,-2,-2,-2,2,45,3,-2,-2,4,-2,-2,15,30,-2,-2,7,-2,-2],[0.02777777798473835,0.5,0.6604999899864197,0.1915000006556511,-2.0,-2.0,0.44749999046325684,-2.0,-2.0,0.99549999833107,0.9850000143051147,-2.0,-2.0,-2.0,0.9249999821186066,0.5445000231266022,0.7620000243186951,-2.0,-2.0,0.1770000010728836,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.019999999552965164,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.004215,0.040689,0.002351,0.043967,-0.043007,1.010302,-0.102375,-1.171686,-0.434676,0.112223,0.157639,1.431186,0.186026,-0.172587,-0.022808,0.029798,0.043216,0.65879,-0.329744,-0.053815,0.734841,-1.410163,-0.04423,-0.055161,-0.514216,0.379111,0.080429,1.108862,-0.794685]],[[19,42,69,38,-2,-2,4,-2,-2,2,10,-2,-2,1,-2,-2,42,55,-2,-2,-2],[0.5,0.13076923787593842,1.5,0.5,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.9350000023841858,0.4166666716337204,-2.0,-2.0,0.5,-2.0,-2.0,0.3166666775941849,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.001904,0.012896,0.050861,0.008843,-0.153915,1.264129,0.178522,1.563604,0.338522,-0.013908,0.047713,0.914813,0.235241,-0.039913,-0.318866,1.320704,-0.09803,-0.114866,-1.225751,-1.136931,-0.914205]],[[10,15,63,44,-2,-2,-2,-2,17,9,7,-2,-2,42,-2,-2,-2],[0.2678571492433548,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.3815000057220459,0.04450000077486038,-2.0,-2.0,0.47727273404598236,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.004077,-0.032671,-0.040916,-0.050561,-0.188376,-0.601821,0.510774,0.543126,0.033301,0.044814,-0.008352,0.384325,-1.018176,0.078705,0.843331,-0.777673,-1.297672]],[[16,42,69,2,-2,-2,7,-2,-2,2,2,-2,-2,15,-2,-2,9,-2,-2],[0.5,0.02777777798473835,1.5,0.9850000143051147,-2.0,-2.0,0.2290000021457672,-2.0,-2.0,0.9350000023841858,0.8250000178813934,-2.0,-2.0,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.00027,0.007679,0.059756,0.031565,0.594867,-0.429667,0.180109,0.555758,1.738957,-0.014409,0.036177,0.248511,0.898139,-0.035041,-0.382495,0.582961,-0.135696,-1.579083,-0.480916]],[[63,8,16,42,-2,-2,-2,12,-2,6,-2,-2,8,-2,-2],[0.5,0.6304999887943268,0.5,0.029411764815449715,-2.0,-2.0,-2.0,0.24699999392032623,-2.0,0.3955000042915344,-2.0,-2.0,0.8989999890327454,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,14,-1,-1],[0.0017,-0.005373,0.022405,0.031121,0.993038,0.105616,-0.6421,-0.036693,1.340156,-0.047018,-0.852865,-0.186639,0.114293,-0.165962,1.310568]],[[1,32,21,9,-2,-2,-2,12,-2,-2,46,6,12,-2,-2,3,-2,-2,8,-2,-2],[0.5,0.5,0.5,0.9514999985694885,-2.0,-2.0,-2.0,0.4989999979734421,-2.0,-2.0,1.5,0.948500007390976,0.7354999780654907,-2.0,-2.0,0.05199999921023846,-2.0,-2.0,0.04449999891221523,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,-1],[-0.01185,-0.025564,-0.036972,-0.044604,-0.720875,-0.139176,0.711166,0.079403,0.976776,0.238838,0.044068,0.035152,0.00456,-0.415124,1.091004,0.055218,1.05765,1.135198,0.174834,1.131024,1.39055]],[[42,22,17,16,-2,-2,12,-2,-2,2,-2,-2,5,-2,8,-2,3,-2,-2],[0.4580419659614563,0.5,0.5,0.5,-2.0,-2.0,0.4960000216960907,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.01850000023841858,-2.0,0.20600000023841858,-2.0,0.6790000200271606,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,15,-1,17,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,16,-1,18,-1,-1],[0.001953,0.01034,0.01853,0.025512,0.267261,-0.58929,-0.103637,0.090474,-1.410213,-0.108581,-1.307086,-1.070749,-0.099339,0.2322,-0.132356,-1.579954,-0.096968,-0.627017,-1.371652]],[[11,21,63,44,-2,-2,6,-2,-2,-2,17,10,3,-2,-2,12,-2,-2,-2],[0.20714285969734192,0.5,0.5,0.5,-2.0,-2.0,0.6420000195503235,-2.0,-2.0,-2.0,0.5,0.5833333432674408,0.051500000059604645,-2.0,-2.0,0.7080000042915344,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.000837,-0.021395,-0.02795,-0.038039,-0.072582,-0.579811,0.079055,1.173168,-0.253516,0.749964,0.043603,0.052523,0.07694,0.664003,1.179927,0.020363,-0.240273,0.920332,-0.80657]],[[42,69,53,8,-2,-2,6,-2,-2,7,9,-2,-2,4,-2,-2,2,39,45,-2,-2,-2,33,4,-2,-2,-2],[0.02777777798473835,0.5,0.5,0.27549999952316284,-2.0,-2.0,0.4584999978542328,-2.0,-2.0,0.10349999740719795,0.9855000078678131,-2.0,-2.0,0.07400000095367432,-2.0,-2.0,0.9350000023841858,0.5,0.5365000069141388,-2.0,-2.0,-2.0,0.5,0.05849999934434891,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.001983,0.041288,-0.002424,0.053431,0.560322,1.116072,-0.057102,-1.072088,0.004056,0.125017,0.022701,0.711856,-0.384208,0.215528,1.890516,0.966473,-0.014469,0.028358,0.037254,0.654877,-0.506992,-1.014446,-0.033129,-0.027976,0.109043,-0.401966,-1.527355]],[[38,53,46,2,-2,-2,4,-2,-2,11,5,-2,-2,7,-2,-2,42,6,9,-2,-2,-2,12,2,-2,-2,-2],[0.5,0.5,1.5,0.9850000143051147,-2.0,-2.0,0.10249999910593033,-2.0,-2.0,0.4166666716337204,0.21049999445676804,-2.0,-2.0,0.005499999970197678,-2.0,-2.0,0.22474747896194458,0.9884999990463257,0.9404999911785126,-2.0,-2.0,-2.0,0.5195000171661377,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.000846,-0.011562,0.032294,0.022434,0.549207,-1.20147,0.13426,0.831771,1.337425,-0.032564,-0.048308,-0.671046,-0.200898,0.043176,1.005808,-0.457664,0.0564,0.106996,0.144085,0.960827,1.276108,-0.314233,-0.069664,0.032385,0.905918,-0.320953,-1.307748]],[[21,19,17,2,-2,-2,37,-2,-2,55,2,-2,-2,3,-2,-2,-2],[0.5,0.5,0.5,0.9600000083446503,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.29749999940395355,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.000346,-0.005106,0.004158,0.011236,0.352179,-0.21468,-0.110515,-1.482461,-0.310987,-0.080521,-0.114957,-1.242955,-1.170847,-0.052585,-1.023313,-0.284474,1.005013]],[[43,65,53,39,-2,-2,4,-2,-2,6,-2,-2,7,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.08950000256299973,-2.0,-2.0,0.4455000013113022,-2.0,-2.0,0.031000000424683094,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.004156,-0.003583,-0.012326,0.025133,0.483979,-0.823476,-0.034151,0.029935,-0.587692,0.107173,0.584405,0.906607,0.103845,1.10902,0.024084]],[[42,2,5,11,-2,-2,3,-2,-2,7,-2,4,-2,-2,2,4,37,-2,-2,6,-2,-2,1,46,-2,-2,-2],[0.029411764815449715,0.9850000143051147,0.21199999749660492,0.4166666716337204,-2.0,-2.0,0.061000000685453415,-2.0,-2.0,0.5690000057220459,-2.0,0.031000000424683094,-2.0,-2.0,0.9350000023841858,0.19299999624490738,1.5,-2.0,-2.0,0.9539999961853027,-2.0,-2.0,0.5,2.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[-0.008063,0.029674,0.06463,0.014959,-0.472607,1.128823,0.127073,0.457781,1.226958,-0.048236,0.246483,-0.086155,-1.175172,-0.423589,-0.024568,0.022448,0.05748,0.527053,1.286664,-0.003173,-0.701655,0.660324,-0.044292,-0.049422,-0.336828,-1.162448,1.320249]],[[54,38,16,2,-2,-2,9,-2,-2,42,3,-2,-2,5,-2,-2,-2],[0.5,0.5,0.5,0.9600000083446503,-2.0,-2.0,0.8730000257492065,-2.0,-2.0,0.22474747896194458,0.020999999716877937,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.006518,0.001277,-0.009697,-0.001384,0.212101,-0.309699,-0.132594,-1.4787,-0.745863,0.055005,0.107206,0.284817,1.176925,-0.066163,0.660929,-0.876654,0.982625]],[[19,42,69,70,-2,-2,-2,66,7,-2,-2,-2,8,8,-2,-2,55,-2,-2],[0.5,0.11324786394834518,2.5,0.5,-2.0,-2.0,-2.0,0.5,0.004500000039115548,-2.0,-2.0,-2.0,0.7374999821186066,0.42249999940395355,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,-1,-1,17,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,16,15,-1,-1,18,-1,-1],[0.008388,0.016787,0.050957,0.028878,0.121683,1.243098,1.564038,-0.003215,0.005255,0.480493,-0.206892,-1.625348,-0.068437,-0.039549,-0.870061,0.07177,-0.099478,-1.288483,-1.124296]],[[42,69,38,6,-2,-2,-2,4,-2,-2,5,10,44,-2,-2,2,-2,-2,39,2,-2,-2,-2],[0.11324786394834518,1.5,0.5,0.972000002861023,-2.0,-2.0,-2.0,0.0364999994635582,-2.0,-2.0,0.02850000001490116,0.3693181872367859,0.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.5,0.9350000023841858,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.000212,0.030431,0.003909,-0.015323,-0.425186,0.725427,1.230995,0.123801,1.570417,0.311614,-0.016951,0.032091,-0.019619,0.93005,-0.499121,0.078408,0.65016,1.051397,-0.036852,-0.031361,0.419652,-0.442952,-1.032963]],[[16,19,5,7,-2,-2,37,-2,-2,8,8,-2,-2,12,-2,-2,3,-2,-2],[0.5,0.5,0.9835000038146973,0.9354999959468842,-2.0,-2.0,1.5,-2.0,-2.0,0.6675000190734863,0.330499991774559,-2.0,-2.0,0.5439999997615814,-2.0,-2.0,0.0949999988079071,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.001467,0.007305,0.015458,0.024866,0.164183,0.910411,-0.050294,-0.89911,0.039396,-0.065427,-0.029985,-0.801945,0.175231,-0.09795,-1.171196,-1.247057,-0.106418,-0.645946,-1.426874]],[[8,4,9,10,-2,-2,-2,12,53,-2,-2,12,-2,-2,63,48,44,-2,-2,-2,-2],[0.09149999916553497,0.8700000047683716,0.99549999833107,0.3693181872367859,-2.0,-2.0,-2.0,0.47450000047683716,0.5,-2.0,-2.0,0.5760000050067902,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[-0.001861,0.044941,0.077192,0.05854,0.088021,0.989134,1.491996,0.014562,-0.057459,0.693021,-1.253767,0.050451,1.492213,0.13562,-0.018077,-0.027084,-0.035906,-0.081649,-0.582716,0.624586,0.618627]],[[48,8,9,9,-2,-2,8,-2,-2,42,38,-2,-2,12,-2,-2,8,-2,-2],[0.5,0.1314999982714653,0.5284999907016754,0.2070000022649765,-2.0,-2.0,0.06149999983608723,-2.0,-2.0,0.22649572789669037,0.5,-2.0,-2.0,0.38099999725818634,-2.0,-2.0,0.45000000298023224,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.003446,-0.010687,0.033014,0.000876,0.378366,-0.882742,0.056848,0.400361,1.465904,-0.027538,-0.008724,-0.233885,0.695776,-0.064317,-0.082414,-0.879176,0.09229,0.111694,1.242483]],[[42,12,6,4,-2,-2,-2,17,12,-2,-2,9,-2,-2,5,44,-2,-2,4,3,-2,-2,-2],[0.42217741906642914,0.2580000013113022,0.9749999940395355,0.012500000186264515,-2.0,-2.0,-2.0,0.5,0.3894999921321869,-2.0,-2.0,0.9010000228881836,-2.0,-2.0,0.01899999938905239,0.5,-2.0,-2.0,0.9050000011920929,0.6790000200271606,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,19,20,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,22,21,-1,-1,-1],[0.001971,0.010932,0.106397,0.179954,-0.399161,1.821249,-0.408946,0.004218,0.011396,-0.546062,0.235583,-0.116821,-1.322354,-0.51364,-0.07818,0.020656,1.099232,-0.243077,-0.108359,-0.085613,-0.468579,-1.53241,-1.476349]],[[17,5,19,42,-2,-2,42,-2,-2,12,6,-2,-2,9,-2,-2,9,-2,-2],[0.5,0.9895000159740448,0.5,0.4580419659614563,-2.0,-2.0,0.3166666775941849,-2.0,-2.0,0.5534999966621399,0.6554999947547913,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,0.9050000011920929,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.002513,0.002856,0.010895,0.018579,0.23302,-0.729454,-0.061435,-1.115138,-0.433736,-0.072647,-0.108377,-0.53022,-1.319982,-0.027985,0.18999,-1.20919,-0.103538,-1.284609,-0.49988]],[[2,7,3,7,-2,-2,5,-2,-2,42,8,-2,-2,-2,12,-2,1,42,-2,-2,-2],[0.9600000083446503,0.23349999636411667,0.7254999876022339,0.006500000134110451,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.39444445073604584,0.9370000064373016,-2.0,-2.0,-2.0,0.18850000202655792,-2.0,0.5,0.12004802003502846,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,-1],[0.001796,0.023512,0.006925,0.022543,0.618155,-0.17175,-0.075982,-1.348698,-0.380013,0.104443,0.131636,0.459079,1.413225,-0.522757,-0.027491,1.320852,-0.033605,-0.038088,0.009451,-0.616191,1.294499]],[[21,11,44,37,-2,-2,6,-2,-2,9,4,-2,-2,44,-2,-2,-2],[0.5,0.36666667461395264,0.5,0.5,-2.0,-2.0,0.03150000050663948,-2.0,-2.0,0.9784999787807465,0.16250000149011612,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.005714,-0.010899,-0.023974,0.000594,-0.471737,0.263768,-0.057105,1.387799,-0.520029,0.02788,-0.001672,0.919895,-0.566397,0.086875,0.324001,1.443732,0.987654]],[[17,42,19,16,-2,-2,12,-2,-2,5,9,-2,-2,2,-2,-2,4,-2,-2],[0.5,0.33908046782016754,0.5,0.5,-2.0,-2.0,0.6349999904632568,-2.0,-2.0,0.006500000134110451,0.8260000050067902,-2.0,-2.0,0.875,-2.0,-2.0,0.5965000092983246,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.004339,0.001406,0.010926,0.018517,0.215188,-0.710857,-0.058882,-0.585999,-1.193796,-0.044983,0.08542,1.310977,0.488212,-0.069238,0.791564,-1.030052,-0.107078,-1.464899,-0.422799]],[[19,5,44,69,-2,-2,6,-2,-2,2,42,-2,-2,5,-2,-2,9,-2,2,4,-2,-2,-2],[0.5,0.9715000092983246,0.5,0.5,-2.0,-2.0,0.025500000454485416,-2.0,-2.0,0.9850000143051147,0.36666667461395264,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,0.6209999918937683,-2.0,0.9600000083446503,0.23149999231100082,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,-1],[-0.000117,0.006872,0.016645,0.04325,0.336521,0.978673,-0.012765,1.214632,-0.157563,-0.048927,0.021934,0.422389,-1.23578,-0.083437,-0.941517,-1.167991,-0.064039,-0.131688,-0.077712,-0.104437,-1.165525,-1.241622,-1.073569]],[[39,42,22,16,-2,-2,6,-2,-2,42,2,-2,-2,-2,12,-2,-2],[0.5,0.5657142996788025,0.5,0.5,-2.0,-2.0,0.9625000059604645,-2.0,-2.0,0.7321428656578064,0.9750000238418579,-2.0,-2.0,-2.0,0.4544999897480011,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.001864,0.00222,0.005667,0.011137,0.159332,-0.648361,-0.072854,-1.19907,-0.735011,-0.129136,-0.088472,-1.392887,-1.056342,-1.779952,-0.121312,0.390118,-1.395207]],[[71,11,4,69,-2,-2,5,-2,-2,10,44,-2,-2,9,-2,-2,-2],[0.5,0.36666667461395264,0.351500004529953,2.5,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.8333333432674408,0.5,-2.0,-2.0,0.0625,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.001355,0.004474,-0.008883,0.008117,-0.045973,0.642572,-0.040309,0.543117,-0.629853,0.042301,0.071089,0.550018,1.010864,0.008437,1.06431,-0.696968,-0.952379]],[[43,12,6,9,-2,-2,3,-2,-2,17,42,-2,-2,4,-2,-2,12,-2,-2],[0.5,0.2580000013113022,0.800000011920929,0.9880000054836273,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,0.5,0.02777777798473835,-2.0,-2.0,0.5965000092983246,-2.0,-2.0,0.5259999930858612,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.00387,-0.002195,0.068354,0.176265,0.68082,2.448456,-0.030886,0.981663,-0.874319,-0.007414,-0.002589,0.341683,-0.185845,-0.11299,-1.550016,-0.36966,0.091491,1.071315,-0.107583]],[[35,56,65,16,-2,-2,42,-2,-2,5,-2,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0,0.13350000232458115,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[-0.002003,-0.006171,-0.001522,-0.007657,-0.012231,-0.943529,0.07791,1.317027,0.065689,-0.103433,-0.295716,-1.378118,1.000424]],[[35,42,69,2,-2,-2,4,-2,-2,44,5,-2,-2,11,-2,-2,-2],[0.5,0.03125,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.9905000030994415,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.004262,0.000316,0.031472,0.004941,0.290666,-0.535693,0.138681,1.304695,0.428962,-0.013202,0.010829,0.311872,-0.582795,-0.041334,-0.539238,0.228583,1.060783]],[[42,22,3,3,-2,-2,-2,8,-2,-2,5,9,3,-2,-2,-2,45,42,-2,-2,-2],[0.33974359929561615,0.5,0.99549999833107,0.992499977350235,-2.0,-2.0,-2.0,0.7374999821186066,-2.0,-2.0,0.006500000134110451,0.9469999969005585,0.0005000000237487257,-2.0,-2.0,-2.0,0.0005000000237487257,0.7321428656578064,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.005403,0.01493,0.02098,0.0244,0.185414,1.462289,-1.083425,-0.065973,-0.49489,-1.231324,-0.041754,0.066369,0.093721,0.775234,1.389666,-0.166356,-0.06103,-0.071961,-0.808228,-1.718396,0.916109]],[[10,12,5,69,-2,-2,2,-2,-2,5,69,-2,-2,63,-2,-2,4,8,11,-2,-2,-2,7,10,-2,-2,9,-2,-2],[0.2678571492433548,0.3994999974966049,0.053999999538064,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.00849999999627471,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.16099999845027924,0.2590000033378601,0.4166666716337204,-2.0,-2.0,-2.0,0.007500000298023224,0.3541666716337204,-2.0,-2.0,0.9479999840259552,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.000101,-0.017889,-0.074927,0.046489,0.991154,-0.515186,-0.106042,-1.161031,-0.753492,-0.000554,-0.147255,-0.092531,-1.418206,0.008917,-0.021144,0.976316,0.023589,0.069327,0.043113,-0.2215,0.935894,1.326693,0.003197,0.04073,-0.275075,0.819342,-0.042541,-0.747552,0.224492]],[[17,2,46,69,-2,-2,-2,6,3,-2,-2,5,-2,-2,42,-2,-2],[0.5,0.9600000083446503,1.5,2.5,-2.0,-2.0,-2.0,0.9149999916553497,0.001500000071246177,-2.0,-2.0,0.06850000098347664,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.003427,0.001641,0.020159,0.027656,0.163052,1.070002,-1.188768,-0.023192,0.002378,1.188152,-0.155722,-0.055852,0.12968,-0.804295,-0.101413,-1.226824,-0.220569]],[[19,42,37,70,-2,-2,7,-2,-2,12,42,-2,-2,70,-2,-2,42,2,-2,-2,-2],[0.5,0.20256410539150238,1.5,0.5,-2.0,-2.0,0.945499986410141,-2.0,-2.0,0.3824999928474426,0.31643131375312805,-2.0,-2.0,0.5,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.002993,0.009965,0.026364,0.001642,-0.157022,1.200537,0.080794,0.427187,1.490663,-0.015385,0.041786,0.894611,-0.078689,-0.042285,-0.298836,-1.048056,-0.05759,-0.074463,-1.129502,-1.073194,-0.598412]],[[5,3,44,2,-2,-2,6,-2,-2,8,-2,-2,8,-2,2,-2,69,-2,-2],[0.9794999957084656,0.9895000159740448,0.5,0.9950000047683716,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.9794999957084656,-2.0,0.9850000143051147,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,15,-1,17,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,16,-1,18,-1,-1],[0.0017,0.009903,0.005997,0.025938,0.583662,-0.504144,-0.014933,1.270832,-0.193887,0.126449,1.39513,-0.432013,-0.053312,-1.354988,-0.029172,0.347334,-0.069869,-0.695397,-1.034483]],[[42,19,43,37,-2,-2,5,-2,-2,6,-2,69,-2,-2,5,12,-2,-2,9,3,-2,-2,2,-2,-2],[0.42396312952041626,0.5,0.5,4.5,-2.0,-2.0,0.2045000046491623,-2.0,-2.0,0.6944999992847443,-2.0,0.5,-2.0,-2.0,0.019499999471008778,0.43449999392032623,-2.0,-2.0,0.9699999988079071,0.2514999955892563,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,19,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.000287,0.006205,0.012559,0.00702,0.110233,-0.585269,0.087021,0.178072,1.011402,-0.056419,-0.314246,-0.080663,-1.199259,-1.101861,-0.05193,0.037364,1.112812,-0.031598,-0.083184,-0.042828,-0.17262,-1.216632,-0.110087,-1.472477,-1.147023]],[[10,63,44,6,-2,-2,6,-2,-2,-2,17,39,5,-2,-2,1,-2,-2,-2],[0.2678571492433548,0.5,0.5,0.2939999997615814,-2.0,-2.0,0.038999998942017555,-2.0,-2.0,-2.0,0.5,0.5,0.5819999873638153,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.000739,-0.018338,-0.025767,-0.003262,-0.689108,0.228837,-0.05473,0.693677,-0.538369,0.52531,0.022569,0.028415,0.034383,0.546843,-0.535117,-0.079668,-1.185315,1.245543,-0.787096]],[[8,8,3,6,-2,-2,-2,-2,21,63,37,-2,-2,-2,-2],[0.0925000011920929,0.08249999955296516,0.048499999567866325,0.7590000033378601,-2.0,-2.0,-2.0,-2.0,0.5,0.5,4.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,-1],[-0.000734,0.034996,0.028033,0.021396,0.782827,0.076717,0.892815,1.473203,-0.013251,-0.018241,-0.024816,-0.190538,-1.384473,0.389698,0.908947]],[[44,2,4,6,-2,-2,4,-2,-2,5,11,-2,-2,8,-2,-2,6,-2,43,63,-2,-2,-2],[0.5,0.9950000047683716,0.4389999955892563,0.9824999868869781,-2.0,-2.0,0.5725000202655792,-2.0,-2.0,0.5449999868869781,0.2678571492433548,-2.0,-2.0,0.9945000112056732,-2.0,-2.0,0.016999999526888132,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,-1],[-0.000375,0.016167,0.0311,0.060033,0.429433,1.912338,-0.004685,-1.30061,0.125702,-0.043133,0.088322,1.220093,0.977563,-0.072019,-1.085476,0.204996,-0.021358,1.251597,-0.029767,-0.041011,-0.435772,0.467756,0.480153]],[[12,8,5,42,-2,-2,-2,-2,8,9,4,-2,-2,7,-2,-2,3,69,-2,-2,69,-2,-2],[0.2580000013113022,0.38449999690055847,0.13350000232458115,0.2678571492433548,-2.0,-2.0,-2.0,-2.0,0.6304999887943268,0.7695000171661377,0.060499999672174454,-2.0,-2.0,0.887499988079071,-2.0,-2.0,0.007500000298023224,0.5,-2.0,-2.0,2.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.002927,0.072353,0.006037,0.065837,1.362787,0.40256,-0.927033,1.518587,-0.008294,0.014111,-0.020032,1.048238,-0.595761,0.04471,0.61459,-0.92534,-0.030231,-0.125102,-0.554512,-1.363226,-0.019494,-0.333805,0.494201]],[[17,42,70,6,-2,-2,6,-2,-2,8,3,-2,-2,1,-2,-2,9,-2,-2],[0.5,0.2752525359392166,0.5,0.9514999985694885,-2.0,-2.0,0.9354999959468842,-2.0,-2.0,0.05849999934434891,0.009499999694526196,-2.0,-2.0,0.5,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.00399,0.008459,0.021254,0.012277,-0.115781,0.498837,0.077608,1.079807,-0.084451,-0.024171,0.032617,0.848726,-0.757941,-0.042553,-0.585693,0.976872,-0.080012,-1.1792,-0.371287]],[[5,44,2,7,-2,-2,46,-2,-2,12,48,-2,-2,-2,8,-2,2,-2,6,-2,-2],[0.9905000030994415,0.5,0.9950000047683716,0.9505000114440918,-2.0,-2.0,1.5,-2.0,-2.0,0.8810000121593475,0.5,-2.0,-2.0,-2.0,0.9799999892711639,-2.0,0.9850000143051147,-2.0,0.3215000033378601,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,20,-1,-1],[-0.000545,0.005022,0.024526,0.040672,0.381568,1.401666,-0.042232,-1.067689,0.98368,-0.015228,-0.024197,-0.288787,0.3092,1.188763,-0.061333,-1.347377,-0.036393,0.240856,-0.078567,-1.063577,-1.165223]],[[71,43,8,4,-2,-2,42,-2,-2,7,-2,-2,-2],[0.5,0.5,0.1314999982714653,0.9460000097751617,-2.0,-2.0,0.19090909510850906,-2.0,-2.0,0.031000000424683094,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[-0.000666,0.001808,-0.002961,0.028317,0.67796,-0.041134,-0.015184,0.065199,-0.553884,0.072034,1.131075,-0.228599,-0.741828]],[[19,2,7,46,-2,-2,42,-2,-2,42,37,-2,-2,8,-2,-2,42,2,-2,-2,-2],[0.5,0.9750000238418579,0.8925000131130219,1.5,-2.0,-2.0,0.3166666775941849,-2.0,-2.0,0.11437908560037613,1.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.002147,0.008365,0.023824,0.014161,0.215144,-0.862079,0.154238,1.497574,-0.064082,-0.017575,0.020351,-0.310948,0.939751,-0.038617,0.434398,-0.797623,-0.056186,-0.068879,-1.086412,-1.088176,-0.785134]],[[37,42,12,6,-2,-2,17,-2,-2,-2,-2],[4.5,0.7321428656578064,0.2580000013113022,0.792000025510788,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.003192,3.2e-05,0.001318,0.062394,1.382118,-0.124445,-0.003213,0.010705,-0.764855,-1.751231,-0.660285]],[[56,54,30,8,-2,-2,8,-2,-2,-2,69,-2,-2],[0.5,0.5,0.5,0.590499997138977,-2.0,-2.0,0.7730000019073486,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[-0.005773,-0.001738,-0.005588,-0.010825,0.127621,-0.313171,0.050791,-0.713144,1.021319,0.84421,-0.071357,0.419946,-1.03556]],[[16,42,12,42,-2,-2,54,-2,-2,2,-2,-2,4,-2,-2],[0.5,0.5857143104076385,0.21249999850988388,0.3166666775941849,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.8734999895095825,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[-0.000685,0.003539,0.006375,0.113965,1.736847,0.328241,0.002978,-0.007622,0.812562,-0.106074,-1.671553,-1.174307,-0.074507,-1.314746,-0.06989]],[[5,44,8,5,-2,-2,4,-2,-2,4,37,-2,-2,11,-2,-2,42,-2,12,2,-2,-2,5,-2,-2],[0.9794999957084656,0.5,0.9494999945163727,0.6845000088214874,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.030499999411404133,1.5,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.04545454680919647,-2.0,0.507999986410141,0.9950000047683716,-2.0,-2.0,0.99549999833107,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,24,-1,-1],[-0.005082,0.00207,0.023726,0.010632,0.441054,-0.347923,0.083266,-0.929369,1.145535,-0.019913,0.02863,-0.243227,0.904438,-0.036227,-0.521858,0.14702,-0.058042,-0.093046,-0.076223,-0.104397,-1.460948,-1.078079,-0.053188,-1.07265,-0.539612]],[[16,66,56,51,-2,-2,6,-2,-2,69,-2,-2,9,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.9695000052452087,-2.0,-2.0,1.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[-0.005378,-0.000971,0.002857,0.007477,0.039237,0.829948,-0.07404,0.121419,-1.114117,-0.091569,-0.155594,-1.402825,-0.075858,-1.11485,-0.213716]],[[21,4,4,69,-2,-2,-2,1,6,-2,-2,46,-2,-2,9,-2,-2],[0.5,0.05849999934434891,0.05049999989569187,2.5,-2.0,-2.0,-2.0,0.5,0.9975000023841858,-2.0,-2.0,1.5,-2.0,-2.0,0.9774999916553497,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.001552,-0.005479,0.0189,0.009147,-0.060531,0.71462,1.299114,-0.021656,-0.039061,-0.300727,-1.269766,0.022029,0.300061,1.284781,0.079736,0.915081,0.441512]],[[42,69,6,9,-2,-2,5,-2,-2,4,-2,6,-2,-2,69,10,42,-2,-2,7,-2,-2,9,3,-2,-2,-2],[0.13076923787593842,1.5,0.3095000088214874,0.7725000083446503,-2.0,-2.0,0.19249999523162842,-2.0,-2.0,0.07649999856948853,-2.0,0.9309999942779541,-2.0,-2.0,1.5,0.8333333432674408,0.4580419659614563,-2.0,-2.0,0.04800000041723251,-2.0,-2.0,0.99549999833107,0.011500000022351742,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[0.003541,0.026458,0.003816,-0.064054,1.398671,-1.267077,0.024502,-0.153089,0.641825,0.093477,1.080183,0.009075,0.729334,-0.669428,-0.012191,0.001454,0.011335,0.266249,-0.607469,-0.065815,-0.276388,-1.421763,-0.052353,-0.080667,-0.948426,-0.584525,0.332894]],[[70,6,3,12,-2,-2,16,-2,-2,4,42,-2,-2,2,-2,-2,42,9,-2,-2,-2],[0.5,0.9535000026226044,0.001500000071246177,0.6599999964237213,-2.0,-2.0,0.5,-2.0,-2.0,0.08449999988079071,0.1339285746216774,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.25357143580913544,0.9819999933242798,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.008351,0.000533,-0.015643,0.057106,1.137949,-0.400237,-0.026113,-0.191955,-1.378942,0.025453,0.085796,1.004231,0.228692,0.008792,0.406742,-0.334173,0.068386,0.103613,0.477028,1.224497,-0.390936]],[[42,69,3,3,-2,-2,7,-2,-2,4,-2,-2,66,2,46,-2,-2,69,-2,-2,-2],[0.11324786394834518,1.5,0.9245000183582306,0.3410000056028366,-2.0,-2.0,0.016000000294297934,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.9600000083446503,1.5,-2.0,-2.0,1.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[0.002788,0.025426,0.007453,0.02184,0.06083,0.660714,-0.055051,-1.346006,-0.174814,0.093495,1.242425,0.179676,-0.009765,-0.004935,0.0145,0.198051,-0.835926,-0.030443,-0.089955,-0.882727,-1.127626]],[[71,32,73,1,-2,-2,-2,6,6,-2,-2,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9699999988079071,0.5595000088214874,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,-1],[0.000626,0.002815,-0.002028,-0.004658,-0.129734,0.651307,0.949355,0.050073,0.077842,0.730601,1.06615,-0.401761,-0.858077]],[[11,63,44,5,-2,-2,9,-2,-2,-2,9,4,6,-2,-2,12,-2,-2,44,13,-2,-2,12,-2,-2],[0.36666667461395264,0.5,0.5,0.9905000030994415,-2.0,-2.0,0.9934999942779541,-2.0,-2.0,-2.0,0.7730000019073486,0.09699999913573265,0.013499999884516,-2.0,-2.0,0.6335000097751617,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5164999961853027,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.00084,-0.010129,-0.015173,0.004158,0.223528,-0.767116,-0.04062,-0.164464,-0.750154,0.442122,0.02614,-0.010137,0.056326,0.148527,1.235312,-0.036255,-1.032782,0.128877,0.062837,0.018863,0.647647,-1.105478,0.118614,0.641907,1.390831]],[[6,64,6,44,-2,-2,70,-2,-2,-2,12,4,-2,69,-2,-2,12,4,-2,-2,2,-2,-2],[0.9575000107288361,0.5,0.02850000001490116,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.5024999976158142,0.017500000074505806,-2.0,1.5,-2.0,-2.0,0.6270000040531158,0.5194999873638153,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.001888,-0.013999,-0.020403,0.029453,-0.150654,1.296133,-0.029725,-0.398633,0.333922,1.034248,0.018319,-0.012653,0.751482,-0.026332,-0.031108,-1.011744,0.056354,0.119848,1.917295,0.806826,0.008489,0.917037,-0.925421]],[[5,2,42,69,-2,-2,42,-2,-2,4,-2,3,-2,-2,4,2,3,-2,-2,4,-2,-2,-2],[0.6845000088214874,0.9950000047683716,0.28285714983940125,2.5,-2.0,-2.0,0.2952069789171219,-2.0,-2.0,0.02350000012665987,-2.0,0.0025000000605359674,-2.0,-2.0,0.9510000050067902,0.9950000047683716,0.8039999902248383,-2.0,-2.0,0.08549999818205833,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.003821,0.016828,0.009065,0.023243,0.169585,0.746078,-0.026593,-1.162303,-0.114736,0.085445,1.668492,0.027784,1.317675,-0.104872,-0.026121,-0.014934,0.01181,-0.242433,0.549836,-0.064786,-1.159586,-0.562957,-1.312434]],[[3,42,39,38,-2,-2,5,-2,-2,-2,7,-2,-2],[0.99549999833107,0.7321428656578064,0.5,0.5,-2.0,-2.0,0.04399999976158142,-2.0,-2.0,-2.0,0.9560000002384186,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[-0.000741,0.002158,0.00335,0.005997,-0.019059,0.47401,-0.077257,0.371606,-1.039568,-1.665513,-0.127107,-1.467686,-1.267573]],[[67,3,3,9,-2,-2,2,-2,-2,-2,-2],[0.5,0.9964999854564667,0.9895000159740448,0.9985000193119049,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002712,-0.000359,0.00138,-0.001947,-0.053982,1.003226,0.076211,1.154774,0.053975,-1.482661,0.834618]],[[38,16,17,6,-2,-2,9,-2,-2,9,-2,-2,46,42,53,-2,-2,3,-2,-2,-2],[0.5,0.5,0.5,0.7054999768733978,-2.0,-2.0,0.93299999833107,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,2.5,0.2613636404275894,0.5,-2.0,-2.0,0.012500000186264515,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,-1],[-0.003817,-0.012134,-0.006903,-0.002535,-0.273941,0.13483,-0.079901,-1.229341,-0.452071,-0.088405,-1.35914,-0.642673,0.035211,0.048371,0.080753,0.587449,1.035889,-0.039251,0.758821,-1.176503,-1.143769]],[[71,5,5,3,-2,-2,42,-2,-2,3,6,-2,-2,6,-2,-2,-2],[0.5,0.5494999885559082,0.21199999749660492,0.34700000286102295,-2.0,-2.0,0.39444445073604584,-2.0,-2.0,0.057499999180436134,0.9809999763965607,-2.0,-2.0,0.9914999902248383,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.004112,-0.00193,0.010583,-0.000591,0.167629,-0.459141,0.052949,0.743803,-1.0192,-0.028005,-0.089095,-0.579689,-1.336036,-0.004779,-0.216031,1.372745,-0.908666]],[[71,42,12,9,-2,-2,12,-2,-2,-2,-2],[0.5,0.7321428656578064,0.2580000013113022,0.9774999916553497,-2.0,-2.0,0.3110000044107437,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[4.1e-05,0.002428,0.003607,0.061849,0.10567,1.188667,-0.000763,-0.88225,0.052051,-1.608972,-0.864489]],[[54,44,6,2,-2,-2,3,-2,-2,70,6,-2,-2,42,-2,-2,-2],[0.5,0.5,0.9824999868869781,0.9850000143051147,-2.0,-2.0,0.3595000058412552,-2.0,-2.0,0.5,0.02850000001490116,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.000431,-0.003474,0.010951,-0.005888,0.209155,-0.708354,0.052412,0.337471,1.804026,-0.02334,-0.039229,1.018265,-0.424982,0.046182,0.843576,-0.693831,0.790556]],[[3,44,2,3,-2,-2,5,-2,-2,6,-2,11,-2,-2,-2],[0.9964999854564667,0.5,0.9950000047683716,0.8760000169277191,-2.0,-2.0,0.2485000044107437,-2.0,-2.0,0.009499999694526196,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.003477,0.005401,0.01924,0.034605,0.306503,1.076721,-0.035759,1.149368,-0.83391,-0.01145,1.520792,-0.017605,-0.305503,0.318191,-1.392029]],[[2,7,7,42,-2,-2,12,-2,-2,6,12,-2,-2,4,-2,-2,46,6,3,-2,-2,6,-2,-2,3,53,-2,-2,-2],[0.9750000238418579,0.25949999690055847,0.004500000039115548,0.22649572789669037,-2.0,-2.0,0.6344999969005585,-2.0,-2.0,0.45350000262260437,0.8324999809265137,-2.0,-2.0,0.38500000536441803,-2.0,-2.0,1.5,0.9715000092983246,0.007500000298023224,-2.0,-2.0,0.9945000112056732,-2.0,-2.0,0.12850000336766243,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,-1],[0.002764,0.014179,9e-05,0.022431,0.521204,-0.144438,-0.026287,-0.503124,0.198435,0.079508,0.01605,-0.362787,0.918946,0.139081,1.782897,-0.601449,-0.016845,-0.027378,-0.013395,0.781186,-0.442874,-0.066521,-1.228658,-0.123687,0.060773,0.023646,1.030881,-0.836947,1.209228]],[[42,70,50,6,-2,-2,-2,9,-2,-2,8,9,3,-2,-2,-2,1,12,-2,-2,-2],[0.2695630192756653,0.5,0.5,0.9519999921321869,-2.0,-2.0,-2.0,0.9799999892711639,-2.0,-2.0,0.07050000131130219,0.99549999833107,0.013500000350177288,-2.0,-2.0,-2.0,0.5,0.3244999945163727,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,-1],[0.001674,0.012116,0.00299,0.007842,-0.099999,0.385381,-0.94332,0.079231,0.369601,1.107312,-0.022982,0.024937,0.008212,0.49895,-0.928115,1.36166,-0.03906,-0.048006,0.088928,-0.827083,1.032951]],[[16,2,39,53,-2,-2,-2,15,7,-2,-2,7,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.06650000065565109,-2.0,-2.0,0.029000000096857548,-2.0,-2.0,0.9580000042915344,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[0.000507,0.004541,0.030026,0.03602,0.639603,0.967108,-0.944874,-0.006699,-0.013417,-0.336859,0.074651,0.065226,1.138892,-0.213204,-0.058274,-1.237167,0.339201]],[[11,44,6,2,-2,-2,3,-2,-2,2,9,-2,-2,4,-2,-2,7,8,42,-2,-2,2,-2,-2,12,9,-2,-2,7,-2,-2],[0.4166666716337204,0.5,0.9779999852180481,0.9850000143051147,-2.0,-2.0,0.260500006377697,-2.0,-2.0,0.9950000047683716,0.9934999942779541,-2.0,-2.0,0.0669999998062849,-2.0,-2.0,0.004500000039115548,0.051500000059604645,0.2613636404275894,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.7240000069141388,0.35100001096725464,-2.0,-2.0,0.035499999299645424,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.00088,-0.011381,0.006365,-0.011065,0.1676,-0.795598,0.063729,0.094119,1.736771,-0.033518,-0.047181,-0.234602,-0.806895,0.060439,1.206933,-0.543631,0.028337,0.055812,0.023931,0.350971,1.151119,0.133309,1.180704,1.290235,-2.9e-05,-0.018962,-1.031026,0.15057,0.042075,0.977037,0.751597]],[[21,17,66,70,-2,-2,4,-2,-2,4,-2,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.042000001296401024,-2.0,-2.0,0.6445000171661377,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.002042,-0.001478,0.002586,0.005882,-0.012804,0.645336,-0.078783,0.029037,-1.09731,-0.086118,-1.429282,-0.134228,0.837959]],[[5,12,8,5,-2,-2,-2,12,53,-2,-2,63,-2,-2,-2],[0.9964999854564667,0.2580000013113022,0.38449999690055847,0.19749999791383743,-2.0,-2.0,-2.0,0.31049999594688416,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[0.002709,0.005454,0.057729,0.012516,1.056963,-0.912939,1.09326,0.00159,-0.055673,0.597602,-0.926346,0.00616,0.014161,0.626508,-1.067905]],[[3,55,12,8,-2,-2,17,-2,-2,42,-2,-2,42,-2,2,-2,-2],[0.992499977350235,0.5,0.2580000013113022,0.6084999740123749,-2.0,-2.0,0.5,-2.0,-2.0,0.2967033088207245,-2.0,-2.0,0.0357142873108387,-2.0,0.9650000035762787,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,15,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,16,-1,-1],[-0.001178,0.001846,0.005805,0.055435,0.198354,1.328505,0.001882,0.06552,-0.69423,-0.052947,-1.178467,-0.649668,-0.078804,-0.138679,-0.117792,-1.440507,-1.060998]],[[7,3,2,8,-2,-2,8,-2,-2,3,4,-2,-2,12,-2,-2,4,8,9,-2,-2,12,-2,-2,0,46,-2,-2,-2],[0.007500000298023224,0.7685000002384186,0.9600000083446503,0.8425000011920929,-2.0,-2.0,0.5405000150203705,-2.0,-2.0,0.9665000140666962,0.04999999888241291,-2.0,-2.0,0.7129999995231628,-2.0,-2.0,0.14949999749660492,0.9845000207424164,0.9774999916553497,-2.0,-2.0,0.44099999964237213,-2.0,-2.0,0.5,1.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,-1],[-0.002688,0.016326,0.025544,0.038867,0.463803,1.166612,-0.013732,0.235186,-0.937465,-0.042164,-0.11378,-1.433718,-0.597984,0.063863,0.916986,-0.272301,-0.016313,0.005548,0.030795,-0.052683,0.640228,-0.031715,-1.294702,-0.114672,-0.040578,-0.034628,-0.334121,-1.072994,-1.041051]],[[19,33,38,6,-2,-2,42,-2,-2,-2,12,8,-2,7,-2,-2,-2],[0.5,0.5,0.5,0.8819999992847443,-2.0,-2.0,0.2899159789085388,-2.0,-2.0,-2.0,0.6435000002384186,0.42249999940395355,-2.0,0.007500000298023224,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,-1],[-0.000729,0.0046,0.006979,0.000277,-0.161761,0.170831,0.040367,0.78121,-0.481158,-0.851176,-0.046617,-0.036747,-0.944656,-0.02377,-0.206036,-0.934526,-1.197984]],[[71,19,5,16,-2,-2,9,-2,-2,42,2,-2,-2,-2,-2],[0.5,0.5,0.9895000159740448,0.5,-2.0,-2.0,0.9945000112056732,-2.0,-2.0,0.3166666775941849,0.9750000238418579,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,-1],[-0.002775,-0.00073,0.004479,0.009978,0.168587,-0.617957,-0.04314,-0.146951,-0.934167,-0.043052,-0.057225,-0.9358,-1.052739,-0.511194,-0.780853]],[[48,8,11,44,-2,-2,7,-2,-2,3,69,-2,-2,37,-2,-2,6,-2,42,-2,-2],[0.5,0.17599999904632568,0.36666667461395264,0.5,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0,0.014500000048428774,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7529999911785126,-2.0,0.18333333730697632,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[-0.00014,-0.004245,0.017305,-0.014658,0.777064,-0.718645,0.027194,0.911693,0.135244,-0.014266,-0.064109,-0.220658,-0.827294,-0.003057,-0.492861,0.218474,0.051129,-0.240261,0.087131,0.288198,0.963579]],[[42,69,6,2,-2,-2,12,-2,-2,7,8,-2,-2,2,-2,-2,66,4,42,-2,-2,4,-2,-2,-2],[0.02777777798473835,0.5,0.6735000014305115,0.44999998807907104,-2.0,-2.0,0.5460000038146973,-2.0,-2.0,0.10349999740719795,0.5044999867677689,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.05849999934434891,0.2613636404275894,-2.0,-2.0,0.0794999971985817,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[-0.003547,0.020708,-0.001628,-0.049916,0.575192,-1.028858,0.026777,-0.429745,0.973526,0.056641,-0.010057,0.460629,-0.681379,0.110248,1.366154,0.615034,-0.013871,-0.00903,0.013454,0.50695,-0.454457,-0.022622,-1.241759,-0.225486,-1.215994]],[[16,56,7,19,-2,-2,-2,5,-2,-2,6,-2,-2],[0.5,0.5,0.9975000023841858,0.5,-2.0,-2.0,-2.0,0.13350000232458115,-2.0,-2.0,0.875,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,12,-1,-1],[0.000803,0.004586,0.00822,0.010418,0.177469,-0.620604,-1.243714,-0.052618,0.171673,-1.212938,-0.067562,-1.395557,-0.258737]],[[2,69,9,2,-2,-2,5,-2,-2,8,-2,-2,5,4,2,-2,-2,7,-2,-2,3,2,-2,-2,4,-2,-2],[0.9600000083446503,2.5,0.9359999895095825,0.75,-2.0,-2.0,0.2135000005364418,-2.0,-2.0,0.6034999787807465,-2.0,-2.0,0.3554999977350235,0.020999999716877937,0.9950000047683716,-2.0,-2.0,0.3544999957084656,-2.0,-2.0,0.8765000104904175,0.9850000143051147,-2.0,-2.0,0.03150000050663948,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.00041,0.013211,0.004986,-0.024635,0.124364,-0.644687,0.025551,-0.04179,0.661066,0.091414,-0.005884,1.152166,-0.01695,0.014722,0.109286,-0.453817,1.707149,-0.004039,-0.184317,1.613248,-0.040048,-0.065576,-1.193117,-0.602257,0.001732,-0.597553,1.29182]],[[2,40,12,-2,12,-2,-2,-2,1,9,6,-2,-2,8,-2,-2,-2],[0.9750000238418579,0.5,0.2150000035762787,-2.0,0.8409999907016754,-2.0,-2.0,-2.0,0.5,0.9914999902248383,0.9864999949932098,-2.0,-2.0,0.531000018119812,-2.0,-2.0,-2.0],[1,2,3,-1,5,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,-1],[8,7,4,-1,6,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,-1],[-0.001841,0.009711,0.00466,1.327935,0.001473,-0.062574,0.691365,1.601467,-0.021226,-0.02458,-0.039888,-0.377434,-1.116645,-0.000131,0.815452,-0.203976,1.321449]],[[63,8,38,16,-2,-2,36,-2,-2,5,9,-2,-2,9,-2,-2,-2],[0.5,0.87950000166893,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.09449999779462814,0.9964999854564667,-2.0,-2.0,0.9914999902248383,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.002428,-0.005532,0.007462,-0.000199,0.075879,-0.749508,0.048249,0.361435,1.026912,-0.028253,-0.088091,-0.597519,-1.068715,-0.015269,-0.701953,0.13839,0.491192]],[[42,17,22,48,-2,-2,8,-2,-2,7,-2,-2,5,42,-2,-2,45,5,-2,-2,-2],[0.33908046782016754,0.5,0.5,0.5,-2.0,-2.0,0.7290000021457672,-2.0,-2.0,0.01699999999254942,-2.0,-2.0,0.004500000039115548,0.44949495792388916,-2.0,-2.0,0.0005000000237487257,0.054999999701976776,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,-1,17,18,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,16,15,-1,-1,20,19,-1,-1,-1],[0.001238,0.007655,0.012057,0.016491,0.141535,0.698615,-0.048181,-0.501232,-1.215965,-0.067327,-1.288335,-0.285482,-0.028958,0.059031,1.027208,1.37932,-0.041286,-0.050182,-0.989387,-0.747338,0.942651]],[[10,63,44,8,-2,-2,2,-2,-2,-2,9,12,7,-2,-2,12,-2,-2,8,5,-2,-2,9,-2,-2],[0.2678571492433548,0.5,0.5,0.992499977350235,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,-2.0,0.45350000262260437,0.7354999780654907,0.02599999960511923,-2.0,-2.0,0.94200000166893,-2.0,-2.0,0.06149999983608723,0.018499999307096004,-2.0,-2.0,0.9839999973773956,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.003601,-0.017014,-0.022298,-0.006424,0.092695,-0.921192,-0.042978,0.489585,-0.568069,0.377687,0.013399,-0.01712,-0.059183,-0.218881,-1.230141,0.027563,0.310518,1.575982,0.034046,0.00746,0.803828,-0.31698,0.058599,0.33745,1.141337]],[[2,69,17,8,-2,-2,-2,6,-2,-2,4,4,12,-2,-2,-2,69,3,-2,-2,10,-2,-2],[0.9600000083446503,2.5,0.5,0.9934999942779541,-2.0,-2.0,-2.0,0.9290000200271606,-2.0,-2.0,0.07649999856948853,0.051500000059604645,0.466499999165535,-2.0,-2.0,-2.0,0.5,0.24900000542402267,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,18,-1,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,20,19,-1,-1,22,-1,-1],[0.000871,0.014891,0.005914,0.011606,0.087617,0.881567,-1.093447,0.091107,1.169067,-0.145277,-0.017829,0.002695,-0.012455,0.473189,-0.661634,0.907785,-0.035936,-0.00225,-0.38731,1.130845,-0.057135,-1.169679,-0.209927]],[[12,1,40,9,-2,-2,6,-2,-2,46,45,-2,-2,-2,8,37,8,-2,-2,3,-2,-2,-2],[0.8680000007152557,0.5,0.5,0.38350000977516174,-2.0,-2.0,0.93299999833107,-2.0,-2.0,3.5,0.4455000013113022,-2.0,-2.0,-2.0,0.9914999902248383,0.5,0.07449999824166298,-2.0,-2.0,0.0885000005364418,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.000208,-0.004109,-0.011105,-0.01479,-0.720035,-0.10289,0.054432,1.338085,-0.217222,0.024178,0.019784,1.047671,-0.044506,1.194959,0.037244,0.01837,-0.034482,1.145281,-0.959175,0.065061,0.605437,1.108743,1.134639]],[[39,9,69,17,-2,-2,3,-2,-2,48,63,-2,-2,-2,5,-2,-2],[0.5,0.945499986410141,1.5,0.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.02350000012665987,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.001571,0.00178,-0.014417,-0.000302,0.102262,-1.202155,-0.077166,-1.067556,-0.64691,0.011844,0.006045,-0.003491,0.794728,0.869136,-0.111734,-0.041905,-1.254616]],[[1,4,4,69,-2,-2,-2,54,9,-2,-2,-2,46,7,3,-2,-2,-2,-2],[0.5,0.05849999934434891,0.05049999989569187,2.5,-2.0,-2.0,-2.0,0.5,0.945499986410141,-2.0,-2.0,-2.0,1.5,0.12050000205636024,0.036000000312924385,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,17,16,-1,-1,-1,-1],[-0.002111,-0.008275,0.010712,0.003034,-0.142904,0.726166,1.130689,-0.023274,-0.029265,-0.695889,-0.104003,0.731519,0.024404,0.018971,0.025446,0.937978,1.109539,-0.227342,1.272323]],[[5,7,5,3,-2,-2,3,-2,-2,42,2,-2,-2,6,-2,-2,42,-2,3,-2,12,-2,-2],[0.9895000159740448,0.9535000026226044,0.6775000095367432,0.6809999942779541,-2.0,-2.0,0.05949999950826168,-2.0,-2.0,0.19090909510850906,0.9950000047683716,-2.0,-2.0,0.0820000022649765,-2.0,-2.0,0.04545454680919647,-2.0,0.9644999802112579,-2.0,0.7964999973773956,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,22,-1,-1],[0.001968,0.006394,0.001033,0.008104,0.226754,-0.275414,-0.031765,-1.08672,-0.046224,0.069912,0.14147,1.659022,-0.202179,-0.033369,-0.106392,-0.944457,-0.039129,0.279026,-0.061775,-1.264796,-0.020628,-1.172517,0.385417]],[[35,42,12,2,-2,-2,12,-2,-2,-2,-2],[0.5,0.7321428656578064,0.2150000035762787,0.9850000143051147,-2.0,-2.0,0.3894999921321869,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002555,0.000224,0.0013,0.076659,1.446282,0.11002,-0.001352,-0.3968,0.083026,-1.538919,0.760577]],[[19,3,3,42,-2,-2,9,-2,-2,8,-2,12,-2,-2,42,55,-2,-2,-2],[0.5,0.9504999816417694,0.9205000102519989,0.02777777798473835,-2.0,-2.0,0.9869999885559082,-2.0,-2.0,0.9724999964237213,-2.0,0.8695000112056732,-2.0,-2.0,0.3166666775941849,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,-1],[0.003066,0.007526,0.012828,0.009072,0.4639,-0.004306,0.122767,0.052902,1.413262,-0.029132,-1.527457,-0.013964,-0.008108,-1.265439,-0.03895,-0.052477,-1.080135,-1.070547,-0.339991]],[[42,2,69,7,-2,-2,7,-2,-2,7,-2,3,-2,-2,42,-2,66,12,-2,-2,-2],[0.02777777798473835,0.9850000143051147,0.5,0.11100000143051147,-2.0,-2.0,0.17549999803304672,-2.0,-2.0,0.5690000057220459,-2.0,0.9205000102519989,-2.0,-2.0,0.08712121471762657,-2.0,0.5,0.2604999989271164,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,20,19,-1,-1,-1],[0.0,0.01879,0.035297,0.008416,0.488372,-0.516945,0.094731,0.598205,1.240898,-0.017775,0.293148,-0.043668,-1.146499,-0.117985,-0.008072,-0.997152,-0.004558,-0.001124,0.588114,-0.081956,-0.979908]],[[3,4,4,7,-2,-2,-2,8,37,-2,-2,-2,3,-2,3,5,-2,-2,12,-2,-2],[0.9504999816417694,0.007500000298023224,0.004500000039115548,0.5219999998807907,-2.0,-2.0,-2.0,0.992499977350235,4.5,-2.0,-2.0,-2.0,0.9665000140666962,-2.0,0.9805000126361847,0.31450000405311584,-2.0,-2.0,0.9124999940395355,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,-1,15,16,-1,-1,19,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,14,-1,18,17,-1,-1,20,-1,-1],[0.002677,0.008247,0.061616,0.013012,0.756375,-0.650136,2.351721,0.003771,0.007238,0.137278,-0.642964,-0.955274,-0.03206,-1.434021,-0.012168,0.050289,0.826716,-0.188205,-0.031751,-0.563979,0.865242]],[[32,43,1,4,-2,-2,45,-2,-2,9,-2,-2,10,-2,6,-2,-2],[0.5,0.5,0.5,0.007500000298023224,-2.0,-2.0,0.29649999737739563,-2.0,-2.0,0.9524999856948853,-2.0,-2.0,0.2916666716337204,-2.0,0.544500008225441,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,15,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,16,-1,-1],[-0.004469,-0.008802,-0.012444,-0.019451,0.182762,-0.32861,0.012879,1.043945,-0.252291,0.040499,-0.201515,0.748623,0.038476,-0.042312,0.067879,0.713078,1.311151]],[[39,17,40,11,-2,-2,12,-2,-2,37,-2,-2,5,-2,-2],[0.5,0.5,0.5,0.36666667461395264,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.5,-2.0,-2.0,0.02350000012665987,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[-0.004676,-0.001912,0.000648,-0.002491,-0.128716,0.438059,0.057768,1.599456,0.173747,-0.061756,-1.323852,-0.209062,-0.083306,0.319821,-1.201272]],[[42,16,7,5,-2,-2,2,-2,-2,-2,5,-2,9,6,-2,-2,8,-2,-2],[0.42217741906642914,0.5,0.06650000065565109,0.5489999949932098,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,-2.0,0.004500000039115548,-2.0,0.9634999930858612,0.9245000183582306,-2.0,-2.0,0.20600000023841858,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,13,14,-1,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,16,15,-1,-1,18,-1,-1],[0.006211,0.010748,0.013812,0.002364,0.130126,-0.869632,0.030488,0.581408,-0.053179,-0.529811,-0.030076,1.419882,-0.044806,-0.010774,-0.624151,0.468778,-0.065948,-1.313696,-0.917557]],[[22,42,48,37,-2,-2,42,-2,-2,42,-2,-2,42,-2,-2],[0.5,0.5857143104076385,0.5,4.5,-2.0,-2.0,0.21111111342906952,-2.0,-2.0,0.7321428656578064,-2.0,-2.0,0.30384616553783417,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.005006,0.008487,0.010572,0.007156,0.13249,-0.631433,0.053842,0.129717,0.902865,-0.08272,-1.095802,-1.577741,-0.042383,-1.162042,-0.297776]],[[54,4,4,42,-2,-2,42,-2,-2,4,6,-2,-2,63,-2,-2,-2],[0.5,0.007500000298023224,0.004500000039115548,0.1547619104385376,-2.0,-2.0,0.19090909510850906,-2.0,-2.0,0.012500000186264515,0.21299999952316284,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.000496,-0.002514,0.027213,-0.000101,-0.554376,0.541526,0.117757,1.383114,-0.156576,-0.008595,-0.069191,-0.307723,-1.091213,-0.00423,-0.109209,0.457971,0.782415]],[[11,16,44,2,-2,-2,3,-2,-2,-2,7,8,9,-2,-2,2,-2,-2,12,9,-2,-2,-2],[0.4166666716337204,0.5,0.5,0.9950000047683716,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,-2.0,0.005499999970197678,0.022499999962747097,0.7449999749660492,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.898499995470047,0.6545000076293945,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.002038,-0.005626,-0.000885,0.013489,0.349371,-0.510176,-0.020491,-0.864875,-0.023571,-1.362557,0.023899,0.043195,0.010854,-0.202117,1.101922,0.088157,1.124722,1.16525,0.000201,-0.008675,-0.817667,0.239747,1.270406]],[[17,56,64,6,-2,-2,-2,5,-2,-2,42,-2,-2],[0.5,0.5,0.5,0.9945000112056732,-2.0,-2.0,-2.0,0.24949999898672104,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,12,-1,-1],[-0.0005,0.002615,0.005499,0.002472,-0.046745,0.52735,0.793,-0.048195,-0.062326,-1.106634,-0.064305,-1.166388,0.056083]],[[48,65,37,6,-2,-2,-2,10,-2,-2,9,-2,-2],[0.5,0.5,4.5,0.9874999821186066,-2.0,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.921999990940094,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,12,-1,-1],[-0.000572,-0.004191,-0.008105,-0.005212,-0.184459,0.317992,-0.966639,0.043781,-0.033305,1.213626,0.04562,-0.405264,0.994824]],[[64,5,4,6,-2,-2,45,-2,-2,6,42,-2,-2,12,-2,-2,-2],[0.5,0.9835000038146973,0.16849999874830246,0.9905000030994415,-2.0,-2.0,0.4455000013113022,-2.0,-2.0,0.26900000870227814,0.21111111342906952,-2.0,-2.0,0.5604999959468842,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.001334,-0.00381,0.001202,0.016823,0.101814,0.711554,-0.011161,-0.102181,-1.373763,-0.043368,-0.003123,0.903384,-0.99311,-0.065613,-0.636724,-1.263405,0.817889]],[[4,69,4,3,-2,-2,-2,-2,5,4,8,-2,-2,3,-2,-2,9,9,-2,-2,-2],[0.0345000009983778,2.5,0.02850000001490116,0.992499977350235,-2.0,-2.0,-2.0,-2.0,0.02850000001490116,0.3879999965429306,0.05199999921023846,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.9975000023841858,0.21649999916553497,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.000211,0.019398,0.007839,0.00217,-0.119261,0.620639,0.874708,0.783166,-0.008579,0.01638,-0.02902,0.761522,-0.840433,0.036906,0.311926,1.016882,-0.018231,-0.013991,-1.005638,-0.108114,-1.165333]],[[65,5,44,6,-2,-2,6,-2,-2,3,-2,3,-2,-2,12,-2,-2],[0.5,0.9895000159740448,0.5,0.9784999787807465,-2.0,-2.0,0.014500000048428774,-2.0,-2.0,0.278999999165535,-2.0,0.9914999902248383,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,-1],[-0.002328,-0.00612,-0.002144,0.011535,-0.070318,0.865848,-0.018649,1.372303,-0.269041,-0.047353,-1.216681,-0.02839,-0.192689,-1.051062,0.046822,0.857656,0.330667]],[[21,12,6,2,-2,-2,3,-2,-2,11,4,-2,-2,39,-2,-2,9,-2,-2],[0.5,0.2580000013113022,0.792000025510788,0.9850000143051147,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0,0.36666667461395264,0.351500004529953,-2.0,-2.0,0.5,-2.0,-2.0,0.973499983549118,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[3e-05,-0.002715,0.042288,0.094305,1.634984,-0.507253,-0.004775,0.775576,-0.843447,-0.006331,-0.01406,-0.008598,-0.516641,0.017056,0.470027,-0.834316,0.059082,0.813806,0.444031]],[[12,2,-2,-2,67,17,20,-2,-2,4,-2,-2,-2],[0.17549999803304672,0.9850000143051147,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.7450000047683716,-2.0,-2.0,-2.0],[1,2,-1,-1,5,6,7,-1,-1,10,-1,-1,-1],[4,3,-1,-1,12,9,8,-1,-1,11,-1,-1,-1],[0.001241,0.091639,1.642641,-0.096511,-0.000554,-0.00281,0.000105,-0.668591,0.049978,-0.065654,-1.402603,-0.032987,0.663083]],[[1,39,38,12,-2,-2,34,-2,-2,-2,2,4,42,-2,-2,6,-2,-2,-2],[0.5,0.5,0.5,0.2575000077486038,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.4749999940395355,0.5245000123977661,0.21111111342906952,-2.0,-2.0,0.8664999902248383,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,-1],[-0.002316,-0.008364,-0.005317,-0.011137,0.420767,-0.179871,0.028347,0.687832,-0.727355,-0.818801,0.024982,0.019344,0.032832,1.057471,0.501586,0.013525,-0.492267,1.038885,1.274472]],[[42,43,7,10,-2,-2,4,-2,-2,8,-2,-2,3,45,42,-2,-2,-2,-2],[0.42396312952041626,0.5,0.048499999567866325,0.20714285969734192,-2.0,-2.0,0.8064999878406525,-2.0,-2.0,0.3725000023841858,-2.0,-2.0,0.9895000159740448,0.0020000000367872417,0.7321428656578064,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,-1],[0.002723,0.007183,0.003034,-0.00978,-0.344238,0.173823,0.019469,0.399595,-0.731009,0.064526,0.069521,1.014047,-0.03537,-0.026482,-0.035803,-0.562293,-1.524091,1.297431,-1.498911]],[[2,19,69,6,-2,-2,8,-2,-2,12,-2,-2,1,9,-2,9,-2,-2,-2],[0.9600000083446503,0.5,2.5,0.9205000102519989,-2.0,-2.0,0.773499995470047,-2.0,-2.0,0.5474999845027924,-2.0,-2.0,0.5,0.19049999862909317,-2.0,0.9824999868869781,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,16,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,15,-1,17,-1,-1,-1],[-0.004182,0.007384,0.012883,0.006896,-0.186888,0.448304,0.071573,-0.007959,0.963774,-0.053909,-0.660075,-1.163327,-0.019268,-0.02214,0.692929,-0.02534,-0.753074,-0.093448,1.268745]],[[9,9,9,6,-2,-2,-2,2,12,-2,-2,-2,42,42,42,-2,-2,7,-2,-2,5,42,-2,-2,1,-2,-2],[0.38199999928474426,0.057499999180436134,0.030499999411404133,0.9964999854564667,-2.0,-2.0,-2.0,0.9600000083446503,0.7360000014305115,-2.0,-2.0,-2.0,0.33908046782016754,0.2952069789171219,0.22474747896194458,-2.0,-2.0,0.8379999995231628,-2.0,-2.0,0.006500000134110451,0.44155845046043396,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.002436,-0.023113,0.017634,0.009572,0.967956,-0.70877,1.10597,-0.058342,-0.088717,-1.185078,0.515149,-0.107265,0.00726,0.014374,0.00842,0.191218,-0.470216,0.070117,0.669722,1.92712,-0.024311,0.045977,0.774709,1.083111,-0.033937,-0.741157,0.870701]],[[71,19,42,2,-2,-2,42,-2,-2,8,2,-2,-2,8,-2,-2,-2],[0.5,0.5,0.5444444715976715,0.9750000238418579,-2.0,-2.0,0.7321428656578064,-2.0,-2.0,0.7374999821186066,0.9600000083446503,-2.0,-2.0,0.9370000064373016,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.003551,0.004843,0.008142,0.009911,0.226091,-0.049578,-0.064361,-0.901469,-1.59854,-0.024775,-0.008849,0.400586,-1.107925,-0.043408,-1.143174,-1.098849,-0.641985]],[[64,9,5,12,-2,-2,4,-2,-2,6,6,-2,-2,5,-2,-2,-2],[0.5,0.9494999945163727,0.8334999978542328,0.8875000178813934,-2.0,-2.0,0.17899999767541885,-2.0,-2.0,0.24050000309944153,0.029500000178813934,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.000316,-0.00268,-0.01679,-0.007035,-0.227131,0.931008,-0.084086,-0.771055,-1.189329,0.006076,-0.035581,0.314416,-1.181657,0.01575,0.31304,-0.548424,0.721354]],[[42,2,7,9,-2,-2,69,-2,-2,3,-2,9,-2,-2,6,44,5,-2,-2,-2,12,4,-2,-2,42,-2,-2],[0.02777777798473835,0.9850000143051147,0.07000000029802322,0.9964999854564667,-2.0,-2.0,0.5,-2.0,-2.0,0.07850000262260437,-2.0,0.8644999861717224,-2.0,-2.0,0.02850000001490116,0.5,0.9600000083446503,-2.0,-2.0,-2.0,0.2604999989271164,0.13000000268220901,-2.0,-2.0,0.10435571894049644,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,26,-1,-1],[0.001372,0.020995,0.039792,0.01542,0.423305,-1.12438,0.09385,0.497486,1.298114,-0.018766,-1.13194,0.001273,0.979003,-0.171238,-0.00687,0.040148,0.007288,0.872897,-0.766796,1.346971,-0.011158,0.036435,1.130554,-0.222343,-0.01575,-0.775612,-0.161688]],[[54,17,16,19,-2,-2,9,-2,-2,9,-2,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.000898,-0.001385,0.001177,0.004166,0.112408,-0.970429,-0.042587,-1.258218,0.140913,-0.05704,-1.232285,-0.3477,0.691283]],[[46,63,6,6,-2,-2,6,-2,-2,-2,69,1,-2,-2,-2],[2.5,0.5,0.4584999978542328,0.030499999411404133,-2.0,-2.0,0.5685000121593475,-2.0,-2.0,-2.0,1.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,14,13,-1,-1,-1],[0.000701,0.00286,-0.000172,-0.018923,0.409125,-0.527879,0.008209,1.025021,0.051311,0.528897,-0.063828,0.030692,-0.167412,1.159049,-1.42256]],[[71,54,19,8,-2,-2,8,-2,-2,-2,-2],[0.5,0.5,0.5,0.1314999982714653,-2.0,-2.0,0.12549999728798866,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.002396,-0.000605,-0.00268,0.001029,0.450003,-0.079674,-0.034984,-1.123224,-0.981855,0.623767,-0.977552]],[[63,8,3,9,-2,-2,-2,7,1,-2,-2,7,-2,-2,-2],[0.5,0.6324999928474426,0.3620000034570694,0.9835000038146973,-2.0,-2.0,-2.0,0.23499999940395355,0.5,-2.0,-2.0,0.7005000114440918,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[0.003447,0.000587,0.011088,0.00685,-0.054186,0.57773,0.998758,-0.011061,-0.037984,-0.549965,1.072134,0.014895,1.272894,-0.024718,0.546132]],[[5,19,33,56,-2,-2,-2,69,37,-2,-2,-2,20,-2,8,9,-2,-2,3,-2,-2],[0.9065000116825104,0.5,0.5,0.5,-2.0,-2.0,-2.0,1.5,0.5,-2.0,-2.0,-2.0,0.5,-2.0,0.9814999997615814,0.9825000166893005,-2.0,-2.0,0.9690000116825104,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,-1,15,16,-1,-1,19,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,14,-1,18,17,-1,-1,20,-1,-1],[0.001666,0.008084,0.012824,0.015146,0.249024,-0.505297,-0.749821,-0.028296,-0.037104,-0.62727,-1.147502,0.097517,-0.024783,-1.208994,-0.013578,-0.043713,-0.375359,-0.987522,0.010511,0.637905,-0.884289]],[[21,17,64,2,-2,-2,-2,4,-2,-2,-2],[0.5,0.5,0.5,0.9350000023841858,-2.0,-2.0,-2.0,0.6245000064373016,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,-1],[-0.003398,-0.005781,-0.003004,-0.005436,0.245821,-0.159568,0.78089,-0.057711,-1.328043,-0.013264,0.773611]],[[2,46,7,3,-2,-2,9,-2,-2,-2,12,-2,5,7,-2,-2,7,-2,-2],[0.9600000083446503,1.5,0.18699999898672104,0.7045000195503235,-2.0,-2.0,0.9370000064373016,-2.0,-2.0,-2.0,0.1835000067949295,-2.0,0.011500000022351742,0.0025000000605359674,-2.0,-2.0,0.9354999959468842,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,13,14,-1,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,16,15,-1,-1,18,-1,-1],[-0.000168,0.009106,0.012683,0.003244,0.198583,-0.473735,0.054225,-0.14707,0.797551,-0.761721,-0.012493,1.166387,-0.01599,0.041274,-0.320237,1.276634,-0.020499,-0.476539,0.077422]],[[16,12,44,42,-2,-2,12,-2,-2,69,3,-2,-2,-2,4,-2,-2],[0.5,0.5760000050067902,0.5,0.612500011920929,-2.0,-2.0,0.5384999811649323,-2.0,-2.0,2.5,0.9895000159740448,-2.0,-2.0,-2.0,0.863999992609024,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[0.000825,0.004021,-0.007078,0.008278,0.212015,-1.511974,-0.025148,-0.151514,-0.989679,0.018369,0.00952,0.045614,0.714929,0.939703,-0.052372,-1.304464,-0.081838]],[[35,55,42,6,-2,-2,10,-2,-2,2,-2,-2,-2],[0.5,0.5,0.5857143104076385,0.02850000001490116,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.003685,0.001731,0.004369,0.005869,0.57032,0.036052,-0.060665,-1.019697,-1.3386,-0.032765,-0.940381,-1.032682,0.861427]],[[42,69,12,-2,6,-2,-2,-2,42,-2,4,4,-2,-2,69,-2,-2],[0.02777777798473835,1.5,0.40549999475479126,-2.0,0.5575000047683716,-2.0,-2.0,-2.0,0.08712121471762657,-2.0,0.05849999934434891,0.05049999989569187,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,5,-1,-1,-1,9,-1,11,12,-1,-1,15,-1,-1],[8,7,4,-1,6,-1,-1,-1,10,-1,14,13,-1,-1,16,-1,-1],[-0.000219,0.015663,0.004121,-0.806595,0.012436,-0.457248,0.552729,0.647563,-0.006825,-0.771252,-0.004221,0.014265,0.049633,1.10311,-0.015625,-0.072754,-0.520307]],[[16,5,42,69,-2,-2,2,-2,-2,7,6,-2,-2,-2,4,-2,-2],[0.5,0.992499977350235,0.06458333507180214,1.5,-2.0,-2.0,0.9350000023841858,-2.0,-2.0,0.9964999854564667,0.5050000101327896,-2.0,-2.0,-2.0,0.863999992609024,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.001684,0.001012,0.004226,0.020736,0.127004,0.748297,-0.003073,0.424012,-0.16519,-0.041014,-0.028292,-1.012179,-0.175898,-1.105506,-0.047891,-1.296614,-0.038623]],[[17,6,70,3,-2,-2,42,-2,-2,12,10,-2,-2,6,-2,-2,4,-2,-2],[0.5,0.8840000033378601,0.5,0.34700000286102295,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.31849999725818634,0.2678571492433548,-2.0,-2.0,0.9274999797344208,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.004339,0.007151,-0.005479,-0.016943,0.043079,-0.519491,0.053364,1.184207,-0.879292,0.018982,-0.029868,-0.974769,0.355661,0.028211,1.09443,0.310845,-0.052568,-1.324326,0.212901]],[[30,54,4,4,-2,-2,8,-2,-2,-2,8,-2,-2],[0.5,0.5,0.00849999999627471,0.004500000039115548,-2.0,-2.0,0.992499977350235,-2.0,-2.0,-2.0,0.7730000019073486,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,-1],[0.002556,-0.000397,-0.002675,0.023545,0.010643,0.767821,-0.00811,-0.067004,-1.020719,0.654101,0.037427,-0.71066,0.927369]],[[6,6,3,53,-2,-2,7,-2,-2,42,-2,8,-2,-2,35,9,9,-2,-2,4,-2,-2,-2],[0.24400000274181366,0.08650000020861626,0.68299999833107,0.5,-2.0,-2.0,0.04800000041723251,-2.0,-2.0,0.1339285746216774,-2.0,0.09200000017881393,-2.0,-2.0,0.5,0.3799999952316284,0.20250000059604645,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.000178,-0.020549,-0.000923,0.022988,1.161222,0.00682,-0.031559,-0.970299,-0.324113,-0.068763,-1.55518,-0.02685,0.450496,-0.844513,0.005793,0.00302,-0.026308,0.09372,-0.864912,0.008163,-0.856353,0.145802,0.953359]],[[21,63,8,3,-2,-2,5,-2,-2,8,-2,-2,-2],[0.5,0.5,0.5194999873638153,0.25850000977516174,-2.0,-2.0,0.013500000350177288,-2.0,-2.0,0.8290000259876251,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,-1],[0.002783,0.000584,-0.002005,0.010556,0.111601,1.08357,-0.012614,-0.849193,-0.083627,0.042684,-0.488467,0.908569,0.609886]],[[3,6,2,12,-2,-2,-2,2,1,-2,-2,12,-2,-2,12,5,5,-2,-2,8,-2,-2,69,3,-2,-2,9,-2,-2],[0.001500000071246177,0.9325000047683716,0.9600000083446503,0.7129999995231628,-2.0,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.39800000190734863,-2.0,-2.0,0.8680000007152557,0.9830000102519989,0.953000009059906,-2.0,-2.0,0.9799999892711639,-2.0,-2.0,0.5,0.9845000207424164,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.001952,0.022856,0.053718,0.020776,0.930893,-0.644531,1.338744,0.002713,0.03319,1.133173,1.021263,-0.040324,-0.071707,-1.079088,-0.002435,-0.006563,-0.001032,-0.073422,0.805297,-0.043116,-1.238231,-0.621825,0.026973,-0.005983,-0.454567,0.672744,0.080403,0.822151,1.061529]],[[65,56,54,4,-2,-2,-2,7,-2,-2,12,-2,-2],[0.5,0.5,0.5,0.05650000087916851,-2.0,-2.0,-2.0,0.009500000160187483,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,12,-1,-1],[-0.001435,-0.005477,-0.002284,-0.004793,0.173322,-0.246475,0.679224,-0.064619,-0.367457,-1.265178,0.053543,1.047131,0.444141]],[[42,43,65,44,-2,-2,5,-2,-2,7,-2,-2,10,-2,-2],[0.5857143104076385,0.5,0.5,0.5,-2.0,-2.0,0.1419999971985817,-2.0,-2.0,0.031000000424683094,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,14,-1,-1],[0.001994,0.00362,0.000653,-0.002706,0.153098,-0.22609,0.04043,0.908033,0.218124,0.041087,0.860308,-0.023058,-0.061028,-1.192552,-1.475361]],[[3,44,2,3,-2,-2,46,-2,-2,5,37,-2,-2,4,-2,-2,5,-2,-2],[0.9934999942779541,0.5,0.9950000047683716,0.8760000169277191,-2.0,-2.0,1.5,-2.0,-2.0,0.6159999966621399,2.5,-2.0,-2.0,0.1314999982714653,-2.0,-2.0,0.940500020980835,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.002431,0.000186,0.01157,0.020451,0.220363,0.843388,-0.022545,-1.040384,0.850881,-0.014212,-0.006846,-0.238926,0.573414,-0.082853,-1.051185,-1.137439,-0.066367,-1.15878,-0.830136]],[[44,2,42,46,-2,-2,3,-2,-2,46,8,-2,-2,7,-2,-2,6,-2,6,10,-2,-2,6,-2,-2],[0.5,0.9950000047683716,0.3452381044626236,1.5,-2.0,-2.0,0.027000000700354576,-2.0,-2.0,1.5,0.99549999833107,-2.0,-2.0,0.7515000104904175,-2.0,-2.0,0.009499999694526196,-2.0,0.24050000309944153,0.4166666716337204,-2.0,-2.0,0.38099999725818634,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,24,-1,-1],[-0.000625,0.009776,0.017826,0.025554,0.545638,-0.773946,-0.019474,0.521574,-0.812503,-0.018391,-0.032309,-1.039966,0.56244,0.03859,1.115821,0.267972,-0.014645,1.192926,-0.019369,-0.063051,-1.038066,0.128625,-0.012985,0.628255,-0.210882]],[[16,2,46,7,-2,-2,-2,8,3,-2,-2,7,-2,-2,9,-2,-2],[0.5,0.9750000238418579,1.5,0.9354999959468842,-2.0,-2.0,-2.0,0.6029999852180481,0.19049999862909317,-2.0,-2.0,0.005499999970197678,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.001294,0.000942,0.008296,0.013307,0.112141,1.014554,-0.779988,-0.011789,0.014526,-0.081147,1.272915,-0.02675,-1.094455,-0.295741,-0.037282,-0.97388,-0.035987]],[[37,72,42,69,-2,-2,14,-2,-2,-2,-2],[4.5,0.5,0.06458333507180214,2.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[0.002753,0.004905,0.00281,0.019563,0.18515,0.845397,-0.004786,-0.030359,-0.622902,0.872578,-0.631769]],[[42,42,70,5,-2,-2,5,-2,-2,12,-2,2,-2,-2,5,3,-2,-2,45,53,-2,-2,-2],[0.33908046782016754,0.2952069789171219,0.5,0.9535000026226044,-2.0,-2.0,0.21450000256299973,-2.0,-2.0,0.29749999940395355,-2.0,0.9600000083446503,-2.0,-2.0,0.006500000134110451,0.001500000071246177,-2.0,-2.0,0.0005000000237487257,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,19,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,22,21,-1,-1,-1],[0.002374,0.006685,0.002069,-0.004511,-0.1615,0.3724,0.047296,0.888949,0.058103,0.048603,1.70014,0.029382,0.845795,-0.613107,-0.018253,0.045265,0.724581,1.351129,-0.029537,-0.034872,-0.944144,-0.646213,1.14745]],[[19,42,46,9,-2,-2,53,-2,-2,5,12,-2,-2,5,-2,-2,4,6,-2,-2,-2],[0.5,0.33908046782016754,2.5,0.9475000202655792,-2.0,-2.0,0.5,-2.0,-2.0,0.019499999471008778,0.628000020980835,-2.0,-2.0,0.05250000022351742,-2.0,-2.0,0.9679999947547913,0.8925000131130219,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.005116,0.008527,0.013276,0.016061,-0.050234,0.330432,-0.071695,0.775701,-1.134253,-0.015475,0.03373,1.325832,-0.704255,-0.032357,-1.00966,-0.487467,-0.025193,-0.033144,-0.513388,-1.113135,0.252533]]]
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
