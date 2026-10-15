"""El Guardián — bot de joseca (versión 5, con lo mejor de las pruebas v5 y v6).

Novedades respecto a la versión 4:
  - Verificador de cuentas ampliado: diferencias con verbo («superó en 35
    millones al del viñedo», «superó en 17 puntos a...»), proporciones
    («triplica», «más de un tercio»), «X más al día» no es una comparación,
    fecha + plazo = fecha, «más del 70 % del total» con el total de la frase,
    y «1.900 personas» ya no se toma por un año.
  - Regla fija: si el verificador rehace una cuenta de la frase y no cuadra,
    se da la alarma.
  - Características B1 + B2 + B4 de Candela (matices por cláusula, paráfrasis
    de matices, sujeto singular -> colectivo), en su propia sección.
  - Respuestas de varias oraciones (el giro tiene 3-4 por respuesta): media
    entre la puntuación de la respuesta entera y la de su peor oración.
  - Validación honesta y prueba de estrés al estilo de Fernando.
  - Pensado para el GIRO (respuestas de 3-4 oraciones, textos técnicos):
      * respuestas de varias oraciones: manda la PEOR oración, con un corte
        propio (UMBRAL_LARGAS) elegido con un «pseudo-giro» etiquetado;
      * tiempos verbales: «plan -> seguro» (la fuente «prevé», «estudia»,
        «recomienda» y la respuesta afirma «repetirá», «limitará»);
      * cambios de unidad («24 mm» = «2,4 centímetros», MWh/GWh, kg/t...);
      * comparaciones: se compara la cifra justo anterior con una cifra (o
        suma de cifras de una misma fuente) que la frase no cita.
  Se probaron y descartaron: equivalencias de cautela propias, «sujeto
  generalizado» propio, «todo cuadra» y «todas las cifras explicadas».

--- Notas de la versión 5 ---

v5: más cuentas en el verificador (diferencias con verbo: «superó en 35
millones al del viñedo», «redujo en 70»; proporciones: «triplica»; «X más al
día» ya no se toma por comparación; fecha + plazo = fecha) y una regla fija:
si el verificador rehace una cuenta de la frase y no cuadra, se da la alarma.
Se probaron y descartaron (por la validación): equivalencias de cautela,
«sujeto generalizado» y «todo cuadra».
Validación por familias: H +0,55 -> +0,60 (AUROC 0,941 -> 0,946).

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
# v5: unidades y su factor a la unidad base de cada magnitud (longitud en mm,
# masa en g, volumen en ml, energía en Wh, potencia en W, tiempo en minutos)
FACTOR_UNIDAD = {
    "mm": 1, "milimetro": 1, "milimetros": 1, "cm": 10, "centimetro": 10, "centimetros": 10,
    "metro": 1e3, "metros": 1e3, "km": 1e6, "kilometro": 1e6, "kilometros": 1e6,
    "g": 1, "gramo": 1, "gramos": 1, "kg": 1e3, "kilo": 1e3, "kilos": 1e3, "kilogramo": 1e3,
    "kilogramos": 1e3, "tonelada": 1e6, "toneladas": 1e6,
    "ml": 1, "mililitro": 1, "mililitros": 1, "litro": 1e3, "litros": 1e3, "hectolitros": 1e5,
    "wh": 1, "kwh": 1e3, "mwh": 1e6, "gwh": 1e9, "w": 1, "kw": 1e3, "mw": 1e6, "gw": 1e9,
    "minuto": 1, "minutos": 1, "hora": 60, "horas": 60, "dia": 1440, "dias": 1440,
}
RE_UNIDAD = re.compile(r"\s*(" + "|".join(sorted(FACTOR_UNIDAD, key=len, reverse=True)) + r")\b")
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
        # v5: cambio de unidad sin perder precisión (lo permite el criterio:
        # «2.000 m = 2 km»). Añadimos el valor en la unidad base de su magnitud,
        # así «2,4 centímetros» y «24 mm» comparten el valor 24 (mm).
        mu = RE_UNIDAD.match(resto)
        if mu:
            valores.add(v * FACTOR_UNIDAD[mu.group(1)])
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
                            # v5: en una comparación («3 menos que en 2025») el dato
                            # solo es «copiado» si la fuente también compara junto
                            # a esa cifra; un «3» suelto en otra frase no cuenta
                            if comparacion_en_curso[0]:
                                cerca_n = fuentes_norm[i][max(0, n["ini"] - 40):n["fin"] + 40]
                                if not re.search(r"\b(mas|menos|frente|respecto|superior|inferior)\b", cerca_n):
                                    continue
                            return True
            return False

        comparacion_en_curso = [False]

        def anotar(k, ok, tipo):
            comparacion_en_curso[0] = tipo in ("comparacion", "diferencia")
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
            if m.group(1) == "en total":
                # v5: en «7 en total» el total va DELANTE de la marca
                tras_marca = [k for k in utiles if 0 <= pos_marca - nums_resp[k]["fin"] <= 3][-1:]
            if tras_marca and tras_marca[0] not in resultado:
                x = tras_marca[0]
                sujeto = resp_norm[ini_f:pos_marca]
                if m.group(1) == "en total":
                    sujeto = resp_norm[ini_f:nums_resp[x]["ini"]]
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
            # v5: el punto de partida es la cifra JUSTO ANTERIOR del mismo tipo
            # («2 mayores y 5 menores, 7 en total, 3 menos que en 2025»: se
            # compara el 7, no cualquier cifra de la frase, que daba aciertos
            # por casualidad como 5 - 8 = 3)
            previas = [k for k in utiles if k != c and nums_resp[k]["ini"] < nums_resp[c]["ini"]
                       and nums_resp[k]["pct"] == nums_resp[c]["pct"]]
            anclas = previas[-1:] or [k for k in utiles if k != c and nums_resp[k]["ini"] < nums_resp[c]["ini"]][-1:]
            idx = set(_fuentes_que_hablan_de(objetivo, fuentes_norm))
            idx |= {i for i, ns in enumerate(nums_por_fuente) for k in anclas
                    if any(_cerca(v, w, 0.001) for n in ns for v in n["valores"] for w in nums_resp[k]["valores"])}
            if not anclas or not idx:
                continue
            otras = _valores_no_anio([n for i in idx for n in nums_por_fuente[i]])
            # v5: la otra parte puede ser una suma («1 mayor y 8 menores» = 9),
            # pero solo de cifras de una MISMA fuente (mezclar fuentes daba
            # aciertos por casualidad)
            # La otra parte de la comparación es una cifra que la frase NO cita
            # (las que cita son el lado «actual»: «2 mayores y 5 menores, 7 en
            # total, 3 menos que en 2025» compara con 1 + 8 de 2025, no con 2 + 8)
            citadas_frase = [v for k in utiles for v in nums_resp[k]["valores"]]
            no_citadas = lambda vs: [v for v in vs if not any(_cerca(v, w, 0.001) for w in citadas_frase)]
            otras = no_citadas(otras)
            for i in idx:
                vals_i = no_citadas(sorted(set(_valores_no_anio(nums_por_fuente[i]))))[:12]
                otras += [a + b for a, b in combinations(vals_i, 2)]
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
            # v6: también «, más del 70 % del total», «, casi el 40 %»
            if not (re.search(r",\s*(lo que supone\s+|es decir,?\s+)?((algo )?(mas|menos) del?\s*|casi (el|un)\s*|"
                              r"cerca del?\s*)?(un|el)?\s*$", entre)
                    and re.match(r"\s*(%|por ciento)\s+(de|del)\b", resp_norm[nums_resp[c]["fin"]:fin_f])):
                continue
            vx = base(x) if base(x) > 1000 or not any(v > 1000 for v in nums_resp[x]["valores"]) else max(nums_resp[x]["valores"])
            # La base del porcentaje es la cifra de PARTIDA: si perdió X, la
            # base B cumple que B - X también está en las fuentes (si ganó, B + X)
            # v6: si la propia frase trae el total («atendió a 1.326 personas, de
            # las que 812 ..., más del 70 % del total»), la base es ese total
            totales_frase = [base(k) for k in utiles if k not in (x, c) and not nums_resp[k]["pct"]
                             and base(k) > vx]
            if totales_frase:
                bases = totales_frase
            elif RE_PERDIDA.search(frase):
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


# v5: formas equivalentes de conservar cada tipo de cautela. El condicional
# del verbo («quedarían», «terminaría», «sumarían») ya expresa que algo no es
# seguro, igual que «si se aprueba» o «prevé».
RE_VERBO_CONDICIONAL = r"\b\w{3,}(aria|eria|iria|arian|erian|irian)\b"
EQUIVALENTES = {
    "restrictivo": [r"\bsolo\b", r"\bunicamente\b", r"\bexclusivamente\b", r"\bse limita", r"\blimitad",
                    r"\breservad", r"\bsalvo\b", r"\bexcepto\b", r"\bsiempre que\b", r"\ba condicion\b",
                    r"\brequisito", r"\bque (hayan|tengan|sean|esten|cumplan|lo soliciten)\b",
                    r"\bningun", r"\bno (cubre|incluye|se aplica|pueden?)\b"],
    "condicional": [r"\bsi\b", r"\bsiempre que\b", r"\bcondicion", r"\bdepend", r"\ben funcion de\b",
                    r"\bprevist", r"\bpreve\b", r"\bpropuesta\b", r"\bpropone\b", r"\bpodria",
                    r"\bestudia\b", r"\bplantea\b", r"\bpendiente", r"\bprovisional", r"\bsujet",
                    r"\bestima", r"\bsegun\b", RE_VERBO_CONDICIONAL],
    "aproximado": [r"\bmas de\b", r"\bmenos de\b", r"\bal menos\b", r"\bcasi\b", r"\bcerca de\b",
                   r"\bunos\b", r"\bunas\b", r"\balrededor\b", r"\baproximad", r"\bhasta\b",
                   r"\bmaximo\b", r"\bminimo\b", r"\ben torno\b", r"\bpor encima\b", r"\bpor debajo\b"],
    "parcial": [r"\balgun", r"\bvari[oa]s\b", r"\bparte de\b", r"\bmayoria\b", r"\b\d[\d.,]* de (los|las|sus)\b",
                r"\bde ell[oa]s\b", r"\bentre\b", r"\bsolo\b"],
}


# v5: lo que restringe a un grupo: cifras, «algunos», «que lo soliciten»,
# «mayores de», «inscritos»...
RE_RESTRICTOR = re.compile(r"\d|\b(algun|vari[oa]s|parte|solo|unicamente|exclusivamente|que|cuy[oa]s?|con|sin|"
                           r"mayores|menores|inscrit|registrad|afectad|empadronad|de menos|de mas|"
                           r"siempre que|salvo|excepto|determinad|ciert[oa]s|seleccionad|adherid|"
                           r"participantes?|beneficiari)\b")


def sujetos_generalizados(oracion_norm, fuentes_norm):
    """Cuenta grupos en plural («los municipios», «las personas sordas») que la
    oración menciona SIN restricción y que en la fuente aparecen CON ella."""
    total = 0
    for m in re.finditer(r"\b(los|las)\s+([a-zñ]{4,}s)\b", oracion_norm):
        nombre = raiz(m.group(2))
        cerca_resp = oracion_norm[m.end():m.end() + 30]
        if RE_RESTRICTOR.search(cerca_resp):
            continue  # la respuesta ya lo restringe
        for fn in fuentes_norm:
            restringido = False
            for mf in re.finditer(r"\b" + re.escape(nombre) + r"[a-zñ]*\b", fn):
                contexto = fn[max(0, mf.start() - 25):mf.end() + 30]
                if RE_RESTRICTOR.search(contexto):
                    restringido = True
                    break
            if restringido:
                total += 1
                break
    return total


# Probado en v5 y DESACTIVADO: en la validación empeoraba H (-0,014 de media).
# Se deja el código por si con el giro (textos legales) conviene reactivarlo.
USAR_EQUIVALENTES = False


def conserva_cautela(tipo, resp_norm):
    """¿La respuesta mantiene alguna cautela de este tipo, dicha como sea?"""
    if not USAR_EQUIVALENTES:
        return False
    return any(re.search(p, resp_norm) for p in EQUIVALENTES.get(tipo, []))


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
# Características B de Candela (v6)
# ---------------------------------------------------------------------------
# Copiadas tal cual de entregas/candela_v2/experimento_b/caracteristicas_b.py
# (compañera de equipo): B1 matices perdidos en las cláusulas relevantes,
# B2 paráfrasis de matices y B4 sujeto singular que pasa a colectivo
# («Torralcina ganó población» -> «los municipios de la comarca ganaron»).
# No se usa B3 (cobertura), que en sus pruebas y en las nuestras no ayudaba.
# En nuestro banco de pruebas: H honesto +0,561 -> +0,595 sobre v5.

STOP = set('a al algo ante antes asi aun cada como con cual cuando de del desde donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba estan estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mientras muy ni no nos o otra otras otro otros para pero por que quien se sea segun ser si sido sin sobre son su sus tambien tan tanto te tiene tienen todo tras tu un una uno unos unas y ya yo'.split())
NAME_STOP = set('el la los las un una unos unas en de del por para con segun este esta estos estas esto eso ademas enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split())
PREDICATE = re.compile(r'\b(?:es|son|sera|seran|fue|fueron|tiene|tienen|hay|durara|duraran|cuenta|cuentan|incluye|incluyen|preve|preven|previsto|prevista|planea|planean|podria|podrian|puede|pueden|aprob\w*|aprueb\w*|realiz\w*|recib\w*|instal\w*|cost\w*|pag\w*|cerr\w*|abr\w*|impart\w*|afect\w*|estudia|estudian|limita|limitan|\w+(?:aron|ieron|aba|aban|ara|aran|eran))\b')
# Cada grupo fuerte mantiene el tipo y la fuerza del matiz. No se mezclan
# previsión, posibilidad, condición, cota superior, aproximación y mayoría.
EQUIVALENCES = {
 'restriction': ('restrictivo', r'\b(?:solo|solamente|unicamente|exclusivamente|se limita(?:n)? a)\b'),
 'exception': ('restrictivo', r'\b(?:salvo|excepto|a excepcion de)\b'),
 'approval': ('condicional', r'\b(?:si se aprueba\w*|en caso de aprobacion|sujet[oa]s? a (?:la )?aprobacion)\b'),
 'condition': ('condicional', r'\b(?:siempre que|a condicion de|si (?:el|la|los|las|lo|se))\b'),
 'forecast': ('condicional', r'\b(?:preve\w*|previst[oa]s?|planea\w*|tiene(?:n)? previsto)\b'),
 'possibility': ('condicional', r'\b(?:podria\w*|es posible|cabe la posibilidad)\b'),
 'study': ('condicional', r'\b(?:se estudia\w*|se contempla\w*)\b'),
 'proposal': ('condicional', r'\b(?:propuesta|propone\w*|proyecta\w*)\b'),
 'approximation': ('aproximado', r'\b(?:aproximadamente|alrededor de|cerca de|unos|unas)\b'),
 'upper': ('aproximado', r'\b(?:hasta|como maximo|a lo sumo)\b'),
 'lower': ('aproximado', r'\b(?:al menos|como minimo)\b'),
 'more': ('aproximado', r'\bmas de\b'),
 'less': ('aproximado', r'\bmenos de\b'),
 'almost': ('aproximado', r'\bcasi\b'),
 'some': ('parcial', r'\b(?:algun[oa]s|determinad[oa]s|parte de)\b'),
 'several': ('parcial', r'\bvari[oa]s\b'),
 'majority': ('parcial', r'\bla mayoria\b'),
 'subset': ('parcial', r'\b(?:de (?:los|las) \d|entre \d|de ell[oa]s)\b'),
}
# Relaciones débiles: nunca cuentan como preservación fuerte.
WEAK_PAIRS = {frozenset(x) for x in [('approval','proposal'),('approval','forecast'),('possibility','study'),('forecast','proposal'),('some','majority'),('some','several'),('upper','approximation')]}
COLLECTIVES = {'municipios':'municipio','empresas':'empresa','centros':'centro','organizaciones':'organizacion','hospitales':'hospital','ayuntamientos':'ayuntamiento','colegios':'colegio','locales':'local','nadadores':'nadador','ciudades':'ciudad','provincias':'provincia','pozos':'pozo','alumnos':'alumno','trabajadores':'trabajador','familias':'familia','personas':'persona','participantes':'participante'}


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFD',str(text or '').lower()) if unicodedata.category(c)!='Mn')


def roots(text):
    words=re.findall(r'[a-z]+',norm(text));out=set()
    for word in words:
        if word in STOP or len(word)<3:continue
        if word.startswith(('aprob','aprueb')):out.add('aprob')
        else:out.add(word[:5])
    return out


def entities(text):
    return {norm(w) for w in re.findall(r'\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,}\b',text or '') if norm(w) not in NAME_STOP}


def numbers(text):
    return set(re.findall(r'\b\d+(?:[.,]\d+)*\b',norm(text)))


def clauses(text):
    """Conserva decimales, listas y complementos; separa proposiciones claras."""
    chunks=re.split(r';|(?<!\d)[.!?](?!\d)|(?<=\d)[.!?](?!\d)|\bpero\b',str(text or ''),flags=re.I)
    out=[]
    for chunk in chunks:
        pieces=re.split(r'(,|\s+y\s+)',chunk)
        current=pieces[0]
        for i in range(1,len(pieces),2):
            sep=pieces[i];right=pieces[i+1]
            qualifier=re.match(r'\s*(?:solo|si\b|aunque|salvo|excepto|en caso|sujeto|sujeta)',norm(right))
            if PREDICATE.search(norm(current)) and PREDICATE.search(norm(right)) and not qualifier:
                if current.strip():out.append(current.strip())
                current=right
            else:current+=sep+right
        if current.strip():out.append(current.strip())
    return out


def relevant(a,b):
    """>=2 raíces compartidas y cobertura sustancial, sin sujeto incompatible."""
    ra,rb=roots(a),roots(b);common=ra&rb
    if len(common)<2:return False
    ea,eb=entities(a),entities(b)
    if ea and eb and not ea&eb:return False
    overlap=len(common)/max(1,min(len(ra),len(rb)))
    anchor=bool((ea&eb) or (numbers(a)&numbers(b)))
    return overlap>=.5 or (anchor and overlap>=.3)


def signals_rank(caso):
    sources=caso.get('fuentes') or [];response=caso.get('respuesta') or '';rr=roots(response);rank=[]
    signals=caso.get('senales') or {};items=(signals.get('oraciones') or [])+(signals.get('fragmentos') or [])
    for i,source in enumerate(sources):
        lexical=len(rr&roots(source))/max(1,len(rr));e=sim=0.;cited=False
        for item in items:
            for key in ('respaldo_por_fuente','similitud_por_fuente'):
                values=item.get(key) or []
                value=float(values[i] or 0.) if i<len(values) else 0.
                if key=='respaldo_por_fuente':e=max(e,value)
                else:sim=max(sim,value)
            cited=cited or (item.get('llm') or {}).get('fuente')==i
        rank.append((.4*e+.3*sim+.25*lexical+.05*cited,i))
    return [i for _,i in sorted(rank,key=lambda pair:(-pair[0],pair[1]))[:3]]


def nuance_groups(text):
    t=norm(text)
    # «Hasta el 31 de julio» expresa fin temporal, no cota aproximada de cifra.
    t=re.sub(r'\bhasta\s+(?:el\s+)?\d{1,2}\s+de\s+[a-z]+','limite temporal',t)
    found={group for group,(_,pattern) in EQUIVALENCES.items() if re.search(pattern,t)}
    if 'approval' in found:
        found.discard('condition')
        if re.search(r'\bsolo si se aprueba',t):
            without=re.sub(r'\bsolo si se aprueba\w*','',t)
            if not re.search(EQUIVALENCES['restriction'][1],without):found.discard('restriction')
    return found


def aligned_response(source_clause,response_clauses):
    """Preservación en toda la respuesta, ligada a la misma entidad/afirmación."""
    aligned=[];subject=entities(source_clause)
    for i,claim in enumerate(response_clauses):
        if relevant(source_clause,claim):aligned.append(claim)
        # Anáfora conservadora: la condición en la siguiente oración debe
        # referir a la oración alineada y no introducir otra entidad.
        elif i and re.match(r'\s*(?:esto|esta medida|dicha medida|este plan|esa actuacion)\b',norm(claim)):
            previous=response_clauses[i-1]
            if relevant(source_clause,previous) and not (entities(claim)-subject):aligned.append(claim)
        elif subject and entities(claim)&subject and not (entities(claim)-subject):
            # Una condición separada sobre el mismo sujeto solo se acepta si
            # comparte también contenido de la acción (no basta el nombre).
            nonentity=roots(claim)-{x[:5] for x in subject}
            if nonentity&(roots(source_clause)-{x[:5] for x in subject}):aligned.append(claim)
    return aligned


def matiz_features(caso):
    response=caso.get('respuesta') or '';sources=caso.get('fuentes') or [];rc=clauses(response)
    lost={k:0 for k in ('restrictivo','condicional','aproximado','parcial')};preserved=missing=uncertain=0;comparisons=0;seen=set()
    for i in signals_rank(caso):
        for source in clauses(sources[i]):
            if not any(relevant(source,claim) for claim in rc):continue
            aligned=aligned_response(source,rc);answer=' '.join(aligned)
            sg=nuance_groups(source);ag=nuance_groups(answer)
            # B1 conserva los patrones literales; B2 añade equivalencias fuertes.
            for category in lost:
                expressions={m.group(0) for group,(cat,pattern) in EQUIVALENCES.items() if cat==category for m in re.finditer(pattern,norm(source))}
                for expression in expressions:
                    key=(norm(source),category,expression)
                    if key in seen:continue
                    seen.add(key);comparisons+=1
                    if not re.search(r'\b'+re.escape(expression)+r'\b',norm(answer)):lost[category]+=1
            for group in sg:
                if group in ag:preserved+=1
                else:
                    missing+=1
                    if any(frozenset((group,other)) in WEAK_PAIRS for other in ag):uncertain+=1
    out={'matiz_clausula_'+k:float(min(v,2)) for k,v in lost.items()}
    out.update(matiz_parafrasis_preservado=float(min(preserved,2)),matiz_parafrasis_perdido=float(min(missing,2)),matiz_equivalencia_incierta=float(min(uncertain,2)))
    out.update({'matiz_clausula_ratio_'+k:lost[k]/max(1,comparisons) for k in lost})
    return out


def subset_scope(text,plural):
    """Subconjunto cuantificado ligado al sustantivo, no una cifra aislada."""
    t=norm(text);noun=re.escape(plural)
    patterns=[r'\b(\d+)\s+de\s+(?:(?:los|las|sus)\s+)?(\d+)\s+'+noun+r'\b',
              r'\bde\s+(?:(?:los|las|sus)\s+)?(\d+)\s+'+noun+r'\s*,?\s*(\d+)\b']
    for j,pattern in enumerate(patterns):
        for m in re.finditer(pattern,t):
            a,b=map(int,m.groups());part,total=(a,b) if j==0 else (b,a)
            if 0<part<total:return .85
    if re.search(r'\b(?:solo|unicamente|solamente|algun[oa]s|determinad[oa]s|parte de (?:los|las))\s+(?:\d+\s+)?'+noun+r'\b',t):return .7
    return 0.


def subject_features(caso):
    sources=caso.get('fuentes') or [];response=caso.get('respuesta') or '';suspicion=0.;confidence=0.
    source_clauses=[c for i in signals_rank(caso) for c in clauses(sources[i])]
    for claim in clauses(response):
        tokens=set(re.findall(r'[a-z]+',norm(claim)));collective=[p for p in COLLECTIVES if p in tokens]
        if not collective:continue
        # Una restricción explícita evita identificar el grupo como universal.
        if re.search(r'\b(?:algunos|algunas|estos|estas|esos|esas|determinados|determinadas|parte de|solo|unicamente|de los \d|de las \d)\b',norm(claim)):continue
        if re.search(r'\b\d+\s+de\s+(?:(?:los|las)\s+)?\d+\b',norm(claim)):continue
        cp=PREDICATE.search(norm(claim))
        if cp is None:continue
        actions={m.group(0)[:5] for m in PREDICATE.finditer(norm(claim))}
        for plural in collective:
            noun=COLLECTIVES[plural];nr=noun[:5]
            action_roots=roots(claim)-{nr}
            # Revisión de cobertura: grupo limitado -> grupo completo. Exige
            # el sustantivo y al menos dos raíces de propiedad/acción comunes,
            # además de una fracción estricta o restricción explícita.
            lexical=[s for s in source_clauses if plural in re.findall(r'[a-z]+',norm(s)) and len((roots(s)&action_roots)-{'enero','febre','marzo','abril','mayo','junio','julio','agost','septi','octub','novie','dicie'})>=2]
            limited=[(s,subset_scope(s,plural)) for s in lexical if subset_scope(s,plural)>0]
            full_support=[s for s in lexical if subset_scope(s,plural)==0]
            if limited and not full_support and re.search(r'\b(?:los|las|todos los|todas las)\s+'+re.escape(plural)+r'\b',norm(claim)):
                score=max(v for _,v in limited)
                question_entities=entities(caso.get('pregunta') or '')
                anchored=any(entities(s)&question_entities for s,_ in limited)
                suspicion=1.;confidence=max(confidence,min(.95,score+.05*anchored))
                continue
            # Si alguna evidencia menciona el colectivo o al menos dos sujetos
            # de esa clase con la misma acción, nos abstenemos.
            candidates=[s for s in source_clauses if len(action_roots&roots(s))>=1 and any(a in {m.group(0)[:5] for m in PREDICATE.finditer(norm(s))} for a in actions)]
            if any(plural in re.findall(r'[a-z]+',norm(s)) for s in candidates):continue
            relevant_sources=[s for s in candidates if re.search(r'\b'+re.escape(noun)+r'\b',norm(s)) or entities(s)]
            names=set().union(*(entities(s) for s in relevant_sources)) if relevant_sources else set()
            claim_names=entities(claim)
            if len(names)>1 or len(claim_names)>1:continue
            for source in relevant_sources:
                content=roots(source)&action_roots
                if len(content)<2 or not PREDICATE.search(norm(source)):continue
                # Colectivo y predicado deben pertenecer a la misma cláusula.
                if norm(source).find(plural)>=0:continue
                explicit_singular=bool(re.search(r'\b'+re.escape(noun)+r'\b',norm(source)))
                if not explicit_singular and len(entities(source))!=1:continue
                question=caso.get('pregunta') or ''
                anchor=bool(entities(source)&entities(question) or re.search(r'\b'+re.escape(noun)+r'\b',norm(question)))
                suspicion=1.;confidence=max(confidence,min(.95,(.85 if explicit_singular else .6)+.05*anchor))
    return dict(sujeto_generalizado=suspicion,sujeto_generalizado_confianza=confidence)


# ---------------------------------------------------------------------------
# Tiempos verbales (v5)
# ---------------------------------------------------------------------------
# Las fuentes técnicas y legales distinguen lo HECHO de lo PREVISTO: «se
# clausurarán 18 plazas», «recomienda repararlos», «prevé repetir la campaña».
# Una respuesta que lo cuenta como ya hecho («se clausuraron», «se repararon»)
# o como seguro («repetirá») está endureciendo la afirmación.
# Se mira el texto CON tildes: sin ellas «clausurarán» (futuro) y «clausuraran»
# (subjuntivo) se confunden.
RE_FUTURO = re.compile(r"\b\w{3,}(?:ará|erá|irá|arán|erán|irán)\b|\b(?:va|van|iba|iban) a \w+(?:ar|er|ir)\b")
RE_CONDICIONAL = re.compile(r"\b\w{3,}(?:aría|ería|iría|arían|erían|irían)\b")
RE_PLAN = re.compile(r"\b(?:prevé|prevén|previst[oa]s?|propone|proponen|propuso|propuesta|estudia|estudian|"
                     r"planea|planean|pretende|pretenden|recomienda|recomiendan|aconseja|aconsejan|"
                     r"plantea|plantean|se espera|espera|esperan|baraja|contempla|tiene previsto|tienen previsto|"
                     r"podría|podrían|puede|pueden|si se)\b", re.I)
RE_HECHO = re.compile(r"\b\w{3,}(?:ó|aron|ieron|ió)\b|\b(?:ha|han|había|habían) \w+(?:ado|ido|ada|ida|ados|idos|adas|idas)\b")


def modo_verbal(texto):
    """'previsto' si habla de algo futuro, condicional o planeado; 'hecho' si lo
    da por realizado; 'otro' si no se puede saber."""
    t = (texto or "").lower()
    if RE_PLAN.search(t) or RE_FUTURO.search(t) or RE_CONDICIONAL.search(t):
        return "previsto"
    if RE_HECHO.search(t):
        return "hecho"
    return "otro"


def tiempos_endurecidos(oraciones, fuentes):
    """Cuenta oraciones que dan por hecho (o por seguro) lo que su fuente solo
    prevé. Devuelve (previsto_a_hecho, plan_a_seguro)."""
    a_hecho = a_seguro = 0
    for o in oraciones:
        texto = o.get("texto") or ""
        idx = set()
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < len(fuentes):
            idx.add(i)
        sim = o.get("similitud_por_fuente") or []
        if sim:
            idx.add(max(range(len(sim)), key=lambda k: sim[k]))
        raices_o = raices_contenido(texto)
        for j in idx:
            if j >= len(fuentes):
                continue
            # Solo la frase de la fuente que habla de lo mismo que la oración
            frases = re.split(r"(?<=[.;])\s+", fuentes[j])
            frase = max(frases, key=lambda fr: len(raices_contenido(fr) & raices_o))
            if len(raices_contenido(frase) & raices_o) < 2:
                continue
            modo_f = modo_verbal(frase)
            t = texto.lower()
            if modo_f == "previsto" and RE_HECHO.search(t) and not (RE_PLAN.search(t) or RE_FUTURO.search(t)
                                                                    or RE_CONDICIONAL.search(t)):
                a_hecho += 1
            elif RE_PLAN.search(frase.lower()) and RE_FUTURO.search(t) and not (RE_PLAN.search(t)
                                                                               or RE_CONDICIONAL.search(t)):
                a_seguro += 1
    return a_hecho, a_seguro


def caracteristicas_candela(caso):
    """B1 + B2 + B4 de Candela, con prefijo cb_ para no mezclarlas con las nuestras."""
    out = matiz_features(caso)
    out.update(subject_features(caso))
    return {"cb_" + k: float(v) for k, v in out.items()}


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
    partes_ok = 0
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
        elif not n["pct"] and n["valores"] and max(n["valores"]) > 100 and any(
                _cerca(v, x, 0.002) for v in n["valores"] for x in partes_de_total):
            # v4: porcentaje de un total («5.520 pinos» = 30 % de 18.400). v6: si
            # la cifra es grande y cuadra casi exacta, cuenta como cuenta verificada
            clases.append("derivado_local")
            partes_ok += 1
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
    # v6 (idea de Fernando): cifras NUEVAS = las que no están tal cual en las
    # fuentes. Si TODAS salen de una cuenta comprobada, es señal fuerte de
    # respuesta fiel; si solo «alguna», no dice tanto.
    nuevas = [c for n, c in zip(nums_resp, clases) if not n["anio"] and c != "directo"]
    f["cifras_nuevas_todas_explicadas"] = float(bool(nuevas) and all(
        c in ("derivado_local", "derivado_anclado", "cuenta_correcta") for c in nuevas))
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
    # v6: los plazos y los «X % de un total» comprobados cuentan también como
    # cuentas verificadas (son raros: sueltos, el modelo no llega a aprenderlos)
    f["verif_cuenta_ok"] = float(min(2, f["verif_cuenta_ok"] + correctos + partes_ok))
    f["verif_cuenta_mal"] = float(min(2, f["verif_cuenta_mal"] + falsos))
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
            # v5: si la respuesta conserva ese TIPO de cautela con otras palabras
            # («quedarían» por «si se aprueba», «se limita a» por «solo»), no
            # se ha perdido nada aunque no use la misma palabra que la fuente
            if conserva_cautela(k, resp_norm):
                continue
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

    # v5: la misma idea, pero aceptando paráfrasis de la cautela y el
    # condicional del verbo («quedarían», «terminaría»)
    cautelas = [p for pats in EQUIVALENTES.values() for p in pats]
    pierde_v5 = 0.0
    generalizados = 0
    for o in oraciones:
        on = normalizar(o.get("texto") or "")
        fuentes_o = set()
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            fuentes_o.add(i)
        sim = o.get("similitud_por_fuente") or []
        if sim:
            fuentes_o.add(max(range(len(sim)), key=lambda k: sim[k]))
        fuentes_o = [j for j in fuentes_o if j < n_fuentes]
        if not any(re.search(p, on) for p in cautelas) and any(
                any(re.search(p, normalizar(fuentes[j])) for p in todos_matices) for j in fuentes_o):
            pierde_v5 = 1.0
        generalizados += sujetos_generalizados(on, [normalizar(fuentes[j]) for j in fuentes_o])
    f["matiz_oracion_perdido_v5"] = pierde_v5
    # v5: sujeto generalizado: la respuesta habla de «los municipios de la
    # comarca» o «los coches eléctricos» sin restricción, y la fuente dice
    # «21 de sus 26 municipios» o «los coches eléctricos que lo soliciten»
    f["sujeto_generalizado"] = float(min(generalizados, 2))
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
    # v5: «todo cuadra»: el LLM la respalda con confianza y ninguna comprobación
    # propia encuentra nada. Es justo donde más falsas alarmas dábamos; con esta
    # señal explícita el modelo puede aprender a fiarse del LLM en estos casos.
    sospechas = (f["num_sospechosos_grandes"] + f["num_cuenta_falsa"] + f["verif_cuenta_mal"]
                 + f["entidad_ajena"] + f["fecha_plazo_falso"] + f["anadido_universal"]
                 + f["anadido_causal"] + f["matiz_oracion_perdido_v5"] + f["sujeto_generalizado"]
                 + sum(f[k] for k in f if k.startswith("matiz_perdido_tramo")))
    f["todo_cuadra"] = float(llm_ok and f["llm_conf_min_respaldada"] >= 0.9 and sospechas == 0)
    f["todo_cuadra_con_cuenta_ok"] = f["todo_cuadra"] * float(f["verif_cuenta_ok"] > 0
                                                              or f["num_cuenta_correcta"] > 0)
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
        if nombre.startswith(("num_", "verif_", "matiz_", "anadido_", "huecos_", "nombre_", "entidad_", "sujeto_",
                              "fecha_", "hora_")):
            f["llmok_" + nombre] = llm_ok * f[nombre]

    # --- 9. Características B de Candela (v6) ---
    f.update(caracteristicas_candela(caso))

    # --- 10. Tiempos verbales: lo previsto contado como hecho o como seguro ---
    a_hecho, a_seguro = tiempos_endurecidos(oraciones, fuentes)
    f["tiempo_previsto_a_hecho"] = float(min(a_hecho, 2))
    f["tiempo_plan_a_seguro"] = float(min(a_seguro, 2))
    f["llmok_tiempo_previsto_a_hecho"] = llm_ok * f["tiempo_previsto_a_hecho"]
    f["llmok_tiempo_plan_a_seguro"] = llm_ok * f["tiempo_plan_a_seguro"]
    return f


def por_oracion(caso):
    """v6: un sub-caso por oración (su texto, sus señales y sus fragmentos),
    para puntuar cada oración como si fuera una respuesta suelta."""
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    if len(oraciones) < 2:
        return []
    fragmentos = senales.get("fragmentos") or []
    return [{"pregunta": caso.get("pregunta") or "", "fuentes": caso.get("fuentes") or [],
             "respuesta": o.get("texto") or "",
             "senales": {"oraciones": [o],
                         "fragmentos": [dict(fr, oracion=0) for fr in fragmentos if fr.get("oracion", 0) == j]}}
            for j, o in enumerate(oraciones)]


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
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas", "cb_matiz_clausula_restrictivo", "cb_matiz_clausula_condicional", "cb_matiz_clausula_aproximado", "cb_matiz_clausula_parcial", "cb_matiz_parafrasis_preservado", "cb_matiz_parafrasis_perdido", "cb_matiz_equivalencia_incierta", "cb_sujeto_generalizado", "cb_sujeto_generalizado_confianza", "tiempo_plan_a_seguro", "llmok_tiempo_plan_a_seguro"]
INICIAL = -0.035456
TASA = 0.05
ARBOLES = [[[2,3,44,-2,-2,4,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.28450000286102295,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003123,0.414307,0.470507,1.998345,1.099628,0.284888,1.604819,0.622315,-0.180073,-0.253411,-1.233059,0.91899,0.161875,1.447672,-0.873913]],[[2,8,53,-2,-2,7,-2,-2,8,4,-2,-2,21,-2,-2],[0.9350000023841858,0.17599999904632568,0.5,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.07600000128149986,0.08900000154972076,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002218,0.38431,0.464013,1.924445,1.330658,0.301767,1.844482,0.989522,-0.175862,0.204134,1.70514,0.247886,-0.239735,-1.096092,1.308118]],[[2,53,3,-2,-2,5,-2,-2,8,4,-2,-2,38,-2,-2],[0.9249999821186066,0.5,0.032500000670552254,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.0885000005364418,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005121,0.363356,0.417287,1.816243,1.40203,0.165456,1.743681,0.189671,-0.170052,0.172121,1.353577,-0.306689,-0.236102,-1.202441,0.348562]],[[2,53,3,-2,-2,5,-2,-2,8,4,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.014579,0.341147,0.380317,1.734096,1.171746,0.170367,1.690962,0.295011,-0.173934,0.139729,1.336434,-0.485364,-0.236698,-1.176535,0.213592]],[[2,53,3,-2,-2,9,-2,-2,15,38,-2,-2,54,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.011281,0.322393,0.369715,1.602773,0.913849,0.126579,-0.316135,1.144417,-0.160647,-0.200154,-1.075297,0.463863,0.262421,0.110598,1.945208]],[[2,7,3,-2,-2,44,-2,-2,8,4,-2,-2,21,-2,-2],[0.9350000023841858,0.007500000298023224,0.636000007390976,-2.0,-2.0,0.5,-2.0,-2.0,0.0885000005364418,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005164,0.321352,0.380876,1.624815,0.851449,0.224055,1.15285,-0.088735,-0.156771,0.15962,1.246934,-0.251666,-0.214385,-1.014848,1.192122]],[[53,69,39,-2,-2,-2,2,52,-2,-2,79,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9750000238418579,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.008126,0.299182,0.350129,1.526753,0.643056,-0.97606,-0.133621,-0.000195,0.292545,-1.697404,-0.273918,-1.305682,0.273715]],[[53,5,3,-2,-2,42,-2,-2,2,22,-2,-2,21,-2,-2],[0.5,0.5550000071525574,0.007500000298023224,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.013596,0.288597,0.338008,1.58565,1.196765,-0.007226,0.672819,-1.516126,-0.152122,0.002148,0.292304,-1.765071,-0.265329,-1.228284,1.235232]],[[53,69,5,-2,-2,-2,70,54,-2,-2,42,-2,-2],[0.5,0.5,0.4529999941587448,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1819852963089943,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00425,0.267404,0.315398,1.466904,0.753454,-0.696976,-0.123658,-0.178543,-0.895506,1.394413,0.174696,1.609373,-0.4795]],[[2,7,3,-2,-2,44,-2,-2,32,15,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.6884999871253967,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002952,0.270208,0.31647,1.454922,0.766378,0.193533,1.069773,-0.332989,-0.118181,-0.152527,-0.809455,0.960645,0.246159,1.809107,0.079765]],[[53,5,46,-2,-2,42,-2,-2,54,65,-2,-2,-2],[0.5,0.6629999876022339,1.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00041,0.24362,0.29219,1.249469,1.77539,-0.052592,0.317757,-1.540528,-0.114199,-0.145169,-0.799526,1.005927,1.407645]],[[2,5,69,-2,-2,-2,38,21,-2,-2,42,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.000569,0.248316,0.269523,1.170526,1.669669,0.277234,-0.113149,-0.169439,-0.841006,1.330182,0.135726,1.34618,-1.116345]],[[2,8,7,-2,-2,9,-2,-2,8,4,-2,-2,79,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,-2.0,-2.0,0.9634999930858612,-2.0,-2.0,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002331,0.226283,0.282273,1.3827,0.981855,0.165962,0.28373,1.034882,-0.101445,0.158533,1.168732,-0.116184,-0.15063,-0.867882,0.500971]],[[2,19,16,-2,-2,5,-2,-2,0,1,-2,-2,12,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.21949999779462814,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2889999896287918,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00373,0.122429,0.16801,0.872939,-1.20883,-0.395899,-1.419831,-1.905662,-0.171076,-0.193456,-0.919381,1.980549,0.398596,1.735003,1.515691]],[[2,19,2,-2,-2,8,-2,-2,0,1,-2,-2,12,-2,-2],[0.9600000083446503,0.5,0.9350000023841858,-2.0,-2.0,0.6970000267028809,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2574999928474426,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000184,0.126836,0.166299,1.199083,0.230855,-0.370836,-1.335474,-1.850018,-0.168799,-0.187865,-0.899175,1.875686,0.372292,1.667263,1.435113]],[[2,3,5,-2,-2,4,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.04749999940395355,0.05650000087916851,-2.0,-2.0,0.44350001215934753,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005528,0.200064,0.242301,1.139619,1.422982,0.125405,0.916123,-0.207113,-0.099422,-0.152074,-0.847869,0.985408,0.153441,1.327124,-1.373532]],[[53,5,46,-2,-2,42,-2,-2,32,43,-2,-2,42,-2,-2],[0.5,0.5550000071525574,1.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003711,0.190436,0.231628,1.09816,1.66282,-0.018328,0.404739,-1.324369,-0.093512,-0.129502,-0.764466,0.929793,0.216757,1.552487,0.071344]],[[53,5,46,-2,-2,2,-2,-2,2,55,-2,-2,51,-2,-2],[0.5,0.6654999852180481,1.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004098,0.191375,0.225593,1.102682,1.615383,0.005591,0.533204,-1.025064,-0.083677,0.030044,0.320816,-1.598161,-0.172862,-0.906322,0.616842]],[[2,5,42,-2,-2,1,-2,-2,31,15,-2,-2,-2],[0.9350000023841858,0.14750000089406967,0.44949495792388916,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002385,0.172817,0.206578,1.160953,0.350684,0.092399,-0.124143,1.040415,-0.078311,-0.099692,-0.589784,0.818236,1.296393]],[[53,5,46,-2,-2,-2,79,43,-2,-2,42,-2,-2],[0.5,0.9629999995231628,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003186,0.174128,0.195251,0.959071,1.563769,-0.610711,-0.073979,-0.116741,-0.714323,0.789161,0.148657,1.116495,-1.393849]],[[2,19,16,-2,-2,12,-2,-2,0,1,-2,-2,10,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.5444999933242798,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.3541666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003397,0.101452,0.13522,0.78032,-1.03747,-0.307641,-1.077866,-1.797964,-0.128042,-0.147268,-0.766755,1.695022,0.316701,1.317986,1.515016]],[[2,7,3,-2,-2,8,-2,-2,15,31,-2,-2,9,-2,-2],[0.9350000023841858,0.007500000298023224,0.02049999963492155,-2.0,-2.0,0.9359999895095825,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002594,0.161011,0.198279,1.094567,1.259474,0.103275,0.28396,1.377052,-0.071721,-0.096986,-0.56151,1.071888,0.202757,-0.007968,1.451648]],[[53,5,46,-2,-2,1,-2,-2,32,48,-2,-2,42,-2,-2],[0.5,0.6629999876022339,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1547619104385376,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001896,0.141263,0.174362,0.924741,1.51092,-0.047993,-0.642311,0.650605,-0.07047,-0.101129,-0.608637,0.693202,0.229462,1.559826,0.094213]],[[2,8,2,-2,-2,8,-2,-2,38,8,-2,-2,42,-2,-2],[0.9350000023841858,0.15200000256299973,0.8250000178813934,-2.0,-2.0,0.9440000057220459,-2.0,-2.0,0.5,0.05649999901652336,-2.0,-2.0,0.25357143580913544,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003671,0.143342,0.186699,1.119313,1.37907,0.099907,0.362044,1.079911,-0.063514,-0.105424,0.532258,-0.660881,0.150229,1.247341,-0.818569]],[[2,8,2,-2,-2,8,-2,-2,8,4,-2,-2,42,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,-2.0,-2.0,0.9440000057220459,-2.0,-2.0,0.0755000002682209,0.9280000030994415,-2.0,-2.0,0.2604166716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007815,0.136158,0.17967,1.109145,1.338823,0.09232,0.29178,1.114335,-0.073341,0.11005,0.902619,-0.363346,-0.103603,-0.203578,-1.088231]],[[53,69,46,-2,-2,-2,70,54,-2,-2,42,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.25357143580913544,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002694,0.143115,0.163989,0.920517,1.492477,-0.503383,-0.059366,-0.095071,-0.598051,1.097842,0.139951,1.12457,-0.794599]],[[19,2,17,-2,-2,8,-2,-2,9,-2,2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.054500000551342964,-2.0,-2.0,0.6389999985694885,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003593,0.026485,0.100872,0.638323,-1.390701,-0.073791,0.896298,-0.544368,-0.271388,-1.022001,-0.297967,-1.746627,-1.337009]],[[2,19,17,-2,-2,8,-2,-2,0,32,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11600000038743019,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005283,0.083928,0.116835,0.734703,-1.190732,-0.286007,-0.810242,-1.710238,-0.093976,-0.110224,-0.685314,0.790309,0.272495,1.343808,1.470258]],[[19,2,17,-2,-2,8,-2,-2,12,7,-2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.6444999873638153,0.007500000298023224,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.006269,0.03327,0.106942,0.689326,-1.31889,-0.073711,0.792542,-0.607761,-0.25012,-0.222231,-1.053382,-1.345005,-1.648227]],[[2,19,16,-2,-2,5,-2,-2,8,5,-2,-2,42,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.22549999505281448,-2.0,-2.0,0.13199999928474426,0.07250000163912773,-2.0,-2.0,0.11437908560037613,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007349,0.065424,0.097392,0.660824,-0.92711,-0.273165,-1.169742,-1.685948,-0.101588,0.081025,0.868002,-0.515169,-0.13835,-0.295728,-0.997761]],[[2,3,7,-2,-2,4,-2,-2,32,15,-2,-2,42,-2,-2],[0.9350000023841858,0.3439999967813492,0.1655000001192093,-2.0,-2.0,0.1889999955892563,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00306,0.1186,0.138197,0.897853,1.321776,0.029553,0.595304,-0.793425,-0.051474,-0.073273,-0.506343,0.803918,0.171374,1.587215,-0.059766]],[[2,8,2,-2,-2,8,-2,-2,42,79,-2,-2,8,-2,-2],[0.9350000023841858,0.17599999904632568,0.8250000178813934,-2.0,-2.0,0.2565000057220459,-2.0,-2.0,0.11324786394834518,0.5,-2.0,-2.0,0.09149999916553497,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006193,0.103237,0.143858,1.072274,1.266933,0.060057,-0.827724,0.532632,-0.056208,0.056755,-0.069567,1.458613,-0.116504,0.239102,-0.800355]],[[19,8,17,-2,-2,2,-2,-2,55,2,-2,-2,12,-2,-2],[0.5,0.17599999904632568,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.47699999809265137,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004802,0.021246,0.127845,0.922159,-1.156271,-0.029929,0.195182,-0.626111,-0.235329,-0.269314,-1.644362,-1.291841,-0.200943,-1.27357,-1.234165]],[[19,2,16,-2,-2,46,-2,-2,42,2,-2,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.3166666775941849,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.0137,0.036996,0.093593,0.643654,-0.968988,-0.056348,-0.485324,0.912966,-0.22251,-0.245338,-1.489788,-1.248666,-1.058259]],[[11,42,37,-2,-2,1,-2,-2,65,5,-2,-2,-2],[0.20714285969734192,0.13076923787593842,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.032500000670552254,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.004117,-0.045156,0.043449,-0.176272,1.017295,-0.109377,-0.718455,1.024649,0.108491,0.089093,1.002247,0.090245,1.455764]],[[53,69,46,-2,-2,-2,42,70,-2,-2,21,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.17914438992738724,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001272,0.103771,0.119844,0.798077,1.340109,-0.241133,-0.046294,0.022651,-0.145783,1.253845,-0.114398,-0.745701,0.683031]],[[22,2,50,-2,-2,5,-2,-2,52,-2,5,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.4545000046491623,-2.0,-2.0,0.5,-2.0,0.26350000500679016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[7.7e-05,0.018558,0.076623,0.545151,-1.153784,-0.058877,0.081814,-0.680296,-0.213647,-1.385421,-0.17784,-1.233885,-1.203635]],[[19,17,2,-2,-2,12,-2,-2,12,12,-2,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.47699999809265137,-2.0,-2.0,0.6444999873638153,0.5160000026226044,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003951,0.023961,0.037856,0.504254,-0.221939,-0.216598,-0.58752,-1.576394,-0.198441,-0.176221,-1.283286,-0.809411,-1.505774]],[[8,6,2,-2,-2,2,-2,-2,42,38,-2,-2,7,-2,-2],[0.08900000154972076,0.2889999896287918,0.4000000059604645,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.33908046782016754,0.5,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003601,0.109424,0.189043,1.139755,1.286147,0.080736,0.824232,-0.08369,-0.031992,-0.003422,-0.195441,0.688263,-0.175112,-0.349291,-1.15998]],[[19,2,16,-2,-2,8,-2,-2,9,-2,2,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,0.04649999924004078,-2.0,-2.0,0.5315000116825104,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.006653,0.026992,0.077299,0.553823,-1.018347,-0.060078,1.078459,-0.548413,-0.180383,-0.586958,-0.203045,-1.440243,-1.202653]],[[8,9,44,-2,-2,44,-2,-2,42,38,-2,-2,48,-2,-2],[0.09149999916553497,0.945499986410141,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.22649572789669037,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002622,0.096725,0.061205,1.05174,-0.095261,0.15529,1.018686,1.248405,-0.030573,0.017015,-0.114619,0.97639,-0.123905,-0.893431,0.302921]],[[53,39,3,-2,-2,1,-2,-2,79,54,-2,-2,42,-2,-2],[0.5,0.5,0.9385000169277191,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003292,0.082609,0.097147,0.850369,-0.32866,-0.107693,-1.640203,1.325133,-0.044445,-0.072752,-0.524075,1.021649,0.114386,0.94063,-0.952559]],[[2,37,53,-2,-2,3,-2,-2,42,79,-2,-2,8,-2,-2],[0.9350000023841858,0.5,0.5,-2.0,-2.0,0.625,-2.0,-2.0,0.13076923787593842,0.5,-2.0,-2.0,0.07400000095367432,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007967,0.081525,0.050323,0.710336,-0.345245,0.113956,1.03312,0.176488,-0.050225,0.021796,-0.160538,1.139579,-0.095289,0.204128,-0.715379]],[[2,9,8,-2,-2,53,-2,-2,79,21,-2,-2,42,-2,-2],[0.9350000023841858,0.9675000011920929,0.153499998152256,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1882352977991104,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004719,0.078584,0.047936,1.112981,-0.198488,0.116828,0.939618,0.911796,-0.042121,-0.067276,-0.500294,0.96084,0.095001,1.002881,-0.767027]],[[17,55,10,-2,-2,52,-2,-2,9,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.5,-2.0,-2.0,0.9285000264644623,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005723,0.017542,0.034287,-0.091492,0.723277,-0.181642,-1.4643,-1.168532,-0.232154,-1.548324,-1.038334]],[[19,16,2,-2,-2,9,-2,-2,55,8,-2,-2,4,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.5600000023841858,-2.0,-2.0,0.5,0.7885000109672546,-2.0,-2.0,0.6675000190734863,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008594,0.009296,0.025677,0.86469,-0.061013,-0.224973,-1.79709,-0.944623,-0.178831,-0.231488,-1.290257,-1.518725,-0.137875,-1.149087,-1.163769]],[[17,55,5,-2,-2,5,-2,-2,9,-2,-2],[0.5,0.5,0.664000004529953,-2.0,-2.0,0.15850000083446503,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005691,0.016767,0.030827,0.426366,-0.275862,-0.160997,-1.449388,-1.131093,-0.20196,-1.54951,-0.622168]],[[53,69,46,-2,-2,-2,38,21,-2,-2,42,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.004528,0.070308,0.089885,0.713432,1.30142,-0.795518,-0.039466,-0.065831,-0.499804,0.956627,0.098433,0.87487,-1.097523]],[[19,17,10,-2,-2,42,-2,-2,3,-2,55,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,0.009500000160187483,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.005908,0.021639,0.032572,-0.091688,0.69014,-0.183487,-1.541481,-0.256042,-0.145615,-0.642914,-0.165173,-1.372421,-1.165974]],[[22,42,17,-2,-2,8,-2,-2,52,-2,6,-2,-2],[0.5,0.33908046782016754,0.5,-2.0,-2.0,0.024000000208616257,-2.0,-2.0,0.5,-2.0,0.9054999947547913,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.005654,0.008061,0.02963,0.27049,-1.050061,-0.096401,0.873697,-0.931506,-0.15029,-1.158993,-0.122726,-1.120994,-1.143425]],[[19,16,5,-2,-2,4,-2,-2,4,55,-2,-2,-2],[0.5,0.5,0.9794999957084656,-2.0,-2.0,0.9004999995231628,-2.0,-2.0,0.9664999842643738,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001787,0.018222,0.031912,0.349554,-0.702083,-0.1821,-1.66072,-0.204032,-0.140863,-0.159794,-1.337341,-1.183463,-0.52747]],[[32,2,9,-2,-2,15,-2,-2,7,9,-2,-2,-2],[0.5,0.9350000023841858,0.9675000011920929,-2.0,-2.0,0.5,-2.0,-2.0,0.12550000101327896,0.6355000138282776,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00524,-0.020601,0.05998,0.301201,0.771553,-0.06001,-0.481928,0.569089,0.159268,0.108964,1.254239,0.441419,1.27955]],[[21,1,63,-2,-2,2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.13049999624490738,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003306,-0.013804,-0.037395,-0.333356,0.807581,0.080411,0.755781,1.494266,0.174224,0.560311,1.308555]],[[19,2,17,-2,-2,15,-2,-2,12,3,-2,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.6444999873638153,0.009500000160187483,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00231,0.017266,0.052598,0.466343,-1.004731,-0.038672,-0.364398,0.892756,-0.127716,-0.105283,-0.50691,-1.180628,-1.402341]],[[38,8,4,-2,-2,43,-2,-2,42,4,-2,-2,12,-2,-2],[0.5,0.1314999982714653,0.8720000088214874,-2.0,-2.0,0.5,-2.0,-2.0,0.22474747896194458,0.9115000069141388,-2.0,-2.0,0.4714999943971634,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007586,-0.012389,0.069812,1.172465,0.076426,-0.043614,-0.408421,0.671708,0.103408,0.17081,1.31515,0.04359,-0.052701,0.399118,-1.114019]],[[16,19,42,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.42217741906642914,-2.0,-2.0,0.3166666775941849,-2.0,-2.0,0.8680000007152557,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000287,0.011904,0.027703,0.280304,-0.757855,-0.132072,-1.261664,-0.849013,-0.178479,-1.642917,-0.416973]],[[42,69,2,-2,-2,4,-2,-2,2,42,-2,-2,21,-2,-2],[0.13076923787593842,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.9350000023841858,0.1547619104385376,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002044,0.050278,0.009675,0.391433,-0.717449,0.180641,1.760442,0.258607,-0.039808,0.049211,-0.468002,0.60163,-0.072919,-0.549439,0.652708]],[[38,53,46,-2,-2,21,-2,-2,42,2,-2,-2,34,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.11324786394834518,0.9249999821186066,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001626,-0.016918,0.050393,0.373548,1.20418,-0.047876,-0.404848,0.860704,0.09184,0.191484,1.109719,1.397459,0.011322,0.312039,-1.01223]],[[2,16,19,-2,-2,-2,51,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000689,0.042027,0.055455,0.501975,-0.897797,-0.994648,-0.052775,-0.065807,-0.525781,1.516089,0.780597]],[[1,21,83,-2,-2,43,-2,-2,2,42,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4749999940395355,0.21111111342906952,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000376,-0.019875,-0.031252,-0.261192,1.332833,0.137427,-0.16056,1.260866,0.079136,0.067626,1.008063,0.405729,1.495104]],[[2,3,53,-2,-2,4,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.3410000056028366,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2567765712738037,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003656,0.058856,0.078207,0.976435,1.107737,0.022141,0.643272,-0.40213,-0.023188,-0.044722,-0.391814,0.532989,0.085284,0.94659,-0.696419]],[[16,2,17,-2,-2,51,-2,-2,3,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.0949999988079071,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000394,0.008681,0.041433,0.403894,-0.998225,-0.036145,-0.355003,0.786226,-0.143015,-0.431715,-1.713614]],[[64,2,39,-2,-2,21,-2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001109,-0.009669,0.04378,0.535482,-1.483072,-0.03723,-0.320835,0.828055,1.126169]],[[21,84,1,-2,-2,-2,48,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.00345,-0.005924,-0.013131,-0.232376,0.885184,1.320738,0.159619,0.553853,1.26803]],[[19,42,16,-2,-2,8,-2,-2,22,6,-2,-2,6,-2,-2],[0.5,0.4580419659614563,0.5,-2.0,-2.0,0.032500000670552254,-2.0,-2.0,0.5,0.7094999849796295,-2.0,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00301,0.009419,0.020069,0.220486,-0.856028,-0.130461,0.60052,-1.150674,-0.114466,-0.151854,-1.166253,-1.394555,-0.081597,-0.724846,-1.16326]],[[42,2,7,-2,-2,7,-2,-2,8,4,-2,-2,30,-2,-2],[0.02777777798473835,0.9950000047683716,0.8774999976158142,-2.0,-2.0,0.7744999825954437,-2.0,-2.0,0.1314999982714653,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005466,0.051001,0.085555,0.427322,1.700156,-0.074662,0.745574,-1.186444,-0.028847,0.039562,0.82947,-0.256969,-0.055913,-0.496611,0.463617]],[[19,42,69,-2,-2,2,-2,-2,55,69,-2,-2,6,-2,-2],[0.5,0.13076923787593842,1.5,-2.0,-2.0,0.9350000023841858,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9244999885559082,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003013,0.015627,0.063172,0.122372,1.166581,-0.017736,0.675789,-0.3083,-0.115971,-0.149886,-1.226759,-1.138059,-0.085605,-1.084923,-1.143578]],[[10,63,15,-2,-2,-2,17,55,-2,-2,-2],[0.2678571492433548,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.004056,-0.037092,-0.049033,-0.439136,0.695588,0.67565,0.03913,0.05299,0.553483,-1.200567,-1.398651]],[[16,42,69,-2,-2,8,-2,-2,9,-2,-2],[0.5,0.02777777798473835,1.5,-2.0,-2.0,0.09149999916553497,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000126,0.009802,0.07266,0.335506,1.325365,-0.016801,0.418941,-0.275148,-0.169468,-1.702243,-0.603476]],[[63,10,2,-2,-2,4,-2,-2,8,-2,-2],[0.5,0.2678571492433548,0.9350000023841858,-2.0,-2.0,0.15699999779462814,-2.0,-2.0,0.8989999890327454,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001184,-0.007716,-0.040021,0.343678,-0.41929,0.031273,0.855723,0.022063,0.142866,-0.133391,1.476199]],[[21,1,4,-2,-2,46,-2,-2,42,-2,-2],[0.5,0.5,0.3310000002384186,-2.0,-2.0,1.5,-2.0,-2.0,0.20256410539150238,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.011948,-0.019405,-0.038214,-0.031104,-0.713926,0.054782,0.605261,1.422985,0.139284,1.247425,0.437334]],[[42,22,17,-2,-2,52,-2,-2,8,-2,4,-2,-2],[0.4580419659614563,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.013999999966472387,-2.0,0.9050000011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002642,0.013577,0.023666,0.248212,-0.906289,-0.111903,-1.221216,-1.107322,-0.129426,0.842605,-0.153647,-1.058114,-1.67262]],[[42,48,8,-2,-2,12,-2,-2,8,2,-2,-2,2,-2,-2],[0.42217741906642914,0.5,0.1314999982714653,-2.0,-2.0,0.5175000131130219,-2.0,-2.0,0.032500000670552254,0.9000000059604645,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001866,0.009861,-0.002307,0.537094,-0.176186,0.177706,1.409132,0.372957,-0.104838,0.060647,1.150875,0.202378,-0.140047,-1.294117,-0.911838]],[[42,69,38,-2,-2,4,-2,-2,5,12,-2,-2,39,-2,-2],[0.13076923787593842,1.5,0.5,-2.0,-2.0,0.0364999994635582,-2.0,-2.0,0.02850000001490116,0.5370000004768372,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002325,0.042113,0.006704,-0.158299,1.333799,0.162461,1.933499,0.199706,-0.025341,0.035205,0.714476,-0.277219,-0.050572,-0.324351,-1.454965]],[[19,2,17,-2,-2,46,-2,-2,12,5,-2,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.6254999935626984,0.09849999845027924,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001575,0.01248,0.040174,0.412708,-0.759031,-0.037781,-0.448464,0.760931,-0.105071,-0.086225,-1.154567,-0.823637,-1.393877]],[[2,17,19,-2,-2,-2,5,7,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.011500000022351742,0.0025000000605359674,-2.0,-2.0,0.08950000256299973,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.001005,0.026764,0.037358,0.42946,-1.007052,-0.990048,-0.036131,0.100633,-0.052261,1.217231,-0.049194,-0.049433,-0.736369]],[[64,43,53,-2,-2,7,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.031000000424683094,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005305,-0.002108,-0.012637,0.388041,-0.278775,0.13332,1.264227,-0.001797,1.164283]],[[42,2,5,-2,-2,7,-2,-2,2,4,-2,-2,1,-2,-2],[0.029411764815449715,0.9850000143051147,0.21199999749660492,-2.0,-2.0,0.5690000057220459,-2.0,-2.0,0.9350000023841858,0.19299999624490738,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008222,0.040822,0.087859,0.240886,1.196382,-0.064015,0.211849,-0.899827,-0.029672,0.03005,0.786537,0.027284,-0.054727,-0.432429,1.483974]],[[21,17,2,-2,-2,4,-2,-2,5,-2,-2],[0.5,0.5,0.9600000083446503,-2.0,-2.0,0.6245000064373016,-2.0,-2.0,0.13049999624490738,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00826,0.000323,0.00842,0.310728,-0.234674,-0.136503,-1.51025,-0.412683,0.146427,0.596098,1.293644]],[[42,69,70,-2,-2,-2,7,15,-2,-2,4,-2,-2],[0.11324786394834518,2.5,0.5,-2.0,-2.0,-2.0,0.004500000039115548,0.5,-2.0,-2.0,0.28550000488758087,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.007539,0.049392,0.027212,0.067296,1.307402,1.471038,-0.016032,0.036587,0.172678,1.161823,-0.048737,-0.067097,-0.789895]],[[19,42,69,-2,-2,2,-2,-2,3,-2,55,-2,-2],[0.5,0.11324786394834518,1.5,-2.0,-2.0,0.9350000023841858,-2.0,-2.0,0.007500000298023224,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001066,0.01081,0.050241,0.129702,1.06983,-0.012279,0.580769,-0.276099,-0.081589,-0.242258,-0.10091,-1.260301,-1.102327]],[[16,19,42,-2,-2,9,-2,-2,3,-2,-2],[0.5,0.5,0.4522727280855179,-2.0,-2.0,0.6389999985694885,-2.0,-2.0,0.0949999988079071,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001958,0.010017,0.020353,0.22601,-0.703969,-0.084593,-0.248081,-1.204888,-0.151588,-0.762364,-1.659665]],[[8,4,3,-2,-2,7,-2,-2,63,48,-2,-2,-2],[0.09149999916553497,0.8700000047683716,0.021499999798834324,-2.0,-2.0,0.029000000096857548,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002136,0.054651,0.096732,0.793325,1.269339,0.015013,0.637343,-0.930705,-0.021811,-0.033534,-0.366456,0.764164,0.725471]],[[43,8,78,-2,-2,63,-2,-2,9,-2,-2],[0.5,0.1314999982714653,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004908,-0.013537,0.037779,0.231072,1.439946,-0.033281,-0.351909,0.539642,0.111791,0.085457,1.442174]],[[42,69,38,-2,-2,5,-2,-2,2,42,-2,-2,21,-2,-2],[0.12916667014360428,1.5,0.5,-2.0,-2.0,0.851500004529953,-2.0,-2.0,0.9350000023841858,0.4833333343267441,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001749,0.040168,0.003461,-0.182826,1.28866,0.155331,0.550689,1.930645,-0.024677,0.033326,0.545716,-0.791599,-0.047501,-0.438945,0.67692]],[[32,8,5,-2,-2,16,-2,-2,12,-2,12,-2,-2],[0.5,0.09149999916553497,0.31199999153614044,-2.0,-2.0,0.5,-2.0,-2.0,0.408500000834465,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.00142,-0.011182,0.041121,0.556452,-0.668296,-0.028456,-0.16442,-1.419961,0.092277,1.587699,0.041386,-0.458827,1.188297]],[[2,7,22,-2,-2,42,-2,-2,45,0,-2,-2,-2],[0.9600000083446503,0.23349999636411667,0.5,-2.0,-2.0,0.39444445073604584,-2.0,-2.0,0.0005000000237487257,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001363,0.028392,0.008594,0.193021,-1.036388,0.12499,1.085427,-0.658922,-0.035087,-0.041194,-0.406927,1.169969,1.449597]],[[21,17,11,-2,-2,9,-2,-2,5,-2,-2],[0.5,0.5,0.36666667461395264,-2.0,-2.0,0.93299999833107,-2.0,-2.0,0.13099999725818634,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.005713,-0.012177,-0.005017,-0.175715,0.517674,-0.14462,-1.39971,-0.596174,0.121084,0.390058,1.235101]],[[17,16,2,-2,-2,6,-2,-2,9,-2,-2],[0.5,0.5,0.9600000083446503,-2.0,-2.0,0.8805000185966492,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.005077,0.001935,0.008797,0.304928,-0.213199,-0.117993,-1.584891,-0.268025,-0.126848,-1.361596,-0.665489]],[[70,1,21,-2,-2,2,-2,-2,42,6,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.2899159789085388,0.9354999959468842,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001432,-0.0083,-0.0234,-0.254254,0.714282,0.046405,0.579605,1.442271,0.075684,0.137953,1.297465,0.151917,-1.240946]],[[39,42,19,-2,-2,42,-2,-2,12,-2,-2],[0.5,0.5396825671195984,0.5,-2.0,-2.0,0.7321428656578064,-2.0,-2.0,0.4544999897480011,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001597,0.003845,0.00842,0.138859,-0.987232,-0.149567,-1.181417,-2.001637,-0.160741,0.294403,-1.857394]],[[71,11,16,-2,-2,10,-2,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.8333333432674408,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001002,0.005,-0.011592,-0.032932,-1.212038,0.051993,0.943576,0.098671,-1.159437]],[[48,83,4,-2,-2,-2,12,-2,-2],[0.5,0.5,0.12250000238418579,-2.0,-2.0,-2.0,0.5394999980926514,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.004407,-0.003418,-0.009028,0.180722,-0.317756,1.275381,0.120329,1.180843,-0.182914]],[[42,69,2,-2,-2,4,-2,-2,8,5,-2,-2,5,-2,-2],[0.13076923787593842,1.5,0.9850000143051147,-2.0,-2.0,0.054500000551342964,-2.0,-2.0,0.17599999904632568,0.012500000186264515,-2.0,-2.0,0.012500000186264515,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002046,0.033832,0.005365,0.329223,-0.589208,0.124424,1.472293,0.042459,-0.027931,0.021902,0.817286,-0.131523,-0.053886,-1.103167,-0.36539]],[[42,69,7,-2,-2,2,-2,-2,44,5,-2,-2,11,-2,-2],[0.03125,0.5,0.5090000182390213,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.8980000019073486,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005402,0.043414,-0.005809,0.308781,-0.886217,0.125064,1.355706,0.129509,-0.010997,0.021124,0.556991,-0.250747,-0.049428,-0.589833,0.260446]],[[42,22,9,-2,-2,2,-2,-2,3,8,-2,-2,46,-2,-2],[0.33974359929561615,0.5,0.945499986410141,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.009499999694526196,0.06300000101327896,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004872,0.017445,0.025962,-0.112983,0.393635,-0.092853,-1.202623,-1.056632,-0.057362,0.029055,0.96646,-0.671863,-0.098683,-1.134114,0.63639]],[[19,42,37,-2,-2,42,-2,-2,3,-2,55,-2,-2],[0.5,0.5857143104076385,4.5,-2.0,-2.0,0.6904762089252472,-2.0,-2.0,0.007500000298023224,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000225,0.008296,0.012579,0.153296,-0.84173,-0.165657,-1.298327,-1.916103,-0.073543,-0.239364,-0.090083,-1.232667,-1.081108]],[[17,19,2,-2,-2,55,-2,-2,9,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.5,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004012,0.002519,0.011289,0.294419,-0.224348,-0.078701,-1.198758,-0.936032,-0.126495,-1.359271,-0.618199]],[[42,63,70,-2,-2,-2,12,42,-2,-2,39,-2,-2],[0.20256410539150238,0.5,0.5,-2.0,-2.0,-2.0,0.3824999928474426,0.31643131375312805,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00329,0.023739,0.012324,0.016366,0.941233,1.118775,-0.027295,0.038549,0.794141,-0.091521,-0.055224,-0.420603,-1.866615]],[[5,56,19,-2,-2,6,-2,-2,8,-2,2,-2,-2],[0.9794999957084656,0.5,0.5,-2.0,-2.0,0.9695000052452087,-2.0,-2.0,0.9805000126361847,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001593,0.011154,0.017491,0.227479,-0.964074,-0.091435,0.368181,-1.168617,-0.062521,-1.364605,-0.030426,0.599196,-0.963181]],[[42,19,50,-2,-2,55,-2,-2,3,8,-2,-2,10,-2,-2],[0.33908046782016754,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.013500000350177288,0.06650000065565109,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001345,0.009123,0.017796,0.195168,-0.909903,-0.080885,-1.26023,-1.082698,-0.048032,0.030444,0.865842,-0.364661,-0.09089,-0.804111,-1.456931]],[[10,79,15,-2,-2,4,-2,-2,39,17,-2,-2,1,-2,-2],[0.2678571492433548,0.5,0.5,-2.0,-2.0,0.3135000020265579,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001268,-0.024193,-0.039684,-0.429844,0.531994,0.067803,1.073364,-0.695976,0.029092,0.037413,0.478783,-0.824788,-0.116861,-1.56377,1.402839]],[[8,9,5,-2,-2,8,-2,-2,16,5,-2,-2,-2],[0.0925000011920929,0.94200000166893,0.051500000059604645,-2.0,-2.0,0.05049999989569187,-2.0,-2.0,0.5,0.00849999999627471,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000806,0.041035,0.012455,0.556112,-0.731979,0.082461,0.589799,1.608259,-0.013288,-0.005622,-0.844351,0.016674,-1.213959]],[[44,2,4,-2,-2,5,-2,-2,6,-2,48,-2,-2],[0.5,0.9950000047683716,0.4389999955892563,-2.0,-2.0,0.5449999868869781,-2.0,-2.0,0.019999999552965164,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000852,0.022305,0.040064,0.72737,-0.015384,-0.048222,1.192071,-0.942881,-0.030226,1.445323,-0.041092,-0.413337,0.570932]],[[12,8,5,-2,-2,-2,1,46,-2,-2,2,-2,-2],[0.2580000013113022,0.38449999690055847,0.13350000232458115,-2.0,-2.0,-2.0,0.5,2.5,-2.0,-2.0,0.4749999940395355,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003977,0.087729,0.019658,1.03209,-0.901883,1.469748,-0.010516,-0.02217,-0.14651,-1.217708,0.040324,0.572724,1.432022]],[[17,42,79,-2,-2,3,-2,-2,9,-2,-2],[0.5,0.2752525359392166,0.5,-2.0,-2.0,0.2905000001192093,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004647,0.010163,0.02637,0.101667,0.774204,-0.031394,0.007879,-0.953061,-0.095973,-1.258393,-0.370124]],[[42,2,37,-2,-2,78,-2,-2,2,3,-2,-2,4,-2,-2],[0.02777777798473835,0.9850000143051147,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9350000023841858,0.33650000393390656,-2.0,-2.0,0.05849999934434891,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002042,0.036742,0.072199,0.493744,1.469254,-0.044135,-0.698397,1.033308,-0.017559,0.032204,0.827782,-0.343302,-0.037842,0.021858,-0.498762]],[[71,10,15,-2,-2,42,-2,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.5857143104076385,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00072,0.003759,-0.016012,-0.200155,0.787378,0.030282,0.385818,-1.700723,-0.8301]],[[44,2,8,-2,-2,5,-2,-2,11,8,-2,-2,7,-2,-2],[0.5,0.9850000143051147,0.9385000169277191,-2.0,-2.0,0.5449999868869781,-2.0,-2.0,0.4166666716337204,0.2824999988079071,-2.0,-2.0,0.005499999970197678,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003677,0.024149,0.049495,0.285686,1.177749,-0.031384,0.823079,-0.64007,-0.022823,-0.047835,-0.996388,-0.225658,0.047583,0.970568,-0.286977]],[[64,69,63,-2,-2,-2,-2],[0.5,4.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.003944,-0.008567,-0.004453,-0.090821,0.516716,-1.205977,0.768365]],[[30,8,5,-2,-2,44,-2,-2,8,-2,-2],[0.5,0.5940000116825104,0.5920000076293945,-2.0,-2.0,0.5,-2.0,-2.0,0.7730000019073486,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004541,-0.010533,0.018672,0.352358,-1.020972,-0.037381,-0.004071,-0.669211,0.065661,-0.786587,1.163951]],[[16,21,42,-2,-2,37,-2,-2,9,-2,-2],[0.5,0.5,0.18465909361839294,-2.0,-2.0,0.5,-2.0,-2.0,0.8680000007152557,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001205,0.004412,-0.001265,0.205796,-0.292106,0.108595,0.554911,1.267251,-0.099379,-1.128693,-0.180621]],[[5,44,8,-2,-2,4,-2,-2,12,2,-2,-2,2,-2,-2],[0.9794999957084656,0.5,0.9494999945163727,-2.0,-2.0,0.02049999963492155,-2.0,-2.0,0.5264999866485596,0.9950000047683716,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006974,0.00216,0.031025,0.201477,0.862215,-0.02714,0.315977,-0.362367,-0.074618,-0.118397,-1.363179,-1.117534,-0.034839,0.564374,-0.977726]],[[16,54,17,-2,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.006544,-0.000703,-0.005742,-0.007447,-0.961621,0.893004,-0.101971,-1.280337,-0.22962]],[[4,4,69,-2,-2,-2,21,1,-2,-2,-2],[0.05849999934434891,0.05049999989569187,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.001299,0.027113,0.015567,-0.103818,0.66954,1.56726,-0.019619,-0.029026,-0.437188,0.466994,0.857694]],[[42,69,6,-2,-2,4,-2,-2,71,33,-2,-2,-2],[0.13076923787593842,1.5,0.3095000088214874,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.004624,0.032356,0.005477,-0.835724,0.309926,0.111919,1.33855,0.010122,-0.014414,-0.01019,-0.060364,-1.388137,-1.005015]],[[70,3,12,-2,-2,16,-2,-2,42,9,-2,-2,-2],[0.5,0.001500000071246177,0.2590000033378601,-2.0,-2.0,0.5,-2.0,-2.0,0.25357143580913544,0.9839999973773956,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.008746,-0.000875,0.052927,1.707276,0.450089,-0.012596,-0.048072,-1.396099,0.082625,0.129655,0.574254,1.330391,-0.488224]],[[42,69,2,-2,-2,4,-2,-2,66,2,-2,-2,-2],[0.11324786394834518,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002719,0.030667,0.009305,0.360704,-0.495765,0.111573,1.551899,0.131246,-0.012778,-0.006797,0.152987,-0.383428,-1.204326]],[[71,55,42,-2,-2,52,-2,-2,-2],[0.5,0.5,0.612500011920929,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[5.6e-05,0.00283,0.00907,0.117281,-1.545671,-0.070437,-1.274453,-0.938348,-0.975675]],[[84,11,16,-2,-2,9,-2,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.7730000019073486,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002446,-0.006741,-0.020573,-0.117513,-1.371709,0.034858,-0.17669,0.957185,1.270246]],[[11,9,-2,1,-2,-2,9,12,-2,-2,44,-2,-2],[0.36666667461395264,0.25950001180171967,-2.0,0.5,-2.0,-2.0,0.5744999945163727,0.7144999802112579,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.00267,-0.014312,-1.584322,-0.009131,-0.154011,0.709786,0.030829,-0.008778,-0.767109,0.835944,0.06747,0.376175,0.973194]],[[5,2,42,-2,-2,4,-2,-2,4,2,-2,-2,-2],[0.6755000054836273,0.9950000047683716,0.28285714983940125,-2.0,-2.0,0.030499999411404133,-2.0,-2.0,0.9510000050067902,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.004406,0.019218,0.009241,0.274813,-0.344105,0.106942,1.741114,0.440547,-0.029175,-0.015695,0.348621,-0.536764,-1.422834]],[[3,56,42,-2,-2,8,-2,-2,7,-2,-2],[0.99549999833107,0.5,0.5857143104076385,-2.0,-2.0,0.734499990940094,-2.0,-2.0,0.9560000002384186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000672,0.002708,0.007868,0.109604,-1.403697,-0.086259,-0.132417,-1.196838,-0.147965,-1.450939,-1.272425]],[[21,63,17,-2,-2,-2,6,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.789000004529953,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.003068,-0.001812,-0.007117,-0.023076,-0.909866,0.556079,0.092586,0.194517,1.232308]],[[38,16,44,-2,-2,9,-2,-2,46,42,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,2.5,0.2613636404275894,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.004067,-0.014684,-0.008513,0.136908,-0.26436,-0.106828,-1.484123,-0.633888,0.04575,0.061281,1.008521,-0.470197,-1.210329]],[[42,69,2,-2,-2,4,-2,-2,7,2,-2,-2,39,-2,-2],[0.02777777798473835,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.007500000298023224,0.9850000143051147,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004744,0.028898,0.009119,0.353019,-0.547488,0.104863,0.99557,0.414616,-0.018686,0.010658,-0.07016,0.88384,-0.043127,-0.335952,-1.266438]],[[71,84,21,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.000182,0.00265,-0.000807,-0.056683,0.810073,1.112641,-0.93249]],[[42,69,2,-2,-2,4,-2,-2,42,8,-2,-2,4,-2,-2],[0.02777777798473835,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.10435571894049644,0.7625000178813934,-2.0,-2.0,0.05849999934434891,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000393,0.031007,0.009544,0.451673,-0.701508,0.120823,1.145975,0.377893,-0.012822,-0.133066,-1.354524,-0.447568,-0.006328,0.259238,-0.237592]],[[10,79,6,-2,-2,42,-2,-2,9,78,-2,-2,8,-2,-2],[0.2678571492433548,0.5,0.32350000739097595,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,0.9684999883174896,0.5,-2.0,-2.0,0.06149999983608723,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003982,-0.01345,-0.025634,-0.818523,-0.043062,0.055648,0.797646,-1.373471,0.026615,0.002911,-0.130979,1.43351,0.071795,0.446858,1.043921]],[[38,3,12,-2,-2,16,-2,-2,6,42,-2,-2,69,-2,-2],[0.5,0.001500000071246177,0.2590000033378601,-2.0,-2.0,0.5,-2.0,-2.0,0.9884999990463257,0.2361111119389534,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003636,-0.005208,0.041716,1.537183,0.296204,-0.014689,-0.075187,-1.449903,0.049188,0.074092,1.137328,-0.142098,-0.040261,1.124429,-0.94779]],[[42,54,50,-2,-2,-2,5,37,-2,-2,2,-2,-2],[0.42217741906642914,0.5,0.5,-2.0,-2.0,-2.0,0.019499999471008778,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00149,0.008528,0.003809,0.094283,-1.017715,0.876961,-0.05524,0.03657,-0.132031,1.21242,-0.08606,-1.230864,-0.762839]],[[16,56,42,-2,-2,37,-2,-2,6,-2,-2],[0.5,0.5,0.5857143104076385,-2.0,-2.0,0.5,-2.0,-2.0,0.9580000042915344,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000324,0.005693,0.01109,0.143121,-1.265793,-0.080107,0.064652,-1.025859,-0.079913,-1.276053,-0.043889]],[[11,44,6,-2,-2,2,-2,-2,7,8,-2,-2,12,-2,-2],[0.4166666716337204,0.5,0.9779999852180481,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,0.004500000039115548,0.051500000059604645,-2.0,-2.0,0.898499995470047,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00128,-0.015442,0.011592,-0.133073,0.989893,-0.049165,-0.502063,0.514447,0.038124,0.073958,0.766447,1.387412,0.001128,-0.146108,1.383056]],[[19,66,70,-2,-2,4,-2,-2,4,55,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.04000000096857548,-2.0,-2.0,0.9564999938011169,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002766,0.00907,0.013439,0.042853,0.734861,-0.104831,0.149348,-1.328237,-0.059441,-0.073838,-1.186874,-1.071781,-0.089149]],[[5,55,9,-2,-2,52,-2,-2,4,2,-2,-2,-2],[0.992499977350235,0.5,0.9465000033378601,-2.0,-2.0,0.5,-2.0,-2.0,0.012500000186264515,0.9950000047683716,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002719,0.007932,0.013855,-0.112712,0.292606,-0.061346,-1.248618,-0.896947,-0.062934,-0.104456,-1.32492,-1.071855,-0.269394]],[[12,6,4,-2,-2,-2,17,42,-2,-2,9,-2,-2],[0.2580000013113022,0.9745000004768372,0.013000000268220901,-2.0,-2.0,-2.0,0.5,0.02777777798473835,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.001537,0.065536,0.106191,-0.544883,1.185786,-0.322713,-0.006649,-0.002222,0.31029,-0.164379,-0.087559,-1.302563,-0.412811]],[[7,30,3,-2,-2,-2,4,6,-2,-2,15,-2,-2],[0.006500000134110451,0.5,0.7685000002384186,-2.0,-2.0,-2.0,0.16350000351667404,0.9934999942779541,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.004009,0.022671,0.014501,0.369446,-0.59831,0.948688,-0.021328,0.006648,-0.062066,1.48861,-0.053685,-0.649779,0.050326]],[[38,53,3,-2,-2,54,-2,-2,35,6,-2,-2,-2],[0.5,0.5,0.2515000030398369,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.6420000195503235,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[9.8e-05,-0.008358,0.022838,0.711221,-0.487771,-0.021746,-0.259204,0.596811,0.042068,0.025964,0.890047,-0.161119,1.151606]],[[71,8,3,-2,-2,79,-2,-2,-2],[0.5,0.6189999878406525,0.3410000056028366,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003366,-0.000643,0.020009,0.168914,1.50731,-0.022223,-0.340231,0.389976,-0.907833]],[[43,8,4,-2,-2,3,-2,-2,6,-2,-2],[0.5,0.17599999904632568,0.9460000097751617,-2.0,-2.0,0.014500000048428774,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000337,-0.005363,0.025168,0.545223,-0.039043,-0.019528,-0.673766,-0.069362,0.073081,-0.226499,0.902454]],[[65,48,2,-2,-2,9,-2,-2,5,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.9784999787807465,-2.0,-2.0,0.04050000011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004746,-0.00952,-0.014788,0.278292,-0.284468,0.065771,-0.156631,1.115382,0.062686,-0.190421,0.878464]],[[16,44,2,-2,-2,6,-2,-2,9,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,0.009499999694526196,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00015,0.004736,0.022328,0.612885,-0.286902,-0.019054,1.974102,-0.230328,-0.090633,-1.173085,-0.644347]],[[2,9,44,-2,-2,12,-2,-2,5,4,-2,-2,3,-2,-2],[0.9600000083446503,0.9399999976158142,0.5,-2.0,-2.0,0.6534999907016754,-2.0,-2.0,0.3554999977350235,0.020999999716877937,-2.0,-2.0,0.8765000104904175,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000332,0.015617,-0.02399,0.101274,-0.630389,0.043771,0.184902,0.82816,-0.021961,0.020152,1.013858,-0.045639,-0.052674,-0.87074,-0.079274]],[[2,41,69,-2,-2,-2,1,12,-2,-2,-2],[0.9750000238418579,0.5,2.5,-2.0,-2.0,-2.0,0.5,0.2575000077486038,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.002089,0.011795,0.005694,-0.044561,0.56525,1.611536,-0.025388,-0.030575,0.611658,-0.42343,1.48859]],[[63,8,5,-2,-2,12,-2,-2,-2],[0.5,0.5940000116825104,0.6104999780654907,-2.0,-2.0,0.2499999925494194,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003313,-0.007578,0.015554,0.308312,-0.662801,-0.029925,1.086226,-0.352537,0.549796]],[[17,42,19,-2,-2,1,-2,-2,42,-2,-2],[0.5,0.33908046782016754,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000338,0.003813,0.013672,0.206392,-0.924839,-0.041851,-0.821252,1.171159,-0.074765,-1.297067,0.38572]],[[17,10,63,-2,-2,9,-2,-2,4,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.45350000262260437,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003919,0.00037,-0.018463,-0.253659,0.485553,0.02496,-0.203777,0.649736,-0.090548,-1.528135,-0.231599]],[[66,70,17,-2,-2,42,-2,-2,6,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.30414746701717377,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000305,0.003341,-0.004172,0.011308,-1.07519,0.069603,1.051641,-0.926123,-0.10674,-0.010457,-1.410729]],[[63,44,2,-2,-2,6,-2,-2,12,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.585999995470047,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000357,-0.003868,0.012158,0.4328,-0.387221,-0.024717,1.507218,-0.337272,0.068705,-0.651235,1.176599]],[[39,17,19,-2,-2,42,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,0.02350000012665987,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002821,0.001626,0.005763,0.126056,-0.933659,-0.071158,-1.240104,0.04456,-0.148988,-0.136394,-1.540765]],[[4,4,69,-2,-2,-2,1,54,-2,-2,46,-2,-2],[0.05849999934434891,0.051500000059604645,2.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002797,0.018657,0.008626,-0.071508,0.808721,1.630587,-0.016272,-0.032983,-0.408465,0.768286,0.030799,0.523853,1.423419]],[[5,7,15,-2,-2,42,-2,-2,42,-2,6,-2,-2],[0.9895000159740448,0.9535000026226044,0.5,-2.0,-2.0,0.19090909510850906,-2.0,-2.0,0.04545454680919647,-2.0,0.26900000870227814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002067,0.008167,0.00048,-0.07635,0.55642,0.099249,1.524646,-0.514032,-0.054566,0.248767,-0.082695,-0.325905,-1.275687]],[[5,44,12,-2,-2,6,-2,-2,42,8,-2,-2,42,-2,-2],[0.9905000030994415,0.5,0.6304999887943268,-2.0,-2.0,0.010499999858438969,-2.0,-2.0,0.2678571492433548,0.99549999833107,-2.0,-2.0,0.41428571939468384,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002702,0.007503,0.02612,0.635254,-0.045373,-0.014339,1.760106,-0.188052,-0.053676,-0.081159,-1.183275,-0.410541,-0.005294,0.3792,-1.448455]],[[19,3,3,-2,-2,8,-2,-2,22,-2,42,-2,-2],[0.5,0.9504999816417694,0.9205000102519989,-2.0,-2.0,0.9724999964237213,-2.0,-2.0,0.5,-2.0,0.3166666775941849,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.004267,0.009664,0.016466,0.123248,1.172714,-0.037346,-1.540649,-0.181994,-0.047792,-1.224468,-0.02364,-0.955998,0.21902]],[[42,69,7,-2,-2,2,-2,-2,66,42,-2,-2,-2],[0.02777777798473835,0.5,0.31050001084804535,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.08712121471762657,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001092,0.024547,-0.00281,0.388987,-0.873515,0.07272,1.138154,-0.108007,-0.012107,-0.006818,-0.897693,-0.039181,-1.240923]],[[71,8,9,-2,-2,5,-2,-2,-2],[0.5,0.5940000116825104,0.7730000019073486,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001886,0.004259,0.023718,-0.089659,0.638607,-0.014259,-0.902528,-0.088339,-0.809261]],[[32,75,43,-2,-2,-2,12,-2,3,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.4440000057220459,-2.0,0.01699999999254942,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[-0.006026,-0.011488,-0.007885,-0.149534,0.472157,-0.898599,0.048111,0.929553,0.022692,-0.646173,0.683482]],[[39,11,44,-2,-2,44,-2,-2,45,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00531,-0.001785,-0.01177,0.102777,-0.309353,0.027432,0.052133,0.567812,-0.10559,-1.304321,1.407893]],[[42,16,3,-2,-2,4,-2,-2,5,-2,1,-2,-2],[0.42217741906642914,0.5,0.99549999833107,-2.0,-2.0,0.8574999868869781,-2.0,-2.0,0.004500000039115548,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.006882,0.013368,0.017569,0.219289,-0.975603,-0.06422,-1.605386,0.211881,-0.044989,1.374,-0.065083,-0.962386,0.937286]],[[42,22,37,-2,-2,52,-2,-2,42,-2,-2],[0.612500011920929,0.5,4.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7321428656578064,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005058,0.007384,0.01195,0.16649,-0.622761,-0.049773,-0.963046,-0.87617,-0.116692,-1.180955,-1.790199]],[[21,4,4,-2,-2,63,-2,-2,5,-2,-2],[0.5,0.007500000298023224,0.004500000039115548,-2.0,-2.0,0.5,-2.0,-2.0,0.13049999624490738,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000468,-0.004086,0.036679,-0.002627,1.406784,-0.012512,-0.200413,0.454984,0.076041,0.362794,1.03018]],[[11,16,44,-2,-2,-2,7,3,-2,-2,12,-2,-2],[0.4166666716337204,0.5,0.5,-2.0,-2.0,-2.0,0.005499999970197678,0.004500000039115548,-2.0,-2.0,0.898499995470047,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003043,-0.00762,-0.001591,0.225872,-0.250073,-1.472972,0.03346,0.060036,0.623609,1.299682,0.000823,-0.148192,1.367355]],[[56,64,9,-2,-2,-2,9,-2,-2],[0.5,0.5,0.945499986410141,-2.0,-2.0,-2.0,0.9745000004768372,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.001454,0.003045,-0.000555,-0.281749,0.138533,0.791846,-0.07546,-1.147909,-0.358822]],[[9,5,4,-2,-2,4,-2,-2,48,63,-2,-2,-2],[0.9465000033378601,0.8040000200271606,0.05949999950826168,-2.0,-2.0,0.403999999165535,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001893,-0.02186,-0.007536,0.547789,-0.317421,-0.100331,-0.747332,-1.228167,0.011468,0.002634,-0.075539,0.897002,1.103165]],[[4,79,8,-2,-2,9,-2,-2,7,17,-2,-2,12,-2,-2],[0.16350000351667404,0.5,0.40949998795986176,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.007500000298023224,0.5,-2.0,-2.0,0.8224999904632568,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002164,0.014281,0.000832,0.882562,-0.218142,0.076198,0.479585,1.274151,-0.017018,0.013592,0.301655,-0.85724,-0.045224,-0.615039,0.622945]],[[37,42,69,-2,-2,5,-2,-2,-2],[4.5,0.13076923787593842,2.5,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000117,0.003336,0.02324,0.106528,1.093944,-0.011079,0.315133,-0.3067,-0.793611]],[[5,7,6,-2,-2,2,-2,-2,3,-2,3,-2,-2],[0.9895000159740448,0.9335000216960907,0.014500000048428774,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,0.4675000160932541,-2.0,0.9884999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.002348,0.003607,-0.002416,1.246939,-0.084371,0.063687,1.148223,-0.425502,-0.066787,-1.179911,-0.040818,-0.145604,-1.087054]],[[11,16,2,-2,-2,-2,7,8,-2,-2,12,-2,-2],[0.36666667461395264,0.5,0.9750000238418579,-2.0,-2.0,-2.0,0.005499999970197678,0.04350000061094761,-2.0,-2.0,0.7269999980926514,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-4.4e-05,-0.009502,-0.004286,0.125441,-0.325385,-1.479065,0.02781,0.051826,0.537329,1.117404,-0.002994,-0.376422,0.825827]],[[9,32,5,-2,-2,12,-2,-2,63,6,-2,-2,-2],[0.9465000033378601,0.5,0.7985000014305115,-2.0,-2.0,0.5379999876022339,-2.0,-2.0,0.5,0.32199999690055847,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00157,-0.019169,-0.031048,-0.234502,-1.22833,0.060881,1.212747,0.540457,0.014982,0.008138,-0.651157,0.251154,0.799921]],[[1,39,79,-2,-2,-2,2,7,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.4749999940395355,0.12050000205636024,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.002155,-0.010001,-0.006245,-0.140563,0.433544,-0.893061,0.033262,0.024018,1.020617,-0.198405,1.407447]],[[42,5,3,-2,-2,43,-2,-2,8,8,-2,-2,2,-2,-2],[0.42396312952041626,0.21049999445676804,0.34700000286102295,-2.0,-2.0,0.5,-2.0,-2.0,0.032500000670552254,0.004500000039115548,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003161,0.009163,-0.010651,0.095923,-0.804292,0.03004,0.211347,1.294843,-0.048094,0.028186,1.134932,-0.057546,-0.069652,-1.155373,-0.673817]],[[44,2,8,-2,-2,46,-2,-2,6,-2,38,-2,-2],[0.5,0.9950000047683716,0.9574999809265137,-2.0,-2.0,1.5,-2.0,-2.0,0.02850000001490116,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004248,0.011005,0.025716,0.184784,0.659168,-0.044693,-1.138682,0.920725,-0.024556,1.087721,-0.03206,-0.419068,0.276015]],[[44,2,4,-2,-2,5,-2,-2,9,4,-2,-2,10,-2,-2],[0.5,0.9950000047683716,0.15949999541044235,-2.0,-2.0,0.6780000030994415,-2.0,-2.0,0.38850000500679016,0.060499999672174454,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002062,0.016465,0.029376,0.734561,0.081931,-0.02758,0.873669,-0.857738,-0.016844,-0.074048,0.522432,-1.200785,-0.005665,-0.202852,0.470731]],[[12,6,2,-2,-2,3,-2,-2,2,5,-2,-2,20,-2,-2],[0.2580000013113022,0.800000011920929,0.9850000143051147,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,0.9600000083446503,0.2030000016093254,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004199,0.053957,0.12462,1.623097,-0.040018,-0.011142,1.288952,-0.923946,0.000308,0.013216,-0.068963,0.472129,-0.017074,-1.014012,-0.123706]],[[42,9,4,-2,-2,63,-2,-2,42,-2,-2],[0.5857143104076385,0.9494999945163727,0.007500000298023224,-2.0,-2.0,0.5,-2.0,-2.0,0.7321428656578064,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000404,0.002816,-0.014681,1.321907,-0.289854,0.014435,0.08547,0.945285,-0.094062,-1.200189,-1.699044]],[[42,2,5,-2,-2,78,-2,-2,4,4,-2,-2,54,-2,-2],[0.02777777798473835,0.9850000143051147,0.19600000232458115,-2.0,-2.0,0.5,-2.0,-2.0,0.05849999934434891,0.0494999997317791,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001827,0.027837,0.051876,0.070565,0.989508,-0.023011,-0.553807,1.132196,-0.009098,0.016,0.037169,1.159718,-0.023185,-0.346731,0.787474]],[[16,42,17,-2,-2,8,-2,-2,9,-2,-2],[0.5,0.6458333432674408,0.5,-2.0,-2.0,0.7425000071525574,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00231,0.005951,0.007833,0.130375,-0.69354,-0.108044,-1.370807,-1.41814,-0.057485,-1.402358,0.105227]],[[19,42,46,-2,-2,1,-2,-2,55,42,-2,-2,5,-2,-2],[0.5,0.42217741906642914,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.2136363685131073,-2.0,-2.0,0.1550000011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000885,0.006033,0.01164,0.164656,-0.682526,-0.043423,-0.761981,1.08453,-0.042323,-0.062627,-0.685528,-1.210146,-0.024623,-1.079643,-0.637904]],[[71,21,8,-2,-2,37,-2,-2,-2],[0.5,0.5,0.0925000011920929,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.004424,-0.002048,-0.005364,0.365913,-0.150946,0.063783,0.197375,1.010658,-1.096034]],[[19,69,3,-2,-2,42,-2,-2,42,4,-2,-2,-2],[0.5,2.5,0.11449999734759331,-2.0,-2.0,0.11324786394834518,-2.0,-2.0,0.3166666775941849,0.5724999904632568,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003707,0.008203,0.002779,0.233145,-0.151797,0.045345,1.334254,-0.01956,-0.03949,-0.051781,-1.216615,-0.524285,-0.505946]],[[5,19,33,-2,-2,12,-2,-2,20,-2,8,-2,-2],[0.9065000116825104,0.5,0.5,-2.0,-2.0,0.617000013589859,-2.0,-2.0,0.5,-2.0,0.9814999997615814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001281,0.009457,0.016053,0.210412,-0.936823,-0.043239,-0.699073,-1.247585,-0.032414,-1.307468,-0.017748,-0.697152,0.196505]],[[64,21,53,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.00321,-0.006165,-0.009454,0.196697,-0.20088,0.713711,0.801219]],[[9,1,4,-2,-2,10,-2,-2,63,8,-2,-2,-2],[0.9494999945163727,0.5,0.05849999934434891,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.5,0.5179999768733978,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000256,-0.018402,-0.034399,0.102466,-0.584435,0.032196,1.354459,0.597641,0.011709,0.004743,0.60494,-0.126341,0.916961]],[[16,12,2,-2,-2,63,-2,-2,6,-2,-2],[0.5,0.5814999938011169,0.9350000023841858,-2.0,-2.0,0.5,-2.0,-2.0,0.7619999945163727,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001086,0.005141,-0.009463,0.486166,-0.230608,0.024755,0.199626,0.897856,-0.068299,-0.525005,-1.061161]],[[35,84,4,-2,-2,-2,-2],[0.5,0.5,0.05849999934434891,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.005157,0.002675,6.3e-05,0.181094,-0.131355,1.060957,0.969132]],[[42,4,6,-2,-2,69,-2,-2,7,-2,2,-2,-2],[0.5505050718784332,0.08950000256299973,0.9905000030994415,-2.0,-2.0,0.5,-2.0,-2.0,0.005500000203028321,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.001056,0.001393,0.016396,0.088045,0.902968,-0.009574,0.166472,-0.42125,-0.072576,-0.291007,-0.101307,-1.791114,-1.143602]],[[5,16,17,-2,-2,4,-2,-2,7,-2,-2],[0.9945000112056732,0.5,0.5,-2.0,-2.0,0.8574999868869781,-2.0,-2.0,0.9964999854564667,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00158,0.002012,0.005646,0.11027,-0.665144,-0.058346,-1.434477,-0.117695,-0.069333,-0.627816,-1.191682]],[[6,70,3,-2,-2,42,-2,-2,3,5,-2,-2,7,-2,-2],[0.9575000107288361,0.5,0.34700000286102295,-2.0,-2.0,0.2613636404275894,-2.0,-2.0,0.38350000977516174,0.6244999766349792,-2.0,-2.0,0.3294999897480011,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00393,-0.008833,-0.01851,0.014948,-0.483393,0.050217,0.981069,-0.96183,0.02397,0.009638,0.392163,-0.821636,0.074684,0.170222,1.349584]],[[5,3,4,-2,-2,37,-2,-2,2,6,-2,-2,42,-2,-2],[0.21049999445676804,0.34700000286102295,0.00849999999627471,-2.0,-2.0,1.5,-2.0,-2.0,0.9600000083446503,0.9205000102519989,-2.0,-2.0,0.1180555559694767,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003232,-0.010285,0.001795,1.248494,-0.103024,-0.058105,-0.830323,0.002297,0.01844,0.061335,0.191364,1.179272,-0.009978,0.363217,-0.597867]],[[44,6,3,-2,-2,3,-2,-2,46,30,-2,-2,-2],[0.5,0.9784999787807465,0.006500000134110451,-2.0,-2.0,0.35450001060962677,-2.0,-2.0,2.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-5e-06,0.012522,-0.004332,0.832796,-0.240187,0.051468,0.538378,1.740101,-0.016301,-0.012347,-0.192241,0.572869,-1.256871]],[[54,17,84,-2,-2,4,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.36800000071525574,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003738,0.000475,0.004604,0.021578,1.185476,-0.073912,-1.571438,-0.428667,0.069633,0.946967,0.252891]],[[5,19,3,-2,-2,42,-2,-2,12,3,-2,-2,-2],[0.9895000159740448,0.5,0.9895000159740448,-2.0,-2.0,0.3541666716337204,-2.0,-2.0,0.8680000007152557,0.24450000375509262,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002304,0.006869,0.012422,0.10221,0.768182,-0.03844,-1.132815,-0.085678,-0.039752,-0.059478,-1.192867,-0.580475,0.931633]],[[65,56,3,-2,-2,4,-2,-2,12,-2,-2],[0.5,0.5,0.99549999833107,-2.0,-2.0,0.5519999861717224,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002074,-0.006993,-0.002716,-0.005682,-1.209631,-0.08624,-0.592422,-1.29637,0.06484,1.063865,0.451043]],[[37,42,55,-2,-2,10,-2,-2,-2],[4.5,0.5857143104076385,0.5,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001704,0.004456,0.006779,0.128777,-1.046918,-0.082261,-1.241437,-1.582339,-0.563917]],[[3,44,2,-2,-2,5,-2,-2,5,-2,-2],[0.9934999942779541,0.5,0.9950000047683716,-2.0,-2.0,0.6159999966621399,-2.0,-2.0,0.940500020980835,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001573,0.001926,0.015645,0.421401,-0.472031,-0.015424,-0.061883,-1.073026,-0.087054,-1.34092,-0.918308]],[[44,6,2,-2,-2,4,-2,-2,6,-2,6,-2,-2],[0.5,0.99549999833107,0.9750000238418579,-2.0,-2.0,0.6590000092983246,-2.0,-2.0,0.009499999694526196,-2.0,0.24050000309944153,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.001544,0.013418,0.005379,0.47515,-0.410046,0.065843,2.555639,0.229348,-0.021714,1.545209,-0.028322,-0.863256,-0.199803]],[[17,10,2,-2,-2,4,-2,-2,42,-2,-2],[0.5,0.3693181872367859,0.9350000023841858,-2.0,-2.0,0.5660000145435333,-2.0,-2.0,0.2928571552038193,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001753,0.001167,-0.00834,0.393795,-0.184569,0.017429,0.696468,-0.092287,-0.053439,-1.288533,0.497636]],[[37,73,42,-2,-2,-2,-2],[4.5,0.5,0.02777777798473835,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.004006,0.007123,0.00485,0.322118,-0.061888,0.873902,-0.744397]],[[42,70,5,-2,-2,3,-2,-2,5,3,-2,-2,45,-2,-2],[0.33908046782016754,0.5,0.9535000026226044,-2.0,-2.0,0.018499999772757292,-2.0,-2.0,0.006500000134110451,0.001500000071246177,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002834,0.008898,0.002731,-0.071344,0.492997,0.055323,0.019332,0.868612,-0.026177,0.070259,0.880603,1.443488,-0.043308,-0.837326,1.293764]],[[22,42,9,-2,-2,5,-2,-2,52,-2,9,-2,-2],[0.5,0.33908046782016754,0.9465000033378601,-2.0,-2.0,0.019499999471008778,-2.0,-2.0,0.5,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.005551,0.009914,0.016928,-0.14218,0.352394,-0.02412,0.703701,-0.768906,-0.04504,-1.135273,-0.018715,-0.435675,-1.05671]]]
UMBRAL = 0.25
UMBRAL_LARGAS = 0.25
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


# v5: regla de alta precisión. Si el verificador ha rehecho una cuenta de la
# frase y NO cuadra, o un plazo entre dos fechas no cuadra, damos la alarma
# aunque el modelo diga lo contrario. En train acierta ~7 de cada 10 veces y
# un escape cuesta 4 veces más que una falsa alarma, así que compensa; pero
# son casos poco frecuentes y los árboles no llegan a aprenderlo solos.
REGLA_CUENTA_MAL = True
PROB_REGLA = 0.5


def aplicar_reglas(p, feats):
    if REGLA_CUENTA_MAL and (feats.get("verif_cuenta_mal", 0) > 0 or feats.get("fecha_plazo_falso", 0) > 0):
        return max(p, PROB_REGLA)
    return p


def prob_caso(caso):
    feats = caracteristicas(caso)
    return aplicar_reglas(probabilidad(feats), feats)


def agregar_oraciones(entero, peor):
    """v5: en respuestas de varias oraciones manda la PEOR oración (con su
    propio corte, UMBRAL_LARGAS). En el pseudo-giro fue mejor que mezclarla
    con la puntuación de la respuesta entera (+0,43 frente a +0,38)."""
    return peor


class Guardian:
    def score(self, caso):
        try:
            subcasos = por_oracion(caso)
            if not subcasos:
                return reescalar(prob_caso(caso), UMBRAL)
            # Varias oraciones (el giro tiene 3-4): una sola afirmación falsa
            # se diluye si miramos la respuesta entera. Puntuamos cada oración
            # como si fuera una respuesta suelta (el modelo se entrenó así) y
            # nos quedamos con la peor, con el corte propio de respuestas largas.
            peor = max(prob_caso(sc) for sc in subcasos)
            return reescalar(agregar_oraciones(prob_caso(caso), peor), UMBRAL_LARGAS)
        except Exception:
            # Ante un caso raro, mejor dar la alarma (un escape cuesta 4 veces más)
            return 1.0
