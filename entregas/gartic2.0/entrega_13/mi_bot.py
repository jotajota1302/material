"""El Guardián — bot de joseca (versión 10).

v10 = v9 con el corte de respuestas LARGAS fijado en 0,18 (el de v4): el
análisis del profesor muestra que v4 (123 alarmas de 220 en el giro) saca
0,497 y que nuestras versiones más «prudentes» con las respuestas largas
(110 alarmas o menos) bajaron a 0,377-0,407. Con 0,18 el modelo de v6 da
127 alarmas en el giro: ordena como v6 y alarma como v4.

--- Notas de la versión 9 ---

v9 = v6 + DOS cortes (sugerencia del profesor): uno para respuestas cortas
(1 oración, UMBRAL) y otro para respuestas largas (2+ oraciones,
UMBRAL_LARGAS), elegido con respuestas reales de varias oraciones del train.
Se sigue puntuando la respuesta entera. Como el test casi no tiene respuestas
largas, el cambio afecta sobre todo al giro. Banco de pruebas, respuestas
largas reales del train: +0,646 (un corte) -> +0,687 (corte propio).

--- Notas de la versión 6 ---

v6 = la forma de puntuar de v4 (la que mejor ha ido con el giro: 0,497 en
test + giro, frente a 0,377 de v5) con las mejoras que lo superan en el banco
de pruebas sin disparar el giro:
  - verificador de cuentas de v5 (cuentas de la frase, unidades, comparaciones);
  - sujeto generalizado (B4 de Candela y «colectivo respaldado»);
  - cobertura multifuente (cuánto de la respuesta cubren 1, 2 o 3 fuentes).
Puntúa la RESPUESTA ENTERA con un único corte, como v4: ni corte por oraciones
ni regla fija (las dos empeoraban el giro). El resto de lo que se calcula
abajo se mantiene, pero solo entran al modelo las características de USAR en
entrenar.py.
Banco de pruebas (H honesto): v4 +0,51 -> v6 +0,63; AUROC 0,941 -> 0,963;
porcentaje del giro marcado 56 % -> 50 %.

--- Notas de la versión 5 (de la que viene este código) ---

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


# ---------------------------------------------------------------------------
# Cobertura multifuente y sujeto colectivo (v6, experimentos 2 y 3 del equipo)
# ---------------------------------------------------------------------------

def fuentes_relevantes(caso, k=3):
    """Las k fuentes más relevantes, de forma determinista: la que cita el LLM
    y las de mayor similitud y respaldo NLI en oraciones y fragmentos."""
    fuentes = caso.get("fuentes") or []
    senales = caso.get("senales") or {}
    items = (senales.get("oraciones") or []) + (senales.get("fragmentos") or [])
    puntos = []
    for i in range(len(fuentes)):
        sim = max([(it.get("similitud_por_fuente") or [])[i] for it in items
                   if i < len(it.get("similitud_por_fuente") or [])] or [0.0])
        resp = max([(it.get("respaldo_por_fuente") or [])[i] for it in items
                    if i < len(it.get("respaldo_por_fuente") or [])] or [0.0])
        citada = any((it.get("llm") or {}).get("fuente") == i for it in items)
        puntos.append((0.4 * sim + 0.4 * resp + 0.2 * citada, -i))
    return sorted(range(len(fuentes)), key=lambda i: puntos[i], reverse=True)[:k]


def cobertura_multifuente(caso):
    """Qué parte de las palabras de la respuesta cubre la mejor fuente sola, la
    mejor pareja y las tres más relevantes juntas. Una respuesta fiel que
    combina fuentes tiene cobertura baja con una sola y alta con la unión;
    así el modelo puede no castigar el NLI bajo de esos casos. (Cobertura
    léxica no es prueba de fidelidad: el modelo decide cuánto pesa.)"""
    fuentes = caso.get("fuentes") or []
    rr = raices_contenido(caso.get("respuesta") or "")
    vacio = dict(cobertura_mejor_fuente=0.0, cobertura_union_2=0.0, cobertura_union_3=0.0,
                 ganancia_cobertura_2=0.0, ganancia_cobertura_3=0.0, fuentes_complementarias=0.0,
                 llm_ok_nli_bajo_cobertura_alta=0.0, nli_bajo_cobertura_union_alta=0.0)
    if not rr or not fuentes:
        return vacio
    conjuntos = [raices_contenido(fuentes[i]) & rr for i in fuentes_relevantes(caso, 3)]
    individual = [len(c) / len(rr) for c in conjuntos]
    mejor = max(individual)
    u2 = max((len(conjuntos[a] | conjuntos[b]) / len(rr) for a in range(len(conjuntos))
              for b in range(a + 1, len(conjuntos))), default=mejor)
    u3 = len(set().union(*conjuntos)) / len(rr)
    complementarias = sum(1 for i, c in enumerate(conjuntos)
                          if c - set().union(*(o for j, o in enumerate(conjuntos) if j != i)))
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    fragmentos = senales.get("fragmentos") or []
    llm_ok = bool(oraciones) and all((o.get("llm") or {}).get("veredicto") == "respaldada" for o in oraciones)
    nli_bajo = min([fr.get("respaldo_todas", 0.0) for fr in fragmentos] or [0.0]) < 0.3
    return dict(cobertura_mejor_fuente=mejor, cobertura_union_2=u2, cobertura_union_3=u3,
                ganancia_cobertura_2=max(0.0, u2 - mejor), ganancia_cobertura_3=max(0.0, u3 - mejor),
                fuentes_complementarias=float(min(complementarias, 3)),
                llm_ok_nli_bajo_cobertura_alta=float(llm_ok and nli_bajo and u3 >= 0.8),
                nli_bajo_cobertura_union_alta=float(nli_bajo and u3 >= 0.8 and u3 - mejor >= 0.15))


def sujeto_colectivo(caso):
    """Complemento del B4 de Candela: la respuesta usa un sujeto colectivo
    («los municipios», «las empresas») y alguna fuente relevante usa ese mismo
    colectivo (respaldado) o ninguna lo usa (sin respaldo)."""
    resp = normalizar(caso.get("respuesta") or "")
    fuentes = caso.get("fuentes") or []
    relevantes = [normalizar(fuentes[i]) for i in fuentes_relevantes(caso, 3)]
    colectivos = [m.group(2) for m in re.finditer(r"\b(los|las)\s+([a-zñ]{4,}(?:es|s))\b", resp)]
    if not colectivos:
        return dict(sujeto_colectivo_respaldado=0.0, sujeto_colectivo_sin_respaldo=0.0)
    respaldados = sum(1 for c in colectivos
                      if any(re.search(r"\b" + re.escape(c[:5]) + r"\w*s\b", f) for f in relevantes))
    return dict(sujeto_colectivo_respaldado=float(respaldados > 0),
                sujeto_colectivo_sin_respaldo=float(respaldados < len(colectivos)))


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

    # --- 11. Cobertura multifuente y sujeto colectivo (v6) ---
    f.update({"cob_" + k: v for k, v in cobertura_multifuente(caso).items()})
    f.update({"suj_" + k: v for k, v in sujeto_colectivo(caso).items()})

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
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas", "cb_sujeto_generalizado", "cb_sujeto_generalizado_confianza", "suj_sujeto_colectivo_respaldado", "suj_sujeto_colectivo_sin_respaldo", "cob_cobertura_mejor_fuente", "cob_cobertura_union_2", "cob_cobertura_union_3", "cob_ganancia_cobertura_2", "cob_ganancia_cobertura_3", "cob_fuentes_complementarias", "cob_llm_ok_nli_bajo_cobertura_alta", "cob_nli_bajo_cobertura_union_alta"]
INICIAL = -0.035456
TASA = 0.03
ARBOLES = [[[2,3,44,-2,-2,4,-2,-2,38,15,-2,-2,79,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.28450000286102295,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003123,0.414307,0.470507,1.998345,1.099628,0.284888,1.604819,0.622315,-0.180073,-0.253411,-1.233059,0.91899,0.161875,-0.839811,1.928435]],[[2,8,53,-2,-2,7,-2,-2,8,4,-2,-2,21,-2,-2],[0.9350000023841858,0.17599999904632568,0.5,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.07600000128149986,0.08900000154972076,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002279,0.392471,0.473255,1.961457,1.354385,0.308807,1.873355,1.016757,-0.179575,0.201896,1.698077,0.236824,-0.243696,-1.112258,1.31322]],[[2,53,3,-2,-2,5,-2,-2,8,4,-2,-2,38,-2,-2],[0.9249999821186066,0.5,0.032500000670552254,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.0885000005364418,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005243,0.378655,0.433529,1.882385,1.447144,0.177293,1.790555,0.235679,-0.17735,0.172542,1.357834,-0.310141,-0.24489,-1.239634,0.339933]],[[2,53,3,-2,-2,5,-2,-2,8,4,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.07600000128149986,0.08900000154972076,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.014926,0.363458,0.404343,1.824469,1.240612,0.185197,1.765024,0.344293,-0.184432,0.171923,1.818912,0.033104,-0.244665,-1.219269,0.261744]],[[2,53,3,-2,-2,9,-2,-2,15,38,-2,-2,21,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.011937,0.351036,0.400954,1.778373,1.258593,0.144476,-0.251527,1.215573,-0.174418,-0.215029,-1.141443,0.480591,0.260474,0.090816,1.947638]],[[2,7,3,-2,-2,44,-2,-2,8,4,-2,-2,54,-2,-2],[0.9350000023841858,0.007500000298023224,0.636000007390976,-2.0,-2.0,0.5,-2.0,-2.0,0.0885000005364418,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005667,0.355389,0.417928,1.740012,0.949285,0.253166,1.262961,-0.023696,-0.173311,0.164432,1.303573,-0.291771,-0.234814,-1.091372,1.200505]],[[2,53,3,-2,-2,5,-2,-2,8,44,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.1314999982714653,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007507,0.340971,0.38732,1.689185,1.003407,0.142233,1.639107,0.239575,-0.150136,0.147539,1.803687,0.087404,-0.212655,-1.109394,0.347225]],[[53,5,39,-2,-2,42,-2,-2,2,52,-2,-2,54,-2,-2],[0.5,0.5550000071525574,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.014669,0.327222,0.380997,1.608696,0.886331,0.005273,0.732687,-1.486328,-0.171394,-0.003342,0.301981,-1.721678,-0.294715,-1.328502,1.24649]],[[53,69,5,-2,-2,-2,38,21,-2,-2,79,-2,-2],[0.5,0.5,0.4529999941587448,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003461,0.308576,0.363511,1.603322,0.878585,-0.77189,-0.144842,-0.20373,-0.9879,1.445811,0.175266,-0.783322,1.93439]],[[2,44,3,-2,-2,7,-2,-2,38,15,-2,-2,79,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001534,0.32047,0.351099,1.638365,1.121669,0.162459,1.26935,-0.324008,-0.143023,-0.193187,-0.975604,1.000399,0.147634,-0.736822,1.800844]],[[53,69,3,-2,-2,-2,21,32,-2,-2,-2],[0.5,0.5,0.032500000670552254,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.001577,0.292338,0.334495,1.606141,0.982721,-0.744919,-0.138627,-0.173048,-0.907708,1.038308,1.499262]],[[2,5,81,-2,-2,-2,38,21,-2,-2,79,-2,-2],[0.9350000023841858,0.6689999997615814,0.1180555559694767,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000318,0.304805,0.329099,1.578486,1.164698,0.392307,-0.140372,-0.20188,-0.957554,1.43908,0.13157,-0.986917,1.710726]],[[2,53,3,-2,-2,9,-2,-2,8,4,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.33650000393390656,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001783,0.285005,0.317937,1.472526,0.742826,0.149298,-0.198574,1.093167,-0.129457,0.164551,1.24225,-0.190706,-0.18508,-0.972704,0.373445]],[[2,8,81,-2,-2,44,-2,-2,8,4,-2,-2,15,-2,-2],[0.9350000023841858,0.153499998152256,0.2792207896709442,-2.0,-2.0,0.5,-2.0,-2.0,0.09149999916553497,0.21649999916553497,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004174,0.268529,0.346245,1.55208,1.211353,0.184834,1.008355,0.057641,-0.131947,0.150647,1.322772,0.03217,-0.185984,-0.904941,0.873739]],[[2,8,7,-2,-2,80,-2,-2,38,21,-2,-2,79,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,-2.0,-2.0,0.7573529481887817,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[1.8e-05,0.274197,0.328576,1.483742,1.030261,0.212731,0.436331,1.164278,-0.128614,-0.181889,-0.901883,1.322452,0.118461,-0.858377,1.630539]],[[2,3,44,-2,-2,4,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.44350001215934753,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007116,0.26307,0.311146,1.453806,0.940293,0.178089,1.081877,0.005114,-0.130509,-0.187486,-0.96355,0.971045,0.143124,1.290451,-1.42512]],[[53,5,46,-2,-2,42,-2,-2,65,15,-2,-2,80,-2,-2],[0.5,0.5550000071525574,1.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004758,0.249705,0.302844,1.274828,1.888338,-0.019614,0.430336,-1.405139,-0.122457,-0.163351,-0.885072,0.954455,0.230085,-0.013362,1.750536]],[[2,44,3,-2,-2,5,-2,-2,8,4,-2,-2,76,-2,-2],[0.9350000023841858,0.5,0.25300000607967377,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003433,0.254303,0.278354,1.36732,0.858025,0.136894,1.519034,0.011637,-0.114307,0.1576,1.358107,-0.270286,-0.167767,-0.970728,0.163017]],[[2,5,42,-2,-2,1,-2,-2,31,15,-2,-2,-2],[0.9350000023841858,0.14750000089406967,0.44949495792388916,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002168,0.237943,0.279703,1.331253,0.551869,0.138472,-0.011039,1.207705,-0.109467,-0.134299,-0.719363,0.871709,1.408978]],[[53,69,46,-2,-2,-2,65,48,-2,-2,80,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001802,0.240438,0.273756,1.173681,1.821235,-0.398655,-0.10592,-0.144629,-0.794608,0.822452,0.232182,-0.039767,1.683462]],[[2,19,16,-2,-2,12,-2,-2,0,1,-2,-2,10,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.5444999933242798,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.3541666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003931,0.139687,0.181564,0.923648,-0.952093,-0.367642,-1.23611,-1.925675,-0.178045,-0.203365,-0.955765,2.001988,0.407687,1.583593,1.746227]],[[2,7,3,-2,-2,80,-2,-2,76,11,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.013500000350177288,-2.0,-2.0,0.7638888955116272,-2.0,-2.0,0.5,0.1180555559694767,-2.0,-2.0,0.1180555559694767,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002823,0.231243,0.277558,1.253905,1.430024,0.159493,-0.109567,1.086961,-0.104331,-0.160729,-0.979472,0.181575,0.072477,1.30065,-0.308007]],[[53,5,46,-2,-2,1,-2,-2,2,52,-2,-2,32,-2,-2],[0.5,0.6629999876022339,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003031,0.20808,0.25412,1.129955,1.754239,-0.055171,-0.738773,0.795832,-0.104153,0.027139,0.38378,-1.640241,-0.204983,-1.041592,0.822663]],[[2,8,2,-2,-2,76,-2,-2,38,8,-2,-2,79,-2,-2],[0.9350000023841858,0.15200000256299973,0.8250000178813934,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.05649999901652336,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004298,0.21367,0.263244,1.273299,1.512995,0.164006,0.53922,1.182595,-0.096414,-0.144514,0.540134,-0.799294,0.148895,-0.545491,1.565735]],[[2,8,2,-2,-2,9,-2,-2,8,4,-2,-2,79,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,-2.0,-2.0,0.9410000145435333,-2.0,-2.0,0.0755000002682209,0.9280000030994415,-2.0,-2.0,0.8708333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008857,0.206772,0.25681,1.26206,1.476962,0.156358,0.29699,1.002606,-0.106995,0.130438,0.968309,-0.358814,-0.146175,-0.92071,0.063731]],[[53,69,46,-2,-2,-2,21,64,-2,-2,78,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5357142984867096,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002271,0.214345,0.242992,1.114702,1.733955,-0.554154,-0.091455,-0.12019,-0.658147,1.297824,0.303878,0.548639,1.699694]],[[2,8,7,-2,-2,76,-2,-2,32,76,-2,-2,80,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003582,0.187354,0.241077,1.30055,0.673844,0.132709,0.371293,1.12703,-0.090813,-0.126773,-0.844044,0.301525,0.223464,0.089834,1.610052]],[[2,19,17,-2,-2,8,-2,-2,0,1,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11600000038743019,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004459,0.124065,0.166733,0.886909,-1.123577,-0.355598,-0.935444,-1.889688,-0.146499,-0.169941,-0.825949,1.884428,0.382248,1.568054,1.75207]],[[53,5,46,-2,-2,79,-2,-2,2,22,-2,-2,65,-2,-2],[0.5,0.6654999852180481,1.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.0068,0.190635,0.226366,1.086188,1.651419,-0.016163,-1.307577,0.578425,-0.080648,0.032766,0.38301,-1.62697,-0.176044,-0.940986,0.715245]],[[2,19,16,-2,-2,8,-2,-2,8,5,-2,-2,76,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7374999821186066,-2.0,-2.0,0.13199999928474426,0.07250000163912773,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007201,0.10613,0.150253,0.837038,-0.837314,-0.361193,-1.30479,-1.827798,-0.153961,0.08825,0.950458,-0.687983,-0.20272,-1.138097,-0.265505]],[[2,3,44,-2,-2,6,-2,-2,15,32,-2,-2,78,-2,-2],[0.9350000023841858,0.3439999967813492,0.5,-2.0,-2.0,0.9465000033378601,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002957,0.185867,0.207148,1.226727,0.496965,0.089166,0.047711,1.486307,-0.083375,-0.109547,-0.65189,0.91306,0.214668,-0.123013,1.529902]],[[2,8,2,-2,-2,9,-2,-2,76,32,-2,-2,42,-2,-2],[0.9350000023841858,0.17599999904632568,0.8250000178813934,-2.0,-2.0,0.945000022649765,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11324786394834518,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007267,0.172294,0.219527,1.211181,1.405796,0.122087,0.050105,0.942943,-0.089334,-0.144074,-0.832366,0.858701,0.094104,1.459007,-0.318328]],[[53,69,46,-2,-2,-2,48,32,-2,-2,78,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.006092,0.168777,0.194255,0.94777,1.682922,-0.772826,-0.087744,-0.120828,-0.711091,0.849571,0.214864,-0.326446,1.696776]],[[2,19,17,-2,-2,8,-2,-2,0,15,-2,-2,12,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2849999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.015322,0.116663,0.152656,0.847571,-1.143322,-0.337875,-1.286087,-1.754941,-0.118489,-0.13758,-0.769953,0.806248,0.338033,1.576806,1.353627]],[[2,7,8,-2,-2,1,-2,-2,38,15,-2,-2,79,-2,-2],[0.9350000023841858,0.007500000298023224,0.6410000026226044,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005341,0.163257,0.209572,1.302258,0.628514,0.092436,-0.042995,0.942183,-0.065985,-0.111435,-0.685237,0.899536,0.15268,-0.499247,1.523878]],[[53,69,46,-2,-2,-2,76,11,-2,-2,42,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.4166666716337204,-2.0,-2.0,0.1180555559694767,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001431,0.168525,0.192823,0.985874,1.579355,-0.261059,-0.076112,-0.126622,-0.795752,0.289134,0.097191,1.22006,-0.209971]],[[2,19,50,-2,-2,8,-2,-2,8,5,-2,-2,76,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.8070000112056732,-2.0,-2.0,0.08150000125169754,0.07400000095367432,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000544,0.091826,0.12869,0.737084,-1.193859,-0.294039,-1.021961,-1.756365,-0.120045,0.108693,1.024451,-0.651615,-0.159325,-1.017239,-0.036682]],[[19,2,17,-2,-2,8,-2,-2,82,55,-2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.23111110925674438,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.005195,0.034705,0.120669,0.725935,-1.255269,-0.081351,0.681385,-0.611061,-0.293275,-0.312401,-1.702548,-1.43194,-0.912556]],[[8,6,2,-2,-2,44,-2,-2,2,19,-2,-2,79,-2,-2],[0.08349999785423279,0.2889999896287918,0.4000000059604645,-2.0,-2.0,0.5,-2.0,-2.0,0.9750000238418579,0.5,-2.0,-2.0,0.8660714328289032,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004348,0.171555,0.259809,1.241009,1.399962,0.139278,1.141535,0.173954,-0.049463,0.030793,0.352189,-1.534467,-0.152161,-1.014251,-0.198261]],[[2,19,16,-2,-2,8,-2,-2,8,6,-2,-2,76,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.6270000040531158,-2.0,-2.0,0.04649999924004078,0.9165000021457672,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.008182,0.098833,0.133477,0.774758,-0.957142,-0.276762,-0.790602,-1.684599,-0.113817,0.180948,1.410439,0.039931,-0.14599,-0.933426,-0.031615]],[[2,19,79,-2,-2,8,-2,-2,8,46,-2,-2,80,-2,-2],[0.9600000083446503,0.5,0.6306818127632141,-2.0,-2.0,0.6270000040531158,-2.0,-2.0,0.05250000022351742,1.5,-2.0,-2.0,0.8708333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005643,0.087327,0.11704,-0.195571,0.828897,-0.254471,-0.722743,-1.642491,-0.108313,0.133033,0.021789,1.399223,-0.139701,-1.015957,-0.084057]],[[53,69,46,-2,-2,-2,70,21,-2,-2,79,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003015,0.142126,0.161755,0.871083,1.526905,-0.304478,-0.072548,-0.107772,-0.646618,1.267945,0.143168,-0.430407,1.656011]],[[2,7,53,-2,-2,37,-2,-2,79,8,-2,-2,2,-2,-2],[0.9350000023841858,0.001500000071246177,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.8708333373069763,0.07400000095367432,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007587,0.142106,0.191358,1.229024,1.152683,0.115954,0.316025,0.951383,-0.078272,-0.130034,0.280119,-0.819736,0.047546,0.936875,-0.53205]],[[2,8,44,-2,-2,9,-2,-2,31,21,-2,-2,-2],[0.9350000023841858,0.153499998152256,0.5,-2.0,-2.0,0.9679999947547913,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00267,0.141067,0.18008,1.206731,1.305699,0.100357,0.021306,1.001399,-0.067204,-0.087742,-0.532807,1.104394,1.135276]],[[8,4,2,-2,-2,44,-2,-2,79,2,-2,-2,38,-2,-2],[0.1314999982714653,0.060499999672174454,0.42500001192092896,-2.0,-2.0,0.5,-2.0,-2.0,0.7573529481887817,0.9350000023841858,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005569,0.141123,0.241507,1.201951,1.381507,0.100231,1.063885,-0.055415,-0.046156,-0.147794,0.102459,-0.954229,0.039915,-0.097117,1.618696]],[[53,46,2,-2,-2,1,-2,-2,21,38,-2,-2,79,-2,-2],[0.5,1.5,0.9850000143051147,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.6904762089252472,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.009613,0.127156,0.10741,0.810502,-1.068731,0.313002,1.232882,1.74477,-0.072682,-0.096192,-0.659346,0.509194,0.255387,0.607731,1.482134]],[[53,3,46,-2,-2,7,-2,-2,48,70,-2,-2,7,-2,-2],[0.5,0.25300000607967377,1.5,-2.0,-2.0,0.8955000042915344,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.031000000424683094,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006373,0.129123,0.161223,0.965529,1.483596,-0.007615,0.374278,-1.36628,-0.050793,-0.080587,-0.599403,0.564358,0.234302,1.485283,0.333798]],[[53,69,46,-2,-2,-2,79,2,-2,-2,70,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.6449579894542694,0.9249999821186066,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.004595,0.125517,0.151767,0.898588,1.44198,-0.746509,-0.06534,-0.191414,0.795792,-1.090245,-0.016491,-0.322194,1.106888]],[[19,2,17,-2,-2,8,-2,-2,9,-2,8,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.6550000011920929,-2.0,0.9424999952316284,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.008135,0.033945,0.09659,0.636485,-1.097769,-0.070554,0.756267,-0.599794,-0.240465,-0.886873,-0.262531,-1.494352,-1.303353]],[[2,5,45,-2,-2,1,-2,-2,79,21,-2,-2,2,-2,-2],[0.9350000023841858,0.023000000044703484,0.7999999821186066,-2.0,-2.0,0.5,-2.0,-2.0,0.8708333373069763,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007123,0.112984,0.15447,1.167465,0.402684,0.078914,0.037607,0.889728,-0.062345,-0.111504,-0.651049,0.799072,0.055608,0.871819,-0.444898]],[[19,2,16,-2,-2,8,-2,-2,4,2,-2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.9664999842643738,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00133,0.029301,0.103853,0.698839,-1.162345,-0.071859,0.686278,-0.596803,-0.241456,-0.26246,-1.537438,-1.313877,-0.878245]],[[2,8,44,-2,-2,9,-2,-2,32,15,-2,-2,42,-2,-2],[0.9350000023841858,0.153499998152256,0.5,-2.0,-2.0,0.4490000009536743,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002932,0.117746,0.149503,1.13412,1.269724,0.086218,-0.648395,0.629047,-0.062437,-0.088916,-0.596381,0.725004,0.230401,1.61993,0.296783]],[[19,2,79,-2,-2,8,-2,-2,9,-2,2,-2,-2],[0.5,0.9750000238418579,0.8090909123420715,-2.0,-2.0,0.1339999958872795,-2.0,-2.0,0.6200000047683716,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004515,0.021631,0.083638,0.085847,0.965354,-0.081568,0.486202,-0.672962,-0.223539,-0.85201,-0.246176,-1.504476,-1.274502]],[[19,2,79,-2,-2,8,-2,-2,12,81,-2,-2,-2],[0.5,0.9750000238418579,0.8090909123420715,-2.0,-2.0,0.05250000022351742,-2.0,-2.0,0.6444999873638153,0.16333333402872086,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002,0.027237,0.089461,0.077066,1.099768,-0.071276,0.72814,-0.560747,-0.217417,-0.192193,-1.315604,-0.817143,-1.563035]],[[19,2,78,-2,-2,8,-2,-2,9,-2,2,-2,-2],[0.5,0.9750000238418579,0.4226190447807312,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.5785000026226044,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.0082,0.032563,0.089029,-0.49814,0.706559,-0.064665,0.753575,-0.600903,-0.221308,-0.615279,-0.250933,-1.583882,-1.251409]],[[2,3,77,-2,-2,4,-2,-2,15,76,-2,-2,9,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.3410000056028366,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9814999997615814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000334,0.10531,0.131498,0.948015,1.295511,0.055045,0.826509,-0.393538,-0.049385,-0.074062,-0.663031,0.326682,0.211683,0.242813,1.664905]],[[19,2,37,-2,-2,79,-2,-2,12,80,-2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.8708333373069763,-2.0,-2.0,0.6349999904632568,0.7416666746139526,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001103,0.01994,0.118059,0.629086,1.059401,-0.028382,-0.416446,0.546322,-0.212844,-0.192159,-1.354859,-0.995817,-1.536237]],[[19,2,8,-2,-2,38,-2,-2,9,-2,2,-2,-2],[0.5,0.9350000023841858,0.9359999895095825,-2.0,-2.0,0.5,-2.0,-2.0,0.6190000176429749,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002339,0.024199,0.120293,0.815735,1.126535,-0.023085,-0.315573,0.696207,-0.203372,-0.663878,-0.228155,-1.478756,-1.23727]],[[2,55,50,-2,-2,-2,51,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000393,0.066512,0.089641,0.612099,-1.069854,-1.439647,-0.085122,-0.104026,-0.664165,1.765328,0.998162]],[[19,2,79,-2,-2,46,-2,-2,3,-2,2,-2,-2],[0.5,0.9750000238418579,0.8153409063816071,-2.0,-2.0,1.5,-2.0,-2.0,0.007500000298023224,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001481,0.022529,0.073192,0.031018,0.960163,-0.064321,-0.558575,0.817885,-0.201618,-0.819121,-0.228775,-1.494272,-1.247437]],[[2,5,69,-2,-2,-2,76,11,-2,-2,4,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,-2.0,-2.0,-2.0,0.5,0.4166666716337204,-2.0,-2.0,0.08100000023841858,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004441,0.103876,0.115531,0.842755,1.083862,-0.036921,-0.043915,-0.083637,-0.65912,0.39065,0.077532,1.222962,-0.31873]],[[2,22,17,-2,-2,-2,18,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.002714,0.05415,0.0752,0.553238,-1.189204,-1.301876,-0.067221,-0.087204,-0.585311,1.701013,1.033499]],[[2,39,77,-2,-2,-2,31,76,-2,-2,-2],[0.9350000023841858,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.001829,0.088894,0.096095,0.635859,1.268906,-0.706586,-0.041588,-0.060811,-0.594149,0.412388,1.362746]],[[21,53,46,-2,-2,76,-2,-2,78,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4833333343267441,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.006729,-0.008502,0.088948,0.526506,1.380634,-0.058783,-0.563754,0.458198,0.260483,0.3693,1.562237]],[[19,5,16,-2,-2,2,-2,-2,22,6,-2,-2,83,-2,-2],[0.5,0.8260000050067902,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,0.5,0.7094999849796295,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003771,0.016694,0.055464,0.457603,-1.059988,-0.089015,0.232352,-0.870334,-0.187303,-0.217654,-1.241278,-1.498788,-0.160621,-1.27878,-0.713747]],[[8,4,78,-2,-2,1,-2,-2,79,21,-2,-2,63,-2,-2],[0.1314999982714653,0.8700000047683716,0.2899159789085388,-2.0,-2.0,0.5,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006259,0.093226,0.156691,0.14142,1.083664,0.031524,-0.389832,1.159712,-0.043986,-0.128652,-0.881143,0.70539,0.029142,-0.020624,1.651285]],[[19,2,79,-2,-2,46,-2,-2,55,69,-2,-2,5,-2,-2],[0.5,0.9750000238418579,0.6339712738990784,-2.0,-2.0,1.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.24950000643730164,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003303,0.023753,0.067029,-0.376524,0.657829,-0.049702,-0.4786,0.857195,-0.189592,-0.217349,-1.426788,-1.210748,-0.16474,-1.271978,-1.183394]],[[8,19,4,-2,-2,-2,43,76,-2,-2,78,-2,-2],[0.1314999982714653,0.5,0.8700000047683716,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5916666686534882,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.00389,0.095975,0.11537,1.080715,0.443813,-1.094005,-0.043945,-0.063111,-0.640716,0.374863,0.207021,0.145819,1.60126]],[[16,79,2,-2,-2,2,-2,-2,9,-2,-2],[0.5,0.8153409063816071,0.9350000023841858,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000572,0.014086,-0.036945,0.602435,-0.477278,0.084693,0.890031,-0.305385,-0.257501,-1.908565,-0.900105]],[[19,2,39,-2,-2,80,-2,-2,9,-2,2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.8708333373069763,-2.0,-2.0,0.6200000047683716,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001641,0.020339,0.105868,0.944574,-0.4452,-0.023457,-0.385776,0.400447,-0.172007,-0.622342,-0.201784,-1.513238,-1.175071]],[[53,46,7,-2,-2,8,-2,-2,54,65,-2,-2,-2],[0.5,1.5,0.007500000298023224,-2.0,-2.0,0.06750000081956387,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.012625,0.072265,0.051513,0.968119,-0.155328,0.284487,1.33446,1.801089,-0.053675,-0.072639,-0.556389,0.832661,1.148391]],[[19,79,8,-2,-2,50,-2,-2,3,-2,55,-2,-2],[0.5,0.6449579894542694,0.0755000002682209,-2.0,-2.0,0.5,-2.0,-2.0,0.007500000298023224,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002404,0.0191,-0.073152,0.639557,-0.77815,0.053768,0.423106,-1.517429,-0.159046,-0.650635,-0.184668,-1.44234,-1.187761]],[[8,79,-2,5,-2,-2,76,43,-2,-2,12,-2,-2],[0.1314999982714653,0.3457317054271698,-2.0,0.07349999994039536,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.38849999010562897,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.001671,0.091334,-0.983976,0.100332,0.907181,0.126736,-0.037969,-0.073044,-0.60282,0.815173,0.070236,-0.56537,0.664762]],[[55,78,53,-2,-2,43,-2,-2,5,-2,7,-2,-2],[0.5,0.5505050718784332,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.15850000083446503,-2.0,0.029000001028180122,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002716,0.017522,-0.058055,0.565483,-0.630175,0.055274,0.237557,1.329884,-0.181585,-1.478057,-0.145556,-1.243259,-1.064322]],[[2,22,17,-2,-2,-2,38,46,-2,-2,79,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,1.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002927,0.048968,0.067108,0.54892,-1.17212,-1.389994,-0.061877,-0.091077,-0.74411,0.830876,0.122749,-0.413391,1.535633]],[[8,79,-2,4,-2,-2,76,21,-2,-2,42,-2,-2],[0.1314999982714653,0.3457317054271698,-2.0,0.8700000047683716,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002594,0.083114,-1.006508,0.091177,0.958654,0.341246,-0.036537,-0.069432,-0.53686,0.960436,0.065899,0.967486,-0.209426]],[[43,64,53,-2,-2,-2,78,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.007096,-0.007195,-0.01941,0.56293,-0.406709,1.352541,0.195335,-0.064942,1.392024]],[[16,2,79,-2,-2,10,-2,-2,9,-2,-2],[0.5,0.9750000238418579,0.7928571403026581,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.010303,0.001957,0.044393,-0.141365,0.778594,-0.067962,-0.690861,0.252193,-0.223588,-1.55381,-0.873158]],[[21,2,22,-2,-2,1,-2,-2,48,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.011819,-0.001172,0.045363,0.419055,-1.173171,-0.062385,-0.462113,1.602699,0.237993,0.829224,1.445474]],[[19,17,16,-2,-2,79,-2,-2,3,-2,55,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.612500011920929,-2.0,-2.0,0.007500000298023224,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.007918,0.02452,0.037705,0.330406,-1.044769,-0.185995,-0.20027,-1.515591,-0.152417,-0.587508,-0.176183,-1.410941,-1.173172]],[[8,5,7,-2,-2,-2,76,21,-2,-2,42,-2,-2],[0.1314999982714653,0.5539999902248383,0.16499999910593033,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11324786394834518,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00095,0.076715,0.08732,0.57287,1.320091,-0.574154,-0.029483,-0.066086,-0.517085,0.943556,0.078338,1.006825,-0.046647]],[[16,19,79,-2,-2,9,-2,-2,12,-2,-2],[0.5,0.5,0.6504524946212769,-2.0,-2.0,0.6200000047683716,-2.0,-2.0,0.5520000159740448,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00252,0.014279,0.031874,-0.323858,0.408102,-0.146773,-0.540152,-1.305761,-0.221525,-0.819412,-1.63295]],[[8,4,3,-2,-2,7,-2,-2,79,48,-2,-2,70,-2,-2],[0.09149999916553497,0.8700000047683716,0.021499999798834324,-2.0,-2.0,0.02650000061839819,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001323,0.091499,0.151768,0.95904,1.404642,0.034728,0.697079,-0.795829,-0.033484,-0.096613,-0.722139,0.595429,0.021771,-0.133635,1.47702]],[[15,5,78,-2,-2,76,-2,-2,8,78,-2,-2,-2],[0.5,0.08300000056624413,0.8452380895614624,-2.0,-2.0,0.5,-2.0,-2.0,0.37450000643730164,0.5505050718784332,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.005529,-0.023647,0.048631,0.159685,1.520601,-0.066869,-0.661746,0.16625,0.132507,0.062737,-0.406589,0.88849,1.342716]],[[38,53,46,-2,-2,21,-2,-2,79,70,-2,-2,53,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001177,-0.020802,0.056459,0.36857,1.319256,-0.056949,-0.466202,1.028969,0.105253,-0.035136,1.073041,-0.583565,0.217906,0.963738,1.491895]],[[32,53,39,-2,-2,76,-2,-2,12,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.408500000834465,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-3.6e-05,-0.015401,0.059491,0.649729,-0.909591,-0.05224,-0.550607,0.355536,0.147443,1.70713,0.087515,-0.219968,1.224137]],[[2,22,16,-2,-2,82,-2,-2,21,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001281,0.047126,0.067281,0.550078,-1.010323,-0.192469,-1.080179,-1.440787,-0.060545,-0.074474,-0.557482,1.554002,1.08428]],[[1,21,76,-2,-2,-2,2,45,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.4749999940395355,0.4455000013113022,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.006947,-0.028554,-0.041504,-0.458605,0.232935,1.039129,0.084883,0.073356,1.077083,0.178536,1.487783]],[[17,2,16,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.12132352963089943,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.006865,0.004615,0.046533,0.435382,-1.318077,-0.051204,0.158737,-0.630238,-0.206226,-1.577554,-0.922998]],[[19,5,17,-2,-2,2,-2,-2,9,-2,2,-2,-2],[0.5,0.9715000092983246,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.6209999918937683,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001894,0.016844,0.036593,0.323914,-0.887468,-0.096218,0.127529,-1.105852,-0.138061,-0.465066,-0.158143,-1.412949,-1.13114]],[[22,17,10,-2,-2,80,-2,-2,52,-2,5,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.6201923191547394,-2.0,-2.0,0.5,-2.0,0.26350000500679016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000546,0.011444,0.021784,-0.10969,0.555144,-0.163984,0.094071,-1.496875,-0.150198,-1.240789,-0.110032,-1.070267,-1.119647]],[[19,5,17,-2,-2,2,-2,-2,12,6,-2,-2,-2],[0.5,0.6755000054836273,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.6359999775886536,0.9239999949932098,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000937,0.015206,0.043618,0.385752,-0.95485,-0.052436,0.231155,-0.966119,-0.140049,-0.118998,-0.886009,-1.246934,-1.42225]],[[48,53,46,-2,-2,32,-2,-2,12,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5394999980926514,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.007524,-0.004417,0.057241,0.397056,1.311251,-0.037519,-0.377879,0.781131,0.184426,1.343099,-0.053114]],[[80,8,4,-2,-2,21,-2,-2,69,2,-2,-2,4,-2,-2],[0.8708333373069763,0.09149999916553497,0.5805000066757202,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.026499999687075615,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002716,-0.037482,0.061102,0.869958,-0.026927,-0.073128,-0.577553,0.615835,0.064209,0.026126,0.589077,-0.695001,0.20143,1.703615,0.635141]],[[19,79,2,-2,-2,69,-2,-2,12,5,-2,-2,-2],[0.5,0.8153409063816071,0.9350000023841858,-2.0,-2.0,1.5,-2.0,-2.0,0.6444999873638153,0.09849999845027924,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.006843,0.020771,-0.017904,0.729817,-0.367236,0.072718,0.209668,1.254865,-0.115706,-0.096349,-1.004334,-0.836089,-1.412319]],[[1,76,11,-2,-2,42,-2,-2,46,3,-2,-2,-2],[0.5,0.5,0.4166666716337204,-2.0,-2.0,0.12132352963089943,-2.0,-2.0,1.5,0.4814999997615814,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00577,-0.015522,-0.042907,-0.490126,0.45506,0.071554,1.221269,-0.104061,0.095853,0.085032,0.988844,1.054081,1.524613]],[[19,16,10,-2,-2,9,-2,-2,12,5,-2,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.6444999873638153,0.09849999845027924,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000319,0.013708,0.024066,-0.100128,0.622446,-0.155167,-1.475608,0.034674,-0.122042,-0.100632,-0.99256,-0.864698,-1.38799]],[[17,19,2,-2,-2,55,-2,-2,9,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.5,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003298,0.006914,0.021775,0.419655,-0.292402,-0.130718,-1.291885,-1.071919,-0.194794,-1.53397,-0.850628]],[[78,8,5,-2,-2,81,-2,-2,48,65,-2,-2,-2],[0.5505050718784332,0.0625,0.024500000290572643,-2.0,-2.0,0.3798076957464218,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000459,-0.058907,0.074648,1.051981,-0.431564,-0.093528,-0.718152,0.742584,0.030322,0.015132,-0.00324,1.093335,1.306118]],[[2,4,8,-2,-2,3,-2,-2,21,79,-2,-2,-2],[0.9350000023841858,0.44200000166893005,0.9690000116825104,-2.0,-2.0,0.06600000336766243,-2.0,-2.0,0.5,0.8708333373069763,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000205,0.058376,0.099259,0.854989,1.110744,0.026197,0.788548,-1.100287,-0.027469,-0.039167,-0.45295,0.226618,0.976544]],[[19,78,53,-2,-2,16,-2,-2,12,6,-2,-2,-2],[0.5,0.4686274528503418,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.6444999873638153,0.9239999949932098,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.003012,0.009481,-0.062959,0.665759,-0.72387,0.032486,0.333287,-0.928483,-0.123085,-0.105362,-0.854863,-1.216202,-1.392586]],[[11,79,1,-2,-2,38,-2,-2,4,4,-2,-2,9,-2,-2],[0.20714285969734192,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.16899999976158142,0.03799999877810478,-2.0,-2.0,0.9449999928474426,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001243,-0.030338,-0.082653,-0.716683,0.744281,0.015507,-0.102474,1.214024,0.062353,0.122294,0.873352,1.229921,0.03385,-0.121016,0.771269]],[[8,9,5,-2,-2,8,-2,-2,79,1,-2,-2,70,-2,-2],[0.0925000011920929,0.94200000166893,0.051500000059604645,-2.0,-2.0,0.06149999983608723,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000934,0.071924,0.035846,0.71086,-0.666002,0.124217,0.817437,1.729701,-0.023936,-0.082962,-0.687622,0.441575,0.028253,-0.037687,1.308565]],[[79,53,46,-2,-2,2,-2,-2,65,55,-2,-2,8,-2,-2],[0.6449579894542694,0.5,1.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1679999977350235,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000767,-0.070005,0.046437,-0.225704,1.349499,-0.112734,0.854821,-0.913523,0.027557,0.012993,0.193445,-1.310968,0.204336,1.125305,1.456058]],[[2,7,69,-2,-2,0,-2,-2,15,32,-2,-2,9,-2,-2],[0.9350000023841858,0.007500000298023224,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9774999916553497,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006061,0.051922,0.078383,0.834278,1.154097,0.008079,0.295276,-0.928946,-0.032728,-0.045395,-0.417848,0.707991,0.121399,-0.384462,1.251369]],[[17,22,79,-2,-2,19,-2,-2,79,-2,-2],[0.5,0.5,0.6449579894542694,-2.0,-2.0,0.5,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005293,0.01478,0.027861,-0.362823,0.406449,-0.12715,-1.411867,-0.96849,-0.167759,-0.336609,-1.475656]],[[2,55,17,-2,-2,-2,0,1,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1015000008046627,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003679,0.031299,0.042826,0.399154,-0.933153,-1.276826,-0.048753,-0.058069,-0.46455,1.452629,0.171014,1.187221,1.141264]],[[48,32,2,-2,-2,78,-2,-2,7,-2,-2],[0.5,0.5,0.9249999821186066,-2.0,-2.0,0.4833333343267441,-2.0,-2.0,0.031000000424683094,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003001,-0.006667,-0.018696,0.482775,-0.339326,0.116693,-0.205813,1.163159,0.148435,1.464014,-0.150293]],[[79,1,78,-2,-2,46,-2,-2,2,7,-2,-2,8,-2,-2],[0.6449579894542694,0.5,0.4833333343267441,-2.0,-2.0,1.5,-2.0,-2.0,0.9850000143051147,0.9309999942779541,-2.0,-2.0,0.05250000022351742,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003977,-0.059495,-0.084825,-0.843914,-0.168613,0.101932,0.421438,1.487538,0.027792,0.057819,0.314221,1.396162,-0.05908,0.598469,-0.57902]],[[64,21,1,-2,-2,78,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4833333343267441,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002419,-0.010024,-0.019082,-0.274835,0.703969,0.139021,-0.309405,1.358614,1.022513]],[[5,8,17,-2,-2,76,-2,-2,79,21,-2,-2,69,-2,-2],[0.04749999940395355,0.4650000035762787,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.8708333373069763,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006451,0.044619,0.076581,0.805381,-0.614604,-0.049503,-0.78689,0.721463,-0.02967,-0.06678,-0.590937,0.871194,0.040417,-0.222781,0.785063]],[[21,17,5,-2,-2,9,-2,-2,43,-2,-2],[0.5,0.5,0.024500000290572643,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002051,-0.010243,-0.002431,0.539505,-0.174861,-0.162762,-1.46414,-0.618424,0.151567,0.448146,1.262306]],[[80,2,82,-2,-2,1,-2,-2,76,2,-2,-2,3,-2,-2],[0.8708333373069763,0.9350000023841858,0.2928571552038193,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.9750000238418579,-2.0,-2.0,0.19099999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007832,-0.035247,0.043782,0.272998,1.292927,-0.068285,-0.526721,1.481182,0.047546,0.009313,0.56888,-0.645231,0.154493,0.175358,1.585821]],[[16,10,76,-2,-2,17,-2,-2,9,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.008659,0.000991,-0.03155,-0.478321,0.352014,0.044851,0.526506,-1.15778,-0.166321,-1.409618,-0.56767]],[[21,8,4,-2,-2,78,-2,-2,78,-2,-2],[0.5,0.0925000011920929,0.8745000064373016,-2.0,-2.0,0.5419580638408661,-2.0,-2.0,0.47727273404598236,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000138,-0.009291,0.055967,0.807989,0.134482,-0.031925,-0.649711,-0.008925,0.151253,-0.060933,1.331397]],[[80,21,8,-2,-2,6,-2,-2,69,2,-2,-2,4,-2,-2],[0.8708333373069763,0.5,0.09699999913573265,-2.0,-2.0,0.8425000011920929,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.0042,-0.020638,-0.031025,0.336604,-0.415262,0.130727,0.451419,1.229755,0.053098,0.01548,0.475992,-0.629459,0.176478,1.461675,0.457085]],[[64,76,53,-2,-2,3,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8735000193119049,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.008234,0.000288,-0.01987,0.526827,-0.357739,0.062103,0.206269,1.209909,1.308975]],[[21,2,39,-2,-2,38,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7685000002384186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00366,-0.004939,0.048357,0.606036,-1.120609,-0.030582,-0.387013,0.460046,0.150175,0.201917,1.347357]],[[55,78,1,-2,-2,21,-2,-2,19,-2,5,-2,-2],[0.5,0.4833333343267441,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,0.24950000643730164,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[9.2e-05,0.010048,-0.053669,-0.591142,1.148821,0.030734,0.156724,1.310088,-0.118809,-1.391678,-0.079467,-1.043154,-1.067874]],[[38,16,10,-2,-2,9,-2,-2,78,-2,42,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.3550420254468918,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.002335,-0.017882,-0.006973,-0.279872,0.334638,-0.169742,-1.684831,-0.53572,0.077798,-1.023444,0.111076,1.167835,-0.053544]],[[16,17,19,-2,-2,9,-2,-2,8,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9849999845027924,-2.0,-2.0,0.19550000131130219,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001503,0.006628,0.014734,0.200017,-1.072941,-0.141838,-1.444843,0.024041,-0.12978,0.161426,-1.529478]],[[11,76,48,-2,-2,7,-2,-2,7,3,-2,-2,3,-2,-2],[0.36666667461395264,0.5,0.5,-2.0,-2.0,0.7384999990463257,-2.0,-2.0,0.005499999970197678,0.004500000039115548,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004794,-0.015721,-0.04327,-0.427075,0.551707,0.066923,0.749691,-0.254901,0.064941,0.101283,0.791825,1.316724,0.022498,1.074626,-0.138475]],[[11,79,1,-2,-2,70,-2,-2,17,19,-2,-2,-2],[0.1180555559694767,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001675,-0.02433,-0.075163,-0.676276,0.473553,0.017349,-0.0664,1.363305,0.044557,0.054727,0.610437,-0.812682,-0.838005]],[[21,17,19,-2,-2,9,-2,-2,43,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002288,-0.005538,0.001726,0.101346,-1.084596,-0.145785,-1.440868,-0.684839,0.145853,0.524899,1.257713]],[[38,16,2,-2,-2,9,-2,-2,42,6,-2,-2,70,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.2899159789085388,0.9855000078678131,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004287,-0.020967,-0.010436,0.473116,-0.23829,-0.178212,-1.681264,-0.921922,0.073981,0.126941,1.173836,0.154416,-0.105388,0.841587,-1.375526]],[[19,79,7,-2,-2,69,-2,-2,12,8,-2,-2,-2],[0.5,0.8153409063816071,0.029000000096857548,-2.0,-2.0,1.5,-2.0,-2.0,0.6444999873638153,0.42249999940395355,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.004184,0.007041,-0.0235,0.200349,-0.643682,0.049403,0.146868,1.049284,-0.10196,-0.083234,-1.241144,-0.824218,-1.366474]],[[78,2,45,-2,-2,1,-2,-2,69,12,-2,-2,42,-2,-2],[0.5505050718784332,0.9249999821186066,0.5305000245571136,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.2575000077486038,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000928,-0.041397,0.04405,0.726169,-1.31037,-0.074292,-0.636045,1.405064,0.023496,0.006284,1.200129,-0.035783,0.141352,1.493815,0.072177]],[[55,2,17,-2,-2,46,-2,-2,19,-2,5,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.5,-2.0,0.19750000536441803,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000931,0.009593,0.033815,0.34921,-0.769469,-0.03107,-0.388102,0.705617,-0.109926,-1.390671,-0.069149,-1.002806,-1.058343]],[[15,38,53,-2,-2,42,-2,-2,9,78,-2,-2,2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,0.9835000038146973,0.5584415793418884,-2.0,-2.0,0.9099999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002362,-0.009053,-0.026489,0.320268,-0.383995,0.068009,1.076079,-0.589588,0.094013,0.036719,-0.571716,0.936524,0.172077,0.894101,1.443996]],[[2,78,2,-2,-2,8,-2,-2,46,85,-2,-2,78,-2,-2],[0.9750000238418579,0.5419580638408661,0.9249999821186066,-2.0,-2.0,0.9490000009536743,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.464102566242218,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005268,0.028635,-0.034503,0.300209,-0.535552,0.062099,0.292451,1.052653,-0.034871,-0.054261,-0.35067,-1.146148,0.108016,0.166134,1.533108]],[[42,70,21,-2,-2,80,-2,-2,3,8,-2,-2,10,-2,-2],[0.343137264251709,0.5,0.5,-2.0,-2.0,0.7573529481887817,-2.0,-2.0,0.013500000350177288,0.06300000101327896,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000782,0.014851,0.001133,-0.067665,1.142549,0.122774,-0.125472,1.311876,-0.063625,0.032857,0.947869,-0.419479,-0.112695,-0.883467,-1.507492]],[[16,2,39,-2,-2,15,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9580000042915344,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001119,0.008442,0.059604,0.844783,-1.190345,-0.014071,-0.202412,0.744549,-0.144024,-1.445594,-0.254735]],[[32,1,48,-2,-2,46,-2,-2,6,6,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.9694999754428864,0.23400000482797623,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-9.5e-05,-0.011451,-0.029596,-0.338007,0.650526,0.060651,0.782695,1.406265,0.113422,0.138111,0.872931,1.200344,0.155003]],[[16,19,17,-2,-2,12,-2,-2,4,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.6504999995231628,-2.0,-2.0,0.9105000197887421,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004501,0.012867,0.023848,0.258474,-0.950314,-0.092464,-0.892458,-1.409117,-0.126666,-1.464922,0.114862]],[[55,5,78,-2,-2,9,-2,-2,52,-2,7,-2,-2],[0.5,0.992499977350235,0.4686274528503418,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,0.5,-2.0,0.029000001028180122,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003165,0.0114,0.020596,-0.256749,0.31154,-0.099748,-0.357738,-1.230843,-0.097066,-1.352794,-0.057688,-1.106456,-0.873806]],[[80,5,8,-2,-2,39,-2,-2,30,69,-2,-2,-2],[0.8708333373069763,0.024500000290572643,0.4035000056028366,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000831,-0.022973,0.039522,0.682903,-0.274113,-0.046801,-0.313193,-1.475895,0.04107,0.023258,-0.194897,0.566673,1.545508]],[[7,22,78,-2,-2,-2,16,78,-2,-2,-2],[0.007500000298023224,0.5,0.4686274528503418,-2.0,-2.0,-2.0,0.5,0.5505050718784332,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.00424,0.036268,0.04765,-0.153625,0.688257,-1.065964,-0.033266,-0.022865,-0.684909,0.012463,-1.490529]],[[38,21,53,-2,-2,43,-2,-2,79,12,-2,-2,53,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.5370000004768372,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000339,-0.013621,-0.021708,0.380333,-0.350618,0.121927,0.173732,1.24853,0.069622,-0.016922,0.335839,-0.673689,0.137933,0.762266,1.406957]],[[5,19,16,-2,-2,12,-2,-2,2,-2,80,-2,-2],[0.9835000038146973,0.5,0.5,-2.0,-2.0,0.6444999873638153,-2.0,-2.0,0.9850000143051147,-2.0,0.8452380895614624,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003443,0.00832,0.020309,0.24024,-0.817019,-0.085857,-0.828544,-1.386571,-0.090556,-0.031207,-0.125992,-1.147598,-1.177162]],[[43,8,4,-2,-2,63,-2,-2,9,-2,-2],[0.5,0.17599999904632568,0.9259999990463257,-2.0,-2.0,0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002064,-0.006711,0.045848,0.720182,0.045045,-0.031094,-0.342833,0.681737,0.114054,0.191488,1.246522]],[[79,8,39,-2,-2,15,-2,-2,2,5,-2,-2,69,-2,-2],[0.8153409063816071,0.39249999821186066,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9850000143051147,0.901500016450882,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005209,-0.034121,0.015921,0.261936,-0.964617,-0.074015,-0.738166,0.541855,0.033135,0.062315,0.389704,1.36702,-0.039885,-0.648446,0.713112]],[[16,76,8,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.6304999887943268,-2.0,-2.0,0.0669642873108387,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001459,0.005991,-0.012007,0.272991,-0.450749,0.062247,1.264255,0.073777,-0.139409,-1.305523,-0.786189]],[[38,1,21,-2,-2,2,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.31129032373428345,0.9855000078678131,-2.0,-2.0,0.4270000010728836,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001744,-0.014447,-0.029886,-0.308269,0.834392,0.051828,0.714496,1.398254,0.061748,0.101996,1.040901,0.134038,-0.103675,0.099278,-1.226445]],[[2,19,50,-2,-2,55,-2,-2,37,46,-2,-2,42,-2,-2],[0.9750000238418579,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,1.5,1.5,-2.0,-2.0,0.11678004637360573,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002488,0.019815,0.033225,0.357629,-0.668287,-0.09935,-1.082364,-0.963551,-0.039917,-0.065734,-0.74768,0.766721,0.017372,1.275727,-0.406791]],[[19,5,16,-2,-2,2,-2,-2,42,55,-2,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.3166666775941849,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.003109,0.006094,0.016837,0.210955,-0.69686,-0.06483,0.223518,-1.1942,-0.088201,-0.110371,-1.332489,-1.109863,-0.649008]],[[17,42,22,-2,-2,5,-2,-2,79,-2,-2],[0.5,0.33908046782016754,0.5,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.0015,0.005358,0.019835,0.242279,-1.090396,-0.061692,1.154476,-0.862138,-0.124497,0.107586,-1.416052]],[[17,10,79,-2,-2,9,-2,-2,9,-2,-2],[0.5,0.2678571492433548,0.7573529481887817,-2.0,-2.0,0.45350000262260437,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001285,0.005878,-0.022238,-0.572578,0.156055,0.042588,-0.124454,0.739114,-0.145939,-1.393979,-0.611329]],[[2,17,19,-2,-2,-2,1,0,-2,-2,-2],[0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000772,0.024186,0.034079,0.406888,-0.979241,-1.036009,-0.039393,-0.045477,-0.456921,1.088411,1.378771]],[[76,17,8,-2,-2,9,-2,-2,4,8,-2,-2,9,-2,-2],[0.5,0.5,0.6324999928474426,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.09349999949336052,0.9740000069141388,-2.0,-2.0,0.9839999973773956,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-8.7e-05,-0.017194,-0.007863,0.289923,-0.408619,-0.149918,-1.399707,-0.761133,0.05437,0.124728,0.422402,1.314657,0.001715,0.356562,-0.615197]],[[39,17,16,-2,-2,9,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.006500000134110451,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004388,0.001052,0.007877,0.139834,-0.884414,-0.11903,-1.417212,-0.555537,-0.183204,1.118497,-1.647745]],[[1,4,76,-2,-2,54,-2,-2,46,7,-2,-2,-2],[0.5,0.05849999934434891,0.5,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.12050000205636024,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.004613,-0.017631,0.020449,-0.131019,0.766728,-0.047714,-0.48454,0.901658,0.051384,0.041646,1.052023,-0.243227,1.382633]],[[19,5,16,-2,-2,42,-2,-2,55,79,-2,-2,5,-2,-2],[0.5,0.9895000159740448,0.5,-2.0,-2.0,0.04545454680919647,-2.0,-2.0,0.5,0.7071428596973419,-2.0,-2.0,0.27650000154972076,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001678,0.010368,0.021012,0.247232,-0.709432,-0.081597,0.166631,-0.997796,-0.082095,-0.116579,-0.625972,-1.317821,-0.049515,-0.926671,-1.043067]],[[21,50,64,-2,-2,-2,78,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.4833333343267441,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.000909,-0.00532,0.000655,-0.054124,0.852209,-1.201474,0.10311,-0.167288,1.319549]],[[19,54,2,-2,-2,-2,22,-2,79,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,-2.0,0.5,-2.0,0.6833333373069763,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.005663,0.01404,0.008486,0.64341,-0.086926,0.961245,-0.075136,-1.259771,-0.047151,0.022672,-1.17842]],[[16,5,19,-2,-2,2,-2,-2,7,-2,-2],[0.5,0.9745000004768372,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.011500000022351742,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001175,0.004935,0.014087,0.203374,-0.896012,-0.062343,0.183437,-1.05954,-0.098881,-0.032451,-1.532607]],[[71,76,8,-2,-2,4,-2,-2,-2],[0.5,0.5,0.5940000116825104,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003541,0.00743,-0.010537,0.350287,-0.481145,0.061577,0.928992,0.039028,-1.177529]],[[32,76,11,-2,-2,4,-2,-2,12,-2,80,-2,-2],[0.5,0.5,0.4166666716337204,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,0.4989999979734421,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.006667,-0.015533,-0.033217,-0.513109,0.400177,0.040633,0.821295,-0.168249,0.081202,1.056929,0.034882,-0.576339,1.059091]],[[38,43,2,-2,-2,78,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.5357142984867096,-2.0,-2.0,0.2361111119389534,0.9855000078678131,-2.0,-2.0,0.5339999794960022,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004596,-0.016479,-0.024332,0.331318,-0.392664,0.095251,-0.191373,1.158459,0.049894,0.1045,1.219851,-0.26844,-0.070064,0.211386,-1.354238]],[[16,76,10,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.12132352963089943,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.006798,0.013052,-0.004211,-0.287593,0.344403,0.065732,1.2821,-0.008696,-0.099496,-1.301369,-0.201858]],[[42,43,32,-2,-2,7,-2,-2,42,-2,-2],[0.5857143104076385,0.5,0.5,-2.0,-2.0,0.019999999552965164,-2.0,-2.0,0.7321428656578064,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.006377,0.010308,0.003617,-0.059127,0.818535,0.10142,1.107,0.178647,-0.141565,-1.163312,-1.849858]],[[21,78,8,-2,-2,48,-2,-2,5,-2,-2],[0.5,0.5505050718784332,0.17499999701976776,-2.0,-2.0,0.5,-2.0,-2.0,0.22850000113248825,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00026,-0.006391,-0.043849,0.253389,-0.629864,0.01248,0.047175,1.11666,0.110658,0.546626,1.21896]],[[11,16,76,-2,-2,-2,7,3,-2,-2,4,-2,-2],[0.4166666716337204,0.5,0.5,-2.0,-2.0,-2.0,0.005499999970197678,0.004500000039115548,-2.0,-2.0,0.05550000071525574,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002179,-0.014497,-0.005384,-0.241633,0.354353,-1.571931,0.049747,0.08641,0.700789,1.438146,0.004723,0.842536,-0.241343]],[[17,64,7,-2,-2,-2,42,-2,-2],[0.5,0.5,0.005499999970197678,-2.0,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.002883,0.003116,-0.002831,0.306603,-0.185544,0.923781,-0.121838,-1.381148,-0.264776]],[[48,32,1,-2,-2,10,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000615,-0.007607,-0.016194,-0.255597,0.614539,0.070264,-0.064254,1.34911,0.088245,-0.252957,1.281289]],[[80,7,8,-2,-2,4,-2,-2,69,2,-2,-2,-2],[0.8708333373069763,0.004500000039115548,0.9255000054836273,-2.0,-2.0,0.20000000298023224,-2.0,-2.0,2.5,0.9850000143051147,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002779,-0.022674,0.022682,0.390519,-0.715385,-0.051558,-0.134197,-0.743342,0.036061,0.017459,0.481625,-0.554141,1.222578]],[[79,5,5,-2,-2,1,-2,-2,69,2,-2,-2,-2],[0.8660714328289032,0.02850000001490116,0.015500000212341547,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.9850000143051147,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000501,-0.017749,0.039234,0.14258,0.940442,-0.040168,-0.451948,0.663099,0.040503,0.023977,0.572334,-0.415532,1.259454]],[[5,71,77,-2,-2,-2,8,-2,80,-2,-2],[0.9895000159740448,0.5,0.5,-2.0,-2.0,-2.0,0.984499990940094,-2.0,0.8166666626930237,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[-0.003461,0.004663,0.008495,0.00155,0.734923,-1.11956,-0.091384,-1.293039,-0.070521,-1.278365,-0.263071]],[[11,16,2,-2,-2,-2,39,44,-2,-2,-2],[0.36666667461395264,0.5,0.9750000238418579,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.00032,-0.014804,-0.006186,0.153702,-0.384038,-1.597861,0.042336,0.051814,0.440476,0.690086,-0.902735]],[[19,55,78,-2,-2,-2,55,2,-2,-2,80,-2,-2],[0.5,0.5,0.5358974635601044,-2.0,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.7750000059604645,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002071,0.010032,0.014554,-0.213675,0.304114,-1.338853,-0.073238,-0.110862,-1.235825,-1.120649,-0.035146,-0.62805,-1.05886]],[[31,2,77,-2,-2,15,-2,-2,7,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.036000000312924385,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003112,-0.009323,0.033725,0.303618,1.198173,-0.029216,-0.330031,0.474346,0.103186,0.458933,1.340502]],[[78,8,79,-2,-2,5,-2,-2,16,19,-2,-2,-2],[0.5505050718784332,0.17549999803304672,0.3457317054271698,-2.0,-2.0,0.013500000350177288,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.005027,-0.036173,0.038736,-1.111685,0.552131,-0.069919,-1.250562,-0.410665,0.026926,0.035753,0.43501,-0.999186,-0.943301]],[[1,82,2,-2,-2,3,-2,-2,2,7,-2,-2,8,-2,-2],[0.5,0.34495192766189575,0.9750000238418579,-2.0,-2.0,0.953000009059906,-2.0,-2.0,0.4749999940395355,0.25200000405311584,-2.0,-2.0,0.04449999891221523,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005186,-0.016765,-0.007846,0.163627,-0.386422,-0.111571,-0.536327,-1.49873,0.044569,0.033642,0.85282,-0.331508,0.19912,1.133292,1.505835]],[[79,8,5,-2,-2,81,-2,-2,4,69,-2,-2,9,-2,-2],[0.6449579894542694,0.09699999913573265,0.00849999999627471,-2.0,-2.0,0.2204861119389534,-2.0,-2.0,0.061000000685453415,1.5,-2.0,-2.0,0.416499987244606,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001035,-0.038209,0.02809,0.90587,-0.162252,-0.06433,-0.470553,-1.133503,0.01586,0.058831,0.247085,0.974085,-0.010682,-0.889578,0.056806]],[[71,76,3,-2,-2,4,-2,-2,-2],[0.5,0.5,0.19700000435113907,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004561,0.007537,-0.005448,0.238928,-0.440271,0.046446,0.753524,-0.039556,-1.103001]],[[5,9,4,-2,-2,78,-2,-2,8,-2,2,-2,-2],[0.9835000038146973,0.9494999945163727,0.05849999934434891,-2.0,-2.0,0.5358974635601044,-2.0,-2.0,0.9835000038146973,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.00025,0.008328,-0.021856,0.452103,-0.458737,0.032613,-0.212061,0.581859,-0.062731,-1.170116,-0.031703,0.288697,-0.930021]],[[76,3,17,-2,-2,6,-2,-2,3,6,-2,-2,78,-2,-2],[0.5,0.24550000578165054,0.5,-2.0,-2.0,0.6335000097751617,-2.0,-2.0,0.14900000393390656,0.5575000047683716,-2.0,-2.0,0.4686274528503418,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002901,-0.010802,0.011792,0.215483,-1.024209,-0.046216,-0.893709,-0.0745,0.047994,-0.010037,0.949715,-0.469098,0.113553,-0.460621,0.996108]],[[16,78,8,-2,-2,70,-2,-2,9,-2,-2],[0.5,0.5505050718784332,0.17499999701976776,-2.0,-2.0,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002855,0.008938,-0.025066,0.378904,-0.466602,0.026633,0.159781,0.903751,-0.097022,-1.592185,-0.051982]],[[78,11,69,-2,-2,83,-2,-2,4,6,-2,-2,2,-2,-2],[0.5358974635601044,0.22500000149011612,3.5,-2.0,-2.0,1.5,-2.0,-2.0,0.09100000187754631,0.7080000042915344,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001613,-0.034903,-0.065496,-0.48054,-1.448189,0.038556,-0.620599,0.97205,0.019704,0.05985,0.148675,1.066538,-0.010048,0.133006,-0.8063]],[[71,21,5,-2,-2,5,-2,-2,-2],[0.5,0.5,0.024500000290572643,-2.0,-2.0,0.11800000071525574,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003625,0.000161,-0.004742,0.437912,-0.176463,0.09748,0.153717,1.142195,-1.462843]],[[80,2,2,-2,-2,5,-2,-2,69,2,-2,-2,4,-2,-2],[0.8708333373069763,0.9350000023841858,0.8250000178813934,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003359,-0.013667,0.031622,0.153642,1.036271,-0.032305,1.321598,-0.326499,0.036526,0.01407,0.46626,-0.624099,0.11499,1.265601,0.267923]],[[8,9,5,-2,-2,42,-2,-2,63,15,-2,-2,-2],[0.4089999943971634,0.945499986410141,0.006500000134110451,-2.0,-2.0,0.4226190447807312,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001381,0.02992,-0.004061,1.183369,-0.336488,0.08517,1.109222,-0.763492,-0.019723,-0.030426,-0.365295,0.782277,0.712682]],[[19,5,16,-2,-2,2,-2,-2,55,-2,5,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,-2.0,0.21800000220537186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003514,0.004326,0.014225,0.199384,-0.838534,-0.066099,0.060412,-1.037304,-0.0741,-1.190514,-0.047033,-1.111337,-1.040329]],[[48,2,44,-2,-2,85,-2,-2,6,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-6.5e-05,-0.005948,0.012325,0.459231,-0.148335,-0.036194,-0.24355,-1.289396,0.080384,-0.344993,1.047623]],[[16,71,76,-2,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5565000176429749,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.000224,0.006099,0.009565,-0.064226,0.475927,-1.052073,-0.108426,-0.492911,-1.507444]],[[38,16,2,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.2567765712738037,0.9855000078678131,-2.0,-2.0,0.48900000751018524,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006715,-0.005201,0.001063,0.506025,-0.120948,-0.109097,-1.425031,-0.575454,0.064628,0.095231,1.158462,-0.329826,-0.023689,0.504997,-1.320028]],[[42,17,7,-2,-2,9,-2,-2,80,-2,3,-2,-2],[0.5505050718784332,0.5,0.007500000298023224,-2.0,-2.0,0.9345000088214874,-2.0,-2.0,0.2361111119389534,-2.0,0.664000004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.00086,0.004667,0.009274,0.369804,-0.08153,-0.090957,-1.497365,-0.197187,-0.110354,-1.7977,-0.083622,-0.562163,-1.512923]],[[20,5,-2,-2,17,16,-2,-2,79,-2,-2],[0.5,0.8325000107288361,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[-0.00143,-0.106563,-0.445212,-1.256178,0.005131,0.010628,0.169508,-0.847556,-0.100068,0.159691,-1.434552]],[[17,19,2,-2,-2,12,-2,-2,4,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.6315000057220459,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003652,0.008964,0.016705,0.366361,-0.152804,-0.061754,-0.688344,-1.29775,-0.100335,-1.468146,-0.13037]],[[80,2,7,-2,-2,1,-2,-2,76,2,-2,-2,3,-2,-2],[0.8708333373069763,0.9600000083446503,0.18699999898672104,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.9850000143051147,-2.0,-2.0,0.048499999567866325,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00312,-0.013719,0.009854,-0.088391,0.721889,-0.042673,-0.52713,1.450607,0.034736,0.007062,0.482403,-0.903293,0.106503,-0.204856,1.285405]],[[55,42,19,-2,-2,2,-2,-2,19,-2,78,-2,-2],[0.5,0.612500011920929,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.5,-2.0,0.6515151560306549,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001288,0.007374,0.010397,0.143627,-0.873937,-0.151614,-1.694923,-1.253185,-0.070094,-1.267891,-0.03912,-0.730936,-1.095138]],[[54,17,20,-2,-2,78,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4833333343267441,-2.0,-2.0,0.9835000038146973,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003895,-0.00153,0.004812,-0.672745,0.110555,-0.115805,-0.358224,-1.340653,0.113485,1.065031,0.503007]],[[21,5,17,-2,-2,78,-2,-2,5,-2,-2],[0.5,0.992499977350235,0.5,-2.0,-2.0,0.7071428596973419,-2.0,-2.0,0.13049999624490738,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002203,-0.002753,0.003723,0.089642,-0.978176,-0.081013,-0.428562,-1.004623,0.09699,0.478201,1.149661]],[[65,56,63,-2,-2,83,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003155,-0.010831,-0.005325,-0.115914,0.571675,-0.112837,-1.296672,-0.76392,0.101246,1.265516,0.427866]],[[82,42,43,-2,-2,79,-2,-2,2,8,-2,-2,2,-2,-2],[0.3060200661420822,0.5857143104076385,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.875,0.3785000145435333,-2.0,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001396,0.010145,0.014326,0.070514,0.767884,-0.128458,-1.708217,-1.236363,-0.051789,0.050124,0.097146,1.007786,-0.093897,-0.932257,-0.080053]],[[5,69,8,-2,-2,-2,21,9,-2,-2,-2],[0.0494999997317791,2.5,0.9180000126361847,-2.0,-2.0,-2.0,0.5,0.9774999916553497,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.000659,0.033987,0.023231,0.509266,-0.938155,0.83338,-0.018592,-0.025926,-0.623038,0.013144,0.795699]],[[44,2,42,-2,-2,46,-2,-2,6,-2,6,-2,-2],[0.5,0.9950000047683716,0.44949495792388916,-2.0,-2.0,1.5,-2.0,-2.0,0.024500000290572643,-2.0,0.24050000309944153,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.002695,0.017531,0.036268,0.542241,-0.835712,-0.048028,-0.982336,1.006582,-0.029962,1.247088,-0.040562,-1.085482,-0.243973]],[[17,2,46,-2,-2,5,-2,-2,42,-2,-2],[0.5,0.9750000238418579,1.5,-2.0,-2.0,0.5489999949932098,-2.0,-2.0,0.2928571552038193,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001313,0.003688,0.019327,0.264737,-0.867697,-0.023029,0.212081,-0.616914,-0.089836,-1.342355,0.113477]],[[76,3,17,-2,-2,6,-2,-2,37,3,-2,-2,-2],[0.5,0.13000000268220901,0.5,-2.0,-2.0,0.3265000134706497,-2.0,-2.0,3.5,0.14900000393390656,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.004663,-0.007601,0.018225,0.317121,-0.95716,-0.037736,-0.911501,-0.132591,0.042,0.0625,0.153947,0.821642,-0.889939]],[[42,70,22,-2,-2,3,-2,-2,8,79,-2,-2,1,-2,-2],[0.33908046782016754,0.5,0.5,-2.0,-2.0,0.009999999776482582,-2.0,-2.0,0.024000000208616257,0.49193547666072845,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003098,0.01255,0.001758,0.088731,-1.204275,0.093802,0.122683,0.988549,-0.042126,0.073393,0.905892,0.894728,-0.070397,-0.929093,1.163038]],[[22,79,8,-2,-2,63,-2,-2,52,-2,78,-2,-2],[0.5,0.6449579894542694,0.07349999994039536,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.006194,0.013167,-0.029156,0.297974,-0.485055,0.02904,0.201017,0.92341,-0.074656,-1.275821,-0.036779,-0.653302,-1.086302]],[[5,6,7,-2,-2,39,-2,-2,2,79,-2,-2,6,-2,-2],[0.6994999945163727,0.02850000001490116,0.08950000256299973,-2.0,-2.0,0.5,-2.0,-2.0,0.9750000238418579,0.7638888955116272,-2.0,-2.0,0.992499977350235,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002812,0.011311,0.100225,0.780268,1.823782,0.004333,0.097944,-1.067604,-0.037058,0.021016,-0.487256,0.680057,-0.07031,-1.060553,0.973566]],[[4,4,78,-2,-2,5,-2,-2,1,9,-2,-2,46,-2,-2],[0.06000000052154064,0.04649999924004078,0.5419580638408661,-2.0,-2.0,0.33250000327825546,-2.0,-2.0,0.5,0.38349999487400055,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002217,0.024159,0.011113,-0.37691,0.342711,0.161888,0.509061,1.611247,-0.019468,-0.038919,-1.035735,-0.211015,0.033869,0.513074,1.358227]],[[65,5,78,-2,-2,8,-2,-2,79,-2,-2],[0.5,0.9835000038146973,0.7638888955116272,-2.0,-2.0,0.9895000159740448,-2.0,-2.0,0.7071428596973419,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00157,-0.004935,0.002397,-0.11812,0.421087,-0.060797,-1.334086,-0.041469,0.081828,-0.089706,1.155164]],[[38,8,9,-2,-2,48,-2,-2,79,5,-2,-2,9,-2,-2],[0.5,0.1314999982714653,0.9745000004768372,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.051500000059604645,-2.0,-2.0,0.9900000095367432,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004476,-0.014949,0.030943,0.103028,0.850501,-0.03213,-0.397734,0.51709,0.046941,-0.017669,0.644551,-0.653146,0.101063,1.07456,1.304278]],[[76,8,9,-2,-2,6,-2,-2,4,8,-2,-2,37,-2,-2],[0.5,0.39249999821186066,0.38199999928474426,-2.0,-2.0,0.3009999990463257,-2.0,-2.0,0.03350000083446503,0.976500004529953,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001542,-0.008869,0.024508,-0.44139,0.624396,-0.033487,-1.040496,-0.120788,0.035065,0.127825,0.555299,1.34588,-0.010798,0.095809,-1.182552]],[[19,42,17,-2,-2,79,-2,-2,55,81,-2,-2,78,-2,-2],[0.5,0.5857143104076385,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.13701201230287552,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001815,0.008205,0.011531,0.154852,-0.550435,-0.115128,-1.690496,-1.087158,-0.058444,-0.088091,-1.09869,-0.975087,-0.033371,-0.56011,-1.083713]],[[71,76,3,-2,-2,4,-2,-2,-2],[0.5,0.5,0.1979999989271164,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002788,-6.9e-05,-0.014419,0.141574,-0.547414,0.04439,0.788146,-0.006782,-0.940064]],[[76,3,4,-2,-2,77,-2,-2,42,3,-2,-2,2,-2,-2],[0.5,0.11449999734759331,0.679500013589859,-2.0,-2.0,0.5,-2.0,-2.0,0.11324786394834518,0.048499999567866325,-2.0,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000283,-0.013432,0.01342,0.487833,-0.179216,-0.045473,-0.586062,0.781292,0.038993,0.118671,0.069303,1.444363,-0.008069,0.380291,-0.741922]],[[64,21,5,-2,-2,43,-2,-2,-2],[0.5,0.5,0.024500000290572643,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001781,-0.002743,-0.007675,0.374768,-0.201406,0.090394,0.279323,1.150865,0.916626]],[[12,6,4,-2,-2,5,-2,-2,20,5,-2,-2,30,-2,-2],[0.2580000013113022,0.7980000078678131,0.023000000044703484,-2.0,-2.0,0.08350000157952309,-2.0,-2.0,0.5,0.8480000197887421,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001195,0.072044,0.146372,0.076686,1.580946,0.010833,1.075391,-0.490356,-0.007008,-0.102833,-0.460472,-1.461416,-0.00122,-0.086298,0.82256]],[[78,8,4,-2,-2,39,-2,-2,2,7,-2,-2,7,-2,-2],[0.8660714328289032,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0,0.9850000143051147,0.6099999845027924,-2.0,-2.0,0.8129999935626984,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004341,-0.013641,0.023253,0.653664,-0.25704,-0.02607,-0.199579,-1.42244,0.032886,0.07128,0.567388,1.87385,-0.029393,0.833271,-0.718853]],[[78,1,4,-2,-2,2,-2,-2,2,5,-2,-2,7,-2,-2],[0.8090909123420715,0.5,0.9884999990463257,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.9850000143051147,0.8559999763965607,-2.0,-2.0,0.2460000067949295,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005906,-0.01736,-0.027403,-0.182899,-1.161859,0.02749,0.243824,1.41933,0.025261,0.05705,0.565485,1.207162,-0.022553,0.594737,-0.584326]],[[2,9,7,-2,-2,78,-2,-2,1,20,-2,-2,-2],[0.9600000083446503,0.9440000057220459,0.005499999970197678,-2.0,-2.0,0.3550420254468918,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000726,0.01757,-0.017238,0.407855,-0.595035,0.041608,-0.532718,0.544643,-0.02,-0.024173,-0.910345,-0.180809,1.392557]],[[79,5,5,-2,-2,21,-2,-2,6,85,-2,-2,4,-2,-2],[0.8708333373069763,0.024500000290572643,0.019499999471008778,-2.0,-2.0,0.5,-2.0,-2.0,0.9835000038146973,0.5,-2.0,-2.0,0.419000007212162,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00222,-0.011692,0.030035,0.151696,1.361121,-0.027245,-0.357231,0.655629,0.031597,0.004291,0.301262,-1.280641,0.103106,1.600248,0.548689]],[[8,4,3,-2,-2,44,-2,-2,43,79,-2,-2,78,-2,-2],[0.09149999916553497,0.9185000061988831,0.032500000670552254,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.7638888955116272,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007161,0.025925,0.060014,0.588416,1.167669,-0.013468,0.779575,-0.784709,-0.018134,-0.026416,-0.605504,-0.014026,0.080714,-0.185157,1.283244]],[[16,21,2,-2,-2,9,-2,-2,9,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.9845000207424164,-2.0,-2.0,0.5600000023841858,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001243,0.004197,-0.001045,0.452368,-0.144157,0.095116,1.173005,0.3063,-0.091639,-1.513048,-0.471033]],[[19,16,17,-2,-2,9,-2,-2,55,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9305000007152557,-2.0,-2.0,0.5,-2.0,0.48900000751018524,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002411,0.008426,0.013724,0.185456,-0.6181,-0.069947,-1.34573,0.688105,-0.061566,-1.03473,-0.035108,-0.701411,-1.095059]],[[38,6,4,-2,-2,79,-2,-2,42,36,-2,-2,12,-2,-2],[0.5,0.9794999957084656,0.06000000052154064,-2.0,-2.0,0.8660714328289032,-2.0,-2.0,0.2899159789085388,0.5,-2.0,-2.0,0.4580000042915344,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000623,-0.008364,-0.02234,0.022837,-0.463545,0.023828,-0.122901,1.191293,0.048259,0.087378,0.752443,1.538786,-0.076435,-0.100652,-1.278965]],[[7,3,17,-2,-2,78,-2,-2,39,16,-2,-2,-2],[0.007500000298023224,0.15599999576807022,0.5,-2.0,-2.0,0.5192307829856873,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001084,0.023197,0.041591,0.598496,-0.764312,-0.017961,-0.707587,0.260578,-0.014816,-0.009108,-0.014198,-1.490656,-1.130322]],[[10,30,78,-2,-2,8,-2,-2,9,4,-2,-2,42,-2,-2],[0.2678571492433548,0.5,0.5505050718784332,-2.0,-2.0,0.8989999890327454,-2.0,-2.0,0.9255000054836273,0.061000000685453415,-2.0,-2.0,0.47727273404598236,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00075,-0.015748,-0.028576,-0.667985,-0.056988,0.091985,0.122979,1.102217,0.021865,-0.00343,0.736692,-0.335984,0.056789,0.874199,-1.03166]],[[22,2,7,-2,-2,5,-2,-2,19,-2,79,-2,-2],[0.5,0.9750000238418579,0.9354999959468842,-2.0,-2.0,0.5489999949932098,-2.0,-2.0,0.5,-2.0,0.6833333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003977,0.009492,0.026496,0.178927,1.209132,-0.017363,0.258123,-0.601598,-0.062219,-1.321896,-0.033306,-0.025587,-1.141569]],[[19,17,78,-2,-2,79,-2,-2,12,8,-2,-2,-2],[0.5,0.5,0.3867521435022354,-2.0,-2.0,0.612500011920929,-2.0,-2.0,0.6444999873638153,0.3799999952316284,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003505,0.010149,0.015005,-0.334157,0.244512,-0.069085,0.420993,-1.237251,-0.053062,-0.040546,-0.904342,-0.667942,-1.267049]],[[63,8,5,-2,-2,5,-2,-2,80,-2,-2],[0.5,0.5940000116825104,0.5920000076293945,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.8257575631141663,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004458,-8.9e-05,0.020737,0.355081,-0.545046,-0.020786,-1.186496,-0.128955,0.081734,-0.181569,1.331734]],[[1,78,2,-2,-2,70,-2,-2,46,11,-2,-2,-2],[0.5,0.4686274528503418,0.9249999821186066,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000461,-0.00855,-0.048676,0.416583,-0.64543,0.003233,-0.082755,0.792292,0.038452,0.030651,0.817923,0.531782,1.377508]],[[19,85,79,-2,-2,12,-2,-2,12,8,-2,-2,-2],[0.5,0.5,0.8660714328289032,-2.0,-2.0,0.7720000147819519,-2.0,-2.0,0.6444999873638153,0.40150000154972076,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001489,0.007503,0.014711,-0.013383,0.561995,-0.049758,-0.629137,1.122023,-0.051365,-0.038337,-0.906491,-0.561647,-1.249708]],[[21,77,38,-2,-2,4,-2,-2,78,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.0729999989271164,-2.0,-2.0,0.4833333343267441,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.005294,0.00099,-0.005823,-0.168976,0.524503,0.054142,1.32091,0.071556,0.078257,-0.241109,1.207505]],[[42,21,2,-2,-2,78,-2,-2,80,-2,3,-2,-2],[0.5857143104076385,0.5,0.9350000023841858,-2.0,-2.0,0.4833333343267441,-2.0,-2.0,0.2361111119389534,-2.0,0.9415000081062317,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.00259,0.000164,-0.003992,0.386179,-0.151223,0.075184,-0.099843,1.12872,-0.104596,-1.694896,-0.073617,-0.869459,-1.487721]],[[21,17,11,-2,-2,9,-2,-2,43,-2,-2],[0.5,0.5,0.36666667461395264,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003931,-0.000434,0.0042,-0.070497,0.54468,-0.088622,-1.312053,-0.370335,0.085271,0.249431,1.201553]],[[4,4,69,-2,-2,-2,21,5,-2,-2,-2],[0.06000000052154064,0.05049999989569187,2.5,-2.0,-2.0,-2.0,0.5,0.024500000290572643,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.001698,0.023934,0.015287,0.022575,0.760798,1.412964,-0.012497,-0.018837,0.393526,-0.363852,0.748863]],[[2,78,9,-2,-2,2,-2,-2,46,85,-2,-2,9,-2,-2],[0.9750000238418579,0.8090909123420715,0.3799999952316284,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001351,0.014952,-9e-05,-0.888475,0.12644,0.060829,0.648029,1.188859,-0.02759,-0.03994,-0.330953,-1.182621,0.063843,0.248583,1.375933]],[[12,6,4,-2,-2,3,-2,-2,39,12,-2,-2,7,-2,-2],[0.2580000013113022,0.7980000078678131,0.023000000044703484,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,0.5,0.3994999974966049,-2.0,-2.0,0.007000000216066837,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002101,0.074452,0.155034,0.337144,1.552254,0.009987,1.516511,-0.620324,-0.003057,0.000702,-0.474098,0.12402,-0.114477,0.489053,-1.198374]],[[21,2,39,-2,-2,81,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.17444219440221786,-2.0,-2.0,0.789000004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004594,-0.009381,0.019072,0.394038,-1.182169,-0.022617,-0.051393,-0.550169,0.082734,-0.192853,1.236677]],[[5,4,77,-2,-2,9,-2,-2,12,12,-2,-2,-2],[0.992499977350235,0.05849999934434891,0.5,-2.0,-2.0,0.3854999989271164,-2.0,-2.0,0.7674999833106995,0.5164999961853027,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.006023,0.010919,0.038494,0.24315,1.485231,-0.00554,-0.796542,0.089289,-0.064582,-0.045687,-0.957839,0.232143,-1.31398]],[[17,55,2,-2,-2,5,-2,-2,79,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.16350000351667404,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001423,0.005555,0.01028,0.263422,-0.149542,-0.055303,-1.240515,-0.83517,-0.073246,0.152761,-1.307441]],[[2,55,82,-2,-2,6,-2,-2,1,0,-2,-2,-2],[0.9750000238418579,0.5,0.34954752027988434,-2.0,-2.0,0.9625000059604645,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000177,0.01383,0.021033,0.326978,-0.557359,-0.076321,-1.328895,-0.831108,-0.02448,-0.029766,-0.394247,1.113817,1.423186]],[[16,20,80,-2,-2,76,-2,-2,9,-2,-2],[0.5,0.5,0.7638888955116272,-2.0,-2.0,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003503,0.008075,-0.071594,-1.273931,-0.198079,0.013126,0.019249,0.470313,-0.076561,-1.136848,-0.064582]],[[54,11,63,-2,-2,10,-2,-2,6,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.8333333432674408,-2.0,-2.0,0.8060000240802765,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[9.6e-05,-0.003941,-0.014988,-0.215921,0.566272,0.026707,0.746512,-0.252678,0.084677,0.039221,1.201496]],[[17,11,63,-2,-2,3,-2,-2,4,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.021499999798834324,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000854,0.002957,-0.008355,-0.154371,0.652078,0.035528,0.381527,0.852156,-0.079502,-1.548036,-0.262112]],[[65,76,2,-2,-2,8,-2,-2,12,-2,-2],[0.5,0.5,0.875,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,0.5089999884366989,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001934,-0.003078,-0.014267,0.368859,-0.325789,0.030686,0.049252,0.825857,0.069496,1.074861,0.157349]],[[5,5,77,-2,-2,79,-2,-2,8,-2,2,-2,-2],[0.9835000038146973,0.9544999897480011,0.5,-2.0,-2.0,0.8660714328289032,-2.0,-2.0,0.9835000038146973,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000281,0.00599,0.001199,-0.063643,0.563945,0.085641,-0.26288,1.44898,-0.049176,-1.402495,-0.006261,0.815189,-0.790249]],[[5,19,78,-2,-2,3,-2,-2,3,-2,6,-2,-2],[0.992499977350235,0.5,0.3550420254468918,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.965499997138977,-2.0,0.04100000113248825,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.005458,0.0106,0.01722,-0.347437,0.260371,-0.046659,-0.101099,-1.134989,-0.061232,-1.127937,-0.005393,-1.625468,0.277466]],[[5,82,6,-2,-2,53,-2,-2,78,39,-2,-2,12,-2,-2],[0.02850000001490116,0.39444445073604584,0.992499977350235,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.5,-2.0,-2.0,0.3824999928474426,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002774,0.025668,0.038256,0.685003,-0.025399,-0.132429,1.085214,-1.432718,-0.013404,-0.031393,-0.254607,-1.54279,0.018028,-0.81263,0.390417]],[[38,6,76,-2,-2,12,-2,-2,39,6,-2,-2,-2],[0.5,0.8899999856948853,0.5,-2.0,-2.0,0.33550000190734863,-2.0,-2.0,0.5,0.9855000078678131,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.003948,-0.012338,-0.03367,-0.611899,0.287722,0.007013,-0.599975,0.261916,0.038288,0.053048,0.891177,-0.409214,-1.203255]],[[20,83,-2,-2,16,19,-2,-2,9,-2,-2],[0.5,1.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[-0.004916,-0.082432,-1.198539,0.134159,-0.000401,0.004687,0.115208,-0.933447,-0.077798,-1.51065,-0.208107]],[[4,4,78,-2,-2,-2,9,44,-2,-2,8,-2,-2],[0.05849999934434891,0.05049999989569187,0.5419580638408661,-2.0,-2.0,-2.0,0.7774999737739563,0.5,-2.0,-2.0,0.5844999849796295,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002361,0.021666,0.012927,-0.472734,0.387402,1.652651,-0.017081,-0.055691,-0.142361,-1.096982,-0.000635,0.320206,-0.259623]],[[70,37,46,-2,-2,-2,42,4,-2,-2,-2],[0.5,4.5,3.5,-2.0,-2.0,-2.0,0.2899159789085388,0.07999999821186066,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.002417,-0.003733,-0.000172,-0.029284,0.94138,-1.063382,0.047544,0.082676,0.502852,0.998061,-0.875986]],[[11,16,76,-2,-2,-2,9,4,-2,-2,44,-2,-2],[0.4166666716337204,0.5,0.5,-2.0,-2.0,-2.0,0.9839999973773956,0.17100000381469727,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001541,-0.009177,-0.002974,-0.195838,0.311267,-1.48534,0.033405,0.01032,0.958807,-0.285459,0.091405,0.417003,1.346184]],[[12,6,4,-2,-2,-2,6,78,-2,-2,80,-2,-2],[0.2580000013113022,0.9735000133514404,0.013000000268220901,-2.0,-2.0,-2.0,0.8935000002384186,0.8090909123420715,-2.0,-2.0,0.8619047701358795,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002785,0.062438,0.098396,-0.262759,1.143167,-0.35202,-0.00195,-0.019251,-0.418037,0.254337,0.015838,-0.034068,0.714171]],[[44,6,3,-2,-2,3,-2,-2,6,-2,9,-2,-2],[0.5,0.9784999787807465,0.006500000134110451,-2.0,-2.0,0.2199999988079071,-2.0,-2.0,0.009999999776482582,-2.0,0.9469999969005585,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004432,0.010068,-0.011035,0.695307,-0.335276,0.062865,0.497088,1.757849,-0.023107,1.349493,-0.03006,-0.614026,-0.087198]],[[64,15,44,-2,-2,7,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.03749999962747097,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.004443,-0.008461,-0.014627,0.050596,-0.365691,0.039901,0.944065,-0.218052,0.797265]],[[32,1,78,-2,-2,2,-2,-2,12,-2,5,-2,-2],[0.5,0.5,0.4833333343267441,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.41499999165534973,-2.0,0.17549999803304672,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[1.3e-05,-0.005174,-0.014192,-0.533261,-0.014956,0.028724,0.52258,1.355857,0.051941,1.143504,0.019972,0.781261,-0.402507]],[[2,78,4,-2,-2,9,-2,-2,15,30,-2,-2,9,-2,-2],[0.9350000023841858,0.41428571939468384,0.042000001296401024,-2.0,-2.0,0.945499986410141,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.921999990940094,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005664,0.0308,0.082358,0.374476,1.197592,0.02298,0.096219,0.765981,-0.005686,-0.012878,-0.201876,0.541299,0.074833,-0.313118,1.140348]],[[38,77,16,-2,-2,4,-2,-2,42,36,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.060499999672174454,-2.0,-2.0,0.31129032373428345,0.5,-2.0,-2.0,0.43949998915195465,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000488,-0.008578,-0.014959,-0.108108,-1.238466,0.05026,1.202079,-0.027994,0.036674,0.066413,0.594153,1.475508,-0.084281,-0.077523,-1.251705]],[[71,67,17,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.001769,0.004132,0.001284,0.057657,-0.766022,1.041264,-1.183485]],[[5,77,21,-2,-2,4,-2,-2,8,-2,8,-2,-2],[0.9835000038146973,0.5,0.5,-2.0,-2.0,0.0729999989271164,-2.0,-2.0,0.9835000038146973,-2.0,0.992499977350235,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003125,0.003924,-0.002804,-0.094185,0.998566,0.062699,1.307673,0.099051,-0.056781,-1.377454,-0.025444,0.40542,-0.828584]],[[38,69,3,-2,-2,42,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,2.5,0.001500000071246177,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,0.22474747896194458,0.9855000078678131,-2.0,-2.0,0.4919999986886978,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000466,-0.006849,-0.012996,0.477975,-0.258212,0.047865,1.228663,-0.068265,0.035859,0.072533,1.238811,-0.161936,-0.042071,0.261072,-1.262259]],[[64,2,69,-2,-2,30,-2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.002933,-0.001231,0.02401,0.340041,0.842058,-0.013994,-0.206785,0.51753,0.899143]],[[78,8,3,-2,-2,5,-2,-2,9,4,-2,-2,2,-2,-2],[0.5505050718784332,0.17549999803304672,0.021499999798834324,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.9465000033378601,0.007500000298023224,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001423,-0.024866,0.027007,0.092297,0.773575,-0.047125,-1.152301,-0.355191,0.010966,-0.015661,1.301535,-0.387829,0.028791,0.470445,-0.353457]],[[81,65,1,-2,-2,80,-2,-2,-2],[0.3798076957464218,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.000822,-0.00355,-0.008306,-0.184859,0.787153,0.057547,0.054844,1.242293,0.883145]],[[54,11,77,-2,-2,80,-2,-2,9,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.41428571939468384,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002151,-0.006155,-0.015923,-0.238877,0.393704,0.02206,-1.17965,0.43739,0.084423,0.960097,0.408622]],[[12,6,4,-2,-2,3,-2,-2,2,12,-2,-2,1,-2,-2],[0.2604999989271164,0.5649999976158142,0.023000000044703484,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.9750000238418579,0.734499990940094,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002498,0.057094,0.134879,-0.185037,1.422326,0.023256,1.154478,-0.11221,-0.007206,0.005338,-0.091671,0.553788,-0.029577,-0.408668,1.39729]],[[78,8,2,-2,-2,4,-2,-2,48,4,-2,-2,-2],[0.5505050718784332,0.17549999803304672,0.9850000143051147,-2.0,-2.0,0.7944999933242798,-2.0,-2.0,0.5,0.09100000187754631,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00063,-0.02498,0.027392,0.15822,0.846801,-0.052323,-0.786237,0.102614,0.013325,0.005965,0.356737,-0.183518,1.294085]],[[70,6,3,-2,-2,12,-2,-2,78,-2,4,-2,-2],[0.5,0.8819999992847443,0.001500000071246177,-2.0,-2.0,0.33550000190734863,-2.0,-2.0,0.3550420254468918,-2.0,0.26899999380111694,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003842,-0.002553,-0.022438,1.018902,-0.385357,0.015437,-0.537881,0.356879,0.052989,-1.039773,0.084427,0.596597,1.12828]],[[2,46,7,-2,-2,-2,46,85,-2,-2,5,-2,-2],[0.9750000238418579,1.5,0.9284999966621399,-2.0,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.6850000023841858,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004496,0.017558,0.022769,0.199846,0.971782,-0.688055,-0.016849,-0.027587,-0.206408,-1.084347,0.061968,1.047844,-0.58118]],[[19,55,20,-2,-2,-2,55,2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.16700000315904617,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001894,0.007822,0.011002,-0.531454,0.170635,-1.282813,-0.053189,-0.082983,-1.155092,-1.097822,-0.026441,-1.0906,-0.568682]],[[2,16,5,-2,-2,-2,85,76,-2,-2,-2],[0.9750000238418579,0.5,0.28949999809265137,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.001689,0.01029,0.015927,-0.003309,0.495893,-0.783619,-0.021538,-0.013053,-0.393099,0.333805,-0.968631]],[[21,5,77,-2,-2,8,-2,-2,8,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,0.5300000160932541,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00362,-0.008353,-0.001722,-0.097769,0.515516,-0.058072,-1.394565,-0.381796,0.08738,0.692047,1.080323]],[[42,19,81,-2,-2,22,-2,-2,-2],[0.612500011920929,0.5,0.23303167521953583,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004571,0.006668,0.011871,0.234563,-0.211013,-0.045278,-1.217293,-0.576909,-1.527253]],[[19,79,-2,43,-2,-2,22,2,-2,-2,12,-2,-2],[0.5,0.232051283121109,-2.0,0.5,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.4599999934434891,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[0.005356,0.010525,-1.547991,0.012113,0.08816,0.613379,-0.040634,-0.06664,-0.937374,-1.09301,-0.019912,0.23972,-1.099204]],[[62,5,44,-2,-2,12,-2,-2,-2],[0.5,0.9895000159740448,0.5,-2.0,-2.0,0.8680000007152557,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003691,-0.006704,-0.001732,0.254695,-0.22886,-0.058148,-0.922089,0.19176,0.721973]],[[79,69,12,-2,-2,2,-2,-2,69,2,-2,-2,6,-2,-2],[0.8708333373069763,0.5,0.47349999845027924,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.715499997138977,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000886,-0.009335,0.012852,0.628303,-0.169587,-0.035278,0.615752,-0.43244,0.022176,0.002052,0.403671,-0.642264,0.090453,0.083474,1.251112]],[[21,17,19,-2,-2,2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.13049999624490738,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-3.7e-05,-0.003691,0.001036,0.069822,-0.92198,-0.091417,-1.228403,-0.815555,0.066178,0.07527,1.051214]],[[63,8,5,-2,-2,44,-2,-2,8,-2,-2],[0.5,0.5940000116825104,0.8614999949932098,-2.0,-2.0,0.5,-2.0,-2.0,0.8290000259876251,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000411,-0.004306,0.011814,0.221826,-0.727105,-0.020506,0.006259,-0.483,0.063338,-0.42223,1.186058]],[[71,76,8,-2,-2,8,-2,-2,-2],[0.5,0.5,0.5940000116825104,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002735,-0.000567,-0.009503,0.15767,-0.369965,0.027976,0.08678,0.724789,-0.853008]],[[43,17,37,-2,-2,9,-2,-2,78,-2,-2],[0.5,0.5,4.5,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002306,-0.002056,0.001378,0.053951,-0.742145,-0.077012,-1.333597,-0.191424,0.057936,-0.267031,1.120932]],[[77,21,78,-2,-2,5,-2,-2,3,69,-2,-2,82,-2,-2],[0.5,0.5,0.8452380895614624,-2.0,-2.0,0.13049999624490738,-2.0,-2.0,0.014500000048428774,0.5,-2.0,-2.0,0.09545454755425453,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000745,-0.004588,-0.00845,-0.205039,0.273481,0.065875,0.476433,0.792144,0.04651,-0.015484,0.801918,-0.821782,0.101838,0.156821,1.534878]],[[4,4,78,-2,-2,5,-2,-2,20,-2,38,-2,-2],[0.05849999934434891,0.045500000938773155,0.5419580638408661,-2.0,-2.0,0.3125,-2.0,-2.0,0.5,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003029,0.01762,0.005722,-0.427571,0.263705,0.139823,0.228461,1.759835,-0.016188,-1.064767,-0.011327,-0.261545,0.492883]],[[81,82,64,-2,-2,11,-2,-2,-2],[0.3798076957464218,0.30306123197078705,0.5,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001835,-0.000947,0.004839,0.01475,0.999749,-0.041619,-0.631238,0.726283,1.018541]],[[9,4,9,-2,-2,5,-2,-2,11,79,-2,-2,44,-2,-2],[0.945499986410141,0.007500000298023224,0.03399999998509884,-2.0,-2.0,0.8040000200271606,-2.0,-2.0,0.4166666716337204,0.7573529481887817,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005871,-0.012946,0.077086,1.076904,1.365477,-0.022825,-0.137646,-1.076581,0.018181,0.008407,-0.231103,0.317394,0.068937,0.554451,1.084751]],[[12,6,42,-2,-2,3,-2,-2,5,82,-2,-2,8,-2,-2],[0.2580000013113022,0.4425000101327896,0.2928571552038193,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.02850000001490116,0.39230769872665405,-2.0,-2.0,0.027499999850988388,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002858,0.056553,0.156152,1.728842,0.479926,0.019424,1.134736,-0.199731,-0.007218,0.020086,0.419799,-0.898365,-0.016892,-1.153515,-0.148292]],[[70,16,76,-2,-2,9,-2,-2,42,9,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8680000007152557,-2.0,-2.0,0.30414746701717377,0.9914999902248383,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000418,-0.006879,-0.001961,-0.139487,0.269208,-0.085202,-1.34694,-0.748872,0.048429,0.092774,0.701946,1.255027,-0.988096]],[[5,44,12,-2,-2,6,-2,-2,2,42,-2,-2,3,-2,-2],[0.8569999933242798,0.5,0.6574999988079071,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.9850000143051147,0.1339285746216774,-2.0,-2.0,0.024500000290572643,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002149,0.006653,0.029439,0.792599,-0.125898,-0.014382,1.018235,-0.203812,-0.031513,0.020527,0.85825,-0.225495,-0.067941,-1.236357,-0.78644]],[[12,4,-2,6,-2,-2,6,70,-2,-2,12,-2,-2],[0.2580000013113022,0.013000000268220901,-2.0,0.792000025510788,-2.0,-2.0,0.8944999873638153,0.5,-2.0,-2.0,0.3110000044107437,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.001069,0.059156,-0.534118,0.082654,1.350305,0.345073,-0.005668,-0.023337,-0.435674,0.517551,0.012413,-0.7808,0.251417]],[[79,-2,48,2,-2,-2,6,-2,-2],[0.232051283121109,-2.0,0.5,0.9350000023841858,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,-1,3,4,-1,-1,7,-1,-1],[2,-1,6,5,-1,-1,8,-1,-1],[0.000652,-1.676095,0.001965,-0.002359,0.368454,-0.138956,0.056298,-0.282911,0.884291]],[[55,3,56,-2,-2,-2,5,-2,7,-2,-2],[0.5,0.99549999833107,0.5,-2.0,-2.0,-2.0,0.15850000083446503,-2.0,0.029000001028180122,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.001709,0.005924,0.007957,0.127218,-0.534699,-1.297719,-0.046836,-1.26742,-0.025116,-1.096495,-0.506531]],[[38,8,8,-2,-2,5,-2,-2,42,83,-2,-2,12,-2,-2],[0.5,0.9855000078678131,0.9814999997615814,-2.0,-2.0,0.26099999248981476,-2.0,-2.0,0.22474747896194458,2.5,-2.0,-2.0,0.4919999986886978,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001676,-0.004304,0.002945,-0.021453,1.280222,-0.03938,-1.06792,-0.209127,0.031965,0.06247,1.168488,-0.287809,-0.042172,0.384882,-1.108054]],[[54,4,4,-2,-2,7,-2,-2,-2],[0.5,0.06000000052154064,0.051500000059604645,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.008351,0.004811,0.023967,0.187076,1.697886,-0.007308,0.334791,-0.29829,0.885259]],[[6,12,4,-2,-2,4,-2,-2,79,10,-2,-2,3,-2,-2],[0.8840000033378601,0.2629999965429306,0.023000000044703484,-2.0,-2.0,0.03749999962747097,-2.0,-2.0,0.8660714328289032,0.3693181872367859,-2.0,-2.0,0.20750000327825546,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007271,-0.009373,0.073891,-0.367829,1.169383,-0.016307,0.114112,-0.432414,0.024596,0.006378,-0.153693,0.611352,0.075253,0.425308,1.337879]],[[71,76,10,-2,-2,3,-2,-2,-2],[0.5,0.5,0.22500000149011612,-2.0,-2.0,0.051500000059604645,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005027,0.007043,-0.00171,-0.270428,0.304797,0.034303,-0.089172,0.61908,-0.862995]],[[5,4,77,-2,-2,54,-2,-2,9,-2,69,-2,-2],[0.9905000030994415,0.05849999934434891,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9945000112056732,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003296,0.001029,0.023381,0.141628,1.234952,-0.011186,-0.210012,0.737302,-0.056701,-0.062099,-0.085265,-0.438433,-1.211299]],[[20,2,-2,-2,19,11,-2,-2,42,-2,-2],[0.5,0.9249999821186066,-2.0,-2.0,0.5,0.36666667461395264,-2.0,-2.0,0.3166666775941849,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[-0.000977,-0.067605,0.625921,-0.994345,0.002667,0.007973,-0.023113,0.590677,-0.040677,-1.013838,-0.342282]],[[17,16,19,-2,-2,82,-2,-2,37,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003278,0.007524,0.012496,0.198426,-0.844507,-0.065972,-0.398737,-1.762704,-0.07875,-1.267379,-0.435182]],[[2,9,12,-2,-2,22,-2,-2,85,5,-2,-2,-2],[0.9600000083446503,0.7679999768733978,0.7354999780654907,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.992499977350235,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001778,0.014369,-0.022368,-0.774572,0.772479,0.030535,0.437667,-1.094212,-0.014867,-0.007603,0.026825,-0.945116,-0.942287]],[[42,55,5,-2,-2,5,-2,-2,2,-2,-2],[0.612500011920929,0.5,0.9835000038146973,-2.0,-2.0,0.15850000083446503,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003916,0.005837,0.009905,0.173131,-0.373441,-0.043572,-1.274524,-0.850439,-0.098706,-1.763328,-1.200997]],[[15,44,2,-2,-2,4,-2,-2,12,8,-2,-2,12,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.5399999916553497,0.16649999469518661,-2.0,-2.0,0.601500004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000774,-0.004499,0.0075,0.391554,-0.404656,-0.020726,0.64977,-0.362314,0.041497,0.077924,-0.006935,0.998183,0.008801,-0.55479,0.903205]],[[79,-2,5,69,-2,-2,8,-2,-2],[0.23538461327552795,-2.0,0.9835000038146973,2.5,-2.0,-2.0,0.9864999949932098,-2.0,-2.0],[1,-1,3,4,-1,-1,7,-1,-1],[2,-1,6,5,-1,-1,8,-1,-1],[0.002885,-1.558709,0.004264,0.009467,0.04962,0.469925,-0.039633,-1.382534,-0.045476]],[[64,77,21,-2,-2,8,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9440000057220459,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001482,-0.004905,-0.010554,-0.197266,0.792196,0.045103,0.132358,1.312926,0.756991]],[[5,82,69,-2,-2,53,-2,-2,65,81,-2,-2,-2],[0.02949999924749136,0.39230769872665405,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.38750000298023224,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00263,0.01913,0.027294,0.291373,0.807968,-0.10153,1.104728,-1.229943,-0.010953,-0.016401,-0.23188,0.769586,0.615788]],[[20,5,-2,-2,63,78,-2,-2,-2],[0.5,0.8659999966621399,-2.0,-2.0,0.5,0.5505050718784332,-2.0,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,-1],[4,3,-1,-1,8,7,-1,-1,-1],[0.000981,-0.069666,-0.25128,-1.155831,0.005493,0.001658,-0.231802,0.162866,0.678647]],[[37,9,82,-2,-2,78,-2,-2,3,81,-2,-2,3,-2,-2],[1.5,0.99549999833107,0.44949495792388916,-2.0,-2.0,0.5192307829856873,-2.0,-2.0,0.9200000166893005,0.17424242943525314,-2.0,-2.0,0.9784999787807465,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001953,-0.007854,0.003332,-0.004091,1.219011,-0.049293,-1.13916,-0.365115,0.024542,0.01163,0.416003,-0.348864,0.100255,1.524753,0.225191]]]
UMBRAL = 0.26
UMBRAL_LARGAS = 0.18
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
REGLA_CUENTA_MAL = False  # v6: no mejoraba y depende de una señal floja fuera de H
PROB_REGLA = 0.5


def aplicar_reglas(p, feats):
    if REGLA_CUENTA_MAL and (feats.get("verif_cuenta_mal", 0) > 0 or feats.get("fecha_plazo_falso", 0) > 0):
        return max(p, PROB_REGLA)
    return p


def prob_caso(caso):
    feats = caracteristicas(caso)
    return aplicar_reglas(probabilidad(feats), feats)


class Guardian:
    def score(self, caso):
        try:
            p = prob_caso(caso)
            # v9: corte distinto para respuestas LARGAS (2+ oraciones), como
            # sugirió el profesor. Se sigue puntuando la respuesta ENTERA (como
            # v4 y v6); solo cambia dónde se corta. El corte de largas se elige
            # con respuestas reales de varias oraciones del train.
            n_oraciones = len((caso.get("senales") or {}).get("oraciones") or [])
            return reescalar(p, UMBRAL_LARGAS if n_oraciones > 1 else UMBRAL)
        except Exception:
            # Ante un caso raro, mejor dar la alarma (un escape cuesta 4 veces más)
            return 1.0
