"""El Guardián — bot de joseca (versión 7).

v7 = v6 + cuatro señales sobre los fallos COMUNES de v4 y v6 (casi todos
falsas alarmas): opinión atribuida igual que en la fuente («según X...»),
LLM negativo pero con la cuenta verificada o el NLI alto, respuesta en
condicional que conserva la cautela, y «X de los Y» / «equivale a N» que
cuadran. En el banco de pruebas empata con v6 (H +0,643 frente a +0,635,
dentro del ruido) y acierta 2 de los 64 fallos comunes.

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


# ---------------------------------------------------------------------------
# Fallos comunes de v4 y v6 (v7)
# ---------------------------------------------------------------------------
# Las dos versiones fallan los mismos ~60 casos de H, casi todos FALSAS
# ALARMAS. Estas señales describen esos patrones; el modelo aprende cuánto
# pesan (no son reglas fijas).

# Verbos con los que una fuente o una respuesta ATRIBUYE una opinión o una
# causa a alguien: «según el sindicato...», «la emisora atribuye la subida a...»
RE_ATRIBUCION = re.compile(r"\b(segun|atribuye\w*|achaca\w*|explica\w*|afirma\w*|considera\w*|senala\w*|"
                           r"asegura\w*|advierte\w*|sostiene\w*|apunta\w*|relaciona\w*|vincula\w*|cree\w*)\b")
# Condicional del verbo («sumarían», «quedarían», «terminaría») o «si ...»
RE_COND_RESPUESTA = re.compile(r"\b\w{3,}(aria|eria|iria|arian|erian|irian)\b|\bsi (el|la|los|las|se)\b")


def atribucion_respaldada(caso):
    """La respuesta atribuye algo a alguien («según X», «X atribuye...») y una
    fuente relevante también lo presenta como atribuido, con el mismo contenido.
    Así «según el sindicato, la mejora se explica por...» no es una causa
    inventada, sino una causa contada tal como la cuenta la fuente."""
    resp = normalizar(caso.get("respuesta") or "")
    if not RE_ATRIBUCION.search(resp):
        return 0.0
    raices_r = raices_contenido(resp)
    fuentes = caso.get("fuentes") or []
    for i in fuentes_relevantes(caso, 3):
        fn = normalizar(fuentes[i])
        if RE_ATRIBUCION.search(fn) and len(raices_contenido(fn) & raices_r) >= 3:
            return 1.0
    return 0.0


def condicional_conservado(caso):
    """La respuesta va en condicional o con «si ...» (conserva la cautela) y su
    fuente relevante también es condicional, prevista o hipotética."""
    resp = normalizar(caso.get("respuesta") or "")
    if not RE_COND_RESPUESTA.search(resp):
        return 0.0, 0.0
    fuentes = caso.get("fuentes") or []
    for i in fuentes_relevantes(caso, 3):
        fn = normalizar(fuentes[i])
        if RE_COND_RESPUESTA.search(fn) or any(re.search(p, fn) for p in MATICES["condicional"]):
            return 1.0, 1.0
    return 1.0, 0.0


RE_PARTE_DE = re.compile(r"(\d[\d.,]*)\s+de\s+(?:los|las|sus)\s+(\d[\d.,]*)")
RE_EQUIVALE = re.compile(r"\b(equivale\w* a|supone\w*|da para|alcanza\w* para)\s+(?:unos?\s+|unas\s+)?(\d[\d.,]*)")


def cuentas_parte_y_reparto(caso):
    """Dos cuentas fieles frecuentes que el verificador no reconocía:
      - «X de los Y» con X calculado: «suman 13 de los 25 concejales» (suma de
        dos cifras), «870 de los jóvenes ya no seguían» (Y menos los que siguen);
      - «equivale a N»: «3,6 millones equivalen a 300 subvenciones de 12.000».
    Devuelve (cuentas que cuadran, cuentas que no cuadran)."""
    resp = normalizar(caso.get("respuesta") or "")
    texto_f = normalizar(" ".join(caso.get("fuentes") or []))
    vals_f = sorted({v for n in extraer_numeros(texto_f) if not n["anio"] for v in n["valores"]})[:40]
    ok = mal = 0
    for m in RE_PARTE_DE.finditer(resp):
        x, total = _valor(m.group(1)), _valor(m.group(2))
        if any(_cerca(x, v, 0.001) for v in vals_f):
            continue  # la parte está tal cual en las fuentes: no es una cuenta
        candidatos = [total - v for v in vals_f] + [a + b for a, b in combinations(vals_f, 2)]
        if any(_cerca(x, c, 0.001) for c in candidatos):
            ok += 1
        elif any(_cerca(total, v, 0.001) for v in vals_f):
            mal += 1  # el total sí está en las fuentes y la parte no sale de ninguna cuenta
    for m in RE_EQUIVALE.finditer(resp):
        x = _valor(m.group(2))
        if any(_cerca(x, v, 0.001) for v in vals_f):
            continue
        nums_r = [v for n in extraer_numeros(resp) for v in n["valores"] if v and not _cerca(v, x, 0.001)]
        cocientes = [a / b for a in nums_r + vals_f for b in nums_r + vals_f if b and a != b]
        if any(_cerca(x, c, 0.01) for c in cocientes):
            ok += 1
        else:
            mal += 1
    return ok, mal


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

    # --- 12. Fallos comunes de v4 y v6 (v7) ---
    # Opiniones o causas atribuidas igual que en la fuente
    f["v7_atribucion_respaldada"] = atribucion_respaldada(caso)
    # El LLM dice «contradicha»/«no respaldada», pero una cuenta de la frase
    # está comprobada y cuadra, o el NLI respalda con fuerza la fuente citada
    llm_neg = max(f["llm_no_respaldada"], f["llm_contradicha"])
    f["v7_llm_neg_cuenta_ok"] = llm_neg * float(f["verif_cuenta_ok"] > 0 or f["num_cuenta_correcta"] > 0)
    f["v7_llm_neg_nli_alto"] = llm_neg * float(f["nli_resp_fuente_citada_min"] >= 0.8
                                               or f["nli_resp_todas_min_frag"] >= 0.8)
    # La respuesta conserva la cautela con el condicional («sumarían», «si el Pleno...»)
    cond, cond_ok = condicional_conservado(caso)
    f["v7_respuesta_condicional"] = cond
    f["v7_condicional_conservado"] = cond_ok
    # «X de los Y» y «equivale a N»
    parte_ok, parte_mal = cuentas_parte_y_reparto(caso)
    f["v7_parte_reparto_ok"] = float(min(parte_ok, 2))
    f["v7_parte_reparto_mal"] = float(min(parte_mal, 2))

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
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas", "cb_sujeto_generalizado", "cb_sujeto_generalizado_confianza", "suj_sujeto_colectivo_respaldado", "suj_sujeto_colectivo_sin_respaldo", "cob_cobertura_mejor_fuente", "cob_cobertura_union_2", "cob_cobertura_union_3", "cob_ganancia_cobertura_2", "cob_ganancia_cobertura_3", "cob_fuentes_complementarias", "cob_llm_ok_nli_bajo_cobertura_alta", "cob_nli_bajo_cobertura_union_alta", "v7_atribucion_respaldada", "v7_llm_neg_cuenta_ok", "v7_llm_neg_nli_alto", "v7_respuesta_condicional", "v7_condicional_conservado", "v7_parte_reparto_ok"]
INICIAL = -0.035456
TASA = 0.03
ARBOLES = [[[2,3,44,-2,-2,4,-2,-2,38,15,-2,-2,79,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.28450000286102295,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003123,0.414307,0.470507,1.998345,1.099628,0.284888,1.604819,0.622315,-0.180073,-0.253411,-1.233059,0.91899,0.161875,-0.839811,1.928435]],[[2,8,53,-2,-2,7,-2,-2,8,4,-2,-2,21,-2,-2],[0.9350000023841858,0.17599999904632568,0.5,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.07600000128149986,0.08900000154972076,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002279,0.392471,0.473255,1.961457,1.354385,0.308807,1.873355,1.016757,-0.179575,0.201896,1.698077,0.236824,-0.243696,-1.112258,1.31322]],[[2,53,3,-2,-2,5,-2,-2,8,4,-2,-2,38,-2,-2],[0.9249999821186066,0.5,0.032500000670552254,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.0885000005364418,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005243,0.378655,0.433529,1.882385,1.447144,0.177293,1.790555,0.235679,-0.17735,0.172542,1.357834,-0.310141,-0.24489,-1.239634,0.339933]],[[2,53,3,-2,-2,5,-2,-2,8,4,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.07600000128149986,0.08900000154972076,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.014926,0.363458,0.404343,1.824469,1.240612,0.185197,1.765024,0.344293,-0.184432,0.171923,1.818912,0.033104,-0.244665,-1.219269,0.261744]],[[2,53,3,-2,-2,9,-2,-2,15,38,-2,-2,54,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.011937,0.351036,0.400954,1.778373,1.258593,0.144476,-0.251527,1.215573,-0.174418,-0.215029,-1.141443,0.480591,0.260474,0.090816,1.947638]],[[2,7,3,-2,-2,44,-2,-2,8,4,-2,-2,21,-2,-2],[0.9350000023841858,0.007500000298023224,0.636000007390976,-2.0,-2.0,0.5,-2.0,-2.0,0.0885000005364418,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005667,0.355389,0.417928,1.740012,0.949285,0.253166,1.262961,-0.023696,-0.173311,0.164432,1.303573,-0.291771,-0.234814,-1.091372,1.200505]],[[2,53,3,-2,-2,5,-2,-2,8,44,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.34849999845027924,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.1314999982714653,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007507,0.340971,0.38732,1.689185,1.003407,0.142233,1.639107,0.239575,-0.150136,0.147539,1.803687,0.087404,-0.212655,-1.109394,0.347225]],[[53,5,39,-2,-2,42,-2,-2,2,52,-2,-2,54,-2,-2],[0.5,0.5550000071525574,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.014674,0.327206,0.380997,1.608696,0.886331,0.005161,0.731978,-1.486328,-0.171394,-0.003342,0.301981,-1.721678,-0.294715,-1.328502,1.24649]],[[53,69,5,-2,-2,-2,38,21,-2,-2,79,-2,-2],[0.5,0.5,0.4529999941587448,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003461,0.308577,0.363511,1.603322,0.878599,-0.771881,-0.144842,-0.20373,-0.9879,1.445811,0.175266,-0.783322,1.93439]],[[2,44,3,-2,-2,7,-2,-2,38,15,-2,-2,79,-2,-2],[0.9350000023841858,0.5,0.032500000670552254,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001529,0.32047,0.351099,1.638365,1.121672,0.162459,1.26935,-0.324008,-0.14303,-0.193196,-0.97564,1.000399,0.147634,-0.736822,1.800845]],[[53,69,3,-2,-2,-2,21,65,-2,-2,-2],[0.5,0.5,0.032500000670552254,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.001582,0.292323,0.334479,1.60602,0.982725,-0.744911,-0.138627,-0.173047,-0.907707,1.038308,1.499262]],[[2,5,81,-2,-2,-2,38,21,-2,-2,79,-2,-2],[0.9350000023841858,0.6689999997615814,0.1180555559694767,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000323,0.304806,0.329099,1.578487,1.1647,0.392316,-0.140379,-0.201889,-0.957553,1.438315,0.13157,-0.986917,1.710728]],[[2,53,3,-2,-2,9,-2,-2,8,4,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.33650000393390656,-2.0,-2.0,0.9480000138282776,-2.0,-2.0,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001779,0.285006,0.317937,1.472528,0.74283,0.149298,-0.198574,1.093166,-0.129464,0.164552,1.242252,-0.190702,-0.185088,-0.972742,0.373446]],[[2,8,81,-2,-2,44,-2,-2,8,4,-2,-2,15,-2,-2],[0.9350000023841858,0.153499998152256,0.2792207896709442,-2.0,-2.0,0.5,-2.0,-2.0,0.09149999916553497,0.21649999916553497,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004178,0.268529,0.346245,1.552082,1.211355,0.184835,1.008359,0.057641,-0.131954,0.150648,1.322774,0.032173,-0.185992,-0.904976,0.873749]],[[2,8,7,-2,-2,80,-2,-2,38,21,-2,-2,79,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,-2.0,-2.0,0.7573529481887817,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[1.4e-05,0.274198,0.328576,1.483744,1.030262,0.212731,0.436332,1.164281,-0.12862,-0.181897,-0.90188,1.321841,0.118462,-0.858375,1.630541]],[[2,3,44,-2,-2,4,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.44350001215934753,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00712,0.26307,0.311147,1.453808,0.940293,0.178089,1.081879,0.005114,-0.130515,-0.187494,-0.963586,0.97106,0.143125,1.290452,-1.425119]],[[53,5,46,-2,-2,42,-2,-2,32,15,-2,-2,80,-2,-2],[0.5,0.5550000071525574,1.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004762,0.24969,0.302844,1.274829,1.888343,-0.019707,0.429777,-1.405138,-0.122456,-0.16335,-0.885068,0.95447,0.230085,-0.013361,1.750538]],[[2,44,3,-2,-2,5,-2,-2,8,4,-2,-2,76,-2,-2],[0.9350000023841858,0.5,0.25300000607967377,-2.0,-2.0,0.023000000044703484,-2.0,-2.0,0.09149999916553497,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003435,0.254304,0.278355,1.367322,0.858032,0.136894,1.519034,0.011637,-0.114305,0.157601,1.358109,-0.270281,-0.167765,-0.970722,0.163026]],[[2,5,42,-2,-2,1,-2,-2,31,15,-2,-2,-2],[0.9350000023841858,0.14750000089406967,0.44949495792388916,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002164,0.237944,0.279704,1.331254,0.55187,0.138474,-0.011033,1.207712,-0.109473,-0.134305,-0.719394,0.871724,1.40898]],[[53,69,46,-2,-2,-2,65,43,-2,-2,80,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001804,0.24044,0.273757,1.173685,1.82124,-0.398635,-0.105918,-0.144627,-0.794604,0.822465,0.232183,-0.039766,1.683464]],[[2,19,16,-2,-2,12,-2,-2,0,1,-2,-2,10,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.5444999933242798,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.3541666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003928,0.139688,0.181565,0.923653,-0.952091,-0.367641,-1.236105,-1.925671,-0.178055,-0.203376,-0.955759,1.999841,0.407688,1.583597,1.746231]],[[2,7,3,-2,-2,80,-2,-2,76,11,-2,-2,42,-2,-2],[0.9350000023841858,0.007500000298023224,0.013500000350177288,-2.0,-2.0,0.7638888955116272,-2.0,-2.0,0.5,0.1180555559694767,-2.0,-2.0,0.1180555559694767,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002819,0.231244,0.277559,1.253906,1.430025,0.159494,-0.109567,1.086966,-0.104337,-0.160727,-0.979467,0.181581,0.072448,1.300657,-0.30821]],[[53,5,46,-2,-2,1,-2,-2,2,52,-2,-2,65,-2,-2],[0.5,0.6629999876022339,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003034,0.208066,0.254121,1.129957,1.754262,-0.055269,-0.738746,0.794463,-0.104152,0.02714,0.383786,-1.640239,-0.204981,-1.041587,0.822665]],[[2,8,2,-2,-2,76,-2,-2,38,8,-2,-2,79,-2,-2],[0.9350000023841858,0.15200000256299973,0.8250000178813934,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.05649999901652336,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004294,0.213671,0.263245,1.273301,1.512995,0.164008,0.539226,1.1826,-0.09642,-0.144521,0.540142,-0.799331,0.148896,-0.545485,1.565738]],[[2,8,2,-2,-2,9,-2,-2,8,4,-2,-2,79,-2,-2],[0.9350000023841858,0.153499998152256,0.8250000178813934,-2.0,-2.0,0.9410000145435333,-2.0,-2.0,0.0755000002682209,0.9280000030994415,-2.0,-2.0,0.8708333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00886,0.206773,0.256811,1.262062,1.476961,0.15636,0.296996,1.002611,-0.107,0.13044,0.968315,-0.3588,-0.146181,-0.920754,0.063739]],[[53,69,46,-2,-2,-2,21,31,-2,-2,78,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5357142984867096,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002268,0.214331,0.242976,1.114707,1.733247,-0.554137,-0.091453,-0.120189,-0.658143,1.297826,0.303885,0.548673,1.699722]],[[2,8,7,-2,-2,76,-2,-2,32,76,-2,-2,80,-2,-2],[0.9350000023841858,0.17599999904632568,0.035499999299645424,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003585,0.187355,0.241078,1.300551,0.673844,0.132711,0.371299,1.127036,-0.090818,-0.126779,-0.844035,0.301395,0.223465,0.08984,1.610054]],[[2,19,17,-2,-2,8,-2,-2,0,1,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11600000038743019,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004456,0.124067,0.166735,0.886916,-1.123575,-0.355597,-0.935436,-1.889679,-0.146508,-0.169951,-0.825941,1.882161,0.382254,1.568068,1.752092]],[[53,5,46,-2,-2,79,-2,-2,2,55,-2,-2,32,-2,-2],[0.5,0.6654999852180481,1.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006802,0.190638,0.226369,1.086189,1.65147,-0.016152,-1.307551,0.578476,-0.080647,0.032768,0.383017,-1.626968,-0.176043,-0.94098,0.715247]],[[2,19,16,-2,-2,8,-2,-2,8,5,-2,-2,76,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7374999821186066,-2.0,-2.0,0.13199999928474426,0.07250000163912773,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007199,0.106131,0.150254,0.837044,-0.837312,-0.361191,-1.304787,-1.827791,-0.153958,0.088256,0.950484,-0.687972,-0.202718,-1.138091,-0.265483]],[[2,3,44,-2,-2,6,-2,-2,15,32,-2,-2,78,-2,-2],[0.9350000023841858,0.3439999967813492,0.5,-2.0,-2.0,0.9465000033378601,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002954,0.185868,0.207149,1.226729,0.496963,0.089168,0.04772,1.486309,-0.08338,-0.109553,-0.651922,0.913065,0.214672,-0.122995,1.529917]],[[2,8,2,-2,-2,9,-2,-2,76,32,-2,-2,42,-2,-2],[0.9350000023841858,0.17599999904632568,0.8250000178813934,-2.0,-2.0,0.945000022649765,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11324786394834518,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00727,0.172295,0.219527,1.211182,1.405796,0.122088,0.050112,0.942948,-0.089339,-0.144073,-0.832357,0.858703,0.094076,1.459022,-0.318545]],[[53,69,46,-2,-2,-2,48,32,-2,-2,78,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.006095,0.168764,0.194241,0.947776,1.682191,-0.772811,-0.087742,-0.120826,-0.711085,0.849573,0.214868,-0.32643,1.696788]],[[2,19,17,-2,-2,8,-2,-2,0,15,-2,-2,12,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.7360000014305115,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2849999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.01532,0.116664,0.152657,0.847578,-1.14332,-0.337874,-1.286085,-1.754942,-0.118498,-0.137589,-0.770006,0.806269,0.338044,1.576833,1.353657]],[[2,7,8,-2,-2,1,-2,-2,38,15,-2,-2,79,-2,-2],[0.9350000023841858,0.007500000298023224,0.6410000026226044,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005338,0.163258,0.209572,1.302258,0.628516,0.092438,-0.042995,0.942194,-0.065989,-0.111441,-0.685271,0.899553,0.152682,-0.499232,1.523883]],[[53,69,46,-2,-2,-2,76,11,-2,-2,42,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.4166666716337204,-2.0,-2.0,0.1180555559694767,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001428,0.168512,0.192809,0.985879,1.578704,-0.261037,-0.07611,-0.12662,-0.795746,0.289138,0.097195,1.220069,-0.20995]],[[2,19,50,-2,-2,8,-2,-2,8,5,-2,-2,76,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.8070000112056732,-2.0,-2.0,0.08150000125169754,0.07400000095367432,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000542,0.091828,0.128691,0.737091,-1.193859,-0.294037,-1.021955,-1.756356,-0.120052,0.108702,1.024491,-0.6516,-0.159335,-1.017232,-0.036926]],[[19,2,17,-2,-2,8,-2,-2,82,55,-2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.23111110925674438,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.005193,0.034702,0.120671,0.725942,-1.255267,-0.08136,0.681421,-0.611122,-0.293274,-0.312399,-1.702539,-1.431938,-0.912546]],[[8,6,2,-2,-2,44,-2,-2,2,19,-2,-2,79,-2,-2],[0.08349999785423279,0.2889999896287918,0.4000000059604645,-2.0,-2.0,0.5,-2.0,-2.0,0.9750000238418579,0.5,-2.0,-2.0,0.8660714328289032,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00435,0.171558,0.259812,1.241009,1.399973,0.13928,1.141547,0.173961,-0.04946,0.030795,0.352198,-1.534462,-0.152158,-1.014236,-0.198248]],[[2,19,16,-2,-2,8,-2,-2,8,6,-2,-2,76,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.6270000040531158,-2.0,-2.0,0.04649999924004078,0.9165000021457672,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.008179,0.098834,0.133479,0.774764,-0.957141,-0.27676,-0.790595,-1.684593,-0.113825,0.180958,1.410458,0.039988,-0.145999,-0.933418,-0.03186]],[[2,19,79,-2,-2,8,-2,-2,8,46,-2,-2,80,-2,-2],[0.9600000083446503,0.5,0.6306818127632141,-2.0,-2.0,0.6270000040531158,-2.0,-2.0,0.05250000022351742,1.5,-2.0,-2.0,0.8708333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005646,0.087328,0.117041,-0.195564,0.828902,-0.254469,-0.722736,-1.642484,-0.108308,0.133043,0.021795,1.399277,-0.139697,-1.01594,-0.084042]],[[53,69,46,-2,-2,-2,70,21,-2,-2,79,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003017,0.142115,0.161742,0.871089,1.526195,-0.304448,-0.072546,-0.10777,-0.646612,1.267979,0.143169,-0.430404,1.656015]],[[2,7,44,-2,-2,37,-2,-2,79,8,-2,-2,2,-2,-2],[0.9350000023841858,0.001500000071246177,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.8708333373069763,0.07400000095367432,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007589,0.142107,0.191358,1.229024,1.15268,0.115955,0.316027,0.951386,-0.078276,-0.13004,0.280148,-0.819785,0.047549,0.936884,-0.532037]],[[2,8,44,-2,-2,9,-2,-2,31,21,-2,-2,-2],[0.9350000023841858,0.153499998152256,0.5,-2.0,-2.0,0.9679999947547913,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002672,0.141067,0.18008,1.206731,1.305697,0.100358,0.02131,1.001401,-0.067208,-0.087746,-0.53279,1.10385,1.135284]],[[8,4,2,-2,-2,44,-2,-2,79,2,-2,-2,38,-2,-2],[0.1314999982714653,0.060499999672174454,0.42500001192092896,-2.0,-2.0,0.5,-2.0,-2.0,0.7573529481887817,0.9350000023841858,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005567,0.141126,0.241508,1.201951,1.381512,0.100234,1.063903,-0.055408,-0.04616,-0.147806,0.102458,-0.954309,0.039918,-0.097103,1.618702]],[[53,46,2,-2,-2,1,-2,-2,54,70,-2,-2,79,-2,-2],[0.5,1.5,0.9850000143051147,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.6904762089252472,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.009615,0.127144,0.107411,0.810507,-1.068702,0.312864,1.232926,1.743135,-0.072679,-0.09619,-0.65934,0.509201,0.255397,0.607785,1.482164]],[[53,3,46,-2,-2,7,-2,-2,48,38,-2,-2,7,-2,-2],[0.5,0.25300000607967377,1.5,-2.0,-2.0,0.8955000042915344,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.031000000424683094,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006371,0.129111,0.161207,0.965533,1.482902,-0.007611,0.374296,-1.366254,-0.05079,-0.080585,-0.599395,0.564365,0.234307,1.485299,0.333824]],[[53,69,46,-2,-2,-2,79,2,-2,-2,38,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,-2.0,0.6449579894542694,0.9249999821186066,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.004597,0.125507,0.151755,0.898594,1.441349,-0.746493,-0.065337,-0.191412,0.795789,-1.090238,-0.016488,-0.322183,1.106893]],[[19,2,17,-2,-2,8,-2,-2,9,-2,8,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.6550000011920929,-2.0,0.9424999952316284,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.008138,0.033948,0.096592,0.636493,-1.097761,-0.070549,0.756314,-0.599775,-0.240463,-0.886862,-0.26253,-1.494351,-1.303352]],[[2,5,45,-2,-2,1,-2,-2,79,21,-2,-2,2,-2,-2],[0.9350000023841858,0.023000000044703484,0.7999999821186066,-2.0,-2.0,0.5,-2.0,-2.0,0.8708333373069763,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007126,0.112984,0.154469,1.167464,0.402682,0.078914,0.037605,0.889733,-0.062348,-0.11151,-0.651029,0.798384,0.05561,0.871827,-0.444889]],[[19,2,16,-2,-2,8,-2,-2,4,2,-2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.9664999842643738,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001328,0.029299,0.103854,0.698846,-1.162344,-0.071866,0.686327,-0.59687,-0.241455,-0.262459,-1.537436,-1.313876,-0.878236]],[[2,8,44,-2,-2,9,-2,-2,32,15,-2,-2,79,-2,-2],[0.9350000023841858,0.153499998152256,0.5,-2.0,-2.0,0.4490000009536743,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002934,0.117746,0.149503,1.134119,1.269722,0.086219,-0.648401,0.629049,-0.06244,-0.08892,-0.596411,0.725031,0.230403,0.296802,1.619931]],[[19,2,79,-2,-2,8,-2,-2,9,-2,2,-2,-2],[0.5,0.9750000238418579,0.8090909123420715,-2.0,-2.0,0.1339999958872795,-2.0,-2.0,0.6200000047683716,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004517,0.021629,0.08364,0.085858,0.965359,-0.081576,0.486251,-0.673046,-0.223538,-0.851999,-0.246175,-1.504476,-1.274499]],[[19,2,79,-2,-2,8,-2,-2,12,81,-2,-2,-2],[0.5,0.9750000238418579,0.8090909123420715,-2.0,-2.0,0.05250000022351742,-2.0,-2.0,0.6444999873638153,0.16333333402872086,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002027,0.027267,0.089489,0.07733,1.099773,-0.071243,0.72819,-0.560586,-0.217415,-0.192192,-1.315602,-0.817134,-1.563027]],[[19,2,78,-2,-2,8,-2,-2,9,-2,2,-2,-2],[0.5,0.9750000238418579,0.4226190447807312,-2.0,-2.0,0.13199999928474426,-2.0,-2.0,0.5785000026226044,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.008198,0.03256,0.08903,-0.498129,0.706564,-0.064675,0.753614,-0.600989,-0.221306,-0.615266,-0.250932,-1.583879,-1.251408]],[[2,3,77,-2,-2,4,-2,-2,15,76,-2,-2,9,-2,-2],[0.9350000023841858,0.04749999940395355,0.5,-2.0,-2.0,0.3410000056028366,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9814999997615814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000323,0.105309,0.131497,0.948013,1.295509,0.055044,0.826506,-0.393537,-0.049368,-0.074044,-0.662878,0.326571,0.211688,0.242832,1.664923]],[[19,2,37,-2,-2,79,-2,-2,12,80,-2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.8708333373069763,-2.0,-2.0,0.6349999904632568,0.7416666746139526,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001077,0.019968,0.118059,0.629082,1.059399,-0.02834,-0.416166,0.546331,-0.212843,-0.192158,-1.35486,-0.995813,-1.536232]],[[19,2,8,-2,-2,38,-2,-2,9,-2,2,-2,-2],[0.5,0.9350000023841858,0.9359999895095825,-2.0,-2.0,0.5,-2.0,-2.0,0.6190000176429749,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002363,0.024226,0.120293,0.815732,1.126534,-0.023044,-0.315462,0.696886,-0.203371,-0.663867,-0.228154,-1.478752,-1.237269]],[[2,55,50,-2,-2,-2,18,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000418,0.066537,0.089668,0.612267,-1.069865,-1.43965,-0.085099,-0.104001,-0.663986,1.763331,0.998198]],[[19,2,79,-2,-2,46,-2,-2,3,-2,2,-2,-2],[0.5,0.9750000238418579,0.8153409063816071,-2.0,-2.0,1.5,-2.0,-2.0,0.007500000298023224,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001504,0.022556,0.073216,0.031258,0.960159,-0.064291,-0.558335,0.817431,-0.201618,-0.819118,-0.228775,-1.49427,-1.247435]],[[2,5,69,-2,-2,-2,76,11,-2,-2,4,-2,-2],[0.9350000023841858,0.6689999997615814,0.5,-2.0,-2.0,-2.0,0.5,0.4166666716337204,-2.0,-2.0,0.08100000023841858,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00445,0.103875,0.115529,0.842748,1.083853,-0.036904,-0.043902,-0.083611,-0.658964,0.39063,0.077506,1.222976,-0.31897]],[[2,22,17,-2,-2,-2,51,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.002737,0.054173,0.075225,0.553395,-1.189218,-1.301883,-0.067199,-0.087181,-0.585133,1.699405,1.033518]],[[2,39,77,-2,-2,-2,31,76,-2,-2,-2],[0.9350000023841858,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.001841,0.088892,0.096094,0.63585,1.268903,-0.706625,-0.041569,-0.060812,-0.594159,0.412415,1.365077]],[[21,53,46,-2,-2,76,-2,-2,78,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4833333343267441,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.006751,-0.008475,0.088951,0.5265,1.38075,-0.058742,-0.563473,0.458194,0.2604,0.367539,1.562262]],[[19,5,16,-2,-2,2,-2,-2,22,6,-2,-2,83,-2,-2],[0.5,0.8260000050067902,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,0.5,0.7094999849796295,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003765,0.016702,0.055475,0.457696,-1.060012,-0.089018,0.232327,-0.870343,-0.187305,-0.217655,-1.241283,-1.498781,-0.160623,-1.278782,-0.713751]],[[8,4,78,-2,-2,1,-2,-2,79,21,-2,-2,63,-2,-2],[0.1314999982714653,0.8700000047683716,0.2899159789085388,-2.0,-2.0,0.5,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006262,0.093225,0.156688,0.14144,1.083647,0.031524,-0.389858,1.159734,-0.043989,-0.128655,-0.881158,0.705446,0.029138,-0.020643,1.651274]],[[19,2,79,-2,-2,46,-2,-2,55,69,-2,-2,5,-2,-2],[0.5,0.9750000238418579,0.6339712738990784,-2.0,-2.0,1.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.24950000643730164,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003295,0.023745,0.067024,-0.376548,0.657806,-0.049716,-0.478618,0.856765,-0.189594,-0.21735,-1.426787,-1.21075,-0.164742,-1.271979,-1.183397]],[[8,19,4,-2,-2,-2,48,76,-2,-2,78,-2,-2],[0.1314999982714653,0.5,0.8700000047683716,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5916666686534882,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.003869,0.095974,0.115369,1.080699,0.443813,-1.09401,-0.043915,-0.063079,-0.640419,0.374718,0.20702,0.145813,1.601255]],[[16,79,2,-2,-2,2,-2,-2,9,-2,-2],[0.5,0.8153409063816071,0.9350000023841858,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000552,0.014108,-0.036905,0.602419,-0.476979,0.084689,0.890009,-0.305405,-0.257507,-1.908577,-0.900127]],[[19,2,39,-2,-2,80,-2,-2,9,-2,2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.8708333373069763,-2.0,-2.0,0.6200000047683716,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001647,0.020346,0.105866,0.944567,-0.445237,-0.023446,-0.385862,0.400811,-0.17201,-0.622349,-0.201787,-1.51324,-1.175075]],[[53,46,7,-2,-2,8,-2,-2,54,32,-2,-2,-2],[0.5,1.5,0.007500000298023224,-2.0,-2.0,0.06750000081956387,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.012634,0.072251,0.051511,0.968112,-0.155336,0.284356,1.334524,1.798746,-0.053682,-0.072646,-0.556417,0.832577,1.148426]],[[19,79,8,-2,-2,50,-2,-2,3,-2,55,-2,-2],[0.5,0.6449579894542694,0.0755000002682209,-2.0,-2.0,0.5,-2.0,-2.0,0.007500000298023224,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002424,0.019123,-0.073174,0.639567,-0.778333,0.053807,0.423371,-1.517443,-0.159049,-0.650643,-0.184671,-1.442343,-1.187764]],[[8,79,-2,5,-2,-2,76,43,-2,-2,12,-2,-2],[0.1314999982714653,0.3457317054271698,-2.0,0.07349999994039536,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.38849999010562897,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.001651,0.091332,-0.983935,0.10033,0.907171,0.126699,-0.03794,-0.072998,-0.602513,0.815154,0.070208,-0.566046,0.664764]],[[55,78,53,-2,-2,43,-2,-2,5,-2,7,-2,-2],[0.5,0.5505050718784332,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.15850000083446503,-2.0,0.029000001028180122,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002726,0.017532,-0.058014,0.56554,-0.629913,0.055268,0.237518,1.329871,-0.181588,-1.478064,-0.14556,-1.243265,-1.064329]],[[2,22,17,-2,-2,-2,38,46,-2,-2,42,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,-2.0,0.5,1.5,-2.0,-2.0,0.2792207896709442,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002945,0.048987,0.067128,0.549063,-1.172145,-1.390007,-0.06186,-0.091093,-0.744128,0.830237,0.122982,1.235872,-0.998148]],[[8,79,-2,4,-2,-2,76,21,-2,-2,42,-2,-2],[0.1314999982714653,0.3457317054271698,-2.0,0.8700000047683716,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002607,0.083062,-1.006486,0.091123,0.958858,0.339929,-0.036535,-0.06942,-0.536684,0.960449,0.065868,0.966422,-0.209253]],[[48,64,53,-2,-2,-2,78,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.00706,-0.007216,-0.01945,0.562916,-0.406895,1.354275,0.195106,-0.068351,1.392016]],[[16,2,79,-2,-2,10,-2,-2,9,-2,-2],[0.5,0.9750000238418579,0.7928571403026581,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.01031,0.001949,0.044419,-0.141299,0.778804,-0.068025,-0.69093,0.250603,-0.223592,-1.55382,-0.873176]],[[21,2,22,-2,-2,1,-2,-2,43,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.011817,-0.001166,0.045384,0.419204,-1.173184,-0.062399,-0.461841,1.602291,0.23785,0.826499,1.445521]],[[19,17,16,-2,-2,79,-2,-2,3,-2,55,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.612500011920929,-2.0,-2.0,0.007500000298023224,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.007909,0.024513,0.037712,0.330601,-1.045997,-0.186242,-0.200263,-1.516006,-0.15244,-0.588115,-0.176175,-1.410944,-1.17311]],[[19,17,10,-2,-2,9,-2,-2,12,5,-2,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.9230000078678131,-2.0,-2.0,0.6504999995231628,0.09849999845027924,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000917,0.017411,0.029373,-0.105243,0.628451,-0.187024,-1.500641,-0.458515,-0.139007,-0.118405,-1.069566,-0.897994,-1.430586]],[[16,19,79,-2,-2,9,-2,-2,12,-2,-2],[0.5,0.5,0.6504524946212769,-2.0,-2.0,0.6200000047683716,-2.0,-2.0,0.5520000159740448,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002426,0.014282,0.031499,-0.32706,0.40578,-0.143314,-0.510085,-1.297843,-0.223471,-0.82935,-1.636387]],[[8,4,3,-2,-2,7,-2,-2,79,43,-2,-2,70,-2,-2],[0.09149999916553497,0.8700000047683716,0.021499999798834324,-2.0,-2.0,0.02650000061839819,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001399,0.092066,0.152188,0.962533,1.40216,0.035433,0.701401,-0.787197,-0.033782,-0.097383,-0.726967,0.599367,0.021887,-0.133828,1.477077]],[[15,5,78,-2,-2,76,-2,-2,8,7,-2,-2,-2],[0.5,0.08300000056624413,0.8452380895614624,-2.0,-2.0,0.5,-2.0,-2.0,0.45000000298023224,0.04650000110268593,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.005667,-0.023823,0.048794,0.161531,1.5169,-0.067247,-0.668893,0.178424,0.132658,0.071838,0.908606,-0.440989,1.432795]],[[38,21,53,-2,-2,43,-2,-2,79,70,-2,-2,53,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001179,-0.020765,-0.034121,0.463263,-0.46669,0.193181,0.458219,1.387109,0.105088,-0.036517,1.071586,-0.590788,0.218717,0.967732,1.491649]],[[32,53,46,-2,-2,76,-2,-2,12,-2,12,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.408500000834465,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000209,-0.015514,0.059449,0.378729,1.31468,-0.052387,-0.554256,0.366941,0.146688,1.699005,0.086986,-0.219665,1.223701]],[[2,22,16,-2,-2,81,-2,-2,21,1,-2,-2,-2],[0.9600000083446503,0.5,0.5,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001274,0.047038,0.067092,0.550076,-1.020523,-0.191362,-1.07638,-1.439043,-0.060442,-0.074532,-0.556548,1.551032,1.103108]],[[21,53,69,-2,-2,76,-2,-2,48,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.006922,-0.01707,0.054677,0.667283,-0.490888,-0.052727,-0.533659,0.260325,0.192139,0.397722,1.418151]],[[17,2,16,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.12132352963089943,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.006898,0.004361,0.046117,0.433779,-1.325708,-0.051242,0.163919,-0.633338,-0.202416,-1.554922,-0.91542]],[[19,5,17,-2,-2,2,-2,-2,9,-2,2,-2,-2],[0.5,0.9715000092983246,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.6209999918937683,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001862,0.016478,0.03617,0.318844,-0.865957,-0.096257,0.127092,-1.103124,-0.134962,-0.434774,-0.155454,-1.406246,-1.127695]],[[22,78,11,-2,-2,54,-2,-2,52,-2,5,-2,-2],[0.5,0.4686274528503418,0.20714285969734192,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,0.26350000500679016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000774,0.011144,-0.066174,-0.753849,0.505466,0.03433,0.146059,1.409162,-0.149546,-1.24797,-0.1076,-1.06197,-1.117717]],[[19,5,17,-2,-2,2,-2,-2,12,6,-2,-2,-2],[0.5,0.6755000054836273,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.6359999775886536,0.9239999949932098,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000983,0.014962,0.043618,0.385871,-0.955974,-0.053262,0.225953,-0.966098,-0.13713,-0.116479,-0.872129,-1.241916,-1.41768]],[[43,53,46,-2,-2,65,-2,-2,12,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5394999980926514,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.007413,-0.004289,0.057304,0.395786,1.317514,-0.037356,-0.375903,0.779591,0.180773,1.334992,-0.071875]],[[80,8,4,-2,-2,21,-2,-2,69,2,-2,-2,4,-2,-2],[0.8708333373069763,0.09149999916553497,0.5805000066757202,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.026499999687075615,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002617,-0.037494,0.062176,0.873907,-0.01464,-0.073533,-0.582214,0.634421,0.064522,0.026142,0.588134,-0.697006,0.202811,1.712299,0.639409]],[[79,2,78,-2,-2,15,-2,-2,69,2,-2,-2,4,-2,-2],[0.8708333373069763,0.9350000023841858,0.7071428596973419,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.09049999713897705,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007001,-0.020859,0.063896,0.787148,-0.060536,-0.056429,-0.487612,0.659508,0.067979,0.021531,0.541478,-0.711941,0.241066,1.84324,0.711297]],[[1,76,11,-2,-2,42,-2,-2,46,8,-2,-2,-2],[0.5,0.5,0.4166666716337204,-2.0,-2.0,0.12132352963089943,-2.0,-2.0,1.5,0.8294999897480011,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.005702,-0.015821,-0.043954,-0.498435,0.453753,0.073637,1.248089,-0.100968,0.096762,0.0856,0.983749,1.062252,1.538127]],[[19,16,10,-2,-2,9,-2,-2,12,5,-2,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.6444999873638153,0.09849999845027924,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000346,0.013827,0.024233,-0.096632,0.619654,-0.155818,-1.477565,0.027959,-0.122864,-0.10096,-0.989636,-0.873032,-1.399675]],[[17,19,2,-2,-2,55,-2,-2,9,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.5,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003336,0.006951,0.021888,0.418372,-0.288264,-0.131381,-1.300751,-1.070876,-0.196256,-1.536907,-0.868901]],[[78,8,5,-2,-2,81,-2,-2,43,65,-2,-2,-2],[0.5505050718784332,0.0625,0.024500000290572643,-2.0,-2.0,0.3798076957464218,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000443,-0.058356,0.074951,1.055143,-0.425027,-0.092911,-0.716364,0.750057,0.030021,0.015255,-0.002312,1.093606,1.291077]],[[2,4,8,-2,-2,3,-2,-2,21,79,-2,-2,-2],[0.9350000023841858,0.44200000166893005,0.9690000116825104,-2.0,-2.0,0.06600000336766243,-2.0,-2.0,0.5,0.8708333373069763,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000338,0.057833,0.098321,0.849034,1.108541,0.025965,0.78946,-1.127795,-0.027015,-0.0385,-0.44844,0.22922,0.97549]],[[76,10,15,-2,-2,17,-2,-2,3,42,-2,-2,4,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.8735000193119049,0.13809524476528168,-2.0,-2.0,0.006000000052154064,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003074,-0.02552,-0.071441,-0.606774,0.605509,0.036722,0.419256,-1.16383,0.066354,0.030964,0.630364,-0.147521,0.249375,0.939647,1.592966]],[[11,79,1,-2,-2,38,-2,-2,4,4,-2,-2,9,-2,-2],[0.20714285969734192,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.16899999976158142,0.03799999877810478,-2.0,-2.0,0.9449999928474426,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001161,-0.029961,-0.082599,-0.721628,0.762067,0.016167,-0.100219,1.222054,0.061789,0.12226,0.870691,1.233796,0.033034,-0.125484,0.759392]],[[8,9,5,-2,-2,8,-2,-2,79,1,-2,-2,70,-2,-2],[0.0925000011920929,0.94200000166893,0.051500000059604645,-2.0,-2.0,0.05049999989569187,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001071,0.072382,0.036652,0.720032,-0.666907,0.124172,0.775987,1.653501,-0.023912,-0.083616,-0.696632,0.45443,0.028877,-0.035636,1.317125]],[[79,53,46,-2,-2,2,-2,-2,65,22,-2,-2,8,-2,-2],[0.6449579894542694,0.5,1.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1679999977350235,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.0007,-0.070122,0.048469,-0.219554,1.366023,-0.113641,0.837925,-0.920614,0.02751,0.01293,0.198684,-1.222223,0.204478,1.124224,1.451577]],[[2,7,69,-2,-2,0,-2,-2,32,15,-2,-2,10,-2,-2],[0.9350000023841858,0.007500000298023224,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2916666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006036,0.051267,0.078334,0.830619,1.161357,0.006422,0.284507,-0.976802,-0.03239,-0.045846,-0.410742,0.703987,0.099637,-0.101361,1.330785]],[[22,17,79,-2,-2,80,-2,-2,19,-2,79,-2,-2],[0.5,0.5,0.6449579894542694,-2.0,-2.0,0.6201923191547394,-2.0,-2.0,0.5,-2.0,0.6833333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.00538,0.01725,0.028039,-0.364422,0.409682,-0.163964,-0.222401,-1.458633,-0.130593,-1.422734,-0.094469,-0.480711,-1.239517]],[[2,22,79,-2,-2,78,-2,-2,0,1,-2,-2,4,-2,-2],[0.9600000083446503,0.5,0.7573529481887817,-2.0,-2.0,0.5192307829856873,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1015000008046627,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003611,0.030682,0.04349,-0.075885,0.674508,-0.135916,-0.557426,-1.288528,-0.047802,-0.057117,-0.460992,1.473214,0.17193,1.187117,1.151578]],[[16,79,2,-2,-2,2,-2,-2,4,-2,-2],[0.5,0.8153409063816071,0.9350000023841858,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.863999992609024,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003046,0.011711,-0.021502,0.597153,-0.386849,0.053433,0.763641,-0.332127,-0.144319,-1.586193,-0.242122]],[[79,1,78,-2,-2,2,-2,-2,2,19,-2,-2,8,-2,-2],[0.6449579894542694,0.5,0.4833333343267441,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.9850000143051147,0.5,-2.0,-2.0,0.05250000022351742,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004039,-0.058247,-0.084052,-0.841105,-0.164174,0.106206,0.433828,1.509905,0.027409,0.056481,0.551967,-1.261521,-0.056699,0.606064,-0.565964]],[[64,21,1,-2,-2,78,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4833333343267441,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002316,-0.010131,-0.019068,-0.276426,0.707414,0.136924,-0.274499,1.34334,1.048088]],[[5,8,17,-2,-2,37,-2,-2,79,21,-2,-2,69,-2,-2],[0.04749999940395355,0.4650000035762787,0.5,-2.0,-2.0,2.5,-2.0,-2.0,0.8708333373069763,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.006435,0.04484,0.076642,0.803865,-0.590232,-0.048812,-0.722334,0.856961,-0.029748,-0.065964,-0.585517,0.864758,0.03865,-0.238619,0.774809]],[[16,21,17,-2,-2,9,-2,-2,82,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9845000207424164,-2.0,-2.0,0.0625,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001938,0.006448,-0.002293,0.05026,-1.110008,0.166863,1.263066,0.747318,-0.148513,-0.512872,-1.899387]],[[80,2,82,-2,-2,1,-2,-2,76,32,-2,-2,3,-2,-2],[0.8708333373069763,0.9350000023841858,0.2928571552038193,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.19099999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007827,-0.034307,0.044302,0.277408,1.29203,-0.067169,-0.523633,1.504835,0.045661,0.007713,-0.175591,1.458673,0.151811,0.170112,1.566871]],[[16,10,76,-2,-2,17,-2,-2,9,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.008526,0.000935,-0.030967,-0.475459,0.357557,0.043932,0.516086,-1.152292,-0.1631,-1.389739,-0.550395]],[[21,8,4,-2,-2,78,-2,-2,78,-2,-2],[0.5,0.0925000011920929,0.8745000064373016,-2.0,-2.0,0.5419580638408661,-2.0,-2.0,0.47727273404598236,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[6e-06,-0.008983,0.05719,0.81324,0.151131,-0.031934,-0.645237,-0.011893,0.148667,-0.038176,1.317079]],[[55,42,17,-2,-2,8,-2,-2,5,-2,-2],[0.5,0.33908046782016754,0.5,-2.0,-2.0,0.024000000208616257,-2.0,-2.0,0.15850000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004196,0.013869,0.02962,0.284326,-0.968938,-0.062608,1.072987,-0.868212,-0.118403,-1.347791,-1.047399]],[[64,76,53,-2,-2,3,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8735000193119049,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.008278,0.000243,-0.020045,0.524591,-0.359148,0.062456,0.215651,1.191195,1.316298]],[[21,2,39,-2,-2,38,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7685000002384186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003595,-0.005012,0.0483,0.607671,-1.173954,-0.030662,-0.387071,0.460174,0.150227,0.194526,1.354562]],[[55,78,1,-2,-2,21,-2,-2,19,-2,5,-2,-2],[0.5,0.4833333343267441,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,0.24950000643730164,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000186,0.009935,-0.052167,-0.588553,1.165506,0.030097,0.152402,1.308809,-0.116236,-1.370062,-0.078157,-1.032787,-1.068378]],[[38,16,10,-2,-2,9,-2,-2,78,-2,42,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.3550420254468918,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.002209,-0.017732,-0.007049,-0.281377,0.335366,-0.166449,-1.661507,-0.509153,0.077804,-1.00435,0.110567,1.159418,-0.035923]],[[11,16,76,-2,-2,-2,9,4,-2,-2,44,-2,-2],[0.36666667461395264,0.5,0.5,-2.0,-2.0,-2.0,0.7730000019073486,0.1730000004172325,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.001339,-0.020308,-0.009567,-0.242677,0.333115,-1.494557,0.053245,0.002633,1.081119,-0.382887,0.11,0.63013,1.146032]],[[65,5,76,-2,-2,2,-2,-2,79,-2,-2],[0.5,0.6755000054836273,0.5,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,0.7321428656578064,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004888,-0.004871,0.019451,-0.039606,0.656346,-0.059987,-0.157477,-1.098622,0.130051,0.181471,1.316795]],[[8,8,5,-2,-2,9,-2,-2,79,39,-2,-2,70,-2,-2],[0.09149999916553497,0.06149999983608723,0.0494999997317791,-2.0,-2.0,0.7209999859333038,-2.0,-2.0,0.7573529481887817,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001716,0.05422,0.040283,0.726054,-0.504809,0.146031,0.003974,1.68986,-0.020557,-0.065993,-0.436821,-1.656243,0.018741,-0.063395,1.385819]],[[21,17,19,-2,-2,9,-2,-2,48,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002204,-0.00575,0.001639,0.104447,-1.102615,-0.148418,-1.458405,-0.682545,0.148139,0.565923,1.256767]],[[38,16,1,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.2567765712738037,0.9855000078678131,-2.0,-2.0,0.466499999165535,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004354,-0.020744,-0.010429,-0.204171,0.809398,-0.174759,-1.64974,-0.905294,0.072547,0.131506,1.267795,0.050226,-0.079414,0.097782,-1.396378]],[[19,79,7,-2,-2,69,-2,-2,55,4,-2,-2,5,-2,-2],[0.5,0.8153409063816071,0.029000000096857548,-2.0,-2.0,1.5,-2.0,-2.0,0.5,0.5575000047683716,-2.0,-2.0,0.24950000643730164,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004234,0.007349,-0.022583,0.212345,-0.64315,0.048866,0.139689,1.057784,-0.10513,-0.137178,-1.375917,-0.792167,-0.076956,-1.151682,-1.067995]],[[2,39,9,-2,-2,-2,78,1,-2,-2,37,-2,-2],[0.9350000023841858,0.5,0.9675000011920929,-2.0,-2.0,-2.0,0.5505050718784332,0.5,-2.0,-2.0,2.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.000977,0.046533,0.053948,0.302845,0.966437,-1.333619,-0.020353,-0.074302,-0.637741,1.424515,0.011152,-0.115591,0.896545]],[[55,17,2,-2,-2,79,-2,-2,19,-2,5,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.612500011920929,-2.0,-2.0,0.5,-2.0,0.19750000536441803,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.00093,0.009555,0.017071,0.346036,-0.196635,-0.116498,0.085924,-1.393113,-0.109464,-1.369088,-0.070657,-1.007232,-1.060405]],[[15,38,53,-2,-2,42,-2,-2,9,78,-2,-2,2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,0.9835000038146973,0.5584415793418884,-2.0,-2.0,0.9099999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002253,-0.009269,-0.026344,0.330655,-0.384789,0.066195,1.06453,-0.579584,0.094766,0.036419,-0.588127,0.937585,0.174263,0.898506,1.446209]],[[2,78,2,-2,-2,7,-2,-2,46,85,-2,-2,9,-2,-2],[0.9750000238418579,0.5419580638408661,0.9249999821186066,-2.0,-2.0,0.9354999959468842,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.9794999957084656,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005196,0.027976,-0.034343,0.301074,-0.534973,0.061005,0.380112,1.481721,-0.033933,-0.054228,-0.350157,-1.135713,0.115619,0.432266,1.728945]],[[5,19,78,-2,-2,12,-2,-2,42,-2,6,-2,-2],[0.992499977350235,0.5,0.5505050718784332,-2.0,-2.0,0.6444999873638153,-2.0,-2.0,0.1339285746216774,-2.0,0.26900000870227814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000818,0.008759,0.019637,-0.215741,0.376815,-0.088062,-0.895444,-1.338864,-0.105994,-0.252977,-0.151532,-1.021951,-1.3186]],[[16,2,39,-2,-2,15,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9580000042915344,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001037,0.00844,0.059651,0.844541,-1.112849,-0.014094,-0.202099,0.74136,-0.142686,-1.440308,-0.23738]],[[32,1,43,-2,-2,2,-2,-2,12,-2,3,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.41499999165534973,-2.0,0.021499999798834324,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[6.9e-05,-0.011149,-0.02934,-0.334915,0.646175,0.061136,0.780428,1.430675,0.112205,1.263703,0.084048,0.124437,1.035613]],[[16,17,19,-2,-2,9,-2,-2,4,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9050000011920929,-2.0,-2.0,0.9105000197887421,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00455,0.012898,0.020834,0.260668,-1.058785,-0.138243,-1.443864,-0.523513,-0.126316,-1.452738,0.10041]],[[55,5,78,-2,-2,9,-2,-2,19,-2,6,-2,-2],[0.5,0.992499977350235,0.4686274528503418,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,0.5,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003414,0.011574,0.020394,-0.244613,0.304439,-0.095036,-0.316466,-1.226011,-0.095901,-1.337153,-0.057213,-0.866792,-1.093836]],[[80,5,8,-2,-2,39,-2,-2,30,69,-2,-2,-2],[0.8708333373069763,0.024500000290572643,0.4035000056028366,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000935,-0.022764,0.042697,0.700605,-0.256043,-0.047724,-0.320078,-1.480077,0.040377,0.022431,-0.208365,0.565041,1.550925]],[[7,22,78,-2,-2,-2,16,78,-2,-2,-2],[0.007500000298023224,0.5,0.4686274528503418,-2.0,-2.0,-2.0,0.5,0.5505050718784332,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.003939,0.036833,0.048087,-0.130178,0.68031,-1.058092,-0.033154,-0.022736,-0.683914,0.014247,-1.488342]],[[38,21,53,-2,-2,43,-2,-2,79,12,-2,-2,53,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.5370000004768372,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000375,-0.013642,-0.022108,0.388128,-0.355909,0.12824,0.244416,1.257832,0.069945,-0.01669,0.325869,-0.66506,0.138327,0.775816,1.410962]],[[5,19,16,-2,-2,12,-2,-2,2,-2,80,-2,-2],[0.9835000038146973,0.5,0.5,-2.0,-2.0,0.6444999873638153,-2.0,-2.0,0.9850000143051147,-2.0,0.8452380895614624,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003365,0.008063,0.020085,0.236383,-0.808523,-0.086376,-0.828172,-1.385571,-0.087993,-0.016418,-0.123294,-1.142537,-1.163574]],[[48,8,4,-2,-2,63,-2,-2,9,-2,-2],[0.5,0.17599999904632568,0.9259999990463257,-2.0,-2.0,0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002054,-0.006773,0.046611,0.721409,0.055449,-0.03154,-0.348536,0.694024,0.114705,0.17641,1.251795]],[[79,8,39,-2,-2,15,-2,-2,2,5,-2,-2,9,-2,-2],[0.8153409063816071,0.39249999821186066,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9850000143051147,0.901500016450882,-2.0,-2.0,0.9934999942779541,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00528,-0.033828,0.016937,0.26527,-0.927847,-0.074298,-0.738175,0.509453,0.032581,0.060887,0.374776,1.370283,-0.038252,-0.766558,0.492239]],[[16,76,8,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.6304999887943268,-2.0,-2.0,0.0669642873108387,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001432,0.005922,-0.012166,0.267357,-0.450266,0.062456,1.255268,0.079499,-0.137594,-1.293453,-0.784178]],[[21,1,70,-2,-2,46,-2,-2,37,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001692,-0.00833,-0.023952,-0.306405,0.424945,0.056571,0.750018,1.428612,0.121358,0.606428,1.231584]],[[2,19,50,-2,-2,55,-2,-2,0,37,-2,-2,12,-2,-2],[0.9750000238418579,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,1.5,-2.0,-2.0,0.31849999725818634,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00258,0.019454,0.032792,0.355718,-0.704254,-0.09907,-1.081663,-0.956769,-0.039558,-0.048486,-0.700144,0.139374,0.13737,1.038971,1.191591]],[[19,16,5,-2,-2,4,-2,-2,55,79,-2,-2,5,-2,-2],[0.5,0.5,0.9835000038146973,-2.0,-2.0,0.8574999868869781,-2.0,-2.0,0.5,0.6833333373069763,-2.0,-2.0,0.16700000315904617,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002977,0.006328,0.013489,0.208958,-0.495407,-0.096757,-1.615903,0.062229,-0.089016,-0.121705,-0.729405,-1.355894,-0.057884,-1.129202,-0.83229]],[[17,42,19,-2,-2,5,-2,-2,79,-2,-2],[0.5,0.33908046782016754,0.5,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001611,0.005485,0.019864,0.249443,-1.006861,-0.061114,1.159835,-0.853369,-0.128856,0.054328,-1.429106]],[[17,10,63,-2,-2,9,-2,-2,9,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.45350000262260437,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001294,0.006044,-0.02173,-0.277471,0.788842,0.042306,-0.119491,0.727344,-0.149495,-1.405872,-0.629147]],[[2,17,19,-2,-2,-2,1,0,-2,-2,-2],[0.9750000238418579,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.000654,0.023946,0.034141,0.403909,-0.966876,-1.07353,-0.039301,-0.045592,-0.457207,1.096834,1.397296]],[[76,17,8,-2,-2,9,-2,-2,4,8,-2,-2,9,-2,-2],[0.5,0.5,0.6324999928474426,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.09349999949336052,0.9740000069141388,-2.0,-2.0,0.9839999973773956,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000104,-0.01719,-0.007687,0.284899,-0.401728,-0.152351,-1.412177,-0.759092,0.055155,0.126667,0.430517,1.332758,0.001635,0.356592,-0.613681]],[[39,21,17,-2,-2,43,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.007999999914318323,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004174,0.001216,-0.005886,0.018471,-1.192603,0.124673,0.333368,1.27746,-0.181325,0.913528,-1.652755]],[[1,4,76,-2,-2,54,-2,-2,2,7,-2,-2,-2],[0.5,0.05849999934434891,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4749999940395355,0.12050000205636024,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.004384,-0.017494,0.021493,-0.126473,0.78109,-0.048293,-0.486264,0.917681,0.05201,0.04171,1.051366,-0.232528,1.404614]],[[19,5,16,-2,-2,42,-2,-2,55,79,-2,-2,5,-2,-2],[0.5,0.9895000159740448,0.5,-2.0,-2.0,0.04545454680919647,-2.0,-2.0,0.5,0.7071428596973419,-2.0,-2.0,0.27650000154972076,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001938,0.010568,0.021053,0.24786,-0.725431,-0.080026,0.181648,-0.984735,-0.081261,-0.115216,-0.648443,-1.311734,-0.04918,-0.921524,-1.043702]],[[21,50,79,-2,-2,-2,78,-2,-2],[0.5,0.5,0.8708333373069763,-2.0,-2.0,-2.0,0.4833333343267441,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.00104,-0.005211,0.000833,-0.168621,0.38288,-1.208583,0.103607,-0.121338,1.302298]],[[64,21,5,-2,-2,5,-2,-2,-2],[0.5,0.5,0.024500000290572643,-2.0,-2.0,0.13049999624490738,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005627,-0.000154,-0.006515,0.46485,-0.204015,0.123319,0.534586,1.212946,0.897923]],[[16,19,78,-2,-2,12,-2,-2,7,-2,-2],[0.5,0.5,0.5419580638408661,-2.0,-2.0,0.6444999873638153,-2.0,-2.0,0.011500000022351742,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001094,0.005098,0.013298,-0.207802,0.284944,-0.068503,-0.710767,-1.332297,-0.100105,-0.050381,-1.528515]],[[71,76,8,-2,-2,4,-2,-2,-2],[0.5,0.5,0.5940000116825104,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003456,0.00736,-0.010696,0.343342,-0.478964,0.061775,0.926139,0.040203,-1.189772]],[[32,76,11,-2,-2,4,-2,-2,12,-2,80,-2,-2],[0.5,0.5,0.4166666716337204,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,0.4989999979734421,-2.0,0.839743584394455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.006639,-0.015418,-0.032867,-0.508127,0.402038,0.040004,0.817878,-0.175902,0.080356,1.032013,0.035013,-0.556921,1.057361]],[[38,3,4,-2,-2,2,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.005499999970197678,0.05849999934434891,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.2361111119389534,0.9855000078678131,-2.0,-2.0,0.5339999794960022,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004396,-0.016349,0.032349,1.097781,-0.04555,-0.03479,-0.010957,-0.650987,0.050418,0.106568,1.222381,-0.250773,-0.072935,0.173742,-1.352235]],[[16,76,10,-2,-2,42,-2,-2,9,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.12132352963089943,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.006997,0.013408,-0.003856,-0.282441,0.342292,0.066089,1.286279,-0.006291,-0.101956,-1.294336,-0.236216]],[[42,48,32,-2,-2,7,-2,-2,42,-2,-2],[0.5857143104076385,0.5,0.5,-2.0,-2.0,0.019999999552965164,-2.0,-2.0,0.7321428656578064,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.006666,0.010607,0.003817,-0.056135,0.811057,0.103052,1.126762,0.155208,-0.141614,-1.141712,-1.861499]],[[21,1,78,-2,-2,79,-2,-2,5,-2,-2],[0.5,0.5,0.5505050718784332,-2.0,-2.0,0.4913793057203293,-2.0,-2.0,0.22850000113248825,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000372,-0.006392,-0.019463,-0.522048,0.029952,0.045278,1.40879,0.615032,0.112644,0.592574,1.187044]],[[11,16,76,-2,-2,-2,7,3,-2,-2,4,-2,-2],[0.4166666716337204,0.5,0.5,-2.0,-2.0,-2.0,0.005499999970197678,0.004500000039115548,-2.0,-2.0,0.05550000071525574,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002399,-0.014418,-0.005261,-0.23802,0.351578,-1.559038,0.050375,0.088195,0.705049,1.432326,0.003928,0.821152,-0.24086]],[[17,64,7,-2,-2,-2,42,-2,-2],[0.5,0.5,0.005499999970197678,-2.0,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[-0.002955,0.003248,-0.002625,0.312116,-0.186064,0.909546,-0.125976,-1.389311,-0.334954]],[[48,32,1,-2,-2,10,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000461,-0.007714,-0.0162,-0.254031,0.615742,0.069238,-0.043507,1.322615,0.091715,-0.260525,1.298242]],[[80,7,8,-2,-2,5,-2,-2,69,2,-2,-2,-2],[0.8708333373069763,0.004500000039115548,0.9255000054836273,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,2.5,0.9850000143051147,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002692,-0.022238,0.023908,0.398818,-0.714459,-0.051625,0.493539,-0.559687,0.035467,0.016401,0.466691,-0.552859,1.24061]],[[79,5,5,-2,-2,89,-2,-2,2,5,-2,-2,9,-2,-2],[0.8153409063816071,0.024500000290572643,0.009499999694526196,-2.0,-2.0,0.5,-2.0,-2.0,0.9850000143051147,0.9284999966621399,-2.0,-2.0,0.99549999833107,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000434,-0.022636,0.044115,0.074013,1.078186,-0.048328,-0.357312,-1.445719,0.031012,0.060801,0.438055,1.340048,-0.046498,-0.819384,0.565404]],[[5,71,17,-2,-2,-2,8,-2,80,-2,-2],[0.9895000159740448,0.5,0.5,-2.0,-2.0,-2.0,0.984499990940094,-2.0,0.8166666626930237,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[-0.003338,0.004896,0.008729,0.137642,-0.902084,-1.131175,-0.092449,-1.296385,-0.071294,-1.269928,-0.272694]],[[11,16,2,-2,-2,-2,7,3,-2,-2,12,-2,-2],[0.36666667461395264,0.5,0.9750000238418579,-2.0,-2.0,-2.0,0.005499999970197678,0.004500000039115548,-2.0,-2.0,0.7084999978542328,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000276,-0.014916,-0.006263,0.153376,-0.383109,-1.58794,0.042839,0.076854,0.697748,1.207357,-0.000791,-0.410491,0.875346]],[[19,55,78,-2,-2,-2,55,2,-2,-2,79,-2,-2],[0.5,0.5,0.5358974635601044,-2.0,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.7750000059604645,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001849,0.009904,0.014524,-0.202229,0.29635,-1.336591,-0.074341,-0.111812,-1.233333,-1.126536,-0.036405,-0.637206,-1.06068]],[[2,77,39,-2,-2,45,-2,-2,31,15,-2,-2,-2],[0.9350000023841858,0.5,0.5,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.003111,0.035273,0.025028,0.4295,-1.261804,0.114843,1.37099,0.827542,-0.020539,-0.029866,-0.336291,0.490597,0.904427]],[[78,8,79,-2,-2,5,-2,-2,16,19,-2,-2,-2],[0.5505050718784332,0.17549999803304672,0.3457317054271698,-2.0,-2.0,0.013500000350177288,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.005094,-0.03541,0.040868,-1.07797,0.567497,-0.069773,-1.251798,-0.404291,0.026622,0.03563,0.431312,-0.998732,-0.960059]],[[1,82,2,-2,-2,3,-2,-2,46,7,-2,-2,8,-2,-2],[0.5,0.34495192766189575,0.9750000238418579,-2.0,-2.0,0.953000009059906,-2.0,-2.0,1.5,0.25200000405311584,-2.0,-2.0,0.04449999891221523,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005253,-0.016991,-0.008138,0.1582,-0.382249,-0.111099,-0.531827,-1.498807,0.045189,0.033908,0.853331,-0.319347,0.204733,1.137781,1.530231]],[[79,8,5,-2,-2,81,-2,-2,4,69,-2,-2,70,-2,-2],[0.6449579894542694,0.09699999913573265,0.00849999999627471,-2.0,-2.0,0.2204861119389534,-2.0,-2.0,0.05849999934434891,1.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001276,-0.037665,0.030815,0.920663,-0.135573,-0.064644,-0.471813,-1.12373,0.015986,0.058744,0.234474,0.991473,-0.01038,-0.218544,0.592875]],[[71,55,2,-2,-2,5,-2,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.15850000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004675,0.007727,0.014048,0.293986,-0.146283,-0.072487,-1.329872,-0.952742,-1.139332]],[[5,9,4,-2,-2,78,-2,-2,8,-2,2,-2,-2],[0.9835000038146973,0.9494999945163727,0.05849999934434891,-2.0,-2.0,0.5358974635601044,-2.0,-2.0,0.9845000207424164,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000306,0.008328,-0.022117,0.454275,-0.461873,0.032823,-0.202892,0.57772,-0.063195,-1.164279,-0.030412,0.326641,-0.93415]],[[76,3,17,-2,-2,6,-2,-2,3,6,-2,-2,78,-2,-2],[0.5,0.24550000578165054,0.5,-2.0,-2.0,0.6335000097751617,-2.0,-2.0,0.14900000393390656,0.5575000047683716,-2.0,-2.0,0.4686274528503418,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003018,-0.010909,0.012581,0.222162,-1.023782,-0.047727,-0.898246,-0.089378,0.048847,-0.009514,0.94031,-0.463262,0.114779,-0.443686,1.003113]],[[16,78,8,-2,-2,70,-2,-2,9,-2,-2],[0.5,0.5505050718784332,0.17499999701976776,-2.0,-2.0,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002983,0.009194,-0.02414,0.402454,-0.461921,0.026542,0.156312,0.908437,-0.099019,-1.582534,-0.096263]],[[78,11,69,-2,-2,83,-2,-2,4,76,-2,-2,2,-2,-2],[0.5358974635601044,0.22500000149011612,3.5,-2.0,-2.0,1.5,-2.0,-2.0,0.09100000187754631,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001654,-0.033858,-0.064982,-0.472047,-1.444277,0.040876,-0.593534,0.973327,0.019248,0.059829,0.203382,1.039239,-0.010826,0.11575,-0.787104]],[[71,21,5,-2,-2,5,-2,-2,-2],[0.5,0.5,0.024500000290572643,-2.0,-2.0,0.11800000071525574,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003812,0.0,-0.004947,0.433635,-0.177449,0.098195,0.196254,1.122857,-1.484962]],[[80,2,2,-2,-2,1,-2,-2,69,2,-2,-2,4,-2,-2],[0.8708333373069763,0.9350000023841858,0.8250000178813934,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.9850000143051147,-2.0,-2.0,0.07649999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003312,-0.013578,0.033044,0.168214,1.047664,-0.032765,-0.321173,1.408779,0.036214,0.01222,0.446682,-0.637035,0.120051,1.302303,0.310789]],[[8,9,5,-2,-2,42,-2,-2,63,15,-2,-2,-2],[0.4089999943971634,0.9284999966621399,0.006500000134110451,-2.0,-2.0,0.4226190447807312,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001388,0.030616,-0.005534,1.193709,-0.363578,0.082487,1.051279,-0.705936,-0.020226,-0.030767,-0.366432,0.769774,0.685808]],[[19,5,16,-2,-2,2,-2,-2,55,-2,5,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,-2.0,0.21800000220537186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003137,0.004736,0.014737,0.204587,-0.846484,-0.066417,0.093633,-1.041574,-0.074018,-1.19216,-0.045735,-1.102462,-1.039963]],[[48,5,81,-2,-2,4,-2,-2,6,-2,-2],[0.5,0.04749999940395355,0.1270161271095276,-2.0,-2.0,0.1445000022649765,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-2.6e-05,-0.006126,0.027793,0.709762,-0.041658,-0.022998,0.058017,-0.521671,0.083408,-0.321526,1.064342]],[[16,71,76,-2,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5565000176429749,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[0.000175,0.006594,0.010127,-0.059783,0.483045,-1.073208,-0.109671,-0.505831,-1.498209]],[[38,16,2,-2,-2,9,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.2567765712738037,0.9855000078678131,-2.0,-2.0,0.48900000751018524,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006792,-0.004932,0.001279,0.526065,-0.124769,-0.107943,-1.406832,-0.570973,0.06377,0.095623,1.153553,-0.333671,-0.028153,0.460323,-1.328821]],[[42,21,17,-2,-2,9,-2,-2,79,-2,3,-2,-2],[0.5505050718784332,0.5,0.5,-2.0,-2.0,0.9845000207424164,-2.0,-2.0,0.23111110925674438,-2.0,0.664000004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001033,0.004847,-1e-06,0.051546,-1.007029,0.099489,1.069411,0.562345,-0.110385,-1.794303,-0.083732,-0.548524,-1.500818]],[[20,5,-2,-2,17,16,-2,-2,79,-2,-2],[0.5,0.8325000107288361,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[-0.001192,-0.104967,-0.461141,-1.218002,0.005284,0.010647,0.168432,-0.850306,-0.097327,0.193441,-1.436316]],[[17,2,19,-2,-2,79,-2,-2,4,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.8819444477558136,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003604,0.008806,0.029407,0.379855,-0.802868,-0.018339,-0.39,0.287771,-0.098221,-1.48129,-0.119363]],[[76,3,5,-2,-2,4,-2,-2,42,2,-2,-2,2,-2,-2],[0.5,0.03849999979138374,0.6700000166893005,-2.0,-2.0,0.341499999165535,-2.0,-2.0,0.12132352963089943,0.9750000238418579,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003307,-0.009578,0.017418,0.355936,-0.731078,-0.03282,-0.101697,-0.851388,0.045012,0.129422,0.576192,1.355527,-0.007209,0.384753,-0.560804]],[[89,55,42,-2,-2,52,-2,-2,53,-2,-2],[0.5,0.5,0.612500011920929,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001428,0.00616,0.012201,0.143806,-1.471063,-0.065796,-1.228171,-0.935917,-0.09485,0.304701,-1.252638]],[[54,17,11,-2,-2,78,-2,-2,9,-2,-2],[0.5,0.5,0.20714285969734192,-2.0,-2.0,0.4833333343267441,-2.0,-2.0,0.9835000038146973,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.0042,-0.001009,0.005189,-0.098231,0.483507,-0.112675,-0.317164,-1.343301,0.109418,1.027601,0.517365]],[[5,19,17,-2,-2,55,-2,-2,8,-2,2,-2,-2],[0.9835000038146973,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9799999892711639,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002241,0.010386,0.019473,0.225738,-0.726581,-0.062807,-1.08,-0.944546,-0.056686,-1.411071,-0.022261,0.532142,-0.944446]],[[65,56,21,-2,-2,83,-2,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.4989999979734421,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003154,-0.010712,-0.00522,-0.107956,0.748622,-0.112457,-1.292272,-0.757201,0.099644,1.247907,0.41159]],[[42,48,32,-2,-2,7,-2,-2,80,-2,-2],[0.5857143104076385,0.5,0.5,-2.0,-2.0,0.031000000424683094,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001586,0.004939,-0.001832,-0.101053,0.680733,0.092411,1.144948,0.055993,-0.128406,-1.708865,-1.22557]],[[5,69,8,-2,-2,-2,21,9,-2,-2,-2],[0.0494999997317791,2.5,0.9180000126361847,-2.0,-2.0,-2.0,0.5,0.9774999916553497,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.00062,0.033067,0.022416,0.497461,-0.939454,0.820725,-0.018057,-0.025289,-0.613189,0.020059,0.79147]],[[44,2,42,-2,-2,6,-2,-2,6,-2,6,-2,-2],[0.5,0.9850000143051147,0.44949495792388916,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.024500000290572643,-2.0,0.24050000309944153,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.002366,0.018743,0.04403,0.64859,-0.835801,-0.031406,-0.601095,1.496287,-0.030823,1.212665,-0.041273,-1.098938,-0.247652]],[[17,2,46,-2,-2,5,-2,-2,12,-2,-2],[0.5,0.9750000238418579,1.5,-2.0,-2.0,0.5489999949932098,-2.0,-2.0,0.511000007390976,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001288,0.003734,0.019778,0.266052,-0.843786,-0.023675,0.200623,-0.614682,-0.090178,-0.256861,-1.415766]],[[76,3,17,-2,-2,6,-2,-2,37,3,-2,-2,-2],[0.5,0.13000000268220901,0.5,-2.0,-2.0,0.3265000134706497,-2.0,-2.0,3.5,0.14900000393390656,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00471,-0.007793,0.018326,0.315978,-0.955685,-0.038269,-0.916198,-0.132673,0.042772,0.06268,0.15358,0.823184,-0.853171]],[[42,70,44,-2,-2,79,-2,-2,8,46,-2,-2,1,-2,-2],[0.33908046782016754,0.5,0.5,-2.0,-2.0,0.7638888955116272,-2.0,-2.0,0.024000000208616257,1.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003228,0.01254,0.00188,0.266335,-0.23134,0.092796,0.05647,1.190448,-0.041331,0.073425,0.767814,1.154951,-0.069414,-0.918605,1.180661]],[[22,79,8,-2,-2,63,-2,-2,52,-2,78,-2,-2],[0.5,0.6449579894542694,0.07349999994039536,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.006319,0.013319,-0.028065,0.31032,-0.475106,0.028841,0.197974,0.913939,-0.074863,-1.274045,-0.035591,-0.626331,-1.080445]],[[5,39,55,-2,-2,12,-2,-2,2,79,-2,-2,6,-2,-2],[0.6994999945163727,0.5,0.5,-2.0,-2.0,0.4544999897480011,-2.0,-2.0,0.9750000238418579,0.7638888955116272,-2.0,-2.0,0.992499977350235,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002705,0.010888,0.015853,0.231383,-1.115608,-0.114609,0.526843,-1.560265,-0.035666,0.02524,-0.438598,0.693904,-0.070539,-1.056741,0.95193]],[[4,4,78,-2,-2,5,-2,-2,1,9,-2,-2,2,-2,-2],[0.06000000052154064,0.04649999924004078,0.5419580638408661,-2.0,-2.0,0.33250000327825546,-2.0,-2.0,0.5,0.38349999487400055,-2.0,-2.0,0.4749999940395355,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002096,0.023434,0.010208,-0.371551,0.325858,0.163048,0.483441,1.638788,-0.018793,-0.038184,-1.038237,-0.201431,0.034381,0.516312,1.376503]],[[65,5,44,-2,-2,8,-2,-2,79,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9895000159740448,-2.0,-2.0,0.7071428596973419,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001752,-0.004491,0.002863,0.297211,-0.174273,-0.060524,-1.32882,-0.055578,0.078769,-0.109302,1.123172]],[[38,8,9,-2,-2,48,-2,-2,79,5,-2,-2,9,-2,-2],[0.5,0.1314999982714653,0.9745000004768372,-2.0,-2.0,0.5,-2.0,-2.0,0.7638888955116272,0.051500000059604645,-2.0,-2.0,0.9900000095367432,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004272,-0.014552,0.031496,0.113352,0.841484,-0.031791,-0.393959,0.523613,0.0462,-0.020253,0.606482,-0.668576,0.101865,1.077712,1.327172]],[[76,8,9,-2,-2,6,-2,-2,4,8,-2,-2,37,-2,-2],[0.5,0.39249999821186066,0.38199999928474426,-2.0,-2.0,0.3009999990463257,-2.0,-2.0,0.03350000083446503,0.976500004529953,-2.0,-2.0,3.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001515,-0.008989,0.0242,-0.456335,0.625031,-0.033468,-1.037978,-0.116308,0.035339,0.127467,0.552065,1.337876,-0.010211,0.092503,-1.145099]],[[19,42,17,-2,-2,80,-2,-2,55,81,-2,-2,78,-2,-2],[0.5,0.5857143104076385,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.13701201230287552,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001703,0.008009,0.011342,0.152453,-0.563321,-0.115579,-1.696888,-1.083256,-0.057767,-0.089262,-1.099184,-0.978353,-0.031131,-0.517473,-1.074963]],[[71,76,3,-2,-2,4,-2,-2,-2],[0.5,0.5,0.1979999989271164,-2.0,-2.0,0.09200000017881393,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002531,0.000291,-0.01436,0.145145,-0.547813,0.045683,0.795952,0.004929,-0.975211]],[[76,3,4,-2,-2,77,-2,-2,42,3,-2,-2,2,-2,-2],[0.5,0.11449999734759331,0.679500013589859,-2.0,-2.0,0.5,-2.0,-2.0,0.11324786394834518,0.061000000685453415,-2.0,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[7e-06,-0.013321,0.013929,0.491701,-0.169659,-0.045838,-0.591442,0.825919,0.039817,0.119045,0.130618,1.457898,-0.00698,0.390886,-0.732906]],[[64,21,5,-2,-2,43,-2,-2,-2],[0.5,0.5,0.024500000290572643,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001945,-0.002489,-0.00729,0.374072,-0.194833,0.088196,0.31512,1.118852,0.89397]],[[12,6,4,-2,-2,5,-2,-2,20,5,-2,-2,30,-2,-2],[0.2580000013113022,0.7980000078678131,0.023000000044703484,-2.0,-2.0,0.08350000157952309,-2.0,-2.0,0.5,0.8480000197887421,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001258,0.073992,0.148762,0.108701,1.570976,0.012418,1.072133,-0.462936,-0.00723,-0.104852,-0.51344,-1.444929,-0.001334,-0.088169,0.827689]],[[2,3,78,-2,-2,4,-2,-2,42,76,-2,-2,1,-2,-2],[0.9350000023841858,0.04749999940395355,0.44949495792388916,-2.0,-2.0,0.28450000286102295,-2.0,-2.0,0.02777777798473835,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004422,0.022557,0.040127,1.182674,0.738474,-0.011191,0.502001,-0.714733,-0.017241,0.029479,-0.081764,0.952046,-0.035162,-0.354421,1.404219]],[[78,71,8,-2,-2,-2,2,5,-2,-2,7,-2,-2],[0.8090909123420715,0.5,0.09149999916553497,-2.0,-2.0,-2.0,0.9850000143051147,0.8559999763965607,-2.0,-2.0,0.5690000057220459,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.005963,-0.017336,-0.014023,0.342626,-0.254271,-0.949729,0.024985,0.058018,0.573493,1.233397,-0.024699,0.518518,-0.67106]],[[2,9,7,-2,-2,78,-2,-2,1,20,-2,-2,-2],[0.9600000083446503,0.9440000057220459,0.005499999970197678,-2.0,-2.0,0.539393961429596,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000626,0.01757,-0.017553,0.382761,-0.583308,0.041825,-0.11245,0.733476,-0.020221,-0.024496,-0.922826,-0.180891,1.413622]],[[79,5,5,-2,-2,39,-2,-2,6,85,-2,-2,4,-2,-2],[0.8708333373069763,0.024500000290572643,0.019499999471008778,-2.0,-2.0,0.5,-2.0,-2.0,0.9835000038146973,0.5,-2.0,-2.0,0.419000007212162,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002345,-0.011138,0.029693,0.150981,1.343571,-0.026358,-0.209943,-1.144595,0.030818,0.003329,0.285568,-1.26472,0.10281,1.608452,0.514919]],[[8,4,3,-2,-2,44,-2,-2,43,79,-2,-2,78,-2,-2],[0.09149999916553497,0.9185000061988831,0.032500000670552254,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.7638888955116272,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.007116,0.025478,0.060063,0.58655,1.138923,-0.014488,0.786715,-0.791997,-0.017926,-0.026242,-0.599957,-0.014317,0.08132,-0.18862,1.278692]],[[16,21,2,-2,-2,9,-2,-2,9,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.9845000207424164,-2.0,-2.0,0.5600000023841858,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000921,0.00455,-0.000718,0.466212,-0.142803,0.095933,1.162187,0.361465,-0.091836,-1.493938,-0.485068]],[[19,16,17,-2,-2,9,-2,-2,55,-2,12,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9305000007152557,-2.0,-2.0,0.5,-2.0,0.48900000751018524,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002782,0.008739,0.014107,0.188424,-0.632324,-0.070664,-1.333735,0.643745,-0.060579,-1.039038,-0.032535,-0.668014,-1.087989]],[[38,6,12,-2,-2,79,-2,-2,42,36,-2,-2,12,-2,-2],[0.5,0.9794999957084656,0.2564999908208847,-2.0,-2.0,0.8660714328289032,-2.0,-2.0,0.2899159789085388,0.5,-2.0,-2.0,0.4580000042915344,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000656,-0.008354,-0.022305,0.57912,-0.290595,0.023783,-0.118811,1.17021,0.04841,0.088259,0.759242,1.544779,-0.078608,-0.118514,-1.29374]],[[64,7,3,-2,-2,39,-2,-2,-2],[0.5,0.007500000298023224,0.15599999576807022,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00124,-0.00261,0.022042,0.476122,-0.143746,-0.020178,-0.145373,-1.101987,0.830055]],[[86,89,37,-2,-2,69,-2,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.000852,0.004704,0.008918,-0.058158,0.389207,-0.088777,0.133789,-1.344791,-0.769886]],[[22,2,7,-2,-2,5,-2,-2,19,-2,79,-2,-2],[0.5,0.9750000238418579,0.9354999959468842,-2.0,-2.0,0.5489999949932098,-2.0,-2.0,0.5,-2.0,0.6833333373069763,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003739,0.009173,0.026014,0.169483,1.232379,-0.017425,0.259275,-0.599663,-0.061494,-1.307921,-0.032457,-0.018125,-1.147071]],[[77,63,48,-2,-2,-2,4,6,-2,-2,44,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.0729999989271164,0.26999999582767487,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.003691,-0.002914,-0.008389,-0.160721,0.540981,0.676182,0.064618,0.15772,0.343774,1.891458,0.008489,0.612174,-0.675775]],[[63,8,5,-2,-2,5,-2,-2,80,-2,-2],[0.5,0.5940000116825104,0.5920000076293945,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.8257575631141663,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004899,0.000305,0.02097,0.351592,-0.521553,-0.020232,-1.198162,-0.12091,0.082973,-0.187362,1.352675]],[[1,78,2,-2,-2,70,-2,-2,46,11,-2,-2,-2],[0.5,0.4686274528503418,0.9249999821186066,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00038,-0.008552,-0.049343,0.425767,-0.652702,0.003426,-0.080149,0.793456,0.038039,0.03021,0.817519,0.506621,1.385605]],[[19,16,2,-2,-2,9,-2,-2,12,8,-2,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.6444999873638153,0.40150000154972076,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001562,0.007731,0.012893,0.246585,-0.2037,-0.073098,-1.122464,-0.077868,-0.052658,-0.039475,-0.952986,-0.568466,-1.248156]],[[42,54,38,-2,-2,9,-2,-2,79,-2,3,-2,-2],[0.5505050718784332,0.5,0.5,-2.0,-2.0,0.9819999933242798,-2.0,-2.0,0.23111110925674438,-2.0,0.664000004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.005322,0.008718,0.004286,-0.046517,0.490809,0.096762,0.917471,0.614188,-0.089785,-1.762584,-0.06426,-0.734918,-1.270604]],[[2,39,78,-2,-2,-2,1,83,-2,-2,8,-2,-2],[0.9350000023841858,0.5,0.44949495792388916,-2.0,-2.0,-2.0,0.5,2.5,-2.0,-2.0,0.05700000002980232,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.00253,0.022905,0.028747,0.873455,0.350976,-1.07644,-0.013732,-0.017117,-0.070275,-0.764852,0.176867,1.096706,1.493514]],[[21,17,11,-2,-2,9,-2,-2,78,-2,-2],[0.5,0.5,0.36666667461395264,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5357142984867096,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003905,-0.000595,0.004245,-0.070527,0.546579,-0.092713,-1.362816,-0.356382,0.087766,0.180808,1.28004]],[[4,4,69,-2,-2,-2,44,6,-2,-2,9,-2,-2],[0.06000000052154064,0.05049999989569187,2.5,-2.0,-2.0,-2.0,0.5,0.9535000026226044,-2.0,-2.0,0.38349999487400055,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00177,0.024155,0.015439,0.025783,0.753294,1.426207,-0.012519,0.009117,-0.244025,0.600333,-0.039494,-1.29662,-0.162648]],[[2,78,9,-2,-2,2,-2,-2,46,85,-2,-2,9,-2,-2],[0.9750000238418579,0.8090909123420715,0.3799999952316284,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00111,0.014921,-0.000606,-0.893532,0.121838,0.062277,0.654065,1.210471,-0.026911,-0.039776,-0.324557,-1.179073,0.068342,0.272745,1.420957]],[[12,6,4,-2,-2,3,-2,-2,20,42,-2,-2,12,-2,-2],[0.2580000013113022,0.7980000078678131,0.023000000044703484,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,0.5,0.2361111119389534,-2.0,-2.0,0.3994999974966049,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002121,0.075021,0.157646,0.360826,1.54096,0.008921,1.50898,-0.630513,-0.003076,-0.083387,-0.31934,-1.404707,0.002039,-0.424324,0.127257]],[[21,2,77,-2,-2,81,-2,-2,6,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.17444219440221786,-2.0,-2.0,0.789000004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004569,-0.009624,0.019197,0.142517,1.184094,-0.023032,-0.051422,-0.5586,0.087651,-0.133494,1.247811]],[[77,16,76,-2,-2,9,-2,-2,4,6,-2,-2,2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.0729999989271164,0.26999999582767487,-2.0,-2.0,0.9249999821186066,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006151,-0.000109,0.004886,-0.083665,0.371801,-0.082999,-1.434033,-0.000128,0.060747,0.152375,0.244765,1.919925,0.010879,1.284358,-0.240575]],[[17,42,19,-2,-2,5,-2,-2,79,-2,-2],[0.5,0.4208333343267441,0.5,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001391,0.005689,0.011422,0.1783,-0.883061,-0.043761,1.243775,-0.761603,-0.076268,0.131608,-1.346432]],[[2,7,55,-2,-2,12,-2,-2,46,38,-2,-2,53,-2,-2],[0.9750000238418579,0.8264999985694885,0.5,-2.0,-2.0,0.5734999775886536,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[8.2e-05,0.014365,0.006391,0.147112,-1.140613,0.094518,1.47851,0.091687,-0.0247,-0.037137,-0.570341,0.596318,0.064944,1.227767,-0.277164]],[[16,20,80,-2,-2,76,-2,-2,4,-2,-2],[0.5,0.5,0.7638888955116272,-2.0,-2.0,0.5,-2.0,-2.0,0.8734999895095825,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003596,0.008034,-0.071208,-1.259174,-0.203054,0.013058,0.022174,0.458305,-0.07413,-1.344923,-0.02819]],[[54,89,11,-2,-2,69,-2,-2,6,-2,-2],[0.5,0.5,0.36666667461395264,-2.0,-2.0,0.5,-2.0,-2.0,0.8060000240802765,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000258,-0.004031,4e-06,-0.110933,0.42416,-0.08981,-0.03435,-1.247481,0.090134,0.098467,1.213359]],[[17,5,19,-2,-2,8,-2,-2,9,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9804999828338623,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000715,0.003309,0.010294,0.18316,-0.937197,-0.050036,-1.146773,-0.238574,-0.083768,-1.3302,-0.414184]],[[65,76,2,-2,-2,8,-2,-2,12,-2,-2],[0.5,0.5,0.875,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,0.5089999884366989,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002327,-0.002379,-0.013351,0.374289,-0.309866,0.030727,0.05631,0.805708,0.065777,1.058923,0.095656]],[[5,5,77,-2,-2,79,-2,-2,8,-2,2,-2,-2],[0.9835000038146973,0.9544999897480011,0.5,-2.0,-2.0,0.8660714328289032,-2.0,-2.0,0.9835000038146973,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000152,0.005884,0.00105,-0.064655,0.567682,0.086242,-0.255426,1.447052,-0.0495,-1.397599,-0.00787,0.812252,-0.81256]],[[5,19,78,-2,-2,3,-2,-2,3,-2,6,-2,-2],[0.992499977350235,0.5,0.3550420254468918,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.965499997138977,-2.0,0.04100000113248825,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.005231,0.01042,0.016873,-0.37025,0.258746,-0.04539,-0.115368,-1.126994,-0.062079,-1.12652,-0.007085,-1.655395,0.255796]],[[11,81,76,-2,-2,8,-2,-2,7,8,-2,-2,12,-2,-2],[0.4166666716337204,0.16163264960050583,0.5,-2.0,-2.0,0.31450000405311584,-2.0,-2.0,0.005499999970197678,0.04350000061094761,-2.0,-2.0,0.6705000102519989,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002766,-0.013246,0.004559,-0.137484,0.422958,-0.045648,-0.968276,-0.254007,0.028344,0.059565,0.531891,1.250165,-0.013484,-0.607217,0.946898]],[[38,6,76,-2,-2,12,-2,-2,39,6,-2,-2,-2],[0.5,0.8899999856948853,0.5,-2.0,-2.0,0.33550000190734863,-2.0,-2.0,0.5,0.9855000078678131,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.003958,-0.012053,-0.03269,-0.595465,0.294573,0.006669,-0.609212,0.258987,0.036793,0.051619,0.878689,-0.444982,-1.220021]],[[20,76,-2,-2,16,89,-2,-2,9,-2,-2],[0.5,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[-0.004821,-0.084553,-0.118198,-1.201688,-0.000176,0.00484,0.097777,-0.727464,-0.076488,-1.480724,-0.205166]],[[4,4,78,-2,-2,-2,9,44,-2,-2,11,-2,-2],[0.05849999934434891,0.05049999989569187,0.5419580638408661,-2.0,-2.0,-2.0,0.7774999737739563,0.5,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002283,0.021723,0.013029,-0.483505,0.389371,1.648952,-0.016989,-0.05628,-0.175272,-1.084895,-0.000253,-0.153152,0.580287]],[[70,37,21,-2,-2,-2,42,9,-2,-2,-2],[0.5,4.5,0.5,-2.0,-2.0,-2.0,0.2899159789085388,0.9839999973773956,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.00241,-0.00364,-0.000236,-0.052347,0.683577,-1.023375,0.046803,0.083806,0.557513,1.003519,-0.952166]],[[11,16,76,-2,-2,-2,9,4,-2,-2,44,-2,-2],[0.4166666716337204,0.5,0.5,-2.0,-2.0,-2.0,0.9839999973773956,0.17100000381469727,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00157,-0.009179,-0.003139,-0.193007,0.301129,-1.46037,0.033524,0.010435,0.955078,-0.302971,0.091533,0.438607,1.326074]],[[89,12,6,-2,-2,20,-2,-2,12,-2,-2],[0.5,0.2580000013113022,0.800000011920929,-2.0,-2.0,0.5,-2.0,-2.0,0.4415000081062317,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002962,0.006794,0.070579,1.031643,0.245354,0.001651,-0.719936,0.070069,-0.075901,-1.327474,-0.451617]],[[17,89,80,-2,-2,6,-2,-2,10,-2,-2],[0.5,0.5,0.24038461595773697,-2.0,-2.0,0.56700000166893,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004317,-0.000614,0.003247,-1.624357,0.053425,-0.077268,0.649325,-1.20285,-0.081525,-0.25956,-1.444287]],[[64,15,44,-2,-2,7,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.03749999962747097,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.00432,-0.008295,-0.014722,0.053838,-0.368719,0.042123,0.947618,-0.178494,0.77402]],[[32,5,82,-2,-2,6,-2,-2,12,-2,5,-2,-2],[0.5,0.02350000012665987,0.39444445073604584,-2.0,-2.0,0.8875000178813934,-2.0,-2.0,0.41499999165534973,-2.0,0.17549999803304672,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000107,-0.004896,0.026316,0.512683,-0.92161,-0.014498,-0.381145,0.103547,0.050191,1.152247,0.016665,0.732445,-0.414318]],[[2,2,42,-2,-2,80,-2,-2,15,30,-2,-2,9,-2,-2],[0.9350000023841858,0.8250000178813934,0.24038461595773697,-2.0,-2.0,0.7888889014720917,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.921999990940094,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005261,0.030922,0.021962,0.756896,-0.043733,0.071315,1.102705,0.140748,-0.006327,-0.01371,-0.212976,0.569891,0.076345,-0.304196,1.1309]],[[17,79,-2,55,-2,-2,4,-2,-2],[0.5,0.232051283121109,-2.0,0.5,-2.0,-2.0,0.5820000171661377,-2.0,-2.0],[1,2,-1,4,-1,-1,7,-1,-1],[6,3,-1,5,-1,-1,8,-1,-1],[-0.000521,0.003395,-1.578465,0.004972,0.103684,-1.103724,-0.077625,-1.533091,-0.11808]],[[71,67,17,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.001809,0.004314,0.001433,0.05828,-0.766223,1.053176,-1.228104]],[[5,77,21,-2,-2,4,-2,-2,8,-2,8,-2,-2],[0.9835000038146973,0.5,0.5,-2.0,-2.0,0.0729999989271164,-2.0,-2.0,0.9835000038146973,-2.0,0.992499977350235,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003148,0.003768,-0.003114,-0.095853,0.977708,0.063896,1.333062,0.086551,-0.055797,-1.375335,-0.025402,0.417652,-0.826942]],[[38,69,53,-2,-2,42,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,2.5,0.5,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,0.22474747896194458,0.9855000078678131,-2.0,-2.0,0.4919999986886978,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000238,-0.007216,-0.013469,0.28117,-0.293184,0.048435,1.284289,-0.080488,0.036301,0.074802,1.245676,-0.125631,-0.045514,0.229187,-1.270735]],[[64,2,69,-2,-2,30,-2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.002909,-0.001247,0.02364,0.335017,0.837099,-0.013832,-0.206244,0.53803,0.882571]],[[78,8,3,-2,-2,5,-2,-2,9,4,-2,-2,2,-2,-2],[0.5505050718784332,0.17549999803304672,0.021499999798834324,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.9465000033378601,0.007500000298023224,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001346,-0.026305,0.028311,0.119233,0.767326,-0.049741,-1.169205,-0.375776,0.011845,-0.015437,1.325267,-0.390088,0.030108,0.489641,-0.353383]],[[81,65,1,-2,-2,80,-2,-2,-2],[0.3798076957464218,0.5,0.5,-2.0,-2.0,0.839743584394455,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.000919,-0.003624,-0.008374,-0.18387,0.793514,0.057406,0.043946,1.228199,0.859806]],[[54,11,77,-2,-2,80,-2,-2,9,-2,-2],[0.5,0.36666667461395264,0.5,-2.0,-2.0,0.41428571939468384,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002237,-0.006316,-0.016096,-0.24032,0.411157,0.021934,-1.150383,0.430204,0.085938,0.956308,0.447717]],[[12,6,4,-2,-2,3,-2,-2,2,78,-2,-2,1,-2,-2],[0.2604999989271164,0.5649999976158142,0.023000000044703484,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.9750000238418579,0.8452380895614624,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002556,0.057231,0.138873,-0.096102,1.405925,0.021714,1.158192,-0.141085,-0.00728,0.005143,-0.081366,0.848757,-0.029437,-0.403084,1.406132]],[[78,8,2,-2,-2,81,-2,-2,43,4,-2,-2,-2],[0.5505050718784332,0.17549999803304672,0.9850000143051147,-2.0,-2.0,0.281746044754982,-2.0,-2.0,0.5,0.09100000187754631,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000477,-0.025851,0.028208,0.17224,0.841194,-0.054075,-0.838921,0.056872,0.013528,0.006012,0.361306,-0.187612,1.294575]],[[70,6,3,-2,-2,12,-2,-2,42,82,-2,-2,-2],[0.5,0.8819999992847443,0.001500000071246177,-2.0,-2.0,0.33550000190734863,-2.0,-2.0,0.2899159789085388,0.1339285746216774,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003628,-0.002791,-0.022466,1.04305,-0.389921,0.015009,-0.546643,0.351848,0.052964,0.095444,1.205484,0.43888,-0.92835]],[[2,46,7,-2,-2,-2,46,85,-2,-2,5,-2,-2],[0.9750000238418579,1.5,0.9284999966621399,-2.0,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.6850000023841858,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004481,0.017402,0.022543,0.192614,0.992843,-0.689736,-0.016631,-0.028207,-0.207333,-1.097267,0.068336,1.092654,-0.583372]],[[19,55,20,-2,-2,-2,55,2,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.16700000315904617,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.00181,0.007684,0.010846,-0.529365,0.168104,-1.260577,-0.052761,-0.083812,-1.146605,-1.102715,-0.024885,-1.084076,-0.556167]],[[69,12,8,-2,-2,1,-2,-2,78,-2,42,-2,-2],[2.5,0.24400000274181366,0.3349999934434891,-2.0,-2.0,0.5,-2.0,-2.0,0.35164836049079895,-2.0,0.1180555559694767,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.001905,-0.007604,0.074409,0.132019,1.382935,-0.011751,-0.223317,0.606856,0.03822,-1.164927,0.061188,1.331271,0.160236]],[[21,5,77,-2,-2,8,-2,-2,8,-2,-2],[0.5,0.9835000038146973,0.5,-2.0,-2.0,0.9835000038146973,-2.0,-2.0,0.5300000160932541,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003732,-0.008623,-0.002194,-0.102764,0.516561,-0.056827,-1.389113,-0.37521,0.090305,0.720454,1.091451]],[[42,19,81,-2,-2,22,-2,-2,-2],[0.612500011920929,0.5,0.23303167521953583,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004708,0.006811,0.012016,0.237565,-0.217151,-0.045151,-1.206269,-0.59309,-1.530535]],[[19,9,4,-2,-2,63,-2,-2,22,2,-2,-2,12,-2,-2],[0.5,0.945499986410141,0.030499999411404133,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.9600000083446503,-2.0,-2.0,0.4599999934434891,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005684,0.01084,-0.009176,0.887663,-0.32366,0.023163,0.161094,0.935924,-0.04018,-0.066412,-0.912848,-1.097759,-0.019279,0.22725,-1.102294]],[[62,5,7,-2,-2,12,-2,-2,-2],[0.5,0.9615000188350677,0.9354999959468842,-2.0,-2.0,0.4505000114440918,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.00369,-0.006863,-0.000134,-0.055695,1.122289,-0.044486,-1.072468,-0.289497,0.751768]],[[16,79,69,-2,-2,69,-2,-2,9,-2,-2],[0.5,0.8708333373069763,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001001,0.004907,-0.006534,0.214501,-0.32838,0.028632,0.084451,0.88489,-0.059495,-1.359507,0.115342]],[[21,17,19,-2,-2,7,-2,-2,5,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.01699999999254942,-2.0,-2.0,0.13049999624490738,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.000176,-0.003511,0.001085,0.069309,-0.928494,-0.088821,-1.296866,-0.909984,0.066994,0.048085,1.060282]],[[63,8,82,-2,-2,44,-2,-2,8,-2,-2],[0.5,0.5940000116825104,0.46514423191547394,-2.0,-2.0,0.5,-2.0,-2.0,0.8290000259876251,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000329,-0.004394,0.012389,0.11227,1.776213,-0.02126,0.010743,-0.49763,0.066199,-0.418794,1.200803]],[[71,76,8,-2,-2,85,-2,-2,-2],[0.5,0.5,0.5940000116825104,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.0026,-0.000369,-0.009226,0.166512,-0.367756,0.027923,0.413635,-0.661537,-0.855513]],[[48,17,20,-2,-2,9,-2,-2,78,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.9294999837875366,-2.0,-2.0,0.5584415793418884,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002149,-0.002478,0.000953,-0.540395,0.061215,-0.077379,-1.374355,-0.175908,0.061164,-0.274407,1.134063]],[[77,21,78,-2,-2,5,-2,-2,3,69,-2,-2,82,-2,-2],[0.5,0.5,0.8452380895614624,-2.0,-2.0,0.13049999624490738,-2.0,-2.0,0.014500000048428774,0.5,-2.0,-2.0,0.09545454755425453,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000804,-0.004759,-0.008754,-0.207167,0.27225,0.068105,0.476446,0.808584,0.048549,-0.014568,0.78023,-0.789476,0.104879,0.191887,1.553339]],[[4,4,8,-2,-2,5,-2,-2,20,-2,38,-2,-2],[0.05849999934434891,0.045500000938773155,0.3994999974966049,-2.0,-2.0,0.3125,-2.0,-2.0,0.5,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003135,0.01671,0.004672,0.760791,-0.137847,0.140353,0.187746,1.791102,-0.015782,-1.04964,-0.010962,-0.254303,0.49175]],[[81,82,5,-2,-2,11,-2,-2,-2],[0.3798076957464218,0.30306123197078705,0.992499977350235,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001806,-0.000964,0.005146,0.119664,-0.712945,-0.043909,-0.633323,0.668258,1.011514]],[[9,4,9,-2,-2,5,-2,-2,42,80,-2,-2,8,-2,-2],[0.945499986410141,0.007500000298023224,0.03399999998509884,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.4580419659614563,0.5505050718784332,-2.0,-2.0,0.2084999978542328,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005954,-0.012688,0.077686,1.0886,1.3397,-0.022605,0.813911,-0.404658,0.01815,0.024757,1.371007,0.215257,-0.058082,-1.329388,-0.672241]],[[12,6,80,-2,-2,3,-2,-2,5,82,-2,-2,78,-2,-2],[0.2580000013113022,0.4425000101327896,0.7071428596973419,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.02850000001490116,0.39230769872665405,-2.0,-2.0,0.8090909123420715,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002701,0.055766,0.157693,0.522681,1.701167,0.01777,1.142079,-0.231842,-0.006993,0.019038,0.403356,-0.889129,-0.016216,-0.321795,0.158706]],[[70,16,76,-2,-2,9,-2,-2,42,9,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8680000007152557,-2.0,-2.0,0.30414746701717377,0.9914999902248383,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000172,-0.006645,-0.001794,-0.133254,0.263277,-0.0839,-1.326945,-0.750223,0.048769,0.095012,0.710566,1.263453,-1.027729]],[[2,6,69,-2,-2,7,-2,-2,5,46,-2,-2,6,-2,-2],[0.9750000238418579,0.9185000061988831,2.5,-2.0,-2.0,0.3739999979734421,-2.0,-2.0,0.4545000046491623,1.5,-2.0,-2.0,0.9945000112056732,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001981,0.010226,-0.011365,-0.270007,0.949325,0.033037,0.254445,1.136526,-0.022321,0.016157,-0.085551,1.060669,-0.048969,-0.794463,0.631248]],[[12,4,-2,6,-2,-2,6,81,-2,-2,12,-2,-2],[0.2580000013113022,0.013000000268220901,-2.0,0.792000025510788,-2.0,-2.0,0.8944999873638153,0.23303167521953583,-2.0,-2.0,0.3110000044107437,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000766,0.058968,-0.548092,0.083141,1.360889,0.357157,-0.005327,-0.022373,-0.107915,-0.839758,0.012115,-0.776337,0.245223]],[[79,-2,43,2,-2,-2,6,-2,-2],[0.232051283121109,-2.0,0.5,0.9350000023841858,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,-1,3,4,-1,-1,7,-1,-1],[2,-1,6,5,-1,-1,8,-1,-1],[0.000829,-1.651603,0.002118,-0.002471,0.377517,-0.141729,0.059782,-0.226959,0.891222]],[[56,22,2,-2,-2,52,-2,-2,9,-2,-2],[0.5,0.5,0.9950000047683716,-2.0,-2.0,0.5,-2.0,-2.0,0.9764999747276306,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001527,0.004937,0.009281,0.190148,-0.291571,-0.044601,-0.893844,-0.855506,-0.055751,-1.217261,-0.164381]],[[78,8,4,-2,-2,5,-2,-2,70,2,-2,-2,8,-2,-2],[0.5505050718784332,0.17549999803304672,0.9189999997615814,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.5,0.9950000047683716,-2.0,-2.0,0.2500000074505806,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00165,-0.017623,0.037494,0.747916,-0.234996,-0.042707,-1.171842,-0.320965,0.011995,0.002843,0.179945,-0.467987,0.076124,0.174329,1.009221]],[[54,89,69,-2,-2,3,-2,-2,-2],[0.5,0.5,2.5,-2.0,-2.0,0.005000000121071935,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00836,0.004757,0.00802,0.031304,0.499553,-0.066697,0.063605,-1.327807,0.888898]],[[6,12,4,-2,-2,4,-2,-2,79,10,-2,-2,3,-2,-2],[0.8840000033378601,0.2629999965429306,0.023000000044703484,-2.0,-2.0,0.03749999962747097,-2.0,-2.0,0.8660714328289032,0.3693181872367859,-2.0,-2.0,0.20750000327825546,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.007614,-0.008682,0.075277,-0.35727,1.173676,-0.015674,0.115746,-0.41784,0.024576,0.00593,-0.157088,0.609614,0.076425,0.419814,1.36407]],[[71,76,77,-2,-2,3,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.051500000059604645,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005153,0.007218,-0.001615,-0.115087,0.705045,0.034725,-0.079261,0.616117,-0.867347]],[[5,4,78,-2,-2,54,-2,-2,9,-2,69,-2,-2],[0.9905000030994415,0.05849999934434891,0.3550420254468918,-2.0,-2.0,0.5,-2.0,-2.0,0.9945000112056732,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003368,0.000955,0.022968,-0.803564,0.364923,-0.011076,-0.207255,0.732234,-0.056745,-0.001737,-0.088,-0.460366,-1.219737]],[[20,2,-2,-2,19,11,-2,-2,55,-2,-2],[0.5,0.9249999821186066,-2.0,-2.0,0.5,0.36666667461395264,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,9,-1,-1],[4,3,-1,-1,8,7,-1,-1,10,-1,-1],[-0.000867,-0.067592,0.624335,-0.993444,0.002782,0.008117,-0.02295,0.596503,-0.0408,-0.902267,-0.841936]],[[17,16,19,-2,-2,81,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.9294999837875366,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003286,0.007442,0.012461,0.195882,-0.844127,-0.066747,-0.422681,-1.736659,-0.077009,-1.259616,-0.433789]],[[89,2,17,-2,-2,85,-2,-2,6,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9099999964237213,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002092,0.005227,0.019471,0.3216,-0.899127,-0.013303,-0.072935,-0.937754,-0.066746,-0.150888,-1.090176]],[[86,89,69,-2,-2,7,-2,-2,-2],[0.5,0.5,1.5,-2.0,-2.0,0.03299999888986349,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00417,0.007524,0.010869,0.00809,0.419741,-0.060672,0.201348,-1.316067,-0.793734]],[[15,44,2,-2,-2,4,-2,-2,12,8,-2,-2,12,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.5399999916553497,0.16649999469518661,-2.0,-2.0,0.601500004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000814,-0.004642,0.008473,0.413257,-0.394635,-0.022378,0.643181,-0.376601,0.042949,0.081383,0.039442,1.005074,0.008452,-0.553703,0.896752]],[[5,79,-2,7,-2,-2,8,-2,2,-2,-2],[0.9835000038146973,0.23538461327552795,-2.0,0.9354999959468842,-2.0,-2.0,0.9864999949932098,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,-1,4,-1,-1,7,-1,9,-1,-1],[6,3,-1,5,-1,-1,8,-1,10,-1,-1],[0.002773,0.008046,-1.574652,0.009569,0.062663,0.676955,-0.041455,-1.36877,-0.007198,0.467165,-0.882756]],[[77,21,64,-2,-2,4,-2,-2,4,81,-2,-2,44,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.23799999803304672,-2.0,-2.0,0.0729999989271164,0.0833333358168602,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001108,-0.006515,-0.011286,-0.193382,0.896687,0.068783,0.422344,1.024962,0.045299,0.120657,0.074963,1.610081,0.007781,0.684716,-0.543054]],[[5,82,69,-2,-2,53,-2,-2,65,81,-2,-2,-2],[0.02949999924749136,0.39230769872665405,2.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.38750000298023224,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002772,0.019222,0.027161,0.300478,0.75966,-0.098113,1.123084,-1.230621,-0.011184,-0.016571,-0.230844,0.771461,0.599068]],[[20,5,-2,-2,63,78,-2,-2,-2],[0.5,0.8659999966621399,-2.0,-2.0,0.5,0.5505050718784332,-2.0,-2.0,-2.0],[1,2,-1,-1,5,6,-1,-1,-1],[4,3,-1,-1,8,7,-1,-1,-1],[0.000927,-0.070067,-0.256287,-1.146696,0.005462,0.001354,-0.239676,0.160526,0.712048]],[[37,9,82,-2,-2,1,-2,-2,89,42,-2,-2,-2],[1.5,0.99549999833107,0.44949495792388916,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.13076923787593842,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002177,-0.007616,0.004123,0.006517,1.206414,-0.051099,-0.775007,0.686948,0.024732,0.032838,0.752137,0.064764,-0.990085]]]
UMBRAL = 0.26
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
            # v6: como v4, se puntúa la respuesta ENTERA con un único corte.
            # (En v5 mandaba la peor oración con un corte propio para respuestas
            # largas: en el giro real marcaba el 71 % y bajó la nota a 0,377.)
            return reescalar(prob_caso(caso), UMBRAL)
        except Exception:
            # Ante un caso raro, mejor dar la alarma (un escape cuesta 4 veces más)
            return 1.0
