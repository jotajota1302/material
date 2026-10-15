"""El Guardián: bot del equipo Syntax Terror (P13, P, D, J). FICHERO GENERADO, no editar a mano.

Se genera con `python herramientas/montar_bot.py` (desde kit/) a partir de
guardian/reglas.py (P), las señales de D (guardian/plantilla_bot.py) y los pesos
aprendidos en los casos H. Solo usa la librería estándar.

Cómo decide: señales del juez LLM y del NLI (peor oración) + reglas de verificación
(cifras, matices, alcance, causas; y cuentas que cuadran como pista a favor)
-> regresión logística ponderada 4:1 -> score. Dos modelos según lo que traiga el caso
(completo / sin juez LLM; sin señales, sospechoso). Cortes óptimos fuera de muestra desplazados
a 0,5: completo 0.53, sin_llm 0.44.
"""

# ===========================================================================
# REGLAS DE VERIFICACIÓN (P)  — copia de guardian/reglas.py
# ===========================================================================
import bisect
import re
import unicodedata

# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

STOP = set("""
a al algo ante asi aun bajo cabe como con contra cual cuales de del desde donde
dos durante e el ella ellas ello ellos en entre era es esa esas ese eso esos esta
estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi
mientras muy ni no nos o os otra otras otro otros para pero poco por que quien
se segun ser si sido sin sobre son su sus tambien te tiene tienen todo toda tras
tu un una uno unos unas y ya yo cada cuyo cuya
""".split())


def normalizar(texto):
    """Minúsculas y sin tildes (la ñ pasa a n). Conserva dígitos y puntuación."""
    texto = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokens(texto):
    return re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", normalizar(texto))


def es_contenido(t):
    return len(t) > 2 and t not in STOP and not t[0].isdigit()


def raiz(t):
    return t[:5]


def raices(texto):
    return {raiz(t) for t in tokens(texto) if es_contenido(t)}


def clausulas(texto):
    """Trocea por punto, punto y coma o dos puntos (sin romper «1.800» ni «10:05»)."""
    partes = re.split(r"(?<!\d)[.;:](?!\d)|[.;:](?=\s)", texto or "")
    return [p.strip() for p in partes if len(p.strip()) > 3]


def clausulas_relevantes(oracion, fuentes_relevantes, minimo=0.25):
    """Cláusulas de las fuentes que hablan de lo mismo que la oración."""
    r_or = raices(oracion)
    puntuadas = []
    for f in fuentes_relevantes:
        for c in clausulas(f):
            r_c = raices(c)
            if not r_c:
                continue
            comun = len(r_or & r_c)
            puntuadas.append((comun / len(r_c), comun, c))
    buenas = [c for frac, comun, c in puntuadas if comun >= 2 and frac >= minimo]
    if not buenas and puntuadas:
        buenas = [max(puntuadas)[2]]
    return buenas


# ---------------------------------------------------------------------------
# Cifras
# ---------------------------------------------------------------------------

MESES = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
PALABRAS_NUM = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "centenar": 100,
}
PALABRAS_RATIO = {"doble": 2, "triple": 3, "cuadruple": 4, "mitad": 0.5, "tercio": 1 / 3}
RAICES_RATIO = {"duplic": 2, "triplic": 3, "cuadruplic": 4}  # duplica, triplicaron, cuadruplicó...


def _ratio(w):
    if w in PALABRAS_RATIO:
        return PALABRAS_RATIO[w]
    return next((v for r, v in RAICES_RATIO.items() if w.startswith(r)), None)
APROX = r"(?:mas de(?:l)?|menos de(?:l)?|al menos|hasta(?: el| la| los| las)?|casi|cerca de(?:l)?|alrededor de(?:l)?|en torno a(?:l)?|aproximadamente|unos|unas|como maximo|como minimo|un maximo de|un minimo de|superior(?:es)? a(?:l)?|inferior(?:es)? a(?:l)?|rozando|rozo|roza)"
RE_NUM = re.compile(r"\d{1,2}:\d{2}|\d+(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")


def _valor(txt):
    if ":" in txt:
        h, m = txt.split(":")
        return int(h) * 60 + int(m)
    return float(txt.replace(".", "").replace(",", "."))


def extraer_cifras(texto):
    """Lista de cifras con su tipo.

    tipo: 'num' (cantidad), 'anio', 'fecha' (día de un mes), 'hora' (en minutos),
    'ratio' (doble, triple, mitad...). 'cmp' indica si va precedida de «más de» (+1),
    «menos de»/«hasta» (-1) o de una aproximación (0); None si es exacta.
    """
    t = normalizar(texto)
    cifras = []
    for m in RE_NUM.finditer(t):
        txt = m.group()
        antes = t[max(0, m.start() - 25):m.start()]
        despues = t[m.end():m.end() + 25]
        v = _valor(txt)
        if ":" in txt:
            tipo = "hora"
        elif re.match(r"\s*de\s+(" + "|".join(MESES) + r")\b", despues) and v <= 31:
            tipo = "fecha"
        elif "," not in txt and "." not in txt and 1900 <= v <= 2100 and not re.match(r"\s*(%|por ciento|euros|personas|habitantes|toneladas|millones)", despues):
            tipo = "anio"
        else:
            tipo = "num"
            mult = re.match(r"\s*(mil millones|millones|millon|mil)\b", despues)
            if not mult:  # «de 6,4 a 6,8 millones»: el «millones» vale para las dos
                m2 = re.match(r"\s*(?:a|y|e|o|al|hasta|frente a)\s+\d[\d.,]*\s*(mil millones|millones|millon)\b", despues)
                mult = m2 if m2 and "," in txt else None
            if mult:
                v *= {"mil millones": 1e9, "millones": 1e6, "millon": 1e6, "mil": 1e3}[mult.group(1)]
        cmp_ = None
        a = re.search(APROX + r"\s*$", antes)
        if a:
            s = a.group()
            cmp_ = 1 if s.startswith(("mas", "al menos", "superior", "como minimo", "un minimo")) else \
                -1 if s.startswith(("menos", "hasta", "inferior", "como maximo", "un maximo")) else 0
        decs = len(txt.split(",")[1]) if "," in txt else 0
        cifras.append({"texto": txt, "valor": v, "tipo": tipo, "cmp": cmp_, "decimales": decs,
                       "pct": bool(re.match(r"\s*(?:%|por ciento)", despues)),
                       "pos": m.start(), "mes": despues.split()[1] if tipo == "fecha" else None})
    # «seis de cada diez», «3 de cada 4» -> porcentaje
    ocupado = set()
    for m in re.finditer(r"\b(\w+) de cada (\w+)\b", t):
        a, b = m.group(1), m.group(2)
        va = PALABRAS_NUM.get(a) or (float(a) if a.isdigit() else None)
        vb = PALABRAS_NUM.get(b) or (float(b) if b.isdigit() else None)
        if va and vb:
            cifras = [c for c in cifras if not (m.start() <= c["pos"] < m.end())]
            cifras.append({"texto": m.group(), "valor": 100.0 * va / vb, "tipo": "num", "cmp": None,
                           "decimales": 0, "pos": m.start(), "mes": None})
            ocupado.update(range(m.start(), m.end()))
    for m in re.finditer(r"\b[a-z]+\b", t):
        w = m.group()
        if m.start() in ocupado:
            continue
        antes = t[max(0, m.start() - 25):m.start()]
        if w in PALABRAS_NUM:
            cifras.append({"texto": w, "valor": float(PALABRAS_NUM[w]), "tipo": "num",
                           "cmp": 1 if re.search(r"mas de\s*$", antes) else None,
                           "decimales": 0, "pos": m.start(), "mes": None})
        elif _ratio(w):
            cifras.append({"texto": w, "valor": float(_ratio(w)), "tipo": "ratio",
                           "cmp": 1 if re.search(r"mas (del?|de la)\s*$", antes) else None,
                           "decimales": 2, "pos": m.start(), "mes": None})
    return cifras


def _ops(a, b):
    """Cuentas directas entre dos cifras: suma, resta, producto, cociente, porcentajes."""
    out = [a + b, abs(b - a), a * b, a * b / 100]
    if a:
        out += [b / a, b / a * 100, (b - a) / a * 100]
    if b:
        out += [a / b, a / b * 100, (a - b) / b * 100]
    if 0 < a < 100:
        out.append((100 - a) * b / 100)  # «el resto» de un porcentaje
    if 0 < b < 100:
        out.append((100 - b) * a / 100)
    return out


def _derivados(nucleo, todas):
    """Todo lo que sale de una cuenta directa.

    nucleo: cifras de las fuentes relevantes y las que la oración toma de las fuentes.
    todas:  cifras de todas las fuentes. Solo se combinan con el núcleo (una cifra
            del núcleo y otra cualquiera), para no inventar coincidencias.
    """
    nucleo = sorted(set(v for v in nucleo if v) | {100.0})[:40]
    todas = sorted(set(v for v in (todas or ()) if v))[:80]
    out = set(nucleo)
    out.update(100 - v for v in nucleo if 0 < v < 100)
    for i, a in enumerate(nucleo):
        for b in nucleo[i + 1:]:
            out.update(_ops(a, b))
        for b in todas or ():
            out.update(_ops(a, b))
    if len(nucleo) <= 25:
        for i, a in enumerate(nucleo):
            for j in range(i + 1, len(nucleo)):
                for k in range(j + 1, len(nucleo)):
                    b, c = nucleo[j], nucleo[k]
                    out.update((a + b + c, c - a - b, a * b * c, a * b * c / 100))
    return sorted(out)


def _cerca(x, ordenados, tol):
    i = bisect.bisect_left(ordenados, x - tol)
    return i < len(ordenados) and ordenados[i] <= x + tol


def _en_rango(x, ordenados, lo, hi):
    i = bisect.bisect_left(ordenados, lo)
    return i < len(ordenados) and ordenados[i] <= hi


def _tolerancia(c):
    unidad = 10 ** (-c["decimales"])
    tol = max(0.5 * unidad, 0.005 * abs(c["valor"]))
    if c["cmp"] == 0:
        tol = max(tol, 0.06 * abs(c["valor"]))
    return tol


def comprobar_cifras(oracion, fuentes, fuentes_relevantes=None):
    """Compara las cifras de la oración con las de las fuentes.

    Devuelve:
      cifra_nueva          cifras que no aparecen tal cual en ninguna fuente
      cifra_no_derivable   cifras que ni aparecen ni salen de una cuenta directa
      fecha_nueva          años o fechas que no están en las fuentes
      ratio_falso          «el doble», «triplica»... que las cifras no sostienen
      cifra_de_otra_fuente la cifra existe, pero solo en fuentes que no hablan
                           de lo mismo que la oración (posible atribución cruzada)

    Las cuentas solo se hacen con cifras de las fuentes relevantes y con las que
    la propia oración toma de las fuentes: con todas las cifras de 8 fuentes,
    casi cualquier número «saldría» por casualidad.
    """
    cifras_or = extraer_cifras(oracion)
    cifras_f = [c for f in fuentes for c in extraer_cifras(f)]
    cifras_rel = [c for f in (fuentes_relevantes or fuentes) for c in extraer_cifras(f)]
    literales = sorted(c["valor"] for c in cifras_f)
    literales_rel = sorted(c["valor"] for c in cifras_rel)
    anios = {c["valor"] for c in cifras_f if c["tipo"] == "anio"}
    fechas = {(c["valor"], c["mes"]) for c in cifras_f if c["tipo"] == "fecha"}
    res = {"cifra_nueva": 0.0, "cifra_no_derivable": 0.0, "cifra_dudosa": 0.0, "fecha_nueva": 0.0, "ratio_falso": 0.0,
           "cifra_de_otra_fuente": 0.0, "cuenta_verificada": 0.0}
    tomadas = [x for x in cifras_or if x["tipo"] in ("num", "hora") and _cerca(x["valor"], literales, _tolerancia(x))]
    derivados = None
    calculadas = 0
    for c in cifras_or:
        if c["tipo"] == "anio":
            if c["valor"] not in anios:
                res["fecha_nueva"] = 1.0
            continue
        if c["tipo"] == "fecha":
            if (c["valor"], c["mes"]) not in fechas:
                res["fecha_nueva"] = 1.0
            continue
        tol = _tolerancia(c)
        if c["tipo"] == "ratio":
            # «triplica»: primero con las cifras de la propia oración
            propias = [t["valor"] for t in tomadas]
            if len(propias) >= 2:
                cocientes = sorted(max(a, b) / min(a, b) for i, a in enumerate(propias)
                                   for b in propias[i + 1:] if min(a, b) > 0)
                cocientes += sorted(min(a, b) / max(a, b) for i, a in enumerate(propias)
                                    for b in propias[i + 1:] if min(a, b) > 0)
                cocientes.sort()
                x = c["valor"]
                ok = _en_rango(x, cocientes, x, x * 1.5) if c["cmp"] == 1 else _cerca(x, cocientes, 0.06 * x)
                if not ok:
                    res["ratio_falso"] = 1.0
                continue
        elif _cerca(c["valor"], literales, tol):
            if c["tipo"] == "num" and c["valor"] > 31 and not _cerca(c["valor"], literales_rel, tol):
                res["cifra_de_otra_fuente"] = 1.0
            continue
        if c["tipo"] != "ratio":
            res["cifra_nueva"] = 1.0
        if c["tipo"] == "num" and c["texto"] in PALABRAS_NUM and c["valor"] <= 3:
            continue  # «dos», «tres» salen de casi cualquier cuenta: no aportan
        if derivados is None:
            nucleo = [x["valor"] for x in cifras_rel if x["tipo"] in ("num", "hora")]
            nucleo += [x["valor"] for x in tomadas]
            derivados = _derivados(nucleo, None)
            derivados_amplios = _derivados(nucleo, [x["valor"] for x in cifras_f if x["tipo"] in ("num", "hora")])
        # «De los 62 millones, 53 se destinaron...»: el «millones» va implícito
        escalas = (1, 1e6, 1e3) if c["valor"] < 1000 and re.search(r"\bmillon|\bmil\b", normalizar(oracion)) else (1,)
        cerca, lejos = _derivable(c, tol, escalas, derivados), _derivable(c, tol, escalas, derivados_amplios)
        if c["tipo"] == "ratio":
            if not lejos:
                res["ratio_falso"] = 1.0
        elif not lejos:
            res["cifra_no_derivable"] = 1.0
        elif not cerca:
            res["cifra_dudosa"] = 1.0
        elif c.get("pct") and len(tomadas) >= 2 and not _derivable(c, tol, (1,), _derivados([t["valor"] for t in tomadas], None)):
            # «812 de 1.326, más del 70 %»: el porcentaje debe salir de las cifras de la frase
            res["cifra_no_derivable"] = 1.0
        else:
            calculadas += 1
    # Pista POSITIVA: la oración hace cuentas y todas cuadran con las fuentes. El NLI
    # no sabe hacer cuentas («suman 8 de los 15») y da poco respaldo a estas frases fieles.
    if calculadas and not (res["cifra_no_derivable"] or res["cifra_dudosa"] or res["ratio_falso"]):
        res["cuenta_verificada"] = 1.0
    return res


def _derivable(c, tol, escalas, derivados):
    for e in escalas:
        x, t = c["valor"] * e, tol * e
        if c["cmp"] == 1:
            ok = _en_rango(x, derivados, x, x * 1.35)
        elif c["cmp"] == -1:
            ok = _en_rango(x, derivados, x * 0.65, x)
        else:
            ok = _cerca(x, derivados, t if c["tipo"] != "ratio" else 0.06 * x)
        if ok:
            return True
    return False


# ---------------------------------------------------------------------------
# Matices que desaparecen (la fuente es prudente, la respuesta no)
# ---------------------------------------------------------------------------

MATICES = {
    # nombre: (marcador en la fuente, equivalentes válidos en la respuesta)
    "modal_perdido": (
        r"\bpodri(?:a|an)\b|\bpodra(?:n)?\b|\bpuede(?:n)? que\b|\bposiblemente\b|\bprobablemente\b|\bprevisiblemente\b|\bquizas?\b|\btal vez\b",
        r"\bpod(?:ri|ra|e|i)\w*|\bpued\w*|\bposib\w*|\bprobab\w*|\bprevisib\w*|\bquizas?\b|\btal vez\b|\b\w{3,}(?:aria|eria|iria)n?\b",
    ),
    "condicion_perdida": (
        r"\bsi\b|\bsiempre que\b|\bsiempre y cuando\b|\ben caso de\b|\ba condicion de\b|\bcon la condicion\b|\bsalvo\b|\bexcepto\b|\ba excepcion\b|\ba menos que\b|\bsujet[oa]s? a\b|\bpendiente de\b|\bprevia\b|\bcondicionad",
        r"\bsi\b|\bsiempre\b|\ben caso\b|\bcondici\w*|\bsalvo\b|\bexcept\w*|\ba menos\b|\bsujet\w*|\bpendiente\b|\bprevi[ao]\b|\brequisito\w*|\bcuando\b|\bmientras\b"
        r"|\bpara que\b|\bpara no\b|\bten(?:er|ia|ian|dra|dran|ga|gan|ido)? que\b|\btiene(?:n)? que\b|\bdeb\w*|\brequi\w*|\bexig\w*|\bnecesit\w*|\bunic[oa]s?\b|\bsino\b",
    ),
    "restriccion_perdida": (
        r"\bsolo\b|\bsolamente\b|\bunicamente\b|\bexclusivamente\b|\bnada mas\b",
        r"\bsolo\b|\bsolamente\b|\bunicamente\b|\bexclusiv\w*|\bnada mas\b|\bpero no\b|\bno (?:a|en|para) (?:los|las|el|la)\b",
    ),
    # «propuso» no es «aprobó»; «ha pedido elevar» no es «subirá»
    "intencion_como_hecho": (
        r"\bpropon\w*|\bpropus\w*|\bpropuesta\w*|\bplantea\w*|\bestudia(?:n)?\b|\bbaraja\w*|\bpiden?\b|\bpidio\b"
        r"|\bpidieron\b|\bha(?:n)? pedido\b|\bsolicita(?:n|do)?\b|\bsolicito\b|\breclama\w*|\bpretende\w*|\baspira\w*"
        r"|\bdepende\w* de\b|\bintencion\b|\bplanea\w*|\bconfia\w* en\b|\bespera(?:n)?\b|\banalizar?\b|\bnegocia\w*",
        r"\bpropon\w*|\bpropus\w*|\bpropuesta\w*|\bplante\w*|\bestudi\w*|\bbaraj\w*|\bpid\w*|\bpedid\w*|\bsolicit\w*"
        r"|\breclam\w*|\bpretend\w*|\baspir\w*|\bdepend\w*|\bintenc\w*|\bplan\w*|\bconfi\w*|\besper\w*|\banaliz\w*"
        r"|\bnegoci\w*|\bquier\w*|\bobjetivo\b|\bpendiente\b|\bsi se\b",
    ),
}

# Condicional («permitiría», «costaría»): se busca con tildes para no confundirlo
# con adjetivos como «primaria». Sustantivos en -ería/-aría que no son verbos:
NO_CONDICIONAL = set("""tesorería consejería ganadería ingeniería galería librería panadería lotería cafetería
mensajería hostelería carnicería pescadería frutería joyería zapatería peluquería cervecería batería
minería cantería carpintería fontanería herrería lavandería lechería papelería perfumería relojería
sastrería tapicería guardería enfermería conserjería secretaría notaría vicesecretaría maquinaría
artillería porquería alfarería orfebrería chatarrería ferretería pastelería platería tintorería""".split())
RE_CONDICIONAL = re.compile(r"\b\w+(?:aría|ería|iría)n?\b")


def condicionales(texto):
    """Posiciones (fin) de verbos en condicional en el texto."""
    t = (texto or "").lower()
    return [m.end() for m in RE_CONDICIONAL.finditer(t) if m.group().rstrip("n") not in NO_CONDICIONAL]


def _relacionado(clausula, pos, r_or, ventana=8, atras=6):
    """¿Lo que rodea al marcador en la fuente aparece en la oración?

    Mira lo que sigue (lo que matiza un «podría») y lo que precede (la frase
    principal a la que se pega un «si se aprueba»)."""
    siguiente = [t for t in tokens(clausula[pos:]) if es_contenido(t)][:ventana]
    previo = [t for t in tokens(clausula[:pos]) if es_contenido(t)][-atras:] if atras else []
    return any(raiz(t) in r_or for t in siguiente + previo)


def matices_perdidos(oracion, relevantes):
    o = normalizar(oracion)
    r_or = raices(oracion)
    res = {k: 0.0 for k in MATICES}
    cond_or = bool(condicionales(oracion))
    for c in relevantes:
        cn = normalizar(c)
        for nombre, (en_fuente, equivalentes) in MATICES.items():
            for m in re.finditer(en_fuente, cn):
                if nombre == "modal_perdido" and cond_or:
                    continue
                if _relacionado(cn, m.end(), r_or) and not re.search(equivalentes, o):
                    res[nombre] = 1.0
        # un condicional en la fuente («permitiría») que pasa a futuro o presente
        if not cond_or and not re.search(MATICES["modal_perdido"][1], o):
            for fin in condicionales(c):
                if _relacionado(cn, fin, r_or, atras=0):
                    res["modal_perdido"] = 1.0
    return res


def aproximacion_perdida(oracion, relevantes):
    """«más de 150.000» -> «150.000»: la cifra se conserva pero pierde su matiz."""
    o = normalizar(oracion)
    cifras_or = extraer_cifras(oracion)
    for c in relevantes:
        cn = normalizar(c)
        for cf in extraer_cifras(c):
            if cf["tipo"] in ("anio", "fecha", "ratio"):
                continue
            # solo matices estrictos: «hasta» o «unos» los parafrasean las respuestas fieles
            if not re.search(r"(?:mas de(?:l)?|menos de(?:l)?|al menos|casi|superior(?:es)? a(?:l)?|inferior(?:es)? a(?:l)?)\s*$",
                             cn[max(0, cf["pos"] - 25):cf["pos"]]):
                continue
            for co in cifras_or:
                if co["tipo"] != cf["tipo"] or abs(co["valor"] - cf["valor"]) > 1e-9 * max(1, cf["valor"]):
                    continue
                cerca = o[max(0, co["pos"] - 30):co["pos"] + len(co["texto"]) + 20]
                if co["cmp"] is None and not re.search(r"\bo mas\b|\bo menos\b|\bminimo\b|\bmaximo\b|\bsupera|\brebasa|\bexcede|\bno llega|\bpor encima|\bpor debajo", cerca):
                    return 1.0
    return 0.0


RE_PARCIAL_F = r"\b\d+(?:[.,]\d+)* de (?:los|las)\b|\balgun(?:o|a|os|as)\b|\bvari(?:os|as)\b|\bparte de\b|\bla mayoria\b|\bciert(?:os|as)\b|\bdeterminad(?:os|as)\b|\bdetermin\w+ (?:casos|supuestos)\b"
RE_PARCIAL_O = r"\b\d[\d.,]* de (?:los|las)\b|\bde (?:los|las) \d|\balgun\w*|\bvari(?:os|as)\b|\bparte\b|\bmayoria\b|\bciert(?:os|as)\b|\bdetermin\w*|\bminoria\b|\bresto\b|\bde ellos\b|\bde ellas\b"
RE_UNIVERSAL = r"\btod[oa]s\b|\bcualquier\w*|\bsiempre\b(?! que| y cuando)|\bnunca\b|\bningun\w*|\bcada\b|\btotalidad\b|\bsin excepcion\b|\bel conjunto\b|\ben general\b|\bgeneralizad\w*|\btodo el\b|\btoda la\b"


def alcance(oracion, relevantes):
    """Sobregeneralización: se pierde un «14 de los 23» o aparece un «todos» nuevo."""
    o = normalizar(oracion)
    r_or = raices(oracion)
    parcial = 0.0
    universal = 0.0
    cns = [normalizar(c) for c in relevantes]
    for cn in cns:
        for m in re.finditer(RE_PARCIAL_F, cn):
            if _relacionado(cn, m.end(), r_or, ventana=5) and not re.search(RE_PARCIAL_O, o):
                parcial = 1.0
    for m in re.finditer(RE_UNIVERSAL, o):
        palabra = m.group().split()[0]
        if not any(re.search(r"\b" + palabra[:4], cn) for cn in cns):
            universal = 1.0
    return {"parcial_perdido": parcial, "universal_anadido": universal}


def modificador_omitido(oracion, fuentes):
    """«vino embotellado europeo» -> «vino europeo».

    Si dos palabras seguidas de la respuesta aparecen en una fuente con una o
    dos palabras de contenido en medio, y esas palabras no están en la
    respuesta, se ha quitado un modificador (suele ampliar el alcance).
    """
    t_or = tokens(oracion)
    set_or = set(t_or)
    pares = {(a, b) for a, b in zip(t_or, t_or[1:]) if es_contenido(a) and es_contenido(b)}
    if not pares:
        return 0.0
    preps = {"en", "de", "con", "sin", "para", "por"}
    for f in fuentes:
        t_f = tokens(f)
        for i, a in enumerate(t_f):
            for hueco in (1, 2, 3):
                j = i + hueco + 1
                if j < len(t_f) and (a, t_f[j]) in pares:
                    medio = t_f[i + 1:j]
                    cont = [m for m in medio if es_contenido(m)]
                    if cont and medio[0] not in preps and all(m in preps or (es_contenido(m) and m not in set_or) for m in medio):
                        return 1.0
                    if cont and medio[0] in preps and len(medio) >= 2 and all(m in preps or (es_contenido(m) and m not in set_or) for m in medio):
                        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Causas y consecuencias añadidas
# ---------------------------------------------------------------------------

RE_CAUSA = (r"\bgracias a\b|\bdebido a\b|\ba causa de\b|\bpor culpa de\b|\bporque\b|\bya que\b|\bpuesto que\b"
            r"|\bdado que\b|\bcon el (?:fin|objetivo|proposito) de\b|\bpara (?:poder |que |\w+(?:ar|er|ir)\b)"
            r"|\bse debe a\b|\bse explica por\b|\bmotivad[oa] por\b|\bimpulsad[oa] por\b|\bal \w+(?:ar|er|ir)\b"
            r"|\bpor (?:su|sus|la|el) (?:mayor|menor|buen|buena|mala|mal|alta|alto|baja|bajo|escasa|escaso|fuerte)\b")
RE_CONSECUENCIA = (r"\blo que (?:signific|supon|supus|permit|implic|demostr|reflej|indic|oblig|provoc|hac|hiz|explic|llev)\w*"
                   r"|\bde (?:modo|manera|forma) que\b|\bpor lo que\b|\basi que\b|\bcon lo que\b|\bpor eso\b"
                   r"|\bpor ello\b|\bpor tanto\b|\blo cual\b|\bcon ello\b")
RE_CAUSA_FUENTE = RE_CAUSA + r"|\bpor\b|\batribu\w+|\bmotivo\w*|\bcausa\w*|\brazon\w*|\bobjetivo\b|\bfin de\b|\bdebe\w*|\bexplica\w*|\bpermit\w*|\bgracias\b"


def conectores_anadidos(oracion, fuentes, relevantes):
    """Causa o consecuencia que la respuesta añade y ninguna fuente sostiene."""
    o = normalizar(oracion)
    r_fuentes = set().union(*(raices(f) for f in fuentes)) if fuentes else set()
    rel = " ".join(normalizar(c) for c in relevantes)
    res = {"causa_anadida": 0.0, "consecuencia_anadida": 0.0}
    for nombre, patron in (("causa_anadida", RE_CAUSA), ("consecuencia_anadida", RE_CONSECUENCIA)):
        for m in re.finditer(patron, o):
            siguiente = [t for t in tokens(o[m.end():]) if es_contenido(t)][:4]
            if not siguiente:
                continue
            nuevas = sum(raiz(t) not in r_fuentes for t in siguiente) / len(siguiente)
            sin_causa_en_fuente = not re.search(RE_CAUSA_FUENTE, rel)
            if nuevas >= 0.5 or (nombre == "causa_anadida" and sin_causa_en_fuente):
                res[nombre] = 1.0
    return res


# ---------------------------------------------------------------------------
# Contenido nuevo
# ---------------------------------------------------------------------------

def contenido_nuevo(oracion, fuentes, pregunta=""):
    """Fracción de palabras de contenido que no salen en las fuentes (ni en la pregunta)
    y la racha más larga de palabras nuevas seguidas (un inciso inventado)."""
    conocidas = set(raices(pregunta))
    for f in fuentes:
        conocidas |= raices(f)
    cont = [t for t in tokens(oracion) if es_contenido(t)]
    if not cont:
        return {"palabras_nuevas": 0.0, "racha_nueva": 0.0}
    nuevas = [raiz(t) not in conocidas for t in cont]
    racha = mejor = 0
    for n in nuevas:
        racha = racha + 1 if n else 0
        mejor = max(mejor, racha)
    return {"palabras_nuevas": sum(nuevas) / len(cont), "racha_nueva": min(1.0, mejor / 4)}


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

NOMBRES = ["cifra_nueva", "cifra_no_derivable", "cifra_dudosa", "fecha_nueva", "ratio_falso", "cifra_de_otra_fuente",
           "cuenta_verificada",
           "modal_perdido", "condicion_perdida", "restriccion_perdida", "intencion_como_hecho", "aproximacion_perdida",
           "parcial_perdido", "universal_anadido", "modificador_omitido",
           "causa_anadida", "consecuencia_anadida", "palabras_nuevas", "racha_nueva"]


# Las que D debe meter en el modelo. Las demás (cifra_nueva, palabras_nuevas,
# universal_anadido, cifra_de_otra_fuente, racha_nueva, fecha_nueva) en H apuntan
# «al revés» (más palabras nuevas = más fiel) porque las alucinaciones de H se
# escribieron editando poco la fuente. Es un atajo del formato: entrenado en H y
# probado en el resto del train hunde la puntuación (de +0,54 a +0,14 con el
# corte en 0,5), así que en el giro haría daño. Se calculan, pero no se usan.
RECOMENDADAS = ["cifra_no_derivable", "cifra_dudosa", "ratio_falso", "cuenta_verificada",
                "modal_perdido", "condicion_perdida", "restriccion_perdida", "intencion_como_hecho",
                "aproximacion_perdida", "parcial_perdido", "modificador_omitido",
                "causa_anadida", "consecuencia_anadida"]


def caracteristicas_reglas(oracion, fuentes_relevantes, fuentes, pregunta=""):
    """Pistas de alucinación de UNA oración. Cada valor está en [0, 1].

    oracion             texto de la oración (o fragmento) de la respuesta
    fuentes_relevantes  str o lista de str: la fuente que cita el LLM y las más
                        parecidas (si no se sabe, pasad todas)
    fuentes             todas las fuentes del caso
    """
    if isinstance(fuentes_relevantes, str):
        fuentes_relevantes = [fuentes_relevantes]
    fuentes = [f for f in (fuentes or []) if f]
    fuentes_relevantes = [f for f in (fuentes_relevantes or []) if f] or fuentes
    res = dict.fromkeys(NOMBRES, 0.0)
    if not oracion or not fuentes:
        return res
    relevantes = clausulas_relevantes(oracion, fuentes_relevantes)
    res.update(comprobar_cifras(oracion, fuentes, fuentes_relevantes))
    res.update(matices_perdidos(oracion, relevantes))
    res["aproximacion_perdida"] = aproximacion_perdida(oracion, relevantes)
    res.update(alcance(oracion, relevantes))
    res["modificador_omitido"] = modificador_omitido(oracion, fuentes_relevantes)
    res.update(conectores_anadidos(oracion, fuentes, relevantes))
    res.update(contenido_nuevo(oracion, fuentes, pregunta))
    return res


def _fuentes_relevantes(entrada, fuentes, top=2):
    """Fuente citada por el LLM + las que más respaldan + las más parecidas."""
    idx = set()
    llm = entrada.get("llm") or {}
    i = llm.get("fuente")
    if isinstance(i, int) and 0 <= i < len(fuentes):
        idx.add(i)
    for campo in ("respaldo_por_fuente", "similitud_por_fuente"):
        vals = entrada.get(campo) or []
        orden = sorted(range(min(len(vals), len(fuentes))), key=lambda k: -vals[k])
        idx.update(orden[:top])
        idx.update(k for k in orden if campo == "respaldo_por_fuente" and vals[k] >= 0.5)
    return [fuentes[k] for k in sorted(idx)] or list(fuentes)


def _oraciones(caso):
    ors = (caso.get("senales") or {}).get("oraciones") or []
    if ors:
        return ors
    partes = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡«])", caso.get("respuesta") or "")
    return [{"texto": p} for p in partes if p.strip()]


def reglas_caso(caso, estricto=False):
    """Pistas del caso completo: el máximo de cada pista sobre todas las oraciones
    (una sola afirmación falsa basta, así que no se promedia)."""
    fuentes = caso.get("fuentes") or []
    total = dict.fromkeys(NOMBRES, 0.0)
    try:
        for o in _oraciones(caso):
            r = caracteristicas_reglas(o.get("texto", ""), _fuentes_relevantes(o, fuentes),
                                       fuentes, caso.get("pregunta", ""))
            for k, v in r.items():
                total[k] = max(total[k], v)
    except Exception:
        if estricto:  # al medir queremos ver el fallo
            raise
        # en el bot, una regla rota no debe tumbarlo: el resto de señales sigue
    return total


# ===========================================================================
# SEÑALES (D) Y MODELO
# ===========================================================================
import math
import unicodedata


SENALES_LLM = ["llm_no_respaldada", "llm_contradicha", "llm_confianza_duda"]
SENALES_NLI = ["nli_min_respaldo", "frag_min_respaldo"]
# Fuera: nli_max_contradiccion y nli_respaldo_todas_min. En H no distinguen (AUROC ~0,5)
# y el modelo les daba pesos con el signo al revés: era ruido.

MODELOS = {
    "completo": SENALES_LLM + SENALES_NLI + RECOMENDADAS,
    "sin_llm": SENALES_NLI + RECOMENDADAS,
}
# Sin LLM ni NLI no hay modelo: uno «solo reglas» daba −0,05 fuera de H (peor que
# marcarlo todo), así que en ese caso se marca como sospechoso (puntuación 0 asegurada).
SIN_SENALES = 0.6


def _veredicto(v):
    """'Respaldada', 'NO RESPALDADA', 'no-respaldada'... -> respaldada / no_respaldada / contradicha / None."""
    if not isinstance(v, str):
        return None
    v = "".join(c for c in unicodedata.normalize("NFD", v.strip().lower()) if unicodedata.category(c) != "Mn")
    v = v.replace("-", "_").replace(" ", "_")
    if v.startswith("contradic"):
        return "contradicha"
    if v.startswith(("no_resp", "no_sop", "sin_resp")):
        return "no_respaldada"
    if v.startswith("respald"):
        return "respaldada"
    return None


def _num(x, defecto=0.0):
    try:
        x = float(x)
        return x if x == x else defecto  # descarta NaN
    except (TypeError, ValueError):
        return defecto


def _max(lista):
    return max((_num(v) for v in (lista or [])), default=None)


def caracteristicas_senales(caso):
    """Señales de D (kit/guardian/senales.py), con estos arreglos:
    - el original devolvía dentro del bucle y solo miraba la primera oración;
    - con "confianza": null lanzaba una excepción;
    - veredictos normalizados (mayúsculas, tildes, espacios);
    - si no hay oraciones se usan los fragmentos para el NLI.
    Devuelve también qué hay disponible (hay_llm, hay_nli) para elegir modelo.
    """
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    fragmentos = senales.get("fragmentos") or []
    no_resp = contra = duda = 0.0
    hay_llm = False
    min_resp = None
    for o in oraciones:
        llm = o.get("llm") or {}
        ver = _veredicto(llm.get("veredicto"))
        # Sin confianza nos quedamos con el veredicto y una confianza típica: con 0,5 el
        # bot veía una «duda» enorme en cada frase y lo marcaba todo.
        conf = min(1.0, max(0.0, _num(llm.get("confianza"), 0.9)))
        if ver == "no_respaldada":
            no_resp = max(no_resp, conf)
        elif ver == "contradicha":
            contra = max(contra, conf)
        elif ver == "respaldada":
            duda = max(duda, 1.0 - conf)
        hay_llm = hay_llm or ver is not None
        m = _max(o.get("respaldo_por_fuente"))
        if m is not None:
            min_resp = m if min_resp is None else min(min_resp, m)
    frag = None
    for f in fragmentos:
        m = _max(f.get("respaldo_por_fuente"))
        if m is not None:
            frag = m if frag is None else min(frag, m)
    if min_resp is None:
        min_resp = frag
    if frag is None:
        frag = min_resp
    hay_nli = min_resp is not None
    return {"llm_no_respaldada": no_resp, "llm_contradicha": contra, "llm_confianza_duda": duda,
            "nli_min_respaldo": min_resp if hay_nli else 0.0, "frag_min_respaldo": frag if hay_nli else 0.0,
            "hay_llm": hay_llm, "hay_nli": hay_nli}


def caracteristicas(caso):
    """Todas las características del caso (señales de D + reglas de P) en un dict."""
    d = caracteristicas_senales(caso)
    d.update(reglas_caso(caso))
    return d


def elegir_modelo(d):
    if d["hay_llm"] and d["hay_nli"]:
        return "completo"
    return "sin_llm" if d["hay_nli"] else None


def vector(d, modelo):
    return [float(d[k]) for k in MODELOS[modelo]]


# === PARAMETROS (los escribe herramientas/montar_bot.py) ===
PARAMS = {
    'completo': {  # corte 0.53 desplazado a 0,5
        "media": [0.059509, 0.127924, 0.030032, 0.409474, 0.484867, 0.050713, 0.045959, 0.00317, 0.244057, 0.064976, 0.079239, 0.063391, 0.034865, 0.00317, 0.052298, 0.045959, 0.049128, 0.011094],
        "desv": [0.218582, 0.326624, 0.040087, 0.38843, 0.387036, 0.219411, 0.209396, 0.05621, 0.429527, 0.246484, 0.270112, 0.243666, 0.183439, 0.05621, 0.222627, 0.209396, 0.216136, 0.10474],
        "coef": [0.722554, 1.340793, 0.631684, -0.176356, -0.554745, 0.540641, 0.070281, 0.056413, -0.785944, 0.387702, 0.46382, 0.385374, 0.142622, 0.131769, 0.474411, 0.35346, 0.093844, 0.140709],
        "sesgo": 1.243455,
    },
    'sin_llm': {  # corte 0.44 desplazado a 0,5
        "media": [0.409474, 0.484867, 0.050713, 0.045959, 0.00317, 0.244057, 0.064976, 0.079239, 0.063391, 0.034865, 0.00317, 0.052298, 0.045959, 0.049128, 0.011094],
        "desv": [0.38843, 0.387036, 0.219411, 0.209396, 0.05621, 0.429527, 0.246484, 0.270112, 0.243666, 0.183439, 0.05621, 0.222627, 0.209396, 0.216136, 0.10474],
        "coef": [-0.477173, -0.49107, 0.479566, 0.024591, 0.015569, -0.868186, 0.317842, 0.373261, 0.33875, 0.186139, 0.213857, 0.436973, 0.287623, 0.138776, 0.155897],
        "sesgo": 1.480688,
    },
}
# === FIN PARAMETROS ===


class Guardian:
    def score(self, caso):
        try:
            d = caracteristicas(caso)
            modelo = elegir_modelo(d)
            if modelo is None:
                return SIN_SENALES
            p = PARAMS[modelo]
            z = p["sesgo"]
            for v, m, s, w in zip(vector(d, modelo), p["media"], p["desv"], p["coef"]):
                z += w * (v - m) / s
            prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            return min(1.0, max(0.0, prob))
        except Exception:
            # Ante cualquier fallo, mejor una falsa alarma (cuesta 1) que un escape (cuesta 4)
            return 0.6
