"""mi_bot.py - El Guardian.

Solo usa la libreria estandar, numpy y scikit-learn (permitidos por las reglas).
No ejecuta ningun modelo de IA: combina las senales precalculadas del caso con
comprobaciones propias en Python (cifras y cuentas, matices eliminados, causas
anadidas, generalizaciones) y un clasificador entrenado con train.jsonl.

El clasificador se entrena al crear el Guardian, leyendo train.jsonl. El umbral
optimo para la metrica (fallo x4) esta en p~0.25; el score se reescala para que
ese corte caiga en 0.5, que es donde corta el organizador.

Entre las pistas del clasificador esta la alarma de condicionales (de
mi_bot.py): la fuente dice "si se aprueba", "podria", "solo"... y la respuesta
lo pierde. Va como pista y no como alarma directa porque asi puntua mas en
validacion cruzada (+0.356 frente a +0.305): el modelo aprende cuando fiarse.

Otras dos pistas comprueban las cuentas: cuantos numeros de la respuesta salen
de una cuenta bien hecha con las fuentes y cuantos no salen de ningun sitio
(inventados o mal calculados). Bajan las falsas alarmas: +0.386 en validacion.

Y dos de cobertura por fragmentos: una respuesta que combina fuentes puede
tener cada trozo respaldado por una fuente distinta. Se mira el trozo peor
cubierto (cada uno con su mejor fuente). +0.02 de media en 4 repartos.
"""
import itertools, re, unicodedata

STOP = set("""el la los las un una unos unas de del al a en y o u e que se su sus con por para
como mas más es son fue fueron ha han lo le les este esta estos estas ese esa eso entre sobre
desde hasta tras durante sin ya también tambien muy pero no ni si sí cual cuál cuales donde
cuando quien quienes cuyo hay ser sido está están estaba otro otra otros otras mismo misma
año años cada todo toda dos tres""".split())

CAUSAL = ["gracias a", "debido a", "porque", "por lo que", "lo que permit", "lo que ha permit",
          "a causa de", "como consecuencia", "por ello", "para ", "con el fin", "lo que provoc",
          "lo que llev", "tras ", "motivo", "razón"]
GENERAL = ["todos", "todas", "siempre", "nunca", "ningún", "ninguna", "cualquier",
           "del país", "de todo", "de toda", "en general", "la mayoría", "totalidad",
           "cada ", "solo ", "sólo ", "únic", "exclusiv", "toda la", "todo el"]
HEDGE = ["más de", "menos de", "unos ", "unas ", "casi", "alrededor", "aproximad",
         "hasta ", "cerca de", "al menos", "previsiblemente", "podría", "estudia",
         "prevé", "propuso", "propone", "provisional"]

NUMWORDS = {"cero":0,"dos":2,"tres":3,"cuatro":4,"cinco":5,"seis":6,"siete":7,
            "ocho":8,"nueve":9,"diez":10,"once":11,"doce":12,"trece":13,"catorce":14,"quince":15,
            "veinte":20,"treinta":30,"cuarenta":40,"cincuenta":50,"sesenta":60,"setenta":70,
            "ochenta":80,"noventa":90,"cien":100,"ciento":100,"mil":1000,}


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def numbers(s):
    # Igual que numeros(): "1,85 millones" = 1.850.000, no 1,85
    return set(numeros(s))
    out = set()
    for m in re.finditer(r"\d[\d\.]*(?:,\d+)?", s):
        t = m.group().rstrip(".")
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        t = t.replace(",", ".")
        try:
            out.add(float(t))
        except ValueError:
            pass
    for w in re.findall(r"[a-záéíóúñ]+", s.lower()):
        if w in NUMWORDS:
            out.add(float(NUMWORDS[w]))
    return out


def words(s):
    return [w for w in re.findall(r"[a-z0-9ñ]+", norm(s)) if w not in STOP and len(w) > 2]


def stem(w):
    return w[:5]


def caps(s):
    # capitalised words not at sentence start
    toks = re.findall(r"\S+", s)
    return {norm(re.sub(r"\W", "", t)) for i, t in enumerate(toks) if i > 0 and t[:1].isupper() and len(re.sub(r"\W", "", t)) > 2}


def num_contexts(resp, fuentes):
    """For each number in the response, check whether some source containing that number
    also shares the content words that surround it in the response (attribution check)."""
    bad_attr, missing, total = 0, 0, 0
    allnums = set()
    for f in fuentes:
        allnums |= numbers(f)
    # derived numbers: sums, ratios, percentages of pairs
    nl = list(allnums)
    derived = set(nl)
    for a in nl:
        for b in nl:
            derived |= {a + b, abs(a - b)}
            if b:
                derived.add(round(a / b, 2)); derived.add(round(100 * a / b))
        derived |= {a * 1000, a / 1000, a * 2, a / 2, round(a / 10, 1)}
    rnums = numbers(resp)
    for n in rnums:
        if n in (1.0,):
            continue
        total += 1
        if n not in allnums:
            if n not in derived:
                missing += 1
            continue
    # attribution: for each number token in response, context window words
    for m in re.finditer(r"\d[\d\.]*(?:,\d+)?", resp):
        ns = numbers(m.group())
        if not ns:
            continue
        n = next(iter(ns))
        ctx = resp[max(0, m.start() - 60): m.end() + 40]
        cw = {stem(w) for w in words(ctx) if not w[0].isdigit()}
        srcs = [f for f in fuentes if n in numbers(f)]
        if not srcs or not cw:
            continue
        best = max(len(cw & {stem(w) for w in words(f)}) / len(cw) for f in srcs)
        if best < 0.25:
            bad_attr += 1
    return missing, bad_attr, total


def lex(caso):
    resp = caso["respuesta"]; fuentes = caso["fuentes"]
    nr, nf = norm(resp), norm(" ".join(fuentes))
    nq = norm(caso.get("pregunta", ""))
    srcstems = {stem(w) for w in words(" ".join(fuentes))} | {stem(w) for w in words(nq)}
    rw = words(resp)
    novel = [w for w in rw if stem(w) not in srcstems and not w[0].isdigit()]
    causal = sum(1 for c in CAUSAL if norm(c) in nr and norm(c) not in nf)
    general = sum(1 for c in GENERAL if norm(c) in nr and norm(c) not in nf)
    hedge_lost = sum(1 for c in HEDGE if norm(c) in nf and norm(c) not in nr)
    newcaps = len(caps(resp) - {w for w in re.findall(r"[a-z0-9ñ]+", nf)})
    missing, bad_attr, total = num_contexts(resp, fuentes)
    return [len(novel), len(novel) / max(1, len(rw)), causal, general, hedge_lost,
            newcaps, missing, bad_attr, total]


LLMV = {"respaldada": 0, "no_respaldada": 1, "contradicha": 2, None: 3}


def features(caso):
    s = caso["senales"]
    O, F = s["oraciones"], s["fragmentos"]
    f = []
    # sentence level (worst sentence)
    sup_all = [o["respaldo_todas"] for o in O]
    con_all = [o["contradiccion_todas"] for o in O]
    sup_max = [max(o["respaldo_por_fuente"], default=0) for o in O]
    con_max = [max(o["contradiccion_por_fuente"], default=0) for o in O]
    f += [min(sup_all), max(con_all), min(sup_max), max(con_max),
          min(max(a, b) for a, b in zip(sup_all, sup_max))]
    # LLM
    v = [o.get("llm", {}) or {} for o in O]
    worst = max(LLMV.get(x.get("veredicto"), 3) for x in v)
    conf_resp = min([x.get("confianza") or 0 for x in v if x.get("veredicto") == "respaldada"] or [1.0])
    conf_bad = max([x.get("confianza") or 0 for x in v if x.get("veredicto") in ("no_respaldada", "contradicha")] or [0.0])
    f += [worst == 1, worst == 2, worst == 3, conf_resp, conf_bad]
    # support of the source the LLM cited
    cited = []
    for o, x in zip(O, v):
        i = x.get("fuente")
        if isinstance(i, int) and 0 <= i < len(o["respaldo_por_fuente"]):
            cited.append((o["respaldo_por_fuente"][i], o["similitud_por_fuente"][i]))
    f += [min([c[0] for c in cited] or [0.5]), min([c[1] for c in cited] or [0.5])]
    # fragment level
    if F:
        fs = [max(max(x["respaldo_por_fuente"], default=0), x["respaldo_todas"]) for x in F]
        fc = [max(max(x["contradiccion_por_fuente"], default=0), x["contradiccion_todas"]) for x in F]
        fcm = [max(x["contradiccion_por_fuente"], default=0) for x in F]
        fsim = [max(x["similitud_por_fuente"], default=0) for x in F]
        f += [min(fs), sorted(fs)[1] if len(fs) > 1 else fs[0], sum(x < 0.5 for x in fs),
              sum(x < 0.2 for x in fs), max(fcm), min(fsim), len(F)]
    else:
        f += [1, 1, 0, 0, 0, 1, 0]
    f += [len(caso["fuentes"]), len(O)]
    f += lex(caso)
    return [float(x) for x in f]


NAMES = ["s_sup_all_min", "s_con_all_max", "s_sup_max_min", "s_con_max_max", "s_sup_best_min",
         "llm_no", "llm_contra", "llm_null", "llm_conf_resp", "llm_conf_bad", "cited_sup", "cited_sim",
         "f_sup_min", "f_sup_2nd", "f_n_lt05", "f_n_lt02", "f_con_max", "f_sim_min", "n_frag",
         "n_fuentes", "n_orac", "novel_n", "novel_frac", "causal", "general", "hedge_lost",
         "newcaps", "num_missing", "num_badattr", "num_total"]


CAUSAL2 = ["lo que", "impuls", "gracias", "debido", "por eso", "por ello", "motiv", "provoc",
           "permit", "a raiz", "favorec", "propici", "consecuencia", "porque", "para ", "a fin de",
           "tras ", "al ", "dado que", "ya que", "causa", "explica", "se debe", "fruto", "result"]
HEDGE2 = ["podria", "preve", "previsiblemente", "estima", "la mayoria", "mayoria", "solo ", "unicamente",
          "en torno", "unos ", "unas ", "mas de", "menos de", "casi", "alrededor", "algunos", "algunas",
          "varios", "varias", "parte de", "inferior", "superior", "hasta ", "al menos", "aproximad",
          "propuso", "propone", "estudia", "plantea", "proyecto", "provisional", "inicial", "previsto",
          "prevista", "si ", "siempre que", "excepto", "salvo", "con ingresos", "de personas", "del casco",
          "cerca de", "un tercio", "la mitad", "en parte", "temporal", "piloto"]
SCOPE = ["todos", "todas", "toda la", "todo el", "los municipios", "las empresas", "del pais",
         "en general", "siempre", "nunca", "ningun", "cualquier", "totalidad", "cada ", "conjunto",
         "la comarca", "la provincia", "la region", "los hospitales", "los centros", "los vecinos",
         "los socios", "los clientes", "los alumnos", "los usuarios"]


def num_list(s):
    # Igual que numeros(): "1,85 millones" = 1.850.000, no 1,85
    return numeros(s)
    out = []
    for m in re.finditer(r"\d[\d\.]*(?:,\d+)?", s):
        t = m.group().rstrip(".")
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        t = t.replace(",", ".")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def derivable(n, src):
    S = sorted(set(src))
    close = lambda a, b: abs(a - b) <= max(0.006 * abs(b), 0.051)
    if any(close(n, a) for a in S):
        return 2
    for a in S:
        for k in (1000, 0.001, 1e6, 1e-6, 2, 0.5, 3, 4, 10, 100, 0.01, 12, 1/12):
            if close(n, a * k):
                return 1
    for i, a in enumerate(S):
        for b in S:
            if b == a:
                continue
            if close(n, a + b) or close(n, abs(a - b)):
                return 1
            if b:
                for v in (a / b, 100 * a / b, 100 * (a - b) / b, 100 * (b - a) / b):
                    if abs(n - v) <= 0.55 or abs(n - round(v, 1)) < 0.06:
                        return 1
            for c in S:
                if c not in (a, b) and close(n, a + b + c):
                    return 1
    return 0


def extra(caso):
    resp = caso["respuesta"]; F = caso["fuentes"]
    nr, nf = norm(resp), norm(" ".join(F))
    src_nums = [n for f in F for n in num_list(f)]
    rn = num_list(resp)
    lit = sum(1 for n in rn if derivable(n, src_nums) == 2)
    der = sum(1 for n in rn if derivable(n, src_nums) == 1)
    bad = sum(1 for n in rn if derivable(n, src_nums) == 0)
    causal = sum(1 for c in CAUSAL2 if c in nr and c not in nf)
    causal_any = sum(1 for c in CAUSAL2 if c in nr)
    # sources used: LLM cited + top by support per sentence/fragment
    used = set()
    for o in caso["senales"]["oraciones"]:
        l = o.get("llm") or {}
        if isinstance(l.get("fuente"), int) and l["fuente"] < len(F):
            used.add(l["fuente"])
        rp = o["respaldo_por_fuente"]
        if rp:
            used.add(max(range(len(rp)), key=lambda i: rp[i]))
    for x in caso["senales"]["fragmentos"]:
        rp = x["respaldo_por_fuente"]; sp = x["similitud_por_fuente"]
        if rp:
            used.add(max(range(len(rp)), key=lambda i: rp[i]))
            used.add(max(range(len(sp)), key=lambda i: sp[i]))
    nu = norm(" ".join(F[i] for i in used))
    hedge = sum(1 for h in HEDGE2 if h in nu and h not in nr)
    scope = sum(1 for h in SCOPE if h in nr and h not in nf)
    # content words of used sources missing (qualifiers dropped): count of stems in cited-llm source not in response
    cov = []
    for o in caso["senales"]["oraciones"]:
        l = o.get("llm") or {}
        i = l.get("fuente")
        if isinstance(i, int) and i < len(F):
            sw = {stem(w) for w in words(F[i])}
            rw = {stem(w) for w in words(resp)}
            cov.append(len(sw - rw) / max(1, len(sw)))
    cov = max(cov or [0.5])
    # response stems not in used sources (but in other sources = mixing / not in any = novel)
    rws = {stem(w) for w in words(resp) if not w[0].isdigit()}
    usedst = {stem(w) for w in words(" ".join(F[i] for i in used))}
    allst = {stem(w) for w in words(" ".join(F))} | {stem(w) for w in words(caso.get("pregunta", ""))}
    not_used = len(rws - usedst) / max(1, len(rws))
    not_any = len(rws - allst)
    return [lit, der, bad, bad > 0, causal, causal_any, hedge, scope, cov, not_used, not_any, len(used)]

NAMES2 = ["num_lit", "num_der", "num_bad", "num_bad_any", "causal2", "causal_any", "hedge2", "scope2",
          "cited_uncov", "not_used", "not_any", "n_used"]


# --- Condicionales perdidos (de mi_bot.py) ---
# Palabras que expresan condicion, duda o limite. Se buscan sin distinguir
# mayusculas. Cada una se ha medido por separado sobre train: se quedan las que
# suben la nota y las condiciones claras que no la cambian (pensando en el giro
# legal). Fuera, por empeorar: "propuesta", "espera", "calcula", "mas de",
# "menos de", "hasta", "como maximo", "segun", "excluye", "proyecto", "aunque".
CONDICIONALES = [
    # Condicion: "si se aprueba", "siempre que llueva"...
    r"\bsi\b", r"s[oó]lo si", r"siempre que", r"siempre y cuando", r"a menos que",
    r"en caso de", r"en el caso de", r"a condici[oó]n", r"condicionad", r"supeditad",
    r"sujet[oa]s? a", r"depend", r"en su caso", r"cuando proceda|si procede",
    r"\bde (aprobarse|cumplirse|confirmarse|mantenerse|producirse|darse)",
    # Excepcion: "salvo los domingos", "excepto menores"...
    r"salvo", r"excepto", r"queda fuera|quedan fuera",
    # Posibilidad: "podria", "puede"...
    r"podr[ií]an?", r"\bpued(e|en)\b", r"posib", r"probab", r"previsiblemente",
    r"quiz[aá]s?|tal vez",
    # Planes e intenciones, aun no hechos: "preve", "propone", "estudia"...
    r"prev[eé]", r"previst", r"propon", r"propuso", r"estudia", r"plantea",
    r"pretend", r"baraj", r"contempl", r"intenci[oó]n",
    # Estimaciones y provisionalidad
    r"estim", r"provisional", r"pendiente", r"preliminar", r"en tr[aá]mite",
    r"borrador|anteproyecto",
    # Limites y aproximaciones: "solo", "al menos", "unos"...
    r"\bs[oó]lo\b|[uú]nicamente|exclusivamente", r"\bno\b", r"\bni\b",
    r"al menos|por lo menos", r"\bun(os|as)\b",

    # --- Ampliacion 2 (medidas igual; fuera por empeorar: "requisito",
    # "en torno a", "debera", "maximo de", "limitado", "voluntario",
    # "como minimo", "objetivo de") ---
    # Mas formas de condicion
    r"salvo que|excepto si|excepto que", r"a no ser que", r"en tanto (no|que)",
    r"hasta que", r"una vez que", r"en funci[oó]n de", r"a la espera de",
    r"de salir adelante|si sale adelante", r"con la condici[oó]n", r"si as[ií]",
    r"\bprevi[oa]s? (solicitud|autorizaci|aprobaci|pago|cita|inscripci|acuerdo|informe)",
    # Incertidumbre: aun no probado o no decidido
    r"presunt|supuestament", r"aparentemente|\bparece\b", r"presumiblemente",
    r"indicio", r"investig", r"negoci", r"en estudio|se valora|est[aá] valorando",
    r"tiene previsto|tienen previsto", r"intentar[aá]|tratar[aá] de",
    r"prometi|compromet|compromiso", r"aspira",
    # Aproximaciones
    r"aproximad", r"cerca de", r"\bcasi\b", r"algo m[aá]s de|algo menos de",
    # Vocabulario legal (pensando en el giro): excepciones, plazos, alcance
    r"con car[aá]cter general|por regla general", r"excepcional", r"en ning[uú]n caso",
    r"con excepci[oó]n|se except[uú]a|exceptuad", r"no ser[aá] de aplicaci[oó]n|no se aplica",
    r"exent|exenci[oó]n", r"potestativ|facultativ", r"obligatori", r"podr[aá]n?\b",
    r"transitori", r"a instancia|a petici[oó]n de", r"dentro del plazo|en el plazo de",
    r"no superior|no inferior|igual o (superior|inferior)", r"temporalmente",
    r"renovable|prorrogable|prorroga", r"cuando concurr",
]


def tiene_condicional(texto):
    texto = texto.lower()
    return any(re.search(p, texto) for p in CONDICIONALES)


def fuentes_de_oracion(caso, oracion):
    """Indices de las fuentes en las que se apoya la oracion: la que cita el
    LLM juez y la que mas se le parece."""
    fuentes = set()
    llm = oracion.get("llm") or {}
    i = llm.get("fuente")
    if isinstance(i, int) and 0 <= i < len(caso["fuentes"]):
        fuentes.add(i)
    sim = oracion["similitud_por_fuente"]
    if sim:
        fuentes.add(sim.index(max(sim)))
    return fuentes


def pierde_condicional(caso):
    """True si alguna oracion usa una fuente con condicion y la condicion
    desaparece en la oracion."""
    for oracion in caso["senales"]["oraciones"]:
        if tiene_condicional(oracion["texto"]):
            continue
        for i in fuentes_de_oracion(caso, oracion):
            if tiene_condicional(caso["fuentes"][i]):
                return True
    return False


# --- Comprobacion de cuentas ---
# Cifras inventadas y cuentas mal hechas. El LLM juez no sabe hacer cuentas: "142 nacimientos y 198 defunciones, es decir, 66 muertes
# mas" (son 56) le parece respaldado. Y al reves: una cuenta bien hecha
# ("320.000 mas que...") no esta tal cual en las fuentes y no debe parecer sospechosa.

NUMEROS_EN_LETRA = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}
# "1.800", "23,6", "1.234,5" o un numero en letra
PATRON_NUMERO = (r"\d+(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?|"
                 + "|".join(sorted(NUMEROS_EN_LETRA, key=len, reverse=True)))


def numeros(texto):
    """Lista de los numeros de un texto, con "millones" y "mil" aplicados
    ("1,6 millones" -> 1600000)."""
    texto = texto.lower()
    # Proporciones y horas que no son cifras: "seis de cada diez",
    # "dos tercios", "de cuatro a siete de la tarde"
    texto = re.sub(r"\b\w+ de cada (diez|cien|mil)\b", " ", texto)
    texto = re.sub(r"\b\w+ (tercios?|cuartos?|quintos?)\b", " ", texto)
    texto = re.sub(r"\b\w+ a \w+ de la (tarde|mañana|noche)\b", " ", texto)
    out = []
    for m in re.finditer(r"\b(" + PATRON_NUMERO + r")\b(\s+(millones|mil)\b)?", texto):
        s = m.group(1)
        if s in NUMEROS_EN_LETRA:
            v = float(NUMEROS_EN_LETRA[s])
        else:
            v = float(s.replace(".", "").replace(",", "."))
        if m.group(3) == "millones":
            v *= 1e6
        elif m.group(3) == "mil":
            v *= 1e3
        out.append(v)
    return out


def cerca(a, b, tol=0.01):
    return abs(a - b) <= max(tol * abs(b), 0.051)


def es_igual(n, lista):
    return any(cerca(n, a, 0.0001) for a in lista)


def sale_de_una_cuenta(n, principales, todas):
    """True si n se obtiene con una cuenta sencilla. Al menos un operando sale
    de la fuente que usa la oracion (principales); el otro puede ser de
    cualquier fuente. Asi no vale cualquier coincidencia casual."""
    for a in principales:
        # cambios de unidad (2.000 m = 2 km), porcentaje complementario, horas
        for k in (1000, 0.001, 1e6, 1e-6, 100, 0.01, 2, 0.5, 3, 4):
            if cerca(n, a * k):
                return True
        if cerca(n, 100 - a) or cerca(n, a - 12) or cerca(n, a + 12):
            return True
    for a in set(principales):
        for b in set(todas):
            if a == b:
                continue
            for x, y in ((a, b), (b, a)):
                # suma, diferencia, porcentaje de una cantidad, producto
                # ("85 euros diarios durante 60 dias" = 5.100)
                if cerca(n, x + y) or cerca(n, x - y) or cerca(n, x * y / 100) or cerca(n, x * y):
                    return True
                # que porcentaje es, cuanto ha variado en %, cuantas veces
                if y and (abs(n - 100 * x / y) < 0.6 or abs(n - 100 * (x - y) / y) < 0.6
                          or cerca(n, x / y, 0.03)):
                    return True
    for a, b, c in itertools.combinations(set(principales), 3):
        if cerca(n, a + b + c):
            return True
    return False


def revisar_cuentas(caso):
    """Clasifica cada numero de la respuesta: literal (esta tal cual en las
    fuentes), cuenta bien hecha (sale de una cuenta sencilla con ellas) o
    inventado (no sale de ningun sitio: cifra inventada o cuenta mal hecha)."""
    fuentes = caso["fuentes"]
    todas = [x for f in fuentes for x in numeros(f)] + numeros(caso.get("pregunta", ""))
    literales = cuentas_ok = inventados = 0
    for oracion in caso["senales"]["oraciones"]:
        en_oracion = numeros(oracion["texto"])
        for n in en_oracion:
            if es_igual(n, todas):
                literales += 1
                continue
            principales = [x for i in fuentes_de_oracion(caso, oracion) for x in numeros(fuentes[i])]
            principales += [m for m in en_oracion if m != n and es_igual(m, todas)]
            if sale_de_una_cuenta(n, principales, todas):
                cuentas_ok += 1
            else:
                inventados += 1
    return literales, cuentas_ok, inventados


# --- Cobertura por fragmentos ---
def cobertura_fragmentos(caso):
    """Respuestas que combinan fuentes: cada fragmento puede estar respaldado
    por una fuente distinta. Respaldo del fragmento peor cubierto (cada uno
    con su mejor fuente) y cuantos fragmentos no cubre ninguna fuente."""
    mejores = [max(f["respaldo_por_fuente"], default=0.0) for f in caso["senales"]["fragmentos"]]
    if not mejores:
        return 1.0, 0.0
    return min(mejores), float(sum(m < 0.5 for m in mejores))


def all_features(caso):
    src = [n for f in caso["fuentes"] for n in num_list(f)]
    d = [deriv2(n, src) for n in num_list(caso["respuesta"])]
    return (features(caso) + [float(v) for v in extra(caso)] + [float(0 in d), float(1 in d)]
            + [float(pierde_condicional(caso))]
            + [float(v) for v in revisar_cuentas(caso)[1:]]  # cuentas bien hechas, inventados
            + list(cobertura_fragmentos(caso)))


def deriv2(n, src):
    S = sorted(set(src))
    close = lambda a, b: abs(a - b) <= max(0.003 * abs(b), 0.011)
    if any(abs(n-a)<1e-9 for a in S): return 2
    for a in S:
        for k in (1000, 0.001, 1e6, 1e-6, 100, 0.01):
            if close(n, a*k): return 1
        if close(n, 100-a): return 1
    for a in S:
        for b in S:
            if a == b: continue
            if close(n, a+b) or close(n, abs(a-b)) or close(n, a*b/100): return 1
            if b and (abs(n - 100*a/b) < 0.51 or abs(n - 100*(a-b)/b) < 0.51 or abs(n-a/b)<0.051): return 1
    return 0


# ---------------------------------------------------------------------------
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UMBRAL = 0.20  # corte para la metrica (fallo x4): el mejor de la regresion logistica en
               # validacion cruzada; entre 0.18 y 0.25 la nota casi no cambia


def _buscar_train():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "train.jsonl"), "train.jsonl", os.path.join("kit", "train.jsonl"),
             os.path.join(here, "..", "..", "..", "kit", "train.jsonl")]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No encuentro train.jsonl: ponedlo junto a mi_bot.py")


def _por_frase(caso):
    """Parte un caso de varias frases en casos de una frase, cada uno con sus
    fragmentos. Un caso de una frase se devuelve tal cual."""
    O = caso["senales"]["oraciones"]
    if len(O) <= 1:
        return [caso]
    out = []
    for i, o in enumerate(O):
        F = [dict(f, oracion=0) for f in caso["senales"]["fragmentos"] if f.get("oracion") == i]
        out.append({"pregunta": caso.get("pregunta", ""), "fuentes": caso["fuentes"],
                    "respuesta": o["texto"], "senales": {"oraciones": [o], "fragmentos": F}})
    return out


class Guardian:
    def __init__(self, casos=None):
        # casos: para validacion cruzada (validar.py); por defecto, todo train.jsonl
        if casos is None:
            casos = [json.loads(l) for l in open(_buscar_train(), encoding="utf-8") if l.strip()]
        X = np.array([all_features(c) for c in casos])
        y = np.array([c["etiqueta"] for c in casos])
        w = np.array([1.0 if c["id"].startswith("H") else 0.5 for c in casos])
        # Regresion logistica: con pocos casos dificiles (H) generaliza mejor que
        # los arboles (HistGradientBoosting), que se aprendian de memoria el train.
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        self.m.fit(X, y, logisticregression__sample_weight=w)

    def score(self, caso):
        # Respuestas largas (giro): cada frase se juzga como un caso de una
        # frase, igual que los de train, y manda la peor. Si se mira la
        # respuesta entera, una frase falsa entre varias fieles se diluye.
        frases = _por_frase(caso)
        X = np.array([all_features(c) for c in frases])
        p = float(self.m.predict_proba(X)[:, 1].max())
        # reescalado lineal a trozos: UMBRAL -> 0.5
        if p < UMBRAL:
            return 0.5 * p / UMBRAL
        return 0.5 + 0.5 * (p - UMBRAL) / (1 - UMBRAL)
