"""mi_bot.py - El Guardian (pcprompt), version 2.

Solo usa la libreria estandar, numpy y scikit-learn. No ejecuta ningun modelo de IA:
combina las senales precalculadas de cada caso con comprobaciones propias en Python y
un clasificador entrenado al arrancar con train.jsonl (reproducible: random_state fijo).

Como decide
-----------
1. Cada respuesta se parte en FRASES y cada frase se juzga como un caso de una frase
   (con sus fragmentos). Manda la peor frase: una frase falsa entre varias fieles no
   se diluye (respuestas largas del giro).
2. Por frase se calculan ~60 pistas:
   - senales del NLI (por fuente, todas juntas, fragmentos) y del LLM juez (veredicto,
     confianza, si la fuente que cita respalda la frase);
   - comprobaciones en Python: cifras inventadas o cuentas mal/bien hechas, atribucion,
     palabras nuevas, causas anadidas, condicionales y matices perdidos;
   - pistas v2 (patrones medidos en los analisis de errores): cautela convertida en
     certeza ("preve/podria/si" -> "llegara"), restriccion perdida EN SU ALCANCE
     ("solo/salvo/siempre que" de la clausula que la frase copia), ambito mas general,
     causa o finalidad anadida, parte convertida en todo, cuenta mal hecha con los
     operandos de la propia fuente, y si todas las cifras nuevas cuadran.
3. Clasificador: media de un HistGradientBoosting lento, un ExtraTrees y una regresion
   logistica (ordena mejor que cada uno por separado: AUROC 0,936 en H).
4. Corte con el coste 4:1, distinto segun la longitud (validado con CV por escenario):
   0,20 para respuestas de una frase y 0,35 para respuestas de varias (cada frase extra
   es una oportunidad mas de falsa alarma). El score se reescala para que ese corte caiga
   en 0,5, donde corta el organizador.

Validacion (CV agrupada por escenario, sin ver los casos evaluados): casos H ~+0,47 y
ensayo de respuestas de 3-4 frases montadas con frases H (como el giro) ~+0,45 con un
37 % de alucinaciones, frente a ~+0,42 y ~+0,31 de la version anterior.
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
from sklearn.ensemble import HistGradientBoostingClassifier



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


# =====================================================================
# Pistas nuevas (v2): patrones medidos en los análisis de distorsión,
# sobregeneralización y no respaldo. Cada una mira la frase contra SU
# fuente (la que cita el LLM, la más parecida y la que más la respalda).
# =====================================================================
def _n(t):
    return " " + norm(t) + " "


def _fuentes_frase(caso, o):
    J = set(fuentes_de_oracion(caso, o))
    rp = o.get("respaldo_por_fuente") or []
    if rp:
        J.add(max(range(len(rp)), key=lambda k: rp[k]))
    return [caso["fuentes"][j] for j in sorted(J) if 0 <= j < len(caso["fuentes"])]


# Cautela -> certeza: la fuente dice "podria", "preve", "estudia", "si"... y la frase lo afirma.
_CAUTELA_F = re.compile(r"\b(podr\w+|prev\w+|estudia\w*|propon\w+|propuesta|propuso|plante\w+|calcula\w*|estim\w+|advierte\w*|advirti\w*|apunta\w*|ha pedido|han pedido|pide|piden|pretend\w+|proyecci\w+|sondeo|si|en caso contrario|depende\w* de|pendiente\w*|posibl\w+|quiza\w*|probabl\w+|previsiblemente)\b")
_CAUTELA_R = re.compile(r"\b(podr\w+|pued\w+|prev\w+|estudia\w*|propon\w+|propuesta|propuso|plante\w+|calcula\w*|estim\w+|advierte\w*|advirti\w*|apunta\w*|pedido|pide|piden|pretend\w+|sondeo|si|depende\w*|pendiente\w*|posibl\w+|quiza\w*|probabl\w+|\w+ria|\w+rian|en principio|en caso de|segun)\b")
_FUTURO = re.compile(r"\b\w{3,}(?:rá|rán)\b")


def cautela_a_certeza(o, F):
    r, f = _n(o["texto"]), _n(" ".join(F))
    if not _CAUTELA_F.search(f) or _CAUTELA_R.search(r):
        return 0.0, 0.0
    return 1.0, float(bool(_FUTURO.search(o["texto"].lower())))


# Restriccion perdida EN SU ALCANCE: la frase copia la clausula del "solo/salvo/siempre que"
# y no conserva ninguna restriccion equivalente.
_RESTR = re.compile(r"\b(solo|unicamente|exclusivamente|salvo|excepto|siempre que|siempre y cuando|a menos que|aunque solo|en caso de|con tal de|previa\w*|previamente)\b")
_EQUIV = re.compile(r"\b(solo|unicamente|exclusivamente|salvo|excepto|siempre que|si|a menos que|unic[oa]s?|se limita\w*|limitad\w*|un minimo de|como minimo|al menos|tenia que|tiene que|tienen que|deben?|deberan?|no \w+ sin|sin \w+|no cubre|queda\w* fuera|reservad\w*|a partir de|siempre y cuando|con la condicion|requisito\w*|previa\w*|previamente|necesita\w*|hace falta|imprescindible|obligatori\w*)\b")
_STOPR = set("el la los las un una unos unas de del al a en y e o u que se su sus por para con es son fue fueron ha han lo le les este esta estos estas ese esa entre sobre como mas".split())


def _cont(t):
    return [w[:5] for w in re.findall(r"[a-z0-9ñ]+", norm(t)) if w not in _STOPR and len(w) > 2]


def restriccion_en_alcance(o, F):
    r = _n(o["texto"])
    if _EQUIV.search(r):
        return 0.0
    R = set(_cont(o["texto"]))
    for fu in F:
        for cl in re.split(r"[;:]|\.\s", _n(fu)):
            m = _RESTR.search(cl)
            if not m:
                continue
            ventana = _cont(cl[:m.start()])[-8:] + _cont(cl[m.end():])[:8]
            if ventana and sum(w in R for w in ventana) / len(ventana) >= 0.35:
                return 1.0
    return 0.0


# Ambito mas general: provincia, comarca, municipios, continentes... que la fuente no dice.
_GEN = re.compile(r"\b(provincia|comarca|region|ciudad|pais|municipios|barrios|distritos|pedanias|juzgados|muelles|conjunto|zona|territorio|europa|asia|america|africa|todos|todas|cualquier\w*|los demas|el resto)\b")


def ambito_general(o, F):
    r, f = _n(o["texto"]), _n(" ".join(F))
    return float(bool(set(_GEN.findall(r)) - set(_GEN.findall(f))))


# Causa o finalidad que ninguna fuente afirma.
_CAUSA = re.compile(r"\b(gracias a|porque|debido a|a causa de|por culpa de|se debe a|ya que|dado que|puesto que|lo que (permitio|explica|impulso|provoco|supuso|alarga|obligo|llevo|facilito|favorecio)|impulsad[oa]s? por|motivad[oa]s? por|provocad[oa]s? por|fruto de|como consecuencia|por (el|la|los|las|su|sus) (falta|cierre|final|retraso|aumento|caida|subida|llegada|tiron|efecto|mayor|menor|reduccion|crisis))\b")
_CAUSA_F = re.compile(_CAUSA.pattern + r"|\b(debers?e|podria deberse|atribuy\w+|explica\w*|motivo|razon|causa)\b")
_FIN = re.compile(r"\b(para (agilizar|aliviar|ahorrar|abaratar|compensar|responder|poder|mejorar|reducir|evitar|facilitar|garantizar|atraer|fomentar|impulsar|acelerar|reforzar)|con el fin de|con el objetivo de|a fin de)\b")


def causa_finalidad(o, caso):
    r, todas = _n(o["texto"]), _n(" ".join(caso["fuentes"]))
    c = float(bool(_CAUSA.search(r)) and not _CAUSA_F.search(todas))
    fi = float(bool(_FIN.search(r)) and not _FIN.search(todas))
    return c, fi


# Parte -> todo: "14 de los 23", "la mayoria"... y la frase pierde la parte.
_PARTE = re.compile(r"\b(\d+(?:[.,]\d+)?) de (?:los|las|sus|ellos|ellas|esos|esas)\b|\bla mayoria\b|\bparte de\b|\bsolo (\d+)\b")


def parte_a_todo(o, F):
    f = _n(" ".join(F)); wr = set(re.findall(r"[a-z0-9ñ]+(?:[.,][0-9]+)*", _n(o["texto"])))
    for m in _PARTE.finditer(f):
        nums = [x for x in m.groups() if x]
        if nums and not any(x in wr for x in nums):
            return 1.0
        if not nums and not _PARTE.search(_n(o["texto"])):
            return 1.0
    return 0.0


# Cuenta mal hecha: numero nuevo con pista de calculo que no sale de las cifras de la frase,
# su fuente y las fuentes de las entidades que nombra (no de todas: casi todo "saldria").
_PROP = re.compile(r"(?<![\wáéíóúñ])[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}")
_PISTA_CUENTA = re.compile(r"\b(es decir|mas que|menos que|sum\w+|en total|un \d|una media|de media|supone\w*|lo que da|tripl\w+|dupl\w+|de modo que|asi que|quedaban|quedan|puntos|diferencia|tercio|mitad|aument\w+ (un|en)|crec\w+ (un|en)|redu\w+ (un|en)|baj\w+ (un|en)|perdi\w+|el doble|algo mas de|mas del|a razon de|por habitante|precio medio|por cada|en conjunto|entre (los|las) dos)\b")


def _deriv_cerca(n, S):
    S = sorted({x for x in S if not (1900 <= x <= 2100)})[:60]

    def ok(v):
        return abs(n - v) <= max(0.51, 0.02 * abs(n))
    for a in S:
        for k in (1, 2, 3, 4, 0.5, 1 / 3, 0.25, 10, 0.1, 100, 0.01, 1000, 0.001, 60, 1 / 60, 12, 24, 7, 365):
            if ok(a * k):
                return True
        if ok(100 - a):
            return True
    for a, b in itertools.permutations(S, 2):
        if ok(a + b) or ok(a - b) or ok(a * b) or ok(a * b / 100) or (b and (ok(a / b) or ok(100 * a / b) or ok(100 * (a - b) / b))):
            return True
    for a, b in itertools.permutations(S, 2):   # dos pasos: "el resto" (100 - a) aplicado a una cantidad
        if a <= 100 and (ok((100 - a) * b / 100) or ok(b - a * b / 100)):
            return True
    if len(S) <= 14:
        for a, b, c in itertools.combinations(S, 3):
            if ok(a + b + c) or ok(100 - a - b) or ok(c - a - b):
                return True
            for x, y, z in itertools.permutations((a, b, c)):
                if x + y <= 100 and ok((100 - x - y) * z / 100):
                    return True
    return False


def cifras_nuevas(o, caso, F):
    """(hay cifras nuevas, todas cuadran con una cuenta, alguna no cuadra) sin exigir pista de calculo."""
    r = o["texto"]
    todas = [x for f in caso["fuentes"] for x in numeros(f)] + numeros(caso.get("pregunta", ""))
    nr = numeros(r)
    nuevos = [x for x in nr if not any(abs(x - a) <= max(0.002 * abs(a), 0.01) for a in todas) and not (1900 <= x <= 2100)]
    if not nuevos:
        return 0.0, 0.0, 0.0
    props = set(_PROP.findall(r))
    F2 = list(F) + [fu for fu in caso["fuentes"] if props & set(_PROP.findall(fu))]
    base = [x for fu in F2 for x in numeros(fu)] + [x for x in nr if x not in nuevos]
    malos = sum(1 for x in nuevos if not _deriv_cerca(x, base))
    return 1.0, float(malos == 0), float(malos > 0)


def cuenta_mal(o, caso, F):
    r = o["texto"]
    if not _PISTA_CUENTA.search(_n(r)):
        return 0.0
    todas = [x for f in caso["fuentes"] for x in numeros(f)] + numeros(caso.get("pregunta", ""))
    nr = numeros(r)
    nuevos = [x for x in nr if not any(abs(x - a) <= max(0.002 * abs(a), 0.01) for a in todas) and not (1900 <= x <= 2100)]
    if not nuevos:
        return 0.0
    props = set(_PROP.findall(r))
    F2 = list(F) + [fu for fu in caso["fuentes"] if props & set(_PROP.findall(fu))]
    base = [x for fu in F2 for x in numeros(fu)] + [x for x in nr if x not in nuevos]
    return float(any(not _deriv_cerca(x, base) for x in nuevos))


def pistas_v2(caso):
    o = caso["senales"]["oraciones"][0]
    F = _fuentes_frase(caso, o)
    c1, c2 = cautela_a_certeza(o, F)
    ca, fi = causa_finalidad(o, caso)
    return [c1, c2, restriccion_en_alcance(o, F), ambito_general(o, F), ca, fi, parte_a_todo(o, F), cuenta_mal(o, caso, F)]         + list(cifras_nuevas(o, caso, F))


NOMBRES_V2 = ["cautela_perdida", "cautela_a_futuro", "restr_alcance", "ambito_general", "causa_anadida",
              "finalidad_anadida", "parte_a_todo", "cuenta_mal", "cifras_nuevas", "cifras_cuadran", "cifras_no_cuadran"]



# ---- configuracion final (elegida con validacion cruzada; ver banco de pruebas) ----
CONFIG = dict(multi_aluc="caso", peso_otros=0.5, peso_fiel_multi=0.5, corte1=0.20, corteN=0.35, agg="max", drop=set(),
              hgb=dict(max_iter=400, learning_rate=0.03, max_depth=3, l2_regularization=1.0),
              mezcla={"hgb": 1.0, "et": 1.0, "lr": 0.5})


def frases(caso):
    return _por_frase(caso)


NOMBRES = NAMES + NAMES2 + ["deriv0", "deriv1", "pierde_cond", "cuentas_ok", "inventados", "frag_cob_min", "frag_n_sin_cubrir"]


def feats(caso):
    v = all_features(caso)
    if CONFIG.get("v2", True):
        extra = pistas_v2(caso)
        v = v + [x for x, n in zip(extra, NOMBRES_V2) if n in CONFIG.get("v2_usar", NOMBRES_V2)]
    if CONFIG.get("drop"):
        nombres = NOMBRES + [n for n in NOMBRES_V2 if CONFIG.get("v2", True) and n in CONFIG.get("v2_usar", NOMBRES_V2)]
        v = [x for x, n in zip(v, nombres) if n not in CONFIG["drop"]]
    return v


def unidades_entrenamiento(casos):
    U = []
    for c in casos:
        e = c["id"].split("::")[0]
        w = 1.0 if c["id"].startswith("H") else CONFIG["peso_otros"]
        F = _por_frase(c)
        if len(F) == 1:
            U.append((c, c["etiqueta"], w, e))
        elif c["etiqueta"] == 0:
            U += [(f, 0, CONFIG["peso_fiel_multi"], e) for f in F]
        else:
            modo = CONFIG["multi_aluc"]
            if modo == "caso":
                U.append((c, 1, w, e))
            elif modo == "llm":   # frases que el LLM no respalda -> 1
                for f in F:
                    v = (f["senales"]["oraciones"][0].get("llm") or {}).get("veredicto")
                    if v in ("no_respaldada", "contradicha"):
                        U.append((f, 1, w, e))
            # "nada": se descartan
    return U


class _Mezcla:
    """Media de probabilidades de varios modelos (cada uno con su peso)."""
    def __init__(self, modelos, pesos):
        self.modelos, self.pesos = modelos, pesos

    def predict_proba(self, X):
        P = sum(p * m.predict_proba(X)[:, 1] for m, p in zip(self.modelos, self.pesos)) / sum(self.pesos)
        return np.column_stack([1 - P, P])


def entrenar(X, y, w, e):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    modelos, pesos = [], []
    for k, peso in CONFIG.get("mezcla", {"hgb": 1.0}).items():
        if k == "hgb":
            m = HistGradientBoostingClassifier(random_state=0, **CONFIG["hgb"])
            m.fit(X, y, sample_weight=w)
        elif k.startswith("hgb_s"):   # otra semilla / otra configuracion
            m = HistGradientBoostingClassifier(random_state=int(k[5:]), **CONFIG["hgb"])
            m.fit(X, y, sample_weight=w)
        elif k == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=CONFIG.get("lr_C", 0.3), max_iter=3000))
            m.fit(X, y, logisticregression__sample_weight=w)
        elif k == "et":
            m = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.5, random_state=0, n_jobs=-1)
            m.fit(X, y, sample_weight=w)
        elif k == "rf":
            m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.4, random_state=0, n_jobs=-1)
            m.fit(X, y, sample_weight=w)
        modelos.append(m); pesos.append(peso)
    return modelos[0] if len(modelos) == 1 else _Mezcla(modelos, pesos)


def agregar(p):
    if CONFIG["agg"] == "max":
        return float(np.max(p))
    return float(1 - np.prod(1 - np.asarray(p)))


def corte(n):
    return CONFIG["corte1"] if n <= 1 else CONFIG["corteN"]


class Guardian:
    def __init__(self, casos=None):
        # casos: para validacion cruzada (validar.py); por defecto, todo train.jsonl
        if casos is None:
            casos = [json.loads(l) for l in open(_buscar_train(), encoding="utf-8") if l.strip()]
        U = unidades_entrenamiento(casos)
        X = np.array([feats(u[0]) for u in U])
        y = np.array([u[1] for u in U])
        w = np.array([u[2] for u in U])
        self.m = entrenar(X, y, w, [u[3] for u in U])

    def score(self, caso):
        fr = frases(caso)
        p = agregar(self.m.predict_proba(np.array([feats(f) for f in fr]))[:, 1])
        t = corte(len(fr))
        # reescalado lineal a trozos: el corte de la metrica cae en 0.5
        if p < t:
            return 0.5 * p / t
        return min(1.0, 0.5 + 0.5 * (p - t) / (1 - t))
