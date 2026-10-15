"""El Guardián — fernando_v1. Solo librería estándar (math, re, unicodedata).

Cascada:
  Capa 1. Si el juez LLM dice "no_respaldada" o "contradicha" en alguna oración,
          la respuesta alucina con probabilidad alta (92 % en los casos H del
          train, por encima del 20 % que exige el coste 4:1 en cualquier tramo de
          confianza): score = 0,85 + 0,10·confianza.
  Capa 2. Si el LLM respalda todas las oraciones (o no contestó: veredicto null,
          que se trata como sospechoso con un indicador y un suelo de 0,5), un
          especialista entrenado SOLO en esa zona gris decide. Sus señales son
          booleanos o proporciones (nada que crezca con la longitud), agregadas
          por el PEOR fragmento/oración:
            - respaldo NLI (fuente que cita el LLM, todas las fuentes, mejor fuente),
            - alineación fragmento -> fuente (cifras ausentes en la fuente alineada
              pero presentes en otra = atribución cruzada; cifras no derivables
              con cuentas exactas; matices de la fuente que desaparecen),
            - universales, causas y comparaciones añadidas que ninguna fuente dice.
  El especialista es una regresión logística entrenada con pesos 4:1 (escape
  pesa 4), así que su corte natural ya es 0,5. Lo entrena lab/entrenar.py y lo
  escribe en MODELO (abajo); este fichero no lee nada del disco.
"""
import math
import re
import unicodedata


# ------------------------------------------------------------------ texto
def norm(t):
    t = unicodedata.normalize("NFD", (t or "").lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


VACIAS = set("""a al algo ante antes asi aun bajo cada como con contra cual cuando de del desde
donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba
estas este esto estos fue fueron ha habia han hasta hay la las le les lo los mas me mientras muy
ni no nos o otra otras otro otros para pero por que quien se segun ser si sido sin sobre son su
sus tambien tan tanto tiene tienen tras un una unas uno unos y ya""".split())


def raices(t):
    return {w[:5] for w in re.findall(r"[a-z]+", norm(t)) if len(w) >= 4 and w not in VACIAS}


# ------------------------------------------------------------------ cifras
RE_NUM = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?")
PALABRAS_NUM = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
                "nueve": 9, "diez": 10, "once": 11, "doce": 12, "quince": 15, "veinte": 20,
                "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
                "ochenta": 80, "noventa": 90, "cien": 100, "doble": None, "mitad": None}
RE_PAL_NUM = re.compile(r"\b(" + "|".join(k for k, v in PALABRAS_NUM.items() if v) + r")\b")
RE_CADA = re.compile(r"\b(\w+) de cada (\w+)\b")
ESCALAS = {"mil": 1e3, "millon": 1e6, "millones": 1e6}


def _valor(s):
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", s):
        s = s.replace(".", "")
    return float(s.replace(",", "."))


def cifras(texto):
    """Valores numéricos de un texto: dígitos, números en letra y 'X de cada Y' (%)."""
    t = norm(texto)
    out = []
    for m in RE_NUM.finditer(t):
        try:
            v = _valor(m.group())
        except ValueError:
            continue
        sig = t[m.end():m.end() + 9].split()
        if sig and sig[0] in ESCALAS:
            out.append(v * ESCALAS[sig[0]])
        out.append(v)
    out += [float(PALABRAS_NUM[m.group(1)]) for m in RE_PAL_NUM.finditer(t)]
    for m in RE_CADA.finditer(t):
        a, b = (PALABRAS_NUM.get(x) or (_valor(x) if x.isdigit() else None) for x in m.groups())
        if a and b:
            out.append(100.0 * a / b)
    return out


MESES = {m: i + 1 for i, m in enumerate("enero febrero marzo abril mayo junio julio agosto "
                                         "septiembre octubre noviembre diciembre".split())}
RE_FECHA = re.compile(r"\b(" + "|".join(MESES) + r")(?: de(?:l)? (\d{4}))?")


def meses_abs(texto):
    """Fechas como nº de mes absoluto (año·12 + mes), para comprobar plazos ('nueve meses después')."""
    out, anio = [], None
    for m in RE_FECHA.finditer(norm(texto)):
        anio = int(m.group(2)) if m.group(2) else anio
        out.append((anio or 0) * 12 + MESES[m.group(1)])
    return out


def cerca(a, b):
    return abs(a - b) <= max(0.002 * max(abs(a), abs(b)), 0.051)


def esta(v, valores):
    return any(cerca(v, w) for w in valores)


def sale_de(v, derivadas):
    """v sale de una cuenta: igual, o redondeo a entero de un porcentaje/cociente (<= 100)."""
    return any(cerca(v, d) or (v <= 100 and v == int(v) and abs(v - d) < 0.5) for d in derivadas)


def derivables(valores):
    """Cifras que salen con cuentas exactas de dos cifras de las fuentes."""
    vs = sorted(set(valores))[:40]
    out = set()
    for v in vs:
        out.update((v * 1e3, v / 1e3, v * 1e6, v / 1e6))  # cambio de unidad
    for i, a in enumerate(vs):
        for b in vs[i + 1:]:
            out.update((a + b, abs(b - a)))
            # "el 18 % de 48.000", "el resto" (100 - p) %, y subidas/bajadas de un p %
            out.update((a * b / 100, a * (100 - b) / 100, b * (100 - a) / 100,
                        a * (1 + b / 100), a * (1 - b / 100), b * (1 + a / 100), b * (1 - a / 100)))
            if a:
                out.update((100 * b / a, 100 * a / b if b else 0, b / a, 100 * (b - a) / a))
    return out


# ------------------------------------------------------------------ léxico general del español
def _rx(lista):
    return [re.compile(p) for p in lista]


UNIVERSALES = _rx([r"\btod[oa]s\b", r"\btodo el\b", r"\btoda la\b", r"\bcualquier", r"\bsiempre\b",
                   r"\bnunca\b", r"\bningun[oa]?\b", r"\btotalidad\b",
                   r"\btotalmente\b",
                   r"\bdel pais\b", r"\bnacional\b",])
CAUSALES = _rx([r"\bgracias a\b", r"\bdebido a\b", r"\bporque\b", r"\bpor lo que\b", r"\bya que\b", r"\bpor eso\b",
                r"\blo que (permitio|provoco|supuso|llevo|hizo|causo|genero|obligo|explica)\b",
                r"\bpermiti[oa]\b", r"\bimpulsad[oa] por\b",
                r"\bcon el (fin|objetivo|proposito) de\b", r"\bpara (que|lograr|conseguir|evitar|garantizar)\b", r"\bdebid[oa]s? a\b"])
COMPARATIVAS = _rx([r"\b(mas|menos) \w+ que\b", r"\b(mas|menos) que\b", r"\bsuperior(es)? a\b",
                    r"\binferior(es)? a\b", r"\bmayor(es)? que\b", r"\bmenor(es)? que\b",
                    r"\bel doble\b", r"\bla mitad\b", r"\btriple\b", r"\bsuper[oa]\b", r"\brecord\b",
                    r"\bel (primer|unico|mayor|menor)\b"])
# Matices que, si están en la fuente alineada y desaparecen, endurecen la afirmación
MATIZ_MODAL = _rx([r"\bpodria", r"\bpuede[n]?\b", r"\bposible", r"\bpreve\b", r"\bprevist[oa]", r"\bplanea", r"\bestudia\b", r"\bestudiara\b",
                   r"\bpropon", r"\bpropuest[oa]", r"\bpropuso\b", r"\bplantea", r"\bsi se aprueba",
                   r"\bpendiente", r"\ben caso de\b", r"\bestimad[oa]", r"\bse espera\b", r"\bsalvo\b", r"\bexcepto\b",])
MATIZ_ALCANCE = _rx([r"\bsolo\b", r"\bunicamente\b", r"\balgun[oa]s\b", r"\bvari[oa]s\b",
                     r"\bparte de\b",
                     r"\b(\d+|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|veinte) de (los|las|sus|cada)\b",
                     r"\bla mayoria\b", r"\bminoria\b", r"\bla mitad de\b",
                     r"\ben parte\b", r"\bprincipalmente\b", r"\bsobre todo\b", r"\bcon mayor\b"])
HEDGE_NUM = r"(mas de|menos de|al menos|casi|cerca de|alrededor de|aproximadamente|en torno a|unos|unas|hasta|por encima de|por debajo de|mas del|menos del|superior a|inferior a)"
RE_HEDGE_NUM = re.compile(r"\b" + HEDGE_NUM + r"\s+(" + RE_NUM.pattern + r")")
RE_HEDGE_CUALQ = re.compile(r"\b" + HEDGE_NUM + r"\s*$")


def hay(patrones, t):
    return any(p.search(t) for p in patrones)


def anadido(patrones, resp, fuentes_txt):
    """Algún patrón está en la respuesta y en ninguna fuente."""
    return float(any(p.search(resp) and not p.search(fuentes_txt) for p in patrones))


def perdido(patrones, fuente, resp):
    """Algún patrón está en la fuente alineada y no en la respuesta."""
    return float(any(p.search(fuente) and not p.search(resp) for p in patrones))


# ------------------------------------------------------------------ modelo (lo escribe lab/entrenar.py)
# >>> MODELO
# Generado por lab/entrenar.py (variante logreg:C=0.3:pen=l2:noh=0). No editar a mano.
MODELO = {"tipo": "logreg", "nombres": ["llm_conf_min", "num_derivado_local", "nli_mejor_min_frag", "frag_sin_respaldo", "modal_perdido_citada", "matiz_alcance_perdido", "num_ausente"], "media": [0.962027833001988, 0.21073558648111332, 0.5360417495029821, 0.22359178263750829, 0.08946322067594434, 0.09741550695825049, 0.0636182902584493], "escala": [0.041834940506624216, 0.40783096875001357, 0.3826953434406719, 0.3405028823069756, 0.2854111995388962, 0.2965227241921225, 0.24407171774509434], "coef": [-0.8355425750883818, -1.13732207036814, -0.36689251544101814, 0.12784608545061907, 0.48693533998227484, 0.3407881339945539, 0.2953816074128715], "sesgo": 0.4595743101400806, "umbral": 0.5, "variante": "logreg:C=0.3:pen=l2:noh=0"}
# <<< MODELO

NEUTRO = {"llm_conf_min": 0.9649, "llm_null": 0.0048, "llm_sin_fuente": 0.0087, "nli_citada_min": 0.3919, "nli_todas_min_orac": 0.5481, "nli_todas_min_frag": 0.4957, "nli_mejor_min_frag": 0.5577, "nli_comb_min_frag": 0.6819, "nli_contra_max": 0.5811, "sim_min_frag": 0.6011, "frag_sin_respaldo": 0.1734, "num_cruzada": 0.1639, "num_cruzada_llm": 0.3433, "num_ausente": 0.055, "num_no_derivable_local": 0.1851, "num_derivado_local": 0.136, "num_prop_fuera": 0.1463, "matiz_num_perdido": 0.0251, "matiz_modal_perdido": 0.0906, "matiz_alcance_perdido": 0.0636, "alcance_perdido_citada": 0.0482, "modal_perdido_citada": 0.0521, "universal_anadido": 0.0231, "causal_anadido": 0.0251, "comparativa_anadida": 0.0897, "prop_raices_nuevas": 0.2181, "nombre_nuevo": 0.0039}  # medias de la zona gris (train)


def mn(xs, defecto):
    xs = list(xs)
    return min(xs) if xs else defecto


def mx(xs, defecto):
    xs = list(xs)
    return max(xs) if xs else defecto


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class Guardian:
    def __init__(self, modelo=None):
        self.modelo = modelo or MODELO
        self.excepciones = 0

    # ---------------------------------------------------------- features
    def extraer(self, caso):
        sen = caso.get("senales") or {}
        ors = [o for o in (sen.get("oraciones") or []) if isinstance(o, dict)]
        frs = [f for f in (sen.get("fragmentos") or []) if isinstance(f, dict)]
        fuentes = [str(x) for x in (caso.get("fuentes") or [])]
        resp = caso.get("respuesta") or " ".join(o.get("texto", "") for o in ors)
        f = {}

        def calc(nombre, fn, neutro=0.0):
            try:
                f[nombre] = float(fn())
            except Exception:
                self.excepciones += 1
                f[nombre] = NEUTRO.get(nombre, neutro)

        llms = [o.get("llm") or {} for o in ors]
        ver = [l.get("veredicto") for l in llms]
        conf = [l.get("confianza") if isinstance(l.get("confianza"), (int, float)) else 0.5 for l in llms]
        neg = [c for v, c in zip(ver, conf) if v in ("no_respaldada", "contradicha")]
        f["_capa"] = 1.0 if neg else 2.0
        f["_conf_neg"] = max(neg) if neg else 0.0
        f["_null"] = float(not ors or any(v is None for v in ver))

        fn_n = [norm(x) for x in fuentes]
        r_n = norm(resp)
        todas_n = " ".join(fn_n) + " " + norm(caso.get("pregunta") or "")
        nf = len(fuentes)

        def lista(item, campo):
            v = item.get(campo) or []
            return [float(x) if isinstance(x, (int, float)) else 0.0 for x in v]

        def alineada(item):
            r, s = lista(item, "respaldo_por_fuente"), lista(item, "similitud_por_fuente")
            m = max(len(r), len(s), 0)
            if not m:
                return None
            r += [0.0] * (m - len(r))
            s += [0.0] * (m - len(s))
            return max(range(m), key=lambda j: r[j] + s[j])

        def citada(o):
            j = (o.get("llm") or {}).get("fuente")
            return j if isinstance(j, int) and 0 <= j < nf else alineada(o)

        # --- juez LLM
        calc("llm_conf_min", lambda: min(conf) if conf else 0.0, 0.5)
        calc("llm_null", lambda: f["_null"])
        calc("llm_sin_fuente", lambda: any((l.get("fuente") is None) for l in llms) if llms else 1.0)

        # --- NLI, peor caso
        def nli_citada():
            vals = []
            for o in ors:
                j, r = citada(o), lista(o, "respaldo_por_fuente")
                vals.append(r[j] if j is not None and j < len(r) else 0.0)
            return mn(vals, 0.5)
        calc("nli_citada_min", nli_citada, 0.5)
        items = frs or ors
        calc("nli_todas_min_orac", lambda: mn((float(o.get("respaldo_todas") or 0) for o in ors), 0.5), 0.5)
        calc("nli_todas_min_frag", lambda: mn((float(x.get("respaldo_todas") or 0) for x in items), 0.5), 0.5)
        calc("nli_mejor_min_frag", lambda: mn((max(lista(x, "respaldo_por_fuente") or [0]) for x in items), 0.5), 0.5)
        def apoyo(x):  # mejor respaldo de un trozo: todas las fuentes juntas o la mejor sola
            return max([float(x.get("respaldo_todas") or 0)] + lista(x, "respaldo_por_fuente"))
        calc("nli_comb_min_frag", lambda: mn((apoyo(x) for x in items), 0.5), 0.5)
        calc("nli_contra_max", lambda: mx((max([float(x.get("contradiccion_todas") or 0)] +
                                               lista(x, "contradiccion_por_fuente")) for x in items), 0.3), 0.3)
        calc("sim_min_frag", lambda: mn((max(lista(x, "similitud_por_fuente") or [0]) for x in items), 0.5), 0.5)
        calc("frag_sin_respaldo", lambda: sum(apoyo(x) < 0.5 for x in items) / len(items) if items else 0.3, 0.3)

        # --- cifras y alineación fragmento -> fuente
        nums_f = [cifras(x) for x in fuentes]
        todas_nums = [v for ns in nums_f for v in ns]
        deriv = derivables(todas_nums)
        nums_r = cifras(resp)

        def cruzada():
            for x in items:
                j = alineada(x)
                if j is None or j >= nf:
                    continue
                for v in cifras(x.get("texto") or ""):
                    if not esta(v, nums_f[j]) and esta(v, todas_nums):
                        return 1.0
            return 0.0

        def cruzada_llm():
            for o in ors:
                j = citada(o)
                if j is None or j >= nf:
                    continue
                for v in cifras(o.get("texto") or ""):
                    if not esta(v, nums_f[j]) and esta(v, todas_nums):
                        return 1.0
            return 0.0

        calc("num_cruzada", cruzada)
        calc("num_cruzada_llm", cruzada_llm)
        calc("num_ausente", lambda: any(not esta(v, todas_nums) and not sale_de(v, deriv) for v in nums_r))

        # Versión LOCAL: cuentas solo con las cifras (y fechas) de las fuentes alineadas
        # o citadas; con todas las fuentes casi cualquier entero pequeño "sale" de alguna cuenta.
        def locales():
            js = {alineada(x) for x in items} | {citada(o) for o in ors}
            js = [j for j in js if j is not None and j < nf]
            loc = [v for j in js for v in nums_f[j]]
            ms = [v for j in js for v in meses_abs(fuentes[j])]
            dloc = derivables(loc) | {float(abs(a - b)) for a in ms for b in ms if a != b}
            return loc, dloc
        try:
            loc, dloc = locales()
        except Exception:
            self.excepciones += 1
            loc, dloc = todas_nums, deriv
        # Respaldo por cuentas: peor caso = solo cuenta si TODAS las oraciones lo tienen
        # (si no, una oración fiel con una cuenta bien hecha taparía otra alucinada).
        textos_or = [o.get("texto") or "" for o in ors] or [resp]

        def deriva_en(texto, pool):
            # la oración tiene cifras que no están literales y TODAS salen de cuentas
            nuevas = [v for v in cifras(texto) if not esta(v, todas_nums)]
            return bool(nuevas) and all(sale_de(v, pool) for v in nuevas)
        calc("num_no_derivable_local", lambda: any(not esta(v, loc) and not sale_de(v, dloc) for v in nums_r))
        calc("num_derivado_local", lambda: all(deriva_en(t, dloc) for t in textos_or))
        calc("num_derivado", lambda: all(deriva_en(t, deriv) for t in textos_or))
        calc("num_prop_fuera", lambda: (sum(not esta(v, todas_nums) for v in nums_r) / len(nums_r))
             if nums_r else 0.0)

        def matiz_num():
            # "más de 150.000" en la fuente -> "150.000" a secas en la respuesta
            for fx in fn_n:
                for m in RE_HEDGE_NUM.finditer(fx):
                    v = _valor(m.group(2))
                    for mr in RE_NUM.finditer(r_n):
                        try:
                            if cerca(_valor(mr.group()), v) and not RE_HEDGE_CUALQ.search(r_n[max(0, mr.start() - 25):mr.start()]):
                                return 1.0
                        except ValueError:
                            pass
            return 0.0
        calc("matiz_num_perdido", matiz_num)

        def oracion_alineada(fuente_n):
            # En fuentes largas (textos legales) solo cuenta la oración que más se parece a
            # la respuesta: si no, cualquier "salvo"/"puede" de otra cláusula saltaría.
            frases = [f for f in re.split(r"(?<=[.;:])\s+", fuente_n) if f.strip()] or [fuente_n]
            rr = raices(r_n)
            return max(frases, key=lambda f: len(raices(f) & rr))

        def fuentes_alineadas():
            js = {alineada(x) for x in items} | {citada(o) for o in ors}
            return " ".join(oracion_alineada(fn_n[j]) for j in js if j is not None and j < nf)
        try:
            al = fuentes_alineadas()
        except Exception:
            self.excepciones += 1
            al = ""
        calc("matiz_modal_perdido", lambda: perdido(MATIZ_MODAL, al, r_n))
        calc("matiz_alcance_perdido", lambda: perdido(MATIZ_ALCANCE, al, r_n))

        def cit_txt():
            js = {citada(o) for o in ors}
            return " ".join(oracion_alineada(fn_n[j]) for j in js if j is not None and j < nf)
        calc("alcance_perdido_citada", lambda: perdido(MATIZ_ALCANCE, cit_txt(), r_n))
        calc("modal_perdido_citada", lambda: perdido(MATIZ_MODAL, cit_txt(), r_n))

        # --- añadidos que ninguna fuente dice
        calc("universal_anadido", lambda: anadido(UNIVERSALES, r_n, todas_n))
        calc("causal_anadido", lambda: anadido(CAUSALES, r_n, todas_n))
        calc("comparativa_anadida", lambda: anadido(COMPARATIVAS, r_n, todas_n))
        calc("prop_raices_nuevas", lambda: (lambda rr: len(rr - raices(todas_n)) / len(rr) if rr else 0.0)(raices(resp)))

        def nombre_nuevo():
            pal = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})", resp.strip())
            return any(norm(p) not in todas_n for p in pal)
        calc("nombre_nuevo", nombre_nuevo)
        return f

    # ---------------------------------------------------------- decisión
    def puntuar_feats(self, f):
        if f.get("_capa") == 1.0:
            return 0.85 + 0.10 * min(1.0, max(0.0, f.get("_conf_neg", 0.0)))
        m = self.modelo
        if not m:
            return 0.5
        if m.get("tipo") == "hgb":  # solo en la validación del lab (objeto sklearn en memoria)
            p = float(m["obj"].predict_proba([[f.get(n, 0.0) for n in m["nombres"]]])[0, 1])
        else:
            z = m["sesgo"] + sum(c * (f.get(n, NEUTRO.get(n, 0.0)) - mu) / sd
                                 for n, mu, sd, c in zip(m["nombres"], m["media"], m["escala"], m["coef"]))
            p = _sig(z)
        t = m.get("umbral", 0.5)  # reescalado lineal a trozos: corte de validación -> 0,5
        p = 0.5 * p / t if p < t else 0.5 + 0.5 * (p - t) / (1 - t)
        if f.get("_null"):
            p = max(p, 0.5)  # el LLM no contestó: sospechoso (5/5 alucinan en el train)
        return min(1.0, max(0.0, p))

    def score(self, caso):
        try:
            return self.puntuar_feats(self.extraer(caso if isinstance(caso, dict) else {}))
        except Exception:
            self.excepciones += 1
            return 0.9  # si todo falla, alarma: un escape cuesta x4
