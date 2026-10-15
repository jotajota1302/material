"""El Guardián — Detector Híbrido GBDT + LogReg con Calibración Adaptativa PD."""
import collections
import math
import re
import struct
import unicodedata
from itertools import combinations


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFD", str(texto or "").lower())
    return "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")


STOPWORDS_ES = set("""
a al algo ante antes asi aun bajo cada como con contra cual cuando de del desde
donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso
esos esta estaba estan estas este esto estos fue fueron ha han hasta hay la las
le les lo los mas me mientras muy ni no nos o otra otras otro otros para pero
por que quien se sea segun ser si sido sin sobre son su sus tambien tan tanto
te tiene tienen todo tras tu un una uno unos unas y ya yo ano anos
""".split())


def tokenizar_palabras(texto):
    return re.findall(r"[a-zñ]+", normalizar_texto(texto))


def stem_es(palabra):
    return palabra[:5] if len(palabra) > 5 else palabra


def obtener_stems(texto):
    return {stem_es(p) for p in tokenizar_palabras(texto) if p not in STOPWORDS_ES and len(p) > 2}


def extraer_entidades_propias(texto):
    entidades = set()
    for orac in re.split(r"(?<=[.!?;:])\s+", str(texto or "")):
        toks = re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", orac)
        for t in toks[1:]:
            if t[0].isupper() and len(t) > 2:
                entidades.add(normalizar_texto(t))
    return entidades


MAPA_NUMEROS = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100
}

RE_PATRON_HORA = re.compile(r"\b\d{1,2}:\d{2}\b")
RE_PATRON_DIGITO = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
RE_PATRON_PAL_NUM = re.compile(
    r"\b(" + "|".join(sorted(MAPA_NUMEROS, key=len, reverse=True)) +
    r")(?:\s+y\s+(uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))?\b"
)

RE_MARCADOR_OTRA_ENT = re.compile(
    r"\b(vecin[oa]s?|competidor[ae]?s?|rival(es)?|por su parte|en cambio|"
    r"mientras que|a diferencia|otr[oa]s? (empresa|entidad|ciudad|municipio|"
    r"comarca|asociacion|ong|fundacion|cooperativa|banco|ayuntamiento|"
    r"hospital|universidad|operadora|region|provincia|centro|club))\b"
)


def a_float(cad):
    return float(cad.replace(".", "").replace(",", "."))


def es_aproximado(a, b, tol_rel=0.01):
    return abs(a - b) <= max(tol_rel * abs(b), 0.051 if not float(a).is_integer() else 0.5)


def analizar_cifras_texto(texto):
    t = normalizar_texto(texto)
    t = RE_PATRON_HORA.sub(lambda m: " " * len(m.group()), t)
    hallazgos = []
    for m in RE_PATRON_DIGITO.finditer(t):
        hallazgos.append((m.start(), m.end(), a_float(m.group())))
    for m in RE_PATRON_PAL_NUM.finditer(t):
        v = MAPA_NUMEROS[m.group(1)]
        if m.group(2):
            v += MAPA_NUMEROS.get(m.group(2), 0) if m.group(2) != "uno" else 1
        hallazgos.append((m.start(), m.end(), float(v)))
    hallazgos.sort()

    lista_nums = []
    consumidos = set()
    for idx, (ini, fin, val) in enumerate(hallazgos):
        if idx in consumidos:
            continue
        cola = t[fin:fin + 26]
        m_prop = re.match(r"\s+de\s+cada\s+", cola)
        if m_prop and idx + 1 < len(hallazgos) and hallazgos[idx + 1][0] == fin + m_prop.end():
            base_prop = hallazgos[idx + 1][2]
            consumidos.add(idx + 1)
            if base_prop:
                lista_nums.append({
                    "valores": {100.0 * val / base_prop}, "pct": True,
                    "anio": False, "aprox": True, "ini": ini, "fin": hallazgos[idx + 1][1]
                })
            continue
        posibles = {val}
        m_mill = re.match(r"\s*(mil\s+)?millones", cola)
        if m_mill:
            posibles.add(val * (1e9 if m_mill.group(1) else 1e6))
        elif re.match(r"\s*mil\b", cola):
            posibles.add(val * 1e3)
        if re.match(r"\s*centimos", cola):
            posibles.add(val / 100.0)
        if re.match(r"\s*(km|kilometros|toneladas)\b", cola):
            posibles.add(val * 1000.0)
        elif re.match(r"\s*(metros|gramos|litros)\b", cola) and val >= 100:
            posibles.add(val / 1000.0)

        es_pct = bool(re.match(r"\s*(%|por\s*ciento)", cola))
        es_anio = val.is_integer() and 1900 <= val <= 2100 and not es_pct
        lista_nums.append({
            "valores": posibles, "pct": es_pct, "anio": es_anio,
            "aprox": False, "ini": ini, "fin": fin
        })
    return lista_nums


RE_OP_SUMA = re.compile(
    r"\b(suman|sumaron|suma|sumo|sumaba|sumando|sumaban|sumarian|en total|"
    r"entre (los|las) (dos|tres|cuatro)|entre ambos|entre ambas|juntos|juntas|en conjunto)\b"
)
RE_OP_DIF_POST = re.compile(r"^\s*((%\s*|por ciento\s*|puntos?\s*|[a-z]+\s+)?(mas|menos)\b|puntos?\b)")
RE_OP_DIF_PRE = re.compile(
    r"(diferencia|aumento|subida|incremento|rebaja|descenso|caida|ganancia|perdida|"
    r"ahorro|crecio|subio|bajo|aumento|cayo|mejoro|empeoro|gano|perdio|redujo|anadio)\s+(en\s+|de\s+)?(unos?\s+|el\s+|un\s+)?$"
)
RE_OP_CALC_PRE = re.compile(
    r"(es decir|lo que supone|lo que significa|lo que equivale|equivale a|"
    r"o sea|lo que representa)[\s,]*(a\s+)?(unos?\s+|unas\s+|el\s+|la\s+|un\s+|una\s+)?$"
)


def detectar_intencion_aritmetica(texto_norm, info_num):
    pre = texto_norm[max(0, info_num["ini"] - 40):info_num["ini"]]
    post = texto_norm[info_num["fin"]:info_num["fin"] + 30]
    if RE_OP_CALC_PRE.search(pre) or re.match(r"\s*veces\b", post):
        return "calculo"
    if RE_OP_DIF_POST.match(post) or RE_OP_DIF_PRE.search(pre):
        return "diferencia"
    if RE_OP_SUMA.search(pre[-25:] + " " + post[:25]):
        return "suma"
    return None


def validar_operacion_anunciada(tipo_op, info_num, anclas, operandos):
    pool = sorted(set(anclas) | set(sorted(operandos)[:60]))
    generados = set()
    if tipo_op == "suma":
        for a, b in combinations(pool, 2):
            generados.update({a + b, a * b})
        for a, b, c in combinations(pool[:30], 3):
            generados.add(a + b + c)
    else:
        pivotes = anclas if anclas else pool
        for a in pivotes:
            for b in pool:
                if a == b:
                    continue
                generados.add(abs(a - b))
                if tipo_op == "calculo":
                    generados.update({a * b / 100.0, a * b})
                    if b:
                        generados.update({100.0 * a / b, a / b})
                    if a:
                        generados.update({100.0 * b / a, b / a})
    return any(es_aproximado(v, c) for v in info_num["valores"] for c in generados)


MESES_NOMBRES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
RE_DIA_DE_MES = re.compile(r"\s+de\s+(" + MESES_NOMBRES + r")\b")
RE_NUM_ORDINAL = re.compile(r"(n\.?\s*[o\u00ba\u00b0]\.?|numero)\s*$")
RE_VERBO_VAR = re.compile(
    r"\b(aumento|aumentaron|crecio|crecieron|subio|subieron|bajo|bajaron|cayo|"
    r"cayeron|redujo|mejoro|empeoro|gano|perdio|un|una)\s*([a-z]+\s+)?$"
)
RE_PREP_DESDE = re.compile(r"\b(de|del|desde)\s*$")
RE_PREP_HASTA = re.compile(r"\b(a|al|hasta)\s+(los\s+|las\s+|el\s+|la\s+)?$")
RE_UNIDAD_TASA = re.compile(
    r"\bpor (habitante|persona|vecino|alumno|escolar|metro|kilo|litro|hora|dia|mes|"
    r"ano|unidad|plaza|cama|hectarea|trabajador|empleado|socio|cliente|vivienda|"
    r"hogar|familia|tripulante|deportista|participante|beneficiario|usuario)|"
    r"\bal (mes|ano|dia)\b|\bmensual|\banual"
)
RE_VERBO_RES = re.compile(
    r"(fue de|es de|sera de|seran|asciende a|ascendio a|supone|suponen|"
    r"pagara|pagaran|paga|pagan|cuesta|costara|costo|ingresara|ingresa|"
    r"recibira|recibe|total de|en total|es decir|lo que da|resulta|"
    r"alcanza|alcanzo|suma|suman)\s+(unos?\s+|unas\s+|el\s+|los\s+|de\s+)?$"
)
RE_COMP_MAS_MENOS = re.compile(
    r"^\s*(%\s*|por ciento\s*|puntos?\s*)?([a-z]+\s+){0,2}?(mas|menos|mayor|menor|superior|inferior)\s+(que|a|al)\s+"
)
RE_VERBO_PERDIDA = re.compile(r"\b(perdio|perdieron|pierde|bajo|bajaron|cayo|cayeron|redujo|redujeron|disminuyo|descendio|menos)\b")
RE_VERBO_GANANCIA = re.compile(r"\b(gano|ganaron|gana|subio|subieron|crecio|crecieron|aumento|aumentaron)\b")
RAICES_GENERICAS = {"total", "entre", "ambos", "ambas", "sumar", "suman"}


def _extraer_vals_sin_anio(lista_n):
    return [v for n in lista_n if not n["anio"] for v in n["valores"]]


def _cuadra_con_modulador(real, dicho, contexto_previo):
    if real is None:
        return False
    if re.search(r"(mas del?|por encima del?|superior al?)\s*(un\s+|el\s+)?$", contexto_previo):
        return dicho * 0.98 <= real <= dicho * 1.6 + 0.5
    if re.search(r"(menos del?|por debajo del?|inferior al?)\s*(un\s+|el\s+)?$", contexto_previo):
        return dicho * 0.4 - 0.5 <= real <= dicho * 1.02
    if re.search(r"(casi|cerca de|alrededor de|unos|unas|aproximadamente)\s*(un\s+|el\s+)?$", contexto_previo):
        return abs(real - dicho) <= max(0.1 * abs(dicho), 0.6)
    return abs(real - dicho) <= max(0.03 * abs(dicho), 0.6 if dicho >= 3 else 0.06)


def _buscar_fuentes_afines(frag_norm, fuentes_norm):
    stems = obtener_stems(frag_norm) - RAICES_GENERICAS
    anios = set(re.findall(r"\b(?:19|20)\d\d\b", frag_norm))
    scores = []
    for i, fn in enumerate(fuentes_norm):
        sc = len(stems & obtener_stems(fn)) + 2 * sum(1 for a in anios if a in fn)
        scores.append(sc)
    top = max(scores) if scores else 0
    return [i for i, sc in enumerate(scores) if sc == top and sc > 0]


def auditar_cuentas_intraroracion(resp_norm, nums_resp, fuentes, nums_por_fuente):
    fuentes_norm = [normalizar_texto(x) for x in fuentes]
    todas_cifras_f = _extraer_vals_sin_anio([n for ns in nums_por_fuente for n in ns])
    auditoria = {}
    cortes = [0] + [m.end() for m in re.finditer(r"[.;](\s|$)", resp_norm)] + [len(resp_norm)]

    for ini_f, fin_f in zip(cortes, cortes[1:]):
        en_orac = [k for k, n in enumerate(nums_resp) if ini_f <= n["ini"] < fin_f]
        utiles = [
            k for k in en_orac
            if not nums_resp[k]["anio"]
            and not RE_DIA_DE_MES.match(resp_norm[nums_resp[k]["fin"]:nums_resp[k]["fin"] + 15])
            and not RE_NUM_ORDINAL.search(resp_norm[max(0, nums_resp[k]["ini"] - 8):nums_resp[k]["ini"]])
        ]

        def es_dato_literal(k):
            stems_orac = obtener_stems(resp_norm[ini_f:fin_f])
            for i, ns in enumerate(nums_por_fuente):
                for n in ns:
                    if n["pct"] == nums_resp[k]["pct"] and any(
                        es_aproximado(v, w, 0.001) for v in n["valores"] for w in nums_resp[k]["valores"]
                    ):
                        if len(stems_orac & obtener_stems(fuentes_norm[i])) >= 2:
                            return True
            return False

        def registrar(k, exito, clase_op):
            if exito:
                auditoria[k] = ("ok", clase_op)
            elif not es_dato_literal(k):
                auditoria[k] = ("mal", clase_op)

        def ctx_antes(k, ventana=30):
            return resp_norm[max(ini_f, nums_resp[k]["ini"] - ventana):nums_resp[k]["ini"]]

        def ctx_despues(k, ventana=40):
            return resp_norm[nums_resp[k]["fin"]:min(fin_f, nums_resp[k]["fin"] + ventana)]

        def val_base(k):
            return min(nums_resp[k]["valores"])

        # A. Variación «de A a B»
        for a in utiles:
            if not RE_PREP_DESDE.search(ctx_antes(a, 12)):
                continue
            siguientes = [k for k in utiles if nums_resp[k]["ini"] > nums_resp[a]["fin"]]
            if not siguientes:
                continue
            b = siguientes[0]
            if not RE_PREP_HASTA.search(resp_norm[nums_resp[a]["fin"]:nums_resp[b]["ini"]]):
                continue
            va, vb = val_base(a), val_base(b)
            for c in utiles:
                if c in (a, b) or c in auditoria:
                    continue
                if nums_resp[c]["ini"] > nums_resp[b]["fin"]:
                    hueco = resp_norm[nums_resp[b]["fin"]:nums_resp[c]["ini"]]
                    adyacente = (len(hueco) <= 45 and ";" not in hueco and re.search(r",\s*([a-z]+\s+){0,6}$", hueco))
                else:
                    hueco = resp_norm[nums_resp[c]["fin"]:nums_resp[a]["ini"]]
                    adyacente = (
                        nums_resp[c]["fin"] <= nums_resp[a]["ini"] and len(hueco) <= 35
                        and ";" not in hueco and RE_VERBO_VAR.search(ctx_antes(c, 25))
                    )
                if not adyacente:
                    continue
                post_c = ctx_despues(c, 25)
                dicho = val_base(c)
                if re.match(r"\s*puntos?\b", post_c):
                    real = abs(vb - va)
                elif nums_resp[c]["pct"]:
                    reales = [abs(vb - va) / va * 100.0 if va else None]
                    if nums_resp[a]["pct"] and nums_resp[b]["pct"]:
                        reales.append(abs(vb - va))
                    registrar(c, any(_cuadra_con_modulador(r, dicho, ctx_antes(c)) for r in reales), "cambio")
                    continue
                elif re.match(r"\s*([a-z]+\s+){0,2}(mas|menos)\b", post_c):
                    real = abs(vb - va)
                elif re.match(r"\s*veces\b", post_c):
                    real = max(va, vb) / min(va, vb) if min(va, vb) else None
                else:
                    continue
                registrar(c, _cuadra_con_modulador(real, dicho, ctx_antes(c)), "cambio")

        # B. Producto con tasa («por habitante», «mensual»)
        orac_str = resp_norm[ini_f:fin_f]
        if RE_UNIDAD_TASA.search(orac_str) and len(utiles) >= 3:
            r = utiles[-1]
            if r not in auditoria and RE_VERBO_RES.search(ctx_antes(r, 35)):
                escalas = [1.0]
                if re.search(r"\bal mes\b|\bmensual", orac_str) and re.search(r"\bal ano\b|\banual", orac_str):
                    escalas += [12.0, 1.0 / 12.0]
                ops = [nums_resp[k]["valores"] for k in utiles[:-1]]
                cuadra_prod = False
                for tam in (2, 3):
                    for combo in combinations(ops, tam):
                        prods = [1.0]
                        for vals in combo:
                            prods = [p * v for p in prods for v in vals]
                        if any(
                            es_aproximado(rv, p * esc, 0.02)
                            for p in prods for esc in escalas for rv in nums_resp[r]["valores"]
                        ):
                            cuadra_prod = True
                registrar(r, cuadra_prod, "producto")

        # C. Suma de entidades nombradas
        m_suma = RE_OP_SUMA.search(orac_str)
        if m_suma:
            pos_marca = ini_f + m_suma.start()
            tras_marca = [
                k for k in utiles
                if ini_f + m_suma.end() <= nums_resp[k]["ini"] <= ini_f + m_suma.end() + 25
                and not re.search(r"entre (los|las|ambos)\s*$", ctx_antes(k, 15))
            ]
            if tras_marca and tras_marca[0] not in auditoria:
                x = tras_marca[0]
                sujeto = resp_norm[ini_f:pos_marca]
                trozos = re.split(r"\s(?:y|e)\s", sujeto)
                if len(trozos) >= 2:
                    trozos = re.split(r",", trozos[0])[-3:] + trozos[1:]
                partes = [p for p in trozos if obtener_stems(p) - RAICES_GENERICAS]
                if len(partes) >= 2:
                    cands_parte = []
                    for p in partes:
                        idxs_f = _buscar_fuentes_afines(p, fuentes_norm)
                        vals = set(_extraer_vals_sin_anio([
                            n for k2, n in enumerate(nums_resp)
                            if k2 in utiles and n["ini"] < pos_marca and p.strip() and p.strip()[:8] in resp_norm
                        ]))
                        for i_f in idxs_f:
                            for n in nums_por_fuente[i_f]:
                                if n["anio"]:
                                    continue
                                entorno = fuentes_norm[i_f][max(0, n["ini"] - 60):n["fin"] + 20]
                                if obtener_stems(p) & obtener_stems(entorno):
                                    vals.update(n["valores"])
                        cands_parte.append(sorted(vals)[:25])
                    if all(cands_parte) and len(cands_parte) <= 4:
                        sumas = [0.0]
                        for vals in cands_parte:
                            sumas = [s + v for s in sumas for v in vals][:20000]
                        ok = any(es_aproximado(xv, s, 0.01) for s in sumas for xv in nums_resp[x]["valores"])
                        registrar(x, ok, "suma")

        # D. Comparación («X más/menos que...»)
        for c in utiles:
            if c in auditoria:
                continue
            post_full = resp_norm[nums_resp[c]["fin"]:fin_f]
            m_cmp = RE_COMP_MAS_MENOS.match(post_full)
            if not m_cmp:
                continue
            target = re.split(r"[,;]|\sy\s", post_full[m_cmp.end():m_cmp.end() + 60])[0]
            anclas = [k for k in utiles if k != c and nums_resp[k]["ini"] < nums_resp[c]["ini"]]
            idxs_f = set(_buscar_fuentes_afines(target, fuentes_norm))
            idxs_f |= {
                i for i, ns in enumerate(nums_por_fuente) for k in anclas
                if any(es_aproximado(v, w, 0.001) for n in ns for v in n["valores"] for w in nums_resp[k]["valores"])
            }
            if not anclas or not idxs_f:
                continue
            otras = _extraer_vals_sin_anio([n for i in idxs_f for n in nums_por_fuente[i]])
            fin_target = nums_resp[c]["fin"] + m_cmp.end() + len(target)
            otras += _extraer_vals_sin_anio([
                nums_resp[k] for k in utiles if nums_resp[c]["fin"] < nums_resp[k]["ini"] < fin_target
            ])
            dicho = val_base(c)
            pares = [(av, o) for k in anclas for av in nums_resp[k]["valores"] for o in otras if o]
            if re.match(r"\s*veces\b", post_full):
                ok = any(_cuadra_con_modulador(max(av, o) / min(av, o), dicho, ctx_antes(c)) for av, o in pares if min(av, o))
            elif nums_resp[c]["pct"] and not any(nums_resp[k]["pct"] for k in anclas):
                ok = any(_cuadra_con_modulador(abs(av - o) / o * 100.0, dicho, ctx_antes(c)) for av, o in pares)
            else:
                ok = any(_cuadra_con_modulador(abs(av - o), dicho, ctx_antes(c)) for av, o in pares)
            registrar(c, ok, "comparacion")

        # E. Parte de un total («perdió A, un X % de...»)
        for c in utiles:
            if c in auditoria or not nums_resp[c]["pct"]:
                continue
            prevs = [k for k in utiles if nums_resp[k]["fin"] < nums_resp[c]["ini"] and not nums_resp[k]["pct"]]
            if not prevs:
                continue
            x = prevs[-1]
            puente = resp_norm[nums_resp[x]["fin"]:nums_resp[c]["ini"]]
            if not (
                re.search(r",\s*(lo que supone\s+|es decir,?\s+)?(un|el)\s*$", puente)
                and re.match(r"\s*(%|por ciento)\s+(de|del)\b", resp_norm[nums_resp[c]["fin"]:fin_f])
            ):
                continue
            vx = val_base(x) if val_base(x) > 1000 or not any(v > 1000 for v in nums_resp[x]["valores"]) else max(nums_resp[x]["valores"])
            if RE_VERBO_PERDIDA.search(orac_str):
                bases = [b for b in todas_cifras_f if any(es_aproximado(b - vx, o) for o in todas_cifras_f)]
            elif RE_VERBO_GANANCIA.search(orac_str):
                bases = [b for b in todas_cifras_f if any(es_aproximado(b + vx, o) for o in todas_cifras_f)]
            else:
                bases = [b for b in todas_cifras_f if b > vx]
            if not bases:
                continue
            dicho = val_base(c)
            registrar(c, any(_cuadra_con_modulador(100.0 * vx / b, dicho, ctx_antes(c)) for b in bases if b), "parte")

    return auditoria


def generar_cierre_derivados(valores_f):
    vals = sorted(valores_f)[:60]
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


def categorizar_cifra(num, valores_f, derivados):
    tol = 0.12 if num["aprox"] else 0.01
    for v in num["valores"]:
        if any(es_aproximado(v, f, tol) for f in valores_f):
            return "directo"
    for v in num["valores"]:
        if any(es_aproximado(v, d, tol) for d in derivados):
            return "derivado"
    return "inexplicado"


TABLA_PROPORCIONES = [
    (r"\bdoble\b|\bduplica", 2.0), (r"\btriple\b|\btriplica", 3.0),
    (r"\bcuadruplica|\bcuadruple\b", 4.0), (r"\bquintuplica", 5.0),
    (r"\bla mitad\b", 0.5), (r"\bun tercio\b|\btercera parte\b", 1.0 / 3.0),
    (r"\bdos tercios\b", 2.0 / 3.0), (r"\bun cuarto\b|\bcuarta parte\b", 0.25),
    (r"\btres cuartos\b", 0.75),
]


def detectar_proporciones_erroneas(texto_resp, nums_resp, valores_f):
    t = normalizar_texto(texto_resp)
    afirmaciones = []

    def tiene_cota_inf(pos_ini):
        return bool(re.search(r"(mas|algo mas) del?\s*(un|una|la|el)?\s*$", t[max(0, pos_ini - 14):pos_ini]))

    for pat, factor in TABLA_PROPORCIONES:
        m = re.search(pat, t)
        if m:
            afirmaciones.append((factor, tiene_cota_inf(m.start())))
    for m in re.finditer(r"(\d+(?:,\d+)?|" + "|".join(MAPA_NUMEROS) + r")\s+veces\b", t):
        pal = m.group(1)
        factor = MAPA_NUMEROS.get(pal) or a_float(pal)
        afirmaciones.append((float(factor), False))
    if not afirmaciones:
        return 0
    vals_r = {v for n in nums_resp for v in n["valores"] if v}
    universo = {v for v in vals_r | set(valores_f) if v}
    erroneas = 0
    for factor, es_min in afirmaciones:
        cuadra = False
        for a in (vals_r or universo):
            for b in universo:
                if a == b:
                    continue
                for ratio in (a / b, b / a):
                    if es_min and factor <= ratio <= factor * 1.6:
                        cuadra = True
                    elif abs(ratio - factor) <= 0.12 * factor:
                        cuadra = True
            if cuadra:
                break
        if not cuadra:
            erroneas += 1
    return erroneas


MAPA_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}
RE_EXPRESION_FECHA = re.compile(r"(?:(\d{1,2}) de )?(" + "|".join(MAPA_MESES) + r")(?: de(?:l)? (\d{4}))?")
RE_EXPRESION_PLAZO = re.compile(
    r"\b(\d+|" + "|".join(MAPA_NUMEROS) + r"|un|una)\s+(dias|semanas|meses|anos)\s+(despues|antes|mas tarde|mas)\b"
)


def _extraer_tuplas_fecha(texto_norm):
    fechas = []
    for m in RE_EXPRESION_FECHA.finditer(texto_norm):
        d = int(m.group(1)) if m.group(1) else None
        a = int(m.group(3)) if m.group(3) else None
        fechas.append((a, MAPA_MESES[m.group(2)], d))
    return fechas


def verificar_plazos_temporales(respuesta, texto_fuentes):
    r = normalizar_texto(respuesta)
    s = normalizar_texto(texto_fuentes)
    falsos = correctos = 0
    for m in RE_EXPRESION_PLAZO.finditer(r):
        pal, unidad = m.group(1), m.group(2)
        n = 1 if pal in ("un", "una") else (MAPA_NUMEROS.get(pal) or int(pal))
        f_f, f_r = _extraer_tuplas_fecha(s), _extraer_tuplas_fecha(r)
        a_f = {int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", s)}
        a_r = {int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", r)}
        difs = set()
        if unidad == "anos":
            orig = a_r or a_f
            difs = {abs(a - b) for a in orig for b in a_f | a_r}
        elif unidad == "meses":
            con_anio = [a for a, _, _ in f_r + f_f if a]
            anio_def = con_anio[0] if con_anio else 2000
            m_f = {(a or anio_def) * 12 + mes for a, mes, _ in f_f}
            m_r = {(a or anio_def) * 12 + mes for a, mes, _ in f_r}
            orig = m_r or m_f
            difs = {abs(x - y) for x in orig for y in m_f | m_r}
        else:
            continue
        if not difs - {0}:
            continue
        if n in difs:
            correctos += 1
        else:
            falsos += 1
    return falsos, correctos


DICC_MATICES = {
    "restrictivo": [
        r"\bsolo\b", r"\bunicamente\b", r"\bexclusivamente\b", r"\bsalvo\b",
        r"\bexcepto\b", r"\bsiempre que\b", r"\ba condicion\b", r"\bse limita", r"\bsolamente\b"
    ],
    "condicional": [
        r"\bsi se\b", r"\bsi lo\b", r"\bsi la\b", r"\bsi el\b", r"\bsi los\b",
        r"\bsi las\b", r"\bpreve\b", r"\bprevista?\b", r"\bprevisto\b",
        r"\bestudia\b", r"\bplantea\b", r"\bpropone\b", r"\bpropuesta\b",
        r"\bpodria", r"\bposible", r"\bpendiente", r"\bprovisional",
        r"\bestima\b", r"\bestimo\b", r"\bespera\b", r"\bpretende\b",
        r"\bproyecto\b", r"\bplanea\b", r"\bdepend", r"\bcondicionad",
        r"\bsujet[oa]s? a\b", r"\ba expensas\b", r"\bse confirm"
    ],
    "aproximado": [
        r"\bmas de\b", r"\bmenos de\b", r"\bal menos\b", r"\bcasi\b",
        r"\bcerca de\b", r"\bunos\b", r"\bunas\b", r"\balrededor de\b",
        r"\baproximadamente\b", r"\bhasta\b", r"\bcomo maximo\b", r"\bcomo minimo\b", r"\bmaximo\b"
    ],
    "parcial": [
        r"\balgunos\b", r"\balgunas\b", r"\bvarios\b", r"\bvarias\b",
        r"\bparte de\b", r"\bmayoria\b", r"\bde los \d", r"\bde las \d",
        r"\bentre \d", r"\bde ell[oa]s\b"
    ],
}

PATRONES_UNIVERSALES = [
    r"\btodos\b", r"\btodas\b", r"\bcualquier", r"\bsiempre\b", r"\bnunca\b",
    r"\bningun", r"\bnadie\b", r"\btotalidad\b", r"\bintegramente\b",
    r"\bcompletamente\b", r"\bdel pais\b", r"\bde espana\b", r"\bsin excepcion"
]

PATRONES_CAUSALES = [
    r"\bporque\b", r"\bya que\b", r"\bpuesto que\b", r"\bdebido a\b",
    r"\bgracias a\b", r"\bpor su\b", r"\bpor sus\b", r"\blo que permit",
    r"\bpermitio\b", r"\bpermitira\b", r"\bpor lo que\b", r"\bde ahi\b",
    r"\ba raiz de\b", r"\bprovoc", r"\bcausad", r"\bcomo consecuencia\b",
    r"\bcomo resultado\b", r"\bfruto de\b", r"\bpor ello\b", r"\bpor eso\b",
    r"\basi que\b", r"\bcon lo que\b", r"\bpara que\b",
    r"\bpara (ampliar|reducir|mejorar|evitar|financiar|cubrir)\b"
]

PARES_VERBOS_ENDURECIDOS = [
    (r"\bvacun", r"\binmuniz"),
    (r"\bpropuso|\bpropone|\bpropuesta", r"\baprobo|\baprueba"),
    (r"\bpidio|\bsolicit", r"\bexigio|\bobligo"),
    (r"\bintent|\btrat", r"\blogr|\bconsigui"),
    (r"\binvestig|\bestudi", r"\bdemostr|\bconfirm"),
]


def contar_perdidos(patrones, fuente_norm, resp_norm):
    return sum(1 for p in patrones if re.search(p, fuente_norm) and not re.search(p, resp_norm))


def contar_anadidos(patrones, fuentes_norm, resp_norm):
    return sum(1 for p in patrones if re.search(p, resp_norm) and not re.search(p, fuentes_norm))


def analizar_tramo_cubierto(fuente, stems_resp):
    tokens = re.findall(r"[a-zñ0-9.,%]+", normalizar_texto(fuente))
    pos = [i for i, tok in enumerate(tokens) if (tok.isalpha() and tok not in STOPWORDS_ES and stem_es(tok) in stems_resp)]
    if len(pos) < 2:
        return "", 0
    tramo = tokens[min(pos):max(pos) + 1]
    huecos = sum(1 for tok in tramo if tok.isalpha() and tok not in STOPWORDS_ES and len(tok) > 2 and stem_es(tok) not in stems_resp)
    return " ".join(tramo), huecos


def extraer_mapa_caracteristicas(caso):
    fuentes = [str(x) for x in (caso.get("fuentes") or []) if x is not None]
    respuesta = str(caso.get("respuesta") or "")
    pregunta = str(caso.get("pregunta") or "")
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    fragmentos = senales.get("fragmentos") or []
    n_fuentes = len(fuentes)
    f = {}

    veredictos = [((o.get("llm") or {}).get("veredicto")) for o in oraciones]
    confianzas = [float((o.get("llm") or {}).get("confianza") or 0.0) for o in oraciones]
    f["llm_no_respaldada"] = float(any(v == "no_respaldada" for v in veredictos))
    f["llm_contradicha"] = float(any(v == "contradicha" for v in veredictos))
    f["llm_sin_veredicto"] = float(any(v is None for v in veredictos))
    conf_resp = [c for v, c in zip(veredictos, confianzas) if v == "respaldada"]
    f["llm_conf_min_respaldada"] = min(conf_resp) if conf_resp else 0.0
    f["llm_conf_media_respaldada"] = sum(conf_resp) / len(conf_resp) if conf_resp else 0.0
    f["llm_frac_alarmas"] = sum(1.0 for v in veredictos if v in ("no_respaldada", "contradicha")) / max(1, len(veredictos))

    def min_campo(lista, clave):
        vals = [float(x.get(clave, 0.0) or 0.0) for x in lista]
        return min(vals) if vals else 0.0

    def max_campo(lista, clave):
        vals = [float(x.get(clave, 0.0) or 0.0) for x in lista]
        return max(vals) if vals else 0.0

    f["nli_resp_todas_min_orac"] = min_campo(oraciones, "respaldo_todas")
    f["nli_resp_todas_min_frag"] = min_campo(fragmentos, "respaldo_todas") if fragmentos else f["nli_resp_todas_min_orac"]
    f["nli_contr_todas_max_orac"] = max_campo(oraciones, "contradiccion_todas")
    f["nli_contr_todas_max_frag"] = max_campo(fragmentos, "contradiccion_todas")
    f["nli_mejor_fuente_min_orac"] = min([max([float(v or 0.0) for v in (o.get("respaldo_por_fuente") or [0.0])]) for o in oraciones] or [0.0])
    f["nli_contr_fuente_max_frag"] = max([max([float(v or 0.0) for v in (x.get("contradiccion_por_fuente") or [0.0])]) for x in fragmentos] or [0.0])
    f["nli_contr_fuente_max_orac"] = max([max([float(v or 0.0) for v in (o.get("contradiccion_por_fuente") or [0.0])]) for o in oraciones] or [0.0])

    citadas = set()
    resp_citada = []
    contr_citada = []
    for o in oraciones:
        llm = o.get("llm") or {}
        i = llm.get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            citadas.add(i)
            rpf = o.get("respaldo_por_fuente") or []
            cpf = o.get("contradiccion_por_fuente") or []
            if i < len(rpf) and rpf[i] is not None:
                resp_citada.append(float(rpf[i]))
            if i < len(cpf) and cpf[i] is not None:
                contr_citada.append(float(cpf[i]))
        sim = o.get("similitud_por_fuente") or []
        if sim:
            citadas.add(max(range(len(sim)), key=lambda k: float(sim[k] or 0.0)))
    f["nli_resp_fuente_citada_min"] = min(resp_citada) if resp_citada else 0.0
    f["nli_contr_fuente_citada_max"] = max(contr_citada) if contr_citada else 0.0

    resp_frag = sorted(
        max([float(x.get("respaldo_todas") or 0.0)] + [float(v or 0.0) for v in (x.get("respaldo_por_fuente") or [])])
        for x in fragmentos
    ) or [0.0]
    f["frag_respaldo_min"] = resp_frag[0]
    f["frag_respaldo_2o"] = resp_frag[1] if len(resp_frag) > 1 else resp_frag[0]
    f["frag_respaldo_media"] = sum(resp_frag) / len(resp_frag)
    f["frag_frac_bajo_05"] = sum(1.0 for v in resp_frag if v < 0.5) / len(resp_frag)
    f["frag_frac_bajo_02"] = sum(1.0 for v in resp_frag if v < 0.2) / len(resp_frag)
    f["frag_similitud_min"] = min([max([float(v or 0.0) for v in (x.get("similitud_por_fuente") or [0.0])]) for x in fragmentos] or [0.0])

    mejor_orac_comb = []
    for o in oraciones:
        rt = float(o.get("respaldo_todas") or 0.0)
        rpf = sorted([float(v or 0.0) for v in (o.get("respaldo_por_fuente") or [0.0])], reverse=True)
        r_top1 = rpf[0] if rpf else 0.0
        mejor_orac_comb.append(max(rt, r_top1))
    f["orac_mejor_resp_min"] = min(mejor_orac_comb) if mejor_orac_comb else 0.0
    f["brecha_orac_frag"] = max(0.0, f["orac_mejor_resp_min"] - f["frag_respaldo_min"])

    texto_fuentes = " ".join(fuentes) + " " + pregunta
    nums_fuente = analizar_cifras_texto(texto_fuentes)
    valores_fuente = {v for n in nums_fuente for v in n["valores"]}
    nums_resp = analizar_cifras_texto(respuesta)
    nums_por_fuente = [analizar_cifras_texto(x) for x in fuentes]

    def aparece_en(n, lista_n):
        return any(es_aproximado(v, w) for v in n["valores"] for m in lista_n for w in m["valores"])

    directos = [n for n in nums_resp if aparece_en(n, nums_fuente)]
    fuentes_base = set(citadas) | {i for i, ns in enumerate(nums_por_fuente) if any(aparece_en(d, ns) for d in directos)}

    def extraer_operandos(lista_n):
        ops = set()
        for n in lista_n:
            if n["anio"]:
                continue
            ops.update(n["valores"])
            if n["pct"]:
                ops.update(100.0 - v for v in n["valores"] if v < 100)
        return ops

    ops_locales = extraer_operandos(directos) | {100.0}
    for i in fuentes_base:
        if 0 <= i < len(nums_por_fuente):
            ops_locales |= extraer_operandos(nums_por_fuente[i])
    ops_globales = extraer_operandos(nums_fuente)

    nombres_fuentes = [extraer_entidades_propias(x) for x in fuentes]
    nombres_conocidos = set().union(*nombres_fuentes) if nombres_fuentes else set()
    nombres_resp = extraer_entidades_propias(respuesta) | {
        normalizar_texto(m) for m in re.findall(r"(?:^|[.;:!?]\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)", respuesta)
        if normalizar_texto(m) in nombres_conocidos
    }
    fuentes_nombradas = set(fuentes_base) | {i for i, nf in enumerate(nombres_fuentes) if nf & nombres_resp}
    ops_nombradas = set(ops_locales)
    for i in fuentes_nombradas:
        if 0 <= i < len(nums_por_fuente):
            ops_nombradas |= extraer_operandos(nums_por_fuente[i])

    der_locales = generar_cierre_derivados(ops_locales) if nums_resp else set()
    der_globales = generar_cierre_derivados(ops_globales) if nums_resp else set()
    resp_norm = normalizar_texto(respuesta)

    verificadas = auditar_cuentas_intraroracion(resp_norm, nums_resp, fuentes, nums_por_fuente)
    pct_fuentes = [v for n in nums_fuente if n["pct"] for v in n["valores"] if v < 100]
    totales = [v for n in nums_fuente if not n["pct"] and not n["anio"] for v in n["valores"]]
    partes_de_total = {p * t / 100.0 for p in pct_fuentes + [100.0 - q for q in pct_fuentes] for t in totales}

    clases = []
    for k, n in enumerate(nums_resp):
        anclas = extraer_operandos([d for d in directos if d is not n])
        tipo = detectar_intencion_aritmetica(resp_norm, n)
        if k in verificadas:
            clases.append("cuenta_correcta" if verificadas[k][0] == "ok" else "cuenta_falsa")
        elif tipo and not n["anio"] and any(aparece_en(n, nums_por_fuente[i]) for i in fuentes_base if 0 <= i < len(nums_por_fuente)):
            clases.append("directo")
        elif tipo and not n["anio"]:
            if (
                validar_operacion_anunciada(tipo, n, anclas, ops_locales)
                or (tipo == "diferencia" and not anclas and validar_operacion_anunciada(tipo, n, set(), ops_nombradas))
            ):
                clases.append("cuenta_correcta")
            else:
                clases.append("cuenta_falsa")
        elif aparece_en(n, nums_fuente):
            clases.append("directo")
        elif anclas and validar_operacion_anunciada("calculo", n, anclas, ops_locales):
            clases.append("derivado_anclado")
        elif categorizar_cifra(n, set(), der_locales) == "derivado":
            clases.append("derivado_local")
        elif not n["pct"] and any(es_aproximado(v, x, 0.005) for v in n["valores"] for x in partes_de_total):
            clases.append("derivado_local")
        elif categorizar_cifra(n, set(), der_globales) == "derivado":
            clases.append("derivado_casual")
        else:
            clases.append("inexplicado")

    sospechosos = [n for n, c in zip(nums_resp, clases) if c in ("inexplicado", "derivado_casual", "cuenta_falsa")]
    f["num_inexplicados"] = float(sum(1 for c in clases if c == "inexplicado"))
    f["num_derivado_casual"] = float(sum(1 for c in clases if c == "derivado_casual"))
    f["num_sospechosos_grandes"] = float(sum(1 for n in sospechosos if max(n["valores"]) > 10))
    f["num_derivado_local"] = float(sum(1 for c in clases if c == "derivado_local"))
    f["num_derivado_anclado"] = float(sum(1 for c in clases if c == "derivado_anclado"))
    f["num_cuenta_falsa"] = float(sum(1 for c in clases if c == "cuenta_falsa"))
    f["num_cuenta_correcta"] = float(sum(1 for c in clases if c == "cuenta_correcta"))
    f["num_hay_cifras"] = float(len(nums_resp) > 0)
    f["verif_cuenta_mal"] = float(min(2, sum(1 for e, _ in verificadas.values() if e == "mal")))
    f["verif_cuenta_ok"] = float(min(2, sum(1 for e, _ in verificadas.values() if e == "ok")))
    f["num_proporcion_falsa"] = float(detectar_proporciones_erroneas(respuesta, nums_resp, valores_fuente))

    nombres_pregunta = {normalizar_texto(p) for p in re.findall(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+", pregunta)}
    frecuencia = collections.Counter(n for nf in nombres_fuentes for n in nf)
    principales = nombres_pregunta | {n for n, k in frecuencia.items() if k >= 3}
    secundarios = [nf - principales for nf in nombres_fuentes]
    otra_entidad = [
        bool(secundarios[i]) and (bool(RE_MARCADOR_OTRA_ENT.search(normalizar_texto(x))) or bool(nombres_fuentes[i] & nombres_pregunta))
        for i, x in enumerate(fuentes)
    ]
    ajenas = ajenas_fuerte = 0
    for n, c in zip(nums_resp, clases):
        if c != "directo" or n["anio"]:
            continue
        donde = [i for i, ns in enumerate(nums_por_fuente) if aparece_en(n, ns)]
        if donde and all(otra_entidad[i] and not (secundarios[i] & nombres_resp) for i in donde):
            ajenas += 1
            if nombres_resp & principales:
                ajenas_fuerte += 1
    f["entidad_ajena"] = float(min(ajenas, 2))
    f["entidad_ajena_fuerte"] = float(min(ajenas_fuerte, 2))

    todos_nombres = set().union(*nombres_fuentes) if nombres_fuentes else set()
    conocidos = todos_nombres | {normalizar_texto(p) for p in re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", pregunta)}
    f["nombre_inventado"] = float(len(nombres_resp - conocidos))

    falsos_pl, correctos_pl = verificar_plazos_temporales(respuesta, texto_fuentes)
    f["fecha_plazo_falso"] = float(min(falsos_pl, 2))
    f["fecha_plazo_correcto"] = float(min(correctos_pl, 2))
    horas_fuentes = set(RE_PATRON_HORA.findall(" ".join(fuentes)))
    f["hora_inventada"] = float(any(h not in horas_fuentes for h in RE_PATRON_HORA.findall(respuesta)))

    fuentes_norm = normalizar_texto(" ".join(fuentes))
    base_norm = normalizar_texto(" ".join(fuentes[i] for i in sorted(fuentes_base) if 0 <= i < n_fuentes))
    stems_resp = obtener_stems(respuesta) | {tok for tok in re.findall(r"\d[\d.,]*", resp_norm)}
    perdidos_tramo = {k: 0 for k in DICC_MATICES}
    perdidos_fuente = {k: 0 for k in DICC_MATICES}
    huecos = []
    for i in citadas:
        if 0 <= i < n_fuentes:
            fn = normalizar_texto(fuentes[i])
            tramo, h = analizar_tramo_cubierto(fuentes[i], stems_resp)
            huecos.append(h)
            for k, pats in DICC_MATICES.items():
                perdidos_fuente[k] += contar_perdidos(pats, fn, resp_norm)
                if tramo:
                    perdidos_tramo[k] += contar_perdidos(pats, tramo, resp_norm)
    for k in DICC_MATICES:
        f["matiz_perdido_tramo_" + k] = float(min(perdidos_tramo[k], 2))
        f["matiz_perdido_fuente_" + k] = float(min(perdidos_fuente[k], 2))
    f["huecos_tramo_max"] = float(max(huecos) if huecos else 0)

    verbo_endurecido = 0.0
    for pat_suave, pat_fuerte in PARES_VERBOS_ENDURECIDOS:
        if re.search(pat_fuerte, resp_norm) and not re.search(pat_fuerte, fuentes_norm) and re.search(pat_suave, fuentes_norm):
            verbo_endurecido = 1.0
            break
    f["verbo_endurecido"] = verbo_endurecido

    usadas = set(citadas)
    for x in fragmentos:
        rpf = x.get("respaldo_por_fuente") or []
        if rpf:
            usadas.add(max(range(len(rpf)), key=lambda k: float(rpf[k] or 0.0)))
    usadas = {i for i in usadas if 0 <= i < n_fuentes}
    raices_r = obtener_stems(respuesta)
    fuera = []
    for o in oraciones:
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            rf = obtener_stems(fuentes[i])
            fuera.append(len(rf - raices_r) / max(1, len(rf)))
    f["cobertura_citada_fuera"] = max(fuera) if fuera else 0.5
    raices_usadas = obtener_stems(" ".join(fuentes[i] for i in usadas) + " " + pregunta)
    f["resp_fuera_de_usadas"] = len(raices_r - raices_usadas) / max(1, len(raices_r))

    todos_matices = [p for pats in DICC_MATICES.values() for p in pats]
    pierde = 0.0
    for o in oraciones:
        on = normalizar_texto(o.get("texto") or "")
        if any(re.search(p, on) for p in todos_matices):
            continue
        fuentes_o = set()
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            fuentes_o.add(i)
        sim = o.get("similitud_por_fuente") or []
        if sim:
            fuentes_o.add(max(range(len(sim)), key=lambda k: float(sim[k] or 0.0)))
        if any(any(re.search(p, normalizar_texto(fuentes[j])) for p in todos_matices) for j in fuentes_o if j < n_fuentes):
            pierde = 1.0
    f["matiz_oracion_perdido"] = pierde
    f["anadido_universal"] = float(contar_anadidos(PATRONES_UNIVERSALES, base_norm, resp_norm))
    f["anadido_causal"] = float(contar_anadidos(PATRONES_CAUSALES, base_norm, resp_norm))
    f["anadido_causal_todas"] = float(contar_anadidos(PATRONES_CAUSALES, fuentes_norm, resp_norm))

    raices_fuentes = obtener_stems(texto_fuentes)
    nuevas = raices_r - raices_fuentes
    f["palabras_nuevas"] = float(len(nuevas))
    f["palabras_nuevas_frac"] = len(nuevas) / max(1, len(raices_r))

    llm_ok = 1.0 - max(f["llm_no_respaldada"], f["llm_contradicha"])
    f["llm_ok_y_num_sospechoso"] = llm_ok * min(1.0, f["num_sospechosos_grandes"])
    f["llm_ok_y_nli_bajo"] = llm_ok * float(f["nli_resp_fuente_citada_min"] < 0.3)
    cifras_ok = float(f["num_sospechosos_grandes"] == 0 and f["entidad_ajena"] == 0 and f["fecha_plazo_falso"] == 0)
    f["llm_contra_sin_apoyo"] = f["llm_contradicha"] * cifras_ok * (1.0 - f["nli_contr_todas_max_frag"])
    f["n_oraciones"] = float(len(oraciones))

    for nombre in list(f):
        if nombre.startswith(("num_", "verif_", "matiz_", "anadido_", "huecos_", "nombre_", "entidad_", "fecha_", "hora_", "verbo_")):
            f["llmok_" + nombre] = llm_ok * f[nombre]

    return f




NOMBRES = ["anadido_causal", "anadido_causal_todas", "anadido_universal", "brecha_orac_frag", "cobertura_citada_fuera", "entidad_ajena", "entidad_ajena_fuerte", "fecha_plazo_correcto", "fecha_plazo_falso", "frag_frac_bajo_02", "frag_frac_bajo_05", "frag_respaldo_2o", "frag_respaldo_media", "frag_respaldo_min", "frag_similitud_min", "hora_inventada", "huecos_tramo_max", "llm_conf_media_respaldada", "llm_conf_min_respaldada", "llm_contra_sin_apoyo", "llm_contradicha", "llm_frac_alarmas", "llm_no_respaldada", "llm_ok_y_nli_bajo", "llm_ok_y_num_sospechoso", "llm_sin_veredicto", "llmok_anadido_causal", "llmok_anadido_causal_todas", "llmok_anadido_universal", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_fecha_plazo_correcto", "llmok_fecha_plazo_falso", "llmok_hora_inventada", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_parcial", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_aproximado", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_tramo_restrictivo", "llmok_nombre_inventado", "llmok_num_cuenta_correcta", "llmok_num_cuenta_falsa", "llmok_num_derivado_anclado", "llmok_num_derivado_casual", "llmok_num_derivado_local", "llmok_num_hay_cifras", "llmok_num_inexplicados", "llmok_num_proporcion_falsa", "llmok_num_sospechosos_grandes", "llmok_verbo_endurecido", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "matiz_oracion_perdido", "matiz_perdido_fuente_aproximado", "matiz_perdido_fuente_condicional", "matiz_perdido_fuente_parcial", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_aproximado", "matiz_perdido_tramo_condicional", "matiz_perdido_tramo_parcial", "matiz_perdido_tramo_restrictivo", "n_oraciones", "nli_contr_fuente_citada_max", "nli_contr_fuente_max_frag", "nli_contr_fuente_max_orac", "nli_contr_todas_max_frag", "nli_contr_todas_max_orac", "nli_mejor_fuente_min_orac", "nli_resp_fuente_citada_min", "nli_resp_todas_min_frag", "nli_resp_todas_min_orac", "nombre_inventado", "num_cuenta_correcta", "num_cuenta_falsa", "num_derivado_anclado", "num_derivado_casual", "num_derivado_local", "num_hay_cifras", "num_inexplicados", "num_proporcion_falsa", "num_sospechosos_grandes", "orac_mejor_resp_min", "palabras_nuevas", "palabras_nuevas_frac", "resp_fuera_de_usadas", "verbo_endurecido", "verif_cuenta_mal", "verif_cuenta_ok"]
INICIAL_GB = -0.03545633
TASA_GB = 0.050000
ARBOLES_GB = [[[21,17,77,35,-2,-2,67,-2,-2,37,78,-2,-2,89,-2,-2,75,88,12,-2,-2,-2,71,73,-2,-2,-2],[0.04545454680919647,0.9549999833106995,0.5,0.5,-2.0,-2.0,0.02800000086426735,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1547619104385376,-2.0,-2.0,0.05350000038743019,0.5147058963775635,0.8786931931972504,-2.0,-2.0,-2.0,0.8560000061988831,0.7585000097751617,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[-0.000534,-0.171274,0.012347,0.082022,-0.003662,1.464484,-0.427443,-1.965165,-1.113834,-0.314188,-0.350385,-1.507699,0.285542,0.102083,1.672342,-1.195692,0.449567,0.503173,0.505969,2.036092,1.876042,1.369216,0.320507,0.360715,1.625707,0.257756,-0.692038]],[[21,17,77,17,-2,-2,75,-2,-2,78,35,-2,-2,88,-2,-2,81,86,79,-2,-2,88,-2,-2,-2],[0.04545454680919647,0.9549999833106995,0.5,0.9375,-2.0,-2.0,0.017999999690800905,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.3541666716337204,-2.0,-2.0,0.5,0.6834999918937683,0.5,-2.0,-2.0,0.24038461595773697,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[0.005579,-0.170652,0.004526,0.063651,1.403685,-0.068664,-0.391611,-0.843798,-1.884562,-0.299907,-0.333435,-1.550271,0.004755,0.156404,1.314259,-1.235577,0.429701,0.444342,0.467996,1.900364,0.678674,0.3037,1.599887,-0.418472,-0.039809]],[[21,18,77,35,-2,-2,13,-2,-2,59,78,-2,-2,88,-2,-2,75,4,13,-2,-2,-2,71,4,-2,-2,68,-2,-2],[0.04545454680919647,0.9600000083446503,0.5,0.5,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.07200000062584877,0.8090909123420715,0.7615000009536743,-2.0,-2.0,-2.0,0.5,0.48076923191547394,-2.0,-2.0,0.7174999713897705,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[-0.003572,-0.162674,-0.009442,0.044113,-0.109404,1.240968,-0.376731,-0.924259,-1.818139,-0.288884,-0.32594,-1.433575,0.529536,0.172372,1.618142,-0.884553,0.411512,0.457073,0.459729,1.855739,1.723241,1.63836,0.294143,0.392509,1.853063,1.119609,0.150107,1.768914,0.136081]],[[21,17,45,37,-2,-2,67,-2,-2,57,46,-2,-2,87,-2,-2,75,14,-2,12,-2,-2,88,12,-2,-2,71,-2,-2],[0.04545454680919647,0.9562499821186066,0.5,0.5,-2.0,-2.0,0.02800000086426735,-2.0,-2.0,0.5,0.5,-2.0,-2.0,1.5,-2.0,-2.0,0.035499999299645424,0.10900000110268593,-2.0,0.8786931931972504,-2.0,-2.0,0.2361111119389534,0.2645000070333481,-2.0,-2.0,0.4974999874830246,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.006582,-0.143458,0.023794,0.082767,0.096861,1.692539,-0.357292,-1.74611,-0.856461,-0.263225,-0.303662,-1.358042,0.500791,0.009672,1.629315,-1.668967,0.390645,0.435284,1.551747,0.436646,1.778771,1.710431,0.278814,0.351564,0.497706,1.56472,0.074925,1.599149,-0.743651]],[[21,17,45,57,-2,-2,3,-2,-2,37,55,-2,-2,88,-2,-2,75,12,79,-2,-2,71,-2,-2,71,89,-2,-2,88,-2,-2],[0.04545454680919647,0.9549999833106995,0.5,0.5,-2.0,-2.0,0.019499999471008778,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.035499999299645424,0.8387500047683716,0.5,-2.0,-2.0,0.847000002861023,-2.0,-2.0,0.5,0.4365079402923584,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.003805,-0.153941,0.00115,0.061672,-0.052254,1.310016,-0.374162,-1.703126,-0.792446,-0.266003,-0.29836,-1.286017,0.667209,0.130653,1.801898,-0.75042,0.375688,0.412381,0.416357,1.712721,1.754268,0.37018,0.853339,1.722676,0.294495,0.413677,1.752649,1.179765,0.139645,1.139389,-0.770227]],[[17,77,81,79,-2,-2,-2,-2,13,3,68,-2,-2,70,-2,-2,57,78,-2,-2,87,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.07600000128149986,0.005499999970197678,0.9690000116825104,-2.0,-2.0,0.9160000085830688,-2.0,-2.0,0.5,0.5,-2.0,-2.0,2.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.005554,0.332681,0.360505,0.379593,1.625658,-0.211514,-0.516447,-0.92624,-0.162496,0.161578,0.336819,1.680423,0.341664,-0.048932,0.612688,-1.377971,-0.217745,-0.267454,-1.21114,0.372668,0.059639,1.050783,-1.623553]],[[21,17,89,47,-2,-2,13,-2,-2,37,53,-2,-2,69,-2,-2,74,79,73,-2,-2,-2,16,68,-2,-2,-2],[0.04545454680919647,0.9549999833106995,0.3693181872367859,0.5,-2.0,-2.0,0.03999999910593033,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.13099999725818634,-2.0,-2.0,0.25300000607967377,0.5,0.025500000454485416,-2.0,-2.0,-2.0,2.5,0.13249999657273293,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.007153,-0.120413,0.024047,0.105797,0.575525,-1.599795,-0.241873,1.840163,-1.409624,-0.225629,-0.255853,-1.198813,0.268491,0.14467,1.824036,-0.390222,0.348624,0.366076,0.371559,1.611332,1.448755,0.56229,0.233975,0.328068,0.817507,1.662815,-0.442819]],[[21,37,24,17,-2,-2,88,-2,-2,69,4,-2,-2,89,-2,-2,81,72,79,-2,-2,89,-2,-2,-2],[0.04545454680919647,0.5,0.5,0.9375,-2.0,-2.0,0.40833333134651184,-2.0,-2.0,0.13099999725818634,0.5277777910232544,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.5,0.7840000092983246,0.5,-2.0,-2.0,0.20416666567325592,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[-0.00441,-0.134714,-0.175565,-0.216731,0.78876,-1.053281,0.158983,1.076509,-1.883754,0.236663,0.386315,1.785628,1.136755,0.05992,1.366997,-0.66932,0.320142,0.333374,0.344858,1.512884,0.401421,0.182278,1.220657,-0.501959,-0.26785]],[[21,17,89,57,-2,-2,18,-2,-2,12,70,-2,-2,88,-2,-2,75,88,13,-2,-2,-2,11,14,-2,-2,88,-2,-2],[0.04545454680919647,0.9549999833106995,0.3693181872367859,0.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.651666671037674,0.060499999672174454,-2.0,-2.0,0.11324786394834518,-2.0,-2.0,0.07200000062584877,0.5147058963775635,0.535999983549118,-2.0,-2.0,-2.0,0.9864999949932098,0.801499992609024,-2.0,-2.0,0.2539682611823082,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.009346,-0.107491,0.026089,0.09666,0.084402,1.364304,-0.199632,0.869372,-1.176059,-0.20826,-0.021281,1.539186,-0.50593,-0.252054,-0.576516,-1.419993,0.307092,0.338669,0.340827,1.525821,1.380066,1.014053,0.228042,0.165802,0.364566,1.58873,0.347544,1.572707,0.847849]],[[17,77,81,79,-2,-2,-2,-2,57,55,21,-2,-2,87,-2,-2,87,68,-2,-2,35,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.10000000149011612,-2.0,-2.0,1.5,-2.0,-2.0,2.5,0.9354999959468842,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.007357,0.264076,0.293978,0.310294,1.438898,-0.538254,-0.634906,-1.090757,-0.116318,-0.166431,-0.190709,-0.904015,1.470459,0.325008,1.914266,-0.0096,0.123888,0.262737,1.535404,0.169275,-0.268674,1.553054,-1.691637]],[[21,37,91,18,-2,-2,88,-2,-2,12,9,-2,-2,14,-2,-2,75,79,12,-2,-2,-2,88,67,-2,-2,71,-2,-2],[0.04545454680919647,0.5,0.5,0.9249999821186066,-2.0,-2.0,0.2678571492433548,-2.0,-2.0,0.883576899766922,0.25757576525211334,-2.0,-2.0,0.5494999885559082,-2.0,-2.0,0.035499999299645424,0.5,0.8786931931972504,-2.0,-2.0,-2.0,0.24038461595773697,0.8614999949932098,-2.0,-2.0,0.4974999874830246,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[-0.000417,-0.106231,-0.140494,-0.165644,0.636501,-0.830365,0.306976,1.788014,-0.217039,0.224979,0.346478,0.996091,1.673621,-0.018019,-1.319922,0.844229,0.273307,0.309019,0.307963,1.450507,1.285438,1.615195,0.196139,0.250756,1.256427,0.058391,0.037009,1.292812,-0.578958]],[[17,77,81,79,-2,-2,-2,-2,13,70,89,-2,-2,69,-2,-2,57,85,-2,-2,89,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.0755000002682209,0.08900000154972076,0.41428571939468384,-2.0,-2.0,0.3334999978542328,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.005254,0.241302,0.268102,0.282197,1.370336,-0.5367,-0.506105,-1.335267,-0.105691,0.159767,0.314658,1.617212,-0.402831,0.051343,-0.764829,0.754239,-0.148793,-0.193257,-0.972871,0.495348,0.079783,1.366138,-0.854228]],[[17,77,81,79,-2,-2,-2,-2,57,86,77,-2,-2,63,-2,-2,87,89,-2,-2,21,-2,-2],[0.9350000023841858,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.09349999949336052,0.5,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.12916667014360428,-2.0,-2.0,0.04545454680919647,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.001458,0.232605,0.251991,0.269933,1.326735,-0.12764,-0.697452,-0.889094,-0.107695,-0.15485,0.078452,0.614201,-1.546711,-0.196952,-0.92574,1.306289,0.118076,0.257895,1.625024,0.511042,-0.263692,-1.597076,1.455233]],[[17,77,81,79,-2,-2,-2,-2,13,70,18,-2,-2,69,-2,-2,57,85,-2,-2,89,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.09200000017881393,0.8700000047683716,0.9600000083446503,-2.0,-2.0,0.3334999978542328,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.015375,0.212733,0.238931,0.252911,1.292944,-0.390528,-0.562478,-1.558177,-0.120862,0.099,0.239923,1.599366,0.629861,-0.077154,-1.689885,0.163071,-0.164129,-0.201204,-1.016799,0.284606,0.036489,1.080529,-0.969249]],[[21,57,13,70,-2,-2,53,-2,-2,87,68,-2,-2,74,-2,-2,75,79,14,-2,-2,-2,71,73,-2,-2,-2],[0.04545454680919647,0.5,0.1314999982714653,0.060499999672174454,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.9869999885559082,-2.0,-2.0,0.5259999930858612,-2.0,-2.0,0.05550000071525574,0.5,0.18250000476837158,-2.0,-2.0,-2.0,0.8165000081062317,0.8095000088214874,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.003146,-0.082876,-0.12687,0.078104,1.638649,-0.194647,-0.167614,-0.899956,0.48534,0.130091,0.249224,1.328404,0.101477,-0.276848,-1.500848,-0.484956,0.229723,0.257744,0.256109,1.085389,1.340006,1.574531,0.157335,0.201884,1.12502,-0.059499,-0.742484]],[[17,77,81,79,-2,-2,-2,-2,57,85,21,-2,-2,11,-2,-2,89,68,-2,-2,21,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.10000000149011612,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,0.2899159789085388,0.9864999949932098,-2.0,-2.0,0.10000000149011612,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.005766,0.197708,0.214732,0.23118,1.244098,-0.432541,-0.698306,-0.982683,-0.084928,-0.120087,-0.15326,-0.78829,1.276718,0.144123,-0.013518,1.552736,0.101396,0.242306,1.514125,-0.217308,-0.216248,-1.229794,1.412631]],[[17,77,81,2,-2,-2,-2,-2,13,77,79,-2,-2,-2,57,85,-2,-2,88,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.0885000005364418,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.11324786394834518,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[0.000312,0.205378,0.218955,0.228889,1.244518,0.310592,-0.282709,-0.514334,-0.091661,0.119003,0.166939,0.915761,-1.233765,-1.525074,-0.131308,-0.164911,-0.886696,0.198111,0.045852,1.45251,-0.466242]],[[17,77,79,4,-2,-2,-2,-2,57,13,70,-2,-2,85,-2,-2,89,68,-2,-2,14,-2,-2],[0.9375,0.5,0.5,0.47727273404598236,-2.0,-2.0,-2.0,-2.0,0.5,0.1314999982714653,0.8700000047683716,-2.0,-2.0,0.5,-2.0,-2.0,0.2567765712738037,0.9884999990463257,-2.0,-2.0,0.4985000044107437,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.005262,0.180251,0.199335,0.209943,1.259879,0.852111,-0.422908,-0.780796,-0.079009,-0.115103,0.062984,0.909587,-0.436476,-0.15318,-0.856468,0.457051,0.109357,0.253912,1.48531,-0.177564,-0.142582,-0.073374,-1.65416]],[[21,17,77,49,-2,-2,87,-2,-2,88,16,-2,-2,13,-2,-2,75,79,66,-2,-2,-2,16,75,-2,-2,-2],[0.04545454680919647,0.9725000262260437,0.5,0.5,-2.0,-2.0,2.5,-2.0,-2.0,0.12916667014360428,1.5,-2.0,-2.0,0.08800000324845314,-2.0,-2.0,0.053999999538064,0.5,3.5,-2.0,-2.0,-2.0,2.5,0.07600000128149986,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[-0.001085,-0.075972,0.006644,0.048394,0.37231,-1.489427,-0.25847,-1.429027,-0.597894,-0.160899,-0.066293,-0.745139,0.511086,-0.225246,0.054237,-1.255737,0.189046,0.220731,0.218569,1.272489,1.388385,1.544247,0.10994,0.146254,-0.768696,0.927281,-0.378066]],[[17,77,81,47,-2,-2,86,-2,-2,75,-2,69,-2,-2,21,88,35,-2,-2,78,-2,-2,13,74,-2,-2,88,-2,-2],[0.9549999833106995,0.5,0.5,0.5,-2.0,-2.0,0.14249999821186066,-2.0,-2.0,0.019499999936670065,-2.0,0.4155000001192093,-2.0,-2.0,0.04545454680919647,0.12916667014360428,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.07250000163912773,0.007500000298023224,-2.0,-2.0,0.3245614022016525,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.004901,0.098198,0.127531,0.152383,0.878812,-1.150043,-0.28289,0.011229,-1.692326,-0.239037,-0.475554,-0.300212,-1.365454,-1.605394,-0.114399,-0.1368,-0.037331,-0.445278,1.16347,-0.204226,-1.161254,0.095334,0.246907,0.217739,1.301744,0.784116,0.346911,1.461588,1.654593]],[[21,17,89,71,-2,-2,86,-2,-2,10,89,-2,-2,71,-2,-2,75,79,14,-2,-2,-2,71,74,-2,-2,-2],[0.04545454680919647,0.9725000262260437,0.18465909361839294,0.8030000030994415,-2.0,-2.0,0.022499999962747097,-2.0,-2.0,0.4166666716337204,0.12916667014360428,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.05550000071525574,0.5,0.1835000067949295,-2.0,-2.0,-2.0,0.8560000061988831,0.25300000607967377,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[-0.005706,-0.078403,0.00384,0.136154,0.884906,-0.777514,-0.082191,1.109092,-0.552498,-0.161167,-0.191088,-0.631318,-1.206627,-0.031884,1.84901,-0.454005,0.173206,0.199429,0.197688,0.952826,1.246189,1.465957,0.107423,0.135264,1.127909,0.4027,-0.966605]],[[17,77,81,67,-2,-2,-2,-2,57,85,21,-2,-2,11,-2,-2,87,73,-2,-2,21,-2,-2],[0.9375,0.5,0.5,0.09200000017881393,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.10000000149011612,-2.0,-2.0,0.9830000102519989,-2.0,-2.0,2.5,0.0025000000605359674,-2.0,-2.0,0.04545454680919647,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.004851,0.16148,0.177437,0.192171,1.255672,1.002585,-0.685723,-0.911591,-0.068054,-0.101227,-0.129376,-0.723653,1.191318,0.135694,-0.177508,1.462595,0.095318,0.207704,-0.045755,1.215011,-0.195909,-1.430279,1.313216]],[[21,17,77,89,-2,-2,87,-2,-2,41,91,-2,-2,-2,75,79,72,-2,-2,-2,71,68,-2,-2,-2],[0.04545454680919647,0.9725000262260437,0.5,0.18465909361839294,-2.0,-2.0,2.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0,0.05550000071525574,0.5,0.001500000071246177,-2.0,-2.0,-2.0,0.8560000061988831,0.948500007390976,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.002218,-0.057257,0.019746,0.060299,0.819355,-0.145443,-0.232976,-1.36967,-0.573225,-0.135274,-0.148969,-0.854054,0.546051,0.808459,0.155533,0.180804,0.178574,1.275029,1.17258,1.415965,0.097476,0.122648,0.504257,1.291919,-0.944471]],[[17,77,81,79,-2,-2,-2,-2,63,85,57,-2,-2,75,-2,-2,68,70,-2,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.009000000078231096,-2.0,-2.0,0.9594999849796295,0.024000000208616257,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.002031,0.139085,0.153904,0.164428,1.125924,-0.493331,-0.542001,-0.66369,-0.064297,-0.084864,-0.108885,-0.739989,0.355692,0.126668,1.627608,0.094639,0.237575,0.316352,1.253612,1.681578,0.150528]],[[21,89,34,14,-2,-2,70,-2,-2,9,88,-2,-2,71,-2,-2,72,11,72,-2,-2,-2,74,88,-2,-2,-2],[0.04545454680919647,0.12916667014360428,0.5,0.8149999976158142,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.22500000149011612,0.3060200661420822,-2.0,-2.0,0.013999999966472387,-2.0,-2.0,0.4529999941587448,0.9985000193119049,0.12050000205636024,-2.0,-2.0,-2.0,0.4450000077486038,0.2863636463880539,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.002402,-0.055148,0.046871,-0.097049,-0.204133,-1.399451,0.156262,1.25203,0.206122,-0.104176,-0.147036,-0.560658,-1.160016,0.035515,1.100261,-0.245277,0.145938,0.159706,0.164232,1.182402,0.868561,-0.124484,0.050414,0.142237,1.225529,0.05715,-0.490421]],[[21,41,17,49,-2,-2,70,-2,-2,68,70,-2,-2,-2,75,79,66,-2,-2,-2,71,73,-2,-2,14,-2,-2],[0.04545454680919647,0.5,0.9549999833106995,0.5,-2.0,-2.0,0.14400000125169754,-2.0,-2.0,0.9594999849796295,0.018000000156462193,-2.0,-2.0,-2.0,0.05550000071525574,0.5,3.5,-2.0,-2.0,-2.0,0.534500002861023,0.00849999999627471,-2.0,-2.0,0.7339999973773956,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,26,-1,-1],[-0.002005,-0.055288,-0.072811,-0.000103,0.147554,-1.476734,-0.124565,-0.438333,-0.952329,0.216375,0.286843,1.146791,1.594119,0.094164,0.140373,0.160906,0.159074,1.170313,1.280185,1.377495,0.081834,0.130244,0.175658,1.205382,-0.043625,-0.741977,1.267168]],[[17,77,81,2,-2,-2,-2,-2,13,70,17,-2,-2,12,-2,-2,88,57,-2,-2,65,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.0885000005364418,0.551499992609024,0.9566666483879089,-2.0,-2.0,0.43441666662693024,-2.0,-2.0,0.12916667014360428,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.003305,0.120653,0.136434,0.147682,1.064392,-0.181657,-0.616031,-0.997202,-0.052677,0.095662,0.204893,1.540428,0.631227,-0.020579,-1.006259,0.569219,-0.080245,-4.8e-05,-0.229811,1.063306,-0.134054,-0.7722,0.436638]],[[17,86,81,34,-2,-2,-2,86,68,-2,-2,67,-2,-2,63,85,61,-2,-2,75,-2,-2,88,-2,-2],[0.9375,0.7650000154972076,0.5,0.5,-2.0,-2.0,-2.0,0.9415000081062317,0.9340000152587891,-2.0,-2.0,0.04450000077486038,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.009000000078231096,-2.0,-2.0,0.16025641560554504,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[-0.002226,0.114054,0.138753,0.149649,1.014686,1.520019,-0.469332,-0.006781,-0.18101,-1.587414,0.584759,0.122334,0.963884,-0.149193,-0.056692,-0.073223,-0.09339,-0.614764,0.608543,0.12119,1.637996,0.033095,0.208387,1.500042,0.299524]],[[21,17,45,16,-2,-2,67,-2,-2,87,57,-2,-2,73,-2,-2,75,87,79,-2,-2,19,-2,-2,71,67,-2,-2,88,-2,-2],[0.04545454680919647,0.9758333563804626,0.5,2.5,-2.0,-2.0,0.03150000050663948,-2.0,-2.0,1.5,0.5,-2.0,-2.0,0.006500000134110451,-2.0,-2.0,0.035499999299645424,5.5,0.5,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0,0.5,0.0025000000605359674,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.000379,-0.045639,0.019191,0.050249,0.067164,1.047907,-0.175047,-1.300696,-0.429052,-0.112558,-0.064944,-0.568291,1.102449,-0.17815,-0.449761,-1.266058,0.115573,0.142685,0.13865,1.13812,1.326427,0.228334,1.240801,1.396148,0.057197,0.123082,0.271346,1.22707,-0.031941,0.388076,-1.006223]],[[17,81,79,77,-2,-2,-2,-2,63,85,61,-2,-2,75,-2,-2,4,35,-2,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.009000000078231096,-2.0,-2.0,0.4722222238779068,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.003586,0.105503,0.118014,0.127758,1.068493,-0.480459,-0.736376,-0.741436,-0.042867,-0.057719,-0.075668,-0.509626,0.548628,0.120004,1.526664,0.065794,0.196618,0.2786,1.571825,1.202849,0.135734]],[[17,81,77,34,-2,-2,-2,-2,89,16,61,-2,-2,70,-2,-2,21,46,-2,-2,13,-2,-2],[0.9375,0.5,0.5,1.5,-2.0,-2.0,-2.0,-2.0,0.12916667014360428,1.5,0.5,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.04545454680919647,0.5,-2.0,-2.0,0.07250000163912773,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.006922,0.106161,0.118224,0.126965,0.936667,1.372658,-0.373774,-0.853821,-0.039223,0.047699,-0.020895,-0.329419,1.565524,0.180648,1.589937,0.029276,-0.08184,-0.09952,-0.637624,0.453467,0.194792,1.205837,1.435834]],[[21,17,89,79,-2,-2,13,-2,-2,78,88,-2,-2,88,-2,-2,72,11,18,-2,-2,-2,74,73,-2,-2,-2],[0.10000000149011612,0.9549999833106995,0.3693181872367859,0.5,-2.0,-2.0,0.03999999910593033,-2.0,-2.0,0.5,0.11324786394834518,-2.0,-2.0,0.3541666716337204,-2.0,-2.0,0.4529999941587448,0.9985000193119049,0.4749999940395355,-2.0,-2.0,-2.0,0.40949998795986176,0.04449999891221523,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.00664,-0.034899,0.031532,0.079591,0.494779,-1.259091,-0.12314,1.038505,-0.977472,-0.086226,-0.102056,-0.126189,-0.923477,0.135395,1.185705,-0.916451,0.111566,0.123845,0.128302,1.072654,1.226936,-0.243163,0.02792,0.117263,0.325234,1.209506,-0.571434]],[[21,88,91,35,-2,-2,69,-2,-2,13,74,-2,-2,12,-2,-2,75,87,18,-2,-2,22,-2,-2,70,73,-2,-2,69,-2,-2],[0.04545454680919647,0.33908046782016754,0.5,0.5,-2.0,-2.0,0.8350000083446503,-2.0,-2.0,0.0625,0.006500000134110451,-2.0,-2.0,0.44474999606609344,-2.0,-2.0,0.035499999299645424,5.5,0.9600000083446503,-2.0,-2.0,0.5,-2.0,-2.0,0.17599999904632568,0.6279999911785126,-2.0,-2.0,0.9009999930858612,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[0.003756,-0.033647,-0.006628,-0.022224,-0.278379,0.65481,0.332119,1.272169,1.644804,-0.153585,0.089684,1.050748,-0.313652,-0.184715,-1.6891,-1.055837,0.098973,0.120529,0.116341,1.100698,1.189497,0.205294,1.353375,1.208706,0.051153,0.141805,1.090493,1.225693,-0.01559,-0.489252,0.942614]],[[13,45,5,11,-2,-2,-2,-2,17,77,81,-2,-2,13,-2,-2,88,34,-2,-2,21,-2,-2],[0.09149999916553497,0.5,0.5,0.9174999892711639,-2.0,-2.0,-2.0,-2.0,0.9725000262260437,0.5,0.5,-2.0,-2.0,0.6385000050067902,-2.0,-2.0,0.1180555559694767,1.5,-2.0,-2.0,0.10000000149011612,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.010264,0.117674,0.12901,0.138368,0.712651,1.283624,-0.572675,-1.136796,-0.027402,0.027272,0.053725,0.425082,-0.995951,-0.161905,-0.430936,-1.296365,-0.099473,-0.021738,-0.516433,0.771531,-0.145422,-0.998417,1.344412]],[[17,73,34,74,-2,-2,-2,77,73,-2,-2,-2,13,86,79,-2,-2,-2,88,16,-2,-2,85,-2,-2],[0.9375,0.007500000298023224,1.5,0.7529999911785126,-2.0,-2.0,-2.0,0.5,0.00849999999627471,-2.0,-2.0,-2.0,0.09200000017881393,0.8740000128746033,0.5,-2.0,-2.0,-2.0,0.12916667014360428,1.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[-0.000217,0.092859,0.126027,0.113396,1.130715,-0.173798,1.560297,0.041084,0.063158,-1.671662,0.614983,-1.10042,-0.044293,0.079587,0.09908,0.666058,-1.155914,-1.478233,-0.067872,-0.004591,-0.364803,0.66497,-0.109666,-0.777755,0.087804]],[[17,77,81,79,-2,-2,86,-2,-2,13,-2,69,-2,-2,21,46,88,-2,-2,88,-2,-2,13,4,-2,-2,-2],[0.9549999833106995,0.5,0.5,0.5,-2.0,-2.0,0.14249999821186066,-2.0,-2.0,0.6385000050067902,-2.0,0.3374999910593033,-2.0,-2.0,0.04545454680919647,0.5,0.11324786394834518,-2.0,-2.0,0.36666667461395264,-2.0,-2.0,0.07250000163912773,0.5773809552192688,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,-1],[0.00213,0.058698,0.075967,0.089959,0.684831,-0.779313,-0.131966,0.799566,-1.342388,-0.148532,-0.194782,-0.211541,-1.208803,-1.377178,-0.066613,-0.080561,-0.095166,-0.093753,-0.979876,0.142239,1.313147,-0.82526,0.150859,0.12602,1.161127,0.403152,1.380284]],[[13,45,3,11,-2,-2,-2,-2,17,89,77,-2,-2,24,-2,-2,4,72,-2,-2,67,-2,-2],[0.1314999982714653,0.5,0.7974999845027924,0.992499977350235,-2.0,-2.0,-2.0,-2.0,0.9725000262260437,0.18465909361839294,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.4365079402923584,0.24499999731779099,-2.0,-2.0,0.4870000034570694,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.002017,0.085465,0.099159,0.105842,0.603619,1.460836,-1.49054,-1.154075,-0.037097,0.009172,0.084311,0.623995,-1.270167,-0.050262,-0.437073,0.823615,-0.094177,-0.052767,0.685767,-0.587231,-0.155188,-1.166604,0.414375]],[[13,45,49,86,-2,-2,-2,-2,17,89,77,-2,-2,24,-2,-2,88,4,-2,-2,67,-2,-2],[0.1314999982714653,0.5,0.5,0.8819999992847443,-2.0,-2.0,-2.0,-2.0,0.9725000262260437,0.18465909361839294,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.1180555559694767,0.1180555559694767,-2.0,-2.0,0.7639999985694885,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.004578,0.076572,0.087666,0.098677,0.797176,-1.422274,-0.780718,-1.126224,-0.036969,0.00585,0.075994,0.605758,-1.303205,-0.053067,-0.485986,0.952541,-0.090662,-0.027876,-1.184376,0.219839,-0.131704,-1.053873,1.159158]],[[17,81,79,86,-2,-2,-2,-2,59,85,61,-2,-2,68,-2,-2,70,13,-2,-2,-2],[0.9375,0.5,0.5,0.7635000050067902,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.3334999978542328,-2.0,-2.0,0.5940000116825104,0.8510000109672546,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.001388,0.074259,0.08444,0.090416,1.047482,0.140716,-0.538032,-0.701447,-0.033558,-0.049889,-0.065707,-0.507478,0.576896,0.09819,-0.818671,0.749593,0.134871,0.185875,1.336366,0.305315,-0.535037]],[[77,17,89,79,-2,-2,81,-2,-2,86,12,-2,-2,13,-2,-2,75,67,-2,-2,16,67,-2,-2,68,-2,-2],[0.5,0.9725000262260437,0.18465909361839294,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.6845000088214874,0.7631785571575165,-2.0,-2.0,0.9684999883174896,-2.0,-2.0,0.017999999690800905,0.029000001028180122,-2.0,-2.0,0.5,0.04700000025331974,-2.0,-2.0,0.6644999980926514,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,22,-1,-1,25,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.003006,0.01739,0.053877,0.116606,0.898931,-0.858809,0.001921,0.136983,-1.215443,-0.042998,0.030371,0.548813,-0.60063,-0.089802,-1.014248,-0.278183,-0.150633,-0.085335,-1.130317,0.230995,-0.17945,-0.15726,-1.156235,-1.254422,-0.202274,-1.210014,-1.370328]],[[77,13,86,75,-2,-2,-2,88,79,-2,-2,18,-2,-2,71,-2,71,88,-2,-2,-2],[0.5,0.09149999916553497,0.8740000128746033,0.33149999380111694,-2.0,-2.0,-2.0,0.12916667014360428,0.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0,0.006500000134110451,-2.0,0.9865000247955322,0.0555555559694767,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,18,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,20,19,-1,-1,-1],[-0.004635,0.010726,0.088454,0.09598,0.720058,1.475595,-1.383918,-0.017749,0.04097,0.35506,-1.526993,-0.06139,0.384777,-0.542426,-0.14754,-0.419934,-0.161422,-0.17567,-1.290532,-1.225319,-0.345817]],[[13,45,86,74,-2,-2,-2,-2,88,55,23,-2,-2,-2,19,85,-2,-2,-2],[0.09149999916553497,0.5,0.8740000128746033,0.048499999567866325,-2.0,-2.0,-2.0,-2.0,0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,0.0024999999441206455,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,15,16,-1,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,18,17,-1,-1,-1],[0.000267,0.085239,0.094466,0.101394,0.740798,1.452125,-1.359608,-1.038939,-0.030329,-0.010549,-0.021268,0.163535,-0.391691,1.35733,-0.128648,-0.147216,-1.031556,-1.116499,1.029391]],[[21,55,67,9,-2,-2,11,-2,-2,88,13,-2,-2,-2,74,11,79,-2,-2,-2,74,-2,19,-2,-2],[0.04545454680919647,0.5,0.010499999858438969,0.4166666716337204,-2.0,-2.0,0.9554999768733978,-2.0,-2.0,0.2678571492433548,0.36499999463558197,-2.0,-2.0,-2.0,0.33650000393390656,0.9985000193119049,0.5,-2.0,-2.0,-2.0,0.5209999978542328,-2.0,0.715499997138977,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,22,-1,24,-1,-1],[-0.003749,-0.032665,-0.044492,-0.005394,-0.18789,0.868125,-0.113815,-0.998209,-0.346595,0.169241,0.273792,1.301366,1.475802,-0.240916,0.071895,0.084003,0.088118,1.0491,-0.232108,-0.556453,-0.002712,-1.852736,0.069288,-0.077768,1.157342]],[[17,77,79,81,-2,-2,11,-2,-2,87,69,-2,-2,-2,63,21,4,-2,-2,13,-2,-2,-2],[0.9741666913032532,0.5,0.5,0.5,-2.0,-2.0,0.9050000011920929,-2.0,-2.0,2.5,0.38599999248981476,-2.0,-2.0,-2.0,0.5,0.04545454680919647,0.19090909510850906,-2.0,-2.0,0.11899999901652336,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.003059,0.038789,0.054544,0.066155,0.545453,-0.603703,-0.136573,-1.380712,0.173863,-0.120191,-0.16184,-1.112924,-1.248678,0.161378,-0.053794,-0.063926,-0.07807,-1.141848,-0.392002,0.11847,1.010691,1.32433,1.069835]],[[13,45,86,75,-2,-2,-2,-2,20,35,55,-2,-2,89,-2,-2,68,-2,21,-2,-2],[0.09149999916553497,0.5,0.8740000128746033,0.042000001296401024,-2.0,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.2899159789085388,-2.0,-2.0,0.13249999657273293,-2.0,0.75,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[-0.008123,0.068137,0.078579,0.085458,0.602601,0.991765,-1.330728,-1.04848,-0.034906,-0.049391,-0.066417,-0.487985,0.54996,0.054011,0.808301,-1.339483,0.067551,-0.654332,0.081917,1.356474,0.762548]],[[13,45,86,74,-2,-2,-2,-2,89,16,69,-2,-2,70,-2,-2,86,14,-2,-2,20,-2,-2],[0.0885000005364418,0.5,0.8740000128746033,0.048499999567866325,-2.0,-2.0,-2.0,-2.0,0.12916667014360428,0.5,0.9645000100135803,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.029000000096857548,0.4560000002384186,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.003462,0.070968,0.081166,0.088013,0.705097,1.339996,-1.311245,-1.028184,-0.028511,0.032584,-0.055172,-0.543469,1.405714,0.096363,1.071096,0.015835,-0.06155,0.093451,1.683464,-0.739107,-0.071027,-0.541103,0.295864]],[[17,47,77,73,-2,-2,75,-2,-2,-2,21,70,68,-2,-2,11,-2,-2,13,88,-2,-2,-2],[0.9549999833106995,0.5,0.5,0.18699999898672104,-2.0,-2.0,0.017999999690800905,-2.0,-2.0,-2.0,0.04545454680919647,0.07649999856948853,0.9945000112056732,-2.0,-2.0,0.8815000057220459,-2.0,-2.0,0.09050000086426735,0.4522727280855179,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.002699,0.043128,0.052528,0.065399,0.334298,1.03994,-0.087845,0.257557,-1.203113,-1.11912,-0.048679,-0.060231,-0.013649,-0.285883,2.632865,-0.101758,-1.07513,-0.499972,0.11205,0.089795,1.113004,0.150385,1.302295]],[[47,77,86,70,-2,-2,88,-2,-2,86,-2,71,-2,-2,11,-2,70,-2,-2],[0.5,0.5,0.7120000123977661,0.02350000012665987,-2.0,-2.0,0.38750000298023224,-2.0,-2.0,0.029000000096857548,-2.0,0.9865000247955322,-2.0,-2.0,0.9230000078678131,-2.0,0.23749999701976776,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,-1],[0.000991,0.00878,0.022275,0.058261,1.156184,0.300098,-0.024775,-0.029526,-1.111507,-0.110436,0.010495,-0.125717,-1.191837,-0.186311,-0.175394,-1.43023,-0.099997,-1.261398,0.009429]],[[13,77,34,75,-2,-2,13,-2,-2,-2,57,17,71,-2,-2,85,-2,-2,87,68,-2,-2,17,-2,-2],[0.09149999916553497,0.5,0.5,0.04649999924004078,-2.0,-2.0,0.03849999979138374,-2.0,-2.0,-2.0,0.5,0.9350000023841858,0.026000000536441803,-2.0,-2.0,0.5,-2.0,-2.0,1.5,0.9894999861717224,-2.0,-2.0,0.9562499821186066,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.000626,0.074094,0.08324,0.058349,0.86297,-0.526343,0.160705,0.657653,1.345645,-1.077322,-0.024819,-0.041726,0.022899,0.768032,-0.048385,-0.060909,-0.507767,0.229589,0.067687,0.159436,1.331632,-0.035942,-0.040224,0.418897,-1.22432]],[[17,81,34,79,-2,-2,69,-2,-2,73,-2,11,-2,-2,12,70,88,-2,-2,72,-2,-2,14,-2,61,-2,-2],[0.9549999833106995,0.5,2.5,0.5,-2.0,-2.0,0.8339999914169312,-2.0,-2.0,0.0035000001080334187,-2.0,0.6730000078678131,-2.0,-2.0,0.7603500187397003,0.3605000078678131,0.0416666679084301,-2.0,-2.0,0.06299999915063381,-2.0,-2.0,0.24649999290704727,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,26,-1,-1],[0.000196,0.036033,0.048144,0.035263,0.369244,-0.93634,0.192015,1.4565,-0.58077,-0.127379,0.47174,-0.245913,-1.489216,-1.288523,-0.045808,0.015547,0.108124,2.200231,0.335679,-0.057305,0.356858,-0.877622,-0.076545,1.789783,-0.086125,-0.805704,0.412314]],[[17,81,79,2,-2,-2,-2,-2,59,87,57,-2,-2,21,-2,-2,12,86,-2,-2,14,-2,-2],[0.9375,0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0,0.5,1.5,0.5,-2.0,-2.0,0.10000000149011612,-2.0,-2.0,0.8841666579246521,0.06550000049173832,-2.0,-2.0,0.5964999794960022,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-2.8e-05,0.049581,0.059609,0.066181,0.832164,-0.600819,-0.562385,-1.006873,-0.023321,-0.037671,-0.002761,-0.188771,1.026015,-0.080419,-0.648667,0.936743,0.113959,0.173884,0.524294,1.431502,-0.005123,-0.858522,0.960324]],[[17,73,34,79,-2,-2,-2,89,70,-2,-2,-2,70,12,89,-2,-2,70,-2,-2,55,72,-2,-2,-2],[0.9375,0.007500000298023224,1.5,0.5,-2.0,-2.0,-2.0,0.47727273404598236,0.44350001215934753,-2.0,-2.0,-2.0,0.05849999934434891,0.6828333139419556,0.41428571939468384,-2.0,-2.0,0.05049999989569187,-2.0,-2.0,0.5,0.0494999997317791,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[-0.004681,0.05166,0.081911,0.071146,1.035638,-0.883752,1.529255,0.005555,0.01985,0.744322,-0.203141,-1.467187,-0.031578,0.016283,0.154932,1.283742,-0.981552,-0.017315,-0.211447,1.637831,-0.068564,-0.080795,-0.016984,-0.74689,0.902889]],[[77,89,34,18,-2,-2,75,-2,-2,74,73,-2,-2,14,-2,-2,10,88,-2,68,-2,-2,-2],[0.5,0.12916667014360428,1.5,0.9850000143051147,-2.0,-2.0,0.901500016450882,-2.0,-2.0,0.0925000011920929,0.02949999924749136,-2.0,-2.0,0.1860000044107437,-2.0,-2.0,0.5833333432674408,0.1180555559694767,-2.0,0.9120000004768372,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,-1],[0.005996,0.019396,0.069457,0.033883,0.686439,-0.576404,0.185074,0.614206,1.932413,-0.007349,0.031531,0.560663,-0.303775,-0.051947,2.706959,-0.419229,-0.117466,-0.131286,-1.222465,-0.117486,-1.105455,-1.164418,-0.315778]],[[21,88,55,35,-2,-2,-2,86,-2,12,-2,-2,75,79,66,-2,-2,-2,86,-2,16,-2,-2],[0.04545454680919647,0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,0.03349999897181988,-2.0,0.46449999511241913,-2.0,-2.0,0.03500000014901161,0.5,3.5,-2.0,-2.0,-2.0,0.07349999994039536,-2.0,2.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,21,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,22,-1,-1],[-0.010157,-0.03169,-0.013645,-0.022594,-0.287977,0.557607,1.219812,-0.11226,0.456442,-0.132307,-1.751212,-1.019294,0.045943,0.066387,0.064503,1.017683,1.151732,1.22218,0.001647,-1.491011,0.023039,0.507484,-0.693627]],[[18,77,89,73,-2,-2,75,-2,-2,87,88,-2,-2,-2,12,11,68,-2,-2,69,-2,-2,68,75,-2,-2,16,-2,-2],[0.9750000238418579,0.5,0.49193547666072845,0.2264999970793724,-2.0,-2.0,0.05550000071525574,-2.0,-2.0,2.5,0.1339285746216774,-2.0,-2.0,-2.0,0.7631785571575165,0.9884999990463257,0.012499999720603228,-2.0,-2.0,0.22950000315904617,-2.0,-2.0,0.9945000112056732,0.8765000104904175,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.000793,0.028391,0.040204,0.054969,0.260862,0.971655,-0.066201,0.534045,-1.016731,-0.093794,-0.126656,-1.207238,-1.089516,0.198988,-0.047395,0.033792,-0.007744,2.303579,-0.217211,0.195048,0.169922,2.08467,-0.082898,-0.098491,-1.046323,-0.552815,0.037267,-0.684155,1.455324]],[[13,77,75,75,-2,-2,11,-2,-2,-2,39,20,91,-2,-2,4,-2,-2,3,14,-2,-2,-2],[0.0925000011920929,0.5,0.042000001296401024,0.018499999307096004,-2.0,-2.0,0.910999983549118,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.0555555559694767,-2.0,-2.0,0.001500000013038516,0.7195000052452087,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[-0.007026,0.051098,0.059439,0.04382,0.635323,-0.617978,0.1427,0.16367,1.331387,-1.042756,-0.027052,-0.036702,-0.049349,-0.399935,0.7,0.045273,-0.861092,0.713974,0.09402,0.197289,0.764143,1.566202,-0.882653]],[[77,89,16,69,-2,-2,66,-2,-2,86,74,-2,-2,86,-2,-2,74,-2,17,-2,16,-2,-2],[0.5,0.12916667014360428,0.5,0.30150000751018524,-2.0,-2.0,2.5,-2.0,-2.0,0.8980000019073486,0.9364999830722809,-2.0,-2.0,0.9824999868869781,-2.0,-2.0,0.001500000071246177,-2.0,0.9249999821186066,-2.0,2.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,22,-1,-1],[0.00467,0.016031,0.058103,-0.003018,-0.715547,0.551331,0.10457,0.88592,-1.264659,-0.008679,0.02158,0.128271,3.580263,-0.069106,-1.007371,-0.2452,-0.10406,-0.123432,-0.118837,-1.225152,-0.109174,-1.121484,-1.338768]],[[55,50,12,86,-2,-2,-2,70,13,-2,-2,72,-2,-2,67,-2,-2],[0.5,0.5,0.9962500035762787,0.9884999990463257,-2.0,-2.0,-2.0,0.351500004529953,0.609499990940094,-2.0,-2.0,0.07499999925494194,-2.0,-2.0,0.005499999853782356,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[0.003706,-0.002228,0.044558,0.049368,0.547812,1.701244,-1.334475,-0.025328,0.00148,0.611527,-0.188928,-0.077644,0.060823,-0.857628,0.162889,0.043939,1.375457]],[[89,77,47,4,-2,-2,-2,89,-2,-2,86,85,12,-2,-2,-2,14,-2,64,-2,-2],[0.18465909361839294,0.5,0.5,0.09545454755425453,-2.0,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,0.022499999962747097,0.5,0.7682666480541229,-2.0,-2.0,-2.0,0.17750000208616257,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,-1,17,-1,19,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,16,15,14,-1,-1,-1,18,-1,20,-1,-1],[0.002166,0.03923,0.049515,0.057765,-0.171769,0.596962,-0.981044,-0.115874,-1.177606,-0.930326,-0.028087,0.059346,0.047036,0.807459,-0.678766,1.501542,-0.048441,1.41595,-0.054422,-0.459076,1.043435]],[[79,88,77,72,-2,-2,75,-2,-2,75,86,-2,-2,86,-2,-2,71,11,-2,-2,-2],[0.5,0.4686274528503418,0.5,0.9895000159740448,-2.0,-2.0,0.016000000294297934,-2.0,-2.0,0.010499999858438969,0.004500000039115548,-2.0,-2.0,0.12000000104308128,-2.0,-2.0,0.8770000040531158,0.93299999833107,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.001463,0.009095,0.01809,0.027585,0.301794,-0.53052,-0.083161,-0.226428,-1.149375,-0.095051,0.029559,1.41293,-0.165726,-0.136385,-1.998724,-1.031968,-0.162354,-0.208252,-1.443051,-1.079699,-0.344405]],[[17,34,88,81,-2,-2,-2,-2,87,55,57,-2,-2,-2,21,73,-2,-2,13,-2,-2],[0.9375,2.5,0.5916666686534882,0.5,-2.0,-2.0,-2.0,-2.0,2.5,0.5,0.5,-2.0,-2.0,-2.0,0.10000000149011612,0.18049999326467514,-2.0,-2.0,0.06350000016391277,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,-1,15,16,-1,-1,19,-1,-1],[8,7,6,5,-1,-1,-1,-1,14,13,12,-1,-1,-1,18,17,-1,-1,20,-1,-1],[0.000575,0.045359,0.038949,0.041765,0.619022,-0.477949,-2.25872,1.280225,-0.019732,-0.001276,-0.012549,-0.224649,0.589283,1.311099,-0.084314,-0.110803,-1.072588,-0.553882,0.094026,0.84549,1.30509]],[[77,79,81,12,-2,-2,11,-2,-2,75,86,-2,-2,-2,69,-2,17,-2,68,-2,-2],[0.5,0.5,0.5,0.7716666758060455,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,0.7310000061988831,0.022499999962747097,-2.0,-2.0,-2.0,0.060499999672174454,-2.0,0.9249999821186066,-2.0,0.9120000004768372,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,18,-1,20,-1,-1],[0.001129,0.010113,0.017584,0.026525,0.617643,-0.040223,-0.102052,-1.129724,0.716777,-0.117021,-0.171411,0.319722,-1.330843,0.593111,-0.089067,0.02705,-0.10672,-1.212127,-0.096767,-1.085909,-1.168241]],[[49,17,73,79,-2,-2,89,-2,-2,12,88,-2,-2,67,-2,-2,70,12,-2,75,-2,-2,-2],[0.5,0.9741666913032532,0.23349999636411667,0.5,-2.0,-2.0,0.563492089509964,-2.0,-2.0,0.7631785571575165,0.11437908560037613,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.9339999854564667,0.5927500128746033,-2.0,0.9059999883174896,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,-1],[0.003848,0.011465,0.034154,0.013733,0.201256,-0.934582,0.120815,0.914105,-1.671498,-0.024765,0.044734,1.358754,-0.042213,-0.055306,-0.959318,-0.115732,-0.133651,-0.171559,-1.324754,-0.136116,-1.139063,-1.348418,-0.114941]],[[13,86,75,3,-2,-2,34,-2,-2,-2,4,39,14,-2,-2,13,-2,-2,14,67,-2,-2,89,-2,-2],[0.0885000005364418,0.8740000128746033,0.033000001683831215,0.005499999970197678,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.4580419659614563,0.5,0.2575000077486038,-2.0,-2.0,0.7950000166893005,-2.0,-2.0,0.796999990940094,0.47450000047683716,-2.0,-2.0,0.3484848588705063,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.003108,0.043077,0.048944,0.032388,0.705048,-0.367208,0.126274,0.25538,1.311492,-1.270564,-0.018833,0.007192,-0.003511,1.110276,-0.116185,0.154611,-0.324927,1.513469,-0.054711,-0.067398,-0.610247,0.238006,0.07819,1.022225,-0.879727]],[[17,81,67,34,-2,-2,20,-2,-2,-2,85,59,61,-2,-2,12,-2,-2,87,4,-2,-2,72,-2,-2],[0.9375,0.5,0.08999999985098839,1.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.5,0.5,0.5,-2.0,-2.0,0.883576899766922,-2.0,-2.0,1.5,0.4642857164144516,-2.0,-2.0,0.32200000435113907,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.005641,0.047121,0.053002,0.077632,0.871697,1.312475,0.025407,-0.731781,0.813839,-0.749003,-0.013464,-0.025205,-0.039112,-0.380547,0.449174,0.095963,1.091191,0.100224,0.109685,0.219004,1.60937,0.901186,0.003403,-0.853903,0.644304]],[[89,39,16,50,-2,-2,4,-2,-2,-2,17,88,67,-2,-2,34,-2,-2,14,11,-2,-2,21,-2,-2],[0.12916667014360428,0.5,0.5,0.5,-2.0,-2.0,0.5505050718784332,-2.0,-2.0,-2.0,0.9350000023841858,0.1547619104385376,0.007500000298023224,-2.0,-2.0,1.5,-2.0,-2.0,0.18200000375509262,0.9769999980926514,-2.0,-2.0,0.04545454680919647,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.002096,0.035773,0.022047,-0.036747,1.101151,-0.688923,0.068918,0.75502,-0.518627,1.199525,-0.022131,0.032859,-0.050423,-1.404868,0.363256,0.054816,0.568568,1.467311,-0.043798,0.140953,-0.053879,1.978113,-0.050895,-0.429987,1.047328]],[[18,89,73,79,-2,-2,74,-2,-2,13,3,-2,-2,85,-2,-2,68,12,69,-2,-2,86,-2,-2,70,-2,73,-2,-2],[0.9750000238418579,0.44949495792388916,0.23349999636411667,0.5,-2.0,-2.0,0.804999977350235,-2.0,-2.0,0.032500000670552254,0.0005000000237487257,-2.0,-2.0,0.5,-2.0,-2.0,0.9945000112056732,0.7482380867004395,0.009499999694526196,-2.0,-2.0,0.9544999897480011,-2.0,-2.0,0.1525000035762787,-2.0,0.005500000203028321,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,26,-1,28,-1,-1],[0.002183,0.022531,0.035801,0.016007,0.225105,-1.005466,0.121662,0.478387,1.654322,-0.060816,0.08113,0.56759,1.399328,-0.107932,-0.670184,-1.733965,-0.032917,-0.047356,0.019021,1.51758,-0.061009,-0.072874,-1.090482,-0.354272,0.07045,2.136772,-0.058295,0.567371,-1.035061]],[[17,34,23,13,-2,-2,9,-2,-2,88,69,-2,-2,-2,59,21,11,-2,-2,13,-2,-2,12,-2,-2],[0.9741666913032532,2.5,0.5,0.992499977350235,-2.0,-2.0,0.4166666716337204,-2.0,-2.0,0.2899159789085388,0.13750000298023224,-2.0,-2.0,-2.0,0.5,0.10000000149011612,0.9824999868869781,-2.0,-2.0,0.09050000086426735,-2.0,-2.0,0.883576899766922,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[0.00351,0.022744,0.009966,0.038365,0.615002,-1.080298,-0.029951,-0.402236,0.406625,0.139746,0.179822,1.591931,0.720653,-0.512723,-0.026216,-0.034813,-0.043142,-0.74289,-0.073098,0.082752,1.071722,1.243682,0.075629,1.348505,-0.329277]],[[46,47,81,77,-2,-2,73,-2,-2,71,12,-2,-2,-2,74,68,13,-2,-2,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0,0.8199999928474426,0.8776666522026062,-2.0,-2.0,-2.0,0.8479999899864197,0.9850000143051147,0.35750000178813934,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[0.004596,-0.001897,0.003974,0.011827,0.194762,-0.871368,-0.098384,0.146286,-1.249468,-0.143645,-0.181562,-1.33381,-1.171463,-0.611251,0.119052,0.159252,0.241506,0.922607,1.766618,0.103299,-1.057118]],[[28,9,55,70,-2,-2,-2,11,71,-2,-2,23,-2,-2,-2],[0.5,0.36666667461395264,0.5,0.351500004529953,-2.0,-2.0,-2.0,0.7730000019073486,0.053999999538064,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[-0.001738,0.001452,-0.013598,-0.01848,0.042076,-0.525393,0.849541,0.044411,0.005863,1.05912,-0.549435,0.084166,0.631032,0.96097,-1.312983]],[[17,39,23,34,-2,-2,86,-2,-2,-2,78,70,68,-2,-2,21,-2,-2,73,-2,-2],[0.9549999833106995,0.5,0.5,1.5,-2.0,-2.0,0.13099999725818634,-2.0,-2.0,-2.0,0.5,0.08950000256299973,0.9939999878406525,-2.0,-2.0,0.10000000149011612,-2.0,-2.0,0.019499999471008778,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,-1],[-0.003076,0.018747,0.011473,0.037613,0.401537,1.077311,-0.025743,0.329153,-0.433514,1.166816,-0.029608,-0.037701,0.000931,-0.141249,1.598465,-0.073086,-0.781046,1.024296,0.082069,1.160618,-0.36548]],[[79,88,24,70,-2,-2,11,-2,-2,13,74,-2,-2,13,-2,-2,16,86,-2,18,-2,-2,-2],[0.5,0.33908046782016754,0.5,0.05849999934434891,-2.0,-2.0,0.9895000159740448,-2.0,-2.0,0.060499999672174454,0.013500000350177288,-2.0,-2.0,0.09549999982118607,-2.0,-2.0,1.5,0.03500000014901161,-2.0,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,-1],[0.000798,0.006412,0.018365,0.010335,0.3504,-0.099224,0.121012,0.217145,1.566794,-0.051986,0.057269,1.100849,-0.848516,-0.086436,-1.529641,-0.795027,-0.108204,-0.14735,0.270889,-0.174918,-1.304783,-1.107999,0.125372]],[[12,11,79,3,-2,-2,69,-2,-2,88,4,-2,-2,71,-2,-2,39,23,68,-2,-2,68,-2,-2,72,-2,-2],[0.7716666758060455,0.9284999966621399,0.5,0.005499999970197678,-2.0,-2.0,0.5639999955892563,-2.0,-2.0,0.47727273404598236,0.40833333134651184,-2.0,-2.0,0.09399999864399433,-2.0,-2.0,0.5,0.5,0.4584999978542328,-2.0,-2.0,0.017500000074505806,-2.0,-2.0,0.6860000193119049,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,26,-1,-1],[-0.002312,0.024803,-0.001972,0.009303,0.346434,-0.644707,-0.175345,-1.484202,-0.616746,0.065639,0.07963,0.451888,1.177493,-0.143877,-0.220325,-1.631713,-0.026168,-0.036061,-0.00634,-0.655759,0.367599,-0.077132,1.426095,-0.663025,0.091336,1.323462,-0.061281]],[[37,17,81,77,-2,-2,-2,85,89,-2,-2,87,-2,-2,-2],[1.5,0.9375,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.029411764815449715,-2.0,-2.0,2.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[-0.001038,-0.004436,0.031871,0.039004,0.712609,-0.707593,-0.736226,-0.021987,-0.030898,0.187512,-0.388511,0.064668,0.641562,-0.758107,1.237316]],[[89,17,73,11,-2,-2,13,-2,-2,66,12,-2,-2,-2,72,13,72,-2,-2,74,-2,-2,12,11,-2,-2,46,-2,-2],[0.12916667014360428,0.9850000143051147,0.8644999861717224,0.9284999966621399,-2.0,-2.0,0.9804999828338623,-2.0,-2.0,1.5,0.6693333387374878,-2.0,-2.0,-2.0,0.02850000001490116,0.925000011920929,0.025500000454485416,-2.0,-2.0,0.9699999988079071,-2.0,-2.0,0.3882499933242798,0.05849999934434891,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.002884,0.030261,0.058994,0.038973,-0.091148,0.616644,0.219363,0.726822,2.19331,-0.039164,-0.065289,0.8033,-0.783051,1.854287,-0.019923,0.021431,0.038172,0.296831,1.699602,-0.120521,-1.336553,0.206887,-0.036887,-0.140262,1.206044,-1.288804,-0.025302,-0.333345,0.549285]],[[12,47,11,81,-2,-2,88,-2,-2,72,-2,-2,39,34,85,-2,-2,-2,13,-2,11,-2,-2],[0.7716666758060455,0.5,0.773499995470047,0.5,-2.0,-2.0,0.47727273404598236,-2.0,-2.0,0.05049999989569187,-2.0,-2.0,0.5,4.5,0.5,-2.0,-2.0,-2.0,0.8290000259876251,-2.0,0.99549999833107,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,-1,21,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,20,-1,22,-1,-1],[0.002356,0.024553,0.0321,0.003025,0.232222,-1.070358,0.061421,0.780604,-0.83411,-0.128514,-1.274325,-0.588238,-0.018548,-0.030298,-0.024316,-0.285133,0.514588,-1.330531,0.116814,-0.892024,0.201261,1.584054,0.478971]],[[12,47,11,71,-2,-2,86,-2,-2,12,-2,-2,61,68,75,-2,-2,85,-2,-2,13,-2,89,-2,-2],[0.7716666758060455,0.5,0.773499995470047,0.03350000083446503,-2.0,-2.0,0.921500027179718,-2.0,-2.0,0.539000004529953,-2.0,-2.0,0.5,0.4584999978542328,0.46049998700618744,-2.0,-2.0,0.5,-2.0,-2.0,0.7730000019073486,-2.0,0.1339285746216774,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,22,-1,24,-1,-1],[0.001489,0.029196,0.037119,0.005927,0.963638,-0.299129,0.067926,0.834628,-0.962006,-0.126555,-1.258804,-0.520756,-0.023855,-0.038217,-0.099185,0.049379,-1.084212,-0.012368,-0.221016,0.909834,0.11291,-0.907157,0.166079,1.705375,0.457851]],[[37,89,39,16,-2,-2,-2,72,34,-2,-2,12,-2,-2,-2],[1.5,0.12916667014360428,0.5,0.5,-2.0,-2.0,-2.0,0.02850000001490116,2.5,-2.0,-2.0,0.37300001084804535,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,-1],[0.004906,0.001964,0.037033,0.025461,-0.241926,0.559363,1.001533,-0.015061,0.02788,0.151753,1.321101,-0.033299,-0.986534,-0.207395,1.432145]],[[17,23,14,-2,67,-2,-2,68,11,-2,-2,73,-2,-2,59,68,72,-2,-2,89,-2,-2,69,-2,-2],[0.9725000262260437,0.5,0.17549999803304672,-2.0,0.004500000039115548,-2.0,-2.0,0.019999999552965164,0.9569999873638153,-2.0,-2.0,0.014500000048428774,-2.0,-2.0,0.5,0.9945000112056732,0.26100000739097595,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,0.13099999725818634,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,7,4,-1,6,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[-0.000313,0.018202,0.046418,2.609018,0.041826,0.996998,0.265089,-0.01752,0.172374,0.758583,1.850575,-0.030773,0.020202,-0.699753,-0.0294,-0.03788,-0.049494,-0.009977,-0.803685,0.030162,2.086493,-0.479652,0.076389,1.406494,-0.233791]],[[12,28,79,88,-2,-2,70,-2,-2,-2,39,23,18,-2,-2,68,-2,-2,72,-2,-2],[0.7716666758060455,0.5,0.5,0.7321428656578064,-2.0,-2.0,0.9625000059604645,-2.0,-2.0,-2.0,0.5,0.5,0.9950000047683716,-2.0,-2.0,0.021000000648200512,-2.0,-2.0,0.6860000193119049,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,-1],[-0.001528,0.02181,0.026625,0.033868,0.403381,-1.904765,-0.110042,-1.186559,0.785177,-1.411108,-0.022146,-0.032383,0.000885,0.338851,-0.834448,-0.07308,1.47922,-0.645123,0.097979,1.301882,0.034302]],[[77,88,11,70,-2,-2,72,-2,-2,13,69,-2,-2,12,-2,-2,71,-2,69,69,-2,-2,-2],[0.5,0.3452381044626236,0.9494999945163727,0.06000000052154064,-2.0,-2.0,0.992499977350235,-2.0,-2.0,0.024000000208616257,0.34299999475479126,-2.0,-2.0,0.5611666738986969,-2.0,-2.0,0.007499999832361937,-2.0,0.926499992609024,0.7074999809265137,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,-1],[0.000119,0.007037,0.01718,-0.017522,0.586737,-0.444926,0.040322,0.431554,-0.563517,-0.042777,0.061264,0.546344,1.2444,-0.066659,-1.148,-0.529829,-0.064952,0.055752,-0.078276,-0.092664,-1.111303,-1.262038,-0.142993]],[[89,4,72,86,-2,-2,73,-2,-2,70,68,-2,-2,17,-2,-2,28,86,72,-2,-2,91,-2,-2,-2],[0.12916667014360428,0.1180555559694767,0.11099999770522118,0.6730000078678131,-2.0,-2.0,0.21050000190734863,-2.0,-2.0,0.07649999856948853,0.9905000030994415,-2.0,-2.0,0.9549999833106995,-2.0,-2.0,0.5,0.02949999924749136,0.02650000061839819,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[0.001055,0.028862,-0.042307,0.054279,1.18237,-0.986453,-0.129668,-1.481543,-1.111486,0.056434,0.106466,0.575651,1.865969,0.011709,0.628791,-0.422747,-0.013273,-0.009973,0.041139,0.403238,1.615764,-0.022217,-0.254875,0.556144,-1.207551]],[[86,47,89,11,-2,-2,72,-2,-2,-2,81,14,12,-2,-2,14,-2,-2,11,17,-2,-2,-2],[0.1770000010728836,0.5,0.3452381044626236,0.3840000033378601,-2.0,-2.0,0.04050000011920929,-2.0,-2.0,-2.0,0.5,0.24650000035762787,0.9507499933242798,-2.0,-2.0,0.34950000047683716,-2.0,-2.0,0.9794999957084656,0.9650000035762787,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[-0.001775,0.031037,0.036497,0.054335,0.079089,0.949902,-0.007586,0.360188,-0.882806,-0.773879,-0.019494,-0.013395,0.113314,-0.106868,2.741861,-0.019694,-0.900336,-0.058343,-0.122524,-0.153185,-1.382807,-1.146093,-0.363299]],[[77,88,17,73,-2,-2,68,-2,-2,36,88,-2,-2,3,-2,-2,75,-2,88,4,-2,-2,68,-2,-2],[0.5,0.029411764815449715,0.9850000143051147,0.9165000021457672,-2.0,-2.0,0.9814999997615814,-2.0,-2.0,0.5,0.08712121471762657,-2.0,-2.0,0.007999999914318323,-2.0,-2.0,0.004500000039115548,-2.0,0.3205128312110901,0.5857143104076385,-2.0,-2.0,0.9000000059604645,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,23,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,24,-1,-1],[0.005823,0.013265,0.04817,0.085216,0.642415,2.144585,-0.037474,-0.577342,0.577672,-0.001627,0.003825,-0.996418,0.078034,-0.131887,-0.527226,-1.335227,-0.067624,-0.012294,-0.081732,-0.093268,-1.145121,-1.07187,-0.055777,-1.047374,-1.293386]],[[86,47,77,30,-2,-2,68,-2,-2,12,-2,-2,70,68,14,-2,-2,75,-2,-2,70,-2,68,-2,-2],[0.7120000123977661,0.5,0.5,0.5,-2.0,-2.0,0.4659999907016754,-2.0,-2.0,0.539000004529953,-2.0,-2.0,0.44200000166893005,0.26900000870227814,0.7010000050067902,-2.0,-2.0,0.6705000102519989,-2.0,-2.0,0.5104999840259552,-2.0,0.49799999594688416,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,22,-1,24,-1,-1],[0.002934,0.022138,0.029603,0.039497,0.372564,1.751181,-0.050693,0.360048,-0.819377,-0.113338,-1.251628,-0.756684,-0.022426,-0.002717,-0.055031,-1.111132,-0.171708,0.026011,-0.350236,0.36468,-0.103913,-2.406548,-0.080252,0.620149,-1.155234]],[[77,79,87,35,-2,-2,34,-2,-2,14,-2,71,-2,-2,71,-2,86,-2,4,-2,-2],[0.5,0.5,1.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.38199999928474426,-2.0,0.8770000040531158,-2.0,-2.0,0.006500000134110451,-2.0,0.06549999862909317,-2.0,0.5857143104076385,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[-0.002702,0.003998,0.008747,0.02769,0.117968,1.092653,-0.015768,0.159181,-0.417473,-0.072619,0.83464,-0.11521,-1.183386,-0.190609,-0.06483,0.053126,-0.077575,-1.237624,-0.072346,-1.116833,-1.064477]],[[55,17,81,34,-2,-2,-2,70,70,-2,-2,21,-2,-2,69,-2,-2],[0.5,0.9375,0.5,2.5,-2.0,-2.0,-2.0,0.351500004529953,0.3190000057220459,-2.0,-2.0,0.04545454680919647,-2.0,-2.0,0.7335000038146973,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,-1],[-0.007921,-0.01172,0.020815,0.027753,0.370231,1.365893,-0.797561,-0.027988,-0.007128,-0.11181,1.713092,-0.065077,-0.638438,1.206656,0.112671,0.205955,1.342742]],[[21,63,11,71,-2,-2,67,-2,-2,4,12,-2,-2,-2,74,86,20,-2,-2,-2,70,71,-2,-2,-2],[0.04545454680919647,0.5,0.945499986410141,0.001500000071246177,-2.0,-2.0,0.001500000071246177,-2.0,-2.0,0.5277777910232544,0.6234999895095825,-2.0,-2.0,-2.0,0.25300000607967377,0.8305000066757202,0.5,-2.0,-2.0,-2.0,0.4310000091791153,0.4974999874830246,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[-0.002201,-0.015118,-0.021492,-0.067391,1.071459,-0.724652,0.001443,-0.290368,0.27251,0.092495,0.125541,1.13689,1.11557,0.295506,0.030311,0.037298,0.035135,0.436945,1.031998,1.320107,-0.006552,0.045328,0.942942,-0.290811,-1.546108]],[[86,77,81,17,-2,-2,11,-2,-2,68,-2,69,-2,-2,14,68,-2,14,-2,-2,-2],[0.9975000023841858,0.5,0.5,0.9725000262260437,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,0.3139999955892563,-2.0,0.976500004529953,-2.0,-2.0,0.7540000081062317,0.23749999701976776,-2.0,0.5365000069141388,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,17,-1,19,-1,-1,-1],[0.002471,0.006553,0.013073,0.018912,0.351687,-0.090449,-0.070875,-1.139092,0.849736,-0.057663,0.064961,-0.067142,-1.096473,0.127407,-0.12639,-0.10608,-1.796268,-0.078688,-1.083696,-0.565344,-1.65011]],[[72,88,46,70,-2,-2,75,-2,-2,74,13,-2,-2,85,-2,-2,57,14,75,-2,-2,-2,-2],[0.992499977350235,0.33908046782016754,0.5,0.08950000256299973,-2.0,-2.0,0.05850000120699406,-2.0,-2.0,0.013500000350177288,0.06300000101327896,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.848499983549118,0.99549999833107,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[-0.000812,0.005433,0.014607,0.009195,0.336919,-0.126883,0.136588,1.548319,0.33662,-0.039353,0.015548,0.917351,-0.434739,-0.073452,-0.665234,-1.521961,-0.08278,-0.101509,-0.086989,-0.988511,-1.350084,-1.304944,0.020636]],[[17,34,75,75,-2,-2,11,-2,-2,69,74,-2,-2,-2,12,86,12,-2,-2,71,-2,-2,14,-2,74,-2,-2],[0.9549999833106995,2.5,0.9964999854564667,0.9945000112056732,-2.0,-2.0,0.9975000023841858,-2.0,-2.0,0.8339999914169312,0.6499999761581421,-2.0,-2.0,-2.0,0.7690833210945129,0.6044999957084656,0.6825000047683716,-2.0,-2.0,0.19599999487400055,-2.0,-2.0,0.24649999290704727,-2.0,0.9205000102519989,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,26,-1,-1],[-0.00208,0.013335,0.003083,0.007243,0.019668,1.321074,-0.14846,-0.287252,-1.64041,0.115523,0.157155,1.320709,0.292324,-0.623138,-0.021818,0.016024,0.052638,0.221491,1.7494,-0.073849,-1.201028,-0.16919,-0.041102,1.453603,-0.049669,-0.822815,0.026242]],[[72,34,13,79,-2,-2,68,-2,-2,67,-2,-2,70,68,14,-2,-2,73,-2,-2,55,20,-2,-2,-2],[0.02850000001490116,2.5,0.9214999973773956,0.5,-2.0,-2.0,0.05299999937415123,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,0.341499999165535,0.7080000042915344,0.278999999165535,-2.0,-2.0,0.23499999940395355,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[-0.004249,0.027181,0.01123,0.028944,0.528451,-0.942848,-0.127558,0.103522,-1.192847,0.200215,0.906141,1.702399,-0.015185,0.003601,-0.027234,0.925532,-0.37518,0.037767,-0.036517,0.76911,-0.047289,-0.056534,-0.746106,0.477488,1.055759]],[[17,73,17,81,-2,-2,89,-2,-2,67,68,-2,-2,53,-2,-2,10,71,12,-2,-2,75,-2,-2,70,13,-2,-2,16,-2,-2],[0.9725000262260437,0.18699999898672104,0.9350000023841858,0.5,-2.0,-2.0,0.269696980714798,-2.0,-2.0,0.0035000001080334187,0.3870000094175339,-2.0,-2.0,0.5,-2.0,-2.0,0.3452381044626236,0.019499999471008778,0.6876666843891144,-2.0,-2.0,0.9440000057220459,-2.0,-2.0,0.29500000178813934,0.11749999597668648,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.001194,0.014116,0.000576,0.025669,0.62284,-0.749485,-0.031621,0.041479,-0.762117,0.071143,0.153043,0.043515,1.589279,0.020696,-0.124108,1.155167,-0.025503,-0.042134,-0.076168,1.270646,-0.908269,-0.020426,-0.412201,2.389257,0.036245,0.134279,0.385941,1.824213,-0.019255,-0.800752,0.772479]],[[79,77,87,18,-2,-2,4,-2,-2,11,-2,17,-2,-2,88,71,71,-2,-2,73,-2,-2,-2],[0.5,0.5,1.5,0.9850000143051147,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.47850000858306885,-2.0,0.9249999821186066,-2.0,-2.0,0.39230769872665405,0.8770000040531158,0.39899998903274536,-2.0,-2.0,0.004000000073574483,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[-0.000914,0.003409,0.009511,0.027894,0.505267,-0.218522,-0.015215,0.608988,-0.307597,-0.056507,0.33928,-0.067643,-1.156408,-1.074144,-0.081858,-0.104788,-0.146116,-1.125446,-1.399827,-0.019748,0.378561,-0.802947,0.650688]],[[74,70,68,17,-2,-2,17,-2,-2,72,72,-2,-2,64,-2,-2,70,4,-2,-2,-2],[0.99549999833107,0.08950000256299973,0.9945000112056732,0.9950000047683716,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.024500000290572643,0.019499999471008778,-2.0,-2.0,0.5,-2.0,-2.0,0.001500000071246177,0.47727273404598236,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[0.003961,0.006662,0.02966,0.021894,0.348455,-0.332536,0.164429,-0.083944,2.252865,-0.009542,0.033422,0.319251,1.224297,-0.02608,-0.318143,1.140404,-0.096821,-0.056585,-0.076997,-1.192334,-1.38714]],[[46,23,18,12,-2,-2,21,-2,-2,70,17,-2,-2,73,-2,-2,72,-2,74,4,-2,-2,-2],[0.5,0.5,0.9850000143051147,0.948583334684372,-2.0,-2.0,0.22500000149011612,-2.0,-2.0,0.007500000298023224,0.9850000143051147,-2.0,-2.0,0.015500000212341547,-2.0,-2.0,0.0989999957382679,-2.0,0.8479999899864197,0.5227272808551788,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,22,21,-1,-1,-1],[-0.003301,-0.007665,0.008267,0.030839,0.175893,0.790914,-0.039172,-0.699314,1.203037,-0.030541,0.060309,0.141075,1.895193,-0.044696,-0.139639,-0.817323,0.083864,-0.6128,0.142106,0.185459,1.520875,0.831733,-0.693518]],[[57,9,23,68,-2,-2,68,-2,-2,11,11,-2,-2,23,-2,-2,89,12,12,-2,-2,-2,23,68,-2,-2,14,-2,-2],[0.5,0.4166666716337204,0.5,0.9784999787807465,-2.0,-2.0,0.012500000186264515,-2.0,-2.0,0.6414999961853027,0.17500000447034836,-2.0,-2.0,0.5,-2.0,-2.0,0.2899159789085388,0.9863499999046326,0.9562499821186066,-2.0,-2.0,-2.0,0.5,0.9535000026226044,-2.0,-2.0,0.5094999819993973,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.001354,-0.007036,-0.019179,0.003965,-0.234465,0.972992,-0.049933,1.660685,-0.431189,0.02957,-0.004124,0.375035,-1.069252,0.061764,0.449207,0.982401,0.044219,0.085836,0.066249,1.006576,-0.617979,1.329679,-0.049132,0.004777,-0.902955,1.123382,-0.115686,-0.193721,-1.347297]],[[81,86,37,23,-2,-2,-2,72,-2,14,-2,-2,71,69,69,-2,-2,-2,-2],[0.5,0.9975000023841858,1.5,0.5,-2.0,-2.0,-2.0,0.7820000052452087,-2.0,0.7554999887943268,-2.0,-2.0,0.9255000054836273,0.6969999969005585,0.05250000022351742,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,18,17,16,-1,-1,-1,-1],[0.001733,0.006566,0.010008,0.007432,0.347774,-0.123512,1.125334,-0.091631,0.227714,-0.109374,-1.175133,-1.554235,-0.082394,-0.134139,-0.092127,-0.505427,-1.257863,-1.37997,0.718029]],[[13,75,3,89,-2,-2,87,-2,-2,69,75,-2,-2,72,-2,-2,70,70,34,-2,-2,-2,91,23,-2,-2,-2],[0.09149999916553497,0.042000001296401024,0.001500000071246177,0.5086206793785095,-2.0,-2.0,1.5,-2.0,-2.0,0.7400000095367432,0.08700000122189522,-2.0,-2.0,0.05849999934434891,-2.0,-2.0,0.05849999934434891,0.05049999989569187,2.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[-0.003458,0.031364,0.018105,0.033863,0.863842,-0.575502,-0.021609,0.607335,-0.884458,0.093693,0.023653,0.947448,-0.585726,0.190748,1.614387,0.885643,-0.016063,0.004991,-0.006093,-0.243576,0.799965,1.285623,-0.031485,-0.037143,-0.137521,-0.57257,0.639795]],[[77,88,2,57,-2,-2,72,-2,-2,71,74,-2,-2,-2,68,-2,67,17,-2,-2,-2],[0.5,0.5916666686534882,0.5,0.5,-2.0,-2.0,0.02350000012665987,-2.0,-2.0,0.10349999740719795,0.9830000102519989,-2.0,-2.0,-2.0,0.2604999989271164,-2.0,0.6899999976158142,0.9249999821186066,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,-1],[0.003851,0.009797,0.012573,0.01612,0.072988,0.617534,-0.091444,0.364917,-1.417874,-0.114121,-0.087526,-0.956074,-1.531841,-1.763594,-0.05103,0.101287,-0.05956,-0.067783,-1.02154,-1.106428,-0.010062]],[[89,39,68,70,-2,-2,70,-2,-2,23,-2,-2,75,24,17,-2,-2,-2,70,75,-2,-2,70,-2,-2],[0.20820433646440506,0.5,0.39249999821186066,0.5164999961853027,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,-2.0,-2.0,0.015500000212341547,0.5,0.9350000023841858,-2.0,-2.0,-2.0,0.3254999965429306,0.05650000087916851,-2.0,-2.0,0.5085000097751617,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[0.00183,0.017986,0.010975,-0.038384,-0.713743,0.987341,0.026975,0.559481,0.046401,0.116313,0.366889,1.519033,-0.014507,0.015269,-0.000248,0.713249,-0.461993,1.050027,-0.029992,-0.007494,0.957356,-0.213226,-0.069882,-1.890655,-0.535955]],[[17,73,79,39,-2,-2,71,-2,-2,89,67,-2,-2,-2,12,70,69,-2,-2,11,-2,-2,67,14,-2,-2,12,-2,-2],[0.9741666913032532,0.23349999636411667,0.5,0.5,-2.0,-2.0,0.9169999957084656,-2.0,-2.0,0.563492089509964,0.0005000000237487257,-2.0,-2.0,-2.0,0.7712499797344208,0.3605000078678131,0.4009999930858612,-2.0,-2.0,0.9370000064373016,-2.0,-2.0,0.001500000071246177,0.909500002861023,-2.0,-2.0,0.98887500166893,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.00433,0.01856,0.007283,0.012655,0.093392,0.770315,-0.101798,-1.177352,0.303077,0.065946,0.082815,1.339674,0.495043,-1.369817,-0.018655,0.023222,0.094617,0.133279,1.840449,-0.024915,-0.70875,0.686076,-0.038222,-0.074394,-0.926785,0.894828,-0.01415,-0.664499,0.592537]],[[55,9,39,23,-2,-2,13,-2,-2,2,89,-2,-2,18,-2,-2,87,-2,-2],[0.5,0.36666667461395264,0.5,0.5,-2.0,-2.0,0.7730000019073486,-2.0,-2.0,0.5,0.7071428596973419,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.000114,-0.003445,-0.013311,-0.01869,-0.003539,-0.35521,0.061988,-0.946305,1.112835,0.025748,0.033839,0.560411,-1.73414,-0.080608,1.030068,-1.23453,0.091682,1.23324,-0.071611]],[[23,17,88,24,-2,-2,85,-2,-2,21,68,-2,-2,13,-2,-2,68,75,72,-2,-2,12,-2,-2,68,13,-2,-2,68,-2,-2],[0.5,0.9950000047683716,0.44949495792388916,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.10000000149011612,0.9930000007152557,-2.0,-2.0,0.09050000086426735,-2.0,-2.0,0.02850000001490116,0.08950000256299973,0.03150000050663948,-2.0,-2.0,0.9881666600704193,-2.0,-2.0,0.24050000309944153,0.05599999986588955,-2.0,-2.0,0.28599999845027924,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.002281,0.012956,0.028832,0.038177,0.447515,1.598257,-0.063973,-0.441601,-1.593014,-0.045157,-0.065156,-1.105735,0.421891,0.063195,1.046643,1.38098,-0.021583,0.111078,0.055561,1.175796,-0.17565,0.191269,1.694311,0.237472,-0.029397,-0.116894,1.23534,-1.235226,-0.018829,1.317931,-0.204814]],[[88,34,14,70,-2,-2,69,-2,-2,71,71,-2,-2,71,-2,-2,86,71,75,-2,-2,28,-2,-2,14,18,-2,-2,11,-2,-2],[0.029411764815449715,1.5,0.3674999922513962,0.3464999943971634,-2.0,-2.0,0.2915000021457672,-2.0,-2.0,0.07499999925494194,0.006000000052154064,-2.0,-2.0,0.29500000178813934,-2.0,-2.0,0.7120000123977661,0.005499999970197678,0.004500000039115548,-2.0,-2.0,0.5,-2.0,-2.0,0.24699999392032623,0.9850000143051147,-2.0,-2.0,0.99549999833107,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.001336,0.025439,0.008369,-0.115936,-0.610776,-1.412339,0.020565,-0.359612,0.553999,0.08832,0.175922,-0.098295,1.706411,-0.001528,-1.008974,0.851817,-0.013149,0.004643,0.058601,0.501136,1.140123,-0.008879,-0.052683,-1.154297,-0.040849,0.127418,2.247967,-1.137975,-0.052359,-0.756329,-0.066767]],[[55,23,18,88,-2,-2,66,-2,-2,74,39,-2,-2,72,-2,-2,87,-2,-2],[0.5,0.5,0.9950000047683716,0.44949495792388916,-2.0,-2.0,1.5,-2.0,-2.0,0.9895000159740448,0.5,-2.0,-2.0,0.0669999998062849,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.000416,-0.003255,0.011145,0.026775,0.471165,-0.7682,-0.041351,-0.911455,0.912712,-0.023024,-0.031051,-0.337299,0.62144,0.139961,-0.20219,1.606225,0.093022,1.229636,-0.069672]],[[88,55,23,18,-2,-2,68,-2,-2,69,-2,-2,13,2,88,-2,-2,-2,85,21,-2,-2,72,-2,-2],[0.33908046782016754,0.5,0.5,0.9950000047683716,-2.0,-2.0,0.024500000290572643,-2.0,-2.0,0.8339999914169312,-2.0,-2.0,0.0409999992698431,0.5,0.5527777969837189,-2.0,-2.0,-2.0,0.5,0.10000000149011612,-2.0,-2.0,0.44700001180171967,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[0.005157,0.012522,0.009399,0.024657,0.597814,-0.694964,-0.011744,1.251027,-0.1699,0.109534,0.598262,1.2077,-0.031446,0.041672,0.055804,1.184505,-0.662736,-0.844596,-0.053286,-0.035914,-0.917708,1.140399,-0.169676,-1.523438,-0.663239]],[[86,86,12,12,-2,-2,67,-2,-2,-2,4,17,-2,88,-2,-2,18,75,-2,-2,68,-2,-2],[0.992499977350235,0.9914999902248383,0.9948750138282776,0.9938333332538605,-2.0,-2.0,0.0035000001080334187,-2.0,-2.0,-2.0,0.2613636404275894,0.9600000083446503,-2.0,0.19090909510850906,-2.0,-2.0,0.9750000238418579,0.9964999854564667,-2.0,-2.0,0.9575000107288361,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[0.001266,0.008024,0.005948,0.009658,0.083046,1.242274,-0.095214,-1.194395,-0.60252,1.355366,-0.051995,-0.10662,-1.195257,-0.078368,-1.14439,-1.061463,-0.025589,0.034872,1.186919,-0.622494,-0.06627,-0.663614,-1.201509]],[[77,88,34,69,-2,-2,-2,72,73,-2,-2,69,-2,-2,88,18,-2,10,-2,-2,18,-2,-2],[0.5,0.029411764815449715,2.5,0.2915000021457672,-2.0,-2.0,-2.0,0.9905000030994415,0.9874999821186066,-2.0,-2.0,0.2524999976158142,-2.0,-2.0,0.3166666775941849,0.9249999821186066,-2.0,0.4166666716337204,-2.0,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,18,-1,-1,21,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,17,-1,19,-1,-1,22,-1,-1],[0.003296,0.00873,0.036495,0.023435,-0.256257,0.614776,1.295019,-0.002698,0.003601,0.002583,1.666382,-0.073073,-0.710083,-1.128548,-0.051291,-0.062275,-1.017643,-0.052607,-1.050097,-1.095814,-0.02693,0.251876,-1.162252]],[[81,77,88,79,-2,-2,71,-2,-2,74,-2,17,-2,-2,69,-2,67,-2,11,-2,-2],[0.5,0.5,0.612500011920929,0.5,-2.0,-2.0,0.09750000014901161,-2.0,-2.0,0.001500000071246177,-2.0,0.9249999821186066,-2.0,-2.0,0.035499999299645424,-2.0,0.0035000001080334187,-2.0,0.828000009059906,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[0.000887,0.00657,0.012115,0.014244,0.197655,-0.485397,-0.116905,-1.190686,-1.63431,-0.046889,0.132523,-0.057643,-1.004494,-1.092129,-0.081071,0.188101,-0.09873,-1.458267,-0.077863,-1.180112,-0.237967]],[[9,20,34,1,-2,-2,89,-2,-2,87,11,-2,-2,19,-2,-2,12,75,68,-2,-2,87,-2,-2,23,73,-2,-2,72,-2,-2],[0.36666667461395264,0.5,2.5,0.5,-2.0,-2.0,0.0416666679084301,-2.0,-2.0,5.5,0.26999999582767487,-2.0,-2.0,0.1574999988079071,-2.0,-2.0,0.38974998891353607,0.06200000084936619,0.9695000052452087,-2.0,-2.0,1.5,-2.0,-2.0,0.5,0.929500013589859,-2.0,-2.0,0.0755000002682209,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.001467,-0.012704,-0.020312,-0.02952,-0.337518,0.806496,0.030985,1.471915,-0.027848,0.039931,0.032478,-0.53404,0.835167,0.120106,1.083519,1.363763,0.03293,0.006821,0.022323,0.882349,-0.436516,-0.076443,0.327068,-1.117656,0.059638,0.018671,0.802837,-0.911103,0.113177,1.22697,0.417551]],[[29,37,70,14,-2,-2,86,-2,-2,-2,-2],[1.5,1.5,0.08950000256299973,0.46699999272823334,-2.0,-2.0,0.7129999995231628,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.004454,-0.002502,-0.004356,0.01224,0.592026,-0.095233,-0.016301,-0.004259,-0.627722,0.93362,-1.120678]],[[42,81,70,88,-2,-2,34,-2,-2,68,69,-2,-2,-2,12,-2,-2],[0.5,0.5,0.08950000256299973,0.2613636404275894,-2.0,-2.0,0.5,-2.0,-2.0,0.9709999859333038,0.6969999969005585,-2.0,-2.0,-2.0,0.9258333444595337,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[-0.000866,-0.00354,4.3e-05,0.016432,0.350738,-0.376443,-0.012062,0.141323,-0.450143,-0.057806,-0.100588,-0.723646,-1.33643,0.632272,0.086705,1.220285,-0.123241]],[[85,72,11,73,-2,-2,34,-2,-2,17,-2,69,-2,-2,72,87,11,-2,-2,-2,69,73,-2,-2,-2],[0.5,0.9895000159740448,0.9985000193119049,0.9874999821186066,-2.0,-2.0,0.5,-2.0,-2.0,0.9562499821186066,-2.0,0.9524999856948853,-2.0,-2.0,0.4284999966621399,2.5,0.9524999856948853,-2.0,-2.0,-2.0,0.6965000033378601,0.7355000078678131,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.002896,-0.003014,0.002192,-0.00069,-0.029177,1.233724,0.134678,0.318101,1.324562,-0.053806,0.21259,-0.070687,-0.923793,-1.218001,0.046827,0.010364,0.034578,-0.137317,1.161716,-1.409932,0.141821,0.084347,1.220331,-0.742348,1.539591]],[[53,59,61,23,-2,-2,12,-2,-2,86,10,-2,-2,-2,68,-2,67,72,-2,-2,14,-2,-2],[0.5,0.5,0.5,0.5,-2.0,-2.0,0.9039999842643738,-2.0,-2.0,0.9705000221729279,0.2916666716337204,-2.0,-2.0,-2.0,0.3060000091791153,-2.0,0.05299999937415123,0.22049999982118607,-2.0,-2.0,0.5480000078678131,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[0.006555,0.001769,-0.003216,-0.008324,0.10988,-0.271281,0.058581,-0.473343,1.00641,0.055787,0.034373,-0.456684,1.15442,1.081301,0.066593,-0.856082,0.096564,0.15183,0.433,1.544976,0.009979,0.734446,-1.282803]],[[88,34,79,14,-2,-2,-2,71,71,-2,-2,71,-2,-2,89,21,18,-2,-2,-2,67,4,-2,-2,89,-2,-2],[0.029411764815449715,1.5,0.5,0.38499999046325684,-2.0,-2.0,-2.0,0.07250000163912773,0.013500000350177288,-2.0,-2.0,0.35999999940395355,-2.0,-2.0,0.10577819496393204,0.125,0.9600000083446503,-2.0,-2.0,-2.0,0.0005000000237487257,0.12142857536673546,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,25,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,26,-1,-1],[0.001532,0.025177,0.011112,0.018735,-0.865594,0.402474,-1.011843,0.08659,0.191005,0.603372,1.778266,0.004351,-0.781998,1.110214,-0.008631,-0.111526,-0.151843,-1.2414,-0.685093,1.081487,-0.005129,0.027931,1.84396,0.107129,-0.014066,0.529841,-0.251257]],[[79,35,12,70,-2,-2,69,-2,-2,89,68,-2,-2,89,-2,-2,70,11,-2,-2,11,-2,-2],[0.5,0.5,0.7690833210945129,0.351500004529953,-2.0,-2.0,0.048499999567866325,-2.0,-2.0,0.2899159789085388,0.9884999990463257,-2.0,-2.0,0.3452381044626236,-2.0,-2.0,0.36800000071525574,0.99549999833107,-2.0,-2.0,0.9345000088214874,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.00188,0.002165,-0.004243,0.013395,0.593106,-0.104731,-0.020028,-0.757571,-0.045862,0.048867,0.089579,1.060371,0.023385,-0.070195,-1.227148,-0.575152,-0.092457,-0.136915,-1.444891,-1.048521,-0.053421,-0.914711,0.227988]],[[35,23,18,12,-2,-2,21,-2,-2,68,-2,12,-2,-2,89,68,68,-2,-2,-2,89,-2,69,-2,-2],[0.5,0.5,0.9950000047683716,0.948583334684372,-2.0,-2.0,0.10000000149011612,-2.0,-2.0,0.004500000039115548,-2.0,0.9911249876022339,-2.0,-2.0,0.2899159789085388,0.9869999885559082,0.2279999926686287,-2.0,-2.0,-2.0,0.3452381044626236,-2.0,0.31049999594688416,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,21,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,22,-1,24,-1,-1],[-0.003848,-0.009517,0.00497,0.02082,0.105091,0.565932,-0.046937,-1.09673,1.086166,-0.030948,1.527866,-0.034329,-0.213037,-0.826613,0.038581,0.077546,0.109762,0.311418,1.241091,-0.459915,-0.070314,-1.29517,-0.02293,-1.091762,0.35109]],[[88,11,13,69,-2,-2,-2,14,-2,35,-2,-2,13,3,70,-2,-2,-2,19,86,-2,-2,3,-2,-2],[0.33908046782016754,0.945499986410141,0.9305000007152557,0.009499999694526196,-2.0,-2.0,-2.0,0.17549999803304672,-2.0,0.5,-2.0,-2.0,0.024000000208616257,0.036000000312924385,0.9429999887943268,-2.0,-2.0,-2.0,0.0005000000237487257,0.057499999180436134,-2.0,-2.0,0.020499999169260263,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,7,6,5,-1,-1,-1,9,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[0.001144,0.007553,-0.014469,-0.009526,1.393118,-0.247061,-1.347882,0.022423,1.634625,0.019036,0.107132,0.679774,-0.030295,0.040267,0.024416,0.683855,1.091474,0.949177,-0.043422,-0.05362,-1.08379,-0.686265,0.05601,0.536236,1.38699]],[[12,79,13,9,-2,-2,72,-2,-2,11,72,-2,-2,-2,13,68,-2,10,-2,-2,14,12,-2,-2,39,-2,-2],[0.7716666758060455,0.5,0.34300000965595245,0.19090909510850906,-2.0,-2.0,0.359499990940094,-2.0,-2.0,0.9824999868869781,0.03999999910593033,-2.0,-2.0,-2.0,0.6024999916553497,0.6465000212192535,-2.0,0.2361111119389534,-2.0,-2.0,0.2459999993443489,0.9526250064373016,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.000189,0.017879,0.024182,0.014787,-0.443232,0.397514,0.066838,1.181362,-0.035739,-0.07782,-0.112004,-1.194612,-0.95889,0.459801,-0.016001,-0.095166,-2.590587,-0.048758,-0.931124,0.932022,-0.009785,0.119719,-0.422101,2.035817,-0.015472,-0.221634,0.551408]],[[23,86,18,12,-2,-2,21,-2,-2,68,-2,-2,68,72,14,-2,-2,-2,35,73,-2,-2,89,-2,-2],[0.5,0.9975000023841858,0.9950000047683716,0.948583334684372,-2.0,-2.0,0.10000000149011612,-2.0,-2.0,0.23749999701976776,-2.0,-2.0,0.02850000001490116,0.08300000056624413,0.8620000183582306,-2.0,-2.0,-2.0,0.5,0.019499999471008778,-2.0,-2.0,0.2899159789085388,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,-1,19,20,-1,-1,23,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,17,16,-1,-1,-1,22,21,-1,-1,24,-1,-1],[-0.003356,0.010127,0.014905,0.026073,0.166694,0.663603,-0.027211,-0.784797,1.145976,-0.100035,-1.762304,-1.106984,-0.020723,0.116501,0.064643,0.899614,0.107238,1.620023,-0.028681,-0.043154,-0.171779,-0.815343,0.039398,0.907229,-0.994245]],[[37,11,21,71,-2,-2,74,-2,-2,10,11,-2,-2,23,-2,-2,-2],[1.5,0.945499986410141,0.04545454680919647,0.001500000071246177,-2.0,-2.0,0.4450000077486038,-2.0,-2.0,0.4166666716337204,0.9554999768733978,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.001473,-0.000497,-0.017771,-0.038078,0.917015,-0.51108,0.015524,0.624701,-0.81739,0.010716,-0.000523,1.003169,-0.057454,0.050792,0.523213,0.83009,1.112191]],[[28,81,72,23,-2,-2,69,-2,-2,11,70,-2,-2,-2,3,-2,-2],[0.5,0.5,0.9895000159740448,0.5,-2.0,-2.0,0.9524999856948853,-2.0,-2.0,0.9794999957084656,0.8734999895095825,-2.0,-2.0,-2.0,0.03799999970942736,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,16,-1,-1],[0.000987,0.002864,0.006579,0.011417,0.439202,-0.050486,-0.037439,-0.274917,-1.256778,-0.055306,-0.083382,-1.260863,-0.406307,0.427081,-0.10611,-1.298381,0.206203]],[[12,12,88,47,-2,-2,-2,88,11,-2,-2,-2,4,75,21,-2,-2,69,-2,-2,36,70,-2,-2,-2],[0.6338749825954437,0.5837500095367432,0.7638888955116272,0.5,-2.0,-2.0,-2.0,0.3252032548189163,0.9010000228881836,-2.0,-2.0,-2.0,0.4580419659614563,0.039000000804662704,0.10000000149011612,-2.0,-2.0,0.032500000670552254,-2.0,-2.0,0.5,0.6654999852180481,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,-1],[0.003347,0.023287,0.015613,0.017865,0.370582,-0.678279,-1.703128,0.131279,0.067521,-0.225169,1.168683,1.765685,-0.007336,0.010171,-0.039373,-0.6841,0.751459,0.029938,-0.360882,0.504874,-0.033503,-0.025723,-0.450589,0.347822,-1.353167]],[[12,79,89,45,-2,-2,-2,71,-2,-2,39,4,14,-2,-2,22,-2,-2,13,-2,23,-2,-2],[0.770416647195816,0.5,0.7638888955116272,0.5,-2.0,-2.0,-2.0,0.8770000040531158,-2.0,-2.0,0.5,0.4365079402923584,0.5999999940395355,-2.0,-2.0,0.5,-2.0,-2.0,0.8290000259876251,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,22,-1,-1],[0.001364,0.017541,0.023715,0.025898,0.465406,-0.660548,-1.497937,-0.077097,-1.25771,-0.15964,-0.013216,-0.020515,0.001848,0.434705,-0.452747,-0.047475,-0.434326,-1.892074,0.067068,-0.861851,0.124571,0.527909,1.247688]],[[17,67,73,4,-2,-2,89,-2,-2,18,88,-2,-2,75,-2,-2,11,75,70,-2,-2,68,-2,-2,73,-2,-2],[0.9562499821186066,0.009499999694526196,0.2175000011920929,0.6076923310756683,-2.0,-2.0,0.38750000298023224,-2.0,-2.0,0.9249999821186066,0.24264705926179886,-2.0,-2.0,0.012500000186264515,-2.0,-2.0,0.9985000193119049,0.9675000011920929,0.09100000187754631,-2.0,-2.0,0.9874999821186066,-2.0,-2.0,0.9690000116825104,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,26,-1,-1],[0.002772,0.018141,0.038951,0.021479,0.033728,0.828526,0.131604,1.167072,-0.625478,-0.004809,0.020103,0.79224,-0.488987,-0.055192,-1.280726,-0.198512,-0.016007,-0.019802,-0.011123,0.184088,-0.388219,-0.070723,-0.924951,-1.245595,0.066958,1.612513,-1.069788]],[[37,17,34,79,-2,-2,11,-2,-2,72,74,-2,-2,46,-2,-2,-2],[1.5,0.9350000023841858,1.5,0.5,-2.0,-2.0,0.9745000004768372,-2.0,-2.0,0.2045000046491623,0.11749999970197678,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[-0.003558,-0.00551,0.014313,0.008308,0.290851,-0.993613,0.0878,-0.153344,1.298604,-0.014514,-0.036192,-0.103118,-0.675139,-0.000457,-0.092508,0.785736,1.056721]],[[55,4,34,67,-2,-2,3,-2,-2,73,74,-2,-2,14,-2,-2,87,-2,-2],[0.5,0.464102566242218,2.5,0.010499999858438969,-2.0,-2.0,0.0020000000367872417,-2.0,-2.0,0.001500000071246177,0.9739999771118164,-2.0,-2.0,0.8055000007152557,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.000814,-0.003585,0.006986,0.001054,0.218892,-0.291158,0.070149,1.234911,-0.224783,-0.02031,0.022235,0.14472,1.198095,-0.034833,-0.479024,0.833475,0.072951,1.191507,-0.038215]],[[18,81,34,2,-2,-2,-2,-2,88,4,18,-2,-2,14,-2,-2,12,71,-2,-2,12,-2,-2],[0.9350000023841858,0.5,1.5,0.5,-2.0,-2.0,-2.0,-2.0,0.029411764815449715,0.1180555559694767,0.9650000035762787,-2.0,-2.0,0.8970000147819519,-2.0,-2.0,0.3857499957084656,0.053999999538064,-2.0,-2.0,0.5337499976158142,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.003204,0.017834,0.023287,0.017537,0.540313,-0.930553,1.156676,-0.755938,-0.013199,0.018566,-0.068668,-1.008012,-0.74013,0.050648,0.337108,1.507604,-0.024837,-0.082026,0.07933,-1.171084,-0.018432,0.541755,-0.289531]],[[87,17,14,-2,81,-2,-2,11,67,-2,-2,-2,70,67,67,-2,-2,71,-2,-2,86,68,-2,-2,86,-2,-2],[1.5,0.9850000143051147,0.24400000274181366,-2.0,0.5,-2.0,-2.0,0.9985000193119049,0.001500000071246177,-2.0,-2.0,-2.0,0.28550000488758087,0.10049999877810478,0.0005000000237487257,-2.0,-2.0,0.16200000047683716,-2.0,-2.0,0.022499999962747097,0.9280000030994415,-2.0,-2.0,0.05849999934434891,-2.0,-2.0],[1,2,3,-1,5,-1,-1,8,9,-1,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,7,4,-1,6,-1,-1,11,10,-1,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.002295,0.014889,0.028371,1.585753,0.023404,0.393304,-0.511742,-0.019804,-0.0305,-0.878056,-0.103078,1.023338,-0.014315,0.009241,-0.001773,0.590993,-0.339414,0.117285,-0.449223,1.82119,-0.043013,0.040494,1.188457,0.615923,-0.062566,-1.453183,-0.535387]],[[72,77,72,13,-2,-2,71,-2,-2,71,68,-2,-2,-2,17,14,68,-2,-2,-2,89,89,-2,-2,74,-2,-2],[0.9835000038146973,0.5,0.953000009059906,0.99549999833107,-2.0,-2.0,0.036000000312924385,-2.0,-2.0,0.9895000159740448,0.3034999966621399,-2.0,-2.0,-2.0,0.9850000143051147,0.8680000007152557,0.4345000088214874,-2.0,-2.0,-2.0,0.1547619104385376,0.04545454680919647,-2.0,-2.0,0.988999992609024,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,25,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,26,-1,-1],[0.003342,0.008086,0.013054,0.009231,0.132511,-0.925773,0.072686,-0.136353,1.569897,-0.035258,-0.042032,0.038552,-1.058343,0.261371,-0.035317,0.016369,-0.021292,-1.013384,0.576622,1.537074,-0.061853,-0.092011,-0.96386,-1.140723,-0.042193,-1.047077,-1.112297]],[[87,35,34,86,-2,-2,69,-2,-2,18,72,-2,-2,-2,34,88,75,-2,-2,2,-2,-2,74,88,-2,-2,86,-2,-2],[1.5,0.5,2.5,0.6474999785423279,-2.0,-2.0,0.6159999966621399,-2.0,-2.0,0.9850000143051147,0.10999999940395355,-2.0,-2.0,-2.0,0.5,0.23303167521953583,0.7714999914169312,-2.0,-2.0,0.5,-2.0,-2.0,0.007500000298023224,0.22777777910232544,-2.0,-2.0,0.13300000131130219,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.001331,0.009988,0.002571,-0.004378,0.297493,-0.35457,0.083231,0.261789,1.218837,0.067177,0.048638,0.21113,1.13669,1.012323,-0.017431,0.005918,0.077394,0.518392,1.48746,-0.008611,-0.004017,-0.939002,-0.045651,-0.120205,-1.516131,-0.658905,-0.023816,0.687026,-0.481364]],[[46,81,70,70,-2,-2,73,-2,-2,11,69,-2,-2,73,-2,-2,69,87,11,-2,-2,-2,-2],[0.5,0.5,0.351500004529953,0.3244999945163727,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.9794999957084656,0.6969999969005585,-2.0,-2.0,0.005500000203028321,-2.0,-2.0,0.968500018119812,2.5,0.9819999933242798,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[0.000812,-0.002578,0.000742,0.011004,0.081205,1.790959,-0.013783,0.277976,-0.552417,-0.049776,-0.078956,-0.541149,-1.193673,0.021407,1.029837,-1.233027,0.064434,0.026644,0.080215,0.311827,1.276456,-1.020089,1.393594]],[[20,46,71,71,-2,-2,74,-2,-2,69,87,-2,-2,-2,87,79,74,-2,-2,-2,74,14,-2,-2,-2],[0.5,0.5,0.7770000100135803,0.5185000002384186,-2.0,-2.0,0.02650000061839819,-2.0,-2.0,0.9134999811649323,2.5,-2.0,-2.0,-2.0,4.5,0.5,0.05199999921023846,-2.0,-2.0,-2.0,0.02199999988079071,0.3659999966621399,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,21,22,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,24,23,-1,-1,-1],[0.000899,-0.005543,-0.00963,-0.002017,-0.088573,0.733294,-0.05307,0.000546,-1.324497,0.049269,0.005872,0.504966,-1.016634,1.319457,0.028681,0.025325,0.02426,0.942102,0.858598,0.926548,0.091954,0.066162,1.051605,1.110766,1.26107]],[[37,29,77,72,-2,-2,71,-2,-2,-2,-2],[1.5,1.5,0.5,0.9895000159740448,-2.0,-2.0,0.9865000247955322,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,-1],[-0.002028,-0.00366,-0.002257,0.001429,0.068053,-0.497735,-0.038307,-0.976758,0.264016,-0.882262,1.04712]],[[72,34,74,12,-2,-2,72,-2,-2,68,86,-2,-2,-2,70,67,71,-2,-2,75,-2,-2,16,-2,-2],[0.6545000076293945,2.5,0.9905000030994415,0.9908333420753479,-2.0,-2.0,0.11349999904632568,-2.0,-2.0,0.99549999833107,0.7045000195503235,-2.0,-2.0,-2.0,0.8815000057220459,0.006500000134110451,0.030500000342726707,-2.0,-2.0,0.41200000047683716,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,24,-1,-1],[-0.004722,0.003959,-0.002551,-0.005595,0.001299,-1.002303,0.134715,-0.177277,1.501554,0.063032,0.087531,0.406508,1.112099,-0.935255,-0.025387,-0.01456,0.009248,-0.275444,0.729386,-0.04966,0.051909,-0.920878,-0.120375,-0.943069,-1.615935]],[[53,37,17,81,-2,-2,71,-2,-2,89,89,-2,-2,-2,69,69,-2,-2,13,11,-2,-2,55,-2,-2],[0.5,0.5,0.9375,0.5,-2.0,-2.0,0.019499999471008778,-2.0,-2.0,0.2899159789085388,0.1180555559694767,-2.0,-2.0,-2.0,0.10850000008940697,0.02850000001490116,-2.0,-2.0,0.45000000298023224,0.9544999897480011,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,19,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.001526,-0.002598,-0.006388,0.013634,0.454113,-0.632605,-0.017388,-0.403153,-0.041026,0.045743,0.071773,0.732622,1.064112,-0.843025,0.052734,-0.048472,0.026956,-1.140629,0.085301,0.01772,-0.518167,0.952744,0.178324,1.391968,1.023532]],[[85,68,41,68,-2,-2,72,-2,-2,14,88,-2,-2,74,-2,-2,72,87,4,-2,-2,-2,88,16,-2,-2,-2],[0.5,0.9575000107288361,0.5,0.929500013589859,-2.0,-2.0,0.04899999871850014,-2.0,-2.0,0.3330000042915344,0.10555555671453476,-2.0,-2.0,0.38350000977516174,-2.0,-2.0,0.40699999034404755,2.5,0.5857143104076385,-2.0,-2.0,-2.0,0.45299145579338074,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,-1],[0.000547,-0.003763,-0.012325,-0.01689,-0.148136,-0.936433,0.064111,0.027488,1.219986,0.011641,-0.059483,-1.445788,-0.341179,0.027741,0.087869,0.836247,0.034338,0.008272,0.026602,0.111872,1.025487,-1.239615,0.101655,0.156548,1.51351,0.628879,-1.195976]],[[86,79,86,71,-2,-2,74,-2,-2,70,-2,50,-2,-2,75,-2,12,4,-2,-2,14,-2,-2],[0.652999997138977,0.5,0.590499997138977,0.014500000048428774,-2.0,-2.0,0.153499998152256,-2.0,-2.0,0.38100001215934753,-2.0,0.5,-2.0,-2.0,0.0025000000605359674,-2.0,0.9404166638851166,0.48076923191547394,-2.0,-2.0,0.25699999928474426,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,18,-1,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,20,19,-1,-1,22,-1,-1],[0.001769,0.013545,0.018475,0.01426,0.696506,0.045537,0.119351,0.631622,1.786441,-0.07061,-1.32161,-0.030794,1.173741,-0.676357,-0.012887,-1.376898,-0.006903,-0.036699,-0.173231,-0.912647,0.010399,1.876314,0.009677]],[[14,12,72,16,-2,-2,70,-2,-2,-2,14,11,-2,50,-2,-2,46,39,-2,-2,69,-2,-2],[0.2580000013113022,0.9526250064373016,0.2929999977350235,1.5,-2.0,-2.0,0.27799999713897705,-2.0,-2.0,-2.0,0.3110000044107437,0.5979999899864197,-2.0,0.5,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.7884999811649323,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.003804,0.049052,0.015084,0.05217,0.583124,1.167489,-0.05604,-1.286542,-0.140361,1.731221,-0.007655,-0.064849,-1.429316,-0.04914,0.585442,-0.946003,-0.003399,-0.007175,-0.148529,0.568814,0.065616,-0.123701,1.273226]],[[79,88,11,11,-2,-2,53,-2,-2,75,14,-2,-2,71,-2,-2,88,71,71,-2,-2,88,-2,-2,-2],[0.5,0.33908046782016754,0.945499986410141,0.9185000061988831,-2.0,-2.0,0.5,-2.0,-2.0,0.0035000001080334187,0.570499986410141,-2.0,-2.0,0.9370000064373016,-2.0,-2.0,0.39230769872665405,0.875,0.39899998903274536,-2.0,-2.0,0.1547619104385376,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,-1],[0.003226,0.00612,0.012072,-0.00702,0.025577,-0.93503,0.023847,0.17384,1.017439,-0.023323,0.035232,1.239878,-0.506205,-0.040638,-0.559103,-1.46371,-0.056622,-0.074954,-0.106658,-1.07016,-1.333882,-0.013528,0.38816,-0.957137,0.522945]],[[72,73,35,68,-2,-2,88,-2,-2,-2,14,69,57,-2,-2,-2,18,-2,-2],[0.9895000159740448,0.9874999821186066,0.5,0.9535000026226044,-2.0,-2.0,0.25357143580913544,-2.0,-2.0,-2.0,0.8680000007152557,0.9705000221729279,0.5,-2.0,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,18,-1,-1],[0.001136,0.004657,0.003367,-0.001391,-0.201387,0.281065,0.037746,0.850685,-0.92705,1.052652,-0.038046,-0.05641,-0.045217,-1.070367,0.405148,-1.213769,0.059632,1.31427,-1.092221]],[[17,74,34,69,-2,-2,72,-2,-2,-2,21,70,68,-2,-2,67,-2,-2,13,89,-2,-2,14,-2,-2],[0.9549999833106995,0.9964999854564667,2.5,0.3894999921321869,-2.0,-2.0,0.0625,-2.0,-2.0,-2.0,0.04545454680919647,0.08950000256299973,0.9945000112056732,-2.0,-2.0,0.7439999878406525,-2.0,-2.0,0.07250000163912773,0.4365079402923584,-2.0,-2.0,0.3825000077486038,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.005373,0.019114,0.021333,0.015829,-0.063467,0.469972,0.08127,1.215799,0.227696,-1.465675,-0.01192,-0.016313,0.003852,-0.09093,0.93763,-0.035183,-0.693784,0.369959,0.051698,0.026923,1.041702,-0.190371,0.147259,1.129666,1.379499]],[[81,77,3,29,-2,-2,67,-2,-2,71,74,-2,-2,-2,70,71,-2,88,-2,-2,-2],[0.5,0.5,0.007500000298023224,0.5,-2.0,-2.0,0.09699999913573265,-2.0,-2.0,0.9865000247955322,0.04050000011920929,-2.0,-2.0,-2.0,0.9325000047683716,0.018000000156462193,-2.0,0.2967033088207245,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,-1],[-0.002223,0.001835,0.005886,0.014456,0.242975,-0.647075,-0.019449,-0.437887,0.786319,-0.03759,-0.044082,-1.117169,-0.976894,0.219182,-0.057829,-0.075207,-0.192171,-0.098156,-1.224266,-1.046907,0.053619]],[[12,11,28,71,-2,-2,-2,75,75,-2,-2,34,-2,-2,72,23,-2,88,-2,-2,17,0,-2,-2,68,-2,-2],[0.8136666715145111,0.9774999916553497,0.5,0.019499999471008778,-2.0,-2.0,-2.0,0.054500000551342964,0.005499999970197678,-2.0,-2.0,0.5,-2.0,-2.0,0.012500000186264515,0.5,-2.0,0.2792207896709442,-2.0,-2.0,0.9725000262260437,0.5,-2.0,-2.0,0.9945000112056732,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,17,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[-0.000499,0.013361,0.000432,0.004623,0.635243,-0.121953,-1.223427,0.041599,0.015663,0.848846,-0.687191,0.106603,0.516273,1.315401,-0.014158,-0.083634,1.091988,-0.111748,-1.245207,-0.574555,-0.009432,0.011907,0.023776,1.466594,-0.030468,-0.67201,0.655739]],[[47,14,4,70,-2,-2,88,-2,-2,14,70,-2,-2,29,-2,-2,87,67,-2,-2,-2],[0.5,0.41850000619888306,0.4365079402923584,0.0819999985396862,-2.0,-2.0,0.2158385068178177,-2.0,-2.0,0.4675000011920929,0.006000000052154064,-2.0,-2.0,1.5,-2.0,-2.0,2.5,0.006500000134110451,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,20,19,-1,-1,-1],[-0.002186,0.000273,-0.018303,0.004208,0.615268,-0.368954,-0.051589,-1.141688,-0.179281,0.007137,0.050218,1.952465,0.456477,0.001041,0.037539,-1.209623,-0.067184,-0.085027,-0.533615,-1.149106,-0.399987]],[[12,70,4,68,-2,-2,18,-2,-2,86,11,-2,-2,86,-2,-2,17,-2,89,-2,-2],[0.9974166750907898,0.09100000187754631,0.696153849363327,0.7080000042915344,-2.0,-2.0,0.9000000059604645,-2.0,-2.0,0.7470000088214874,0.8030000030994415,-2.0,-2.0,0.7950000166893005,-2.0,-2.0,0.9650000035762787,-2.0,0.2928571552038193,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,19,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,20,-1,-1],[-0.000219,0.001479,0.016913,0.024625,0.075097,0.530257,-0.051766,0.939548,-0.806919,-0.00927,0.001703,-0.329998,0.262205,-0.038819,-2.026948,-0.295325,-0.076954,-1.574751,-0.048639,-1.089516,-1.038265]],[[79,3,39,3,-2,-2,73,-2,-2,86,86,-2,-2,68,-2,-2,88,70,-2,67,-2,-2,-2],[0.5,0.007500000298023224,0.5,0.004500000039115548,-2.0,-2.0,0.007499999832361937,-2.0,-2.0,0.6234999895095825,0.051500000059604645,-2.0,-2.0,0.32350000739097595,-2.0,-2.0,0.3798076957464218,0.36800000071525574,-2.0,0.3804999962449074,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,-1],[-0.003181,4.9e-05,0.007062,0.002989,0.005894,0.675787,0.075435,1.209488,0.341458,-0.022179,0.021078,-1.121451,0.535506,-0.050011,-1.767968,-0.39802,-0.064228,-0.081057,-1.235586,-0.054787,-1.058706,-0.242999,0.687506]],[[29,77,12,70,-2,-2,17,-2,-2,69,-2,69,-2,-2,-2],[1.5,0.5,0.9974166750907898,0.351500004529953,-2.0,-2.0,0.9650000035762787,-2.0,-2.0,0.060499999672174454,-2.0,0.926499992609024,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.00084,0.002073,0.005632,0.007481,0.20726,-0.104756,-0.077303,-1.524667,-1.070296,-0.032112,0.084564,-0.040648,-1.098083,0.057949,-1.1691]],[[86,72,86,74,-2,-2,-2,-2,11,14,18,-2,-2,72,-2,-2,46,39,-2,-2,74,-2,-2],[0.02949999924749136,0.02650000061839819,0.022499999962747097,0.0025000000605359674,-2.0,-2.0,-2.0,-2.0,0.38350000977516174,0.7354999780654907,0.9600000083446503,-2.0,-2.0,0.07099999859929085,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.8554999828338623,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,11,-1,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[8,7,6,5,-1,-1,-1,-1,16,13,12,-1,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.002975,0.020234,0.012365,0.021372,0.271718,1.067505,-1.142147,1.480303,-0.008159,-0.052482,-0.113634,-1.248994,-0.696798,0.018721,1.040176,-0.169848,-0.002824,-0.00669,-0.133461,0.520666,0.062826,0.821224,-0.855817]],[[74,74,12,12,-2,-2,70,-2,-2,-2,73,-2,69,-2,-2],[0.99549999833107,0.9945000112056732,0.9895833134651184,0.98887500166893,-2.0,-2.0,0.03399999998509884,-2.0,-2.0,-2.0,0.2225000038743019,-2.0,0.02199999988079071,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,-1,13,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,12,-1,14,-1,-1],[-0.001635,0.00026,-0.00089,0.00361,0.019649,1.001331,-0.029974,0.062845,-0.914278,0.915765,-0.082492,-1.33844,-0.06011,-0.673681,-1.257343]],[[12,11,71,86,-2,-2,23,-2,-2,75,75,-2,-2,50,-2,-2,73,57,88,-2,-2,68,-2,-2,72,-2,68,-2,-2],[0.7716666758060455,0.9835000038146973,0.02850000001490116,0.6649999916553497,-2.0,-2.0,0.5,-2.0,-2.0,0.04350000061094761,0.005499999970197678,-2.0,-2.0,0.5,-2.0,-2.0,0.23349999636411667,0.5,0.15192308276891708,-2.0,-2.0,0.679500013589859,-2.0,-2.0,0.3034999966621399,-2.0,0.99549999833107,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,26,-1,28,-1,-1],[-0.001773,0.010438,-0.000129,0.040398,0.958329,-1.042683,-0.014752,0.282892,-0.475075,0.041322,0.012783,0.842239,-0.778271,0.121974,-0.337467,1.264981,-0.012912,-0.031343,-0.045562,-0.765914,-0.176932,0.047763,1.088802,0.000933,0.004981,1.561349,-0.001728,-0.120941,1.032353]],[[18,26,73,73,-2,-2,68,-2,-2,-2,66,59,16,-2,-2,69,-2,-2,12,19,-2,-2,-2],[0.9750000238418579,0.5,0.18699999898672104,0.024000000208616257,-2.0,-2.0,0.3930000066757202,-2.0,-2.0,-2.0,1.5,0.5,1.5,-2.0,-2.0,0.09300000220537186,-2.0,-2.0,0.9757142961025238,0.0800000000745058,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,-1],[0.000545,0.009162,0.005484,-0.004078,0.08987,-0.591653,0.042946,-0.094136,0.731894,1.085774,-0.013444,-0.021234,-0.026644,-0.646935,0.044068,0.044445,1.06806,-0.41618,0.045051,0.017322,0.184148,1.395706,1.497319]],[[17,81,17,2,-2,-2,70,-2,-2,-2,69,69,73,-2,-2,17,-2,-2,88,14,-2,-2,14,-2,-2],[0.9375,0.5,0.8350000083446503,0.5,-2.0,-2.0,0.5419999957084656,-2.0,-2.0,-2.0,0.9504999816417694,0.46150000393390656,0.06650000065565109,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.3252032548189163,0.3279999941587448,-2.0,-2.0,0.4819999933242798,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,20,-1,-1,23,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.000976,0.013864,0.017533,0.007828,0.405577,-0.961564,0.062676,0.57965,1.33029,-0.600493,-0.008066,-0.001303,-0.016808,-0.464838,0.112681,0.035298,0.824627,0.086847,-0.03835,-0.066887,-1.300134,-0.440865,0.093724,-0.282051,1.519189]],[[81,77,88,73,-2,-2,72,-2,-2,69,69,-2,-2,-2,86,72,-2,-2,72,-2,11,-2,-2],[0.5,0.5,0.33908046782016754,0.06650000065565109,-2.0,-2.0,0.006500000134110451,-2.0,-2.0,0.926499992609024,0.723499983549118,-2.0,-2.0,-2.0,0.14350000023841858,0.06149999983608723,-2.0,-2.0,0.16700000315904617,-2.0,0.9780000150203705,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,-1,19,-1,21,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,18,17,-1,-1,20,-1,22,-1,-1],[0.003637,0.006592,0.010265,0.015942,0.056472,0.375263,-0.019492,1.04053,-0.529957,-0.031253,-0.039762,-0.849649,-1.137499,0.244482,-0.039102,0.027853,-0.304346,0.744764,-0.071514,-1.346878,-0.054885,-1.115144,-0.395506]],[[74,74,70,86,-2,-2,12,-2,-2,-2,4,73,11,-2,-2,18,-2,-2,89,-2,68,-2,-2],[0.9504999816417694,0.9444999992847443,0.03749999962747097,0.9884999990463257,-2.0,-2.0,0.9746666848659515,-2.0,-2.0,-2.0,0.5505050718784332,0.011500000022351742,0.9909999966621399,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.2361111119389534,-2.0,0.021499999798834324,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,22,-1,-1],[-0.001797,0.002385,-2.1e-05,0.025123,0.231825,1.03276,-0.007348,-0.000125,-0.578935,1.619373,-0.030225,-0.051076,-0.135027,0.096654,-1.380115,-0.031395,0.435463,-0.898318,0.024672,1.050901,-0.049768,0.959976,-0.978291]],[[34,69,14,72,-2,-2,57,-2,-2,69,-2,14,-2,-2,68,72,-2,-2,4,88,-2,-2,69,-2,-2],[2.5,0.9255000054836273,0.2615000009536743,0.8504999876022339,-2.0,-2.0,0.5,-2.0,-2.0,0.929500013589859,-2.0,0.33550000190734863,-2.0,-2.0,0.50450000166893,0.4624999910593033,-2.0,-2.0,0.44949495792388916,0.1678321734070778,-2.0,-2.0,0.965499997138977,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,-1,-1,19,20,-1,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,18,17,-1,-1,22,21,-1,-1,24,-1,-1],[-0.00042,-0.004565,-0.011394,0.04366,0.149695,1.737988,-0.015791,-0.294786,0.297018,0.017874,2.032604,0.010168,-0.881751,0.43505,0.032519,0.126193,1.381072,0.491565,0.004897,0.065512,1.139468,-0.57774,-0.031471,0.037706,-1.065221]],[[72,11,48,86,-2,-2,-2,39,73,-2,-2,68,-2,-2,69,14,17,-2,-2,67,-2,-2,-2],[0.9895000159740448,0.945499986410141,0.5,0.8020000159740448,-2.0,-2.0,-2.0,0.5,0.1914999932050705,-2.0,-2.0,0.7279999852180481,-2.0,-2.0,0.9524999856948853,0.8549999892711639,0.9600000083446503,-2.0,-2.0,0.0005000000237487257,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[5.8e-05,0.004581,-0.008431,-0.004817,0.072635,-0.651835,-0.987865,0.014414,0.009924,-0.037344,0.464241,0.097441,1.25541,0.57055,-0.045099,-0.036786,-0.023482,0.877885,-1.063848,-0.081612,-1.307593,-0.827967,-0.995821]],[[37,12,79,11,-2,-2,71,-2,-2,39,72,-2,-2,88,-2,-2,-2],[1.5,0.7603500187397003,0.5,0.5645000040531158,-2.0,-2.0,0.8645000159740448,-2.0,-2.0,0.5,0.20400000363588333,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.000367,-0.001068,0.010403,0.015058,-0.162066,0.499288,-0.076054,-1.19644,-0.539597,-0.011276,-0.016162,-0.491437,-0.041073,0.048183,1.046514,-0.147438,1.013035]],[[72,34,86,79,-2,-2,4,-2,-2,13,-2,-2,88,4,86,-2,-2,4,-2,-2,23,68,-2,-2,69,-2,-2],[0.02850000001490116,2.5,0.8759999871253967,0.5,-2.0,-2.0,0.5857143104076385,-2.0,-2.0,0.7995000183582306,-2.0,-2.0,0.029411764815449715,0.1180555559694767,0.1119999997317791,-2.0,-2.0,0.3798076957464218,-2.0,-2.0,0.5,0.7675000131130219,-2.0,-2.0,0.003999999957159162,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.00107,0.019169,0.009391,0.018996,0.565964,-0.841041,-0.098873,-1.196899,-0.000289,0.109484,0.479896,1.471847,-0.005542,0.017874,-0.035426,1.262033,-0.951195,0.037198,0.937216,0.06266,-0.015784,-0.000243,-0.351034,0.353605,-0.03609,1.473886,-0.490395]],[[88,85,70,81,-2,-2,75,-2,-2,72,87,-2,-2,89,-2,-2,88,12,67,-2,-2,-2,-2],[0.5916666686534882,0.5,0.08950000256299973,0.5,-2.0,-2.0,0.40549999475479126,-2.0,-2.0,0.4284999966621399,1.5,-2.0,-2.0,0.3541666716337204,-2.0,-2.0,0.8166666626930237,0.9903749823570251,0.005500000203028321,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[0.000661,0.00222,-0.001716,0.010785,0.186896,-0.968261,-0.011265,0.007127,-0.586786,0.031406,0.004333,0.657321,-0.661002,0.098305,1.388018,0.148098,-0.069083,-0.053198,-0.035945,-0.709914,-1.049474,-1.348335,-1.594041]],[[72,73,77,68,-2,-2,11,-2,-2,-2,17,69,-2,-2,69,18,-2,-2,-2],[0.9895000159740448,0.9874999821186066,0.5,0.9575000107288361,-2.0,-2.0,0.47850000858306885,-2.0,-2.0,-2.0,0.9562499821186066,0.009500000160187483,-2.0,-2.0,0.9524999856948853,0.9850000143051147,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,-1,15,16,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,14,13,-1,-1,18,17,-1,-1,-1],[0.001913,0.005454,0.00421,0.007892,-0.022332,0.308413,-0.028777,0.241149,-0.851981,1.08884,-0.032195,0.041433,-1.162317,0.649519,-0.049583,-0.038599,-0.284823,-1.069829,-1.167202]],[[77,68,74,75,-2,-2,39,-2,-2,86,74,-2,-2,11,-2,-2,10,18,-2,13,-2,-2,-2],[0.5,0.9575000107288361,0.34849999845027924,0.14100000262260437,-2.0,-2.0,0.5,-2.0,-2.0,0.9794999957084656,0.9065000116825104,-2.0,-2.0,0.9975000023841858,-2.0,-2.0,0.5833333432674408,0.9249999821186066,-2.0,0.398499995470047,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,-1],[-0.000302,0.003016,-0.006214,0.006548,-0.118442,0.406137,-0.023533,-0.396252,1.073889,0.018065,0.028714,0.264195,2.269516,-0.051739,-1.062314,0.72417,-0.031825,-0.037003,-0.939387,-0.031487,-1.122586,-1.042604,0.048965]],[[81,72,77,79,-2,-2,87,-2,-2,3,74,-2,-2,-2,68,72,-2,14,-2,-2,-2],[0.5,0.9964999854564667,0.5,0.5,-2.0,-2.0,2.5,-2.0,-2.0,0.001500000071246177,0.988999992609024,-2.0,-2.0,-2.0,0.9709999859333038,0.12350000068545341,-2.0,0.55799999833107,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,-1],[0.001763,0.004497,0.006481,0.009901,0.171613,-0.443874,-0.023864,-0.859812,0.355804,-0.052553,-0.03529,-0.547702,-1.09462,-1.190056,-0.040775,-0.067578,-1.32746,-0.040027,0.284136,-1.108563,0.298875]],[[89,47,86,11,-2,-2,14,-2,-2,74,-2,-2,14,88,-2,72,-2,-2,28,70,-2,-2,-2],[0.343137264251709,0.5,0.21149999648332596,0.41449999809265137,-2.0,-2.0,0.36149999499320984,-2.0,-2.0,0.09449999779462814,-2.0,-2.0,0.22849999368190765,0.39047619700431824,-2.0,0.17099999263882637,-2.0,-2.0,0.5,0.715499997138977,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,-1,16,-1,-1,19,20,-1,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,18,15,-1,17,-1,-1,22,21,-1,-1,-1],[-0.001278,0.004571,0.007733,0.024693,-0.197879,0.690333,-0.001825,-0.615885,0.094234,-0.060894,-0.600514,-1.126374,-0.018632,0.058266,2.11649,-0.003768,1.051598,-0.837209,-0.024125,-0.020344,-0.606592,0.195264,-1.080946]],[[17,34,72,75,-2,-2,67,-2,-2,68,67,-2,-2,-2,71,11,-2,73,-2,-2,70,-2,88,-2,-2],[0.9549999833106995,2.5,0.2135000005364418,0.017500000074505806,-2.0,-2.0,0.005499999970197678,-2.0,-2.0,0.972000002861023,0.0035000001080334187,-2.0,-2.0,-2.0,0.3869999945163727,0.34450000524520874,-2.0,0.0035000001080334187,-2.0,-2.0,0.3695000112056732,-2.0,0.0839160867035389,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,-1,18,-1,-1,21,-1,23,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,17,-1,19,-1,-1,22,-1,24,-1,-1],[0.000306,0.010706,0.004641,-0.007401,0.445328,-0.516426,0.027993,0.715303,-0.081037,0.074633,0.111193,0.957726,1.233336,-0.427152,-0.013064,-0.024394,0.619495,-0.029022,-0.903644,-0.318915,0.014416,1.198363,-0.005916,-1.083084,0.130546]],[[11,21,71,89,-2,-2,55,-2,-2,73,12,-2,-2,-2,13,88,75,-2,-2,71,-2,-2,61,26,-2,-2,23,-2,-2],[0.945499986410141,0.04545454680919647,0.030499999411404133,0.2632978707551956,-2.0,-2.0,0.5,-2.0,-2.0,0.9180000126361847,0.8224374949932098,-2.0,-2.0,-2.0,0.40949998795986176,0.47727273404598236,0.04350000061094761,-2.0,-2.0,0.09399999864399433,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.000707,-0.012598,-0.028891,0.019884,0.658539,-0.429871,-0.048691,-0.808629,0.515436,0.015508,0.018413,0.484567,1.089097,-0.679942,0.007005,0.034086,0.044065,0.522019,0.973117,-0.088039,-0.06209,-1.39902,-0.002238,-0.008106,-0.140776,0.849711,0.053166,0.259978,1.025553]],[[12,12,11,72,-2,-2,53,-2,-2,-2,11,74,-2,72,-2,-2,72,69,-2,-2,18,-2,-2],[0.9802500009536743,0.9785833358764648,0.45350000262260437,0.044999999925494194,-2.0,-2.0,0.5,-2.0,-2.0,-2.0,0.9905000030994415,0.49549999833106995,-2.0,0.8325000107288361,-2.0,-2.0,0.09449999779462814,0.27299998700618744,-2.0,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,17,18,-1,-1,21,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,20,19,-1,-1,22,-1,-1],[-0.000752,0.006655,0.004307,-0.017019,0.20631,-0.684087,0.010627,0.094851,0.530198,1.393484,-0.023491,-0.100862,-1.501835,-0.033127,0.391335,-1.037456,-0.009029,-0.083141,-1.324523,-0.168935,0.003865,0.363806,-0.385505]],[[55,87,17,73,-2,-2,72,-2,-2,86,72,-2,-2,14,-2,-2,67,-2,-2],[0.5,0.5,0.9950000047683716,0.9079999923706055,-2.0,-2.0,0.3564999997615814,-2.0,-2.0,0.02949999924749136,0.02650000061839819,-2.0,-2.0,0.17750000208616257,-2.0,-2.0,0.011500000022351742,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.002197,0.000284,0.014383,0.027314,0.180501,1.119909,-0.03909,0.747616,-1.045895,-0.005836,0.021181,0.276515,1.406833,-0.011737,0.862095,-0.180356,0.052341,0.002658,1.14518]],[[59,23,18,74,-2,-2,21,-2,-2,68,-2,11,-2,-2,71,-2,12,57,-2,-2,-2],[0.5,0.5,0.9950000047683716,0.8735000193119049,-2.0,-2.0,0.10000000149011612,-2.0,-2.0,0.009499999694526196,-2.0,0.37700000405311584,-2.0,-2.0,0.001500000071246177,-2.0,0.961208313703537,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,-1,17,18,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,16,-1,20,19,-1,-1,-1],[0.000915,-0.001954,0.005926,0.014723,0.120502,0.589723,-0.025641,-0.822939,1.067497,-0.013207,1.08497,-0.017496,-0.764091,-0.099182,0.03284,1.258332,0.021598,0.033173,0.947891,0.396875,-0.426887]],[[42,38,37,68,-2,-2,-2,-2,13,-2,-2],[0.5,0.5,1.5,0.02850000001490116,-2.0,-2.0,-2.0,-2.0,0.8540000021457672,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,-1,-1],[8,7,6,5,-1,-1,-1,-1,10,-1,-1],[-0.001024,-0.002627,-0.001391,-0.002596,0.448187,-0.081399,0.925209,-0.84176,0.056855,1.000843,-0.053225]],[[28,88,14,69,-2,-2,69,-2,-2,88,18,-2,-2,70,-2,-2,-2],[0.5,0.029411764815449715,0.3824999928474426,0.6659999787807465,-2.0,-2.0,0.3135000020265579,-2.0,-2.0,0.08712121471762657,0.9600000083446503,-2.0,-2.0,0.057499999180436134,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,-1],[0.001371,0.002664,0.01804,-0.04083,0.311918,-1.097836,0.02645,-0.180764,0.712256,-0.004087,-0.076536,-0.998264,-0.549082,-0.001871,0.208422,-0.160255,-0.861972]],[[12,13,79,12,-2,-2,71,-2,-2,-2,39,34,4,-2,-2,-2,69,-2,-2],[0.7435624897480011,0.5760000050067902,0.5,0.2902499884366989,-2.0,-2.0,0.8645000159740448,-2.0,-2.0,-2.0,0.5,4.5,0.5505050718784332,-2.0,-2.0,-2.0,0.3020000010728836,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,13,-1,-1,-1,17,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,15,14,-1,-1,-1,18,-1,-1],[0.000375,0.013108,0.009837,0.013955,-0.159147,0.378645,-0.05156,-1.158958,-0.145672,1.333542,-0.009952,-0.014706,-0.011815,-0.277112,0.181543,-0.878951,0.061395,1.012402,-0.040752]],[[75,12,28,74,-2,-2,-2,20,14,-2,-2,17,-2,-2,71,74,-2,81,-2,-2,-2],[0.20749999582767487,0.8347499966621399,0.5,0.060499999672174454,-2.0,-2.0,-2.0,0.5,0.6654999852180481,-2.0,-2.0,0.987500011920929,-2.0,-2.0,0.7260000109672546,0.001500000071246177,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,18,-1,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,20,17,-1,19,-1,-1,-1],[0.000234,-0.008121,0.002947,0.005698,0.282032,-0.383681,-1.21182,-0.041306,-0.052593,-0.584045,-1.47104,0.025184,0.255006,1.274885,0.01041,0.00606,1.330177,0.001536,0.068704,-1.177677,1.235308]],[[77,12,87,35,-2,-2,74,-2,-2,17,-2,89,-2,-2,71,71,74,-2,-2,56,-2,-2,-2],[0.5,0.9974166750907898,1.5,0.5,-2.0,-2.0,0.9945000112056732,-2.0,-2.0,0.9650000035762787,-2.0,0.3484848588705063,-2.0,-2.0,0.9865000247955322,0.34950000047683716,0.013999999966472387,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[0.000729,0.003761,0.005248,0.013926,0.101177,1.017497,-0.006613,-0.135544,1.246069,-0.069847,-1.533651,-0.041893,-1.067137,-1.031136,-0.028509,-0.032993,-0.023939,-1.070634,-0.803848,-0.044902,-1.127682,-1.036246,0.20002]],[[88,67,67,69,-2,-2,74,-2,-2,73,20,-2,-2,50,-2,-2,88,17,-2,18,-2,-2,72,40,-2,-2,69,-2,-2],[0.029411764815449715,0.010499999858438969,0.005499999970197678,0.7355000078678131,-2.0,-2.0,0.9320000112056732,-2.0,-2.0,0.007500000298023224,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.10577819496393204,0.9249999821186066,-2.0,0.9600000083446503,-2.0,-2.0,0.9895000159740448,0.5,-2.0,-2.0,0.2524999976158142,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,19,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.000239,0.014816,0.037743,0.022426,-0.033478,0.872844,0.148981,2.023253,-1.076741,-0.006194,0.033487,0.959651,1.032916,-0.024558,0.333101,-0.767101,-0.006556,-0.079734,1.270218,-0.103847,-1.288522,-0.640518,-0.003077,0.001749,0.047501,-1.405383,-0.045868,-0.659546,-1.073865]],[[89,77,9,73,-2,-2,12,-2,-2,11,-2,88,-2,-2,-2],[0.7888889014720917,0.5,0.36666667461395264,0.23499999940395355,-2.0,-2.0,0.2822500020265579,-2.0,-2.0,0.5055000185966492,-2.0,0.1339285746216774,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,-1],[0.00086,0.001651,0.004609,-0.000828,-0.144936,0.204758,0.019498,-0.034061,0.652105,-0.026853,0.138591,-0.033833,-1.076303,-0.941563,-1.340959]],[[2,77,88,89,-2,-2,83,-2,-2,87,88,-2,-2,75,-2,-2,74,12,17,-2,-2,-2,-2],[0.5,0.5,0.5505050718784332,0.5536398589611053,-2.0,-2.0,0.5,-2.0,-2.0,2.5,0.1180555559694767,-2.0,-2.0,0.019499999936670065,-2.0,-2.0,0.06299999915063381,0.6382272839546204,0.4749999940395355,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,-1],[0.00183,0.0035,0.006813,0.008577,0.092516,0.933526,-0.054388,-0.88947,-1.580107,-0.026413,-0.034353,-1.07673,-1.045422,0.001585,0.704361,-1.14351,-0.0515,-0.019413,-0.045948,1.023807,-1.089991,2.204333,-1.331757]],[[89,23,17,12,-2,-2,21,-2,-2,12,87,-2,-2,74,-2,-2,86,-2,-2],[0.7888889014720917,0.5,0.9950000047683716,0.942166656255722,-2.0,-2.0,0.10000000149011612,-2.0,-2.0,0.9895833134651184,0.5,-2.0,-2.0,0.9754999876022339,-2.0,-2.0,0.8184999823570251,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[0.000867,0.00165,0.008643,0.017342,0.105481,0.562869,-0.023586,-0.787133,1.122534,-0.007538,-0.001514,0.384119,-0.136282,-0.052683,-1.131361,0.128686,-0.09804,-1.408716,-1.070758]],[[68,46,69,69,-2,-2,75,-2,-2,69,-2,-2,73,17,4,-2,-2,71,-2,-2,74,13,-2,-2,86,-2,-2],[0.9794999957084656,0.5,0.23249999433755875,0.21150000393390656,-2.0,-2.0,0.9424999952316284,-2.0,-2.0,0.25949999690055847,-2.0,-2.0,0.23499999940395355,0.9675000011920929,0.6076923310756683,-2.0,-2.0,0.47300001978874207,-2.0,-2.0,0.21199999749660492,0.17149999737739563,-2.0,-2.0,0.9794999957084656,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.003182,-0.00265,-0.005485,0.008692,0.087335,1.270661,-0.021296,-0.430933,0.337307,0.057109,-0.084361,1.029219,0.017546,0.005951,0.021974,0.182583,0.785007,-0.033807,-0.963105,0.007324,0.066027,-0.026526,0.579256,-0.949779,0.169412,1.708304,0.237944]],[[79,73,27,72,-2,-2,-2,81,88,-2,-2,88,-2,-2,70,-2,23,-2,69,-2,-2],[0.5,0.006500000134110451,0.5,0.1015000008046627,-2.0,-2.0,-2.0,0.5,0.11324786394834518,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.23799999803304672,-2.0,0.5,-2.0,0.840499997138977,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,-1,17,-1,19,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,16,-1,18,-1,20,-1,-1],[0.00068,0.003029,0.015041,0.011959,0.03252,0.466961,1.125585,-0.004169,-0.000613,0.215765,-0.170862,-0.053976,-1.169719,-0.693446,-0.044936,-1.108941,-0.020146,0.591523,-0.0466,-0.385283,-1.157146]],[[53,3,3,47,-2,-2,-2,71,12,-2,-2,74,-2,-2,69,67,-2,-2,67,72,-2,-2,14,-2,-2],[0.5,0.007500000298023224,0.004500000039115548,0.5,-2.0,-2.0,-2.0,0.051500000059604645,0.6603333353996277,-2.0,-2.0,0.3645000010728836,-2.0,-2.0,0.10850000008940697,0.004500000039115548,-2.0,-2.0,0.06949999928474426,0.22049999982118607,-2.0,-2.0,0.5480000078678131,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,19,20,-1,-1,23,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,22,21,-1,-1,24,-1,-1],[0.001248,-0.001589,0.004017,0.001893,0.076929,-0.843412,1.089639,-0.019727,-0.051366,0.408799,-0.942005,0.007864,-0.089317,1.315489,0.036148,-0.050847,-0.919674,0.225496,0.065718,0.116375,0.672701,1.225755,-0.006046,0.668341,-1.259688]],[[72,72,72,68,-2,-2,88,-2,-2,71,4,-2,-2,88,-2,-2,17,14,68,-2,-2,-2,89,75,-2,-2,68,-2,-2],[0.9835000038146973,0.9535000026226044,0.9325000047683716,0.9594999849796295,-2.0,-2.0,0.0555555559694767,-2.0,-2.0,0.036000000312924385,0.3541666716337204,-2.0,-2.0,0.1339285746216774,-2.0,-2.0,0.9850000143051147,0.8680000007152557,0.4345000088214874,-2.0,-2.0,-2.0,0.1547619104385376,0.512499988079071,-2.0,-2.0,0.9754999876022339,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[-0.002006,0.001581,-0.001504,0.000424,-0.135865,0.226178,-0.099828,-0.500739,-1.334961,0.045418,-0.004417,0.63267,-1.104146,0.12017,1.421936,-0.257636,-0.027427,0.00861,-0.020182,-1.092012,0.435648,1.198964,-0.044069,-0.067886,-1.116308,-0.976463,-0.028279,-1.033912,-1.087819]],[[13,5,3,13,-2,-2,-2,-2,13,70,-2,-2,13,-2,72,-2,-2],[0.09149999916553497,0.5,0.8599999845027924,0.02049999963492155,-2.0,-2.0,-2.0,-2.0,0.1119999997317791,0.48100000619888306,-2.0,-2.0,0.13199999928474426,-2.0,0.007500000298023224,-2.0,-2.0],[1,2,3,4,-1,-1,-1,-1,9,10,-1,-1,13,-1,15,-1,-1],[8,7,6,5,-1,-1,-1,-1,12,11,-1,-1,14,-1,16,-1,-1],[-0.001473,0.013184,0.016476,0.0194,0.042355,0.767782,-1.171338,-0.988898,-0.006494,-0.077752,-1.121512,-0.228928,-0.004632,1.488165,-0.006113,-0.568076,-0.045659]],[[72,72,72,72,-2,-2,12,-2,-2,-2,65,18,-2,69,-2,-2,-2],[0.9895000159740448,0.9874999821186066,0.9835000038146973,0.9535000026226044,-2.0,-2.0,0.9864999949932098,-2.0,-2.0,-2.0,0.5,0.9600000083446503,-2.0,0.9524999856948853,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,-1,11,12,-1,14,-1,-1,-1],[10,9,6,5,-1,-1,8,-1,-1,-1,16,13,-1,15,-1,-1,-1],[0.001755,0.005306,0.004367,0.005599,0.040733,0.661,-0.068213,-1.289315,-0.757388,0.778976,-0.033416,-0.046279,-0.70279,-0.038418,-0.952795,-1.130963,0.514701]],[[88,70,70,72,-2,-2,12,-2,-2,74,70,-2,-2,-2,88,12,67,-2,-2,-2,-2],[0.5916666686534882,0.03749999962747097,0.02850000001490116,0.31949999928474426,-2.0,-2.0,0.9692499935626984,-2.0,-2.0,0.9465000033378601,0.05049999989569187,-2.0,-2.0,-2.0,0.8166666626930237,0.9937500059604645,0.005500000203028321,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,20,19,18,-1,-1,-1,-1],[0.001808,0.003108,0.016434,0.007823,0.370163,-0.184799,0.085598,0.211198,1.688757,-0.00307,-0.004826,-0.962477,-0.015329,1.853724,-0.054255,-0.042007,-0.030794,-0.834661,-1.032342,-1.295101,-1.535991]],[[23,68,53,17,-2,-2,89,-2,-2,74,21,-2,-2,-2,72,69,17,-2,-2,69,-2,-2,74,11,-2,-2,88,-2,-2],[0.5,0.99549999833107,0.5,0.9873333275318146,-2.0,-2.0,0.4722222238779068,-2.0,-2.0,0.3595000058412552,0.22500000149011612,-2.0,-2.0,-2.0,0.5029999911785126,0.4814999997615814,0.9249999821186066,-2.0,-2.0,0.5275000035762787,-2.0,-2.0,0.7839999794960022,0.9214999973773956,-2.0,-2.0,0.22875817120075226,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[-0.001979,0.00626,0.000939,-0.0023,0.169698,-0.61219,0.083223,1.293544,-0.597643,0.04357,0.010234,-0.833048,1.082858,1.836433,-0.013253,-0.006394,-0.025276,0.857773,-0.455398,0.013582,1.550061,0.053302,-0.052919,-0.091852,-1.055128,-1.375424,0.031065,0.731602,-0.960496]],[[79,17,81,72,-2,-2,13,-2,-2,18,-2,88,-2,-2,87,67,17,-2,-2,-2,-2],[0.5,0.9549999833106995,0.5,0.9524999856948853,-2.0,-2.0,0.21050000190734863,-2.0,-2.0,0.9600000083446503,-2.0,0.12132352963089943,-2.0,-2.0,2.5,0.6635000109672546,0.9600000083446503,-2.0,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,12,-1,-1,15,16,17,-1,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,11,-1,13,-1,-1,20,19,18,-1,-1,-1,-1],[-0.000386,0.001457,0.009796,0.013693,0.157784,0.72001,-0.046753,0.433808,-0.985672,-0.009278,-1.007139,-0.006375,0.185938,-0.309504,-0.038493,-0.07346,-0.093371,-1.152215,-1.062564,0.075037,0.773867]],[[59,68,14,13,-2,-2,3,-2,-2,13,17,-2,-2,12,-2,-2,3,12,86,-2,-2,-2,69,-2,68,-2,-2],[0.5,0.9794999957084656,0.2580000013113022,0.8620000183582306,-2.0,-2.0,0.0025000000605359674,-2.0,-2.0,0.9665000140666962,0.9741666913032532,-2.0,-2.0,0.9894166588783264,-2.0,-2.0,0.004500000039115548,0.9602499902248383,0.07250000163912773,-2.0,-2.0,-2.0,0.07000000029802322,-2.0,0.9309999942779541,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,-1,25,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,24,-1,26,-1,-1],[0.000893,-0.001807,-0.007704,0.044795,-0.208577,1.451974,-0.011175,-0.046683,-0.53668,0.01181,-0.001366,0.226296,-0.60465,0.073554,1.739542,0.259114,0.027051,0.011527,0.024881,0.083814,1.0928,-0.642676,0.05701,1.160451,0.019115,1.001492,-0.309564]],[[13,77,14,69,-2,-2,34,-2,-2,67,69,-2,-2,-2,18,73,69,-2,-2,75,-2,-2,-2],[0.99549999833107,0.5,0.3110000044107437,0.9494999945163727,-2.0,-2.0,2.5,-2.0,-2.0,0.651500016450882,0.38449999690055847,-2.0,-2.0,-2.0,0.9950000047683716,0.8445000052452087,0.1314999982714653,-2.0,-2.0,0.9964999854564667,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,-1],[8.8e-05,0.002427,0.005645,-0.019008,-0.00755,-1.21528,0.009721,0.077266,0.493961,-0.027747,-0.031591,-0.698432,-1.088207,0.047755,-0.051779,-0.067532,-0.121752,-0.429774,-1.437928,-0.041117,-0.793915,-1.173424,-0.086995]],[[4,46,70,70,-2,-2,71,-2,-2,71,72,-2,-2,-2,11,21,68,-2,-2,9,-2,-2,3,4,-2,-2,89,-2,-2],[0.6547619104385376,0.5,0.08950000256299973,0.04649999924004078,-2.0,-2.0,0.21700000017881393,-2.0,-2.0,0.6620000004768372,0.2224999964237213,-2.0,-2.0,-2.0,0.9799999892711639,0.22500000149011612,0.9864999949932098,-2.0,-2.0,0.18333333730697632,-2.0,-2.0,0.04899999871850014,0.8660714328289032,-2.0,-2.0,0.4652777910232544,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,-1,15,16,17,-1,-1,20,-1,-1,23,24,-1,-1,27,-1,-1],[14,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,-1,22,19,18,-1,-1,21,-1,-1,26,25,-1,-1,28,-1,-1],[0.001492,0.00464,0.002451,0.015498,0.096539,0.670199,-0.007042,-0.535084,0.125986,0.052919,0.004265,-0.370648,0.679447,1.193052,-0.016824,-0.039073,-0.058103,-0.638602,-1.241328,0.042644,1.11945,1.059207,0.00692,-0.012317,-0.487641,1.370345,0.089078,1.591162,-0.362796]],[[81,77,3,88,-2,-2,-2,71,69,-2,-2,-2,11,72,-2,34,-2,-2,73,-2,-2],[0.5,0.5,0.8610000014305115,0.612500011920929,-2.0,-2.0,-2.0,0.9835000038146973,0.38449999690055847,-2.0,-2.0,-2.0,0.9794999957084656,0.12350000068545341,-2.0,0.5,-2.0,-2.0,0.005500000203028321,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,-1,13,14,-1,16,-1,-1,19,-1,-1],[12,7,6,5,-1,-1,-1,11,10,-1,-1,-1,18,15,-1,17,-1,-1,20,-1,-1],[0.001393,0.004105,0.006836,0.00754,0.126242,-1.209338,-1.172096,-0.023732,-0.029344,-0.601546,-1.083934,0.353342,-0.040599,-0.062556,-1.19321,-0.042692,-1.111536,-0.371235,0.015005,0.955883,-1.206125]],[[9,35,23,18,-2,-2,68,-2,-2,89,69,-2,-2,72,-2,-2,11,75,68,-2,-2,87,-2,-2,23,4,-2,-2,11,-2,-2],[0.4166666716337204,0.5,0.5,0.9850000143051147,-2.0,-2.0,0.017500000074505806,-2.0,-2.0,0.2899159789085388,0.3994999974966049,-2.0,-2.0,0.6324999928474426,-2.0,-2.0,0.6414999961853027,0.06200000084936619,0.9679999947547913,-2.0,-2.0,1.5,-2.0,-2.0,0.5,0.7071428596973419,-2.0,-2.0,0.9120000004768372,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.000482,-0.006137,-0.010193,0.000188,0.2275,-0.462305,-0.02369,1.449416,-0.346955,0.02259,0.048131,1.067523,-0.117172,-0.039401,-1.166804,0.035996,0.01567,-0.000251,0.01098,0.772229,-0.54818,-0.059895,0.616916,-1.103951,0.031695,0.006112,0.505172,-1.71581,0.068587,1.541505,0.59515]],[[70,70,16,69,-2,-2,89,-2,-2,74,-2,-2,34,81,13,-2,-2,71,-2,-2,75,12,-2,-2,71,-2,-2],[0.05849999934434891,0.05049999989569187,2.5,0.9214999973773956,-2.0,-2.0,0.1180555559694767,-2.0,-2.0,0.5509999990463257,-2.0,-2.0,0.5,0.5,0.8939999938011169,-2.0,-2.0,0.9145000278949738,-2.0,-2.0,0.04299999959766865,0.6565000116825104,-2.0,-2.0,0.45749999582767487,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,-1,-1,13,14,15,-1,-1,18,-1,-1,21,22,-1,-1,25,-1,-1],[12,9,6,5,-1,-1,8,-1,-1,11,-1,-1,20,17,16,-1,-1,19,-1,-1,24,23,-1,-1,26,-1,-1],[0.000908,0.014153,0.009188,0.001728,-0.052917,0.506432,0.046997,1.226451,-0.051498,0.117255,0.282902,1.492747,-0.007489,0.005501,0.009594,0.334305,-0.466673,-0.05331,-1.322461,0.008505,-0.026,-0.059376,-0.316464,-1.24737,-0.009885,-0.440585,0.525883]],[[9,57,68,88,-2,-2,14,-2,-2,89,11,-2,-2,14,-2,-2,73,75,24,-2,-2,70,-2,-2,11,87,-2,-2,14,-2,-2],[0.36666667461395264,0.5,0.9535000026226044,0.14835165441036224,-2.0,-2.0,0.6605000197887421,-2.0,-2.0,0.24499999731779099,0.9914999902248383,-2.0,-2.0,0.7854999899864197,-2.0,-2.0,0.02650000061839819,0.03849999979138374,0.5,-2.0,-2.0,0.7524999976158142,-2.0,-2.0,0.15600000321865082,1.5,-2.0,-2.0,0.6514999866485596,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,22,-1,-1,25,26,-1,-1,29,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,24,21,20,-1,-1,23,-1,-1,28,27,-1,-1,30,-1,-1],[-0.001774,-0.007711,-0.012652,-0.022311,-0.510757,-0.064785,0.003887,0.206247,-0.809827,0.015818,0.043169,0.370745,1.132921,-0.017654,0.001935,-1.160742,0.015536,0.027782,0.018167,0.530077,1.075621,0.078781,0.211101,1.233638,-0.012585,-0.062101,0.247752,-1.454044,0.010487,-0.170826,1.091057]],[[79,86,75,75,-2,-2,-2,13,67,-2,-2,71,-2,-2,68,70,-2,-2,11,-2,-2],[0.5,0.7120000123977661,0.6699999868869781,0.6380000114440918,-2.0,-2.0,-2.0,0.948500007390976,0.001500000071246177,-2.0,-2.0,0.030500000342726707,-2.0,-2.0,0.948500007390976,0.5820000171661377,-2.0,-2.0,0.9000000059604645,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,9,-1,-1,12,-1,-1,15,16,-1,-1,19,-1,-1],[14,7,6,5,-1,-1,-1,11,10,-1,-1,13,-1,-1,18,17,-1,-1,20,-1,-1],[-0.000408,0.001535,0.008732,0.007141,0.152614,-1.211866,1.066614,-0.007836,-0.030914,-0.808877,-0.350122,0.00808,-0.148233,0.563795,-0.040008,-0.065,-1.18202,-0.848297,0.007785,-0.737525,0.654221]],[[70,70,68,73,-2,-2,-2,13,-2,-2,12,74,73,-2,-2,72,-2,-2,13,-2,68,-2,-2],[0.0345000009983778,0.02850000001490116,0.99549999833107,0.7885000109672546,-2.0,-2.0,-2.0,0.56700000166893,-2.0,-2.0,0.9895833134651184,0.8760000169277191,0.004500000039115548,-2.0,-2.0,0.7010000050067902,-2.0,-2.0,0.9864999949932098,-2.0,0.8240000009536743,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,12,13,-1,-1,16,-1,-1,19,-1,21,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,18,15,14,-1,-1,17,-1,-1,20,-1,22,-1,-1],[0.000972,0.01331,0.005418,0.002594,0.245246,-0.395335,1.126322,0.115409,-0.802952,1.448105,-0.004584,0.000424,-0.002992,0.241075,-0.214625,0.068806,-0.832905,1.530981,-0.069542,-1.182235,-0.043619,-0.936631,-0.648597]],[[73,79,39,74,-2,-2,-2,71,-2,-2,73,-2,48,69,-2,-2,-2],[0.007500000298023224,0.5,0.5,0.7689999938011169,-2.0,-2.0,-2.0,0.6969999969005585,-2.0,-2.0,0.00849999999627471,-2.0,0.5,0.03350000083446503,-2.0,-2.0,-2.0],[1,2,3,4,-1,-1,-1,8,-1,-1,11,-1,13,14,-1,-1,-1],[10,7,6,5,-1,-1,-1,9,-1,-1,12,-1,16,15,-1,-1,-1],[-0.002161,0.008204,0.010961,0.007784,0.302237,-0.350187,0.670198,-0.055259,-1.204726,-0.013494,-0.009176,-1.347056,-0.006985,-0.004014,0.334164,-0.166994,-0.779075]],[[55,73,18,78,-2,-2,68,-2,-2,14,4,-2,-2,14,-2,-2,67,-2,-2],[0.5,0.006500000134110451,0.9600000083446503,0.5,-2.0,-2.0,0.3774999976158142,-2.0,-2.0,0.22049999982118607,0.3818181902170181,-2.0,-2.0,0.38449999690055847,-2.0,-2.0,0.005499999853782356,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,18,-1,-1],[-0.001537,-0.003079,0.008162,0.017004,0.38682,-1.537118,-0.01831,0.477473,-0.552772,-0.009583,0.042193,-0.09816,1.223432,-0.012062,-0.594958,-0.10263,0.041423,-0.224321,1.048214]],[[23,17,34,24,-2,-2,71,-2,-2,21,75,-2,-2,13,-2,-2,68,75,73,-2,-2,-2,9,39,-2,-2,11,-2,-2],[0.5,0.9950000047683716,0.5,0.5,-2.0,-2.0,0.013500000350177288,-2.0,-2.0,0.10000000149011612,0.9914999902248383,-2.0,-2.0,0.09050000086426735,-2.0,-2.0,0.02850000001490116,0.08950000256299973,0.021499999798834324,-2.0,-2.0,-2.0,0.4166666716337204,0.5,-2.0,-2.0,0.9839999973773956,-2.0,-2.0],[1,2,3,4,-1,-1,7,-1,-1,10,11,-1,-1,14,-1,-1,17,18,19,-1,-1,-1,23,24,-1,-1,27,-1,-1],[16,9,6,5,-1,-1,8,-1,-1,13,12,-1,-1,15,-1,-1,22,21,20,-1,-1,-1,26,25,-1,-1,28,-1,-1],[0.001342,0.009635,0.017766,0.008252,0.154933,0.864332,0.042976,-0.389275,0.827578,-0.019971,-0.031918,-1.042019,1.118717,0.036494,1.024167,1.229866,-0.00875,0.080279,0.014988,1.064597,-0.775978,1.089928,-0.013828,-0.024201,-0.351336,0.690444,0.018306,-0.123066,1.044645]]]
LR_INTER = 4.03068147
LR_COEFS = [-0.09838692,-0.09838692,-1.26137169,-2.55819773,0.36983308,0.11032418,0.36181801,-2.10417382,3.49869789,2.27359774,0.50557142,3.31981371,-1.03280707,-1.60976508,-0.39404749,0.0,0.13764053,-1.85504247,-1.82467658,-1.30416684,3.30197553,-3.26831558,2.33634284,-1.22602021,2.58905194,2.19864655,0.39059813,0.39059813,0.13459705,-0.57155581,-0.06257557,-2.10417382,3.4986968,0.0,-0.05674748,0.85252728,-1.0445259,-0.73759873,0.09188221,0.57019111,-1.64265352,2.02474337,0.45686295,0.65906261,0.06649955,1.08149066,0.175504,0.87105832,1.97360587,3.80014889,-1.7487053,-1.06308938,-0.49188488,0.06180945,-10.99564687,0.47437233,-2.03271299,0.67329483,0.90415897,1.14261102,-0.9713435,0.26777515,0.78596879,-0.92021324,1.84191504,-0.66828826,-0.15080598,0.09568638,0.05386436,0.02080633,-0.93915619,-0.00451068,-0.87082823,-0.94138278,-0.30064651,0.34345601,1.2623485,-4.61346509,-0.72825436,-4.07060872,-1.91917227,-5.55076813,1.56698723,0.04569645,0.82405937,-0.35491507,-0.04946835,-0.04860088,-3.30648293,-1.13001104,3.81025084,1.10398239,2.07526206]
W_GB = 0.8100
UMBRAL_CORTAS = 0.243358
UMBRAL_LARGAS = 0.340000


def _evaluar_arbol(arbol, x):
    variable, umbral, izq, der, valor = arbol
    nodo = 0
    while izq[nodo] != -1:
        nodo = izq[nodo] if x[variable[nodo]] <= umbral[nodo] else der[nodo]
    return valor[nodo]


def _sigmoide(z):
    z_clip = max(-30.0, min(30.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z_clip))


def calcular_probabilidad(feats):
    x = [struct.unpack("f", struct.pack("f", float(feats.get(n, 0.0))))[0] for n in NOMBRES]
    z_gb = INICIAL_GB + TASA_GB * sum(_evaluar_arbol(a, x) for a in ARBOLES_GB)
    p_gb = _sigmoide(z_gb)
    z_lr = LR_INTER + sum(c * val for c, val in zip(LR_COEFS, x))
    p_lr = _sigmoide(z_lr)
    return W_GB * p_gb + (1.0 - W_GB) * p_lr


def reescalar_score(p, corte):
    if p < corte:
        return 0.5 * p / corte if corte > 0 else 0.0
    return 0.5 + 0.5 * (p - corte) / (1.0 - corte) if corte < 1.0 else 1.0


class Guardian:
    def score(self, caso: dict) -> float:
        try:
            feats = extraer_mapa_caracteristicas(caso)
            p = calcular_probabilidad(feats)
            n_orac = len(((caso.get("senales") or {}).get("oraciones")) or [])
            if n_orac <= 2:
                corte = UMBRAL_CORTAS
            else:
                corte = min(0.36, UMBRAL_LARGAS + 0.02 * math.log(max(1.0, n_orac - 2.0)))
            return float(min(1.0, max(0.0, reescalar_score(p, corte))))
        except Exception:
            return 1.0


Bot = Guardian
