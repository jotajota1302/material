"""El Guardian - detector de alucinaciones. Solo Python estandar (re)."""
import re

CAU_RE = re.compile(r'(gracias a|debido a|por culpa|por su mayor|por su menor|porque |ya que |esto demuestra|por lo que|esto confirma)', re.I)
RESTR = re.compile(r'(solo|unicamente|salvo|excepto|durante|frescos?|refrigerad\w*|bajura|altura|maximo|minimo|ultimos?|primeros?)', re.I)
DE_RE = re.compile(r'(\d+)\s+de\s+(los|las|\d+)', re.I)
NUM_RE = re.compile(r'\d+(?:[.,]\d+)?')
HEDGE_RE = re.compile(r'(podr\w*|pued\w*|posible(mente)?|probable(mente)?|si se cumple|si llueve|depend\w* de que|pendiente de|prev[eé]\w* que|estim\w* que)', re.I)
COND_RE = re.compile(r'\bsi\b|en caso de|siempre que|salvo que|a condici[oó]n de', re.I)
CONDV_RE = re.compile(r'\b\w+(ar|er|ir)ia\b', re.I)
PARA_RE = re.compile(r'\bpara\s+(poder\s+)?(\w+\s+){1,4}', re.I)
UNIT_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(millones?|miles?|paginas?|documentos?|toneladas?|kilos?|euros?|centimos?|metros?|km|tiendas?|arboles?|ejemplares?|anos?|meses|dias?)', re.I)
NUMWORDS = {'cero': 0, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5, 'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10, 'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15, 'veinte': 20, 'treinta': 30, 'cien': 100, 'ciento': 100, 'mil': 1000}
PARAPH = {'ultimo': ('reciente', 'actual', 'vigente'), 'solo': ('unicamente',), 'maximo': ('como mucho', 'a lo sumo'), 'minimo': ('como minimo', 'al menos')}


ENT_RE = re.compile(r'[A-ZÁÉÍÓÚÜ][a-záéíóúü]+|[A-ZÁÉÍÓÚÜ]{2,}|[A-ZÁÉÍÓÚÜ]\b')
ENT_STOP = {'El', 'La', 'Los', 'Las', 'Un', 'Una', 'En', 'De', 'Del', 'Con', 'Por', 'Para', 'Según', 'Segun', 'Desde', 'Hasta', 'Entre', 'Como', 'Más', 'Mas', 'Este', 'Esta', 'Eso', 'Sus', 'Su', 'Al', 'Se'}


def _entities_case(fuentes, respuesta):
    vocab = set()
    for text in fuentes + [respuesta]:
        for sent in re.split(r'[.!?]', text):
            s = sent.strip()
            ms = list(ENT_RE.finditer(s))
            for k, m in enumerate(ms):
                w = m.group(0)
                if w in ENT_STOP:
                    continue
                if k == 0 and m.start() < 3:
                    continue
                vocab.add(_norm(w))
    return vocab


def _attrib_hit(fuentes, respuesta, vocab):
    rlow = _norm(respuesta)
    fl = [_norm(f) for f in fuentes]
    for sent in re.split(r'[.!?]', respuesta):
        sents = sent.strip()
        ctx = set()
        for m in ENT_RE.finditer(sents):
            w = m.group(0)
            if w in ENT_STOP:
                continue
            if _norm(w) in vocab:
                ctx.add(_norm(w))
        if not ctx:
            continue
        for m in NUM_RE.finditer(_norm(sents)):
            v = _numval(m.group(0))
            if v is None:
                continue
            holders = []
            for fo, f in zip(fuentes, fl):
                vals = set()
                for x in NUM_RE.finditer(f):
                    vv = _numval(x.group(0))
                    if vv is not None:
                        vals.add(round(vv, 2))
                if round(v, 2) in vals:
                    holders.append(fo)
            if not holders:
                continue
            ok = False
            for h in holders:
                he = set()
                for mm in ENT_RE.finditer(h):
                    w = mm.group(0)
                    if w in ENT_STOP:
                        continue
                    if _norm(w) in vocab:
                        he.add(_norm(w))
                if ctx & he:
                    ok = True
                    break
            if not ok:
                return True
    return False




def _norm(s):
    out = s.lower()
    for a, b in (('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ü', 'u')):
        out = out.replace(a, b)
    for w, v in NUMWORDS.items():
        out = re.sub(r'\b' + w + r'\b', str(v), out)
    return out


def _toks(s):
    return re.findall(r'[a-z0-9]+', s)


def _content(s):
    return [t for t in _toks(s) if len(t) >= 5]


def _ov(a, b):
    sa = set(a)
    if not sa:
        return 0.0
    return len(sa & set(b)) / len(sa)


def _numval(tok):
    t = tok.replace(',', '.')
    if re.match(r'^\d{1,3}(\.\d{3})+$', t):
        t = t.replace('.', '')
    try:
        return float(t)
    except Exception:
        return None


class Guardian:
    def score(self, caso, debug=False):
        fired = []
        ors = (caso.get('senales') or {}).get('oraciones', []) or []
        frs = (caso.get('senales') or {}).get('fragmentos', []) or []
        fuentes = caso.get('fuentes') or []
        respuesta = caso.get('respuesta') or ''
        rlow = _norm(respuesta)
        flow = _norm(' '.join(fuentes))
        rtok = _content(rlow)
        nsrc = len(fuentes)
        srcs = [_norm(f) for f in fuentes]

        per = []
        min_rt = 1.0
        for o in ors:
            rt = o.get('respaldo_todas', 0) or 0
            rb = max(o.get('respaldo_por_fuente', [0]) or [0])
            ct = o.get('contradiccion_todas', 0) or 0
            sim = max(o.get('similitud_por_fuente', [0]) or [0])
            gap = sim - (rb if rb > rt else rt)
            llm = o.get('llm') or {}
            v = llm.get('veredicto')
            cf = llm.get('confianza') or 0
            m = rt if rt < rb else rb
            if v == 'contradicha':
                lr = 1.0
            elif v == 'no_respaldada':
                lr = 0.85
            elif v == 'respaldada':
                if m >= 0.50:
                    lr = 0.05
                elif m >= 0.30:
                    lr = 0.30 if cf < 0.95 else 0.12
                else:
                    lr = 0.60 if cf < 0.95 else 0.25
            else:
                lr = 0.50
            s = 0.06 * (1.0 - rt) + 0.34 * (1.0 - rb) + 0.06 * ct + 1.27 * lr + 0.50 * (gap if gap > 0 else 0)
            per.append(s)
            if rt < min_rt:
                min_rt = rt
        base = max(per) if per else 0.5
        if frs:
            fw = max(1.0 - ((f.get('respaldo_todas', 0) or 0)) for f in frs)
            alt = 0.24 * fw + 0.35
            if alt > base:
                base = alt

        def best_src(oracion_n):
            sims = oracion_n.get('similitud_por_fuente', [0]) or [0]
            if len(sims) != nsrc or nsrc == 0:
                return -1, ''
            mx = max(sims)
            cand = [i for i, v in enumerate(sims) if v >= mx - 0.05]
            ot = _content(_norm(oracion_n.get('texto', '')))
            bi = cand[0]
            bo = -1.0
            for i in cand:
                o = _ov(ot, _content(srcs[i]))
                if o > bo:
                    bo = o
                    bi = i
            return bi, srcs[bi]

        # 1) restriccion caida (por oracion)
        scope_hit = 0.0
        for o in ors:
            bi, src = best_src(o)
            if bi < 0:
                continue
            ot = _content(_norm(o.get('texto', '')))
            if _ov(ot, _content(src)) < 0.30:
                continue
            miss = False
            for mm in RESTR.finditer(src):
                w = mm.group(0).lower()
                if w not in rlow and not any(p in rlow for p in PARAPH.get(w, ())):
                    miss = True
                    break
            if not miss and DE_RE.search(src):
                ns = NUM_RE.findall(src)
                rvals = set()
                for n in NUM_RE.findall(rlow):
                    v = _numval(n)
                    if v is not None:
                        rvals.add(round(v, 2))
                hit = False
                for n in ns:
                    v = _numval(n)
                    if v is not None and round(v, 2) in rvals:
                        hit = True
                        break
                if ns and not hit:
                    miss = True
            if miss:
                scope_hit = 0.55
                if debug:
                    fired.append('scope:F%d' % bi)
                break

        # 5) matiz condicional caido (por oracion)
        hedge_hit = False
        for o in ors:
            bi, src = best_src(o)
            if bi < 0:
                continue
            ot = _content(_norm(o.get('texto', '')))
            if _ov(ot, _content(src)) < 0.30:
                continue
            otxt = _norm(o.get('texto', ''))
            sh = HEDGE_RE.search(src) or COND_RE.search(src) or CONDV_RE.search(src)
            rh = HEDGE_RE.search(otxt) or COND_RE.search(otxt) or CONDV_RE.search(otxt)
            if sh and not rh:
                hedge_hit = True
                if debug:
                    fired.append('hedge:F%d' % bi)
                break
        if hedge_hit:
            base += 0.40

        # numeros por valor
        fflat = flow.replace(',', '.')
        fvals = set()
        for x in NUM_RE.findall(fflat):
            v = _numval(x)
            if v is not None:
                fvals.add(round(v, 2))
        rvals_list = []
        for x in NUM_RE.findall(rlow):
            v = _numval(x)
            if v is not None:
                rvals_list.append(round(v, 2))
        deriv = set(fvals)
        sv = sorted(fvals)
        L = min(len(sv), 12)
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                deriv.add(round(sv[i] + sv[j], 2))
                deriv.add(round(abs(sv[i] - sv[j]), 2))
                deriv.add(round(100 - sv[i] - sv[j], 2))
                deriv.add(round(sv[i] * 100, 2))
                deriv.add(round(sv[i] * 1000, 2))
                if sv[j]:
                    deriv.add(round(sv[i] / sv[j], 2))
                    deriv.add(round(sv[i] / sv[j] * 100, 2))
        nums_r = NUM_RE.findall(rlow)
        max_ov = 0.0
        for s in srcs:
            o = _ov(rtok, _content(s))
            if o > max_ov:
                max_ov = o
        if nums_r:
            derived = 0
            missing = 0
            nearmiss = False
            for n in nums_r:
                v = _numval(n)
                if v is None:
                    continue
                if round(v, 2) in fvals:
                    continue
                missing += 1
                if round(v, 2) in deriv:
                    derived += 1
                elif v:
                    for d in deriv:
                        if d and abs(v - d) / abs(d) < 0.15 and round(v, 2) != round(d, 2):
                            nearmiss = True
                            break
            if missing and derived == missing:
                base -= 0.45
                if debug:
                    fired.append('derived-')
            if nearmiss and max_ov >= 0.35:
                base += 0.45
                if debug:
                    fired.append('near+')

        # 4) unidad cambiada con mismo numero
        if max_ov >= 0.40:
            runs = {}
            for m in UNIT_RE.finditer(rlow):
                v = _numval(m.group(1))
                if v is not None:
                    runs.setdefault(round(v, 2), set()).add(m.group(2).lower())
            funits = {}
            for m in UNIT_RE.finditer(flow):
                v = _numval(m.group(1))
                if v is not None:
                    funits.setdefault(round(v, 2), set()).add(m.group(2).lower())
            for n, us in runs.items():
                if n in funits and not (us & funits[n]):
                    base += 0.50
                    if debug:
                        fired.append('unit+')
                    break
        base += scope_hit

        # 2) causa con contenido nuevo
        mca = CAU_RE.search(rlow)
        if mca and not CAU_RE.search(flow):
            after = rlow[mca.end():mca.end() + 60]
            news = [w for w in _content(after) if w not in flow]
            if len(news) >= 2 and (min_rt < 0.5 or max_ov >= 0.4):
                base += 0.30
                if debug:
                    fired.append('cau+')
            elif min_rt < 0.5:
                conf_max = 0.0
                for o in ors:
                    llm = o.get('llm') or {}
                    if llm.get('veredicto') == 'respaldada':
                        cf = llm.get('confianza') or 0
                        if cf > conf_max:
                            conf_max = cf
                if conf_max < 0.92:
                    base += 0.30
                    if debug:
                        fired.append('cau+')

        # 6) finalidad anadida con contenido ausente
        if max_ov >= 0.40:
            for mp in PARA_RE.finditer(rlow):
                clause = rlow[mp.start():mp.start() + 70]
                cw = [w for w in _content(clause) if w not in ('para', 'poder')]
                if len(cw) >= 3 and sum(1 for w in cw if w not in flow) >= 3:
                    base += 0.35
                    if debug:
                        fired.append('para+')
                    break

        # 6b) cifra atribuida a otra entidad
        try:
            if _attrib_hit(fuentes, respuesta, _entities_case(fuentes, respuesta)):
                base += 0.45
                if debug:
                    fired.append('attr+')
        except Exception:
            pass

        # 7) mitigacion
        if scope_hit == 0.0 and not hedge_hit:
            mit_ok = True
            for o in ors:
                llm = o.get('llm') or {}
                if llm.get('veredicto') != 'respaldada' or (llm.get('confianza') or 0) < 0.95:
                    mit_ok = False
                    break
                if (o.get('contradiccion_todas', 0) or 0) >= 0.40:
                    mit_ok = False
                    break
            if mit_ok:
                if nums_r:
                    if all(((_numval(n) is not None and round(_numval(n), 2) in fvals)) for n in nums_r):
                        base -= 0.40
                        if debug:
                            fired.append('mit-')
                else:
                    clean = True
                    for o in ors:
                        llm = o.get('llm') or {}
                        if (llm.get('confianza') or 0) < 0.97:
                            clean = False
                            break
                        if (o.get('contradiccion_todas', 0) or 0) >= 0.30:
                            clean = False
                            break
                    if clean:
                        base -= 0.40
                        if debug:
                            fired.append('mit-')

        lo, hi = -0.59, 2.00
        s = (base - lo) / (hi - lo)
        if s < 0:
            s = 0.0
        if s > 1:
            s = 1.0
        if debug:
            return float(s), fired
        return float(s)
