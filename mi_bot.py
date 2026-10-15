"""El Guardian - detector de alucinaciones. Solo Python estandar (re)."""
import re

CAU_RE = re.compile(r'(gracias a|debido a|por culpa|por su mayor|por su menor|porque |ya que |esto demuestra|por lo que|esto confirma)', re.I)
RESTR = re.compile(r'(solo|unicamente|salvo|excepto|durante|frescos?|refrigerad\w*|bajura|altura|primeros?|ultimos?|minimo|m[i]nimo|maximo|m[a]ximo)', re.I)
DE_RE = re.compile(r'(\d+)\s+de\s+(los|las|\d+)', re.I)
NUM_RE = re.compile(r'\d+(?:[.,]\d+)?')
HEDGE_RE = re.compile(r'(podr\w*|pued\w*|posible(mente)?|probable(mente)?|si se cumple|si llueve|depend\w* de que|pendiente de|prev[eé]\w* que|estim\w* que)', re.I)
COND_RE = re.compile(r'\bsi\b|en caso de|siempre que|salvo que|a condici[oó]n de', re.I)
CONDV_RE = re.compile(r'\b\w+(ar|er|ir)[íi]a\b', re.I)
PARA_RE = re.compile(r'\bpara\s+(poder\s+)?(\w+\s+){1,4}', re.I)
UNIT_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(millones?|miles?|paginas?|documentos?|toneladas?|kilos?|euros?|centimos?|metros?|km|tiendas?|arboles?|ejemplares?|anos?|meses|dias?)', re.I)
NUMWORDS = {'cero': 0, 'uno': 1, 'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5, 'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10, 'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15, 'veinte': 20, 'treinta': 30, 'cien': 100, 'ciento': 100, 'mil': 1000}


def _norm_nums(s):
    out = s
    for w, v in NUMWORDS.items():
        out = re.sub(r'\b' + w + r'\b', str(v), out)
    return out


def _toks(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def _content(s):
    return [t for t in _toks(s) if len(t) >= 5]


def _overlap(a_toks, b_toks):
    if not a_toks:
        return 0.0
    sa = set(a_toks)
    sb = set(b_toks)
    return len(sa & sb) / max(1, len(sa))


class Guardian:
    def score(self, caso):
        ors = (caso.get('senales') or {}).get('oraciones', []) or []
        frs = (caso.get('senales') or {}).get('fragmentos', []) or []
        fuentes = caso.get('fuentes') or []
        respuesta = caso.get('respuesta') or ''
        rlow = _norm_nums(respuesta.lower())
        flow = _norm_nums(' '.join(fuentes).lower())
        rtok = _content(rlow)

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

        # solape maximo con alguna fuente (nucleo compartido)
        max_ov = 0.0
        best_src = ''
        for fsrc in fuentes:
            sfl = _norm_nums(fsrc.lower())
            ov = _overlap(rtok, _content(sfl))
            if ov > max_ov:
                max_ov = ov
                best_src = sfl

        # 1) restriccion caida en la fuente mas parecida
        scope_hit = 0.0
        if max_ov >= 0.35 and best_src:
            miss = False
            for mm in RESTR.finditer(best_src):
                if mm.group(0).lower() not in rlow:
                    miss = True
                    break
            if not miss and DE_RE.search(best_src):
                ns = NUM_RE.findall(best_src)
                nr = NUM_RE.findall(rlow)
                if ns and not any(n in nr for n in ns):
                    miss = True
            if miss:
                scope_hit = 0.55
        base += scope_hit

        # 2) causa con contenido nuevo
        cau_hit = False
        mca = CAU_RE.search(rlow)
        if mca and not CAU_RE.search(flow):
            after = rlow[mca.end():mca.end() + 60]
            news = [w for w in _content(after) if w not in flow]
            if len(news) >= 2 and (min_rt < 0.5 or max_ov >= 0.4):
                base += 0.30
                cau_hit = True
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
                    cau_hit = True

        # 3) numeros: derivados, casi y literales
        fflat = flow.replace(',', '.')
        src_vals = []
        for x in NUM_RE.findall(fflat):
            try:
                src_vals.append(float(x.replace(',', '.')))
            except Exception:
                pass
        deriv = set([round(v, 2) for v in src_vals])
        L = min(len(src_vals), 12)
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                deriv.add(round(src_vals[i] + src_vals[j], 2))
                deriv.add(round(abs(src_vals[i] - src_vals[j]), 2))
                deriv.add(round(100 - src_vals[i] - src_vals[j], 2))
                deriv.add(round(src_vals[i] * 100, 2))
                deriv.add(round(src_vals[i] * 1000, 2))
                if src_vals[j]:
                    deriv.add(round(src_vals[i] / src_vals[j], 2))
                    deriv.add(round(src_vals[i] / src_vals[j] * 100, 2))
        nums_r = NUM_RE.findall(rlow)
        if nums_r:
            derived = 0
            missing = 0
            nearmiss = False
            for n in nums_r:
                nn = n.replace(',', '.')
                if nn in fflat:
                    continue
                missing += 1
                try:
                    v = float(nn)
                except Exception:
                    continue
                if round(v, 2) in deriv:
                    derived += 1
                elif v:
                    for d in deriv:
                        if d and abs(v - d) / abs(d) < 0.15 and round(v, 2) != round(d, 2):
                            nearmiss = True
                            break
            if missing and derived == missing:
                base -= 0.45
            if nearmiss and max_ov >= 0.35:
                base += 0.45

        # 4) unidad cambiada con mismo numero
        if max_ov >= 0.40:
            try:
                runs = {}
                for m in UNIT_RE.finditer(rlow):
                    runs.setdefault(m.group(1).replace(',', '.'), set()).add(m.group(2).lower())
                funits = {}
                for m in UNIT_RE.finditer(flow):
                    funits.setdefault(m.group(1).replace(',', '.'), set()).add(m.group(2).lower())
                for n, us in runs.items():
                    if n in funits and not (us & funits[n]):
                        base += 0.50
                        break
            except Exception:
                pass

        # 5) matiz condicional caido (podria/si/condicional) con mismo nucleo
        hedge_hit = False
        if max_ov >= 0.45:
            src_has = HEDGE_RE.search(flow) or COND_RE.search(flow) or CONDV_RE.search(flow)
            rsp_has = HEDGE_RE.search(rlow) or COND_RE.search(rlow) or CONDV_RE.search(rlow)
            if src_has and not rsp_has:
                base += 0.40
                hedge_hit = True

        # 6) finalidad anadida con contenido ausente
        if max_ov >= 0.40:
            for mp in PARA_RE.finditer(rlow):
                clause = rlow[mp.start():mp.start() + 70]
                cw = [w for w in _content(clause) if w not in ('para', 'poder')]
                if len(cw) >= 3 and sum(1 for w in cw if w not in flow) >= 3:
                    base += 0.35
                    break

        # 7) mitigacion: LLM muy seguro + sin contradiccion + sin motivo
        if scope_hit == 0.0 and not hedge_hit:
            try:
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
                        allp = True
                        for n in nums_r:
                            if n.replace(',', '.') not in fflat:
                                allp = False
                                break
                        if allp:
                            base -= 0.40
                    else:
                        # sin cifras: exigir confianza extrema y NLI limpio
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
            except Exception:
                pass

        _ = cau_hit
        lo, hi = -0.78, 2.00
        s = (base - lo) / (hi - lo)
        if s < 0:
            s = 0.0
        if s > 1:
            s = 1.0
        return float(s)
