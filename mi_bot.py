"""Guardian v3 - scope ampliado + aritmetica derivada. Puro stdlib."""
import re
CAU_RE = re.compile(r'(gracias a|debido a|por culpa|por su mayor|por su menor|porque |ya que |esto demuestra|por lo que|esto confirma)', re.I)
RESTR = re.compile(r'(solo|únicamente|unicamente|salvo|excepto|durante|hasta|desde|entre|frescos?|refrigerad\w*|bajura|altura|primeros?|últimos?|ultimos?|mínimo|m[ií]nimo|máximo|m[aá]ximo|años?|meses|d[ií]as|horas|semanas?)', re.I)
DE_RE = re.compile(r'(\d+)\s+de\s+(los|las|\d+)', re.I)
NUM_RE = re.compile(r'\d+(?:[.,]\d+)?')

NUMWORDS = {'cero':0,'uno':1,'una':1,'dos':2,'tres':3,'cuatro':4,'cinco':5,'seis':6,'siete':7,'ocho':8,'nueve':9,'diez':10,'once':11,'doce':12,'trece':13,'catorce':14,'quince':15,'dieciseis':16,'dieciocho':18,'diecinueve':19,'veinte':20,'treinta':30,'cuarenta':40,'cincuenta':50,'sesenta':60,'setenta':70,'ochenta':80,'noventa':90,'cien':100,'ciento':100,'mil':1000}

HEDGE_RE = re.compile(r'(podr[íi]a|podr[íi]an|puede(n)?|posible(mente)?|probable(mente)?|si se cumple|si llueve|condicionad\w*|siempre que)', re.I)
UNIT_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(millones?|miles?|millar\w*|p[aá]ginas?|documentos?|toneladas?|kilos?|euros?|c[eé]ntimos?|metros?|km|tiendas?|\xe1rboles?|ejemplares?|a[ñn]os?|meses|d[íi]as?)', re.I)

def _norm_nums(s):
    out = s
    for w,v in NUMWORDS.items():
        out = re.sub(r'\b'+w+r'\b', str(v), out)
    return out

def _toks(s): return re.findall(r'[a-záéíóúñü0-9]+', s.lower())

class Guardian:
    def score(self, caso):
        ors = (caso.get('senales') or {}).get('oraciones', []) or []
        frs = (caso.get('senales') or {}).get('fragmentos', []) or []
        fuentes = caso.get('fuentes') or []
        respuesta = caso.get('respuesta') or ''
        # base NLI+LLM+gap por oracion (peor)
        per = []; min_rt = 1.0
        for o in ors:
            rt = o.get('respaldo_todas', 0) or 0
            rb = max(o.get('respaldo_por_fuente', [0]) or [0])
            ct = o.get('contradiccion_todas', 0) or 0
            sim = max(o.get('similitud_por_fuente', [0]) or [0])
            gap = sim - (rb if rb > rt else rt)
            v = (o.get('llm') or {}).get('veredicto')
            m = rt if rt < rb else rb
            if v == 'contradicha': lr = 1.0
            elif v == 'no_respaldada': lr = 0.85
            elif v == 'respaldada': lr = 0.60 if m < 0.30 else (0.30 if m < 0.50 else 0.05)
            else: lr = 0.50
            s = 0.06*(1.0-rt) + 0.34*(1.0-rb) + 0.06*ct + 1.27*lr + 0.50*(gap if gap > 0 else 0)
            per.append(s)
            if rt < min_rt: min_rt = rt
        base = max(per) if per else 0.5
        if frs:
            fw = max(1.0-((f.get('respaldo_todas', 0) or 0)) for f in frs)
            alt = 0.24*fw + 0.35
            if alt > base: base = alt
        scope_hit = 0.0
        rlow = _norm_nums(respuesta.lower()); flow = _norm_nums(' '.join(fuentes).lower())
        # scope: restricción caída en la fuente más parecida con alto solape
        try:
            rtoks = set(t for t in _toks(respuesta) if len(t) >= 5)
            sb = 0.0
            for o in ors:
                sims = o.get('similitud_por_fuente', [0]) or [0]
                bi = 0
                if len(sims) == len(fuentes) and fuentes:
                    mx = sims[0]; bi = 0
                    for i, v in enumerate(sims):
                        if v > mx: mx = v; bi = i
                src = fuentes[bi] if bi < len(fuentes) else ''
                stoks = set(t for t in _toks(src) if len(t) >= 5)
                if not rtoks: continue
                ov = len(rtoks & stoks) / max(1, len(rtoks))
                if ov < 0.35: continue
                sl = src.lower(); miss = False
                for mm in RESTR.finditer(sl):
                    if mm.group(0).lower() not in rlow:
                        miss = True; break
                if not miss and DE_RE.search(sl):
                    ns = NUM_RE.findall(sl); nr = NUM_RE.findall(rlow)
                    if ns and not any(n in nr for n in ns): miss = True
                if miss:
                    sb = 0.55; break
            base += sb
            scope_hit = sb
        except Exception:
            pass
        # causal solo si NLI flojo y el LLM no lo avala con confianza alta
        try:
            if min_rt < 0.5 and CAU_RE.search(rlow) and not CAU_RE.search(flow):
                conf_max = 0.0
                for o in ors:
                    llm = o.get('llm') or {}
                    if llm.get('veredicto') == 'respaldada':
                        cf = llm.get('confianza') or 0
                        if cf > conf_max: conf_max = cf
                if conf_max < 0.92:
                    base += 0.30
        except Exception:
            pass
        # aritmética derivada: si hay número calculado (no literal pero derivable), mitigar FP del LLM
        try:
            nums_r = NUM_RE.findall(rlow)
            if nums_r:
                fflat = flow.replace(',', '.')
                src_vals = []
                for x in NUM_RE.findall(fflat):
                    try: src_vals.append(float(x.replace(',', '.')))
                    except Exception: pass
                deriv = set([round(v, 2) for v in src_vals])
                for i in range(min(len(src_vals), 12)):
                    for j in range(min(len(src_vals), 12)):
                        if i == j: continue
                        deriv.add(round(src_vals[i]+src_vals[j], 2))
                        deriv.add(round(abs(src_vals[i]-src_vals[j]), 2))
                        deriv.add(round(100-src_vals[i]-src_vals[j], 2))
                        deriv.add(round(src_vals[i]*100, 2))
                        deriv.add(round(src_vals[i]*1000, 2))
                derived = 0; missing = 0
                for n in nums_r:
                    nn = n.replace(',', '.')
                    if nn in fflat: continue
                    missing += 1
                    try:
                        if round(float(nn), 2) in deriv: derived += 1
                    except Exception: pass
                if missing and derived == missing:
                    base -= 0.45
        except Exception:
            pass
        try:
            mit_ok = True
            for o in ors:
                llm = o.get('llm') or {}
                if llm.get('veredicto') != 'respaldada' or (llm.get('confianza') or 0) < 0.95:
                    mit_ok = False; break
                if (o.get('contradiccion_todas', 0) or 0) >= 0.40:
                    mit_ok = False; break
            if mit_ok and scope_hit == 0.0:
                nums2 = NUM_RE.findall(rlow)
                if nums2:
                    fflat2 = flow.replace(',', '.')
                    allp = True
                    for n in nums2:
                        if n.replace(',', '.') not in fflat2:
                            allp = False; break
                    if allp:
                        base -= 0.40
        except Exception:
            pass
        # 1) unidad cambiada: mismo numero, distinta unidad (paginas->documentos)
        try:
            runs = set((m.group(1).replace(',', '.'), (m.group(2) or '').lower()) for m in UNIT_RE.finditer(rlow))
            funits = set((m.group(1).replace(',', '.'), (m.group(2) or '').lower()) for m in UNIT_RE.finditer(flow))
            rnums = dict()
            for n, u in runs:
                rnums.setdefault(n, set()).add(u)
            funits_by_n = dict()
            for n, u in funits:
                funits_by_n.setdefault(n, set()).add(u)
            for n, us in rnums.items():
                if n in funits_by_n and not (us & funits_by_n[n]):
                    base += 0.50
                    break
        except Exception:
            pass
        # 2) cuenta casi: numero cercano (±12%) a un derivable pero no exacto
        try:
            fflat = flow.replace(',', '.')
            sv = []
            for x in NUM_RE.findall(fflat):
                try: sv.append(float(x.replace(',', '.')))
                except Exception: pass
            deriv = set([round(v, 2) for v in sv])
            for i in range(min(len(sv), 10)):
                for j in range(min(len(sv), 10)):
                    if i == j: continue
                    deriv.add(round(sv[i]+sv[j], 2)); deriv.add(round(abs(sv[i]-sv[j]), 2))
            nearmiss = False
            for n in NUM_RE.findall(rlow):
                nn = n.replace(',', '.')
                if nn in fflat: continue
                try: v = float(nn)
                except Exception: continue
                if v == 0: continue
                for d in deriv:
                    if d and abs(v-d)/abs(d) < 0.12 and round(v, 2) != round(d, 2):
                        nearmiss = True; break
                if nearmiss: break
            if nearmiss:
                base += 0.45
        except Exception:
            pass
        # 3) matiz caido: fuente con podria/si y respuesta sin ellos pero mismo nucleo
        try:
            if HEDGE_RE.search(flow) and not HEDGE_RE.search(rlow):
                rtoks2 = set(x for x in _toks(rlow) if len(x) >= 5)
                hit = False
                for fsrc in fuentes:
                    sfl = _norm_nums(fsrc.lower())
                    if not HEDGE_RE.search(sfl): continue
                    st2 = set(x for x in _toks(sfl) if len(x) >= 5)
                    if rtoks2 and len(rtoks2 & st2)/max(1, len(rtoks2)) >= 0.45:
                        hit = True; break
                if hit:
                    base += 0.40
        except Exception:
            pass
        lo, hi = -0.57, 2.00
        s = (base - lo) / (hi - lo)
        if s < 0: s = 0.0
        if s > 1: s = 1.0
        return float(s)
