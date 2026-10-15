"""El Guardian - mi_bot.py (rama marcos).

Solo usa Python estandar (re). No usa modelos.
Recibe caso con pregunta/fuentes/respuesta/senales y devuelve
probabilidad [0,1] de alucinacion (1 = alucina, 0 = fiel).
"""
import re

# Causas inventadas: si la respuesta las trae y las fuentes no, es sospechoso.
CAU_RE = re.compile(
    r'(gracias a|debido a|por culpa|por su mayor|por su menor|porque |ya que |esto demuestra|por lo que|esto confirma)',
    re.I,
)


class Guardian:
    def score(self, caso):
        ors = (caso.get('senales') or {}).get('oraciones', []) or []
        frs = (caso.get('senales') or {}).get('fragmentos', []) or []

        riesgos = []
        min_rt = 1.0
        for o in ors:
            # NLI: respaldo con todas juntas y con la mejor fuente por separado
            rt = o.get('respaldo_todas', 0) or 0
            rb = max(o.get('respaldo_por_fuente', [0]) or [0])
            ct = o.get('contradiccion_todas', 0) or 0
            # Similitud alta pero respaldo bajo = parafrasis que cambia el dato
            sim = max(o.get('similitud_por_fuente', [0]) or [0])
            gap = sim - (rb if rb > rt else rt)

            # LLM juez, pero desconfiando si el NLI no ve respaldo
            v = (o.get('llm') or {}).get('veredicto')
            m = rt if rt < rb else rb
            if v == 'contradicha':
                lr = 1.0
            elif v == 'no_respaldada':
                lr = 0.85
            elif v == 'respaldada':
                if m < 0.30:
                    lr = 0.60  # dice respaldada pero ningun NLI lo ve: caso Montejo
                elif m < 0.50:
                    lr = 0.30
                else:
                    lr = 0.05
            else:
                lr = 0.50

            s = 0.06 * (1.0 - rt) + 0.34 * (1.0 - rb) + 0.06 * ct + 1.27 * lr + 0.50 * (gap if gap > 0 else 0)
            riesgos.append(s)
            if rt < min_rt:
                min_rt = rt

        # Una sola frase falsa basta: nos quedamos con la peor, no el promedio
        base = max(riesgos) if riesgos else 0.5

        # Fragmentos: pillan el trozo falso dentro de una frase larga
        if frs:
            fw = max(1.0 - ((f.get('respaldo_todas', 0) or 0)) for f in frs)
            alt = 0.24 * fw + 0.35
            if alt > base:
                base = alt

        # Chequeo propio: causa añadida ("gracias a", "porque"...) solo si NLI flojo
        try:
            rtxt = (caso.get('respuesta') or '').lower()
            ftxt = ' '.join(caso.get('fuentes') or []).lower()
            if min_rt < 0.5 and CAU_RE.search(rtxt) and not CAU_RE.search(ftxt):
                base += 0.30
        except Exception:
            pass

        # Calibrado para que el corte 0.5 sea el optimo en train
        lo, hi = 0.20, 2.00
        s = (base - lo) / (hi - lo)
        if s < 0:
            s = 0.0
        if s > 1:
            s = 1.0
        return float(s)
