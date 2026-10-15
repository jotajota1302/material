"""Guardian v1 — detector de alucinaciones 'No respaldado'. Solo stdlib.

Idea central: el NLI mide si la oración *se parece / se deduce* de UNA fuente,
y un inciso inventado ("multitudinario", "gracias a", "desde 2019") apenas baja
esa cifra. Aquí medimos lo contrario: ¿qué contenido de la respuesta NO aparece
en NINGUNA fuente (unión de todas)? Eso además arregla el fallo de comparar
oración vs fuente por separado cuando una respuesta fiel mezcla varias fuentes.

Todo está escrito de forma defensiva: si algún campo no existe con el nombre
esperado, se ignora esa señal en vez de romper.
"""
import re
import unicodedata
from difflib import SequenceMatcher

# ---------------- CONFIG (ajustad con vuestro set de validación) ----------------
W_NUM = 0.85        # número/fecha/cifra que no está en las fuentes
W_ENT = 0.70        # nombre propio que no está en fuentes ni pregunta
W_MARKER = 0.70     # causa/atribución/plazo/cuantificador inventado
W_MENTE = 0.50      # adverbio en -mente no respaldado
W_WORD = 0.35       # palabra de contenido novedosa (adjetivo, sustantivo, verbo)
MAX_WORDS_COUNTED = 3
Q_DISCOUNT = 0.4    # si la palabra viene de la pregunta, pesa menos
W_NLI = 0.50        # peso del riesgo NLI (1 - respaldo) en el combinado
W_JUEZ = 0.25       # el juez LLM solo refuerza; nunca absuelve
FUZZ = 0.86         # similitud de cadenas para considerar "misma palabra"

STOP = set("""a al algo algun alguna algunas alguno algunos ante antes aqui asi aun aunque bajo
cada casi como con contra cual cuales cuando cuyo de del desde donde dos e el ella ellas ello
ellos en entre era eran es esa esas ese eso esos esta estaba estan estar estas este esto estos
fue fueron ha han hasta hay la las le les lo los mas me mi mis muy ni no nos o os otra otras otro
otros para pero poco por porque que quien quienes se sea segun ser si sido sin sobre son su sus
tambien tan tanto te tiene tienen todo todos tu tus un una unas uno unos y ya yo""".split())

# Marcadores de causa-efecto, atribución, plazo y cuantificación absoluta.
MARKERS = [
    "gracias a", "debido a", "a causa de", "por culpa de", "como resultado", "como consecuencia",
    "por lo que", "por eso", "por ello", "de modo que", "lo que permitio", "lo que provoco",
    "lo que llevo", "lo que significa", "esto significa", "esto explica", "motivo por el",
    "porque", "ya que", "puesto que", "dado que", "provoco", "causo", "ocasiono", "permitio",
    "segun", "afirmo", "afirma", "aseguro", "asegura", "declaro", "explico", "anuncio",
    "desde hace", "desde entonces", "hace anos", "durante anos", "recientemente", "actualmente",
    "siempre", "nunca", "jamas", "todos", "todas", "ninguno", "ninguna", "cada ano", "cada dia",
    "mas de", "menos de", "alrededor de", "aproximadamente", "mayoria", "millones", "miles",
    "principal", "unico", "unica", "primer", "primera", "primero", "ultimo", "ultima",
    "mejor", "peor", "mas grande", "mas importante", "famoso", "famosa", "conocido por",
]


# ------------------------------ utilidades de texto ------------------------------
def _norm(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _stem(t):
    if t.isdigit() or re.search(r"\d", t):
        return t
    if len(t) <= 6:
        return t[:-2] if t.endswith("es") and len(t) > 4 else t.rstrip("s")
    return t[:6]


def _tokens(s):
    return re.findall(r"\d+(?:[.,]\d+)?|[a-z]+", _norm(s))


def _collect_strings(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            if str(k).lower() in ("id", "url", "titulo", "title", "idx"):
                continue
            out += _collect_strings(v)
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out += _collect_strings(v)
        return out
    return []


def _split_sentences(txt):
    parts = re.split(r"(?<=[.!?])\s+|\n+", txt.strip())
    return [p for p in parts if p.strip()]


def _num_key(t):
    return t.replace(",", ".").rstrip("0").rstrip(".") if "." in t or "," in t else t


class _Vocab:
    """Conjunto de stems + tokens crudos con búsqueda tolerante."""

    def __init__(self, texts):
        self.raw = set()
        self.stems = set()
        self.nums = set()
        self.text = _norm(" ".join(texts))
        for t in _tokens(self.text):
            if re.match(r"\d", t):
                self.nums.add(_num_key(t))
            else:
                self.raw.add(t)
                self.stems.add(_stem(t))

    def has_word(self, w):
        if w in self.raw or _stem(w) in self.stems:
            return True
        if len(w) >= 6:
            for r in self.raw:
                if abs(len(r) - len(w)) <= 3 and SequenceMatcher(None, w, r).ratio() >= FUZZ:
                    return True
        return False

    def has_num(self, n):
        return _num_key(n) in self.nums

    def has_phrase(self, p):
        return re.search(r"\b" + re.escape(p) + r"\b", self.text) is not None


def _noisy_or(ps):
    r = 1.0
    for p in ps:
        r *= 1.0 - max(0.0, min(1.0, p))
    return 1.0 - r


# ------------------------------ extracción de señales ------------------------------
def _best_support(entry):
    v = entry.get("respaldo_por_fuente") or []
    return max(v) if v else 0.0


def _max_contra(entry):
    v = entry.get("contradiccion_por_fuente") or []
    return max(v) if v else 0.0


def _juez_score(caso):
    """Busca de forma tolerante una probabilidad del juez LLM (MiniMax)."""
    sen = caso.get("senales") or {}
    for k, v in sen.items():
        if k in ("oraciones", "fragmentos"):
            continue
        if any(s in str(k).lower() for s in ("minimax", "juez", "judge", "llm")):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)) and any(
                            s in str(kk).lower() for s in ("alucin", "score", "prob", "nota", "no_resp")):
                        return float(vv)
    return None


# ------------------------------------ Guardian ------------------------------------
class Guardian:
    def _sentence_lexical(self, sent, src, qv, is_first_word_cap=True):
        """Devuelve (lista de pesos de evidencia, lista de razones)."""
        ev, why = [], []
        ns = _norm(sent)

        # 1) números / fechas
        for t in re.findall(r"\d+(?:[.,]\d+)?", ns):
            if not src.has_num(t):
                ev.append(W_NUM * (Q_DISCOUNT if qv.has_num(t) else 1.0))
                why.append(f"num:{t}")

        # 2) nombres propios (mayúscula que no es inicio de oración)
        words = re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñü]+", sent)
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and len(w) > 2:
                n = _norm(w)
                if n not in STOP and not src.has_word(n):
                    ev.append(W_ENT * (Q_DISCOUNT if qv.has_word(n) else 1.0))
                    why.append(f"ent:{w}")

        # 3) marcadores causa/atribución/plazo/absolutos
        for m in MARKERS:
            if re.search(r"\b" + re.escape(m) + r"\b", ns) and not src.has_phrase(m):
                ev.append(W_MARKER * (Q_DISCOUNT if qv.has_phrase(m) else 1.0))
                why.append(f"marker:{m}")

        # 4) adverbios en -mente y palabras de contenido novedosas
        novel_words = []
        for t in _tokens(ns):
            if re.match(r"\d", t) or t in STOP or len(t) < 4:
                continue
            if src.has_word(t):
                continue
            q = Q_DISCOUNT if qv.has_word(t) else 1.0
            if t.endswith("mente") and len(t) > 7:
                ev.append(W_MENTE * q)
                why.append(f"mente:{t}")
            else:
                novel_words.append((len(t), t, q))
        novel_words.sort(reverse=True)
        for _, t, q in novel_words[:MAX_WORDS_COUNTED]:
            ev.append(W_WORD * q)
            why.append(f"word:{t}")
        return ev, why

    def explain(self, caso):
        """Útil para depurar falsos positivos/negativos en el hackathon."""
        return self._run(caso, debug=True)

    def score(self, caso):
        return self._run(caso, debug=False)

    def _run(self, caso, debug):
        src_texts = _collect_strings(caso.get("fuentes"))
        src = _Vocab(src_texts)
        qv = _Vocab(_collect_strings(caso.get("pregunta")))
        resp = " ".join(_collect_strings(caso.get("respuesta")))
        sen = caso.get("senales") or {}
        ors = sen.get("oraciones") or []

        # oraciones: preferimos las de senales si traen texto; si no, partimos nosotros
        sents = _split_sentences(resp)
        info = []
        for i, s in enumerate(sents):
            entry = ors[i] if len(ors) == len(sents) else None
            info.append((s, entry))
        if not sents and ors:
            for e in ors:
                t = " ".join(_collect_strings(e.get("texto") or e.get("oracion") or ""))
                info.append((t, e))

        per_sentence, detail = [], []
        for s, entry in info:
            ev, why = self._sentence_lexical(s, src, qv)
            lex = _noisy_or(ev)
            nli_risk = 1.0 - _best_support(entry) if entry else 0.0
            contra = _max_contra(entry) if entry else 0.0
            # NLI solo suma riesgo; un NLI alto NO absuelve a una oración con novedad léxica.
            sc = _noisy_or([lex, W_NLI * nli_risk, 0.3 * contra])
            per_sentence.append(sc)
            detail.append((s, round(sc, 3), round(lex, 3), round(nli_risk, 3), why))

        # fragmentos: el trozo peor respaldado como señal adicional (atrapa incisos)
        frag_risk = 0.0
        frs = sen.get("fragmentos") or []
        if frs:
            worst = min(_best_support(f) for f in frs)
            frag_risk = 1.0 - worst

        base = max(per_sentence) if per_sentence else 0.5
        parts = [base, 0.25 * frag_risk]
        j = _juez_score(caso)
        if j is not None:
            parts.append(W_JUEZ * max(0.0, min(1.0, j)))
        final = max(0.0, min(1.0, _noisy_or(parts)))
        if debug:
            return final, detail, {"frag_risk": frag_risk, "juez": j}
        return final
