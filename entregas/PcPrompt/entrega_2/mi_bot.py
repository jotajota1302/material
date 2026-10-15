"""mi_bot.py - El Guardian.

Parte del bot de ejemplo del profesor (ejemplo_bot.py) y le anade una alarma:

1. Base del profesor (sin cambios): mira cada oracion, se queda con la fuente
   que mas la respalda y marca alucinacion si alguna oracion queda sin respaldo.

2. Alarma de condicionales: si la fuente que usa una oracion expresa una
   condicion, duda o limite ("si se aprueba", "podria", "solo"...) y la oracion
   la pierde, la respuesta convierte un "quiza" en un "seguro". Eso es
   alucinacion aunque todos los datos coincidan, y el LLM juez no lo ve.

Solo usa la libreria estandar de Python.
"""
import re

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


class Guardian:
    def score(self, caso):
        # 1. Base del profesor, sin cambios.
        peor_respaldo = 1.0
        for oracion in caso["senales"]["oraciones"]:
            mejor_fuente = max(oracion["respaldo_por_fuente"], default=0.0)
            peor_respaldo = min(peor_respaldo, mejor_fuente)
        score = 1.0 - peor_respaldo

        # 2. Alarma de condicionales.
        if pierde_condicional(caso):
            score = max(score, 0.8)
        return score
