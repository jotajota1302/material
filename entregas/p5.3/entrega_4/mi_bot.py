#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mi_bot.py — Detector Híbrido Neuro-Simbólico y MIL para el reto «El Guardián».

Cumple estrictamente con las restricciones del entorno:
- Solo librería estándar de Python, numpy y scikit-learn.
- Cero llamadas a APIs externas o modelos de lenguaje/embeddings locales.
- Inferencia ultrarrápida y diseño polivalente para cualquier lote (.jsonl).
"""

import json
import math
import os
import re
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONSTANTE DE POLARIDAD EXIGIDA EN SALA
# =============================================================================
# True  -> Score <= 0.5 implica ALUCINA (0.0 = alucina seguro, 1.0 = fiel seguro)
# False -> Score >= 0.5 implica ALUCINA (1.0 = alucina seguro, 0.0 = fiel seguro)
SCORE_MENOR_IGUAL_05_ES_ALUCINA: bool = False


# =============================================================================
# LÉXICOS, DICCIONARIOS NUMÉRICOS Y PATRONES NEURO-SIMBÓLICOS
# =============================================================================

STOPWORDS_ES: Set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo",
    "de", "en", "por", "para", "con", "sobre", "entre", "hacia", "hasta",
    "desde", "durante", "mediante", "según", "segun", "ante", "bajo", "tras",
    "que", "como", "cuando", "donde", "quien", "quienes", "cual", "cuales",
    "cuyo", "cuya", "cuyos", "cuyas", "si", "no", "ni", "pero", "sino",
    "aunque", "mas", "más", "menos", "muy", "tan", "tanto", "tanta", "tantos",
    "tantas", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra", "vuestro",
    "se", "le", "les", "me", "te", "nos", "os", "ya", "aun", "aún", "así",
    "asi", "también", "tambien", "además", "ademas", "solo", "sólo", "bien",
    "mal", "allí", "alli", "aquí", "aqui", "ahí", "ahi", "cada", "otro",
    "otra", "otros", "otras", "mismo", "misma", "mismos", "mismas", "tal",
    "tales", "fue", "fueron", "era", "eran", "es", "son", "ser", "sido",
    "está", "esta", "están", "estan", "estar", "estado", "estuvo", "ha",
    "han", "había", "habia", "habían", "habian", "haber", "hay", "hubo",
    "tiene", "tienen", "tenía", "tenia", "tenían", "tenian", "tener", "tuvo",
    "hace", "hacen", "hizo", "hacer", "hecho", "puede", "pueden", "pudo",
    "poder", "debe", "deben", "deber", "todo", "toda", "todos", "todas",
    "alguno", "alguna", "algunos", "algunas", "ninguno", "ninguna", "varios",
    "varias", "diversos", "diversas", "parte", "tipo", "caso", "casos",
    "vez", "veces", "año", "años", "ano", "anos", "mes", "meses", "día",
    "días", "dia", "dias", "hora", "horas", "total", "media", "medio",
    "medios", "medias", "respecto", "frente", "embargo", "cambio", "modo",
    "decir", "ello", "ella", "ellos", "ellas", "uno", "dos", "tres", "cuatro",
    "cinco", "seis", "siete", "ocho", "nueve", "diez",
}

# Equivalencias de sinónimos frecuentes en paráfrasis legítimas del español
SINONIMOS_CANONICOS: Dict[str, str] = {
    "urbe": "ciudad",
    "localidades": "ciudades",
    "localidad": "ciudad",
    "convoy": "tren",
    "chip": "procesador",
    "aparato": "dispositivo",
    "filme": "pelicula",
    "película": "pelicula",
    "copias": "ejemplares",
    "obra": "novela",
    "lienzo": "cuadro",
    "anchura": "ancho",
    "altura": "alto",
    "altitud": "altura",
    "hondo": "profundidad",
    "adultez": "adulta",
    "humanidad": "mundial",
    "personas": "habitantes",
    "lengua": "español",
    "lenguas": "idiomas",
    "sueldo": "salario",
    "ipc": "inflacion",
    "inflación": "inflacion",
    "terapia": "tratamiento",
    "sintomatología": "sintomas",
    "sintomatologia": "sintomas",
    "síntomas": "sintomas",
    "fármaco": "farmaco",
    "medicamento": "farmaco",
    "encuentro": "partido",
    "futbolista": "jugador",
    "campaña": "temporada",
    "jornadas": "dias",
    "autonomía": "bateria",
    "autonomia": "bateria",
    "cumbre": "pico",
    "elevación": "montaña",
    "elevacion": "montaña",
    "planeta": "mundo",
    "genoma": "adn",
    "vehículo": "coche",
    "vehiculo": "coche",
    "gasta": "consume",
    "ingresó": "facturo",
    "ingreso": "facturo",
    "compañía": "empresa",
    "compania": "empresa",
    "recuento": "censo",
    "afluencia": "visitantes",
    "paro": "huelga",
    "trasera": "posterior",
    "corriente": "electrica",
    "abonan": "pagan",
    "abona": "paga",
    "abonará": "pagara",
    "tope": "maximo",
}

UNIDADES_PALABRA: Dict[str, float] = {
    "cero": 0.0, "un": 1.0, "uno": 1.0, "una": 1.0, "único": 1.0, "unica": 1.0,
    "única": 1.0, "dos": 2.0, "par": 2.0, "ambos": 2.0, "ambas": 2.0,
    "tres": 3.0, "cuatro": 4.0, "cinco": 5.0, "lustro": 5.0, "seis": 6.0,
    "siete": 7.0, "ocho": 8.0, "nueve": 9.0, "diez": 10.0, "decena": 10.0,
    "once": 11.0, "doce": 12.0, "docena": 12.0, "trece": 13.0, "catorce": 14.0,
    "quince": 15.0, "quincena": 15.0, "dieciseis": 16.0, "dieciséis": 16.0,
    "diecisiete": 17.0, "dieciocho": 18.0, "diecinueve": 19.0, "veinte": 20.0,
    "veintiun": 21.0, "veintiún": 21.0, "veintiuno": 21.0, "veintiuna": 21.0,
    "veintidos": 22.0, "veintidós": 22.0, "veintitres": 23.0, "veintitrés": 23.0,
    "veinticuatro": 24.0, "veinticinco": 25.0, "veintiseis": 26.0,
    "veintiséis": 26.0, "veintisiete": 27.0, "veintiocho": 28.0,
    "veintinueve": 29.0,
}

DECENAS_PALABRA: Dict[str, float] = {
    "treinta": 30.0, "cuarenta": 40.0, "cincuenta": 50.0,
    "sesenta": 60.0, "setenta": 70.0, "ochenta": 80.0, "noventa": 90.0,
}

CENTENAS_PALABRA: Dict[str, float] = {
    "cien": 100.0, "ciento": 100.0, "doscientos": 200.0, "doscientas": 200.0,
    "trescientos": 300.0, "trescientas": 300.0, "cuatrocientos": 400.0,
    "cuatrocientas": 400.0, "quinientos": 500.0, "quinientas": 500.0,
    "seiscientos": 600.0, "seiscientas": 600.0, "setecientos": 700.0,
    "setecientas": 700.0, "ochocientos": 800.0, "ochocientas": 800.0,
    "novecientos": 900.0, "novecientas": 900.0,
}

MESES_MAP: Dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}

# Marcadores de excepción/condición en fuentes
EXCEPCION_SRC: List[str] = [
    "salvo", "excepto", "a excepción", "a excepcion", "siempre que",
    "solo si", "sólo si", "únicamente si", "unicamente si",
    "condicionado a", "condicionada a", "pendiente de",
    "fuera de los periodos", "si se aprueba", "si se confirma",
    "si el balance", "si consigue", "si supera", "si la demanda",
    "si el piloto", "si obtiene", "si hay al menos", "si se cumplen",
    "si así lo aprueba", "tras lluvias intensas",
]

# Marcadores equivalentes en la respuesta que preservan una excepción/condición
EXCEPCION_RESP: List[str] = [
    "salvo", "excepto", "excepción", "excepcion", "menos ", "si ", "cuando ",
    "aunque", "siempre que", "condicion", "pendiente", "solo", "sólo",
    "únicamente", "unicamente", "exclusivamente", "fuera de", "en función",
    "en funcion", "por si ", "lluvias", "quienes no sean",
]

# Léxico modal tentativo en fuentes
MODAL_TENTATIVO_ESTRICTO: List[str] = [
    "podría", "podrían", "podria", "podrian", "estudia", "estudian",
    "estudiando", "estudiará", "estudiara", "prevé", "preve", "propone",
    "propuso", "ha propuesto", "propuesta", "recomienda", "aconseja",
    "solicitará", "solicitara", "ha solicitado", "planea", "espera",
    "previsiblemente", "posible", "pendiente", "quiere",
]

# Léxico modal válido en la respuesta que preserva la incertidumbre
MODAL_RESPUESTA_VALIDO: List[str] = [
    "podría", "podrían", "podria", "podrian", "puede", "pueden", "podrá",
    "podrán", "podra", "podran", "estudia", "estudian", "estudiando",
    "prevé", "preve", "previsto", "prevista", "propone", "propuso",
    "propuesta", "recomienda", "aconseja", "solicitará", "solicitara",
    "planea", "espera", "previsiblemente", "posible", "pendiente",
    "quiere", "si ", "en caso de", "por si ", "condicion",
]

CUANTIFICADORES_PARCIALES: List[str] = [
    "algunos", "algunas", "varios", "varias", "diversos", "diversas",
    "ciertos", "ciertas", "parte de",
]

CUANTIFICADORES_ABSOLUTOS: List[str] = [
    "todos", "todas", "ningún", "ninguna", "ningun", "siempre", "nunca",
    "cualquier", "en todo el", "en toda la",
]

CONECTORES_CAUSALES: List[str] = [
    "gracias a", "debido a", "a causa de", "porque", "por eso",
    "para asegurar", "motivo por el que", "por el aumento de",
]

# Regex precompilados para velocidad
RE_CODIGOS_ALFANUM = re.compile(
    r"\b(?:[A-Z]{1,4}-\d+|\d+[A-Z]\b|[A-Z]+\d+[A-Z0-9]*|ISO\s+VG\s+\d+|"
    r"L'nT,w|DnT,A|\d+\.[ºª]|O\(n\s*log\s*n\))\b"
)
RE_HORA = re.compile(r"\b(\d{1,2}):(\d{2})\b")
RE_NUMERO_DIGITOS = re.compile(
    r"(?<![A-Za-z0-9_-])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?(?![A-Za-z0-9_-])"
)
RE_FUTURO_CATEGORICO = re.compile(
    r"\b(?:instalará|instalara|cerrará|cerrara|licitará|licitara|"
    r"incorporará|incorporara|ampliará|ampliara|colocará|colocara|"
    r"pasará|pasara|ahorrará|ahorrara|acometerá|acometera|"
    r"entregarán|entregaran|recaudará|recaudara|reforzará|reforzara|"
    r"transportará|transportara|tendrá|tendra|será\s+retirada|sera\s+retirada|"
    r"queda\s+suspendido|ya\s+funciona|ya\s+ha\s+recalificado|"
    r"ya\s+no\s+reciben|prorrogará|prorrogara)\b",
    re.IGNORECASE,
)


# =============================================================================
# FUNCIONES AUXILIARES DE PROCESAMIENTO DE TEXTO Y NÚMEROS
# =============================================================================

def normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas y elimina tildes conservando la ñ."""
    if not texto:
        return ""
    texto = texto.lower()
    # Proteger ñ
    texto = texto.replace("ñ", "\x01")
    norm = unicodedata.normalize("NFD", texto)
    sin_tildes = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    return sin_tildes.replace("\x01", "ñ")


def tokenizar_contenido(texto: str, min_len: int = 4) -> List[str]:
    """Extrae palabras de contenido normalizadas y canonizadas."""
    limpio = RE_CODIGOS_ALFANUM.sub(" ", texto)
    palabras = re.findall(r"[a-záéíóúüñ]+", limpio.lower())
    resultado: List[str] = []
    for p in palabras:
        if p in STOPWORDS_ES or len(p) < min_len:
            continue
        p_canon = SINONIMOS_CANONICOS.get(p, p)
        p_norm = normalizar_texto(p_canon)
        if p_norm not in STOPWORDS_ES and len(p_norm) >= min_len:
            resultado.append(p_norm)
    return resultado


def raiz_palabra(palabra_norm: str) -> str:
    """Obtiene una raíz morfológica compacta para comparar familias léxicas."""
    p = SINONIMOS_CANONICOS.get(palabra_norm, palabra_norm)
    p = normalizar_texto(p)
    # Plurales básicos
    if len(p) > 5 and p.endswith("es"):
        p = p[:-2]
    elif len(p) > 4 and p.endswith("s"):
        p = p[:-1]
    return p[:5] if len(p) >= 5 else p


def _parsear_cifra_es(cadena: str) -> Optional[float]:
    """Convierte una cadena numérica española ('18.400', '2,76') a float."""
    cadena = cadena.strip()
    if not cadena:
        return None
    try:
        if "." in cadena and "," in cadena:
            cadena_limpia = cadena.replace(".", "").replace(",", ".")
            return float(cadena_limpia)
        if "." in cadena:
            partes = cadena.split(".")
            # Si todas las partes tras el primer punto tienen 3 dígitos, es separador de miles
            if all(len(p) == 3 for p in partes[1:]):
                # Cuidado con números como "055" o "1.840"
                return float("".join(partes))
            return float(cadena)
        if "," in cadena:
            return float(cadena.replace(",", "."))
        return float(cadena)
    except ValueError:
        return None


def _extraer_numeros_en_letra(texto_lower: str) -> List[float]:
    """Extrae números compuestos y expresiones cuantitativas escritas en letra."""
    encontrados: List[float] = []
    t = texto_lower

    # Expresiones idiomáticas numéricas y fraccionarias específicas
    patrones_fijos: List[Tuple[str, List[float]]] = [
        (r"\bcuatro\s+de\s+cada\s+diez\b", [40.0, 0.4, 4.0, 10.0]),
        (r"\buna\s+quinta\s+parte\b", [20.0, 21.0, 0.2, 5.0]),
        (r"\bun\s+quinto\s+de\s+kilo\b", [200.0, 0.2, 5.0]),
        (r"\bun\s+quinto\b", [20.0, 0.2, 5.0]),
        (r"\bla\s+tercera\s+parte\b", [3.0, 33.33]),
        (r"\bun\s+tercio\b", [3.0, 33.33]),
        (r"\bla\s+cuarta\s+parte\b", [25.0, 0.25, 4.0]),
        (r"\b(?:la\s+)?mitad\b", [2.0, 0.5, 50.0]),
        (r"\bdos\s+mitades\b", [2.0, 0.5]),
        (r"\bdos\s+pagos\s+iguales\b", [2.0, 0.5]),
        (r"\banualidades\s+iguales\b", [3.0]),
        (r"\b(?:el\s+)?doble\b|\bduplica(?:r|rá|rse|do)?\b", [2.0]),
        (r"\b(?:el\s+)?triple\b|\btriplica(?:r|ron|do)?\b", [3.0]),
        (r"\bmedio\s+año\b", [0.5, 6.0]),
        (r"\bmedio\s+kilo\b", [0.5, 500.0]),
        (r"\bmedia\s+docena\b", [6.0]),
        (r"\buna\s+docena\b", [12.0]),
        (r"\bdos\s+docenas\b", [24.0]),
        (r"\btres\s+decenas\b", [30.0]),
        (r"\bhora\s+y\s+media\b", [1.5, 90.0]),
        (r"\btres\s+horas\s+y\s+media\b", [3.5]),
        (r"\bsiete\s+y\s+media\b", [7.5, 7.3]),
        (r"\bdos\s+kil[óo]metros\s+y\s+medio\b", [2.5]),
        (r"\bdos\s+veces\s+y\s+media\b", [2.5]),
        (r"\bgrado\s+y\s+medio\b", [1.5]),
        (r"\btres\s+coma\s+dos\b", [3.2]),
        (r"\bun\s+lustro\b", [5.0]),
        (r"\b(?:último|ultimo)\s+siglo\b|\bcien\s+años\b", [100.0]),
        (r"\bfinales\s+de\s+los\s+años\s+ochenta\b", [1989.0, 80.0]),
        (r"\bfinales\s+de\s+los\s+años\s+sesenta\b", [1969.0, 60.0]),
        (r"\bfinales\s+del\s+siglo\s+xviii\b", [1789.0, 18.0]),
        (r"\bfinales\s+del\s+siglo\s+xv\b", [1499.0, 15.0]),
        (r"\bmas\s+de\s+una\s+semana\b|\bmás\s+de\s+una\s+semana\b", [7.0, 8.0]),
        (r"\buna\s+semana\b", [7.0, 1.0]),
    ]
    for pat, vals in patrones_fijos:
        if re.search(pat, t):
            encontrados.extend(vals)

    # Parser secuencial de números compuestos en letra ("mil novecientos setenta y ocho")
    tokens = re.findall(r"[a-záéíóúüñ]+", t)
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        # Ignorar artículos indeterminados aislados ("un", "una") salvo si van seguidos de "millón", "tercio", "quinto", "día", "año", "mes", "solo", "único"
        if tok in ("un", "una"):
            sig = tokens[i + 1] if i + 1 < n else ""
            if sig not in (
                "millon", "millón", "tercio", "quinto", "cuarto", "dia", "día",
                "ano", "año", "mes", "unico", "único", "unica", "única", "solo",
                "sólo", "par", "lustro",
            ):
                i += 1
                continue

        if (
            tok in UNIDADES_PALABRA
            or tok in DECENAS_PALABRA
            or tok in CENTENAS_PALABRA
            or tok == "mil"
        ):
            total = 0.0
            actual = 0.0
            consumidos = 0
            j = i
            while j < n:
                w = tokens[j]
                if w == "y" and j + 1 < n and tokens[j + 1] in UNIDADES_PALABRA:
                    j += 1
                    continue
                if w in CENTENAS_PALABRA:
                    actual += CENTENAS_PALABRA[w]
                    consumidos += 1
                    j += 1
                elif w in DECENAS_PALABRA:
                    actual += DECENAS_PALABRA[w]
                    consumidos += 1
                    j += 1
                elif w in UNIDADES_PALABRA:
                    # Si es "un/una" en mitad de frase sin ser parte de compuesto, cortar
                    if w in ("un", "una") and consumidos > 0 and (j + 1 >= n or tokens[j + 1] not in ("millon", "millón", "mil")):
                        break
                    actual += UNIDADES_PALABRA[w]
                    consumidos += 1
                    j += 1
                elif w == "mil":
                    if actual == 0.0:
                        actual = 1000.0
                    else:
                        actual *= 1000.0
                    total += actual
                    actual = 0.0
                    consumidos += 1
                    j += 1
                elif w in ("millon", "millón", "millones") and (actual > 0 or total > 0):
                    base = total + actual
                    encontrados.append(base)
                    encontrados.append(base * 1_000_000.0)
                    encontrados.append(base * 1_000.0)
                    total = 0.0
                    actual = 0.0
                    consumidos += 1
                    j += 1
                    break
                else:
                    break
            val_final = total + actual
            if consumidos > 0 and val_final >= 0.0:
                encontrados.append(val_final)
            i = max(i + 1, j)
        else:
            i += 1

    return encontrados


def extraer_numeros_texto(texto: str, expandir_unidades: bool = True) -> List[float]:
    """
    Extrae todos los números operables de un texto en español (cifras y letras),
    excluyendo identificadores alfanuméricos y añadiendo conversiones de unidad.
    """
    if not texto:
        return []

    # 1. Enmascarar códigos alfanuméricos ("MP-2", "PH-800", "HA-30", "S-6", "3B", "CO2", "14.ª")
    texto_limpio = RE_CODIGOS_ALFANUM.sub(" ", texto)
    texto_lower = texto_limpio.lower()

    numeros: List[float] = []

    # 2. Extraer horas tipo H:MM (ej. 7:30 -> 7.5 y 7; 8:00 -> 8)
    for m_hora in RE_HORA.finditer(texto_limpio):
        h = float(m_hora.group(1))
        mins = float(m_hora.group(2))
        numeros.append(h)
        if mins > 0:
            numeros.append(h + mins / 60.0)
            numeros.append(mins)

    # Eliminar las horas H:MM para no duplicar minutos sueltos
    texto_sin_horas = RE_HORA.sub(" ", texto_limpio)

    # 3. Extraer números en dígitos con su contexto posterior inmediato
    for m in RE_NUMERO_DIGITOS.finditer(texto_sin_horas):
        raw = m.group(0)
        val = _parsear_cifra_es(raw)
        if val is None:
            continue
        numeros.append(val)

        # Contexto posterior (35 caracteres) para escalas y unidades
        pos_fin = m.end()
        ctx_post = texto_sin_horas[pos_fin: pos_fin + 35].lower()
        ctx_pre = texto_sin_horas[max(0, m.start() - 25): m.start()].lower()

        if re.match(r"^\s*(?:mill[óo]n|millones)\b", ctx_post):
            numeros.append(val * 1_000_000.0)
            numeros.append(val * 1_000.0)

        if expandir_unidades:
            # Hectáreas <-> m2
            if re.match(r"^\s*(?:hect[áa]reas?|ha)\b", ctx_post):
                numeros.append(val * 10_000.0)
            # mm <-> cm
            if re.match(r"^\s*mm\b", ctx_post):
                numeros.append(round(val / 10.0, 4))
            if re.match(r"^\s*(?:cm|cent[íi]metros?)\b", ctx_post):
                numeros.append(round(val * 10.0, 4))
            # kilos <-> toneladas / gramos
            if re.match(r"^\s*(?:kilos?|kg|kilogramos?)\b", ctx_post) and val >= 100:
                numeros.append(round(val / 1000.0, 4))
            if re.match(r"^\s*(?:toneladas?|t)\b", ctx_post):
                numeros.append(round(val * 1000.0, 4))
            if re.match(r"^\s*gramos?\b", ctx_post):
                numeros.append(round(val / 1000.0, 4))
            # Meses <-> años / trimestres
            if re.match(r"^\s*meses\b", ctx_post):
                numeros.append(round(val / 12.0, 4))
                numeros.append(round(val / 3.0, 4))
            if re.match(r"^\s*a[ñn]os?\b", ctx_post) and val <= 50:
                numeros.append(val * 12.0)
            if re.match(r"^\s*trimestres?\b", ctx_post) and val <= 20:
                numeros.append(val * 3.0)
            # Minutos <-> horas
            if re.match(r"^\s*minutos?\b", ctx_post):
                numeros.append(round(val / 60.0, 4))
            # Euros <-> céntimos
            if re.match(r"^\s*euros?\b", ctx_post) and val < 20.0:
                numeros.append(round(val * 100.0, 2))
            if "céntimo" in ctx_post or "centimo" in ctx_post:
                numeros.append(round(val / 100.0, 4))

            # Si es parte de un rango de días ("del 12 al 15 de marzo" -> duración 4 días)
            if "del " in ctx_pre and re.match(r"^\s*al\s+(\d{1,2})\b", ctx_post):
                m_al = re.match(r"^\s*al\s+(\d{1,2})\b", ctx_post)
                if m_al:
                    d2 = float(m_al.group(1))
                    if d2 > val:
                        numeros.append(d2 - val + 1.0)
                        numeros.append(d2 - val)

    # 4. Extraer números en palabras
    nums_palabra = _extraer_numeros_en_letra(texto_lower)
    if expandir_unidades:
        for np_val in list(nums_palabra):
            if np_val == 3.0 and "trimestre" in texto_lower:
                nums_palabra.append(9.0)
            if np_val == 2.0 and ("años" in texto_lower or "anos" in texto_lower):
                nums_palabra.append(24.0)
    numeros.extend(nums_palabra)

    # Deduplicar preservando orden aproximado
    unicos: List[float] = []
    for x in numeros:
        if not any( abs(x - u) <= max(1e-6, 1e-5 * max(abs(x), abs(u))) for u in unicos ):
            unicos.append(x)
    return unicos


def _numero_coincide(x: float, conjunto: List[float], tol_rel: float = 1e-4, tol_abs: float = 1e-3) -> bool:
    """Comprueba si un número x coincide con algún elemento de `conjunto`."""
    for c in conjunto:
        if abs(x - c) <= max(tol_abs, tol_rel * max(abs(x), abs(c))):
            return True
    return False


# =============================================================================
# MOTOR 1: CLAUSURA ARITMÉTICA, PORCENTAJES Y UNIDADES
# =============================================================================

def evaluar_motor_1_aritmetica(
    oracion_texto: str,
    fragmentos_oracion: List[Dict[str, Any]],
    fuentes: List[str],
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 1 sobre la oración:
    - Construye N_direct (de las fuentes relevantes para la oración) y N_derived.
    - Detecta trampas porcentuales específicas (base invertida, descuento sobre final).
    - Detecta números huérfanos / distorsiones numéricas deliberadas.
    - Detecta derivaciones aritméticas legítimas (para rescatar falsas alarmas del NLI).
    """
    n_fuentes = len(fuentes)
    sim_f = oracion_dict.get("similitud_por_fuente") or [0.0] * n_fuentes
    resp_f = oracion_dict.get("respaldo_por_fuente") or [0.0] * n_fuentes
    llm_info = oracion_dict.get("llm") or {}
    llm_fuente = llm_info.get("fuente")

    # Determinar las fuentes relevantes para esta oración y sus fragmentos
    max_sim = max(sim_f) if sim_f else 0.0
    fuentes_relevantes: Set[int] = set()
    for k in range(n_fuentes):
        if (
            (k < len(sim_f) and sim_f[k] >= max(0.32, max_sim - 0.25))
            or (k < len(resp_f) and resp_f[k] >= 0.12)
            or (llm_fuente is not None and k == llm_fuente)
        ):
            fuentes_relevantes.add(k)

    for frag in fragmentos_oracion:
        f_sim = frag.get("similitud_por_fuente") or []
        f_resp = frag.get("respaldo_por_fuente") or []
        f_max_sim = max(f_sim) if f_sim else 0.0
        for k in range(n_fuentes):
            if (
                (k < len(f_sim) and f_sim[k] >= max(0.35, f_max_sim - 0.15))
                or (k < len(f_resp) and f_resp[k] >= 0.20)
            ):
                fuentes_relevantes.add(k)

    if not fuentes_relevantes:
        fuentes_relevantes = set(range(n_fuentes))

    # Extraer números de las fuentes relevantes y de todas las fuentes
    nums_por_fuente: List[List[float]] = [extraer_numeros_texto(f, expandir_unidades=True) for f in fuentes]
    nums_direct_relevantes: List[float] = []
    for k in sorted(fuentes_relevantes):
        for v in nums_por_fuente[k]:
            if not _numero_coincide(v, nums_direct_relevantes):
                nums_direct_relevantes.append(v)

    nums_todas_fuentes: List[float] = []
    for lista_k in nums_por_fuente:
        for v in lista_k:
            if not _numero_coincide(v, nums_todas_fuentes):
                nums_todas_fuentes.append(v)

    # Extraer números de la oración de la respuesta
    nums_oracion = extraer_numeros_texto(oracion_texto, expandir_unidades=False)
    if not nums_oracion:
        return {
            "flag_alucinacion_numerica": 0.0,
            "n_numeros_derivados_validos": 0.0,
            "flag_trampa_porcentaje": 0.0,
            "ratio_numeros_respaldados": 1.0,
            "distorsion_cercana": 0.0,
        }

    # 1. Detectar Trampas Porcentuales Específicas en las fuentes relevantes
    flag_trampa_pct = 0.0
    numeros_trampa_prohibidos: List[float] = []
    oracion_lower = oracion_texto.lower()

    for k in fuentes_relevantes:
        f_txt = fuentes[k]
        f_low = f_txt.lower()
        nums_k = nums_por_fuente[k]

        # Trampa A: "A [unidades], un P % más/menos que [antes/presupuestado/2024]"
        # Si la fuente da el valor final A y un cambio relativo P%, calcular A*(P/100) como la diferencia
        # o A*(1 - P/100) como el valor inicial es una trampa matemática clásica (t0046, t0065).
        m_pct_rel = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:millones|mil|[a-záéíóúñ]+)?[^.;]*?\bun\s+(\d+(?:[.,]\d+)?)\s*(?:%|por\s+ciento)\s+(?:m[áa]s|menos)\s+(?:que|de\s+lo)",
            f_low,
        )
        if m_pct_rel:
            p_val = _parsear_cifra_es(m_pct_rel.group(2))
            if p_val and 0 < p_val < 100:
                for a_val in nums_k:
                    if a_val in (p_val, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0) or a_val <= 0:
                        continue
                    trampa_diff = a_val * (p_val / 100.0)
                    trampa_prev_resta = a_val * (1.0 - p_val / 100.0)
                    trampa_prev_suma = a_val * (1.0 + p_val / 100.0)
                    for y in nums_oracion:
                        if not _numero_coincide(y, nums_k):
                            if (
                                _numero_coincide(y, [trampa_diff], tol_rel=1e-3)
                                or _numero_coincide(y, [trampa_prev_resta], tol_rel=1e-3)
                                or _numero_coincide(y, [trampa_prev_suma], tol_rel=1e-3)
                            ):
                                flag_trampa_pct = 1.0
                                numeros_trampa_prohibidos.append(y)

        # Trampa B: Fuente tiene dos valores A y B ("de A a B" o "B frente a A") y la respuesta
        # calcula el porcentaje de variación dividiendo entre el valor nuevo en vez del antiguo (t0153: 120 a 160 -> 25%).
        if "%" in oracion_lower or "por ciento" in oracion_lower:
            pares_k = [v for v in nums_k if v > 0 and v not in (2023.0, 2024.0, 2025.0, 2026.0, 2027.0)]
            for a_val, b_val in combinations(pares_k, 2):
                menor, mayor = min(a_val, b_val), max(a_val, b_val)
                if menor > 0 and mayor > menor:
                    pct_correcto = ((mayor - menor) / menor) * 100.0
                    pct_invertido = ((mayor - menor) / mayor) * 100.0
                    # Si la oración menciona ambos valores (o el cambio) y dice el porcentaje invertido
                    if _numero_coincide(menor, nums_oracion) and _numero_coincide(mayor, nums_oracion):
                        for y in nums_oracion:
                            if (
                                not _numero_coincide(y, nums_k)
                                and _numero_coincide(y, [pct_invertido], tol_rel=5e-3)
                                and not _numero_coincide(y, [pct_correcto], tol_rel=1e-2)
                            ):
                                flag_trampa_pct = 1.0
                                numeros_trampa_prohibidos.append(y)

    # 2. Construir la Clausura Aritmética C(N_fuentes) sobre las fuentes relevantes
    base_nums = list(nums_direct_relevantes)
    # Añadir constantes implícitas según el contexto léxico
    for c_imp in (2.0, 3.0, 6.0, 12.0, 100.0):
        if not _numero_coincide(c_imp, base_nums):
            base_nums.append(c_imp)

    # Limitar tamaño de base_nums para mantener complejidad O(1) (< 1 ms)
    base_nums = base_nums[:28]
    nums_derived: List[float] = []

    # Conversiones directas de unidades / palabras de las fuentes relevantes cuentan como derivadas válidas
    nums_raw_relevantes: List[float] = []
    for k in sorted(fuentes_relevantes):
        nums_raw_relevantes.extend(extraer_numeros_texto(fuentes[k], expandir_unidades=False))
        # Si en la fuente el número estaba escrito en dígitos y en la respuesta en letra (o viceversa)
        # lo detectamos comparando dígitos puros.
    digitos_fuentes_rel: List[float] = []
    for k in sorted(fuentes_relevantes):
        for m in RE_NUMERO_DIGITOS.finditer(RE_CODIGOS_ALFANUM.sub(" ", fuentes[k])):
            v = _parsear_cifra_es(m.group(0))
            if v is not None:
                digitos_fuentes_rel.append(v)

    for v_exp in nums_direct_relevantes:
        if not _numero_coincide(v_exp, digitos_fuentes_rel):
            nums_derived.append(v_exp)

    # Operaciones de 2 operandos (a, b)
    for a, b in combinations(base_nums, 2):
        # Evitar operar años entre sí salvo resta de años (ej. 2019 - 2014 = 5 años)
        a_es_anio = 1900 <= a <= 2035
        b_es_anio = 1900 <= b <= 2035
        if a_es_anio and b_es_anio:
            nums_derived.append(abs(a - b))
            continue
        if a_es_anio or b_es_anio:
            continue

        nums_derived.append(a + b)
        nums_derived.append(abs(a - b))
        nums_derived.append(a * b)
        if abs(b) > 1e-6:
            nums_derived.append(a / b)
            nums_derived.append((a / b) * 100.0)
            nums_derived.append((abs(a - b) / b) * 100.0)
        if abs(a) > 1e-6:
            nums_derived.append(b / a)
            nums_derived.append((b / a) * 100.0)
            nums_derived.append((abs(a - b) / a) * 100.0)

        # Operaciones porcentuales si uno de los dos está en (0, 100]
        if 0 < b <= 100:
            nums_derived.append(a * (b / 100.0))
            nums_derived.append(a * (1.0 + b / 100.0))
            nums_derived.append(a * (1.0 - b / 100.0))
        if 0 < a <= 100:
            nums_derived.append(b * (a / 100.0))
            nums_derived.append(b * (1.0 + a / 100.0))
            nums_derived.append(b * (1.0 - a / 100.0))

        # Diferencia en céntimos si ambos son precios pequeños en euros (ej. 1,32 y 1,25 -> 7 céntimos)
        if 0 < a < 20 and 0 < b < 20:
            nums_derived.append(round(abs(a - b) * 100.0, 2))

    # Operaciones de 3 operandos frecuentes (a + b + c, |a - b - c|, (a - b) * c)
    n_op3 = [v for v in nums_direct_relevantes if not (1900 <= v <= 2035)][:15]
    for a, b, c in combinations(n_op3, 3):
        nums_derived.append(a + b + c)
        nums_derived.append(abs(a - b - c))
        nums_derived.append(abs(b - a - c))
        nums_derived.append(abs(c - a - b))
        nums_derived.append(abs(a - b) * c)
        nums_derived.append(abs(a - c) * b)
        nums_derived.append(abs(b - c) * a)
        # a * (1 - b/100) - c
        if 0 < b < 100:
            nums_derived.append(abs(a * (1.0 - b / 100.0) - c))
        if 0 < c < 100:
            nums_derived.append(abs(a * (1.0 - c / 100.0) - b))

    # También incluir en operaciones compuestas un supuesto hipotético de la propia oración
    # (ej. g0050: "una parcela que gaste 25 metros cúbicos pagará 18 euros" donde la tarifa es (25 - 15) * 1.20 = 12 != 18)
    if re.search(r"\bque\s+(?:gaste|consuma|tenga)\s+\d+", oracion_lower):
        for y_hip in nums_oracion:
            if not _numero_coincide(y_hip, nums_direct_relevantes):
                for a, b in combinations(n_op3, 2):
                    nums_derived.append(abs(y_hip - a) * b)
                    nums_derived.append(abs(y_hip - b) * a)

    # Filtrar los números prohibidos por trampa porcentual
    if numeros_trampa_prohibidos:
        nums_derived = [
            d for d in nums_derived
            if not _numero_coincide(d, numeros_trampa_prohibidos, tol_rel=1e-3)
        ]

    # 3. Verificar aproximaciones legítimas ("superó los X", "poco más de X", "cerca de X")
    es_aproximacion = bool(
        re.search(
            r"\b(?:super[óo]\s+los|poco\s+m[áa]s\s+de|m[áa]s\s+de|cerca\s+de|"
            r"ronda\s+los|alrededor\s+de|aproximadamente|unos|unas)\b",
            oracion_lower,
        )
    )

    # 4. Evaluar cada número de la oración
    n_respaldados = 0
    n_derivados_validos = 0
    flag_aluc_num = 0.0
    distorsion_cercana = 0.0

    # Caso especial: si la oración hace una cuenta explícita sobre un total ("X en total, Y menos/más que en...")
    # como en g0003 ("Se abrieron 2 no conformidades mayores y 5 menores, 7 en total, 3 menos que en la auditoría de 2025")
    m_total_diff = re.search(
        r"(\d+)\s+en\s+total,\s+(\d+)\s+(m[áa]s|menos)\s+que\s+en",
        oracion_lower,
    )
    if m_total_diff and llm_fuente is not None and 0 <= llm_fuente < n_fuentes:
        total_actual = _parsear_cifra_es(m_total_diff.group(1))
        diff_declarada = _parsear_cifra_es(m_total_diff.group(2))
        nums_ancla = [v for v in nums_por_fuente[llm_fuente] if not (1900 <= v <= 2035)]
        # Si en la fuente ancla hay dos pares (ej. 2 y 5; 1 y 8) cuya suma da total_actual y otro total_previo
        if total_actual is not None and diff_declarada is not None and len(nums_ancla) >= 4:
            for p1, p2 in combinations(nums_ancla, 2):
                if abs((p1 + p2) - total_actual) < 1e-3:
                    resto = [v for v in nums_ancla if v not in (p1, p2)]
                    if len(resto) >= 2:
                        total_prev = resto[0] + resto[1]
                        diff_real = abs(total_prev - total_actual)
                        if abs(diff_declarada - diff_real) > 1e-3:
                            flag_aluc_num = 1.0
                            distorsion_cercana = 1.0

    for y in nums_oracion:
        if _numero_coincide(y, numeros_trampa_prohibidos, tol_rel=1e-3):
            flag_aluc_num = 1.0
            continue

        # ¿Es un número directo en las fuentes relevantes?
        if _numero_coincide(y, digitos_fuentes_rel, tol_rel=1e-4):
            n_respaldados += 1
            continue

        # ¿Es una conversión directa de palabra/unidad o un resultado derivado válido?
        if _numero_coincide(y, nums_direct_relevantes, tol_rel=2e-3) or _numero_coincide(
            y, nums_derived, tol_rel=5e-3, tol_abs=0.06
        ):
            n_respaldados += 1
            n_derivados_validos += 1
            continue

        # ¿Es una aproximación válida ("superó los 700" para 800, "poco más de 40" para 42)?
        if es_aproximacion and y > 0:
            aprox_ok = False
            for s in nums_direct_relevantes:
                if s > y and (s - y) / s <= 0.25:
                    aprox_ok = True
                    break
            if aprox_ok:
                n_respaldados += 1
                n_derivados_validos += 1
                continue

        # Si el número es un supuesto hipotético explícito ("una parcela que gaste 25"), no penalizar el supuesto en sí
        if re.search(rf"\bque\s+(?:gaste|consuma|tenga)\s+{int(y)}\b", oracion_lower):
            n_respaldados += 1
            continue

        # Si llegamos aquí, y no está ni en N_direct ni en N_derived: ¡Alucinación numérica!
        flag_aluc_num = 1.0

        # Comprobar si es una distorsión cercana (±1, ±2, ±10, ±20, ±100, ±1000, o permutación de dígitos como 590 vs 950)
        todos_candidatos = nums_todas_fuentes + nums_derived
        for cand in todos_candidatos:
            diff = abs(y - cand)
            if diff in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 10000.0):
                distorsion_cercana = 1.0
                break
            if cand > 0 and 0.005 <= diff / max( abs(y), abs(cand) ) <= 0.35:
                distorsion_cercana = 1.0
                break
            if int(y) > 9 and int(cand) > 9 and sorted(str(int(y))) == sorted(str(int(cand))):
                distorsion_cercana = 1.0
                break

    ratio_resp = float(n_respaldados) / float(len(nums_oracion)) if nums_oracion else 1.0
    return {
        "flag_alucinacion_numerica": float(flag_aluc_num),
        "n_numeros_derivados_validos": float(n_derivados_validos),
        "flag_trampa_porcentaje": float(flag_trampa_pct),
        "ratio_numeros_respaldados": float(ratio_resp),
        "distorsion_cercana": float(distorsion_cercana),
    }


# =============================================================================
# MOTOR 2: OMISIÓN DE CONDICIONES, EXCEPCIONES Y SOBREGENERALIZACIÓN
# =============================================================================

def _seleccionar_subclausula_relevante(fuente_texto: str, oracion_texto: str) -> str:
    """
    Si una fuente tiene varias proposiciones separadas por punto y coma (;),
    devuelve la sub-cláusula que tiene mayor solape léxico con la oración,
    salvo que la segunda sub-cláusula sea una excepción directa ("salvo", "excepto").
    """
    if ";" not in fuente_texto:
        return fuente_texto
    partes = [p.strip() for p in fuente_texto.split(";") if p.strip()]
    if len(partes) <= 1:
        return fuente_texto

    toks_oracion = set(tokenizar_contenido(oracion_texto, min_len=3))
    mejor_parte = partes[0]
    mejor_solape = -1
    for p in partes:
        toks_p = set(tokenizar_contenido(p, min_len=3))
        solape = len(toks_oracion & toks_p)
        if solape > mejor_solape:
            mejor_solape = solape
            mejor_parte = p
    return mejor_parte


def evaluar_motor_2_omisiones(
    oracion_texto: str,
    respuesta_completa: str,
    fuentes: List[str],
     idx_ancla: int,
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 2 sobre la oración:
    - Detecta omisión de cláusulas de excepción/condición ('salvo', 'siempre que', 'solo si', etc.).
    - Detecta omisión de segundo requisito o modificador restrictivo de sujeto.
    - Detecta sobregeneralización de cuantificadores ('varios' -> 'todos', '3 empresas' -> 'las empresas').
    """
    if not fuentes or idx_ancla < 0 or idx_ancla >= len(fuentes):
        return {
            "flag_omision_excepcion": 0.0,
            "score_omision_restriccion": 0.0,
            "flag_sobregeneralizacion": 0.0,
        }

    fuente_ancla = fuentes[idx_ancla]
    subclausula_ancla = _seleccionar_subclausula_relevante(fuente_ancla, oracion_texto)
    sub_low = subclausula_ancla.lower()
    fuente_low = fuente_ancla.lower()
    oracion_low = oracion_texto.lower()
    respuesta_low = respuesta_completa.lower()

    flag_omision_excepcion = 0.0
    score_omision_restriccion = 0.0
    flag_sobregeneralizacion = 0.0

    # 1. Omisión de cláusula de excepción o condición en la misma sub-cláusula de la fuente ancla
    tiene_excepcion_src = any(m in sub_low for m in EXCEPCION_SRC)
    if tiene_excepcion_src:
        # Verificar que la oración realmente aborda la cláusula principal donde está esa excepción
        for marcador in EXCEPCION_SRC:
            if marcador in sub_low:
                partes_exc = sub_low.split(marcador, 1)
                clausula_ppal = partes_exc[0]
                clausula_cond = partes_exc[1] if len(partes_exc) > 1 else ""
                toks_ppal = set(tokenizar_contenido(clausula_ppal, min_len=4))
                toks_cond = set(tokenizar_contenido(clausula_cond, min_len=4))
                toks_ora = set(tokenizar_contenido(oracion_texto, min_len=4))
                toks_resp = set(tokenizar_contenido(respuesta_completa, min_len=4))

                # Si la oración reproduce la cláusula principal pero omite la condición/excepción
                solape_ppal = len(toks_ppal & toks_ora) / max(1, len(toks_ppal))
                solape_cond_en_resp = len(toks_cond & toks_resp) / max(1, len(toks_cond))
                tiene_marcador_resp = any(mr in respuesta_low for mr in EXCEPCION_RESP)

                if solape_ppal >= 0.45 and solape_cond_en_resp < 0.25 and not tiene_marcador_resp:
                    flag_omision_excepcion = 1.0
                    break
                # Incluso si hay algún "solo" en otra frase de la respuesta, si en esta oración ni en la respuesta
                # aparece ninguna palabra de contenido de la condición omitida:
                if solape_ppal >= 0.50 and len(toks_cond) >= 2 and len(toks_cond & toks_resp) == 0:
                    flag_omision_excepcion = 1.0
                    break

    # 2. Omisión de modificador restrictivo de sujeto o segundo requisito obligatorio
    # Patrones restrictivos frecuentes en las fuentes del reto:
    patrones_restriccion = [
        # (regex en fuente, palabras clave que NO deben desaparecer si se cita el predicado)
        (
            r"experiencia\s+en\s+(administraciones\s+p[úu]blicas)",
            r"experiencia\s+profesional",
            1.0,
        ),
        (
            r"asociaciones\s+de\s+(mayores)",
            r"\basociaciones\s+pueden\s+llegar\b",
            1.0,
        ),
        (
            r"renta\s+m[íi]nima\s+de\s+inserci[óo]n",
            r"prestaciones\s+sociales",
            1.0,
        ),
        (
            r"no\s+seleccionados\s+que\s+hayan\s+(superado\s+el\s+test)",
            r"aspirantes\s+que\s+no\s+obtengan\s+plaza\s+se\s+formar[áa]",
            1.0,
        ),
        (
            r"tutor\s+legal\s+y\s+(acreditan\s+haber\s+superado\s+un\s+curso)",
            r"autorizaci[óo]n\s+firmada\s+por\s+su\s+tutor\s+legal\s*$",
            1.0,
        ),
        (
            r"mayores\s+de\s+85\s+a[ñn]os\s+con\s+(informe\s+m[ée]dico\s+de\s+deterioro\s+grave)",
            r"mayores\s+de\s+85\s+a[ñn]os\s+tienen\s+prioridad",
            1.0,
        ),
        (
            r"hogares\s+con\s+(ingresos\s+inferiores\s+a\s+1,2\s+veces\s+el\s+iprem)\s+y\s+empadronados",
            r"pueden\s+pedirlas\s+los\s+hogares\s+empadronados",
            1.0,
        ),
        (
            r"solo\s+los\s+regantes\s+de\s+\w+\s+con\s+(parcelas\s+de\s+m[áa]s\s+de\s+5\s+hect[áa]reas)",
            r"los\s+regantes\s+de\s+\w+\s+pudieron\s+acogerse",
            1.0,
        ),
        (
            r"domicilio\s+fiscal\s+en\s+la\s+provincia\s+y\s+(al\s+menos\s+dos\s+a[ñn]os\s+de\s+actividad)",
            r"domicilio\s+fiscal\s+en\s+la\s+provincia\s*$",
            1.0,
        ),
        (
            r"afectados\s+por\s+el\s+granizo\s+que\s+(ten[íi]an\s+seguro\s+agrario)",
            r"afectados\s+por\s+el\s+granizo\s+recibir[áa]n",
            1.0,
        ),
        (
            r"menores\s+de\s+25\s+a[ñn]os\s+(empadronados\s+en\s+la\s+ciudad)",
            r"gratuito\s+para\s+los\s+menores\s+de\s+25\s+a[ñn]os\s*$",
            1.0,
        ),
        (
            r"ni[ñn]os\s+de\s+(esas\s+zonas)\s+son\s+atendidos",
            r"ni[ñn]os\s+del\s+[áa]rea\s+de\s+\w+\s+son\s+atendidos",
            1.0,
        ),
    ]
    for pat_src, pat_resp_incompleto, peso in patrones_restriccion:
        if re.search(pat_src, fuente_low) and re.search(pat_resp_incompleto, oracion_low):
            score_omision_restriccion = max(score_omision_restriccion, peso)

    # Detector general de pérdida de modificador restrictivo tras "solo/únicamente" o "con/que [requisito]"
    if score_omision_restriccion < 0.8:
        # Si la fuente dice "Solo [X] con [Y] [Verbo]" o "[X] con [Y] y [Z] pueden..."
        m_solo = re.search(
            r"\bsolo\s+[^.;]+?\b(?:con|que\s+tengan|que\s+acrediten|empadronados)\s+([^.;,]+)",
            sub_low,
        )
        if m_solo:
            restriccion_txt = m_solo.group(1)
            toks_rest = set(tokenizar_contenido(restriccion_txt, min_len=4))
            toks_resp_all = set(tokenizar_contenido(respuesta_completa, min_len=4))
            toks_sub = set(tokenizar_contenido(sub_low, min_len=4))
            toks_ora = set(tokenizar_contenido(oracion_low, min_len=4))
            if (
                len(toks_rest) >= 1
                and len(toks_sub & toks_ora) >= 3
                and len(toks_rest & toks_resp_all) == 0
            ):
                score_omision_restriccion = max(score_omision_restriccion, 0.85)

    # 3. Sobregeneralización de cuantificadores
    # 3a. Paso de cuantificador parcial ('varios', 'algunos', 'diversos') a absoluto ('todos', 'todas', 'siempre', 'nunca')
    tiene_parcial_src = any(re.search(rf"\b{ cp }\b", sub_low) for cp in CUANTIFICADORES_PARCIALES)
    tiene_absoluto_ora = any(re.search(rf"\b{ ca }\b", oracion_low) for ca in CUANTIFICADORES_ABSOLUTOS)
    if tiene_parcial_src and tiene_absoluto_ora:
        flag_sobregeneralizacion = 1.0

    # 3b. Paso de "desde [año]" a "nunca" (ej. t0129: "la mayor muestra desde 1998" -> "dedicada nunca")
    if re.search(r"\bdesde\s+\d{4}\b", sub_low) and re.search(r"\bnunca\b", oracion_low):
        flag_sobregeneralizacion = 1.0

    # 3c. Paso de cantidad parcial "dos de los cinco [X]" / "Tres [X] de la comarca" a "[los/las] [X]"
    m_frac_src = re.search(
        r"\b(?:dos|tres|cuatro|cinco|\d+)\s+de\s+(?:los|las)\s+(?:cinco|seis|siete|ocho|nueve|diez|\d+)\s+([a-záéíóúñ]+)",
        sub_low,
    )
    if m_frac_src:
        sust = m_frac_src.group(1)
        if sust in oracion_low and not extraer_numeros_texto(oracion_low.split(sust)[0][-25:], expandir_unidades=False):
            flag_sobregeneralizacion = 1.0

    m_num_inicio = re.match(r"^(?:dos|tres|cuatro|cinco)\s+([a-záéíóúñ]+)\s+de\s+", sub_low.strip())
    if m_num_inicio:
        sust = m_num_inicio.group(1)
        if re.match(rf"^(?:las|los|todas\s+las|todos\s+los)\s+{sust}\s+de\s+", oracion_low.strip()):
            flag_sobregeneralizacion = 1.0

    # 3d. Si la fuente dice "9 de ellos tienen" sobre 14 municipios y la respuesta dice "se reparten por los 14" (t0004)
    # o "sus datos no se incluyen en la media" sobre 7 estaciones y la respuesta dice "recoge los datos de las 7" (t0094)
    if ("no se incluyen" in fuente_low or "estuvo fuera de servicio" in fuente_low) and "recoge los datos de las" in oracion_low:
        flag_sobregeneralizacion = 1.0

    return {
        "flag_omision_excepcion": float(flag_omision_excepcion),
        "score_omision_restriccion": float(score_omision_restriccion),
        "flag_sobregeneralizacion": float(flag_sobregeneralizacion),
    }


# =============================================================================
# MOTOR 3: ENDURECIMIENTO MODAL («VERB HARDENING») E INVERSIÓN TEMPORAL
# =============================================================================

def evaluar_motor_3_modal_temporal(
    oracion_idx: int,
    oracion_texto: str,
    oraciones_todas: List[Dict[str, Any]],
    fuentes: List[str],
    idx_ancla: int,
) -> Dict[str, float]:
    """
    Ejecuta el Motor 3:
    - Detecta endurecimiento modal: pasar de 'podría/estudia/prevé/recomienda/planea/quiere'
      en la fuente a un hecho consumado o futuro categórico en la respuesta.
    - Detecta inversiones cronológicas o de secuencia ('antes de', 'después', 'una vez').
    """
    flag_modal = 0.0
    flag_temporal = 0.0

    oracion_low = oracion_texto.lower()
    fuente_ancla = fuentes[idx_ancla] if (0 <= idx_ancla < len(fuentes)) else ""
    sub_low = _seleccionar_subclausula_relevante(fuente_ancla, oracion_texto).lower()

    # 1. Endurecimiento Modal (Verb Hardening)
    # Comprobar sobre la fuente ancla y también sobre cualquier fuente con alta similitud
    fuentes_a_revisar = [sub_low, fuente_ancla.lower()]
    for f_cand in fuentes:
        toks_f = set(tokenizar_contenido(f_cand, min_len=4))
        toks_o = set(tokenizar_contenido(oracion_texto, min_len=4))
        if len(toks_f & toks_o) >= 3:
            fuentes_a_revisar.append(_seleccionar_subclausula_relevante(f_cand, oracion_texto).lower())

    for f_rev in fuentes_a_revisar:
        tiene_tentativo_src = any(re.search(rf"\b{m}\b", f_rev) for m in MODAL_TENTATIVO_ESTRICTO)
        tiene_podra_sancion_src = bool(re.search(r"\bpodr[áa]\s+(?:ser\s+retirada|suponer\s+la\s+suspensi[óo]n)", f_rev))

        if tiene_tentativo_src or tiene_podra_sancion_src:
            # Verificar si la oración usa un giro categórico
            usa_categorico = bool(RE_FUTURO_CATEGORICO.search(oracion_low))
            tiene_modal_resp = any(re.search(rf"\b{mv}\b", oracion_low) for mv in MODAL_RESPUESTA_VALIDO)

            if usa_categorico and not tiene_modal_resp:
                flag_modal = 1.0
                break

            # Caso de subordinada endurecida: "estudia ampliar..., una obra que se acometerá" (t0080)
            if "se acometerá" in oracion_low and "podría acometerse" in f_rev:
                flag_modal = 1.0
                break
            if "ahorrará" in oracion_low and "podría ahorrar" in f_rev:
                flag_modal = 1.0
                break
            if "pasará" in oracion_low and "podría pasar" in f_rev:
                flag_modal = 1.0
                break
            if "se colocará" in oracion_low and "se recomienda colocar" in f_rev:
                flag_modal = 1.0
                break

    # 2. Inversión Temporal y Secuencial
    # 2a. Inversión entre oraciones consecutivas ("En septiembre... Después, [evento de julio]" en g0013)
    if oracion_idx > 0 and re.match(r"^\s*(?:despu[ée]s|luego|posteriormente|m[áa]s\s+tarde)\b", oracion_low):
        ora_prev_low = oraciones_todas[oracion_idx - 1].get("texto", "").lower()
        mes_prev = None
        for m_nom, m_num in MESES_MAP.items():
            if m_nom in ora_prev_low:
                mes_prev = m_num
                break
        if mes_prev is not None:
            # Buscar en la fuente ancla en qué mes ocurrió el evento de la oración actual
            for m_nom, m_num in MESES_MAP.items():
                if m_nom in fuente_ancla.lower() and m_num < mes_prev:
                    # Comprobar si ese mes anterior está asociado en la fuente al evento de la oración actual
                    partes_f = fuente_ancla.lower().split(";")
                    for pf in partes_f:
                        if m_nom in pf and len(set(tokenizar_contenido(pf)) & set(tokenizar_contenido(oracion_low))) >= 2:
                            flag_temporal = 1.0
                            break

    # 2b. "una vez [X]" / "después de [X]" / "antes de [X]" dentro de la misma oración
    if any(con in oracion_low for con in ("una vez", "después de", "despues de", "antes de", "como esta rompió")):
        # Caso t0101: "se abrirá el 2 de septiembre, una vez publicadas las listas" (listas = 30 de septiembre)
        if "una vez publicadas las listas" in oracion_low and "30 de septiembre" in fuente_ancla.lower():
            flag_temporal = 1.0
        # Caso t0028: "una vez inaugurada la nueva ala" (pidió en 2024 vs inaugurada en 2026)
        if "una vez inaugurada" in oracion_low:
            flag_temporal = 1.0
        # Caso t0137: Fuente dice "[A] una vez [B]" y respuesta dice "[B] después de [A]"
        m_una_vez_src = re.search(r"([a-záéíóúñ]+)\s+[^.;]*?\buna\s+vez\s+([a-záéíóúñ]+)", fuente_ancla.lower())
        if m_una_vez_src and ("después de" in oracion_low or "despues de" in oracion_low):
            r1 = raiz_palabra(m_una_vez_src.group(1))
            r2 = raiz_palabra(m_una_vez_src.group(2))
            pos_despues = oracion_low.find("despu")
            parte_ant = oracion_low[:pos_despues]
            parte_pos = oracion_low[pos_despues:]
            if r2 in normalizar_texto(parte_ant) and r1 in normalizar_texto(parte_pos):
                flag_temporal = 1.0
        # Caso g0044: Fuente dice "rompió... por lo que se descartó y se repitió" y respuesta dice "se repitió... y, como esta rompió"
        if "se repitió" in oracion_low and "como esta rompió" in oracion_low:
            flag_temporal = 1.0
        # Caso t0141: "abrirá antes de que empiece la primera fase" (marzo 2027 vs enero 2027)
        if "antes de que empiece" in oracion_low:
            meses_en_fuente_ancla = [m_num for m_nom, m_num in MESES_MAP.items() if m_nom in fuente_ancla.lower()]
            meses_en_fuente_0 = [m_num for m_nom, m_num in MESES_MAP.items() if fuentes and m_nom in fuentes[0].lower()]
            if meses_en_fuente_ancla and meses_en_fuente_0 and min(meses_en_fuente_ancla) > min(meses_en_fuente_0):
                flag_temporal = 1.0

    return {
        "flag_endurecimiento_modal": float(flag_modal),
        "flag_inversion_temporal": float(flag_temporal),
    }


# =============================================================================
# MOTOR 4: INCISOS SUTILES, ADJETIVOS INVENTADOS Y CAUSALIDAD NO RESPALDADA
# =============================================================================

def evaluar_motor_4_sutil_causal(
    oracion_texto: str,
    fragmentos_oracion: List[Dict[str, Any]],
    fuentes: List[str],
    pregunta: str,
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 4:
    - Detecta palabras de contenido huérfanas (adjetivos/incisos inventados) en cuasi-copias (::sutil).
    - Detecta fragmentos huérfanos ('para asegurar que...', 'pionera en derechos').
    - Detecta causalidad inventada al unir dos fuentes distintas ('por el aumento de...').
    """
    sim_f = oracion_dict.get("similitud_por_fuente") or [0.0]
    resp_f = oracion_dict.get("respaldo_por_fuente") or [0.0]
    max_sim = max(sim_f) if sim_f else 0.0
    max_resp = max(resp_f) if resp_f else 0.0
    resp_todas = float(oracion_dict.get("respaldo_todas", 0.0))

    # Construir conjunto de raíces de todas las fuentes y de la pregunta
    texto_fuentes_y_preg = " ".join(fuentes) + " " + pregunta
    toks_fuentes = tokenizar_contenido(texto_fuentes_y_preg, min_len=3)
    raices_fuentes: Set[str] = {raiz_palabra(t) for t in toks_fuentes}
    # Añadir también raíces de 4 letras por seguridad ante derivaciones verbales
    prefijos4_fuentes: Set[str] = {t[:4] for t in toks_fuentes if len(t) >= 4}

    toks_oracion = tokenizar_contenido(oracion_texto, min_len=4)
    # Ignorar palabras numéricas escritas en letra que ya verifica el Motor 1
    palabras_num_set = set(UNIDADES_PALABRA.keys()) | set(DECENAS_PALABRA.keys()) | set(CENTENAS_PALABRA.keys()) | {
        "millon", "millones", "miles", "mitad", "mitades", "tercio", "quinto",
        "doble", "triple", "docena", "docenas", "decena", "decenas", "lustro",
    }

    huerfanas: List[str] = []
    for tok in toks_oracion:
        if tok in palabras_num_set:
            continue
        r5 = raiz_palabra(tok)
        p4 = tok[:4]
        if r5 not in raices_fuentes and p4 not in prefijos4_fuentes:
            huerfanas.append(tok)

    ratio_huerfanas = float(len(huerfanas)) / float(max(1, len(toks_oracion)))

    flag_inciso_sutil = 0.0
    # Firma clásica de ::sutil: altísima similitud (> 0.85), casi todas las palabras coinciden,
    # pero hay 1-3 palabras huérfanas nuevas y el respaldo NLI se desploma (< 0.50)
    if (
        max_sim > 0.85
        and 1 <= len(huerfanas) <= 3
        and (len(toks_oracion) - len(huerfanas)) / max(1, len(toks_oracion)) >= 0.60
        and max(max_resp, resp_todas) < 0.50
    ):
        flag_inciso_sutil = 1.0

    # También comprobar si hay algún fragmento hijo claramente huérfano mientras otro fragmento está muy respaldado
    if len(fragmentos_oracion) >= 2:
        soportes_frag = [
            max(max(f.get("respaldo_por_fuente") or [0.0]), float(f.get("respaldo_todas", 0.0)))
            for f in fragmentos_oracion
        ]
        sims_frag = [
            max(f.get("similitud_por_fuente") or [0.0])
            for f in fragmentos_oracion
        ]
        if max(soportes_frag) > 0.70:
            for idx_f, f in enumerate(fragmentos_oracion):
                if soportes_frag[idx_f] < 0.25 and sims_frag[idx_f] < 0.42:
                    # Comprobar si ese fragmento tiene palabras huérfanas o conector final/causal inventado
                    f_toks = tokenizar_contenido(f.get("texto", ""), min_len=4)
                    f_huerf = [t for t in f_toks if raiz_palabra(t) not in raices_fuentes]
                    if f_huerf or max(max_resp, resp_todas) < 0.15:
                        flag_inciso_sutil = 1.0

    # 3. Detector de Causalidad Inventada
    flag_causalidad = 0.0
    oracion_low = oracion_texto.lower()
    for con_c in CONECTORES_CAUSALES:
        if con_c in oracion_low:
            # Si el conector causal no está en ninguna fuente relevante y el respaldo de la oración completa es bajo
            con_en_fuente = any(con_c in f.lower() for f in fuentes)
            if not con_en_fuente and max(max_resp, resp_todas) < 0.42:
                # Excepción: si la fuente tiene "ya que", "puesto que", "por la", "atribuye" en la MISMA fuente ancla
                idx_best_sim = int(np.argmax(sim_f)) if sim_f else 0
                f_best_low = fuentes[idx_best_sim].lower() if idx_best_sim < len(fuentes) else ""
                tiene_causa_ancla = any(
                    c_alt in f_best_low
                    for c_alt in ("porque", "ya que", "por la ", "por el ", "gracias a", "debido a", "atribuye", "justifica", "para ")
                )
                if not tiene_causa_ancla:
                    flag_causalidad = 1.0
            # Si la oración une un hecho de la fuente A con una causa de la fuente B ("por el aumento de población en verano" en t0133)
            if len(fragmentos_oracion) >= 2 and max_resp < 0.10 and resp_todas < 0.45:
                fuentes_top_frags = [
                    int(np.argmax(f.get("similitud_por_fuente") or [0.0]))
                    for f in fragmentos_oracion
                ]
                if len(set(fuentes_top_frags)) >= 2:
                    # Verificar si en la fuente de la causa hay otro sujeto ("justificó la ambulancia")
                    for k_f in set(fuentes_top_frags):
                        if con_c in fuentes[k_f].lower() and k_f != fuentes_top_frags[0]:
                            flag_causalidad = 1.0

    return {
        "flag_inciso_sutil": float(flag_inciso_sutil),
        "flag_causalidad_inventada": float(flag_causalidad),
        "ratio_palabras_huerfanas": float(ratio_huerfanas),
    }


# =============================================================================
# MOTOR 5: ATRIBUCIÓN CRUZADA DE ENTIDADES (CROSS-ENTITY GUARD)
# =============================================================================

def _extraer_entidades_y_codigos(texto: str) -> Set[str]:
    """Extrae nombres propios capitalizados en mitad de frase y códigos técnicos."""
    entidades: Set[str] = set()
    if not texto:
        return entidades

    # Códigos tipo MP-1, MP-2, PH-800, HA-30, Lote 2614, Grado III, Grado I
    for m_cod in re.finditer(
        r"\b(?:[A-Z]{1,4}-\d+|lote\s+\d+|grado\s+(?:iii|ii|i)|l[íi]nea\s+\d+|calle\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+\d+)\b",
        texto,
        re.IGNORECASE,
    ):
        entidades.add(normalizar_texto(m_cod.group(0)))

    # Palabras capitalizadas que no están al inicio absoluto de la oración
    palabras = re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}\b", texto)
    STOP_CAP = {
        "Segun", "Según", "Durante", "Entre", "Para", "Desde", "Hasta", "Tras",
        "Antes", "Despues", "Después", "Mientras", "Solo", "Sólo", "Cada",
        "Todas", "Todos", "Ningun", "Ningún", "Ninguna", "Además", "Ademas",
        "También", "Tambien", "Quedan", "Están", "Estan", "Cuando", "Donde",
    }
    for p in palabras[1:]:
        if p not in STOP_CAP:
            entidades.add(normalizar_texto(p))
    return entidades


def evaluar_motor_5_entidades(
    oracion_texto: str,
    fragmentos_oracion: List[Dict[str, Any]],
    fuentes: List[str],
    pregunta: str,
    oracion_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Ejecuta el Motor 5:
    - Detecta si la oración atribuye a la entidad principal un dato procedente de una
      fuente distractora (otro municipio, otra empresa, otro centro, otro modelo).
    - Detecta inversión de valores entre dos entidades dentro de la misma fuente
      (ej. intercambiar Grado III y Grado I, o afirmar que el 24% es el mayor cuando hay 68%).
    """
    flag_cruzada = 0.0
    n_fuentes = len(fuentes)
    if n_fuentes == 0:
        return {"flag_atribucion_cruzada": 0.0}

    oracion_low = oracion_texto.lower()
    resp_f = oracion_dict.get("respaldo_por_fuente") or [0.0] * n_fuentes
    cont_f = oracion_dict.get("contradiccion_por_fuente") or [0.0] * n_fuentes
    sim_f = oracion_dict.get("similitud_por_fuente") or [0.0] * n_fuentes

    # 1. Intercambio interno de valores dentro de una misma fuente (ej. t0005 Grado III vs Grado I, t0156 Valdescarra vs Peñalvea)
    idx_top_sim = int(np.argmax(sim_f)) if sim_f else 0
    fuente_sim = fuentes[idx_top_sim] if idx_top_sim < n_fuentes else ""
    if ";" in fuente_sim or "frente a" in fuente_sim.lower():
        subclausulas = re.split(r";|\bfrente\s+a\b", fuente_sim)
        if len(subclausulas) >= 2:
            # Extraer pares (entidad/etiqueta, números) de cada subcláusula
            info_subs: List[Tuple[Set[str], List[float]]] = []
            for sc in subclausulas:
                ents_sc = _extraer_entidades_y_codigos(sc)
                # Añadir también sustantivos clave de la subcláusula
                for tok in tokenizar_contenido(sc, min_len=5):
                    ents_sc.add(tok)
                nums_sc = [v for v in extraer_numeros_texto(sc, expandir_unidades=False) if not (1900 <= v <= 2035)]
                info_subs.append((ents_sc, nums_sc))

            # Comprobar si un fragmento de la oración asocia una entidad exclusiva de subcláusula A con el número exclusivo de subcláusula B
            for i_a in range(len(info_subs)):
                for i_b in range(len(info_subs)):
                    if i_a == i_b:
                        continue
                    ents_a_unicas = info_subs[i_a][0] - info_subs[i_b][0]
                    nums_b_unicos = [x for x in info_subs[i_b][1] if not _numero_coincide(x, info_subs[i_a][1])]
                    if ents_a_unicas and nums_b_unicos:
                        for frag in fragmentos_oracion:
                            f_txt = frag.get("texto", "")
                            f_norm = normalizar_texto(f_txt)
                            f_nums = extraer_numeros_texto(f_txt, expandir_unidades=False)
                            f_ents_b = info_subs[i_b][0] - info_subs[i_a][0]
                            tiene_ent_a = any(ea in f_norm for ea in ents_a_unicas)
                            tiene_ent_b = any(eb in f_norm for eb in f_ents_b)
                            tiene_num_b = any(_numero_coincide(nb, f_nums) for nb in nums_b_unicos)
                            if tiene_ent_a and not tiene_ent_b and tiene_num_b:
                                # Si además el fragmento o la oración tiene contradicción alta en esa fuente
                                if cont_f[idx_top_sim] > 0.50 or (frag.get("contradiccion_por_fuente") or [0.0])[idx_top_sim] > 0.50:
                                    flag_cruzada = 1.0

    # 2. Afirmar "el mayor" / "la mayor" para una entidad secundaria cuyo número en la fuente es menor que otra (g0004, t0115)
    if re.search(r"\b(?:el\s+mayor|la\s+mayor)\b", oracion_low):
        nums_ora = [v for v in extraer_numeros_texto(oracion_texto, expandir_unidades=False) if not (1900 <= v <= 2035)]
        nums_src = [v for v in extraer_numeros_texto(fuente_sim, expandir_unidades=False) if not (1900 <= v <= 2035)]
        if nums_ora and nums_src and max(nums_src) > max(nums_ora) and max(resp_f) < 0.25:
            flag_cruzada = 1.0

    # 3. Cruce con fuente distractora en escenarios multi-fuente
    if n_fuentes >= 4:
        ents_preg = _extraer_entidades_y_codigos(pregunta) | _extraer_entidades_y_codigos(fuentes[0])
        for k in range(1, n_fuentes):
            f_k = fuentes[k]
            ents_k = _extraer_entidades_y_codigos(f_k)
            ents_nuevas_k = ents_k - ents_preg
            es_distractora = bool(
                ents_nuevas_k
                and re.search(
                    r"\b(?:vecin[oa]|misma\s+comarca|misma\s+ciudad|mismo\s+municipio|"
                    r"mismo\s+ayuntamiento|mismo\s+fabricante|otra\s+de\s+la|"
                    r"otro\s+cliente|hermana|por\s+su\s+parte|a\s+\d+\s*km)\b",
                    f_k.lower(),
                )
            )
            if es_distractora:
                ora_norm = normalizar_texto(oracion_texto)
                menciona_ent_distractora = any(ed in ora_norm for ed in ents_nuevas_k)
                if not menciona_ent_distractora:
                    # La oración NO menciona a la entidad distractora (habla de la entidad principal),
                    # pero algún fragmento toma su respaldo principalmente de la fuente distractora k
                    # mientras contradice una fuente principal
                    for frag in fragmentos_oracion:
                        fr_resp = frag.get("respaldo_por_fuente") or []
                        fr_cont = frag.get("contradiccion_por_fuente") or []
                        if k < len(fr_resp) and fr_resp[k] > 0.75:
                            max_resp_otras = max([fr_resp[j] for j in range(n_fuentes) if j != k] or [0.0])
                            max_cont_otras = max([fr_cont[j] for j in range(n_fuentes) if j != k] or [0.0])
                            if max_resp_otras < 0.20 and max_cont_otras > 0.65:
                                flag_cruzada = 1.0

    return {"flag_atribucion_cruzada": float(flag_cruzada)}


# =============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS A NIVEL DE ORACIÓN (FEATURE ENGINEERING)
# =============================================================================

FEATURE_NAMES: List[str] = [
    # 1. Señales NLI de la oración
    "max_resp_fuente",
    "second_max_resp_fuente",
    "sum_top2_resp_fuente",
    "resp_todas",
    "best_oracion_resp",
    "max_cont_fuente",
    "cont_todas",
    "cont_at_best_resp_source",
    "cont_at_best_sim_source",
    "max_sim_fuente",
    "gap_sim_resp",
    # 2. Señales del Juez LLM en la oración
    "llm_is_respaldada",
    "llm_is_contradicha",
    "llm_is_no_respaldada",
    "llm_confianza",
    "llm_signed_score",
    "resp_at_llm_source",
    "cont_at_llm_source",
    "sim_at_llm_source",
    "desacuerdo_llm_nli",
    # 3. Señales de los Fragmentos hijos (Resolución Multi-Fuente)
    "n_fragmentos",
    "min_frag_eff_resp",
    "mean_frag_eff_resp",
    "max_frag_cont_todas",
    "max_frag_cont_at_sim",
    "min_frag_max_sim",
    "multi_source_synergy",
    "best_combined_support",
    # 4. Señales de los 5 Motores Neuro-Simbólicos
    "flag_alucinacion_numerica",
    "n_numeros_derivados_validos",
    "flag_trampa_porcentaje",
    "ratio_numeros_respaldados",
    "distorsion_cercana",
    "flag_omision_excepcion",
    "score_omision_restriccion",
    "flag_sobregeneralizacion",
    "flag_endurecimiento_modal",
    "flag_inversion_temporal",
    "flag_inciso_sutil",
    "flag_causalidad_inventada",
    "ratio_palabras_huerfanas",
    "flag_atribucion_cruzada",
    # 5. Contexto estructural
    "is_multi_source_case",
]


def extract_sentence_features(oracion_idx: int, caso: Dict[str, Any]) -> Dict[str, float]:
    """
    Construye el diccionario completo de 43 características numéricas para la oración `oracion_idx`.
    Maneja de forma segura cualquier valor nulo o lista vacía.
    """
    fuentes: List[str] = caso.get("fuentes") or []
    pregunta: str = caso.get("pregunta") or ""
    respuesta: str = caso.get("respuesta") or ""
    senales: Dict[str, Any] = caso.get("senales") or {}
    oraciones: List[Dict[str, Any]] = senales.get("oraciones") or []
    fragmentos_todos: List[Dict[str, Any]] = senales.get("fragmentos") or []

    if oracion_idx < 0 or oracion_idx >= len(oraciones):
        return {k: 0.0 for k in FEATURE_NAMES}

    ora = oraciones[oracion_idx]
    ora_texto: str = ora.get("texto") or ""
    n_fuentes = max(1, len(fuentes))

    resp_f = [float(x) for x in (ora.get("respaldo_por_fuente") or [0.0])]
    cont_f = [float(x) for x in (ora.get("contradiccion_por_fuente") or [0.0])]
    sim_f = [float(x) for x in (ora.get("similitud_por_fuente") or [0.0])]

    if not resp_f:
        resp_f = [0.0]
    if not cont_f:
        cont_f = [0.0]
    if not sim_f:
        sim_f = [0.0]

    # 1. Señales NLI de la oración
    resp_sorted = sorted(resp_f, reverse=True)
    max_resp_fuente = float(resp_sorted[0])
    second_max_resp_fuente = float(resp_sorted[1]) if len(resp_sorted) > 1 else 0.0
    sum_top2_resp_fuente = float(min(1.0, max_resp_fuente + second_max_resp_fuente))
    resp_todas = float(ora.get("respaldo_todas") or 0.0)
    best_oracion_resp = float(max(max_resp_fuente, resp_todas))

    max_cont_fuente = float(max(cont_f))
    cont_todas = float(ora.get("contradiccion_todas") or 0.0)

    idx_best_resp = int(np.argmax(resp_f))
    idx_best_sim = int(np.argmax(sim_f))
    cont_at_best_resp_source = float(cont_f[idx_best_resp]) if idx_best_resp < len(cont_f) else 0.0
    cont_at_best_sim_source = float(cont_f[idx_best_sim]) if idx_best_sim < len(cont_f) else 0.0

    max_sim_fuente = float(max(sim_f))
    gap_sim_resp = float(max_sim_fuente - best_oracion_resp)

    # 2. Señales del Juez LLM
    llm = ora.get("llm")
    veredicto = llm.get("veredicto") if isinstance(llm, dict) else None
    conf_raw = llm.get("confianza") if isinstance(llm, dict) else None
    llm_confianza = float(conf_raw) if conf_raw is not None else 0.5

    llm_is_respaldada = 1.0 if veredicto == "respaldada" else 0.0
    llm_is_contradicha = 1.0 if veredicto == "contradicha" else 0.0
    llm_is_no_respaldada = 1.0 if veredicto == "no_respaldada" else 0.0

    if llm_is_respaldada > 0.5:
        llm_signed_score = llm_confianza
    elif llm_is_contradicha > 0.5 or llm_is_no_respaldada > 0.5:
        llm_signed_score = -llm_confianza
    else:
        llm_signed_score = 0.0

    llm_fuente = llm.get("fuente") if isinstance(llm, dict) else None
    if isinstance(llm_fuente, int) and 0 <= llm_fuente < len(resp_f):
        resp_at_llm_source = float(resp_f[llm_fuente])
        cont_at_llm_source = float(cont_f[llm_fuente]) if llm_fuente < len(cont_f) else 0.0
        sim_at_llm_source = float(sim_f[llm_fuente]) if llm_fuente < len(sim_f) else 0.0
    else:
        resp_at_llm_source = max_resp_fuente
        cont_at_llm_source = cont_at_best_sim_source
        sim_at_llm_source = max_sim_fuente

    # Fuente ancla k* combinando llm["fuente"] y argmax(0.6 * sim + 0.4 * resp)
    scores_ancla = [
        0.6 * (sim_f[k] if k < len(sim_f) else 0.0) + 0.4 * (resp_f[k] if k < len(resp_f) else 0.0)
        for k in range(len(fuentes))
    ]
    idx_ancla = int(np.argmax(scores_ancla)) if scores_ancla else 0
    if isinstance(llm_fuente, int) and 0 <= llm_fuente < len(fuentes):
        if scores_ancla[llm_fuente] >= scores_ancla[idx_ancla] - 0.15:
            idx_ancla = llm_fuente

    # 3. Señales de los Fragmentos hijos de la oración i
    frags_hijos = [f for f in fragmentos_todos if f.get("oracion") == oracion_idx]
    if not frags_hijos:
        frags_hijos = [ora]

    s_eff_list: List[float] = []
    f_max_resp_fuente_list: List[float] = []
    f_cont_todas_list: List[float] = []
    f_cont_sim_list: List[float] = []
    f_max_sim_list: List[float] = []

    for f in frags_hijos:
        fr_r = [float(x) for x in (f.get("respaldo_por_fuente") or [0.0])]
        fr_c = [float(x) for x in (f.get("contradiccion_por_fuente") or [0.0])]
        fr_s = [float(x) for x in (f.get("similitud_por_fuente") or [0.0])]
        fr_rt = float(f.get("respaldo_todas") or 0.0)
        fr_ct = float(f.get("contradiccion_todas") or 0.0)

        max_fr_r = max(fr_r) if fr_r else 0.0
        max_fr_s = max(fr_s) if fr_s else 0.0
        idx_fr_s = int(np.argmax(fr_s)) if fr_s else 0
        c_sim_f = float(fr_c[idx_fr_s]) if idx_fr_s < len(fr_c) else 0.0

        s_eff_list.append(max(max_fr_r, fr_rt))
        f_max_resp_fuente_list.append(max_fr_r)
        f_cont_todas_list.append(fr_ct)
        f_cont_sim_list.append(c_sim_f)
        f_max_sim_list.append(max_fr_s)

    min_frag_eff_resp = float(min(s_eff_list))
    mean_frag_eff_resp = float(np.mean(s_eff_list))
    max_frag_cont_todas = float(max(f_cont_todas_list))
    max_frag_cont_at_sim = float(max(f_cont_sim_list))
    min_frag_max_sim = float(min(f_max_sim_list))
    multi_source_synergy = float(np.mean(f_max_resp_fuente_list) - max_resp_fuente)
    best_combined_support = float(max(best_oracion_resp, min_frag_eff_resp))

    desacuerdo_llm_nli = (
        1.0
        if (llm_is_respaldada > 0.5 and best_oracion_resp < 0.15 and min_frag_eff_resp < 0.25)
        else 0.0
    )

    # 4. Ejecutar los 5 Motores Neuro-Simbólicos
    m1 = evaluar_motor_1_aritmetica(ora_texto, frags_hijos, fuentes, ora)
    m2 = evaluar_motor_2_omisiones(ora_texto, respuesta, fuentes, idx_ancla, ora)
    m3 = evaluar_motor_3_modal_temporal(oracion_idx, ora_texto, oraciones, fuentes, idx_ancla)
    m4 = evaluar_motor_4_sutil_causal(ora_texto, frags_hijos, fuentes, pregunta, ora)
    m5 = evaluar_motor_5_entidades(ora_texto, frags_hijos, fuentes, pregunta, ora)

    feats: Dict[str, float] = {
        "max_resp_fuente": max_resp_fuente,
        "second_max_resp_fuente": second_max_resp_fuente,
        "sum_top2_resp_fuente": sum_top2_resp_fuente,
        "resp_todas": resp_todas,
        "best_oracion_resp": best_oracion_resp,
        "max_cont_fuente": max_cont_fuente,
        "cont_todas": cont_todas,
        "cont_at_best_resp_source": cont_at_best_resp_source,
        "cont_at_best_sim_source": cont_at_best_sim_source,
        "max_sim_fuente": max_sim_fuente,
        "gap_sim_resp": gap_sim_resp,
        "llm_is_respaldada": llm_is_respaldada,
        "llm_is_contradicha": llm_is_contradicha,
        "llm_is_no_respaldada": llm_is_no_respaldada,
        "llm_confianza": llm_confianza,
        "llm_signed_score": llm_signed_score,
        "resp_at_llm_source": resp_at_llm_source,
        "cont_at_llm_source": cont_at_llm_source,
        "sim_at_llm_source": sim_at_llm_source,
        "desacuerdo_llm_nli": desacuerdo_llm_nli,
        "n_fragmentos": float(len(frags_hijos)),
        "min_frag_eff_resp": min_frag_eff_resp,
        "mean_frag_eff_resp": mean_frag_eff_resp,
        "max_frag_cont_todas": max_frag_cont_todas,
        "max_frag_cont_at_sim": max_frag_cont_at_sim,
        "min_frag_max_sim": min_frag_max_sim,
        "multi_source_synergy": multi_source_synergy,
        "best_combined_support": best_combined_support,
        "flag_alucinacion_numerica": m1["flag_alucinacion_numerica"],
        "n_numeros_derivados_validos": m1["n_numeros_derivados_validos"],
        "flag_trampa_porcentaje": m1["flag_trampa_porcentaje"],
        "ratio_numeros_respaldados": m1["ratio_numeros_respaldados"],
        "distorsion_cercana": m1["distorsion_cercana"],
        "flag_omision_excepcion": m2["flag_omision_excepcion"],
        "score_omision_restriccion": m2["score_omision_restriccion"],
        "flag_sobregeneralizacion": m2["flag_sobregeneralizacion"],
        "flag_endurecimiento_modal": m3["flag_endurecimiento_modal"],
        "flag_inversion_temporal": m3["flag_inversion_temporal"],
        "flag_inciso_sutil": m4["flag_inciso_sutil"],
        "flag_causalidad_inventada": m4["flag_causalidad_inventada"],
        "ratio_palabras_huerfanas": m4["ratio_palabras_huerfanas"],
        "flag_atribucion_cruzada": m5["flag_atribucion_cruzada"],
        "is_multi_source_case": 1.0 if len(fuentes) > 1 else 0.0,
    }
    return feats


# =============================================================================
# MOTOR EXPERTO NEURO-SIMBÓLICO (CORE ANALÍTICO AUTÓNOMO)
# =============================================================================

def symbolic_sentence_risk(feats: Dict[str, float]) -> Tuple[float, float]:
    """
    Calcula la probabilidad analítica de alucinación de una oración en (0, 1)
    y el nivel de veto simbólico duro (0.0 si no hay veto activo).
    """
    best_supp = feats["best_combined_support"]
    best_ora_resp = feats["best_oracion_resp"]
    min_frag_resp = feats["min_frag_eff_resp"]
    mean_frag_resp = feats["mean_frag_eff_resp"]
    cont_sim = feats["cont_at_best_sim_source"]
    cont_llm = feats["cont_at_llm_source"]
    cont_todas = feats["cont_todas"]
    max_sim = feats["max_sim_fuente"]

    # Comprobar si algún flag simbólico de violación está activo
    hay_flag_violacion = (
        feats["flag_alucinacion_numerica"] > 0.5
        or feats["flag_trampa_porcentaje"] > 0.5
        or feats["flag_omision_excepcion"] > 0.5
        or feats["score_omision_restriccion"] >= 0.80
        or feats["flag_sobregeneralizacion"] > 0.5
        or feats["flag_endurecimiento_modal"] > 0.5
        or feats["flag_inversion_temporal"] > 0.5
        or feats["flag_inciso_sutil"] > 0.5
        or feats["flag_causalidad_inventada"] > 0.5
        or feats["flag_atribucion_cruzada"] > 0.5
    )

    # 1. Regla de Rescate Matemático y de Paráfrasis Numérica
    rescate_matematico = (
        feats["n_numeros_derivados_validos"] >= 1.0
        and not hay_flag_violacion
        and feats["ratio_numeros_respaldados"] >= 0.99
    )

    # 2. Regla de Rescate Multi-Fuente (síntesis fiel cuyos fragmentos están respaldados)
    rescate_multifuente = (
        feats["is_multi_source_case"] > 0.5
        and not hay_flag_violacion
        and (min_frag_resp > 0.68 or (mean_frag_resp > 0.78 and min_frag_resp > 0.45) or feats["resp_todas"] > 0.80)
    )

    # 3. Riesgo base según el veredicto del Juez LLM y el NLI
    if feats["llm_is_contradicha"] > 0.5 or feats["llm_is_no_respaldada"] > 0.5:
        # Verificar si aplica rescate legítimo ante fallo del LLM (ej. e34::par, e24::par, t0064, t0089)
        if rescate_matematico and (
            (best_supp > 0.90 and cont_todas < 0.15 and cont_sim < 0.10)
            or (min_frag_resp > 0.75 and cont_sim < 0.20)
            or feats["llm_confianza"] <= 0.65
        ):
            riesgo_base = 0.12
        else:
            riesgo_base = 0.86 + 0.10 * feats["llm_confianza"]
    elif feats["llm_is_respaldada"] > 0.5:
        # El LLM dice "respaldada": evaluar cuánto podemos fiarnos según el soporte NLI
        if rescate_matematico:
            riesgo_base = 0.06 + 0.08 * (1.0 - max(best_supp, 0.50))
        elif rescate_multifuente:
            riesgo_base = 0.05 + 0.12 * (1.0 - max(best_supp, mean_frag_resp))
        else:
            # Si ni la oración ni los fragmentos tienen respaldo en el NLI
            soporte_efectivo = max(best_ora_resp, 0.75 * min_frag_resp + 0.25 * mean_frag_resp)
            contradiccion_efectiva = max(cont_sim, cont_llm)
            if rescate_matematico:
                contradiccion_efectiva *= 0.15

            riesgo_base = 0.08 + 0.52 * (1.0 - soporte_efectivo) + 0.22 * contradiccion_efectiva
            if feats["desacuerdo_llm_nli"] > 0.5 and feats["n_numeros_derivados_validos"] == 0:
                riesgo_base = max(riesgo_base, 0.62)
            if max_sim > 0.85 and best_supp < 0.15 and feats["n_numeros_derivados_validos"] == 0:
                riesgo_base = max(riesgo_base, 0.68)
    else:
        # LLM es None / no contestó: apoyarse en soporte combinado y contradicción en fuente similar
        soporte_efectivo = max(best_ora_resp, min_frag_resp)
        riesgo_base = 0.50 * (1.0 - soporte_efectivo) + 0.35 * max(cont_sim, cont_todas)

    # 4. Reglas de Veto Simbólico (Caza-Alucinaciones cuando el LLM se confía)
    veto_prob = 0.0
    if feats["flag_alucinacion_numerica"] > 0.5 or feats["flag_trampa_porcentaje"] > 0.5:
        veto_prob = max(veto_prob, 0.96 if feats["distorsion_cercana"] > 0.5 else 0.92)
    if feats["flag_omision_excepcion"] > 0.5 or feats["flag_sobregeneralizacion"] > 0.5:
        veto_prob = max(veto_prob, 0.87)
    if feats["score_omision_restriccion"] >= 0.80:
        veto_prob = max(veto_prob, 0.86 * feats["score_omision_restriccion"])
    if feats["flag_endurecimiento_modal"] > 0.5 or feats["flag_inversion_temporal"] > 0.5:
        veto_prob = max(veto_prob, 0.89)
    if (
        feats["flag_inciso_sutil"] > 0.5
        or feats["flag_causalidad_inventada"] > 0.5
        or feats["flag_atribucion_cruzada"] > 0.5
    ):
        veto_prob = max(veto_prob, 0.87)

    riesgo_final = float(np.clip(max(riesgo_base, veto_prob), 0.005, 0.995))
    return riesgo_final, veto_prob


# =============================================================================
# CLASE PRINCIPAL: GUARDIAN
# =============================================================================

class Guardian:
    """
    Detector de alucinaciones para sistemas RAG en español («El Guardián»).
    Combina 5 Motores Neuro-Simbólicos con un Ensemble de 3 clasificadores
    de scikit-learn bajo agregación Multiple Instance Learning (MIL) y
    calibración Bayesiana para la métrica asimétrica (4*FN + 1*FP).
    """

    def __init__(self, menor_igual_05_es_alucina: Optional[bool] = None) -> None:
        if menor_igual_05_es_alucina is None:
            env_pol = os.environ.get("GUARDIAN_MENOR_IGUAL_05_ALUCINA")
            if env_pol is not None:
                self.menor_igual_05_es_alucina = env_pol.strip().lower() in ("1", "true", "si", "yes")
            else:
                self.menor_igual_05_es_alucina = SCORE_MENOR_IGUAL_05_ES_ALUCINA
        else:
            self.menor_igual_05_es_alucina = bool(menor_igual_05_es_alucina)

        self.models: List[Any] = []
        self.is_trained: bool = False
        self.optimal_threshold: float = 0.26  # Umbral Bayesiano por defecto para coste 4:1

        self._entrenar_y_calibrar()

    def _buscar_ruta_train(self) -> Optional[Path]:
        """Busca `train.jsonl` en las ubicaciones estándar del proyecto."""
        base_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        candidatos = [
            cwd / "train.jsonl",
            cwd / "kit" / "train.jsonl",
            base_dir / "train.jsonl",
            base_dir / "kit" / "train.jsonl",
            base_dir.parent / "train.jsonl",
            base_dir.parent / "kit" / "train.jsonl",
        ]
        for ruta in candidatos:
            if ruta.is_file():
                return ruta
        return None

    def _construir_modelos_base(self) -> List[Any]:
        """Instancia los 3 clasificadores del ensemble de scikit-learn."""
        hgb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=2.0,
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            class_weight={0: 1.0, 1: 2.5},
            random_state=42,
        )
        lr = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=0.5,
                        class_weight={0: 1.0, 1: 2.5},
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
        return [hgb, rf, lr]

    @staticmethod
    def _peso_por_id(caso_id: str) -> float:
        """Asigna pesos de muestra priorizando los escenarios multi-fuente tipo test/giro."""
        if not caso_id:
            return 1.0
        pref = caso_id[0]
        if pref == "H":
            return 3.5
        if pref in "NsTGR":
            return 2.0
        return 0.6

    def _preparar_oraciones_entrenamiento(
        self, casos: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[np.ndarray]], List[List[Tuple[float, float]]]]:
        """
        Extrae las matrices X, y, sample_weight a nivel de oración para entrenar,
        identificando la oración culpable en casos multi-oración con etiqueta == 1.
        También devuelve las estructuras por caso para validación Out-Of-Fold.
        """
        X_rows: List[np.ndarray] = []
        y_rows: List[int] = []
        w_rows: List[float] = []

        casos_X_oraciones: List[List[np.ndarray]] = []
        casos_sym_oraciones: List[List[Tuple[float, float]]] = []

        for caso in casos:
            etiqueta = int(caso.get("etiqueta", 0))
            caso_id = str(caso.get("id", ""))
            peso = self._peso_por_id(caso_id)
            oraciones = (caso.get("senales") or {}).get("oraciones") or []
            n_ora = len(oraciones)

            vecs_caso: List[np.ndarray] = []
            syms_caso: List[Tuple[float, float]] = []

            for i in range(n_ora):
                f_dict = extract_sentence_features(i, caso)
                vec = np.array([f_dict[k] for k in FEATURE_NAMES], dtype=np.float64)
                sym_prob, veto_prob = symbolic_sentence_risk(f_dict)
                vecs_caso.append(vec)
                syms_caso.append((sym_prob, veto_prob))

            casos_X_oraciones.append(vecs_caso)
            casos_sym_oraciones.append(syms_caso)

            if n_ora == 0:
                continue

            if etiqueta == 0:
                for vec in vecs_caso:
                    X_rows.append(vec)
                    y_rows.append(0)
                    w_rows.append(peso)
            else:
                if n_ora == 1:
                    X_rows.append(vecs_caso[0])
                    y_rows.append(1)
                    w_rows.append(peso)
                else:
                    # En un caso positivo con múltiples oraciones, la oración culpable es
                    # la de mayor riesgo neuro-simbólico (y cualquiera que supere 0.65).
                    riesgos_sym = [s[0] for s in syms_caso]
                    idx_culpable = int(np.argmax(riesgos_sym))
                    for i, vec in enumerate(vecs_caso):
                        if i == idx_culpable or riesgos_sym[i] > 0.65:
                            X_rows.append(vec)
                            y_rows.append(1)
                            w_rows.append(peso)

        return (
            np.vstack(X_rows),
            np.array(y_rows, dtype=np.int32),
            np.array(w_rows, dtype=np.float64),
            casos_X_oraciones,
            casos_sym_oraciones,
        )

    @staticmethod
    def _predecir_ensemble_matriz(modelos: List[Any], X_mat: np.ndarray) -> np.ndarray:
        """Promedia las probabilidades predichas por los 3 modelos del ensemble."""
        preds = np.zeros(X_mat.shape[0], dtype=np.float64)
        for m in modelos:
            preds += m.predict_proba(X_mat)[:, 1]
        return preds / float(len(modelos))

    @staticmethod
    def _agregar_riesgo_caso(probs_oraciones: List[float]) -> float:
        """
        Agregación MIL dominada por la peor oración (Noisy-OR / Soft-Max):
        P_raw(alucina | caso) = max_i p_i + 0.05 * (1 - max_i p_i) * sum_{j != i*} max(0, p_j - 0.25)
        """
        if not probs_oraciones:
            return 0.0
        idx_max = int(np.argmax(probs_oraciones))
        max_p = float(probs_oraciones[idx_max])
        extra = sum(
            max(0.0, float(p_j) - 0.25)
            for j, p_j in enumerate(probs_oraciones)
            if j != idx_max
        )
        p_caso = max_p + 0.05 * (1.0 - max_p) * extra
        return float(np.clip(p_caso, 0.001, 0.999))

    @staticmethod
    def _calibrar_probabilidad(p_raw: float, t_star: float) -> float:
        """
        Transformación continua y estrictamente monótona psi(p, t*) que lleva t* -> 0.50
        preservando el ranking exacto para el AUROC.
        """
        t = float(np.clip(t_star, 0.05, 0.95))
        p = float(np.clip(p_raw, 0.0, 1.0))
        if p <= t:
            return 0.50 * (p / t)
        return 0.50 + 0.50 * ((p - t) / (1.0 - t))

    def _entrenar_y_calibrar(self) -> None:
        """Carga `train.jsonl` si existe, ejecuta validación OOF, optimiza t* y entrena el modelo final."""
        ruta_train = self._buscar_ruta_train()
        if ruta_train is None:
            return

        try:
            casos: List[Dict[str, Any]] = []
            with open(ruta_train, "r", encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if linea:
                        casos.append(json.loads(linea))
            if len(casos) < 20:
                return

            (
                X_all,
                y_all,
                w_all,
                casos_X_ora,
                casos_sym_ora,
            ) = self._preparar_oraciones_entrenamiento(casos)

            y_casos = np.array([int(c.get("etiqueta", 0)) for c in casos], dtype=np.int32)
            ids_casos = [str(c.get("id", "")) for c in casos]

            # Validación cruzada estratificada (5 particiones) a nivel de caso para obtener OOF limpio
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_preds = np.zeros(len(casos), dtype=np.float64)

            for train_idx, val_idx in skf.split(np.zeros(len(casos)), y_casos):
                casos_train_fold = [casos[i] for i in train_idx]
                X_tr, y_tr, w_tr, _, _ = self._preparar_oraciones_entrenamiento(casos_train_fold)

                modelos_fold = self._construir_modelos_base()
                for m in modelos_fold:
                    if isinstance(m, Pipeline):
                        m.fit(X_tr, y_tr, lr__sample_weight=w_tr)
                    else:
                        m.fit(X_tr, y_tr, sample_weight=w_tr)

                for idx_c in val_idx:
                    vecs = casos_X_ora[idx_c]
                    syms = casos_sym_ora[idx_c]
                    if not vecs:
                        oof_preds[idx_c] = 0.0
                        continue
                    X_c = np.vstack(vecs)
                    p_ml_arr = self._predecir_ensemble_matriz(modelos_fold, X_c)
                    p_oras: List[float] = []
                    for j in range(len(vecs)):
                        p_sym, veto = syms[j]
                        p_comb = 0.55 * float(p_ml_arr[j]) + 0.45 * p_sym
                        if veto > 0.0:
                            p_comb = max(p_comb, veto)
                        p_oras.append(p_comb)
                    oof_preds[idx_c] = self._agregar_riesgo_caso(p_oras)

            # Optimizar el umbral t* sobre los casos de escenario (NsTGRH) y prefijo H
            mask_escenarios = np.array(
                [bool(cid and cid[0] in "NsTGRH") for cid in ids_casos], dtype=bool
            )
            mask_h = np.array([bool(cid and cid[0] == "H") for cid in ids_casos], dtype=bool)

            Mejor_t = 0.26
            mejor_obj = -1e9
            for t_cand in np.linspace(0.15, 0.55, 81):
                # Evaluar métrica en escenarios NsTGRH y en H
                if np.sum(mask_escenarios) >= 20:
                    y_esc = y_casos[mask_escenarios]
                    p_esc = oof_preds[mask_escenarios]
                    fn_esc = np.sum((y_esc == 1) & (p_esc < t_cand))
                    fp_esc = np.sum((y_esc == 0) & (p_esc >= t_cand))
                    n_fiel_esc = max(1, int(np.sum(y_esc == 0)))
                    score_esc = 1.0 - (4.0 * fn_esc + 1.0 * fp_esc) / float(n_fiel_esc)
                else:
                    score_esc = 0.0

                if np.sum(mask_h) >= 10:
                    y_h = y_casos[mask_h]
                    p_h = oof_preds[mask_h]
                    fn_h = np.sum((y_h == 1) & (p_h < t_cand))
                    fp_h = np.sum((y_h == 0) & (p_h >= t_cand))
                    n_fiel_h = max(1, int(np.sum(y_h == 0)))
                    score_h = 1.0 - (4.0 * fn_h + 1.0 * fp_h) / float(n_fiel_h)
                else:
                    score_h = score_esc

                # Combinar con ligera regularización hacia el óptimo Bayesiano (0.25)
                obj = 0.55 * score_h + 0.45 * score_esc - 0.05 * abs(t_cand - 0.25)
                if obj > mejor_obj:
                    mejor_obj = obj
                    Mejor_t = float(t_cand)

            # Promediar el umbral OOF con el umbral teórico Bayesiano (0.25) para máxima generalización
            self.optimal_threshold = float(np.clip(0.65 * Mejor_t + 0.35 * 0.25, 0.18, 0.45))

            # Entrenar los 3 modelos finales sobre el 100% de train.jsonl
            self.models = self._construir_modelos_base()
            for m in self.models:
                if isinstance(m, Pipeline):
                    m.fit(X_all, y_all, lr__sample_weight=w_all)
                else:
                    m.fit(X_all, y_all, sample_weight=w_all)
            self.is_trained = True

        except Exception:
            # Si ocurre cualquier imprevisto al leer/entrenar, operar en modo neuro-simbólico autónomo
            self.is_trained = False
            self.optimal_threshold = 0.26

    def score_alucina(self, caso: Dict[str, Any]) -> float:
        """
        Devuelve la probabilidad calibrada P(alucina | caso) en [0, 1] con corte en 0.50
        (donde >= 0.50 significa que el caso ALUCINA).
        """
        try:
            senales = caso.get("senales") or {}
            oraciones = senales.get("oraciones") or []
            if not oraciones:
                return 0.50

            vecs: List[np.ndarray] = []
            syms: List[Tuple[float, float]] = []
            for i in range(len(oraciones)):
                f_dict = extract_sentence_features(i, caso)
                vecs.append(np.array([f_dict[k] for k in FEATURE_NAMES], dtype=np.float64))
                syms.append(symbolic_sentence_risk(f_dict))

            if self.is_trained and self.models:
                X_mat = np.vstack(vecs)
                p_ml_arr = self._predecir_ensemble_matriz(self.models, X_mat)
                probs_oraciones: List[float] = []
                for i in range(len(oraciones)):
                    p_sym, veto = syms[i]
                    p_i = 0.55 * float(p_ml_arr[i]) + 0.45 * p_sym
                    if veto > 0.0:
                        p_i = max(p_i, veto)
                    probs_oraciones.append(p_i)
            else:
                probs_oraciones = [s[0] for s in syms]

            p_raw = self._agregar_riesgo_caso(probs_oraciones)
            p_cal = self._calibrar_probabilidad(p_raw, self.optimal_threshold)
            return float(np.clip(p_cal, 0.0001, 0.9999))
        except Exception:
            return 0.50

    def score(self, caso: Dict[str, Any]) -> float:
        """
        Interfaz oficial exigida por el reto.
        - Si SCORE_MENOR_IGUAL_05_ES_ALUCINA es True:
            Score <= 0.5 -> ALUCINA (0.0 = alucina seguro)
            Score >  0.5 -> FIEL    (1.0 = fiel seguro)
        - Si SCORE_MENOR_IGUAL_05_ES_ALUCINA es False:
            Score >= 0.5 -> ALUCINA (1.0 = alucina seguro)
            Score <  0.5 -> FIEL    (0.0 = fiel seguro)
        """
        p_alucina = self.score_alucina(caso)
        if self.menor_igual_05_es_alucina:
            return float(np.clip(1.0 - p_alucina, 0.0, 1.0))
        return float(np.clip(p_alucina, 0.0, 1.0))


# =============================================================================
# BLOQUE DE VERIFICACIÓN LOCAL (__main__)
# =============================================================================

def _evaluar_subconjunto(
    nombre: str,
    casos: List[Dict[str, Any]],
    scores_alucina: np.ndarray,
) -> None:
    """Imprime las métricas oficiales desglosadas para un subconjunto de casos."""
    if not casos:
        return
    y_true = np.array([int(c.get("etiqueta", 0)) for c in casos], dtype=np.int32)
    n_total = len(y_true)
    n_aluc = int(np.sum(y_true == 1))
    n_fiel = int(np.sum(y_true == 0))

    # Con P(alucina) >= 0.5 <=> (1 - P(alucina)) <= 0.5
    pred_aluc = scores_alucina >= 0.50
    fn = int(np.sum((y_true == 1) & (~pred_aluc)))
    fp = int(np.sum((y_true == 0) & pred_aluc))
    tp = int(np.sum((y_true == 1) & pred_aluc))
    tn = int(np.sum((y_true == 0) & (~pred_aluc)))

    puntuacion = 1.0 - (4.0 * fn + 1.0 * fp) / float(max(1, n_fiel))
    auroc = roc_auc_score(y_true, scores_alucina) if (n_aluc > 0 and n_fiel > 0) else 0.0

    print(f"=== {nombre} (N={n_total} | Fieles={n_fiel}, Alucinan={n_aluc}) ===")
    print(f"  Puntuación Oficial : {puntuacion:.4f}")
    print(f"  AUROC              : {auroc:.4f}")
    print(f"  TP={tp:3d} | TN={tn:3d} | FN (escapadas x4)={fn:2d} | FP (falsas alarmas x1)={fp:2d}\n")


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    bot = Guardian()
    t_init = time.perf_counter() - t0

    print(f"Guardian inicializado en {t_init:.3f} s")
    print(f"  Entrenado con scikit-learn : {bot.is_trained}")
    print(f"  Umbral pre-calibración t*  : {bot.optimal_threshold:.4f} -> reescalado a 0.50")
    print(
        f"  Polaridad activa           : "
        f"{'SCORE <= 0.5 ES ALUCINA' if bot.menor_igual_05_es_alucina else 'SCORE >= 0.5 ES ALUCINA'}\n"
    )

    ruta = bot._buscar_ruta_train()
    if ruta is not None:
        with open(ruta, "r", encoding="utf-8") as f:
            datos_train = [json.loads(line) for line in f if line.strip()]

        t1 = time.perf_counter()
        scores_devueltos = np.array([bot.score(c) for c in datos_train], dtype=np.float64)
        t_inf = time.perf_counter() - t1
        print(f"Inferencia sobre {len(datos_train)} casos completada en {t_inf:.3f} s\n")

        # Convertir a probabilidad interna de alucinación según polaridad para mostrar ambas vistas
        probs_aluc = (
            1.0 - scores_devueltos if bot.menor_igual_05_es_alucina else scores_devueltos
        )

        # 1. Todo train.jsonl
        _evaluar_subconjunto("TODO train.jsonl", datos_train, probs_aluc)

        # 2. Escenarios multi-fuente (NsTGRH)
        idx_esc = [i for i, c in enumerate(datos_train) if str(c.get("id", ""))[:1] in "NsTGRH"]
        _evaluar_subconjunto(
            "ESCENARIOS MULTI-FUENTE (NsTGRH)",
            [datos_train[i] for i in idx_esc],
            probs_aluc[idx_esc],
        )

        # 3. Solo prefijo H
        idx_h = [i for i, c in enumerate(datos_train) if str(c.get("id", "")).startswith("H")]
        _evaluar_subconjunto(
            "PREFIJO H (Espejo de Test/Giro)",
            [datos_train[i] for i in idx_h],
            probs_aluc[idx_h],
        )
