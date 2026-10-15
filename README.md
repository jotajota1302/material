# El Guardián — material del reto

Bienvenidos al hackathon. Vais a construir un **detector de alucinaciones**: ante la
respuesta de una IA y sus fuentes, vuestro bot decide si la respuesta afirma algo
que las fuentes **no respaldan**.

Solo necesitáis **Python 3.9 o superior**. No se instala ni se ejecuta ningún modelo.

> **Giro publicado (15:30):** ya está [`kit/giro_publico.jsonl`](kit/giro_publico.jsonl) (220 casos: normativa e informes técnicos con respuestas largas). Haced `git pull` o descargadlo. Desde ahora cada entrega es **un único CSV con test + giro**:
>
> `python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl giro_publico.jsonl --salida predicciones.csv`

## 1. Leed el enunciado

**[Enunciado_El_Guardian.docx](Enunciado_El_Guardian.docx)**: el problema, un caso de
ejemplo, qué recibís, cómo se puntúa y el calendario. Pulsad *Download* (o el icono
de descarga) para abrirlo en Word.

## 2. Descargad el kit

- Sin git: botón verde **Code → Download ZIP** y descomprimidlo.
- Con git: `git clone https://github.com/jotajota1302/material.git`

Todo el código y los datos están en la carpeta [`kit/`](kit/). Leed [`kit/README.md`](kit/README.md) (reglas,
señales, métrica) y [`kit/EJEMPLOS.md`](kit/EJEMPLOS.md) (qué cuenta como alucinación).

| Fichero | Para qué |
|---|---|
| `kit/README.md` | Reglas, señales, entrega y puntuación. |
| `kit/EJEMPLOS.md` | Casos comentados de cada tipo de alucinación. |
| `kit/ejemplo_bot.py` | Un bot que ya funciona: vuestro punto de partida. |
| `kit/evaluar_local.py` | Mide vuestro bot sobre `train.jsonl` con la métrica del leaderboard. |
| `kit/generar_predicciones.py` | Genera el CSV que entregáis. |
| `kit/train.jsonl` | Casos etiquetados para desarrollar y calibrar. |
| `kit/test_publico.jsonl` | Los casos que hay que puntuar (sin etiquetas). |

## 3. Primeros pasos

```
cd kit
python evaluar_local.py --bot ejemplo_bot.py --datos train.jsonl
```

Copiad `ejemplo_bot.py` a `mi_bot.py`, mejoradlo y medid cada cambio con `evaluar_local.py`.

## 4. Calendario

**Jueves 15 de octubre**

| Hora | Qué pasa |
|---|---|
| 10:00 | Presentación técnica del reto. Material disponible. Entregas sin límite (solo test). |
| 14:15 | Comida. |
| **15:30** | **Giro:** aparecerá aquí `kit/giro_publico.jsonl`. Desde entonces cada entrega lleva test + giro en un solo CSV. |
| **19:00** | Cierre de entregas. Se revela la clasificación final. |
| 19:00–20:15 | Los 5 primeros reproducen su bot sobre un lote nuevo delante del jurado. |
| **20:30** | Anuncio de los finalistas. |

**Viernes 16 de octubre**

| Hora | Qué pasa |
|---|---|
| 9:30 | Pitch de los finalistas (5 minutos), en el Salón de Actos. |
| 12:00 | Entrega de premios y clausura. |

A las 15:30 actualizad el repositorio (`git pull`), o descargad solo `kit/giro_publico.jsonl`.

## 5. Cómo entregar

Cada entrega es **`predicciones.csv` + `mi_bot.py`**, en la carpeta
[`entregas/`](entregas/) de este repositorio, mediante un *Pull Request*.
Las instrucciones paso a paso están en [`entregas/README.md`](entregas/README.md).
