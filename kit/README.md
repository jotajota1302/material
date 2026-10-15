# El Guardián — kit de participantes

Construid un **detector de alucinaciones**: ante la respuesta de una IA y sus
fuentes, decidid si la respuesta afirma algo que las fuentes **no respaldan**.

**No hay que instalar nada más que Python 3** (3.9 o superior), y no se ejecuta
ningún modelo de IA. El trabajo pesado (un modelo de NLI y un LLM juez) ya lo hemos hecho
nosotros: cada caso trae sus resultados precalculados. Funciona en cualquier
portátil.

## Qué hay en este kit

| Fichero | Para qué |
|---|---|
| `README.md` | Este documento: reglas, señales, entrega y métrica. |
| `EJEMPLOS.md` | Casos comentados de cada tipo de alucinación y el criterio de etiquetado. |
| `ejemplo_bot.py` | Un bot que ya funciona. Es vuestro punto de partida. |
| `evaluar_local.py` | Mide vuestro bot sobre `train.jsonl` con la misma métrica del leaderboard. |
| `generar_predicciones.py` | Genera el CSV que entregáis. |
| `train.jsonl` | Casos etiquetados, con sus señales, para desarrollar y calibrar. |
| `test_publico.jsonl` | Los casos que hay que puntuar (sin etiquetas). A las 15:30 recibiréis `giro_publico.jsonl`. |

## Qué construís

Un fichero Python con una clase `Guardian`:

```python
class Guardian:
    def score(self, caso) -> float:
        # caso = {"pregunta": str, "fuentes": [str, ...], "respuesta": str, "senales": {...}}
        # devolved la probabilidad [0,1] de que la respuesta ALUCINE
        ...
```

## Las señales: lo que el modelo ya ha calculado

Hemos pasado cada caso por tres modelos y os damos sus resultados:

- un modelo de **NLI** (*natural language inference*: dado un texto A y una frase
  B, estima si A respalda B, la contradice o ninguna de las dos);
- un modelo de **similitud** de textos;
- un **LLM juez** (MiniMax), al que le preguntamos, oración a oración, si las
  fuentes la respaldan.

Todo está en `caso["senales"]`:

```
caso["senales"]["oraciones"]    una entrada por oración de la respuesta
caso["senales"]["fragmentos"]   trozos más pequeños (cortados por comas y conjunciones);
                                cada uno indica en "oracion" de qué oración sale
```

Cada entrada (oración o fragmento) trae:

| Campo | Qué es |
|---|---|
| `texto` | El trozo de respuesta evaluado. |
| `respaldo_por_fuente` | Una cifra 0–1 por fuente: cuánto respalda esa fuente, ella sola, al texto. |
| `contradiccion_por_fuente` | Una cifra 0–1 por fuente: cuánto lo contradice. |
| `similitud_por_fuente` | Una cifra 0–1 por fuente: cuánto se parecen (tema y palabras, no verdad). |
| `respaldo_todas` | Cuánto respaldan al texto todas las fuentes leídas juntas. |
| `contradiccion_todas` | Cuánto lo contradicen todas las fuentes juntas. |
| `llm` (solo en oraciones) | El veredicto del LLM juez: `{"veredicto": "respaldada" / "no_respaldada" / "contradicha", "fuente": índice de la fuente (desde 0) o null, "confianza": 0–1}`. Si el LLM no contestó bien, `veredicto` es `null`. |

Las señales **no son la respuesta**: los modelos se equivocan, y no en los
mismos sitios. El NLI da falsas alarmas con las respuestas que combinan varias
fuentes y se lía con los números. El LLM juez acierta mucho más, pero se fía
demasiado: a veces da por buena una frase que añade un detalle que nadie ha dicho. Vuestro trabajo es decidir qué señales creer, cómo
combinarlas y qué comprobaciones añadir en Python (por ejemplo, cifras, fechas y
a quién pertenece cada dato) para cubrir lo que el modelo no ve.

## Puesta en marcha

1. Probad el bot de ejemplo:
   ```
   python evaluar_local.py --bot ejemplo_bot.py --datos train.jsonl
   ```
2. Copiad `ejemplo_bot.py` a `mi_bot.py` y mejoradlo. Medid cada cambio con
   `evaluar_local.py`. **Ojo: el train no es igual que el test.** Los casos con id `H…` se
   escribieron como el test; el resto del train es más fácil. Si validáis y
   calibráis con todo el train, os vais a confiar: medid sobre todo con
   `--prefijo H` (y validad sin entrenar sobre esos mismos casos).

## Cómo entregáis

Ejecutáis vuestro bot **en vuestra máquina** y entregáis un fichero de predicciones:

```
python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl --salida predicciones.csv
```

Desde las 15:30, con el giro publicado, generáis un único CSV con los dos ficheros:

```
python generar_predicciones.py --bot mi_bot.py --test test_publico.jsonl giro_publico.jsonl --salida predicciones.csv
```

Mandáis **`predicciones.csv` y vuestro `mi_bot.py`** juntos en cada entrega.

**Formato del CSV:** una línea `id,score` por caso, separador coma y **punto
decimal** (`t0001,0.73`). La cabecera `id,score` es opcional. Tiene que haber una
línea por cada caso, sin ids repetidos. Si lo abrís y guardáis con Excel en
español, lo convertirá a `;` y `0,73` y se rechazará: generadlo siempre con el
script.

**Entregas:** sin límite. Podéis entregar tantas veces como queráis antes y
después del giro. Ojo: la clasificación final usa vuestra entrega de después del
giro con mejor nota **pública**, puntuada en la parte **privada**, así que subir
muchas variantes para afinar contra el leaderboard no os ayuda.

## Cómo se puntúa

El organizador convierte cada score en una decisión: **score ≥ 0,5 = alucina**.

```
puntuación = 1 − (4 · alucinaciones que se escapan + falsas alarmas) / nº de casos fieles
```

- **1** = no falláis ninguno.
- **0** = lo mismo que marcar todo como alucinación (no hace falta un bot para eso).
- **Menos de 0** = peor que no hacer nada.

Dejar pasar una alucinación pesa **4 veces** más que una falsa alarma, porque en
un guardrail real es mucho peor. Si vuestro mejor corte no es 0,5, reescalad el
score para que lo sea (`evaluar_local.py` os dice cuál es vuestro mejor corte).
El **AUROC** (si ordenáis bien los casos, al margen del corte) se muestra al lado
y sirve para desempatar.

**Parte pública y privada.** El test (y el giro) se divide en dos mitades que no
conocéis. El leaderboard en directo muestra vuestra puntuación en la mitad
**pública**. La clasificación final se calcula con la mitad **privada**, usando
vuestra entrega de después del giro con mejor puntuación pública, y se revela al
cierre. Afinar el bot a base de subir variantes y mirar el leaderboard no os
servirá: un bot que generaliza puntúa igual en las dos mitades.

**Nota final:** 60 × vuestra puntuación final (mínimo 0) + hasta 40 puntos del
jurado. Los 5 primeros de la clasificación final son los finalistas: se anuncian
el jueves a las 20:30 y presentan su pitch (5 minutos) el viernes a las 9:30.

## Reglas

- **Un bot por equipo**, en Python.
- **Podéis usar:** las señales de cada caso, la librería estándar de Python,
  `numpy` y `scikit-learn` (por ejemplo, para combinar señales o entrenar un
  clasificador con `train.jsonl`). Nada de eso es obligatorio: se puede ganar con
  Python puro.
- **No podéis usar:** modelos de IA de ningún tipo (ni APIs externas, ni modelos
  locales, ni otros modelos de NLI o embeddings). Así nadie gana por tener mejor
  portátil o pagar una API: todos parten de las mismas señales.
- **Podéis leer** `test_publico.jsonl` para entender el formato. **No podéis**
  etiquetarlo a mano ni meter en el código reglas o datos pensados para casos
  concretos del test.
- **Reproducción (jueves, de 19:00 a 20:15):** los 5 primeros ejecutan su
  `mi_bot.py` delante del jurado sobre un **lote nuevo** que nadie ha visto, del
  mismo tipo que test y giro. Quedaos hasta el anuncio de las 20:30: si un equipo
  queda descalificado, entra el siguiente. Un bot honesto puntúa parecido. Si el resultado cae muy por debajo de
  vuestra puntuación final, o el código no corresponde con lo entregado, se
  descalifica la entrega.

## Los 4 tipos de alucinación

1. **Distorsión:** dato equivocado. Puede ser una cifra, una fecha, una unidad o
   un dato que sí está en las fuentes pero es de **otra entidad**.
2. **Contradicción:** dice lo contrario que una fuente, muchas veces sin ningún
   «no» («aumentó» frente a «se redujo»).
3. **No respaldado:** añade algo plausible que ninguna fuente dice: una causa, una
   fecha, una atribución, un detalle.
4. **Sobregeneralización:** amplía el alcance («en tres hospitales» → «en los
   hospitales del país»).

El criterio completo que usamos para etiquetar está en `EJEMPLOS.md`.

## Pistas

- **Las respuestas combinan varias fuentes.** Una frase fiel puede juntar datos
  de tres fuentes distintas. Si miráis solo `respaldo_por_fuente`, ninguna fuente
  la respalda entera y daréis falsa alarma (es lo que le pasa al bot de ejemplo).
  Mirad también `respaldo_todas` y los fragmentos.
- **Una sola afirmación falsa basta.** En una respuesta larga, si promediáis las
  señales de todas las oraciones, la falsa se diluye. Buscad la peor.
- **Cuidado con quién hizo qué.** Muchas fuentes hablan de dos entidades parecidas
  (dos empresas, dos tarifas, dos dosis). Que un número aparezca en las fuentes no
  significa que esté bien atribuido.
- **El LLM juez no es infalible.** Dejar pasar una alucinación cuesta 4 veces más
  que una falsa alarma: pensad cuándo conviene desconfiar de un «respaldada»
  (poca confianza, el NLI no encuentra respaldo, la frase añade algo que no está
  en la fuente que cita…).
- **No os fiéis de atajos.** La longitud de la respuesta, los superlativos o las
  negaciones no predicen la etiqueta en el test.
- **El giro de las 15:30 cambia de terreno:** documentos técnicos y legales y
  respuestas más largas. Lo que solo funciona en el formato del train caerá ahí.
