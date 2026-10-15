"""Bot de ejemplo para El Guardian. Solo usa Python estandar: no necesita
instalar ni ejecutar ningun modelo.

Entregais una clase `Guardian` con un metodo:
  - score(self, caso) -> float en [0,1]: probabilidad de que la respuesta afirme
    algo que las fuentes NO respaldan (1 = alucina, 0 = fiel).

Cada caso trae, ademas de pregunta/fuentes/respuesta, el campo `senales` con lo
que un modelo de NLI ya ha calculado por vosotros (ver README):
  caso["senales"]["oraciones"]  -> una entrada por oracion de la respuesta
  caso["senales"]["fragmentos"] -> trozos mas pequenos de cada oracion
Cada entrada tiene `respaldo_por_fuente`, `contradiccion_por_fuente`,
`similitud_por_fuente` (una cifra por fuente, de 0 a 1).

Este ejemplo mira cada oracion, se queda con la fuente que mas la respalda y
marca alucinacion si alguna oracion queda sin respaldo. Es un punto de partida
deliberadamente simple: compara cada oracion con cada fuente POR SEPARADO, asi
que falla cuando una respuesta fiel combina datos de varias fuentes.
"""


class Guardian:
    def score(self, caso):
        # codigo
        return;
