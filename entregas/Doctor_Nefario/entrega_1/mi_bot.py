"""Bot de ejemplo para El Guardián. Solo usa Python estándar: no necesita
instalar ni ejecutar ningún modelo.

Entregáis una clase `Guardian` con un método:
  - score(self, caso) -> float en [0,1]: probabilidad de que la respuesta afirme
    algo que las fuentes NO respaldan (1 = alucina, 0 = fiel).

Cada caso trae, además de pregunta/fuentes/respuesta, el campo `senales` con lo
que un modelo de NLI ya ha calculado por vosotros (ver README):
  caso["senales"]["oraciones"]  -> una entrada por oración de la respuesta
  caso["senales"]["fragmentos"] -> trozos más pequeños de cada oración
Cada entrada tiene `respaldo_por_fuente`, `contradiccion_por_fuente`,
`similitud_por_fuente` (una cifra por fuente, de 0 a 1).

Este ejemplo mira cada oración, se queda con la fuente que más la respalda y
marca alucinación si alguna oración queda sin respaldo. Es un punto de partida
deliberadamente simple: compara cada oración con cada fuente POR SEPARADO, así
que falla cuando una respuesta fiel combina datos de varias fuentes.
"""


class Guardian:

    def reescalar_score(self, score_original):
        umbral_bot = 0.167

        if score_original < umbral_bot:
            return (score_original / umbral_bot) * 0.5
        else:
            return 0.5 + ((score_original - umbral_bot) / (1.0 - umbral_bot)) * 0.5

    def score(self, caso):
        peor_respaldo = 1.0
        for oracion in caso["senales"]["oraciones"]:
            mejor_fuente = max(oracion["respaldo_por_fuente"], default=0.0)
            peor_respaldo = min(peor_respaldo, mejor_fuente)
        
        # 1. Calculamos la probabilidad original de alucinación
        score_original = 1.0 - peor_respaldo
        
        # 2. La reescalamos usando self. y la devolvemos
        return self.reescalar_score(score_original)