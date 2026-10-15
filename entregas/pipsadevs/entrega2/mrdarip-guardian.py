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


from pathlib import Path
from train_mrdarip_guardian import load_model
from infernal_flatternify import infernal_flatternify

class Guardian:
    def score(self, caso):

      # 1. Load the predictor function once
      predict_fn = load_model(Path("train.model.joblib"))

      # 3. Get predictions (returns a numpy array of 0s and 1s, e.g., array([1, 0]))
      predictions = predict_fn([caso])

      for record, res in zip([caso], predictions):
          print(
              f"Predicted EXPECTED={res['prediction']} "
              f"(confidence: {res['confidence']:.1%}, P(True)={res['prob_true']:.3f}) "
              f"for {record}"
          )

      return predictions[0]['prob_true']  # Return the probability of being True (alucina)