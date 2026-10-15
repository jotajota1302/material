import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

class Guardian:
    def __init__(self):
        """
        This runs once when the Guardian is created.
        It loads the data and trains your model pipeline.
        """
        print("Initializing Guardian and training ML model...")

        # 1. Load the data
        data = pd.read_json(path_or_buf="train.jsonl", lines=True)

        # 2. Extract and flatten features for training
        data['texto_oracion'] = data['senales'].str['oraciones'].str[0].str['texto']
        data['respaldo'] = data['senales'].str['oraciones'].str[0].str['respaldo_todas']
        data['contradiccion'] = data['senales'].str['oraciones'].str[0].str['contradiccion_todas']
        data['llm_confianza'] = data['senales'].str['oraciones'].str[0].str['llm'].str['confianza']

        veredicto_map = {"respaldada": 0, "contradicha": 1}
        data['llm_veredicto'] = data['senales'].str['oraciones'].str[0].str['llm'].str['veredicto'].map(veredicto_map)

        data['pregunta'] = data['pregunta'].fillna("")
        data['texto_oracion'] = data['texto_oracion'].fillna("")

        # 3. Setup X and y
        features = ['pregunta', 'texto_oracion', 'respaldo', 'contradiccion', 'llm_confianza', 'llm_veredicto']
        X = data[features]
        y = data['etiqueta']

        # 4. Build the pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('text_pregunta', TfidfVectorizer(), 'pregunta'),
                ('text_oracion', TfidfVectorizer(), 'texto_oracion'),
                ('numeric', numeric_transformer, ['respaldo', 'contradiccion', 'llm_confianza', 'llm_veredicto'])
            ]
        )

        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # 5. Train and store the model as a class attribute
        self.model_pipeline.fit(X, y)
        print("Model trained and ready!")

    def score(self, caso):
        pregunta = caso.get("pregunta", "") or ""
        oraciones = caso.get("senales", {}).get("oraciones", [])

        if len(oraciones) > 0:
            oracion_0 = oraciones[0]
            texto_oracion = oracion_0.get("texto", "") or ""
            respaldo = float(oracion_0.get("respaldo_todas") if oracion_0.get("respaldo_todas") is not None else 0.0)
            contradiccion = float(oracion_0.get("contradiccion_todas") if oracion_0.get("contradiccion_todas") is not None else 0.0)

            llm_info = oracion_0.get("llm") or {}
            conf = llm_info.get("confianza")
            llm_confianza = float(conf) if conf is not None else 0.0

            veredicto_str = llm_info.get("veredicto")
            if veredicto_str == "respaldada":
                llm_veredicto = 0.0
            elif veredicto_str == "contradicha":
                llm_veredicto = 1.0
            elif veredicto_str == "no_respaldada":
                llm_veredicto = 2.0
            else:
                llm_veredicto = -1.0
        else:
            texto_oracion = ""
            respaldo = 0.0
            contradiccion = 0.0
            llm_confianza = 0.0
            llm_veredicto = -1.0

        df_caso = pd.DataFrame([{
            'pregunta': pregunta,
            'texto_oracion': texto_oracion,
            'respaldo': respaldo,
            'contradiccion': contradiccion,
            'llm_confianza': llm_confianza,
            'llm_veredicto': llm_veredicto
        }])

        prediction = self.model_pipeline.predict(df_caso)[0]
        return float(prediction)
