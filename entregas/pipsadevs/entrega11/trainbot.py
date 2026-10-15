import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data = pd.read_json(path_or_buf="train.jsonl", lines=True)

data['texto_oracion'] = data['senales'].str['oraciones'].str[0].str['texto']

data['respaldo'] = data['senales'].str['oraciones'].str[0].str['respaldo_todas']
data['contradiccion'] = data['senales'].str['oraciones'].str[0].str['contradiccion_todas']
data['llm_confianza'] = data['senales'].str['oraciones'].str[0].str['llm'].str['confianza']

data['llm_veredicto'] = data['senales'].str['oraciones'].str[0].str['llm'].str['veredicto']

veredicto_map = {
    "respaldada": 0,
    "contradicha": 1
}

data['llm_veredicto'] = data['llm_veredicto'].map(veredicto_map)
data['pregunta'] = data['pregunta'].fillna("")
data['texto_oracion'] = data['texto_oracion'].fillna("")
data['llm_veredicto'] = data['llm_veredicto'].fillna(-1)
features = ['pregunta', 'texto_oracion', 'respaldo', 'contradiccion', 'llm_confianza', 'llm_veredicto']
X = data[features]
y = data['etiqueta']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # Replaces missing numbers with 0
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text_pregunta', TfidfVectorizer(), 'pregunta'),
        ('text_oracion', TfidfVectorizer(), 'texto_oracion'),
        ('numeric', numeric_transformer, ['respaldo', 'contradiccion', 'llm_confianza', 'llm_veredicto'])
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("Training model...")
model_pipeline.fit(X_train, y_train)
print("Model trained successfully!")

accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
