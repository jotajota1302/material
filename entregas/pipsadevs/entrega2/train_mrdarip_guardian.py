import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from infernal_flatternify import infernal_flatternify


def load_model(artifact_path: Path):
    """Load a saved artifact and return a predictor with confidence scores."""
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    def predict(records):
        flattened = [infernal_flatternify(record) for record in records]
        features = [
            {key: int(value) if isinstance(value, bool) else value
             for key, value in record.items() if key != "EXPECTED"}
            for record in flattened
        ]
        X = vectorizer.transform(features)
        
        preds = model.predict(X)
        # predict_proba returns [[prob_0, prob_1], ...] for each row
        probs = model.predict_proba(X)

        results = []
        for pred, (prob_false, prob_true) in zip(preds, probs):
            is_true = bool(pred)
            results.append({
                "prediction": is_true,
                "confidence": float(prob_true if is_true else prob_false),
                "prob_true": float(prob_true),
            })
        return results

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL records and train a Logistic Regression model."
    )
    parser.add_argument("input", type=Path, help="Path to input .jsonl file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output path (default: <input>.preprocessed.jsonl)",
    )
    parser.add_argument(
        "-m", "--model-output", type=Path,
        help="Model artifact path (default: <input>.model.joblib)",
    )
    args = parser.parse_args()

    if args.input.suffix.lower() != ".jsonl":
        parser.error("input must be a .jsonl file")
    output = args.output or args.input.with_name(
        f"{args.input.stem}.preprocessed.jsonl"
    )
    model_output = args.model_output or args.input.with_name(
        f"{args.input.stem}.model.joblib"
    )

    with args.input.open(encoding="utf-8") as infile, output.open(
        "w", encoding="utf-8"
    ) as outfile:
        features = []
        targets = []
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            flattened = infernal_flatternify(record)
            if not isinstance(flattened, dict):
                raise TypeError("infernal_flatternify must return a dictionary")
            if "EXPECTED" not in flattened:
                raise ValueError(f'Missing "EXPECTED" target on line {line_number}')
            outfile.write(json.dumps(flattened, ensure_ascii=False) + "\n")
            targets.append(int(bool(flattened["EXPECTED"])))
            features.append({
                key: int(value) if isinstance(value, bool) else value
                for key, value in flattened.items()
                if key != "EXPECTED"
            })

    if not features:
        raise ValueError("Input dataset contains no records")
    if len(set(targets)) < 2:
        raise ValueError(
            "Logistic Regression requires both target classes (0 and 1); "
            "the input data has only one class in its EXPECTED values"
        )

    vectorizer = DictVectorizer(sparse=True, dtype=float)
    X = vectorizer.fit_transform(features)
    model = LogisticRegression()
    model.fit(X, targets)
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "feature_names": vectorizer.get_feature_names_out(),
        },
        model_output,
    )


if __name__ == "__main__":
    main()