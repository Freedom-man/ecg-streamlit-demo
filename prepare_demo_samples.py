from pathlib import Path
import ast
import pickle

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ptbxl"
EXP_DIR = ROOT / "output" / "final_superdiagnostic_30ep"
OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def main():
    metadata = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    raw = np.load(DATA_DIR / "raw100.npy", mmap_mode="r", allow_pickle=True)
    y_test = np.load(EXP_DIR / "data" / "y_test.npy", allow_pickle=True)
    y_pred_fcn = np.load(
        EXP_DIR / "models" / "fastai_fcn_wang" / "y_test_pred.npy",
        allow_pickle=True,
    )
    y_pred_resnet = np.load(
        EXP_DIR / "models" / "fastai_resnet1d_wang" / "y_test_pred.npy",
        allow_pickle=True,
    )

    with open(EXP_DIR / "data" / "mlb.pkl", "rb") as f:
        classes = list(pickle.load(f).classes_)

    statements = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
    diagnostic_map = (
        statements[statements.diagnostic == 1.0]["diagnostic_class"]
        .dropna()
        .to_dict()
    )

    def aggregate_superclasses(value):
        codes = ast.literal_eval(value)
        return sorted({diagnostic_map[code] for code in codes if code in diagnostic_map})

    aggregated = metadata.scp_codes.apply(aggregate_superclasses)
    eligible = aggregated.apply(len).to_numpy() > 0
    test_positions = np.flatnonzero((metadata.strat_fold.to_numpy() == 10) & eligible)
    test_ids = metadata.index.to_numpy()[test_positions]
    test_labels = aggregated.iloc[test_positions].tolist()
    reconstructed_y = np.asarray(
        [[int(class_name in labels) for class_name in classes] for labels in test_labels]
    )

    if not np.array_equal(reconstructed_y, y_test):
        raise RuntimeError("PTB-XL test signal order does not match y_test.npy")

    desired = ["NORM", "MI", "CD", "STTC", "HYP"]
    selected = []
    for class_name in desired:
        class_idx = classes.index(class_name)
        candidates = np.flatnonzero(
            (y_test[:, class_idx] == 1)
            & (y_test.sum(axis=1) == 1)
            & ((y_pred_fcn >= 0.5).astype(int) == y_test).all(axis=1)
            & ((y_pred_resnet >= 0.5).astype(int) == y_test).all(axis=1)
        )
        if len(candidates) == 0:
            raise RuntimeError(f"No stable demonstration sample found for {class_name}")

        other_indices = [idx for idx in range(len(classes)) if idx != class_idx]
        target_confidence = np.minimum(
            y_pred_fcn[candidates, class_idx],
            y_pred_resnet[candidates, class_idx],
        )
        false_confidence = np.maximum(
            y_pred_fcn[candidates][:, other_indices].max(axis=1),
            y_pred_resnet[candidates][:, other_indices].max(axis=1),
        )
        margin = target_confidence - false_confidence
        selected.append(int(candidates[np.argmax(margin)]))

    multi_mask = y_test.sum(axis=1) >= 2
    exact_fcn = ((y_pred_fcn >= 0.5).astype(int) == y_test).all(axis=1)
    exact_resnet = ((y_pred_resnet >= 0.5).astype(int) == y_test).all(axis=1)
    multi_candidates = np.flatnonzero(multi_mask & exact_fcn & exact_resnet)
    if len(multi_candidates):
        margins = []
        for idx in multi_candidates:
            positive = y_test[idx] == 1
            negative = ~positive
            positive_confidence = min(
                y_pred_fcn[idx, positive].min(),
                y_pred_resnet[idx, positive].min(),
            )
            negative_confidence = max(
                y_pred_fcn[idx, negative].max(),
                y_pred_resnet[idx, negative].max(),
            )
            margins.append(positive_confidence - negative_confidence)
        selected.append(int(multi_candidates[int(np.argmax(margins))]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "demo_samples.npz",
        signals=np.asarray(raw[test_positions[selected]], dtype=np.float32),
        ecg_ids=test_ids[selected],
        labels=y_test[selected],
        classes=np.asarray(classes),
    )

    for idx in selected:
        labels = [c for c, value in zip(classes, y_test[idx]) if value == 1]
        fcn_values = {c: round(float(v), 3) for c, v in zip(classes, y_pred_fcn[idx])}
        resnet_values = {c: round(float(v), 3) for c, v in zip(classes, y_pred_resnet[idx])}
        print(idx, int(test_ids[idx]), labels, "FCN", fcn_values, "ResNet", resnet_values)


if __name__ == "__main__":
    main()
