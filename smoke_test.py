from pathlib import Path

import numpy as np

from inference import ECGPredictor


def main():
    app_dir = Path(__file__).resolve().parent
    demo = np.load(app_dir / "artifacts" / "demo_samples.npz", allow_pickle=True)
    signals = demo["signals"]
    labels = demo["labels"]
    ecg_ids = demo["ecg_ids"]
    classes = demo["classes"].tolist()

    for model_label in ["FCN-1D", "ResNet-1D"]:
        predictor = ECGPredictor(model_label)
        for signal, target, ecg_id in zip(signals, labels, ecg_ids):
            result = predictor.predict(signal)
            probabilities = result["probabilities"]

            assert probabilities.shape == (5,)
            assert np.isfinite(probabilities).all()
            assert ((probabilities >= 0) & (probabilities <= 1)).all()
            assert np.array_equal((probabilities >= 0.5).astype(int), target.astype(int))

            target_names = [c for c, value in zip(classes, target) if value == 1]
            values = ", ".join(
                f"{class_name}={probability:.3f}"
                for class_name, probability in zip(result["classes"], probabilities)
            )
            print(f"{model_label} ECG {int(ecg_id)} {target_names}: {values}")


if __name__ == "__main__":
    main()
