from pathlib import Path
import pickle
import sys

import numpy as np
import torch


APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.basic_conv1d import fcn_wang  # noqa: E402
from models.resnet1d import resnet1d_wang  # noqa: E402


MODEL_NAMES = {
    "FCN-1D": "fastai_fcn_wang",
    "ResNet-1D": "fastai_resnet1d_wang",
}


class ECGPredictor:
    def __init__(self, model_label, device=None):
        if model_label not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_label}")

        self.model_label = model_label
        self.model_name = MODEL_NAMES[model_label]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with open(APP_DIR / "artifacts" / "mlb.pkl", "rb") as f:
            self.classes = list(pickle.load(f).classes_)
        with open(APP_DIR / "artifacts" / "standard_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        if model_label == "FCN-1D":
            model = fcn_wang(
                num_classes=len(self.classes),
                input_channels=12,
                lin_ftrs_head=[128],
                ps_head=0.5,
            )
        else:
            model = resnet1d_wang(
                num_classes=len(self.classes),
                input_channels=12,
                kernel_size=5,
                lin_ftrs_head=[128],
                ps_head=0.5,
            )

        checkpoint_path = (
            APP_DIR
            / "artifacts"
            / "models"
            / self.model_name
            / "models"
            / f"{self.model_name}.pth"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()

    @staticmethod
    def validate_signal(signal):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2:
            raise ValueError("Ожидается двумерный массив: отсчеты x 12 отведений.")
        if signal.shape[1] != 12 and signal.shape[0] == 12:
            signal = signal.T
        if signal.shape[1] != 12:
            raise ValueError(f"Ожидается 12 отведений, получена форма {signal.shape}.")
        if signal.shape[0] < 250:
            raise ValueError("Длина сигнала должна быть не менее 250 отсчетов.")
        if not np.isfinite(signal).all():
            raise ValueError("Сигнал содержит NaN или бесконечные значения.")
        return signal

    def preprocess(self, signal):
        signal = self.validate_signal(signal)
        shape = signal.shape
        return self.scaler.transform(signal.reshape(-1, 1)).reshape(shape).astype(np.float32)

    @staticmethod
    def make_windows(signal, window_size=250, stride=125):
        if len(signal) == window_size:
            return signal[np.newaxis, ...]

        starts = list(range(0, len(signal) - window_size + 1, stride))
        last_start = len(signal) - window_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return np.stack([signal[start : start + window_size] for start in starts])

    def predict(self, signal):
        standardized = self.preprocess(signal)
        windows = self.make_windows(standardized)
        tensor = torch.from_numpy(windows.transpose(0, 2, 1)).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            window_probabilities = torch.sigmoid(logits)
            probabilities = window_probabilities.max(dim=0).values

        return {
            "classes": self.classes,
            "probabilities": probabilities.cpu().numpy(),
            "window_probabilities": window_probabilities.cpu().numpy(),
            "window_count": len(windows),
            "device": str(self.device),
        }


def predict_ensemble(predictors, signal):
    results = [predictor.predict(signal) for predictor in predictors]
    probabilities = np.mean([result["probabilities"] for result in results], axis=0)
    return {
        "classes": results[0]["classes"],
        "probabilities": probabilities,
        "window_probabilities": None,
        "window_count": results[0]["window_count"],
        "device": results[0]["device"],
    }
