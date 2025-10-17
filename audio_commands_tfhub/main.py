import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub

def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
    wav = wav.astype("float32")
    m = np.max(np.abs(wav)) or 1.0
    if m > 1.0:
        wav = wav / m
    return wav, target_sr

def load_yamnet_labels(yamnet_model) -> list[str]:
    if hasattr(yamnet_model, "class_map_path"):
        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    else:
        assets = getattr(yamnet_model, "assets", None)
        class_map_path = None
        if assets:
            for a in assets:
                p = a.numpy().decode()
                if p.endswith("yamnet_class_map.csv"):
                    class_map_path = p
                    break
        if class_map_path is None:
            raise RuntimeError("yamnet_class_map.csv not found")
    names = []
    with open(class_map_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row.get("display_name", "").strip())
    if not names:
        raise RuntimeError("empty class map")
    return names

def main():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    wav_path = Path(__file__).parent / "samples" / "sample.wav"
    if not wav_path.exists():
        print("⚠️ add samples/sample.wav")
        return
    wav, _ = load_audio(wav_path, target_sr=16000)
    waveform = tf.convert_to_tensor(wav, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top5_idx = tf.argsort(mean_scores, direction="DESCENDING")[:5].numpy()
    labels = load_yamnet_labels(model)
    print("=== Top predicted audio events ===")
    for i in top5_idx:
        name = labels[i] if i < len(labels) else f"Class #{i}"
        score = float(mean_scores[i].numpy())
        print(f"- {name} (score={score:.4f})")

if __name__ == "__main__":
    main()
