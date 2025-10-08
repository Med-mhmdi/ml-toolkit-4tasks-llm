import tensorflow as tf, tensorflow_hub as hub, numpy as np, soundfile as sf, librosa
from pathlib import Path

def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav, target_sr

def main():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    wav_path = Path(__file__).parent / "samples" / "sample.wav"
    if not wav_path.exists():
        print("⚠️ Please add a short sample.wav file to samples/")
        return
    wav, sr = load_audio(wav_path)
    scores, embeddings, spectrogram = model(wav)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top5 = tf.argsort(mean_scores, direction='DESCENDING')[:5]
    print("Top predicted audio events:")
    for i in top5.numpy():
        print(f"- Class #{i} (score={mean_scores[i].numpy():.4f})")

if __name__ == "__main__":
    main()
