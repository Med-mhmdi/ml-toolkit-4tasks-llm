from transformers import pipeline
from pathlib import Path

def load_samples(path: Path):
    if not path.exists():
        return ["I love this!", "This is bad.", "It’s fine."]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    clf = pipeline("sentiment-analysis")
    samples = load_samples(Path(__file__).parent / "samples" / "sample_texts.txt")
    print("=== Sentiment Inference ===")
    for s in samples:
        res = clf(s)[0]
        print(f"{s} → {res['label']} ({res['score']:.4f})")

if __name__ == "__main__":
    main()
