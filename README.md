# ML Toolkit — 4 Tasks + Local LLM (Full Version)

This repository demonstrates **four ML tasks using ready-made pre-trained models** plus a **local LLM**.
Frameworks used: Hugging Face, TensorFlow Hub, Keras, PyTorch/Ultralytics, Ollama.

## Tasks
| # | Type | Framework | Description |
|---|------|------------|--------------|
| 1 | Text | Hugging Face | Sentiment Analysis |
| 2 | Audio | TensorFlow Hub | Audio Event / Command Detection |
| 3 | Image | Keras Applications | Image Classification |
| 4 | Video | PyTorch (YOLOv8) | Object Detection |
| + | LLM | Ollama | Local LLM Run (offline) |

---

## How to Run

### 1️⃣ Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1
```

### 2️⃣ Install dependencies
Install only what you need:
```bash
pip install -r env/requirements_text.txt
pip install -r env/requirements_audio.txt
pip install -r env/requirements_image.txt
pip install -r env/requirements_video.txt
```
Or everything:
```bash
pip install -r env/requirements_all.txt
```

### 3️⃣ Run tasks
```bash
python text_sentiment_hf/main.py
python audio_commands_tfhub/main.py
python image_classification_keras/main.py
python video_detection_yolo/main.py
```

### 4️⃣ Run local LLM
```bash
ollama run mistral:7b-instruct < llm_local/prompts/hello.txt
```

---

## License
MIT
