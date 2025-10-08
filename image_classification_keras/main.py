import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from pathlib import Path

def main():
    model = MobileNetV2(weights="imagenet")
    img_path = Path(__file__).parent / "samples" / "cat.jpg"
    if not img_path.exists():
        print("⚠️ Please add a test image (cat.jpg) to samples/")
        return
    img = image.load_img(img_path, target_size=(224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    for label, desc, score in decode_predictions(preds, top=3)[0]:
        print(f"{desc}: {score:.4f}")

if __name__ == "__main__":
    main()
