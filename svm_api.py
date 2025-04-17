from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime

app = FastAPI()

# Model ve scaler dosyalarını yükle
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sınıf sıralamasını al (örn. array([0, 1]) → 0: Successful, 1: Unsuccessful)
model_classes = model.classes_

# API'ye gönderilecek veri modeli
class Candidate(BaseModel):
    experience: int
    technical_point: int

@app.get("/")
def read_root():
    return {"message": "🚀 Aday Tahmin API'si çalışıyor ve hazır!"}

@app.post("/predict_proba", summary="Aday başarı tahmini", tags=["Tahmin"])
def predict_with_score(candidate: Candidate):
    """
    🔍 Girilen aday bilgilerine göre başarı durumu tahmini yapılır.

    - 0 → Successful (Geçer)
    - 1 → Unsuccessful (Geçemez)
    """

    # 1. Aday verisini ölçekle
    data = np.array([[candidate.experience, candidate.technical_point]])
    data_scaled = scaler.transform(data)

    # 2. Tahmin ve karar skoru al
    true_class = model.predict(data_scaled)[0]  # 0 veya 1
    decision_score = model.decision_function(data_scaled)[0]

    # 3. Sınıf etiketine göre açıklama üret
    result = "Successful" if true_class == 0 else "Unsuccessful"

    # 4. Konsola log yaz
    print("📚 Model sınıf sırası:", model.classes_)
    print("🔍 Tahmin edilen sınıf:", true_class)
    print("📉 Decision score:", decision_score)

    # 5. Tahmin kaydını dosyaya logla
    log_input(candidate, endpoint="/predict_proba", prediction=result)

    # 6. JSON cevap
    return {
        "prediction": result,
        "decision_score": round(decision_score, 3)
    }


# Tahmin girişlerini logs.txt'ye kaydeden yardımcı fonksiyon
import os  # Ekstra: dosya kontrolü için lazım

def log_input(candidate: Candidate, endpoint: str, prediction: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] {endpoint} → "
        f"experience={candidate.experience}, "
        f"technical_point={candidate.technical_point}, "
        f"prediction={prediction}\n"
    )

    log_path = "C:/GYK/GYK1/ML/SVM_ODEV/logs.txt"

    # 🚀 Eğer dosya yoksa otomatik oluştur (gerekirse klasörü de oluşturabiliriz)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)
    
    print(log_entry.strip())  # Terminalde de göster
    print("Log dosyasının yazıldığı yer:", os.path.abspath(log_path))


