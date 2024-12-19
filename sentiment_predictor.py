import joblib
import re

class SentimentPredictor:
    def __init__(self, model_path, vectorizer_path):
        """
        Inisialisasi predictor dengan model dan vectorizer
        
        Args:
            model_path: Path ke model tersimpan
            vectorizer_path: Path ke vectorizer tersimpan
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
    
    def clean_text(self, text):
        """Membersihkan teks input"""
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
    
    def predict(self, text):
        """
        Melakukan prediksi sentiment
        
        Args:
            text: Teks yang akan dianalisis
        
        Returns:
            Dict berisi prediksi dan probabilitas
        """
        cleaned_text = self.clean_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)[0]
        probabilities = self.model.predict_proba(vectorized_text)[0]
        
        return {
            'sentiment': 'Positif' if prediction == 1 else 'Negatif',
            'confidence': max(probabilities) * 100
        }

# Contoh penggunaan
if __name__ == "__main__":
    predictor = SentimentPredictor(
        model_path='/content/drive/MyDrive/Model-KCB/model_export/sentiment_model.joblib',
        vectorizer_path='/content/drive/MyDrive/Model-KCB/model_export/tfidf_vectorizer.joblib'
    )
    
    # Uji coba prediksi
    text = "Ini adalah contoh teks untuk dianalisis"
    result = predictor.predict(text)
    print(result)