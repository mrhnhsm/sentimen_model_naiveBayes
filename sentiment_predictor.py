import joblib
import re
import gzip

class SentimentPredictor:
    def __init__(self, model_path, vectorizer_path):
        """
        Inisialisasi predictor dengan model dan vectorizer terkompresi
        
        Args:
            model_path: Path ke model terkompresi
            vectorizer_path: Path ke vectorizer terkompresi
        """
        # Baca model dengan gzip
        with gzip.open(model_path, 'rb') as f:
            self.model = joblib.load(f)
        
        # Baca vectorizer dengan gzip
        with gzip.open(vectorizer_path, 'rb') as f:
            self.vectorizer = joblib.load(f)
    
    def clean_text(self, text):
        """Membersihkan teks input"""
        text = re.sub(r'@\w+', '', str(text))
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower().strip()
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
            'text': text,
            'sentiment': 'Positif' if prediction == 1 else 'Negatif',
            'confidence': f"{max(probabilities) * 100:.2f}%"
        }

# Contoh penggunaan
if __name__ == "__main__":
    try:
        predictor = SentimentPredictor(
            model_path='model_export/sentiment_model.joblib.gz',
            vectorizer_path='model_export/tfidf_vectorizer.joblib.gz'
        )
        
        # Uji coba prediksi
        test_texts = [
            "Produk ini sangat bagus dan recommended",
            "Layanan yang buruk dan tidak memuaskan",
            "Biasa saja, tidak ada yang istimewa"
        ]
        
        for text in test_texts:
            result = predictor.predict(text)
            print("\nHasil Analisis:")
            print(f"Teks: {result['text']}")
            print(f"Sentimen: {result['sentiment']}")
            print(f"Kepercayaan: {result['confidence']}")
    
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")