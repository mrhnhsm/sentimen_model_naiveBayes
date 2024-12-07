from flask import Flask, request, jsonify
from sentiment_predictor import SentimentPredictor

app = Flask(__name__)

# Inisialisasi predictor
predictor = SentimentPredictor(
    model_path='model_export/sentiment_model.joblib.gz',
    vectorizer_path='model_export/tfidf_vectorizer.joblib.gz'
)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400
    
    result = predictor.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)