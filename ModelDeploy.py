import pandas as pd
import csv
import os
import re
import gzip
import joblib
import gc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Lighter model
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghapus mention (contoh: @user)
    text = re.sub(r'@\w+', '', str(text))
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Menghapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    # Mengubah teks menjadi huruf kecil
    text = text.lower().strip()
    return text

# Fungsi untuk mengekspor model dengan kompresi
def export_compressed_model(model, vectorizer, output_dir='model_export'):
    """
    Fungsi untuk mengekspor model dan vectorizer dengan kompresi
    
    Args:
        model: Model yang telah dilatih
        vectorizer: TF-IDF Vectorizer
        output_dir: Direktori untuk menyimpan model
    """
    # Pastikan direktori export ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Compress dan simpan model dengan gzip
    with gzip.open(os.path.join(output_dir, 'sentiment_model.joblib.gz'), 'wb') as f:
        joblib.dump(model, f, compress=('gzip', 3))
    
    # Compress dan simpan vectorizer dengan gzip
    with gzip.open(os.path.join(output_dir, 'tfidf_vectorizer.joblib.gz'), 'wb') as f:
        joblib.dump(vectorizer, f, compress=('gzip', 3))
    
    print(f"Model terkompresi berhasil disimpan di {output_dir}")

# Fungsi untuk membaca data dengan robust error handling
def read_csv_robust(file_path):
    """
    Membaca file CSV dengan metode robust
    
    Args:
        file_path (str): Path ke file CSV
    
    Returns:
        pandas.DataFrame: DataFrame yang telah dibaca
    """
    try:
        # Metode pertama dengan parameter lengkap
        df = pd.read_csv(
            file_path, 
            encoding='utf-8', 
            sep=';',  
            quoting=csv.QUOTE_ALL,  
            quotechar='"',  
            escapechar='\\',  
            doublequote=True,  
            engine='python',  
            on_bad_lines='skip'
        )
        
        # Membersihkan nama kolom
        df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
        
    except Exception as e:
        print(f"Error membaca file dengan metode pertama: {str(e)}")
        print("Mencoba metode alternatif...")
        
        try:
            # Metode alternatif
            df = pd.read_csv(
                file_path, 
                sep=',',  
                encoding='utf-8', 
                quotechar='"',  
                escapechar='\\',  
                doublequote=True,  
                skipinitialspace=True,  
                engine='python',  
                on_bad_lines='skip'
            )
            df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
            
        except Exception as e2:
            print(f"Error pada metode alternatif: {str(e2)}")
            raise
    
    return df

# Fungsi untuk menentukan sentimen
def assign_sentiment(text):
    text = str(text).lower()
    positive_words = ['bagus', 'pintar', 'berguna', 'cerdas', 'tepat', 'keren', 'mantul', 'gokil']
    negative_words = ['bodoh', 'gagal', 'jelek', 'lebay', 'pecundang', 'tidak berguna', 'tolol']
    
    # Hitung kemunculan kata positif dan negatif
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Logika penentuan sentimen yang disempurnakan
    if negative_count > 0:
        return 0  # Sentimen Negatif
    elif positive_count > 0:
        return 1  # Sentimen Positif
    else:
        return -1  # Netral

def train_sentiment_model(training_file_path):
    """
    Fungsi untuk melatih model sentimen dengan optimasi
    
    Args:
        training_file_path (str): Path ke file CSV training
    
    Returns:
        tuple: Model, Vectorizer
    """
    # Baca data training
    df = read_csv_robust(training_file_path)
    
    # Identifikasi kolom 'full_text'
    full_text_col = None
    for col in df.columns:
        if 'full_text' in col.lower():
            full_text_col = col
            df = df.rename(columns={col: 'full_text'})
            break
    
    if full_text_col is None:
        raise KeyError("Kolom 'full_text' tidak ditemukan!")
    
    # Bersihkan teks
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Buat label sentimen
    labels = df['cleaned_text'].apply(assign_sentiment)
    
    # Hapus data netral
    df = df[labels != -1]
    labels = labels[labels != -1]
    
    # Vectorization dengan fitur yang dikurangi
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Kurangi jumlah fitur
        ngram_range=(1, 1),  # Hanya unigram
        strip_accents='unicode',
        lowercase=True
    )
    
    # Transformasi teks
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Model Logistik Regresi yang lebih ringan
    model = LogisticRegression(
        max_iter=1000, 
        solver='lbfgs', 
        class_weight='balanced'
    )
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    
    # Tampilkan metrik
    print("\nPerforma Model:")
    print("----------------")
    print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    print("\nLaporan Klasifikasi Terperinci:")
    print(classification_report(y_test, y_pred))
    
    # Bersihkan memori
    del X_train, X_test, y_train, y_test
    gc.collect()
    
    return model, vectorizer

def load_compressed_model(model_path, vectorizer_path):
    """
    Memuat model dan vectorizer yang terkompresi
    
    Args:
        model_path (str): Path ke model terkompresi
        vectorizer_path (str): Path ke vectorizer terkompresi
    
    Returns:
        tuple: Model, Vectorizer
    """
    with gzip.open(model_path, 'rb') as f:
        model = joblib.load(f)
    
    with gzip.open(vectorizer_path, 'rb') as f:
        vectorizer = joblib.load(f)
    
    return model, vectorizer

def analyze_sentiment(text, model, vectorizer):
    """
    Menganalisis sentimen untuk teks tunggal
    
    Args:
        text (str): Teks yang akan dianalisis
        model: Model sentimen yang telah dilatih
        vectorizer: Vectorizer yang telah dilatih
    
    Returns:
        dict: Hasil analisis sentimen
    """
    # Bersihkan teks
    cleaned_text = clean_text(text)
    
    # Vektorisasi
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Prediksi
    prediction = model.predict(vectorized_text)[0]
    
    # Probabilitas
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = max(probabilities) * 100
    
    # Mapping sentimen
    sentiment_map = {0: 'Negatif', 1: 'Positif'}
    
    return {
        'text': text,
        'sentiment': sentiment_map.get(prediction, 'Netral'),
        'confidence': f"{confidence:.2f}%"
    }

def main():
    # Path file
    training_file_path = r"D:\Dhea-Sayang\DataTraining.csv"
    testing_file_path = r"D:\Dhea-Sayang\DataTesting.csv"
    model_export_dir = 'model_export'
    
    # Latih model
    model, vectorizer = train_sentiment_model(training_file_path)
    
    # Ekspor model terkompresi
    export_compressed_model(model, vectorizer, model_export_dir)
    
    # Analisis data testing
    test_df = read_csv_robust(testing_file_path)
    test_df['cleaned_text'] = test_df['full_text'].apply(clean_text)
    
    X_test = vectorizer.transform(test_df['cleaned_text'])
    test_predictions = model.predict(X_test)
    
    # Ringkasan sentimen
    sentiment_counts = {
        'Positif': sum(test_predictions == 1),
        'Negatif': sum(test_predictions == 0),
        'Total': len(test_predictions)
    }
    
    # Tampilkan ringkasan
    print("\nAnalisis Data Testing:")
    print(f"Total teks: {sentiment_counts['Total']}")
    print(f"Positif: {sentiment_counts['Positif']} ({sentiment_counts['Positif']/sentiment_counts['Total']*100:.2f}%)")
    print(f"Negatif: {sentiment_counts['Negatif']} ({sentiment_counts['Negatif']/sentiment_counts['Total']*100:.2f}%)")

if __name__ == "__main__":
    main()

    # Interaktif analisis sentimen
    print("\nAnalisis Sentimen Interaktif")
    print("Ketik 'keluar' untuk berhenti")
    
    # Muat model terkompresi
    model, vectorizer = load_compressed_model(
        'model_export/sentiment_model.joblib.gz', 
        'model_export/tfidf_vectorizer.joblib.gz'
    )
    
    while True:
        user_input = input("\nMasukkan teks untuk dianalisis: ")
        
        if user_input.lower() == 'keluar':
            print("Keluar dari analisis...")
            break
        
        # Analisis sentimen
        result = analyze_sentiment(user_input, model, vectorizer)
        
        # Tampilkan hasil
        print("\nHasil Analisis:")
        print(f"Teks: {result['text']}")
        print(f"Sentimen: {result['sentiment']}")
        print(f"Kepercayaan: {result['confidence']}")