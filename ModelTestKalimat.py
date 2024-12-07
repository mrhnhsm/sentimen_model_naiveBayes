import pandas as pd
import csv
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

def read_csv_robust(file_path):
    """
    Fungsi untuk membaca CSV dengan berbagai metode
    """
    # Daftar encoding yang mungkin
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'utf-16']
    
    # Cek apakah file ada
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")
    
    # Coba berbagai metode pembacaan
    for encoding in encodings:
        try:
            # Metode pertama
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,  # Ubah dari QUOTE_ALL
                engine='python',
                on_bad_lines='skip',
                dtype=str  # Membaca semua kolom sebagai string
            )
            
            # Membersihkan nama kolom
            df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
            
            # Cari kolom 'full_text' dengan case-insensitive
            full_text_col = None
            for col in df.columns:
                if 'full_text' in col.lower():
                    full_text_col = col
                    df = df.rename(columns={col: 'full_text'})
                    break
            
            # Pastikan kolom 'full_text' ada
            if 'full_text' not in df.columns:
                print("Kolom yang tersedia:", list(df.columns))
                raise KeyError("Kolom 'full_text' tidak ditemukan dalam DataFrame")
            
            print(f"Berhasil membaca file dengan encoding: {encoding}")
            print(f"Jumlah baris: {len(df)}")
            print(f"Kolom: {df.columns.tolist()}")
            
            return df
        
        except Exception as e:
            print(f"Gagal dengan encoding {encoding}: {str(e)}")
    
    # Jika semua metode gagal
    raise ValueError("Tidak dapat membaca file CSV dengan metode apapun")

def assign_sentiment(text):
    text = str(text).lower()
    positive_words = ['bagus', 'pintar', 'berguna', 'cerdas', 'tepat', 'keren', 'mantul', 'gokil']
    negative_words = ['bodoh', 'gagal', 'jelek', 'lebay', 'pecundang', 'tidak berguna', 'tolol']
    
    # Count occurrences of positive and negative words
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    # Improved logic for sentiment assignment
    if negative_count > 0:
        return 0  # Negative sentiment
    elif positive_count > 0:
        return 1  # Positive sentiment
    else:
        return -1  # Neutral

def main():
    # Read training data
    try:
        file_path = r"D:\Dhea-Sayang\DataTraining.csv"
        df = read_csv_robust(file_path)
        
        # Clean text
        df['cleaned_text'] = df['full_text'].apply(clean_text)
        
        # Create sentiment labels
        labels = df['cleaned_text'].apply(assign_sentiment)
        
        # Remove neutral sentiments
        df = df[labels != -1]
        labels = labels[labels != -1]
        
        # Vectorization
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            strip_accents='unicode',
            lowercase=True
        )
        
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        # Train model
        model = MultinomialNB(class_prior=[0.5, 0.5])
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        print("\nModel Performance:")
        print("----------------")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nDetailed Training Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Read and predict on test data
        test_file_path = r"D:\Dhea-Sayang\DataTesting.csv"
        test_df = read_csv_robust(test_file_path)
        test_df['cleaned_text'] = test_df['full_text'].apply(clean_text)
        
        # Predict on test data
        X_new_test = vectorizer.transform(test_df['cleaned_text'])
        test_predictions = model.predict(X_new_test)
        
        # Count and display sentiments
        sentiment_counts = {
            'Positive': sum(test_predictions == 1),
            'Negative': sum(test_predictions == 0),
            'Total': len(test_predictions)
        }
        
        print("\nTesting Data Analysis Results:")
        print(f"Total texts analyzed: {sentiment_counts['Total']}")
        print(f"Positive sentiments: {sentiment_counts['Positive']} ({(sentiment_counts['Positive']/sentiment_counts['Total']*100):.2f}%)")
        print(f"Negative sentiments: {sentiment_counts['Negative']} ({(sentiment_counts['Negative']/sentiment_counts['Total']*100):.2f}%)")
        
        test_df['predicted_sentiment'] = test_predictions
        
        # Interactive analysis
        while True:
            user_input = input("\nMasukkan teks untuk dianalisis (ketik 'exit' untuk keluar): ")
            
            if user_input.lower() == 'exit':
                break
            
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            
            print("\nHasil Analisis:")
            print(f"Teks: {user_input}")
            print(f"Sentimen: {'Positif' if prediction == 1 else 'Negatif'}")
    
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()