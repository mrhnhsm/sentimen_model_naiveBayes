import pandas as pd
import csv
import os
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghapus mention (contoh: @user)
    text = re.sub(r'@\w+', '', text)
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Menghapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    return text

# Read the CSV data
file_path = r"D:\Dhea-Sayang\sentimen-model\DataTraining.csv"  # Path Menuju Data Training

# Tambahkan pengecekan file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")

try:
    # Membaca file dengan parameter yang lebih spesifik
    df = pd.read_csv(
       file_path, 
       encoding='utf-8', 
       sep=';',  # Delimiter yang jelas
       quoting=csv.QUOTE_ALL,  # Menangani quote yang kompleks
       quotechar='"',  # Karakter quote
       escapechar='\\',  # Karakter escape
       doublequote=True,  # Mengizinkan double quote
       engine='python',  # Parser Python untuk fleksibilitas
       on_bad_lines='skip'  # Skip baris bermasalah
    )
    # Membersihkan nama kolom
    df.columns = [col.strip().strip('"').strip("'") for col in df.columns]

except Exception as e:
    print(f"Error membaca file: {str(e)}")
    # Mencoba alternatif jika metode pertama gagal
    try:
        print("Mencoba metode alternatif...")
        # Membaca dengan delimiter yang tepat
        df = pd.read_csv(
            file_path, 
            sep=',',  # Delimiter yang jelas
            encoding='utf-8', 
            quotechar='"',  # Karakter quote
            escapechar='\\',  # Karakter escape
            doublequote=True,  # Mengizinkan double quote
            skipinitialspace=True,  # Mengabaikan spasi di awal
            engine='python',  # Parser Python
            on_bad_lines='skip'  # Skip baris bermasalah
        )
        df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
        print("Berhasil membaca dengan metode alternatif!")
    except Exception as e2:
        print(f"Error pada metode alternatif: {str(e2)}")
        raise

# Ganti nama kolom yang mengandung 'full_text'
for col in df.columns:
    if 'full_text' in col.lower():
        df = df.rename(columns={col: 'full_text'})
        break

# Pastikan kolom 'full_text' ada
if 'full_text' in df.columns:
    df['cleaned_text'] = df['full_text'].apply(clean_text)
else:
    print("Kolom 'full_text' tidak ditemukan!")
    raise KeyError("Kolom 'full_text' tidak ditemukan dalam DataFrame")

def assign_sentiment(text):
    text = str(text).lower()
    positive_words = ['bagus', 'pintar', 'berguna', 'cerdas', 'tepat', 'keren', 'mantul', 'gokil']
    negative_words = ['bodoh', 'gagal', 'jelek', 'lebay', 'pecundang', 'tidak berguna', 'tolol']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if negative_count > 0:
        return 0  # Negative sentiment
    elif positive_count > 0:
        return 1  # Positive sentiment
    else:
        return -1  # Neutral

# Create sentiment labels
labels = df['cleaned_text'].apply(assign_sentiment)

# Remove neutral sentiments
df = df[labels != -1]
labels = labels[labels != -1]

# Preprocessing and vectorizing data
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    strip_accents='unicode',
    lowercase=True
)

# Menggunakan teks yang sudah dibersihkan
X = vectorizer.fit_transform(df['cleaned_text'])
y = labels

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Model training and evaluation
model = MultinomialNB(class_prior=[0.5, 0.5])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Menampilkan Hasil Training
print("\nModel Performance:")
print("----------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Training Classification Report:")
print(classification_report(y_test, y_pred))

# Fungsi untuk mengekspor model dan vectorizer
def export_model_and_vectorizer(model, vectorizer, output_dir='model_export'):
    """
    Menyimpan model dan vectorizer ke dalam file menggunakan joblib.
    
    Args:
        model: Model sentiment analysis (contoh: MultinomialNB)
        vectorizer: TfidfVectorizer yang digunakan untuk pemrosesan teks
        output_dir: Direktori untuk menyimpan file model dan vectorizer (default: 'model_export')
    """
    # Memastikan direktori output ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Menyimpan model ke dalam file
    model_filename = os.path.join(output_dir, 'sentiment_model.joblib')
    joblib.dump(model, model_filename)
    print(f"Model berhasil disimpan di: {model_filename}")
    
    # Menyimpan vectorizer ke dalam file
    vectorizer_filename = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_filename)
    print(f"Vectorizer berhasil disimpan di: {vectorizer_filename}")

# Ekspor model dan vectorizer
export_model_and_vectorizer(model, vectorizer, output_dir='model_export')

# Uji Coba Untuk Testing Data
def read_test_data(test_file_path):
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {test_file_path}")

    try:
        test_df = pd.read_csv(
            test_file_path,
            sep=',',
            header=0,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            doublequote=True,
            engine='python',
            on_bad_lines='skip'
        )
        test_df.columns = [col.strip().strip('"').strip("'") for col in test_df.columns]
    except Exception as e:
        print(f"Error membaca file dengan metode pertama: {str(e)}")
        print("Mencoba metode alternatif...")
        try:
            test_df = pd.read_csv(
                test_file_path,
                encoding='utf-8',
                sep=',',
                quotechar='"',
                escapechar='\\',
                engine='python',
                on_bad_lines='skip'
            )
            test_df.columns = [col.strip().strip('"').strip("'") for col in test_df.columns]
        except Exception as e2:
            print(f"Error pada metode alternatif: {str(e2)}")
            raise

    full_text_col = None
    for col in test_df.columns:
        if 'full_text' in col.lower():
            full_text_col = col
            test_df = test_df.rename(columns={col: 'full_text'})
            break

    if full_text_col is None:
        raise KeyError("Kolom 'full_text' tidak ditemukan dalam DataFrame")

    return test_df

# Usage example:
try:
    test_file_path = r"D:\Dhea-Sayang\sentimen-model\DataTesting.csv"  # Path Menuju File Data Testing
    test_df = read_test_data(test_file_path)
    
    # Process testing data
    test_df['cleaned_text'] = test_df['full_text'].apply(clean_text)
    X_new_test = vectorizer.transform(test_df['cleaned_text'])
    test_predictions = model.predict(X_new_test)
    
    # Count sentiments
    sentiment_counts = {
        'Positive': sum(test_predictions == 1),
        'Negative': sum(test_predictions == 0),
        'Netral': sum(test_predictions == -1),
        'Total': len(test_predictions)
    }
    
    print("\nTesting Data Analysis Results:")
    print(f"Total texts analyzed: {sentiment_counts['Total']}")
    print(f"Positive sentiments: {sentiment_counts['Positive']} ({(sentiment_counts['Positive']/sentiment_counts['Total']*100):.2f}%)")
    print(f"Negative sentiments: {sentiment_counts['Negative']} ({(sentiment_counts['Negative']/sentiment_counts['Total']*100):.2f}%)")
    print(f"Netral sentiments: {sentiment_counts['Netral']} ({(sentiment_counts['Netral']/sentiment_counts['Total']*100):.2f}%)")
    
    # Add predictions to the testing DataFrame
    test_df['predicted_sentiment'] = test_predictions

except Exception as e:
    print(f"Error processing testing data: {str(e)}")

# Interactive input analysis
def analyze_user_input():
    print("\nInteractive Sentiment Analysis")
    print("Type 'exit' to quit")
    print("-" * 30)
    
    while True:
        user_input = input("\nEnter text to analyze: ")
        
        if user_input.lower() == 'exit':
            print("Exiting interactive analysis...")
            break
        
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = max(probabilities) * 100
        
        print("\nAnalysis Results:")
        print(f"Text: {user_input}")
        print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative' if prediction == 0 else 'Netral'}")
        print(f"Confidence: {confidence:.2f}%")

# Run interactive analysis
if __name__ == "__main__":
    print("\nStarting interactive analysis...")
    analyze_user_input()
