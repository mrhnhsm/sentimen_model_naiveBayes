import pandas as pd
import csv
import os
import re
import joblib  # Tambahkan import joblib
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

# Fungsi untuk mengekspor model
def export_model(model, vectorizer, output_dir='model_export'):
    """
    Fungsi untuk mengekspor model dan vectorizer
    
    Args:
        model: Model Naive Bayes yang telah dilatih
        vectorizer: TF-IDF Vectorizer
        output_dir: Direktori untuk menyimpan model
    """
    # Pastikan direktori export ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan model
    joblib.dump(model, os.path.join(output_dir, 'sentiment_model.joblib'))
    
    # Simpan vectorizer
    joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
    
    print(f"Model dan vectorizer berhasil disimpan di {output_dir}")

# Read the CSV data
file_path = r"D:\Dhea-Sayang\DataTraining.csv" #Path Menuju Data Training

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
    # Clean 'full_text' column
    df['cleaned_text'] = df['full_text'].apply(clean_text)
else:
    # Print kolom yang tersedia untuk debugging
    print("Kolom yang tersedia:", list(df.columns))
    print("Kolom 'full_text' tidak ditemukan!")
    raise KeyError("Kolom 'full_text' tidak ditemukan dalam DataFrame")

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
        return -1  # Neutral (you might want to handle this case differently)

# Create sentiment labels with the modified function
labels = df['cleaned_text'].apply(assign_sentiment)

# Remove neutral sentiments if you want to focus only on positive and negative
df = df[labels != -1]
labels = labels[labels != -1]

# Preprocessing and vectorizing data
# Vectorization and model training
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
    stratify=y  # This ensures balanced split
)

# Model training and evaluation
model = MultinomialNB(class_prior=[0.5, 0.5])  # Give equal prior probabilities
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Menampilkan Hasil Training
print("\nModel Performance:")
print("----------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Training Classification Report:")
print(classification_report(y_test, y_pred))

# Tambahkan pemanggilan export_model di akhir script
export_model(model, vectorizer)

#Uji Coba Untuk Testing Data
def read_test_data(test_file_path):
    """
    Reads and processes test data CSV with robust error handling and column checking.
    
    Args:
        test_file_path (str): Path to the test CSV file
    
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    # Check if file exists
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {test_file_path}")

    print("\nProcessing Testing Data:")
    print("----------------------")

    try:
        # First attempt to read the CSV
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
        
        # Clean column names
        test_df.columns = [col.strip().strip('"').strip("'") for col in test_df.columns]
        
    except Exception as e:
        print(f"Error membaca file dengan metode pertama: {str(e)}")
        print("Mencoba metode alternatif...")
        
        try:
            # Alternative reading method
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
            print("Berhasil membaca dengan metode alternatif!")
            
        except Exception as e2:
            print(f"Error pada metode alternatif: {str(e2)}")
            raise

    # Print available columns for debugging
    # print("\nKolom yang tersedia:", list(test_df.columns))

    # Look for full_text column with case-insensitive matching
    full_text_col = None
    for col in test_df.columns:
        if 'full_text' in col.lower():
            full_text_col = col
            test_df = test_df.rename(columns={col: 'full_text'})
            # print(f"Menemukan kolom full_text dengan nama asli: {col}")
            break

    if full_text_col is None:
        print("\nPeringatan: Kolom 'full_text' tidak ditemukan!")
        print("Mohon periksa nama kolom yang tersedia di atas.")
        raise KeyError("Kolom 'full_text' tidak ditemukan dalam DataFrame")

    print("\nBerhasil memproses file CSV:")
    print(f"- Jumlah baris: {len(test_df)}")
    print(f"- Jumlah kolom: {len(test_df.columns)}")
    
    return test_df

# Usage example:
try:
    test_file_path = r"D:\Dhea-Sayang\DataTesting.csv" #Path Menuju File Data Testing
    test_df = read_test_data(test_file_path)
    
    # Continue with your sentiment analysis
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
    
    # Menampilkan Contoh Data Serta Klasifikasinya
    # print("\nSample of Testing Results:")
    # for i, (text, sentiment) in enumerate(zip(test_df['full_text'][:5], test_predictions[:5])):
    #     print(f"\nText {i+1}: {text}")
    #     print(f"Predicted Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")

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
        
        # Clean and process user input
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = max(probabilities) * 100
        
        # Display results
        print("\nAnalysis Results:")
        print(f"Text: {user_input}")
        print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative' if prediction == 0 else 'Netral'}")
        print(f"Confidence: {confidence:.2f}%")

# Run interactive analysis
if __name__ == "__main__":
    print("\nStarting interactive analysis...")
    analyze_user_input()