import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(input_path, output_path):

    df = pd.read_csv(input_path, encoding='latin1')

    df = df.dropna().drop_duplicates()

    if 'Email No.' in df.columns:
        df = df.drop(columns=['Email No.'])

    df['Prediction'] = df['Prediction'].astype(int)

    text_data = df.drop(columns=['Prediction']).astype(str).agg(' '.join, axis=1)

    vectorizer = TfidfVectorizer(max_features=3000)
    X_tfidf = vectorizer.fit_transform(text_data)

    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    df_tfidf['Prediction'] = df['Prediction'].values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_tfidf.to_csv(output_path, index=False)

    size_mb = round(os.path.getsize(output_path)/1024/1024, 2)
    print(f"Berhasil disimpan ke: {output_path} | Ukuran: {size_mb} MB")

if __name__ == "__main__":
    raw_dataset_path = 'dataset/emails.csv'
    processed_dataset_path = 'preprocessing/namadataset_preprocessing/emails_preprocessed.csv'
    
    preprocess_data(raw_dataset_path, processed_dataset_path)
