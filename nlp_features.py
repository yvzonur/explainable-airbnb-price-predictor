import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer

def add_nlp_features(df):
    """
    Applies Natural Language Processing feature engineering to the 'description' column.
    Extracts metrics like length, sentiment, specific keywords, and language.
    
    Dependencies required:
    pip install pandas textblob langdetect scikit-learn
    """
    print("Starting NLP feature engineering on 'description' column...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_nlp = df.copy()
    
    # Handle missing descriptions by filling them with an empty string
    df_nlp['description'] = df_nlp['description'].fillna('')
    
    # ---------------------------------------------------------
    # 1. Text Length Metrics
    # ---------------------------------------------------------
    print("Calculating word and character counts...")
    df_nlp['desc_word_count'] = df_nlp['description'].apply(lambda x: len(str(x).split()))
    df_nlp['desc_char_count'] = df_nlp['description'].apply(lambda x: len(str(x)))
    
    # ---------------------------------------------------------
    # 2. Automated Keyword Extraction (TF-IDF)
    # ---------------------------------------------------------
    # Removed from here to prevent Data Leakage!
    # TF-IDF will now be applied via a Scikit-Learn Pipeline during training.
        
    # ---------------------------------------------------------
    # 3. Sentiment Analysis
    # ---------------------------------------------------------
    print("Calculating sentiment scores...")
    def get_sentiment(text):
        if not text.strip():
            return 0.0 # Neutral sentiment if empty
        try:
            # Polarity ranges from -1.0 (highly negative) to 1.0 (highly positive)
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0
            
    df_nlp['desc_sentiment_score'] = df_nlp['description'].apply(get_sentiment)
    
    # ---------------------------------------------------------
    # 4. Language Detection
    # ---------------------------------------------------------
    print("Detecting if description is in English...")
    def detect_english(text):
        if len(str(text)) < 15: # Text too short to reliably detect language
            return 0
        try:
            return 1 if detect(str(text)) == 'en' else 0
        except LangDetectException:
            return 0
            
    df_nlp['desc_is_english'] = df_nlp['description'].apply(detect_english)
    
    print("NLP feature engineering complete!")
    return df_nlp

if __name__ == "__main__":
    print("Loading listings.csv...")
    # low_memory=False to prevent DtypeWarnings for mixed types in some columns
    df = pd.read_csv('listings.csv', low_memory=False)
    
    print(f"Original shape: {df.shape}")
    df_enhanced = add_nlp_features(df)
    
    output_file = 'listings_with_nlp.csv'
    print(f"Saving to {output_file}...")
    df_enhanced.to_csv(output_file, index=False)
    print(f"New shape: {df_enhanced.shape}")
    
    print("Pipeline optimization: TF-IDF keywords will be calculated during training.")
