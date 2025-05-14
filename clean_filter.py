# Set random seed for reproducibility
import numpy as np
np.random.seed(42)

import os
import re
import pickle
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import nltk

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Create cache directory if it doesn't exist
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def get_cache_path(filename):
    """Get full path for a cache file"""
    return os.path.join(cache_dir, filename)

def save_to_cache(obj, filename):
    """Save object to cache"""
    with open(get_cache_path(filename), 'wb') as f:
        pickle.dump(obj, f)

def load_from_cache(filename):
    """Load object from cache if it exists"""
    cache_path = get_cache_path(filename)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def clean_article(text):
    """Clean article text by removing HTML, extra whitespace, etc."""
    # Handling none or empty strings
    if not text or pd.isna(text):
        return ""

    # Removing the HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Removing the URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Removing the extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Removing the special characters
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)

    return text

def extract_domain(url):
    """Extract domain from URL"""
    if not url or pd.isna(url):
        return ""

    try:
        # Domain using regex
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            return domain_match.group(1)
    except:
        pass

    return ""

def is_relevant(text):
    """Check if article is relevant to AI's impact on industries/jobs"""
    if not text or pd.isna(text):
        return False

    text_lower = text.lower()

    # Checking for AI related terms
    ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'llm', 'large language model', 'chatgpt', 'generative ai']

    # Checking for industry impact terms
    impact_terms = ['impact', 'effect', 'transform', 'disrupt', 'replace', 'automate',
                    'job', 'employment', 'workforce', 'career', 'industry', 'sector',
                    'profession', 'work', 'labor market', 'skill']

    contains_ai = any(term in text_lower for term in ai_terms)
    contains_impact = any(term in text_lower for term in impact_terms)

    if not (contains_ai and contains_impact):
        return False

    # Proximity within same paragraph for better accuracy
    paragraphs = text_lower.split('\n')

    for para in paragraphs:
        para_has_ai = any(term in para for term in ai_terms)
        para_has_impact = any(term in para for term in impact_terms)

        if para_has_ai and para_has_impact:
            return True

    # Fallback for short texts without paragraphs, checking the sentence proximity
    sentences = text_lower.split('.')

    ai_sentences = [i for i, sent in enumerate(sentences) if any(term in sent for term in ai_terms)]
    impact_sentences = [i for i, sent in enumerate(sentences) if any(term in sent for term in impact_terms)]

    # AI and impact sentences are close to each other within 3 sentences
    for ai_idx in ai_sentences:
        for impact_idx in impact_sentences:
            if abs(ai_idx - impact_idx) <= 3:
                return True

    return False

def clean_and_filter_data(df, force_recompute=False):
    """
    Main function to clean and filter the dataset
    
    Args:
        df: DataFrame with raw data
        force_recompute: Whether to force recomputation even if cached results exist
        
    Returns:
        DataFrame with cleaned and filtered data
    """
    cache_file = "cleaned_data.pkl"

    if not force_recompute:
        df_clean = load_from_cache(cache_file)
        if df_clean is not None:
            print("Loaded cleaned data from cache")
            return df_clean

    print("Cleaning and filtering data...")

    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_article)

    # Parsing dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['yearmonth'] = df['date'].dt.strftime('%Y-%m')

    # Relevance
    df['is_relevant'] = df['cleaned_text'].apply(is_relevant)
    df_relevant = df[df['is_relevant']].copy()

    # Extract for source analysis
    df_relevant['source_domain'] = df_relevant['url'].apply(extract_domain)

    save_to_cache(df_relevant, cache_file)

    print(f"Filtered to {len(df_relevant)} relevant articles")
    return df_relevant

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_parquet('https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet', 
                         engine='pyarrow')
    print(f"Dataset shape: {df.shape}")
    df.info()
    
    # Clean and filter data - saves to cache
    df_clean = clean_and_filter_data(df, force_recompute=True)
    
    # Save the full DataFrame for topic modeling
    print(f"Saving all {len(df_clean)} articles to cache for topic modeling")
    save_to_cache(df_clean, "cleaned_data_for_lda.pkl")
    
    # Also save a version with just the necessary columns to reduce file size
    df_minimal = df_clean[['cleaned_text', 'date', 'year', 'month', 'yearmonth']].copy()
    save_to_cache(df_minimal, "cleaned_data_minimal.pkl")
    
    print("Data cleaning and sampling complete!")