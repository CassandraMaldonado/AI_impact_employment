# Data cleaning and preprocessing
# Set random seed for reproducibility
import numpy as np
np.random.seed(42)

import os
import re
import pickle
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

print("Starting data preprocessing...")

# Create cache directory if it doesn't exist
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

def get_cache_path(filename):
    """Get full path for a cache file"""
    return os.path.join(cache_dir, filename)

def save_to_cache(obj, filename):
    """Save object to cache"""
    with open(get_cache_path(filename), 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved {filename} to cache")

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

print("Loading dataset...")
try:
    df = pd.read_parquet('https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet', 
                        engine='pyarrow')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Trying alternative approach...")
    try:
        # If direct URL doesn't work, try with requests
        import requests
        import io
        
        url = 'https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet'
        response = requests.get(url)
        
        if response.status_code == 200:
            df = pd.read_parquet(io.BytesIO(response.content), engine='pyarrow')
            print(f"Dataset loaded via requests. Shape: {df.shape}")
        else:
            print(f"Failed to download file: HTTP {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"Alternative approach also failed: {e}")
        print("Creating a small sample dataset for testing...")
        # Create a small sample dataset for testing
        df = pd.DataFrame({
            'title': ['AI is transforming jobs', 'Machine learning and employment', 'AI impact on workforce'],
            'text': [
                'Artificial intelligence is having a significant impact on jobs across many industries.',
                'Machine learning technologies are changing how companies think about their workforce.',
                'The artificial intelligence revolution is transforming the job market in multiple sectors.'
            ],
            'date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'url': ['https://example.com/1', 'https://example.com/2', 'https://example.com/3']
        })
        print(f"Created sample dataset with {len(df)} rows for testing")

print("Cleaning and processing text...")
# Clean text
df['cleaned_text'] = df['text'].apply(clean_article)

# Parsing dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
print(f"After removing rows with invalid dates: {len(df)} rows")

# Time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['yearmonth'] = df['date'].dt.strftime('%Y-%m')

# Filter for relevance
print("Filtering for relevance...")
df['is_relevant'] = df['cleaned_text'].apply(is_relevant)
df_relevant = df[df['is_relevant']].copy()
print(f"After filtering for relevance: {len(df_relevant)} rows")

# Extract domain
df_relevant['source_domain'] = df_relevant['url'].apply(extract_domain)

# Save multiple versions to cache
print("Saving data to cache...")

# Full version
save_to_cache(df_relevant, "cleaned_data.pkl")

# Version with all columns for LDA
save_to_cache(df_relevant, "cleaned_data_for_lda.pkl")

# Minimal version with just necessary columns
df_minimal = df_relevant[['cleaned_text', 'date', 'year', 'month', 'yearmonth']].copy()
save_to_cache(df_minimal, "cleaned_data_minimal.pkl")

print("Data preprocessing complete!")
print(f"Processed {len(df)} articles, with {len(df_relevant)} relevant articles saved to cache")
print("You can now run LDA.py for topic modeling")