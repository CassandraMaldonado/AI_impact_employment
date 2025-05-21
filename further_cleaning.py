import os
import re
import pickle
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import html

print("Starting further cleaning of cached dataset...")

# Define paths
cache_dir = "cache"
input_file = os.path.join(cache_dir, "cleaned_data.pkl")
output_file = os.path.join(cache_dir, "trafilatura_quality_data.pkl")


def further_clean_text(text):
    """
    Further clean text to remove navigation elements, dates, and non-article content
    to more closely match Trafilatura quality
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string just in case
    text = str(text)
    
    # Remove common navigation/menu items and non-English content patterns
    
    # Remove navigation-like patterns
    text = re.sub(r'(?i)(Home|Menu|Navigation|Search|Login|Sign up|Subscribe|Follow us)\s*[|\-•]', '', text)
    
    # Remove common news site section markers
    text = re.sub(r'(?i)(News|Sports|Technology|Business|Entertainment|Politics|Opinion|Weather|AIXov XwmSpace|technology|Satellite|Science|US|Tiv tauj|Xov Xwm)\s*[|\-•]', '', text)
    
    # Remove dates in various formats
    text = re.sub(r'(?i)(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', '', text)
    text = re.sub(r'(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', '', text)
    text = re.sub(r'(?i)(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?\s+\d{1,2},?\s+\d{4}', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    
    # Remove bylines and publishing info
    text = re.sub(r'(?i)By\s+[A-Za-z\s\.]+\s*[,|\-]\s*[A-Za-z\s\.]+', '', text)
    text = re.sub(r'(?i)By\s+[A-Za-z\s\.]+\s*$', '', text)
    text = re.sub(r'(?i)Published\s+\d{1,2}[a-z]{0,2}\s+[A-Za-z]+\s+\d{4}', '', text)
    
    # Remove time specifications
    text = re.sub(r'(?i)\d{1,2}:\d{2}\s*[ap]\.?m\.?(\s+[A-Za-z]+)?', '', text)
    
    # Remove non-English text patterns (like the Hmong example)
    text = re.sub(r'(?i)Hla mus rau cov ntsiab lus', '', text)
    text = re.sub(r'(?i)Lub neej hauv nroog', '', text)
    
    # Remove share/social media buttons text
    text = re.sub(r'(?i)(Share|Tweet|Email|Print|Facebook|Twitter|LinkedIn|Pinterest|Instagram|WhatsApp)\s*[|\-•]', '', text)
    
    # Remove copyright notices
    text = re.sub(r'(?i)©\s*\d{4}.*?(rights reserved|all rights)', '', text)
    text = re.sub(r'(?i)Copyright\s*©?\s*\d{4}.*?$', '', text)
    
    # Remove comment section indicators
    text = re.sub(r'(?i)(Comments|Leave a comment|Add a comment|Join the conversation)', '', text)
    
    # Clean up excessive newlines and spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with just 2
    text = re.sub(r' {2,}', ' ', text)      # Replace 2+ spaces with just 1
    
    # Remove lines that are too short (likely navigation/menu items)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 15 or line.strip() == '']
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup of leading/trailing whitespace
    text = text.strip()
    
    return text


def further_clean_title(title):
    """
    Further clean title to remove publisher names and other noise
    """
    if not title or pd.isna(title):
        return ""
    
    # Convert to string just in case
    title = str(title)
    
    # Remove common publisher indicators
    patterns = [
        r'\s*\|\s*[^|]+$',                    # | Business News This Week
        r'\s*–\s*[^–]+$',                     # – MeriTalk
        r'\s*-\s*[^-]+\.(com|org|net)$',      # - allAfrica.com
        r'\s*-\s*[A-Za-z\s]+Times$',          # - Asia Times
        r'\s*\(\s*[^)]+\s*\)$',               # (Publisher Name)
        r'\s*\|\s*[A-Za-z0-9\s\.]+$',         # | Website Name
        r'\s*-\s*[A-Za-z0-9\s\.]+$',          # - Website Name
        r'\s*—\s*[A-Za-z0-9\s\.]+$',          # — Website Name
        r'\s*:\s*[A-Za-z0-9\s\.]+\.(com|org|net|edu)$', # : website.com
        r'\s*[·•]\s*[A-Za-z0-9\s\.]+$',       # · Website Name
    ]
    
    for pattern in patterns:
        title = re.sub(pattern, '', title)
    
    # Remove quotation marks that might remain from previous cleaning
    title = re.sub(r'^"(.*)"$', r'\1', title)
    title = re.sub(r'^\'(.*)\'$', r'\1', title)
    
    # Clean up any remaining special characters and whitespace
    title = title.strip()
    
    return title


def process_cleaned_dataset():
    """
    Load the previously cleaned dataset and apply further cleaning
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Make sure you've run your original cleaning script first.")
        return None
    
    # Load the dataset
    print(f"Loading cached dataset from {input_file}...")
    try:
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Make a copy to preserve the original
    df_clean = df.copy()
    
    # Apply further cleaning to text
    print("Applying further text cleaning...")
    if 'cleaned_text' in df_clean.columns:
        df_clean['trafilatura_text'] = df_clean['cleaned_text'].apply(further_clean_text)
    else:
        print("Warning: 'cleaned_text' column not found. Looking for 'text' column instead.")
        if 'text' in df_clean.columns:
            df_clean['trafilatura_text'] = df_clean['text'].apply(further_clean_text)
        else:
            print("Error: Neither 'cleaned_text' nor 'text' column found.")
            return None
    
    # Apply further cleaning to title
    print("Applying further title cleaning...")
    if 'title' in df_clean.columns:
        df_clean['trafilatura_title'] = df_clean['title'].apply(further_clean_title)
    else:
        print("Warning: 'title' column not found.")
        df_clean['trafilatura_title'] = ["Unknown Title" for _ in range(len(df_clean))]
    
    # Save the further cleaned dataset
    print(f"Saving further cleaned dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(df_clean, f)
    print(f"Dataset saved successfully.")
    
    # Also save a minimal version with just the essential columns
    minimal_cols = ['trafilatura_title', 'trafilatura_text']
    if 'date' in df_clean.columns:
        minimal_cols.append('date')
    if 'year' in df_clean.columns:
        minimal_cols.append('year')
    if 'month' in df_clean.columns:
        minimal_cols.append('month')
    if 'yearmonth' in df_clean.columns:
        minimal_cols.append('yearmonth')
    if 'source_domain' in df_clean.columns:
        minimal_cols.append('source_domain')
    
    minimal_file = os.path.join(cache_dir, "trafilatura_quality_minimal.pkl")
    df_minimal = df_clean[minimal_cols].copy()
    
    with open(minimal_file, 'wb') as f:
        pickle.dump(df_minimal, f)
    print(f"Minimal dataset saved to {minimal_file}")
    
    # Return the first few rows for inspection
    return df_clean[['trafilatura_title', 'trafilatura_text']].head()


def analyze_cleaning_differences(df_sample=5):
    """
    Analyze and show the differences between original and further cleaned text
    """
    # Load the dataset
    try:
        with open(input_file, 'rb') as f:
            df = pickle.load(f)
        
        with open(output_file, 'rb') as f:
            df_clean = pickle.load(f)
        
        # Select columns for comparison
        title_cols = ['title', 'trafilatura_title'] if 'title' in df.columns else ['trafilatura_title']
        text_cols = ['cleaned_text', 'trafilatura_text'] if 'cleaned_text' in df.columns else ['trafilatura_text']
        
        # Get random sample
        sample_idx = df.sample(df_sample).index
        
        print("\n" + "="*80)
        print("TITLE CLEANING COMPARISON")
        print("="*80)
        
        for idx in sample_idx:
            if len(title_cols) > 1:
                print(f"\nOriginal title: {df.loc[idx, title_cols[0]]}")
                print(f"Cleaned title:  {df_clean.loc[idx, title_cols[1]]}")
            else:
                print(f"\nCleaned title:  {df_clean.loc[idx, title_cols[0]]}")
            print("-"*80)
        
        print("\n" + "="*80)
        print("TEXT CLEANING COMPARISON")
        print("="*80)
        
        for idx in sample_idx:
            if len(text_cols) > 1:
                orig_text = df.loc[idx, text_cols[0]]
                clean_text = df_clean.loc[idx, text_cols[1]]
                
                print(f"\nOriginal text (first 200 chars):")
                print(f"{orig_text[:200]}..." if len(orig_text) > 200 else orig_text)
                
                print(f"\nCleaned text (first 200 chars):")
                print(f"{clean_text[:200]}..." if len(clean_text) > 200 else clean_text)
                
                # Calculate character reduction percentage
                orig_len = len(orig_text)
                clean_len = len(clean_text)
                reduction = ((orig_len - clean_len) / orig_len) * 100 if orig_len > 0 else 0
                
                print(f"\nCharacter count: {orig_len} → {clean_len} ({reduction:.1f}% reduction)")
            else:
                clean_text = df_clean.loc[idx, text_cols[0]]
                print(f"\nCleaned text (first 200 chars):")
                print(f"{clean_text[:200]}..." if len(clean_text) > 200 else clean_text)
            
            print("-"*80)
        
        return True
    except Exception as e:
        print(f"Error analyzing cleaning differences: {e}")
        return False


if __name__ == "__main__":
    # Process the dataset
    result = process_cleaned_dataset()
    
    if result is not None:
        print("\nSample of further cleaned data:")
        print(result)
        
        # Analyze differences
        print("\nAnalyzing cleaning differences...")
        analyze_cleaning_differences()
        
        print("\nFurther cleaning complete! Your data is now closer to Trafilatura quality.")
        print(f"The cleaned data is saved at: {output_file}")
        print(f"A minimal version is saved at: {os.path.join(cache_dir, 'trafilatura_quality_minimal.pkl')}")
