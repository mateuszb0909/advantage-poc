import pandas as pd
import nltk
from collections import defaultdict

# --- NLTK Resource Downloader ---
# This section ensures that all required NLTK data models are downloaded
# before any functions that depend on them are called.

# Download the 'punkt' tokenizer models from NLTK if not already present.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' model...")
    nltk.download('punkt')

# Download the 'punkt_tab' resource, which is also required by the tokenizer.
# The traceback indicated this resource was missing.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt_tab' resource...")
    nltk.download('punkt_tab')


def generate_ngrams(text, n):
    """
    Generates n-grams from a given text string.

    Args:
        text (str): The text to process.
        n (int): The length of the n-grams (e.g., 1 for unigram, 2 for bigram).

    Returns:
        list: A list of n-gram strings.
    """
    # Tokenize the text (split it into words) and convert to lower case
    # Added str() for safety to handle potential non-string inputs.
    tokens = [word.lower() for word in nltk.word_tokenize(str(text))] 
    
    # Use NLTK's ngrams function to create the word combinations
    return [" ".join(gram) for gram in nltk.ngrams(tokens, n)]

def analyze_ngrams(search_terms_df, min_ngram=1, max_ngram=3):
    
    """
    Performs n-gram analysis on a DataFrame of search terms.

    Args:
        search_terms_df (pandas.DataFrame): DataFrame containing search term data.
        min_ngram (int): The minimum n-gram length to analyze.
        max_ngram (int): The maximum n-gram length to analyze.

    Returns:
        dict: A dictionary of DataFrames, where each key is an n-gram length
              (e.g., '1-grams') and the value is the analysis DataFrame.
    """
    # This dictionary will store the aggregated data for each n-gram
    ngram_performance = defaultdict(lambda: defaultdict(float))
    
    # A dictionary to store the n-gram length for each phrase
    ngram_lengths = {}

    # Define the metrics we want to aggregate
    metrics = ['Impressions', 'Clicks', 'Cost', 'Conversions', 'Conv. value']

    # Iterate over each row in the search terms DataFrame
    for index, row in search_terms_df.iterrows():
        search_term = row['Search term']
        
        # Generate n-grams for the current search term for all desired lengths
        for n in range(min_ngram, max_ngram + 1):
            ngrams = generate_ngrams(search_term, n)
            
            # For each n-gram found, add the performance metrics from the row
            for ngram in ngrams:
                ngram_lengths[ngram] = n
                for metric in metrics:
                    ngram_performance[ngram][metric] += row[metric]

    # Convert the aggregated data into a pandas DataFrame
    if not ngram_performance:
        print("No n-grams were generated. Check the input data.")
        return {}
        
    performance_df = pd.DataFrame.from_dict(ngram_performance, orient='index')
    performance_df.index.name = 'N-Gram'
    performance_df.reset_index(inplace=True)
    
    # Add the n-gram length as a column
    performance_df['N-Gram Length'] = performance_df['N-Gram'].map(ngram_lengths)

    # --- Calculate Derived Metrics ---
    # To avoid division by zero, we replace 0s with 1 in denominators.
    # This is a safe way to prevent errors without affecting the result
    # (e.g., x / 1 is x, and if the numerator is 0, the result is still 0).
    
    # Click-Through Rate (CTR)
    performance_df['CTR'] = (performance_df['Clicks'] / performance_df['Impressions'].replace(0, 1))
    
    # Conversion Rate
    performance_df['Conversion Rate'] = (performance_df['Conversions'] / performance_df['Clicks'].replace(0, 1))
    
    # Cost-Per-Acquisition (CPA)
    performance_df['CPA'] = (performance_df['Cost'] / performance_df['Conversions'].replace(0, 1))
    
    # Return on Ad Spend (ROAS)
    performance_df['ROAS'] = (performance_df['Conv. value'] / performance_df['Cost'].replace(0, 1))

    # Split the results into separate DataFrames for each n-gram length
    analyzed_data = {}
    for n in range(min_ngram, max_ngram + 1):
        key = f'{n}-grams'
        df = performance_df[performance_df['N-Gram Length'] == n].copy()
        
        # Sort by cost to see the biggest spenders first
        df.sort_values(by='Cost', ascending=False, inplace=True)
        analyzed_data[key] = df

    return analyzed_data


