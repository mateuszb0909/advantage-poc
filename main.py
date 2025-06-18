import os
import pandas as pd
from google import genai

# --- 1. Import functions from other project modules ---
# Note: These files (data_loader.py, etc.) must exist in the same directory.
from app.data_loader import load_data
from app.ngram_analyzer import analyze_ngrams
from app.ad_analyzer import find_underperforming_ads, find_best_ngrams, find_mismatched_ngrams
from app.ad_generator import generate_suggestions

from app.gui import create_gui
from nicegui import ui

# Create the user interface
create_gui()

# Run the NiceGUI app
ui.run(title="Advantage+ Ad Optimizer", port=8080)

def main():
    """
    Main function to run the AdVantage POC workflow.
    """
    print("\n--- 1. Starting AdVantage POC Analysis ---")

    # --- Initialize API Client Once ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n--- Analysis halted: GEMINI_API_KEY not found. ---")
        return
    try:
        gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"\n--- Analysis halted: Could not initialize Gemini client: {e} ---")
        return

    # --- 2. Load Data ---
    # Define the paths to your data files
    SEARCH_TERMS_FILE = 'data/search_terms.csv'
    ADS_FILE = 'data/ads.csv'

    search_terms_df = load_data(SEARCH_TERMS_FILE)
    ads_df = load_data(ADS_FILE)

    # Exit if data loading failed
    if search_terms_df is None or ads_df is None:
        print("\n--- Analysis halted due to data loading errors. ---")
        return

    # --- 3. Perform N-Gram Analysis ---
    print("\n--- 2. Running N-Gram Analysis on Search Terms ---")
    ngram_analysis = analyze_ngrams(search_terms_df, min_ngram=2, max_ngram=3)

    # # Display top 5 most expensive 2-grams for review
    # if '2-grams' in ngram_analysis and not ngram_analysis['2-grams'].empty:
    #     print("\nTop 5 Most Expensive 2-Grams:")
    #     print(ngram_analysis['2-grams'].head(5)[['N-Gram', 'Cost', 'CTR', 'CPA', 'ROAS']])

    # --- 4. Identify Opportunities ---
    print("\n--- 3. Identifying Optimization Opportunities ---")

    # Find ads that are not performing well
    underperforming_ads = find_underperforming_ads(ads_df)

    # Find the n-grams that perform the best ("Gold Nuggets")
    best_ngrams = find_best_ngrams(ngram_analysis)

    # Find n-grams with high impressions but low CTR ("Mismatches")
    mismatched_ngrams = find_mismatched_ngrams(ngram_analysis)

    # --- Display all findings ---
    if not underperforming_ads.empty:
        print("\nUnderperforming Ad(s) Identified:")
        print(underperforming_ads[['Campaign', 'Ad group', 'Headline 1', 'CTR']])

    if not best_ngrams.empty:
        print("\nTop 'Gold Nugget' N-Grams Found (High Conversion, Low CPA):")
        print(best_ngrams.head(5)[['N-Gram', 'Conversions', 'CPA', 'ROAS']])

    if not mismatched_ngrams.empty:
        print("\n'Mismatched' N-Grams Found (High Impressions, Low CTR):")
        print(mismatched_ngrams.head(5)[['N-Gram', 'Impressions', 'CTR']])

    # --- 5. Generate AI-Powered Suggestions ---
    print("\n--- 4. Generating Ad Copy Suggestions ---")
    if underperforming_ads.empty:
        print("No underperforming ads to generate suggestions for.")
    else:
        # For each underperforming ad, generate suggestions
        for index, ad_row in underperforming_ads.iterrows():
            print(f"\nProcessing Ad Group: '{ad_row['Ad group']}'")

            # Pass the initialized client to the suggestion function
            suggestions = generate_suggestions(gemini_client, ad_row, best_ngrams, mismatched_ngrams)

            for s in suggestions:
                print(s)

    print("\n--- AdVantage POC Analysis Complete ---")


if __name__ == '__main__':
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("dotenv library not found, assuming environment variables are set.")
    main()
