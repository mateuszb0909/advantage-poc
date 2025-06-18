import pandas as pd

def find_underperforming_ads(ads_df, min_impressions=10000, max_ctr=0.04):
    """
    Identifies ads that are underperforming based on impression and CTR thresholds.
    This simulates finding a "Mismatched" ad (Profile 3 from the methodology).

    Args:
        ads_df (pandas.DataFrame): DataFrame containing ad copy performance.
        min_impressions (int): The minimum number of impressions for an ad to be considered.
        max_ctr (float): The maximum CTR for an ad to be considered underperforming.

    Returns:
        pandas.DataFrame: A DataFrame of underperforming ads.
    """
    # Calculate CTR for each ad
    # We replace 0 impressions with 1 to avoid division-by-zero errors.
    ads_df['CTR'] = ads_df['Clicks'] / ads_df['Impressions'].replace(0, 1)
    
    # Filter for ads that have high impressions but a low CTR
    underperforming = ads_df[
        (ads_df['Impressions'] > min_impressions) & 
        (ads_df['CTR'] < max_ctr)
    ]
    
    print(f"\nFound {len(underperforming)} underperforming ad(s) based on criteria (Impressions > {min_impressions} and CTR < {max_ctr:.2%}).")
    return underperforming

def find_best_ngrams(ngram_analysis, min_conversions=5, max_cpa=50.0):
    """
    Identifies the best performing n-grams ("Gold Nuggets") from the analysis.
    This simulates finding "Gold Nuggets" (Profile 4 from the methodology).

    Args:
        ngram_analysis (dict): The dictionary of n-gram DataFrames from the analyzer.
        min_conversions (int): Minimum conversions for an n-gram to be considered.
        max_cpa (float): The maximum CPA for an n-gram to be considered a top performer.

    Returns:
        pandas.DataFrame: A DataFrame of the best performing n-grams.
    """
    # We are most interested in 2-grams and 3-grams for ad copy.
    # We use .get() to safely access the keys, providing an empty DataFrame as a default.
    relevant_ngrams_df = pd.concat([
        ngram_analysis.get('2-grams', pd.DataFrame()), 
        ngram_analysis.get('3-grams', pd.DataFrame())
    ])

    if relevant_ngrams_df.empty:
        print("No 2-gram or 3-gram data found to analyze for 'Gold Nuggets'.")
        return pd.DataFrame()

    # Filter for n-grams that have a good number of conversions and a low CPA
    gold_nuggets = relevant_ngrams_df[
        (relevant_ngrams_df['Conversions'] >= min_conversions) &
        (relevant_ngrams_df['CPA'] <= max_cpa) &
        (relevant_ngrams_df['CPA'] > 0) # Ensure it has conversions to have a valid CPA
    ].copy()
    
    # Sort by ROAS to find the absolute best performers
    gold_nuggets.sort_values(by='ROAS', ascending=False, inplace=True)
    
    print(f"Found {len(gold_nuggets)} 'Gold Nugget' n-grams (Conversions >= {min_conversions} and CPA <= ${max_cpa:.2f}).")
    return gold_nuggets

def find_mismatched_ngrams(ngram_analysis, min_impressions=5000, max_ctr=0.05):
    """
    Identifies n-grams with high impressions but low CTR ("Mismatches").
    This simulates finding "Mismatches" (Profile 3 from the methodology).

    Args:
        ngram_analysis (dict): The dictionary of n-gram DataFrames from the analyzer.
        min_impressions (int): Minimum impressions for an n-gram to be considered.
        max_ctr (float): The maximum CTR for an n-gram to be considered a mismatch.

    Returns:
        pandas.DataFrame: A DataFrame of the mismatched n-grams.
    """
    # Combine 2-gram and 3-gram data for analysis
    relevant_ngrams_df = pd.concat([
        ngram_analysis.get('2-grams', pd.DataFrame()),
        ngram_analysis.get('3-grams', pd.DataFrame())
    ])

    if relevant_ngrams_df.empty:
        print("No 2-gram or 3-gram data found to analyze for 'Mismatches'.")
        return pd.DataFrame()

    # Filter for n-grams with high impressions but a low CTR
    mismatches = relevant_ngrams_df[
        (relevant_ngrams_df['Impressions'] >= min_impressions) &
        (relevant_ngrams_df['CTR'] < max_ctr)
    ].copy()

    # Sort by Impressions descending to see the biggest opportunities first
    mismatches.sort_values(by='Impressions', ascending=False, inplace=True)

    print(f"Found {len(mismatches)} 'Mismatched' n-grams (Impressions >= {min_impressions} and CTR < {max_ctr:.2%}).")
    return mismatches
