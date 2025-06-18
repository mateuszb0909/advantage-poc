import pandas as pd

def find_underperforming_ads(ads_df, min_impressions=10000, max_ctr=0.04):
    """
    Identifies ads that are underperforming based on impression and CTR thresholds.
    """
    ads_df['CTR'] = ads_df['Clicks'] / ads_df['Impressions'].replace(0, 1)
    
    underperforming = ads_df[
        (ads_df['Impressions'] > min_impressions) & 
        (ads_df['CTR'] < max_ctr)
    ]
    
    return underperforming

def find_best_ngrams(ngram_analysis, min_conversions=5, max_cpa=50.0):
    """
    Identifies the best performing n-grams ("Gold Nuggets") from the analysis.
    """
    relevant_ngrams_df = pd.concat([
        ngram_analysis.get('2-grams', pd.DataFrame()), 
        ngram_analysis.get('3-grams', pd.DataFrame())
    ])

    if relevant_ngrams_df.empty:
        return pd.DataFrame()

    gold_nuggets = relevant_ngrams_df[
        (relevant_ngrams_df['Conversions'] >= min_conversions) &
        (relevant_ngrams_df['CPA'] <= max_cpa) &
        (relevant_ngrams_df['CPA'] > 0)
    ].copy()
    
    gold_nuggets.sort_values(by='ROAS', ascending=False, inplace=True)
    
    return gold_nuggets

def find_mismatched_ngrams(ngram_analysis, min_impressions=5000, max_ctr=0.05):
    """
    Identifies n-grams with high impressions but low CTR ("Mismatches").
    """
    relevant_ngrams_df = pd.concat([
        ngram_analysis.get('2-grams', pd.DataFrame()),
        ngram_analysis.get('3-grams', pd.DataFrame())
    ])

    if relevant_ngrams_df.empty:
        return pd.DataFrame()

    mismatches = relevant_ngrams_df[
        (relevant_ngrams_df['Impressions'] >= min_impressions) &
        (relevant_ngrams_df['CTR'] < max_ctr)
    ].copy()

    mismatches.sort_values(by='Impressions', ascending=False, inplace=True)

    return mismatches