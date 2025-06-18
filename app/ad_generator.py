import json
from google import genai
from google.genai import types

def generate_suggestions(client, underperforming_ad, best_ngrams_df, mismatched_ngrams_df):
    """
    Generates new ad copy suggestions using the Google Gemini AI model.
    Args:
        client (genai.Client): The initialized Gemini API client.
        underperforming_ad (pd.Series): The ad to improve.
        best_ngrams_df (pd.DataFrame): Top-performing "Gold Nugget" n-grams.
        mismatched_ngrams_df (pd.DataFrame): "Mismatched" n-grams.
    Returns:
        list: A list of suggested new ad variations, or an error message.
    """
    # --- 1. Prepare Prompt Context ---
    top_best_ngrams = best_ngrams_df.head(5)['N-Gram'].tolist()
    top_mismatched_ngrams = mismatched_ngrams_df.head(5)['N-Gram'].tolist()

    if not top_mismatched_ngrams:
        return ["No 'Mismatched' n-grams found to generate suggestions from."]

    prompt = f"""
    **Persona:** Expert Google Ads copywriter.
    **Task:** Generate three new Responsive Search Ad variations for an underperforming ad.
    **Context:**
    * **Ad Group:** "{underperforming_ad['Ad group']}"
    * **Original Headline:** "{underperforming_ad['Headline 1']}"
    * **Problem:** The ad has a low CTR ({underperforming_ad['CTR']:.2%}) because it doesn't match what people search for.
    * **Problematic Phrases (High Impressions, Low CTR):** {top_mismatched_ngrams}
    * **Proven "Gold Nugget" Phrases (High Conversion):** {top_best_ngrams}
    **Instructions:**
    1.  **Fix Relevance:** MUST use "Problematic Phrases" in the new headlines to increase CTR.
    2.  **Drive Conversions:** Use "Gold Nugget" phrases in the descriptions.
    3.  **Limits:** Headlines <= 30 chars. Descriptions <= 90 chars.
    4.  **Output:** Create 3 distinct variations (3 headlines, 2 descriptions each).
    """

    # --- 2. Define Output Schema & Generation Config ---
    response_schema=types.Schema(
        type=types.Type.OBJECT,
        required=["ad_variations"],
        properties={"ad_variations": types.Schema(type=types.Type.ARRAY,
                                                  items=types.Schema(type=types.Type.OBJECT,
                                                                     required=["headlines", "descriptions"], 
                                                                     properties={"headlines": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)), "descriptions": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))}))})
    
    generation_config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema)
    
    contents = [types.Content(role="user",
                              parts=[types.Part(text=prompt)])]
    
    model_name = "gemini-2.5-flash-preview-05-20"

    # --- 3. Call API & Format Response ---
    print(f"\n   > Calling Gemini AI for '{underperforming_ad['Ad group']}'...")
    try:
        # THE FIX: The keyword argument should be 'config', not 'generation_config'.
        response_stream = client.models.generate_content_stream(model=model_name, contents=contents, config=generation_config)
        
        full_response_text = "".join(chunk.text for chunk in response_stream)
        
        if not full_response_text.strip():
            return ["  An error occurred: Received empty response from API."]
            
        response_json = json.loads(full_response_text)
        
        formatted_suggestions = []
        for i, variation in enumerate(response_json.get('ad_variations', [])):
            formatted_suggestions.append(f"  Suggestion Set {i+1}:")
            for j, headline in enumerate(variation.get('headlines', [])):
                formatted_suggestions.append(f"    H{j+1}: {headline}")
            for k, description in enumerate(variation.get('descriptions', [])):
                formatted_suggestions.append(f"    D{k+1}: {description}")
        return formatted_suggestions
    except Exception as e:
        return [f"  An error occurred: {e}"]
