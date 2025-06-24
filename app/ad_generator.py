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
    
      # THE FIX: Validate that all prompt components are strings and not None.
    ad_group = underperforming_ad.get('Ad group', 'Unknown Ad Group')
    headline = underperforming_ad.get('Headline 1', '') # Use empty string if headline is missing
    ctr = underperforming_ad.get('CTR', 0.0)

    # Ensure headline is a string to prevent errors.
    if not isinstance(headline, str):
        headline = str(headline) if headline is not None else ''

    if not top_mismatched_ngrams:
        return ["No 'Mismatched' n-grams found to generate suggestions from."]

    prompt = f"""
**Persona**: Highly skilled Google Ads copywriter specializing in performance optimization and deeply knowledgeable in Google Ads policy guidelines (Prohibited Content, Restricted Content, Editorial & Technical Requirements, etc.) to ensure all ad variations are compliant.
**Task**: Develop two distinct, high-performing Search Ad variations to replace an underperforming ad.
**Context:**
Ad Group: "{underperforming_ad['Ad group']}"
Current Primary Headline: "{underperforming_ad['Headline 1']}"
Performance Issue: The existing ad is experiencing a low Click-Through Rate (CTR) of {underperforming_ad['CTR']:.2%} primarily due to a disconnect between the ad copy and user search queries.
Identified Keyword Gaps (High Impressions, Low CTR - indicating poor relevance): {top_mismatched_ngrams}
High-Converting Value Propositions (Proven "Gold Nugget" Phrases): {top_best_ngrams}
*Core Objectives for New Ad Variations*:

Maximize Relevance (Increase CTR): Integrate key phrases from "Identified Keyword Gaps" into new headlines to directly address user search intent.
Optimize for Conversion (Drive Action): Incorporate the "High-Converting Value Propositions" into the descriptions to highlight benefits and encourage desired actions.
Ad Creation Guidelines:

*Google Ads Policy Compliance (CRITICAL):*

All headlines and descriptions MUST strictly adhere to Google Ads policies. Before generating, thoroughly review and self-correct any potential policy violations. This includes, but is not limited to:
Prohibited Content: Absolutely no promotion of illegal activities, dangerous products/services (e.g., weapons, recreational drugs), counterfeit goods, or content that promotes hate speech or discrimination.
Restricted Content: If the product/service relates to sensitive categories (e.g., alcohol, gambling, healthcare, financial services, political content), ensure the copy respects all relevant restrictions, legal requirements, and necessary disclosures (e.g., age restrictions, licensing, "Terms Apply" disclaimers where appropriate).
Editorial & Technical Standards:
Maintain professional, clear, and accurate language.
Avoid excessive capitalization (e.g., "FREE!!!"), gimmicky symbols, jargon, vague phrasing, or misspellings.
Ensure a strong, direct relationship between the ad copy and the likely content of the landing page.
Do not use misleading claims or superlatives that cannot be substantiated (e.g., "Best in the World" without credible evidence).
Avoid asking for sensitive personal information directly in the ad copy (e.g., "Enter your SSN here").
Trademark Compliance: Do not use trademarked terms unless explicitly authorized by the trademark owner for this specific use case.

*Headlines:*
Craft three compelling and unique headlines per ad variation.
Each headline must be 30 characters or less.
Prioritize clarity and directness. Aim to include at least one "Identified Keyword Gap" phrase across the three headlines for each variation.
*Descriptions:*
Write two distinct, benefit-driven descriptions per ad variation.
Each description must be 90 characters or less.
Effectively leverage "High-Converting Value Propositions" to articulate the unique selling points and call users to action.
**Output Format:**
Generate 2 complete and distinct ad variations. Each variation should include:
3 unique headlines.
2 unique descriptions.
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
        response_stream = client.models.generate_content_stream(model=model_name, contents=contents, config=generation_config)
        full_response_text = "".join(chunk.text for chunk in response_stream)

        if not full_response_text.strip():
            return ["An error occurred: Received empty response from API."]

        response_json = json.loads(full_response_text)

        # Improved User-Friendly Output Formatting
        formatted_suggestions = []

        # BEFORE Section
        before_str = (
            "**BEFORE (Underperforming Ad)**\n\n"
            f"> **Headline:** {headline}\n"
            f"> **Description:** {underperforming_ad.get('Description 1', 'N/A')}\n"
            f"> **CTR:** {ctr:.2%}\n\n"
            f" **Top mismatched n-grams:** {', '.join(top_mismatched_ngrams)}\n\n"
            f" **Top gold nugget n-grams:** {', '.join(top_best_ngrams)}"

        )
        formatted_suggestions.append(before_str)

        # AFTER Header
        formatted_suggestions.append("\n---\n\n**AFTER (AI-Generated Suggestions)**")

        # Each suggestion as a separate card-like item
        for i, variation in enumerate(response_json.get('ad_variations', [])):
            suggestion_str = f"**Suggestion Set {i+1}**\n"
            headlines = variation.get('headlines', [])
            descriptions = variation.get('descriptions', [])

            suggestion_str += "> **Headlines:**\n"
            for j, h in enumerate(headlines):
                suggestion_str += f"> - H{j+1}: {h}\n"

            suggestion_str += "> \n" # spacer
            suggestion_str += "> **Descriptions:**\n"
            for k, d in enumerate(descriptions):
                suggestion_str += f"> - D{k+1}: {d}\n"

            formatted_suggestions.append(suggestion_str)
        
        # Print for console view
        print("\n".join(formatted_suggestions))
        return formatted_suggestions
    
    except Exception as e:
        return [f"An error occurred: {e}"]