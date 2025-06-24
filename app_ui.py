import os
import io
import pandas as pd
from nicegui import ui, run
from nicegui.events import UploadEventArguments
from dotenv import load_dotenv

# Import functions from your project modules
from app.data_loader import load_data
from app.ngram_analyzer import analyze_ngrams
from app.ad_analyzer import find_underperforming_ads, find_best_ngrams, find_mismatched_ngrams
from app.ad_generator import generate_suggestions

# Load environment variables from a .env file
load_dotenv()

# This dictionary will hold the content of the uploaded files.
uploaded_file_content = {
    'ads': None,
    'search_terms': None,
}

def load_data_from_stream(stream):
    """Loads a CSV file from a memory stream into a pandas DataFrame."""
    try:
        stream.seek(0)
        df = pd.read_csv(stream)
        return df
    except Exception as e:
        ui.notify(f"Error loading data: {e}", type='negative')
        return None

@ui.page('/')
def main_page():
    """Defines the main user interface of the application."""
    # --- UI Setup ---

    with ui.card().classes('w-full max-w-4xl mx-auto'):
        ui.label('AdVantage POC Analysis').classes('text-h4 font-bold text-center my-4')
        with ui.card_section():
            ui.label('1. Upload Your Data').classes('text-h6')
            ui.label('Provide your ad performance and search term data in CSV format.')

        def handle_ads_upload(e: UploadEventArguments):
            uploaded_file_content['ads'] = e.content.read()
            ui.notify(f"Successfully uploaded {e.name}", type='positive')

        def handle_search_terms_upload(e: UploadEventArguments):
            uploaded_file_content['search_terms'] = e.content.read()
            ui.notify(f"Successfully uploaded {e.name}", type='positive')

        # File Upload Section
        with ui.row().classes('w-full items-center gap-4 p-4'):
            ui.upload(
                label="Upload Ads Data (CSV)",
                auto_upload=True,
                on_upload=handle_ads_upload
            ).props('accept=.csv').classes('flex-grow')
            ui.upload(
                label="Upload Search Terms Data (CSV)",
                auto_upload=True,
                on_upload=handle_search_terms_upload
            ).props('accept=.csv').classes('flex-grow')

        # Action Button
        with ui.card_section():
             ui.label('2. Run Analysis').classes('text-h6')
             ui.label('Click the button to analyze your data and generate AI-powered suggestions.')
             run_button = ui.button('Run Analysis & Generate Suggestions', on_click=lambda: run_analysis()).classes('mt-2')


    # --- Tabbed Results Area ---
    # This section will be populated after the analysis runs.
    
    with ui.row().classes('w-full mx-0'):
         with ui.card().classes('w-full mx-0'):
            # Create a container for the tabs that will be added dynamically.
            tabs_container = ui.column().classes('w-full')


    async def run_analysis():
        """
        The main function that orchestrates the data loading, analysis,
        and AI content generation.
        """
        # --- 1. Validation ---
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            ui.notify("GEMINI_API_KEY not found in environment variables.", type='negative')
            return

        if not uploaded_file_content['ads'] or not uploaded_file_content['search_terms']:
            ui.notify("Please upload both ads.csv and search_terms.csv.", type='negative')
            return

        # --- 2. Initialization ---
        run_button.disable()
        tabs_container.clear() # Clear previous results
        with tabs_container:
            ui.spinner(size='lg').classes('self-center my-8')
            ui.label('Processing... This may take a moment.').classes('self-center')

        try:
            from google import genai
            gemini_client = genai.Client(api_key=api_key)
        except Exception as e:
            ui.notify(f"Failed to initialize Gemini client: {e}", type='negative')
            run_button.enable()
            return

        # --- 3. Data Processing (IO-Bound) ---
        ads_df, search_terms_df, ngram_analysis, underperforming_ads, best_ngrams, mismatched_ngrams = await run.io_bound(
            process_data_files
        )
        if ads_df is None: # Check if data loading failed
            run_button.enable()
            return

        # --- 4. Render Results in Tabs ---
        tabs_container.clear() # Clear the spinner
        with tabs_container:
            with ui.tabs().classes('w-full') as tabs:
                analysis_tab = ui.tab('📊 Analysis Results')
                suggestions_tab = ui.tab('🚀 AI-Powered Ad Copy Suggestions')

            with ui.tab_panels(tabs, value=analysis_tab).classes('w-full bg-transparent p-4'):
                # --- Analysis Tab Panel ---
                with ui.tab_panel(analysis_tab):
                    populate_analysis_tab(underperforming_ads, best_ngrams, mismatched_ngrams)
                # --- Suggestions Tab Panel ---
                with ui.tab_panel(suggestions_tab):
                    await populate_suggestions_tab(gemini_client, underperforming_ads, best_ngrams, mismatched_ngrams)

        run_button.enable()


    def process_data_files():
        """Loads data and runs initial analysis. Bundled for io_bound call."""
        ads_df = load_data_from_stream(io.BytesIO(uploaded_file_content['ads']))
        search_terms_df = load_data_from_stream(io.BytesIO(uploaded_file_content['search_terms']))
        if ads_df is None or search_terms_df is None:
            return None, None, None, None, None, None # Return Nones on failure

        ngram_analysis = analyze_ngrams(search_terms_df, min_ngram=2, max_ngram=3)
        underperforming_ads = find_underperforming_ads(ads_df)
        best_ngrams = find_best_ngrams(ngram_analysis)
        mismatched_ngrams = find_mismatched_ngrams(ngram_analysis)
        return ads_df, search_terms_df, ngram_analysis, underperforming_ads, best_ngrams, mismatched_ngrams


    def populate_analysis_tab(underperforming_ads, best_ngrams, mismatched_ngrams):
        """Creates the UI content for the Analysis Results tab."""
        # Underperforming Ads Section
        ui.label("Underperforming Ad(s) Identified:").classes('text-xl font-semibold')
        ui.markdown(
            "**Methodology:** These ads are identified as underperforming as they have a high number of "
            "impressions (>10,000) but a low Click-Through Rate (CTR) (<4%)."
        ).classes('text-sm my-2')
        if not underperforming_ads.empty:
            ui.table(columns=[{'name': col, 'label': col, 'field': col} for col in underperforming_ads.columns], rows=underperforming_ads.to_dict('records')).classes('w-full my-4')
        else:
             ui.label("No underperforming ads were found.").classes('text-lg')

        # Gold Nugget N-Grams Section
        ui.label("Top 'Gold Nugget' N-Grams:").classes('text-xl font-semibold mt-6')
        ui.markdown(
            "**Methodology:** These 'Gold Nugget' phrases are sourced from search terms that lead to strong performance. "
            "They are identified by having a healthy number of conversions (>=5) and a cost per acquisition (CPA) "
            "below a specific target (< $50), indicating high profitability. They are sorted by Return On Ad Spend (ROAS) "
            "to prioritize the most valuable terms."
        ).classes('text-sm my-2')
        if not best_ngrams.empty:
            ui.table(columns=[{'name': col, 'label': col, 'field': col} for col in best_ngrams.head().columns], rows=best_ngrams.head().to_dict('records')).classes('w-full my-4')

        # Mismatched N-Grams Section
        ui.label("'Mismatched' N-Grams:").classes('text-xl font-semibold mt-6')
        ui.markdown(
            "**Methodology:** These 'Mismatched' phrases have high impression volume (e.g., >5,000) but a low CTR "
            "(e.g., <5%). This suggests a relevance gap between what users are searching for and what your ad headline "
            "communicates. Using these phrases in ad copy can directly address user queries and improve click rates."
        ).classes('text-sm my-2')
        if not mismatched_ngrams.empty:
            ui.table(columns=[{'name': col, 'label': col, 'field': col} for col in mismatched_ngrams.head().columns], rows=mismatched_ngrams.head().to_dict('records')).classes('w-full my-4')


    async def populate_suggestions_tab(gemini_client, underperforming_ads, best_ngrams, mismatched_ngrams):
        """Creates the UI content for the AI-Powered Ad Copy Suggestions tab."""
        ui.label("AI-Powered Ad Copy Suggestions").classes('text-xl font-semibold')
        if underperforming_ads.empty:
            ui.label("No underperforming ads to generate suggestions for.").classes('mt-4')
            return

        # Process each ad and generate suggestions
        for index, ad_row in underperforming_ads.iterrows():
            with ui.card().classes('w-full my-4'):
                with ui.card_section():
                    ui.label(f"Suggestions for Ad Group: '{ad_row['Ad group']}'").classes('text-lg font-medium')
                
                suggestion_container = ui.column().classes('w-full p-4')
                with suggestion_container:
                    ui.spinner(size='md')
                    ui.label(f"Calling Gemini AI for '{ad_row['Ad group']}'...").classes('ml-2')

                # Run the blocking generate_suggestions function in a background thread
                suggestions = await run.io_bound(
                    generate_suggestions,
                    gemini_client,
                    ad_row,
                    best_ngrams,
                    mismatched_ngrams
                )

                # Clear the spinner and display the results
                suggestion_container.clear()
                with suggestion_container:
                    if suggestions and isinstance(suggestions, list):
                        for suggestion in suggestions:
                            ui.markdown(suggestion)
                    else:
                        ui.label("Could not generate suggestions for this ad.")

# Standard entry point for running the app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
