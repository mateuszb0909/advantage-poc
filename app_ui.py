import os
import io
import pandas as pd
from nicegui import ui
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
# This approach is more compatible with different NiceGUI versions.
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
    ui.label('AdVantage POC Analysis').classes('text-h4')

    # --- Upload Handlers ---
    # These functions are called when a file is uploaded.
    # They store the file's content in our dictionary.
    def handle_ads_upload(e: UploadEventArguments):
        """Handle the ads.csv file upload."""
        uploaded_file_content['ads'] = e.content.read()
        ads_upload_button.set_text(f"Uploaded: {e.name}")
        ui.notify(f"Successfully uploaded {e.name}", type='positive')

    def handle_search_terms_upload(e: UploadEventArguments):
        """Handle the search_terms.csv file upload."""
        uploaded_file_content['search_terms'] = e.content.read()
        search_terms_upload_button.set_text(f"Uploaded: {e.name}")
        ui.notify(f"Successfully uploaded {e.name}", type='positive')


    # --- File Uploaders ---
    with ui.row():
        with ui.column():
            ui.label("Upload Ads Data (CSV)")
            ui.upload(auto_upload=True, on_upload=handle_ads_upload)
            ads_upload_button = ui.button("Upload ads.csv")

        with ui.column():
            ui.label("Upload Search Terms Data (CSV)")
            ui.upload(auto_upload=True, on_upload=handle_search_terms_upload)
            search_terms_upload_button = ui.button("Upload search_terms.csv")

    # --- Results Area ---
    results_area = ui.column().classes('mt-4')

    async def run_analysis():
        """Main function to run the AdVantage POC workflow."""
        # Load API Key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            ui.notify("GEMINI_API_KEY not found in environment variables.", type='negative')
            ui.notify("Please set it in a .env file or your system environment.", type='negative')
            return

        # Check if files have been uploaded by inspecting our dictionary
        if not uploaded_file_content['ads'] or not uploaded_file_content['search_terms']:
            ui.notify("Please upload both ads.csv and search_terms.csv.", type='negative')
            return

        results_area.clear()
        with results_area:
            with ui.spinner(size='lg'):
                ui.label('Running Analysis... This may take a moment.')

                try:
                    from google import genai
                    gemini_client = genai.Client(api_key=api_key)
                except Exception as e:
                    ui.notify(f"Failed to initialize Gemini client: {e}", type='negative')
                    return

                # Load data from the stored file content
                ads_df = load_data_from_stream(io.BytesIO(uploaded_file_content['ads']))
                search_terms_df = load_data_from_stream(io.BytesIO(uploaded_file_content['search_terms']))

                if ads_df is None or search_terms_df is None:
                    return

                # --- Analysis Steps ---
                ngram_analysis = analyze_ngrams(search_terms_df, min_ngram=2, max_ngram=3)
                underperforming_ads = find_underperforming_ads(ads_df)
                best_ngrams = find_best_ngrams(ngram_analysis)
                mismatched_ngrams = find_mismatched_ngrams(ngram_analysis)

                # --- Display Findings ---
                results_area.clear()
                ui.label('Analysis Results').classes('text-h5')

                if not underperforming_ads.empty:
                    ui.label("Underperforming Ad(s) Identified:").classes('text-lg')
                    ui.table(
                        columns=[{'name': col, 'label': col, 'field': col} for col in underperforming_ads.columns],
                        rows=underperforming_ads.to_dict('records')
                    ).classes('w-full')

                if not best_ngrams.empty:
                    ui.label("Top 'Gold Nugget' N-Grams:").classes('text-lg mt-4')
                    ui.table(
                        columns=[{'name': col, 'label': col, 'field': col} for col in best_ngrams.head().columns],
                        rows=best_ngrams.head().to_dict('records')
                    ).classes('w-full')

                if not mismatched_ngrams.empty:
                    ui.label("'Mismatched' N-Grams:").classes('text-lg mt-4')
                    ui.table(
                        columns=[{'name': col, 'label': col, 'field': col} for col in mismatched_ngrams.head().columns],
                        rows=mismatched_ngrams.head().to_dict('records')
                    ).classes('w-full')

                # --- Generate AI Suggestions ---
                ui.label("AI-Powered Ad Copy Suggestions").classes('text-h5 mt-4')
                if underperforming_ads.empty:
                    ui.label("No underperforming ads to generate suggestions for.")
                else:
                    for index, ad_row in underperforming_ads.iterrows():
                        with ui.card().classes('w-full mt-2'):
                            ui.label(f"Suggestions for Ad Group: '{ad_row['Ad group']}'").classes('text-lg')
                            suggestions = generate_suggestions(gemini_client, ad_row, best_ngrams, mismatched_ngrams)
                            for suggestion in suggestions:
                                ui.markdown(suggestion)

    ui.button('Run Analysis', on_click=run_analysis).classes('mt-4')


ui.run()