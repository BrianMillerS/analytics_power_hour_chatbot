import pickle
import os
import openai
import pandas as pd
import nest_asyncio
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

# Load the podcast data from the pickle file
with open('podcast_data.pkl', 'rb') as file:
    podcast_data = pickle.load(file)

# Load environment variables from the .env file
load_dotenv()

# Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

#############################################################################################
episode_number_to_match = ['042', '043', '132', '133', '200', '201']

# Create a subset of the podcast data where the episode number matches the specified values
podcast_data = podcast_data[podcast_data['episode number'].isin(episode_number_to_match)]
#############################################################################################

def generate_summary(transcript: str, existing_summary: str) -> str:
    """
    Uses GPT-4o-mini to summarize the key takeaway points from both the transcript
    and the existing summary. Returns a concise bullet-point list.
    """
    prompt_content = (
        "Below is the transcript and an existing summary for a podcast episode. "
        "Summarize the key takeaway points in bullet-point form, with each bullet representing a distinct key point. You can use multiple sentences if you need to. "
        "Make sure to include all the important details and main ideas. "
        "Do not include any additional commentary.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Existing Summary:\n{existing_summary}"
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that creates clear, concise summaries of podcasts. "
                    "You think from the perspective of a listener who wants to quickly grasp the main points. "
                    "Your audience is data scientists and machine learning engineers who are listening to try to expand their skillsets and learn from the experiences of the podcasters."
                )
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def add_gpt_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in the DataFrame, sends both the transcript and the existing summary to the model for summarization,
    then stores the result in a new 'GPT Summary' column.
    """
    summaries = []
    for _, row in df.iterrows():
        print(f"Generating summary for episode {row['episode number']}")
        transcript = row['Transcript']
        # Use the 'Summary' column if it exists; otherwise, default to an empty string.
        existing_summary = row.get('Summary', '')
        summary_text = generate_summary(transcript, existing_summary)
        summaries.append(summary_text)
    df['GPT Summary'] = summaries
    return df


def generate_memorable_quotes(transcript: str, existing_summary: str) -> str:
    """
    Uses GPT-4o-mini to extract the most memorable quotes from the podcast transcript,
    explaining each quote with context and including the last time stamp for that quote.
    Returns a bullet-point list of quotes with their context.
    """
    prompt_content = (
        "Below is the transcript and an existing summary for a podcast episode. "
        "Extract the most memorable quotes from the transcript and, for each quote, "
        "provide a brief explanation of its context along with the last time stamp where that quote appears. "
        "Present the results as a bullet-point list, where each bullet contains the quote, context, and time stamp. "
        "Do not include any additional commentary.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Existing Summary:\n{existing_summary}"
    )
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant skilled in extracting memorable quotes from podcasts. "
                    "Your output should be clear, concise, and formatted as a bullet-point list. "
                    "Your audience is data scientists and machine learning engineers who are listening to try to expand their skillsets and learn from the experiences of the podcasters."
                    "When extracting quotes, focus on the ones that are impactful, insightful, or thought-provoking. "
                    "When stating people's names, use their full names. "
                    "Each bullet should contain the quote, the speakers full name, a brief explanation of its context, and the last time stamp in the transcript where that quote occurs."
                    "Format the output as a bullet-point list like this: -'quote'\n*Speaker's full name\n*explanation\n*last time stamp"
                )
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def add_gpt_memorable_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in the DataFrame, sends both the transcript and the existing summary to the model for extracting memorable quotes,
    then stores the result in a new 'GPT Memorable Quotes' column.
    """
    quotes_list = []
    for _, row in df.iterrows():
        print(f"Generating memorable quotes for episode {row['episode number']}")
        transcript = row['Transcript']
        # Use the 'Summary' column if it exists; otherwise, default to an empty string.
        existing_summary = row.get('GPT Summary', '')
        quotes_text = generate_memorable_quotes(transcript, existing_summary)
        quotes_list.append(quotes_text)
    df['GPT Memorable Quotes'] = quotes_list
    return df

def get_embedding(text: str) -> List[float]:
    """Generate an embedding vector using OpenAI's updated SDK."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]  # input must be a list in the new SDK
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0] * 1536

def insert_quotes(
    episode_number: int,
    title: str,
    url: str,
    published_date: str,
    participants: str,
    transcript: str,
    gpt_memorable_quotes: str
):
    """Insert quote data with embedding into Supabase (synchronously)."""
    try:
        embedding = get_embedding(gpt_memorable_quotes)

        data = {
            "episode_number": episode_number,
            "title": title,
            "url": url,
            "published_date": published_date,
            "participants": participants,
            "transcript": transcript,
            "gpt_memorable_quotes": gpt_memorable_quotes,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table("podcast_quotes").insert(data).execute()
        print(f"Inserted quotes for episode {episode_number}")
        return result
    except Exception as e:
        print(f"❌ Error inserting quotes for episode {episode_number}: {e}")
        return None


def process_quotes_sequentially(df: pd.DataFrame):
    """Sequentially processes and uploads memorable quotes to Supabase."""
    try:
        supabase.table("podcast_quotes").delete().neq("id", 0).execute()
        print("Cleared podcast_quotes table before re-uploading.")
    except Exception as e:
        print(f"Error clearing table: {e}")

    for _, row in df.iterrows():
        try:
            episode_number = int(row["episode number"])
            title = row["Title"]
            url = row["URL"]
            published_date = row["Date Published"]
            participants = row.get("Participants", "")
            transcript = row.get("Transcript", "")
            gpt_memorable_quotes = row["GPT Memorable Quotes"]

            insert_quotes(
                episode_number,
                title,
                url,
                published_date,
                participants,
                transcript,
                gpt_memorable_quotes
            )

        except Exception as e:
            print(f"Skipping episode {row['episode number']}: {e}")


if __name__ == "__main__":
    print("Adding GPT summaries to the podcast data, used for quote creation:")
    podcast_data = add_gpt_summaries(podcast_data)
    
    print("\nAdding GPT memorable quotes to the podcast data:")
    podcast_data = add_gpt_memorable_quotes(podcast_data)
    
    # subset the podcast data to only include the columns we need
    podcast_quotes = podcast_data[['episode number','Title','URL','Date Published','Participants','Transcript','GPT Memorable Quotes']]
    
    print("\nUploading memorable quotes to Supabase:")
    process_quotes_sequentially(podcast_quotes)
    print("Script Complete")
    





