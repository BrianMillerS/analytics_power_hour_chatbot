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

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

#### Generate Summaries with GPT-4o-mini ####

# Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        existing_summary = row.get('Summary', '')
        summary_text = generate_summary(transcript, existing_summary)
        summaries.append(summary_text)
    df['GPT Summary'] = summaries
    return df

# Add GPT-4o-mini summaries to the podcast data
podcast_data = add_gpt_summaries(podcast_data)
print("All summaries generated...\n")
print("Beginning to upload summaries to Supabase...")

#### Upload summaries to Supabase ####
podcast_summaries = podcast_data[['episode number','Title','URL','GPT Summary','Date Published']]

# Load environment variables
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Limit concurrent API calls (adjust as needed)
API_SEMAPHORE = asyncio.Semaphore(4)

async def get_embedding(text: str) -> List[float]:
    """Generate an embedding vector from OpenAI for retrieval."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def insert_summary(episode_number: int, title: str, url: str, gpt_summary: str, published_date: str):
    """Insert a summary with its embedding into Supabase."""
    try:
        embedding = await get_embedding(gpt_summary)

        data = {
            "episode_number": episode_number,
            "title": title,
            "url": url,
            "gpt_summary": gpt_summary,
            "published_date": published_date,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        result = supabase.table("podcast_summaries").insert(data).execute()
        print(f"Inserted summary for episode {episode_number}")
        return result
    except Exception as e:
        print(f"Error inserting summary: {e}")
        return None

async def process_all_summaries(df: pd.DataFrame):
    """Processes all summaries and uploads them to Supabase."""
    
    # Clear existing data before inserting new summaries
    try:
        supabase.table("podcast_summaries").delete().neq("id", 0).execute()
        print("Cleared podcast_summaries table before re-uploading.")
    except Exception as e:
        print(f"Error clearing table: {e}")

    tasks = []
    for _, row in df.iterrows():
        try:
            episode_number = int(row["episode number"])
            title = row["Title"]
            url = row["URL"]
            gpt_summary = row["GPT Summary"]
            published_date = row["Date Published"]

            tasks.append(insert_summary(episode_number, title, url, gpt_summary, published_date))
        except ValueError:
            print(f"Skipping row due to invalid episode number: {row['episode number']}")

    await asyncio.gather(*tasks)

def main():
    """Main function to execute the async event loop properly."""
    nest_asyncio.apply()  # Allows running nested event loops in some environments
    asyncio.run(process_all_summaries(podcast_summaries))

if __name__ == "__main__":
    main()  
