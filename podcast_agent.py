from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import os
import asyncio
from typing import List

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

load_dotenv()

# Initialize OpenAI model
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

@dataclass
class PodcastAIDeps:
    """Dependencies for the Podcast AI agent, including Supabase and OpenAI clients."""
    supabase: Client
    openai_client: AsyncOpenAI

# Define system prompt
system_prompt = """
* You are smart and witty chatbot and specialize in answering questions about the Analytics Power Hour podcast You talk candidly and casually.
* You can answer general questions about the articles and provide relevant transcripts, when appropriate.
* You will always refer to the specific name of the article you are citing and hyperlink to its url, as such: [Title](URL).
* If you can't answer, you will explain why and suggest sending the question to brismiller@gmail.com where Brian can answer it directly!
* Facts about The Analytics Power Hour podcast: The Analytics Power Hour is a biweekly podcast that delves into various data and analytics topics, offering listeners insightful discussions and practical takeaways. Established in 2015, the podcast has released over 260 episodes, covering a wide range of subjects from AI projects to data storytelling. The show is co-hosted by a dynamic team of five professionals: Michael Helbling, Moe Kiss, Tim Wilson, Val Kroll, and Julie Hoyer. Each episode features these hosts, often accompanied by guest experts, sharing their thoughts and experiences in an open forum format. Whether you're a seasoned data professional or just starting in the analytics field, The Analytics Power Hour offers valuable insights and engaging discussions to enhance your understanding of the industry.
* You can access The Analytics Power Hour on various platforms, including their official website (https://analyticshour.io/), Apple Podcasts, Spotify, and Audible.
* Brian Miller (https://github.com/BrianMillerS) created you. Your code can be found at https://github.com/BrianMillerS/analytics_power_hour_chatbot. You are trained on the 256 most recent Stratechery articles.You are not approved by the authors of The Analytics Power Hour podcast.
"""

# Define the Podcast AI Agent
podcast_ai_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PodcastAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get an embedding vector from OpenAI for similarity search."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        print(f"🔹 Query Embedding: {text[:100]} -> {embedding[:5]}...")  # Log first 5 values
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0] * 1536  # Return a zero vector if an error occurs

import numpy as np

@podcast_ai_agent.tool
async def retrieve_relevant_podcast_transcripts(ctx: RunContext[PodcastAIDeps], user_query: str) -> str:
    """
    Retrieve the most relevant transcript chunks based on the user query.
    """
    try:
        # Get query embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Convert embedding to float32 and ensure it's a list
        query_embedding_vector = np.array(query_embedding, dtype=np.float32).tolist()  

        # Debugging logs
        print(f"DEBUG - Query Embedding Shape: {len(query_embedding_vector)}")  # Should be 1536
        print(f"DEBUG - First 10 Values: {query_embedding_vector[:10]}")  # Log sample values

        # Query Supabase for similar transcript chunks
        result = ctx.deps.supabase.rpc(
            'match_podcast_transcripts',
            {'query_embedding': query_embedding_vector, 'match_count': 5}
        ).execute()

        if not result.data:
            return "No relevant podcast excerpts found."

        print("\n🔹 Retrieved Podcast Transcripts:")
        formatted_chunks = []
        for doc in result.data:
            similarity_score = doc['similarity']
            print(f"Episode {doc['episode_number']} - Similarity: {similarity_score:.4f}")  # Log similarity score

            chunk_text = f"""
🎙 **Episode {doc['episode_number']}: [{doc['title']}]({doc['url']})**  
📌 **Similarity Score:** {similarity_score:.4f}  
📜 **Excerpt:**  
{doc['content'][:500]}...
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving podcast transcripts: {e}")
        return "Error retrieving podcast excerpts."




import asyncio
from pydantic_ai import RunContext
from openai import AsyncOpenAI
from supabase import create_client

# Initialize Supabase and OpenAI clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Create a test context
ctx = RunContext(
    model=model,  # ✅ Use the initialized OpenAI model
    usage={},  # ✅ Empty dictionary for now (could be used for tracking API calls)
    prompt="",  # ✅ Optional prompt, can be left empty
    deps=PodcastAIDeps(supabase=supabase, openai_client=openai_client)  # ✅ Inject dependencies
)

async def test_agent():
    user_query = "Tell me about the first episode of the Analytics Power Hour podcast."
    
    # Call the agent's tool directly
    response = await retrieve_relevant_podcast_transcripts(ctx, user_query)
    
    print("🔹 Agent Response:")
    print(response)

# Run the test
asyncio.run(test_agent())


