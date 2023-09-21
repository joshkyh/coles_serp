import os
import openai
import pandas as pd
from tqdm import tqdm
import pandas as pd


# Move get_embedding to the global scope
def get_embedding(text_to_embed):
    embedding = openai.Embedding.create(
        input=text_to_embed, model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return embedding

def get_ada_embeddings():
    # Get the reviews
    reviews = pd.read_parquet('data/google_maps_reviews.parquet', engine='pyarrow')

    # Get non-null reviews
    reviews = reviews[reviews['snippet'].notnull()]

    # Get Quartile 1 character length
    reviews['snippet'].str.len().quantile(0.25)
    reviews['snippet'].str.len().quantile(0.50)
    reviews['snippet'].str.len().quantile(0.75)


    # Get OPENAI KEY
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize tqdm with pandas
    tqdm.pandas()

    # Apply the function and show a progress bar
    reviews["embedding"] = reviews["snippet"].astype(str).progress_apply(get_embedding)

    # Write the reviews to a parquet file
    reviews.to_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')


def get_sentiment(text):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=f'''Return the language and sentiment score out of 5 for the input, in the following JSON format:
        {
            "language": "en",
            "sentiment": 0.5
        }
        
        input: {text}''',
        temperature=0.0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    sentiment = response.choices[0].text.strip()
    return sentiment

def get_sentiment_language():
    # Get the reviews
    reviews = pd.read_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')

    # Get non-null reviews
    reviews = reviews[reviews['snippet'].notnull()]

if __name__ == '__main__':
    get_ada_embeddings()

