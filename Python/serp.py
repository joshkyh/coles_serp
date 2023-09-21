from serpapi import GoogleSearch
import os
import pandas as pd
from tqdm import tqdm


def refresh_google_maps_data():
  # Get SERPAPI_KEY from environment variable
  api_key = os.environ.get("SERPAPI_KEY")
  assert len(api_key) > 0


  # Define the latlongs to search
  latlongs = [
    ("@-33.8617068,151.2218132,14.29z", 'Sydney'),
    ("@-37.9715653,144.7235152,10z", 'Melbourne'),
    ("@-32.0390554,115.6319122,10z", 'Perth'),
    ("@-27.3811459,152.6635707,10z", 'Brisbane'),
    ("@-35.0003743,138.4463569,11z", 'Adelaide')
    ]

  num_pages = 5

  # For loop to get results for each latlong, and two pages of results only. And append all results to a results dataframe
  # Declare a place holder for the dataframe that has predefined columns
  column_names = ['search_city', 'search_latlong','page', 'position', 'title', 'place_id', 'data_id', 'data_cid', 'reviews_link',
         'photos_link', 'gps_coordinates', 'place_id_search', 'provider_id',
         'rating', 'reviews', 'type', 'types', 'address', 'phone', 'website',
         'thumbnail', 'open_state', 'hours', 'operating_hours']
  results = pd.DataFrame(columns=column_names)

  # Nested for loop tqdm to show progress bar
  for latlong in tqdm(latlongs):
      # Get the first page of results
      for page in range(1, num_pages+1):
        result_offset_str = str((page - 1) * 20)

        params = {
          "api_key": api_key,
          "engine": "google_maps",
          "type": "search",
          "google_domain": "google.com.au",
          "q": "Coles Supermarket",
          "ll": latlong[0],
          "hl": "en",
          "gl": "au",
          "start": result_offset_str
        }

        search = GoogleSearch(params)
        api_response = search.get_dict()

        # If api_response['search_information']['local_results_state'] is 'Fully empty' then continue loop
        # somethings search_information is not here
        if 'search_information' in api_response:
          if api_response['search_information']['local_results_state'] == 'Fully empty':
            continue

        # Convert results['local_results'] to dataframe
        new_data = pd.DataFrame(api_response['local_results'])

        # Add the search_latlong column to the dataframe and reorder such that position and search_latlong are the first two columns
        new_data['search_city'] = latlong[1]
        new_data['search_latlong'] = latlong[0]
        new_data['page'] = page
        new_data = new_data.reindex(columns=column_names)

        # Row bind the new_data to the results dataframe
        results = pd.concat([results, new_data], ignore_index=True)

  # Write the results dataframe to a parquet file
  results.to_parquet('data/google_maps_results.parquet', engine='pyarrow')

def refresh_reviews_data():
  google_maps = pd.read_parquet('data/google_maps_results.parquet', engine='pyarrow')

  # count number of locations
  google_maps['place_id'].nunique()

  # Ensure not duplicates using place_id
  google_maps = google_maps.drop_duplicates(subset=['place_id'])

  # Filter for places with reviews
  google_maps = google_maps[google_maps['reviews'] > 0]


  # Results dataframe
  col_names = ['data_id', 'link', 'rating', 'date', 'source', 'review_id', 'user', 'snippet','images', 'extracted_snippet', 'likes']

  reviews = pd.DataFrame(columns=col_names)

  def get_all_reviews(data_id: str, reviews: pd.DataFrame, next_page_token: str = None) -> pd.DataFrame:
    '''
    Function to get all reviews for a given data_id. Will call itself recursively if there is a next_page_token returned by api_response
    :param data_id: The ID for which reviews are to be fetched
    :param reviews: DataFrame to which new reviews will be added
    :param next_page_token: Token for the next page in pagination
    :return: pd.DataFrame containing all reviews
    '''

    # Get SERPAPI_KEY from environment variable
    api_key = os.environ.get("SERPAPI_KEY")

    params = {
      "api_key": api_key,
      "engine": "google_maps_reviews",
      "hl": "en",
      "data_id": data_id,
      "sort_by": "newestFirst"
    }

    if next_page_token is not None:
      params['next_page_token'] = next_page_token

    try:
      search = GoogleSearch(params)
      api_response = search.get_dict()
    except Exception as e:
      print(f"An error occurred: {e}")
      return reviews

    if 'reviews' in api_response:
      new_data = pd.DataFrame(api_response['reviews'])
      new_data['data_id'] = data_id

      # Align the columns and add missing columns as NaN
      new_data = new_data.reindex(columns=col_names)

      reviews = pd.concat([reviews, new_data], ignore_index=True)

    # only one page of reviews
    next_page_token = api_response.get('serpapi_pagination', {}).get('next_page_token', None)
    next_page_token = None

    if next_page_token is not None:
      return get_all_reviews(data_id=data_id, reviews=reviews, next_page_token=next_page_token)


    print(f"Finished fetching reviews for {data_id}, length of reviews is {len(reviews)}")

    return reviews

  # For loop that iterates through data_id column of google_maps dataframe and calls get_all_reviews function
  for data_id in tqdm(google_maps['data_id']):
    reviews = get_all_reviews(data_id=data_id, reviews=reviews)


  # Write the results dataframe to a parquet file
  reviews.to_parquet('data/google_maps_reviews.parquet', engine='pyarrow')

