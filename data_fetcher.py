import pandas as pd
import requests
import os
import time

# Config section for wase
API_KEY = "My Key was here" # Not sharing due to privacy
dir = 'C:/Users/pragy/OneDrive/Desktop/CDC_Submission Project'
datasets = dir + '/dataset'
image_loc = dir + '/property_images'

class SatelliteImageFetcher:     #class to download satellite images for a list of coordinates with Google Maps API.
    
    def __init__(self, api_key, base_url="https://maps.googleapis.com/maps/api/staticmap"):
        self.api_key = api_key
        self.base_url = base_url
        
    def fetch_single_image(self, lat, lon, zoom=18, size="224x224", maptype="satellite", save_dir="./images"):

        # Create the parameters for the API request
        params = {
            'center': f'{lat},{lon}',
            'zoom': zoom,
            'size': size,
            'maptype': maptype,
            'key': self.api_key
        }
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate a filename based on coordinates
        filename = f"lat{lat}_lon{lon}_z{zoom}.png"
        filepath = os.path.join(save_dir, filename)
        
        try:
            # Make the request to the API
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Saved image for ({lat}, {lon}) to {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to fetch image for ({lat}, {lon}): {e}")
            return None
    
    def fetch_from_dataframe(self, df, lat_col='lat', lon_col='long', delay=0.1, **kwargs):
        
        saved_paths = []
        
        for idx, row in df.iterrows():
            lat = row[lat_col]
            lon = row[lon_col]
            
            # Fetch the image
            saved_path = self.fetch_single_image(lat, lon, **kwargs)
            if saved_path:
                saved_paths.append(saved_path)
            
            # Small delay for API rate limits
            time.sleep(delay)
        
        print(f"\nCompleted. Successfully saved {len(saved_paths)} out of {len(df)} images.")
        return saved_paths


# Usage here only.
if __name__ == "__main__":
    
    fetcher = SatelliteImageFetcher(api_key=API_KEY)
    
    df_train = pd.read_csv(datasets + "/train(1)(train(1)).csv")
    df_test = pd.read_csv(datasets + "/test2(test(1)).csv")

    if not os.path.exists(image_loc + "/train"):   # make directory to store train images
        os.makedirs(image_loc + "/train")

    if not os.path.exists(image_loc + "/test"):  # directory to store test images
        os.makedirs(image_loc + "/test")

    saved_images_1 = fetcher.fetch_from_dataframe(
        df_train,
        lat_col='lat',          
        lon_col='long',         
        zoom=18,                
        size="600x400",         
        maptype="satellite",    
        save_dir= image_loc + "/train",  
        delay=0.2               
    )

    saved_images_2 = fetcher.fetch_from_dataframe(
        df_test,
        lat_col='lat',          
        lon_col='long',         
        zoom=18,                
        size="600x400",         
        maptype="satellite",    
        save_dir= image_loc + "/test",  
        delay=0.2               
    )