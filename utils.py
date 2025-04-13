import os
import requests
import numpy as np
import pandas as pd
import spotipy
import h5py
import re

from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input


import pickle
# import lmdb



DEBUG = os.getenv("DEBUG") == "True"
if DEBUG:
    print("Mode débogage activé")
    
    
def load_model(lite=False):
    """Loads a MobileNet model, either in its full form or as a TensorFlow Lite (TFLite) model.
    Args:
        lite (bool): If True, loads the model as a TFLite interpreter. If False, loads the full MobileNet model.
    Returns:
        If lite is True:
                tuple: A tuple containing:
                    - interpreter (tf.lite.Interpreter): The TFLite interpreter for the model.
                    - input_details (list): Details about the input tensors of the TFLite model.
                    - output_details (list): Details about the output tensors of the TFLite model.
            If lite is False:
                keras.Model: The full MobileNet model with ImageNet weights, excluding the top classification layer.
    """
    if lite:
        model = MobileNet(weights='imagenet',
                        include_top=False,
                        pooling='avg',
                        input_shape=(224, 224, 3))
        
        if os.path.exists("mobilenet_v2.tflite"):
            interpreter = tf.lite.Interpreter(model_path="mobilenet_v2.tflite")
        else:
            # Convert the model to TFLite format
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            # Save the TFLite model to file
            with open("mobilenet_v2.tflite", "wb") as f:
                f.write(tflite_model)

            print("TFLite model saved as mobilenet_v2.tflite")
            
            interpreter = tf.lite.Interpreter(model_path="mobilenet_v2.tflite")
        
        interpreter.allocate_tensors()
        # Get input & output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()         
                
        return interpreter, input_details, output_details
    else:
        model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
        return model



def normalize_feature(features):
    """Function to normalize features

    Args:
        features (np.array): Features to normalize

    Returns:
        np.array: Normalized features
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features/norms


def get_test_image_features(img_path, lite=False):
    """
    Extracts feature vectors from an image using either a TensorFlow Lite model or a MobileNet model.
    Args:
        img_path (str): The file path to the image to be processed.
        lite (bool, optional): If True, uses a TensorFlow Lite model for feature extraction. 
                               If False, uses a standard MobileNet model. Defaults to False.
    Returns:
        numpy.ndarray: A flattened feature vector extracted from the image."""
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0  # Normalize like MobileNetV2

    if lite:
        # Load the TFLite model
        interpreter, input_details, output_details = load_model(lite=lite)
        

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get the output tensor (feature vector)
        features = interpreter.get_tensor(output_details[0]['index'])
    else:
        # Load the MobileNet model
        model = load_model(lite=lite)
        # Extract features using the model
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
    
    
    return features.flatten()

def connect_to_spotify():
    """
    Connect to the Spotify API using OAuth authentication.

    This function loads environment variables, checks if debug mode is enabled,
    and initializes the Spotify client using OAuth credentials.

    Args:
        None

    Returns:
        spotipy.Spotify: An authenticated Spotify client instance.
    """
    
    load_dotenv()

    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

    # client_credential_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    # sp = spotipy.Spotify(client_credentials_manager=client_credential_manager)

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                                client_secret=CLIENT_SECRET,
                                                redirect_uri=REDIRECT_URI,
                                                scope='user-library-read'))
    print("connected to spotify")
    return sp

def get_tracks_list(sp_rqrmt, url_playlist="https://open.spotify.com/collection/tracks", debug_mode=False):
    """
    Retrieves a list of tracks from a Spotify playlist or the user's liked songs.

    Args:
        sp_rqrmt (object): An authenticated Spotify client instance (e.g., spotipy.Spotify).
        url_playlist (str, optional): The URL of the Spotify playlist to fetch tracks from.
            Defaults to the URL for the user's liked songs.
        debug_mode (bool, optional): If True, limits the number of tracks retrieved to 100 for debugging purposes.
            Defaults to False.

    Returns:
        list: A list of track objects retrieved from the specified playlist or liked songs.
    """
    all_tracks = []
    limit = 50 if url_playlist == "https://open.spotify.com/collection/tracks" else 100
    offset = 0
    max_tracks = 100 if debug_mode else None
    
    # Choose the correct method based on URL
    is_liked_songs = url_playlist == "https://open.spotify.com/collection/tracks"
    fetch_tracks = sp_rqrmt.current_user_saved_tracks if is_liked_songs else sp_rqrmt.playlist_tracks
    playlist_uri = None if is_liked_songs else url_playlist.split("/")[-1].split("?")[0]
    
    # Retrieve tracks with pagination
    while True:
        # Fetch a batch of tracks
        if is_liked_songs:
            response = fetch_tracks(limit=limit, offset=offset)
        else:
            response = fetch_tracks(playlist_id=playlist_uri, limit=limit, offset=offset)
        
        # Extend all_tracks with the current batch
        all_tracks.extend(response["items"])
        
        # Break conditions for pagination
        if len(response['items']) < limit or (max_tracks and len(all_tracks) >= max_tracks):
            break
        
        # Update offset for next batch
        offset += limit
    
    # Trim to max_tracks if in debug mode
    if max_tracks:
        all_tracks = all_tracks[:max_tracks]

    return all_tracks

    
def extract_liked_song(sp, debug_mode=False):
    """
        Extracts metadata of liked songs from a Spotify account and saves it to a CSV file.
    This function retrieves all liked tracks from a Spotify account using the provided
    Spotify client object. It extracts metadata such as track name, album name, artist(s),
    and album image URL, and stores the data in a pandas DataFrame. The metadata is then
    saved to a CSV file for further use.
    Args:
        sp (spotipy.Spotify): A Spotipy client object authenticated with the user's Spotify account.
        debug_mode (bool, optional): If True, enables debug mode and saves the output to a test CSV file.
                                     Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing metadata of the liked songs, including:
                    - id: Index of the song.
                    - name: Name of the song.
                    - album: Name of the album.
                    - artists: Name of the primary artist.
                    - url: URL of the album image.
    Raises:
        Exception: If there is an issue retrieving tracks or saving the CSV file.
    Notes:
        - The function removes duplicate entries based on the album name.
        - Commas in the 'name' and 'album' fields are replaced with spaces to avoid CSV formatting issues.
        - If multiple artists are present for a track, only the first artist is currently included.
        - The CSV file is saved in the 'data/' directory with the name 'liked_tacks_playlist.csv' by default.
          If debug_mode is True, the file is saved as 'test_playlist.csv'.
    """
    
    print("retrieving liked tracks from Spotify...")
    tracks = get_tracks_list(sp_rqrmt=sp,
                                url_playlist="https://open.spotify.com/collection/tracks",
                                debug_mode=debug_mode)
    print('all liked tracks retrieved')

    #Extracting the metadata
    print("Extracting the metadata...")
    music_id_data = pd.DataFrame()
    for index, song in enumerate(tracks):
        if len(tracks[index]['track']["album"]["images"]) != 0:
            #TODO: check length to concat all artists         
            song_data = {
                'id': index,
                'name': song['track']['name'],
                'album': song['track']['album']['name'],
                'artists': song['track']['artists'][0]['name'],
                'url': song['track']["album"]["images"][1]['url']
            }
            music_id_data = music_id_data._append(song_data, ignore_index=True)

    music_id_data = music_id_data.drop_duplicates(subset=['album'], ignore_index = False)
    music_id_data['name'] = music_id_data.loc[:, 'name'].apply(lambda x: re.sub(',', ' ', x))
    music_id_data['album'] = music_id_data.loc[:, 'album'].apply(lambda x: re.sub(',', ' ', x))
    #apply sur artists aussi si ut prends pluierus artistes dansget tracks playlist
    # music_id_data['artists'] = music_id_data['artists'].apply(lambda x: re.sub(',', ' ', x))
    print('music metadata extracted')
    
    #  csv save to keep access to full information
    if DEBUG:
        music_id_data.to_csv('data/test_playlist.csv', index=False)
    else:
        music_id_data.to_csv('data/liked_tacks_playlist.csv', index=False)
    print("csv track list saved")

    return music_id_data


def extract_and_save_liked_music_covers_features(music_id_data=None, lite=False):
    """
    Extracts features from music cover images, saves them in an HDF5 file, and optionally saves metadata in a CSV file.
    This function processes a dataset of music cover images, extracts their features using a MobileNet-based model, 
    and stores the features in an HDF5 file. It also saves album names and other metadata for further use.

    Args:
        music_id_data (pd.DataFrame, optional): A DataFrame containing music metadata, including URLs of cover images 
            and album names. If None, the function will load a default dataset based on the DEBUG flag.
        lite (bool, optional): If True, extracts features without batching. If False, processes the images in batches 
            for efficiency. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - music_cover_features (np.ndarray): Extracted features of the music cover images.
            - album_names (pd.Series): A pandas Series containing the album names corresponding to the covers.
    """
    if music_id_data is None:
        if DEBUG:
            music_id_data = pd.read_csv("data/test_playlist.csv")
        else:
            music_id_data = pd.read_csv("data/liked_tacks_playlist.csv")
    
    #Extract features for all liked covers 
    print("Start features' calculation with MobileNet... ")
    urls = music_id_data['url']
    
    if lite:
        music_cover_features = extract_features(urls)
    else:
        batch_size = 32
        music_cover_features = extract_features(urls, batch_size=batch_size)

    print("features extraction ended")

    #hdf5 file creation
    if DEBUG:
        hdf5_file_name = "data/DEBUG_liked_tracks_cover_features.h5"
    else:
        hdf5_file_name = "data/liked_tracks_cover_features.h5"
        
    with h5py.File(hdf5_file_name, "w") as h5file:
        #Store features as datset
        h5file.create_dataset("features", data=music_cover_features, dtype="float32")
        
        #Store album name as a dataset
        h5file.create_dataset("album_names", data=music_id_data['album'], dtype=h5py.string_dtype(encoding="utf-8"))
    print("hdf5 file saved")
    #  csv save to keep access to all informations 
    if DEBUG:
        music_id_data.to_csv('data/test_playlist.csv', index=False)
    else:
        music_id_data.to_csv('data/liked_tacks_playlist.csv', index=False)
    print("csv file saved")
    
    return music_cover_features, music_id_data['album']


def extract_features(urls, lite=False, batch_size=32):
    """
    Extract features from a list of image URLs.

    Args:
        urls (list): List of image URLs.
        model (keras.Model, optional): Pretrained model for batch processing. Required if lite=False.
        lite (bool): Whether to use the lite mode (process one image at a time).
        batch_size (int): Batch size for batch processing. Ignored if lite=True.

    Returns:
        np.array: Extracted features.
    """
    features = []  # Store extracted features

    if lite:
        # Load pretrained MobileNet model in TFLite format
        interpreter, input_details, output_details = load_model(lite=lite)

        for i, url in enumerate(urls):
            try:
                # Download and preprocess image
                response = requests.get(url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img = img.resize((224, 224))  # Resize to target input size
                    img_array = image.img_to_array(img)
                    img_array = preprocess_input(img_array)
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                else:
                    print(f"Failed to download image from {url}")
                    img_array = np.zeros((224, 224, 3))  # Placeholder for failed downloads
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                img_array = np.zeros((224, 224, 3))  # Placeholder for errors

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array)
            # Run inference
            interpreter.invoke()
            # Get the output tensor (feature vector)
            feature = interpreter.get_tensor(output_details[0]['index'])
            features.append(feature.flatten())  # Add to results
    else:
        batch_images = []  # Temporary storage for batch
        model = load_model(lite=lite)
        for i, url in enumerate(urls):
            try:
                # Download and preprocess image
                response = requests.get(url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img = img.resize((224, 224))  # Resize to target input size
                    img_array = image.img_to_array(img)
                    img_array = preprocess_input(img_array)
                    batch_images.append(img_array)
                else:
                    print(f"Failed to download image from {url}")
                    batch_images.append(np.zeros((224, 224, 3)))  # Placeholder for failed downloads
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                batch_images.append(np.zeros((224, 224, 3)))  # Placeholder for errors

            # Process in batches
            if len(batch_images) == batch_size or i == len(urls) - 1:
                # Convert to NumPy array for batch prediction
                batch_images = np.array(batch_images)
                batch_features = model.predict(batch_images)  # Extract features in batch
                features.extend(batch_features)  # Add to results
                batch_images = []  # Reset batch

    return np.array(features)


