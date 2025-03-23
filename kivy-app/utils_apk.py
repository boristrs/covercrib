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
    
    
def load_model():
    """_summary_

    Returns:
        _type_: _description_
    """
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


def normalize_feature(features):
    """Function to normalize features

    Args:
        features (np.array): Features to normalize

    Returns:
        np.array: Normalized features
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features/norms


def get_image_features(img_path):
    """Function to extract image features using MobileNetV2 TFLite

    Args:
        img_path (str): Path to the image file

    Returns:
        np.array: Extracted features
    """
    # Load the TFLite model
    interpreter, input_details, output_details = load_model()
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0  # Normalize like MobileNetV2

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor (feature vector)
    features = interpreter.get_tensor(output_details[0]['index'])
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
    """_summary_

    Args:
        sp (_type_): _description_
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


def extract_and_save_liked_music_covers_features(music_id_data=None):
    
    if music_id_data is None:
        if DEBUG:
            music_id_data = pd.read_csv("data/test_playlist.csv")
        else:
            music_id_data = pd.read_csv("data/liked_tacks_playlist.csv")
    
    #Extract features for all liked covers 
    print("Start features' calculation with MobileNet... ")
    urls = music_id_data['url']
    music_cover_features = extract_all_songs_features(urls)
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


def extract_all_songs_features(urls, batch_size=32):
    features = []  # Store extracted features
    
    # Load pretrained MobileNet model
    interpreter, input_details, output_details = load_model()
    
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

    return np.array(features)




