import spotipy
import os
import pandas as pd
# from PIL import Image
import utils
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()
DEBUG = os.getenv("DEBUG") == "True"

if DEBUG:
    print("Mode débogage activé")

# CLIENT_ID=44405bf3c1dc428fa65da245e70ed8d9
# CLIENT_SECRET=60678404c8e140c7912bee97c2334ded

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

# client_credential_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
# sp = spotipy.Spotify(client_credentials_manager=client_credential_manager)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope='user-library-read'))

# creer une classe par utilisateur avec ses infos dedans ses tracks etc ? 
