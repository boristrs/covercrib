import spotipy
import os
import pandas as pd
import utils
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from dotenv import load_dotenv


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

    DEBUG = os.getenv("DEBUG") == "True"
    if DEBUG:
        print("Mode débogage activé")

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
    
    # creer une classe par utilisateur avec ses infos dedans ses tracks etc ? 
    # Non car qu'un utilisateur à la fois ? 