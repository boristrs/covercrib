import pandas as pd
import os
from _sptf_auth import sp
import utils
import re
DEBUG = os.getenv("DEBUG") == "True"
if DEBUG:
    print("Mode débogage activé")

tracks = utils.get_tracks_list(sp_rqrmt=sp,
                               url_playlist="https://open.spotify.com/collection/tracks",
                               debug_mode=DEBUG)

# Fetching track data
#extracting the image cover
# music_img_data = {}
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
# filtered_index = music_id_data["index"].astype(str).tolist()
# music_img_data = {key: value for key, value in music_img_data.items() if key in filtered_index}

# for song_name in song_data:
#     if song_name
music_id_data['name'] = music_id_data.loc[:,'name'].apply(lambda x: re.sub(',', ' ', x))
music_id_data['album'] = music_id_data.loc[:,'album'].apply(lambda x: re.sub(',', ' ', x))
#apply sur artists aussi si ut prends pluierus artistes dansget tracks playlist
# music_id_data['artists'] = music_id_data['artists'].apply(lambda x: re.sub(',', ' ', x))
    
#  csv save for mobilenet 
music_id_data.to_csv('data/test_playlist.csv', index=False)


# store_img_lmbd(tracks, ) #add directory 


# Extracting song names and other details
# song_data = []
# for track in tracks:
#     track_name = track["track"]["name"]
#     artist_names = [artist["name"] for artist in track["track"]["artists"]]
#     song_data.append({"track_name": track_name, "artists": artist_names})

# # Displaying extracted song data
# for song in song_data:
#     print(f"Track: {song['track_name']}, Artists: {', '.join(song['artists'])}")
    


#idees: generer une ia par ia en funnction de la cover de l'album 
#  Ajouter d'autres parametres pour le matching telles que les émotions percues sur la photo avec celles de la musique 

