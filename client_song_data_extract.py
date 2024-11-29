import pandas as pd
import os
from _sptf_auth import sp
import utils
import re
import numpy as np
import utils as ut
import h5py

DEBUG = os.getenv("DEBUG") == "True"
if DEBUG:
    print("Mode débogage activé")

#Retrieve liked tracks
tracks = utils.get_tracks_list(sp_rqrmt=sp,
                               url_playlist="https://open.spotify.com/collection/tracks",
                               debug_mode=DEBUG)

#Extracting the image cover
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


#Load pretrained MobileNet model
model = ut.load_model()
#Extract features for all dataset images 
music_cover_features = []
for img_url in music_id_data['url']:
    try:
        features = ut.extract_features(ut.get_img(img_url), model)
        music_cover_features.append(features)
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        # You could also choose to append a default value to maintain the index
        music_cover_features.append([])
        

#cannot append empty list if dl interrupted
#hdf5 file creation
with h5py.File("data/liked_tracks_cover_features.h5", "w") as h5file:
    #Store features as datset
    h5file.create_dataset("features", data=music_cover_features, dtype="float32")
    
    #Store album name as a dataset
    h5file.create_dataset("album_names", data=music_id_data['album'], dtype=h5py.string_dtype(encoding="utf-8"))

#  csv save to double check indexation
music_id_data.to_csv('data/test_playlist.csv', index=False)



#idees: generer une ia par ia en funnction de la cover de l'album 
#  Ajouter d'autres parametres pour le matching telles que les émotions percues sur la photo avec celles de la musique 

