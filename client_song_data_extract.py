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
print("retrieving liked tracks from Spotify...")
tracks = ut.get_tracks_list(sp_rqrmt=sp,
                               url_playlist="https://open.spotify.com/collection/tracks",
                               debug_mode=DEBUG)
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
music_id_data['name'] = music_id_data.loc[:,'name'].apply(lambda x: re.sub(',', ' ', x))
music_id_data['album'] = music_id_data.loc[:,'album'].apply(lambda x: re.sub(',', ' ', x))
#apply sur artists aussi si ut prends pluierus artistes dansget tracks playlist
# music_id_data['artists'] = music_id_data['artists'].apply(lambda x: re.sub(',', ' ', x))
print('music metadata extracted')


#Load pretrained MobileNet model
model = ut.load_model()
#Extract features for all dataset images 
print("Start features' calculation with MobileNet... ")
urls = music_id_data['url']
batch_size = 32
music_cover_features = ut.batch_extract_features(urls, model, batch_size=batch_size)
# music_cover_features = []
# for img_url in music_id_data['url']:
#     try:
#         features = ut.extract_features(ut.get_img(img_url), model)
#         music_cover_features.append(features)
#     except Exception as e:
#         print(f"Error processing image {img_url}: {e}")
#         # You could also choose to append a default value to maintain the index
#         music_cover_features.append([])
print("features extraction ended")

#cannot append empty list if dl interrupted
#hdf5 file creation
if DEBUG:
    hdf5_file_name = "data/DEBUG_liked_tracks_cover_features.h5"
else:
     hdf5_file_name = "data/liked_tracks_cover_features.h5"
     
with h5py.File("data/liked_tracks_cover_features.h5", "w") as h5file:
    #Store features as datset
    h5file.create_dataset("features", data=music_cover_features, dtype="float32")
    
    #Store album name as a dataset
    h5file.create_dataset("album_names", data=music_id_data['album'], dtype=h5py.string_dtype(encoding="utf-8"))
print("hdf5 file saved")
#  csv save to double check indexation
if DEBUG:
    music_id_data.to_csv('data/test_playlist.csv', index=False)
else:
     music_id_data.to_csv('data/liked_tacks_playlist.csv', index=False)
print("csv file saved")




#idees: generer une ia par ia en funnction de la cover de l'album 
#  Ajouter d'autres parametres pour le matching telles que les émotions percues sur la photo avec celles de la musique 

