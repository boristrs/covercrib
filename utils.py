import pickle
import lmdb
import spotipy

# def save_img(image_list, directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     for img in image_list :
    

def store_image_lmbd(images_list, img_db_path):
    map_size = len(images_list) * 300 * 300 * 10 
    env = lmdb.open(img_db_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for i, img in enumerate(images_list):
            txn.put(f'image_{i}'.encode(), pickle.dumps(img))

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