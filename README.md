# covercrib

CoverCrib aims to provide the most similar music covers images among your spotify liked music regarding to an input image. 
Functionality:
- If a precomputed features file exists (`data/liked_tracks_cover_features.h5`), it loads the features and album names.
- If the features file does not exist, it connects to Spotify, extracts liked songs, and computes with mobileNet v2 features  for them.
- Normalizes the dataset features.
- Extracts and normalizes features for the external image provided via `img_path`.
- Computes cosine similarity between the external image features and the dataset features.
- Outputs a sorted DataFrame of album names and their similarity scores.


tf_lite can be used for a future integration to mobile app with kivy.
Personnal keys to connect to my spotify are not given here. For now it is necessary to create your owns with spotify developers and add it to an .env file like: <br>
SPOTIFY_CLIENT_SECRET="" <br>
SPOTIFY_CLIENT_ID="" <br>
SPOTIFY_REDIRECT_URI="http://localhost:8888/callback" <br>

Thus my liked songs are not given neither, because yours are automatically extracted when you connect to your spotify. 


TODO: 
- Add requirements.txt
- To be able to extract other playlist than just liked songs <br>
- Update .h5 features files if asked only with songs not already featurized <br>
- Add a function to display the input image and the top 5 similar albums <br>
- Switch to Kivy for a mobile GUI usage <br>

**Exemple:** <br>
Input: <br>
@hrs.wear <br>

![Input Image](data/input_image/flo-horus.png)  

Output:
| Album          | Similarity  |
|----------------|-------------|
| Gotham Ville   | 0.64898646  |
| Ma vision      | 0.64482456  |
| Dolo           | 0.644059    |
<br>
![Image](https://i.scdn.co/image/ab67616d00001e02463773d3c373be91f3c85b1f) <br>
![Image](https://i.scdn.co/image/ab67616d00001e0274bfdb7062f4b72399f91419) <br>
![Image](https://i.scdn.co/image/ab67616d00001e0275804cf0eac99faae98ddfe5) <br>



