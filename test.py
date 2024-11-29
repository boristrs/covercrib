import requests
from PIL import Image
from io import BytesIO

# URL de l'image
url = "https://i.scdn.co/image/ab67616d00001e0225ccd808fb6b2c07c6dcc0e3"

# Télécharger l'image depuis l'URL
response = requests.get(url)
if response.status_code == 200:  # Vérifie si la requête est réussie
    # Charger l'image avec Pillow
    image = Image.open(BytesIO(response.content))
    image.show()  # Affiche l'image dans un visualiseur (selon votre système)
    # Optionnel : Sauvegarder l'image localement
    image.save("spotify_image.jpg")
    print("Image téléchargée et sauvegardée sous 'spotify_image.jpg'.")
else:
    print(f"Erreur lors du téléchargement : {response.status_code}")
