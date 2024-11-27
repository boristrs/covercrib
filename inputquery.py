import pandas as pd 

DEBUG = os.getenv("DEBUG") == "True"

if DEBUG:
    print("Mode débogage activé")
    
print(os.getcwd())

#Est ce que l'user est le meme
    # si oui on recupere juste les infos des sp et data liked 
    # sinon relance les .py


local_img_path = 'C:\Users\torresdb\OneDrive - MANE\Desktop\perso project\matching_rap_cover\data\I_basel.jpg'
input_img_data = pd.DataFrame(data={'index': ['a'], 'url': [local_img_path]})

input_img_data.to_csv('data/test_i_basel.csv', index = False)