from dash import Dash, html, dcc, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

import pandas as pd 
from pathlib import Path

import client_song_data_extract
import _sptf_auth

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    style={"padding": "20px", "font-family": "Arial, sans-serif"},
    children=[
        # Title
        html.H1("COVERCRIB", style={"text-align": "center", "margin-bottom": "20px"}),

        # Connect to Spotify button
        html.Div([
            # "Click if you want to connect to spotify and update your list of saved tracks",
            dbc.Button("Connect to Spotify", id="connect-spotify", color="primary")],
            style={"margin-bottom": "20px"}
        ),

        # Main interface
        html.Div(
            style={"display": "flex", "justify-content": "space-between"},
            children=[
                # Upload picture section
                html.Div(
                    style={
                        "border": "1px solid black",
                        "width": "40%",
                        "height": "300px",
                        "display": "flex",
                        "align-items": "center",
                        "justify-content": "center",
                    },
                    children=[
                        dcc.Upload(
                            id="upload-picture",
                            children=html.Div(["Upload picture"]),
                            style={
                                "color": "red", 
                                "textAlign": "center",
                                "cursor": "pointer",
                                "border": "2px dashed black",
                                "padding":"50px",
                                "width": "100%",
                            },
                        )
                    ],
                ),
                #Find match button
                html.Div(
                    dbc.Button("Find Match", id="find-match", color="success"),
                    style={"align-self": "center", "margin": "0 20px"},
                ),
                #List of Matching Covers section
                html.Div(
                    style={
                        "border": "1px solid black",
                        "width": "50%",
                        "padding": "10px",
                    },
                    children=[
                        html.H4("List of matching covers", style={"text-align": "center"}),
                        dash_table.DataTable(
                            id="matching-covers-table",
                            columns=[
                                {"name": "Cover", "id": "cover"},
                                {"name": "Album name", "id": "album_name"},
                                {"name": "Similarity score", "id": "similarity_score"},
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "5px"},
                            style_header={"backgroundColor": "lightgrey",
                                          "fontWeight": "bold",
                            },
                        ),
                    ],
                ),
            ],
        ),
        
        #log div
        html.Div(children=None,
                 id="log-output"),
        
        dcc.Store(id="spotify-client-store"),
        #footer button
        html.Div(
            style={"margin-top": "20px", "display": "flex", "justify-content": "space-between"},
            children=[dbc.Button("Update song list", id="update-song-list", color="info"),
                    dbc.Button("Quit", id="quit", color="danger"),
                    ],
        ),     
    ],
)









# @app.callback(
#     Output("matching-covers-table", "data"),  # Specify the component and property
#     Input("find-match", "n_clicks"),  # The trigger input
#     prevent_initial_call=True
# )





@app.callback(
    [Output("log-output", "children"),
    Output("spotify-client-store", "data")],
    [Input("connect-spotify", "n_clicks"),
     Input("find-match", "n_clicks"),
     Input("update-song-list", "n_clicks"),
     Input("quit", "n_clicks")],
    [State("spotify-client-store", "data")]
)

def handle_buttons(connect_clicks, match_clicks, update_clicks, quit_clicks, spotify_client_data):
    """Handle actions to trigger in function of the clicked button 

    Args:
        connect_clicks (integer): n_clicks component property for 'connect-spotify' input
        match_clicks (integer): n_clicks component property for 'find-match' input
        update_clicks (integer): n_clicks component property for 'update-song-list' input
        quit_clicks (integer): n_clicks component property for 'quit' input

    Returns:
        (list): message for button status, spotify client manager
    """
    # ctx = ctx
    if not ctx.triggered:
        return 'No action performed yet!', None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "connect-spotify":
        print("Connecting to Spotify...")
        sp = _sptf_auth.connect_to_spotify()
        access_token = sp.auth_manager.get_access_token(as_dict=False)

        return "Connected to Spotify!", {"access_token": access_token}
    elif button_id == "find-match":
        return "Finding matching covers...", Dash.no_update
    elif button_id == "upload-picture":
        return "upload picture button clicked...", Dash.no_update
    elif button_id == "update-song-list":
        return "Updating the song list...", Dash.no_update
    elif button_id == "quit":
        return "Quitting the application...", Dash.no_update
    else:
        return "No action performed yet!", Dash.no_update




def use_of_spotify_client(sp):
    
    if sp:
        print("Spotify client ready to use")
        current_liked_track_list = extract_liked_song(sp) 
        return current_liked_track_list
    else:
        return None


def update_matching_covers(n_clicks):
    if not n_clicks:
        return dash.no_update  # Prevent updates if not clicked
    # Example: Return an empty list or fetched data
    return []

#mettre une condition dans find match pour savoir si on utilise la variable locale ou le fichier csv pour trouver le match 
# en fonction de si on vuet mettre à jour la liste ou pas 
#Après s'être connecté à sptify, les derniers sons sont ajoutés en arriere plan, une comparaison avec la liste saved est faite pour annoncer combien de nouveaux sons sont apparus;
#Un radio button pour choisir si on veut mettre à jour la liste ou pas au moment ou on clique sur find match
        # #root and path of savved tracks list
        # # to save in constant file 
        # file_name = "liked_tracks_playlist.csv"
        # root_folder = "data"
        
        # file_path = next(root_folder.rglob(file_name), None)
        # #if #liked tracks are same or not


#Run the app 
if __name__ == "__main__":
    app.run_server(debug=True)