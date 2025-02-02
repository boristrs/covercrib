from dash import Dash, html, dcc, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    style={"padding": "20px", "font-family": "Arial, sans-serif"},
    children=[
        # Title
        html.H1("COVERCRIB", style={"text-align": "center", "margin-bottom": "20px"}),

        # Connect to Spotify button
        html.Div(
            dbc.Button("Connect to Spotify", id="connect-spotify", color="primary"),
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
        
        #footer button
        html.Div(
            style={"margin-top": "20px", "display": "flex", "justify-content": "space-between"},
            children=[dbc.Button("Update song list", id="update-song-list", color="info"),
                    dbc.Button("Quit", id="quit", color="danger"),
                    ],
        ),     
    ],
)


@app.callback(
    Output("log-output", "children"),  # Target element to update
    [Input("connect-spotify", "n_clicks"),
     Input("find-match", "n_clicks"),
     Input("update-song-list", "n_clicks"),
     Input("quit", "n_clicks")],
)

def handle_buttons(spotipy_co_button, find_match_button, update_song_button, quit_button):
    triggered = ctx.triggered_id  # Detect which button was clicked
    if triggered == "connect-spotify":
        return print("Connecting to Spotify...")
    elif triggered == "find-match":
        return print("Finding matching covers...")
    elif triggered == "upload-picture":
        return print("upload picture button clicked...")
    elif triggered == "update-song-list":
        return print("Updating the song list...")
    elif triggered == "quit":
        return print("Quitting the application...")
    else:
        return print("No action performed yet!")









#Run the app 
if __name__ == "__main__":
    app.run_server(debug=True)