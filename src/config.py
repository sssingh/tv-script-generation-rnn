"""All app-specific user defined configurations are defined here"""

import os
from dataclasses import dataclass

# import plotly.express as px


### define all plotting configuration here,
### should not be accessed and changed directly hence leading "__"
@dataclass
class __PlotConfig:
    """All plotting configurations are defined here"""

    # Available themes (templates):
    # ['ggplot2', 'seaborn', 'simple_white', 'plotly',
    #  'plotly_white', 'plotly_dark', 'presentation',
    #  'xgridoff', 'ygridoff', 'gridon', 'none']
    # theme = "plotly_dark"
    # cat_color_map = px.colors.qualitative.T10
    # cat_color_map_r = px.colors.qualitative.T10_r
    # cont_color_map = px.colors.sequential.amp
    # cont_color_map_r = px.colors.sequential.amp_r


### define all app-wide configuration here,
### should not be accessed and changed directly hence leading "__"
@dataclass
class __AppConfig:
    """app-wide configurations"""

    # get current working directory
    cwd = os.getcwd()
    banner_image = f"{cwd}/"
    logo_image = f"{cwd}/assets/logo.svg"
    app_title = "Script Writer"
    app_icon = f"{cwd}/assets/title.png"
    app_short_desc = "Generate TV Script via AI"
    md_about = f"{cwd}/artifacts/about.md"
    processed_data = f"{cwd}/artifacts/preprocess.p"
    model_weights = f"{cwd}/artifacts/rnn_model_weights.pt"
    s3_model_file_uri = "s3://tv-script-generation/artifacts/rnn_model_weights.pt"
    s3_data_file_uri = "s3://tv-script-generation/artifacts/preprocess.p"
    sidebar_state = "expanded"  # collapsed
    layout = "centered"  # wide
    icon_question = "❓"
    icon_important = "🎯"
    icon_info = "ℹ️"
    icon_stop = "⛔"
    icon_about = "👋"


### make configs available to any module that imports this module
app_config = __AppConfig()
