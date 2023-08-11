"""Application entry point, global configuration, application structure"""

from config import app_config
import utils
import tab_about
import tab_writer
import streamlit as st
import os


def init():
    """setup app-wide configuration and download data only once"""
    utils.setup_app(app_config)

    ### Download model weights and preprocessed data from S3 if not downloaded already
    if not os.path.exists(app_config.model_weights) or not os.path.exists(
        app_config.processed_data
    ):
        utils.download_from_s3(
            source_s3_uri=app_config.s3_model_file_uri,
            target_file=app_config.model_weights,
        )
        utils.download_from_s3(
            source_s3_uri=app_config.s3_data_file_uri,
            target_file=app_config.processed_data,
        )

    ### setup app tab structure
    about, writer = utils.create_tabs(["ABOUT 👋", "SCRIPT WRITER 📝"])
    with about:
        tab_about.render()
    with writer:
        tab_writer.render()


### Application entry point
if __name__ == "__main__":
    init()
