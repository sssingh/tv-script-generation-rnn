"""All app-specific data and disk-IO related functionality implemented here"""

from config import app_config
import model
import streamlit as st
import torch
import pickle

SPECIAL_WORDS = {"PADDING": "<PAD>"}
pad_word = SPECIAL_WORDS["PADDING"]


@st.cache_resource
def load_preprocessed_data():
    """
    Load the Preprocessed Training data
    """
    with open(app_config.processed_data, mode="rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model():
    """
    Instantiate the model and load the model weights
    """
    _, vocab_to_int, _, _ = load_preprocessed_data()
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = vocab_size
    # Embedding Dimension
    embedding_dim = 300
    # Hidden Dimension
    hidden_dim = 512
    # Number of RNN Layers
    n_layers = 2
    # Instantiate the model class and populate trained weights
    rnn = model.RNN(
        vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5
    )
    st.spinner("Loading the trained model...")
    rnn.load_state_dict(torch.load(app_config.model_weights))
    return rnn
