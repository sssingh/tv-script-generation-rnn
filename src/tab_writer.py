"""About tab rendering functionality"""

from config import app_config
import utils
import model
import data
import json
import streamlit as st


###
### INTERNAL FUNCTIONS
###
def __get_model():
    """Instantiate and cache the downloaded model and preprocessed data"""
    _, vocab_to_int, int_to_vocab, token_dict = data.load_preprocessed_data()
    trained_rnn = data.load_model()
    return trained_rnn, vocab_to_int, int_to_vocab, token_dict


def __generate_script(rnn, vocab_to_int, int_to_vocab, token_dict):
    """Read user inputs and generate the script"""
    help_text = (
        "prime word could be any word in dictionary (i.e any word present "
        + "in the original Seinfeld script), but it's best to start with a name:  "
        + "***kessler, jerry, george, claire, kramer, laura, elaine, pamela, "
        + "vanessa, roger***. Vocabulary is available for download below"
    )
    prime_word = st.text_input("Enter a prime word:", help=help_text).strip()
    vocab = json.dumps(vocab_to_int)
    utils.download_file(
        btn_label="Download Vocabulary",
        data=vocab,
        file_name="vocab.json",
        mime_type="application/json",
    )
    gen_length = int(
        st.slider(
            "Choose Script Length:", min_value=100, max_value=500, value=200, step=50
        )
    )
    if st.button("**Generate Script**"):
        ### ensure that prime-word is valid and is present in vocabulary
        if len(prime_word) > 0 and vocab_to_int.get(prime_word, False):
            st.spinner("Processing...")
            script = model.generate_script(
                rnn,
                vocab_to_int[prime_word + ":"],
                int_to_vocab,
                token_dict,
                vocab_to_int[data.pad_word],
                gen_length,
            )
            st.divider()
            st.subheader("Generated Script:")
            st.write(script)
            st.divider()
        else:
            st.error("Please provide a valid prime-word", icon=app_config.icon_stop)


def __section(header):
    """Render the section on this page"""
    st.header(header)
    rnn, vocab_to_int, int_to_vocab, token_dict = __get_model()
    __generate_script(rnn, vocab_to_int, int_to_vocab, token_dict)


###
### MAIN FLOW, entry point
###
def render():
    __section("Generate Script")
