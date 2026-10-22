# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Load saved files
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     model = load_model("lstm_model.h5")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     with open("max_len.pkl", "rb") as f:
#         max_len = pickle.load(f)
#     return model, tokenizer, max_len

# model, tokenizer, max_len = load_resources()

# # ------------------------------
# # Prediction function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

#     preds = model.predict(sequence, verbose=0)
#     predicted_index = np.argmax(preds)

#     for word, index in tokenizer.word_index.items():
#         if index == predicted_index:
#             return word
#     return ""

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.set_page_config(page_title="Next Word Prediction", layout="centered")

# st.title("🧠 Next Word Prediction (LSTM)")
# st.write("Enter a sentence and the model will predict the **next word**.")

# user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

# if st.button("Predict Next Word"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         next_word = predict_next_word(user_input)
#         st.success(f"**Predicted Next Word:** {next_word}")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.caption("LSTM-based Next Word Prediction using Streamlit")


import streamlit as st
import pickle
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="NextWord AI | LSTM Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 5%, #172554 0%, transparent 30%),
            radial-gradient(circle at 90% 15%, #312e81 0%, transparent 28%),
            #080d1b;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 38px 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #172554, #312e81, #4c1d95);
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 18px 55px rgba(0,0,0,0.25);
        margin-bottom: 28px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 30px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        color: #ddd6fe;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(30px, 5vw, 48px);
        font-weight: 800;
        line-height: 1.2;
        margin: 18px 0 12px;
    }

    .hero p {
        color: #ddd6fe;
        font-size: 16px;
        line-height: 1.8;
        max-width: 760px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #f8fafc;
        margin: 22px 0 8px;
    }

    .section-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 20px;
    }

    .panel {
        padding: 25px;
        background: rgba(15,23,42,0.85);
        border: 1px solid #263449;
        border-radius: 20px;
        margin-bottom: 22px;
    }

    .result-card {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        margin-top: 20px;
    }

    .result-label {
        color: #a7f3d0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .result-word {
        color: #ffffff;
        font-size: 40px;
        font-weight: 800;
        margin: 12px 0;
        overflow-wrap: anywhere;
    }

    .result-sentence {
        color: #d1fae5;
        font-size: 17px;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .feature-card {
        padding: 20px;
        min-height: 150px;
        background: rgba(15,23,42,0.8);
        border: 1px solid #263449;
        border-radius: 18px;
    }

    .feature-icon {
        font-size: 27px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #f8fafc;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 7px;
    }

    .feature-desc {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.7;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 15px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border: 1px solid #c4b5fd;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.25);
    }

    .footer {
        text-align: center;
        padding: 25px 10px;
        color: #64748b;
        font-size: 13px;
        border-top: 1px solid #263449;
        margin-top: 40px;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #263449;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .stTextInput label, .stTextArea label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background: #111b2d !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 1px #818cf8 !important;
    }

    .stAlert {
        border-radius: 12px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 25px 20px;
        }

        .result-word {
            font-size: 32px;
        }

        .panel {
            padding: 18px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL AND SAVED RESOURCES
# =========================================================
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "lstm_model.h5")
    tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
    max_len_path = os.path.join(base_dir, "max_len.pkl")

    required_files = {
        "Trained LSTM model": model_path,
        "Tokenizer": tokenizer_path,
        "Maximum sequence length": max_len_path
    }

    missing_files = [
        name for name, path in required_files.items()
        if not os.path.isfile(path)
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required files are missing: " + ", ".join(missing_files)
        )

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(max_len_path, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, int(max_len)


# =========================================================
# NEXT WORD PREDICTION
# =========================================================
def predict_next_word(text, model, tokenizer, max_len):
    text = text.strip().lower()

    if not text:
        return None, None

    sequences = tokenizer.texts_to_sequences([text])

    if not sequences or not sequences[0]:
        return None, None

    input_length = model.input_shape[1]

    if input_length is None:
        input_length = max_len - 1

    sequence = pad_sequences(
        sequences,
        maxlen=int(input_length),
        padding="pre",
        truncating="pre"
    )

    predictions = model.predict(sequence, verbose=0)[0]

    # Select the most probable predicted token.
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    # Build a reverse mapping from the fitted tokenizer.
    index_to_word = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    predicted_word = index_to_word.get(predicted_index)

    if predicted_word is None:
        return None, confidence

    return predicted_word, confidence


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 NextWord AI")
    st.caption("LSTM-powered text prediction")

    st.markdown("---")

    st.markdown("### 🚀 Navigation")
    st.markdown("🏠 Home & Prediction")
    st.markdown("📚 Prediction History")
    st.markdown("⚙️ Model Information")

    st.markdown("---")

    st.markdown("### 💡 How to use")
    st.write("1. Enter a sentence.")
    st.write("2. Click Predict Next Word.")
    st.write("3. View the model's prediction.")
    st.write("4. Generate more text by reusing the result.")

    st.markdown("---")
    st.caption("Built with Python, TensorFlow & Streamlit")


# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">
    <span class="hero-badge">✦ DEEP LEARNING PROJECT</span>
    <h1>Give Your Words<br>the Next Move. ✨</h1>
    <p>
        An intelligent next-word prediction application powered by
        Long Short-Term Memory (LSTM) neural networks.
        Enter your text and let your trained model predict what
        could come next.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD RESOURCES SAFELY
# =========================================================
try:
    with st.spinner("Loading your trained AI model..."):
        model, tokenizer, max_len = load_resources()

except Exception as error:
    st.error("Unable to load the AI model or its saved resources.")
    st.code(str(error))

    st.info(
        "Ensure that app.py, lstm_model.h5, tokenizer.pkl and "
        "max_len.pkl are in the same project folder."
    )

    st.stop()


# =========================================================
# MODEL STATUS
# =========================================================
status1, status2, status3 = st.columns(3)

with status1:
    st.metric("🧠 Model", "LSTM")

with status2:
    st.metric("📏 Sequence Length", max_len)

with status3:
    st.metric("📚 Vocabulary", f"{len(tokenizer.word_index):,}")


# =========================================================
# PREDICTION INTERFACE
# =========================================================
st.markdown(
    '<div class="section-title">✨ Try Your Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Start with a few words and discover the next predicted word.'
    '</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    user_input = st.text_area(
        "Your starting sentence",
        placeholder="Example: You will...",
        height=120,
        max_chars=2000,
        help="Enter words that match the language and style of your training dataset."
    )

    word_count = len(user_input.split())

    st.caption(f"📝 Words entered: {word_count}")

    col1, col2 = st.columns(2)

    with col1:
        predict_button = st.button(
            "✨ Predict Next Word",
            use_container_width=True
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear Input",
            use_container_width=True
        )

    if clear_button:
        st.session_state.prediction_result = None
        st.rerun()

    if predict_button:
        if not user_input.strip():
            st.warning("Please enter a sentence before predicting.")

        else:
            with st.spinner("Your LSTM model is thinking..."):
                try:
                    predicted_word, confidence = predict_next_word(
                        user_input,
                        model,
                        tokenizer,
                        max_len
                    )

                    if predicted_word:
                        full_sentence = (
                            user_input.strip() + " " + predicted_word
                        )

                        st.session_state.prediction_result = {
                            "input": user_input.strip(),
                            "word": predicted_word,
                            "sentence": full_sentence,
                            "confidence": confidence
                        }

                        st.session_state.prediction_history.insert(
                            0, st.session_state.prediction_result.copy()
                        )

                        st.session_state.prediction_history = (
                            st.session_state.prediction_history[:10]
                        )

                    else:
                        st.session_state.prediction_result = {
                            "error": (
                                "The model could not find a prediction. "
                                "Try different words from your training dataset."
                            )
                        }

                except Exception as error:
                    st.error("Prediction failed.")
                    st.code(str(error))


# =========================================================
# DISPLAY PREDICTION RESULT
# =========================================================
result = st.session_state.prediction_result

if result:
    if "error" in result:
        st.warning(result["error"])

    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">✦ Predicted Next Word</div>
            <div class="result-word">{result["word"]}</div>
            <div class="result-sentence">
                {result["sentence"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("### 📊 Prediction Details")

        detail1, detail2 = st.columns(2)

        with detail1:
            st.metric("Input Words", len(result["input"].split()))

        with detail2:
            st.metric("Predicted Word", result["word"])

        st.caption(
            "The displayed model score is the softmax probability for "
            "the selected token, if your model uses a softmax output. "
            "It is not a guarantee that the prediction is correct."
        )

        st.markdown("### 🚀 Continue the sentence")

        if st.button("Use predicted sentence as new input"):
            st.session_state["continued_text"] = result["sentence"]
            st.rerun()


# =========================================================
# CONTINUE PREDICTION INPUT
# =========================================================
if "continued_text" in st.session_state:
    st.info(
        "Your predicted sentence is ready. Copy it into the input box "
        "above to continue the prediction."
    )
    st.code(st.session_state["continued_text"])


# =========================================================
# PREDICTION HISTORY
# =========================================================
st.markdown("---")
st.markdown("## 📚 Recent Predictions")

if st.session_state.prediction_history:
    for i, item in enumerate(st.session_state.prediction_history, start=1):
        with st.expander(
            f"{i}. {item['input']} → {item['word']}"
        ):
            st.write("**Original input:**", item["input"])
            st.write("**Predicted word:**", item["word"])
            st.write("**Completed sentence:**", item["sentence"])

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.session_state.prediction_result = None
        st.rerun()

else:
    st.caption("Your recent predictions will appear here.")


# =========================================================
# FEATURES
# =========================================================
st.markdown("---")
st.markdown("## ⚡ What powers this app?")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">LSTM Neural Network</div>
        <div class="feature-desc">
            Uses a trained sequence model to predict a possible next word.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feature2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔤</div>
        <div class="feature-title">Text Tokenization</div>
        <div class="feature-desc">
            Converts input words into the numerical sequences expected
            by your trained model.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feature3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Real-time Inference</div>
        <div class="feature-desc">
            Runs your saved model to generate a prediction when requested.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <strong>NextWord AI</strong> · LSTM Next Word Prediction<br>
    Built with Python, TensorFlow, NumPy & Streamlit<br><br>
    A deep learning project for educational purposes.
</div>
""", unsafe_allow_html=True)

