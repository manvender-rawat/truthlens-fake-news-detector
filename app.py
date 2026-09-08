import streamlit as st
import joblib
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px 10px 10px 10px;
    }

    .main-title {
        font-size: 45px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Cards */
    .card {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Result cards */
    .fake-result {
        background-color: #ffecec;
        border-left: 7px solid #e63946;
        padding: 22px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .real-result {
        background-color: #eaf8ef;
        border-left: 7px solid #2a9d5b;
        padding: 22px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
    }

    .result-text {
        font-size: 17px;
    }

    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.07);
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
    }

    .metric-label {
        color: #666666;
        font-size: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        height: 45px;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #fff8e1;
        border-left: 6px solid #f4b400;
        padding: 18px;
        border-radius: 10px;
        margin-top: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

# MODEL_PATH = "Model\\svm_model.pkl"
# VECTORIZER_PATH = "Model\\tfidf_vectorizer.pkl"
# DATA_PATH = "Data\\combined_data.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Model", "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "Model", "tfidf_vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Data", "combined_data.csv")

# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


try:
    svm_model, tfidf = load_model()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error("❌ Unable to load the trained model or TF-IDF vectorizer.")
    st.error(str(e))


# ============================================================
# LOAD STOPWORDS
# ============================================================

@st.cache_resource
def load_stopwords():

    try:
        return set(stopwords.words("english"))

    except LookupError:
        import nltk

        nltk.download("stopwords")

        return set(stopwords.words("english"))


stop_words = load_stopwords()

stemmer = PorterStemmer()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text)

    # Keep alphabetic characters only
    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    # Convert to lowercase
    text = text.lower()

    # Split into words
    text = text.split()

    # Remove stopwords + stemming
    text = [
        stemmer.stem(word)
        for word in text
        if word not in stop_words
    ]

    # Join words
    return " ".join(text)


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(decision_score):

    """
    LinearSVC does not directly provide probabilities.

    decision_function() gives a signed distance from the
    decision boundary.

    We convert that score into a probability-like confidence
    value using the sigmoid function.
    """

    confidence = 1 / (1 + np.exp(-abs(decision_score)))

    return confidence * 100


# ============================================================
# SESSION STATE - PREDICTION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📰 TruthLens")

    st.markdown(
        "### AI-Powered Fake News Detection"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📰 News Detection",
            "📊 Article Statistics",
            "🧠 Model Explanation",
            "📈 Visualizations",
            "🕒 Prediction History",
            "📚 How It Works",
            "⚠️ Disclaimer"
        ]
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("TF-IDF + Linear SVM")

    st.markdown("### 🏷️ Labels")

    st.write("0 → REAL")
    st.write("1 → FAKE")

    st.markdown("---")

    st.caption(
        "TruthLens is an academic project and "
        "should not be treated as a professional "
        "fact-checking system."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">

<div class="main-title">
📰 TruthLens
</div>

<div class="subtitle">
AI-Powered Fake News Detection System
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 1. NEWS DETECTION
# ============================================================

if page == "📰 News Detection":

    st.markdown(
        "## 📰 News Detection"
    )

    st.markdown(
        """
        Enter a news article below and TruthLens will analyze
        its textual patterns using a TF-IDF + Linear SVM machine
        learning model.
        """
    )

    st.markdown("---")

    # Input area

    news_text = st.text_area(
        "Paste News Article",
        height=280,
        placeholder=(
            "Paste the complete news article here..."
        )
    )

    # Character and word count

    if news_text:

        words = news_text.split()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Characters",
                len(news_text)
            )

        with col2:
            st.metric(
                "Words",
                len(words)
            )

        with col3:
            st.metric(
                "Sentences",
                len(re.findall(r"[.!?]+", news_text))
            )

    st.markdown("")

    predict_button = st.button(
        "🔍 Analyze News",
        use_container_width=True,
        type="primary"
    )

    if predict_button:

        if not news_text.strip():

            st.warning(
                "⚠️ Please enter a news article first."
            )

        elif not model_loaded:

            st.error(
                "Model could not be loaded."
            )

        else:

            with st.spinner(
                "Analyzing article..."
            ):

                # Preprocess

                cleaned_text = preprocess_text(
                    news_text
                )

                # TF-IDF transformation

                text_vector = tfidf.transform(
                    [cleaned_text]
                )

                # Prediction

                prediction = svm_model.predict(
                    text_vector
                )[0]

                # Decision score

                decision_score = svm_model.decision_function(
                    text_vector
                )[0]

                # Confidence

                confidence = calculate_confidence(
                    decision_score
                )

            # ------------------------------------------------
            # LABEL MAPPING
            # 0 = REAL
            # 1 = FAKE
            # ------------------------------------------------

            if prediction == 1:

                label = "FAKE"
                emoji = "🚨"

            else:

                label = "REAL"
                emoji = "✅"


            # =================================================
            # RESULT
            # =================================================

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="fake-result">

                    <div class="result-title">
                    🚨 FAKE NEWS
                    </div>

                    <div class="result-text">
                    The model classified this article as
                    <b>FAKE</b>.
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="real-result">

                    <div class="result-title">
                    ✅ REAL NEWS
                    </div>

                    <div class="result-text">
                    The model classified this article as
                    <b>REAL</b>.
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # CONFIDENCE
            # =================================================

            st.markdown("### 🎯 Confidence Score")

            st.progress(
                int(confidence)
            )

            st.write(
                f"**Model Confidence: {confidence:.2f}%**"
            )

            st.caption(
                "This is a model confidence estimate based "
                "on the SVM decision score, not a verified "
                "probability of factual truth."
            )


            # =================================================
            # MODEL SCORE
            # =================================================

            with st.expander(
                "🔬 View Model Decision Score"
            ):

                st.write(
                    f"Decision Score: `{decision_score:.4f}`"
                )

                if decision_score > 0:

                    st.write(
                        "Positive score → model leans toward FAKE."
                    )

                else:

                    st.write(
                        "Negative score → model leans toward REAL."
                    )


            # =================================================
            # PREDICTION HISTORY
            # =================================================

            st.session_state.history.append(
                {
                    "Article": news_text[:100] + (
                        "..." if len(news_text) > 100 else ""
                    ),
                    "Prediction": label,
                    "Confidence": round(
                        confidence,
                        2
                    )
                }
            )


            # =================================================
            # ARTICLE STATISTICS
            # =================================================

            st.markdown("---")

            st.markdown(
                "### 📊 Quick Article Statistics"
            )

            words = news_text.split()

            characters = len(news_text)

            sentences = len(
                re.findall(
                    r"[.!?]+",
                    news_text
                )
            )

            paragraphs = len(
                [p for p in news_text.split("\n") if p.strip()]
            )

            avg_word_length = (
                sum(len(word) for word in words)
                / len(words)
                if words
                else 0
            )

            stat1, stat2, stat3, stat4 = st.columns(4)

            with stat1:

                st.metric(
                    "Words",
                    len(words)
                )

            with stat2:

                st.metric(
                    "Characters",
                    characters
                )

            with stat3:

                st.metric(
                    "Sentences",
                    sentences
                )

            with stat4:

                st.metric(
                    "Avg Word Length",
                    f"{avg_word_length:.2f}"
                )


# ============================================================
# 2. ARTICLE STATISTICS
# ============================================================

elif page == "📊 Article Statistics":

    st.markdown(
        "## 📊 Article Statistics"
    )

    st.write(
        "Analyze basic textual characteristics of a news article."
    )

    article = st.text_area(
        "Enter article text",
        height=250
    )

    if article.strip():

        words = article.split()

        characters = len(article)

        sentences = len(
            re.findall(
                r"[.!?]+",
                article
            )
        )

        paragraphs = len(
            [p for p in article.split("\n") if p.strip()]
        )

        unique_words = len(
            set(
                word.lower()
                for word in words
            )
        )

        avg_word_length = (
            sum(
                len(word)
                for word in words
            ) / len(words)
            if words
            else 0
        )

        # ---------------------------------------------
        # METRICS
        # ---------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "📝 Words",
                len(words)
            )

        with c2:
            st.metric(
                "🔤 Characters",
                characters
            )

        with c3:
            st.metric(
                "📄 Sentences",
                sentences
            )

        with c4:
            st.metric(
                "📚 Paragraphs",
                paragraphs
            )


        c5, c6 = st.columns(2)

        with c5:

            st.metric(
                "🔠 Unique Words",
                unique_words
            )

        with c6:

            st.metric(
                "📏 Average Word Length",
                f"{avg_word_length:.2f}"
            )


        # ---------------------------------------------
        # WORD DISTRIBUTION
        # ---------------------------------------------

        st.markdown("---")

        st.markdown(
            "### 📈 Word Length Distribution"
        )

        word_lengths = [
            len(word)
            for word in words
        ]

        if word_lengths:

            fig, ax = plt.subplots()

            ax.hist(
                word_lengths,
                bins=range(
                    1,
                    max(word_lengths) + 2
                ),
                edgecolor="black"
            )

            ax.set_xlabel(
                "Word Length"
            )

            ax.set_ylabel(
                "Frequency"
            )

            ax.set_title(
                "Distribution of Word Lengths"
            )

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# 3. MODEL EXPLANATION
# ============================================================

elif page == "🧠 Model Explanation":

    st.markdown(
        "## 🧠 Model Explanation"
    )

    st.write(
        """
        TruthLens uses a **TF-IDF + Linear Support Vector Machine
        (Linear SVM)** architecture for text classification.
        """
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    st.markdown(
        "### 🔤 TF-IDF"
    )

    st.write(
        """
        TF-IDF (Term Frequency-Inverse Document Frequency)
        converts text into numerical features.

        It gives higher importance to words that are useful
        for distinguishing documents while reducing the
        importance of very common words.
        """
    )

    st.code(
        """
TF-IDF = Term Frequency × Inverse Document Frequency
        """,
        language="text"
    )


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Linear SVM"
    )

    st.write(
        """
        The Linear Support Vector Machine learns a decision
        boundary that separates REAL and FAKE news articles.

        Your model uses:

        • 0 → REAL
        • 1 → FAKE
        """
    )


    # --------------------------------------------------------
    # DECISION SCORE
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Decision Score"
    )

    st.write(
        """
        Linear SVM produces a decision score.

        A positive score indicates that the model leans toward
        FAKE, while a negative score indicates that it leans
        toward REAL.
        """
    )


    # --------------------------------------------------------
    # IMPORTANT FEATURES
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Important Learned Features"
    )

    try:

        feature_names = np.array(
            tfidf.get_feature_names_out()
        )

        coefficients = svm_model.coef_[0]

        # Positive = FAKE
        top_fake_indices = np.argsort(
            coefficients
        )[-15:][::-1]

        # Negative = REAL
        top_real_indices = np.argsort(
            coefficients
        )[:15]

        fake_features = feature_names[
            top_fake_indices
        ]

        real_features = feature_names[
            top_real_indices
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "#### 🚨 Strong FAKE-associated features"
            )

            for feature in fake_features:

                st.write(
                    f"• `{feature}`"
                )

        with col2:

            st.markdown(
                "#### ✅ Strong REAL-associated features"
            )

            for feature in real_features:

                st.write(
                    f"• `{feature}`"
                )

    except Exception as e:

        st.warning(
            "Unable to display model features."
        )

        st.caption(str(e))


    st.info(
        "Important: These features show patterns learned by "
        "the classifier. They do not prove that an article "
        "is factually true or false."
    )


# ============================================================
# 4. VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":

    st.markdown(
        "## 📈 Visualizations"
    )

    st.write(
        "Visual representation of your fake-news classification model."
    )


    # --------------------------------------------------------
    # MODEL FEATURE VISUALIZATION
    # --------------------------------------------------------

    try:

        feature_names = np.array(
            tfidf.get_feature_names_out()
        )

        coefficients = svm_model.coef_[0]


        # ---------------------------------------------
        # TOP FAKE FEATURES
        # ---------------------------------------------

        fake_indices = np.argsort(
            coefficients
        )[-10:][::-1]

        fake_features = feature_names[
            fake_indices
        ]

        fake_values = coefficients[
            fake_indices
        ]


        st.markdown(
            "### 🚨 Top FAKE-associated Features"
        )

        fig, ax = plt.subplots()

        ax.barh(
            fake_features[::-1],
            fake_values[::-1]
        )

        ax.set_xlabel(
            "SVM Coefficient"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Features Most Associated With FAKE"
        )

        st.pyplot(fig)

        plt.close(fig)


        # ---------------------------------------------
        # TOP REAL FEATURES
        # ---------------------------------------------

        real_indices = np.argsort(
            coefficients
        )[:10]

        real_features = feature_names[
            real_indices
        ]

        real_values = coefficients[
            real_indices
        ]


        st.markdown(
            "### ✅ Top REAL-associated Features"
        )

        fig, ax = plt.subplots()

        ax.barh(
            real_features[::-1],
            real_values[::-1]
        )

        ax.set_xlabel(
            "SVM Coefficient"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Features Most Associated With REAL"
        )

        st.pyplot(fig)

        plt.close(fig)


    except Exception as e:

        st.error(
            "Visualization could not be generated."
        )

        st.write(e)


# ============================================================
# 5. PREDICTION HISTORY
# ============================================================

elif page == "🕒 Prediction History":

    st.markdown(
        "## 🕒 Prediction History"
    )

    st.write(
        "Predictions made during the current application session."
    )


    if len(st.session_state.history) == 0:

        st.info(
            "No predictions have been made yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        total = len(history_df)

        fake_count = (
            history_df["Prediction"] == "FAKE"
        ).sum()

        real_count = (
            history_df["Prediction"] == "REAL"
        ).sum()

        avg_confidence = (
            history_df["Confidence"].mean()
        )


        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Predictions",
                total
            )

        with c2:
            st.metric(
                "🚨 FAKE",
                fake_count
            )

        with c3:
            st.metric(
                "✅ REAL",
                real_count
            )

        with c4:
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.2f}%"
            )


        st.markdown("---")


        # ---------------------------------------------
        # TABLE
        # ---------------------------------------------

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        # ---------------------------------------------
        # DOWNLOAD HISTORY
        # ---------------------------------------------

        csv = history_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv,
            file_name="truthlens_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ---------------------------------------------
        # CLEAR HISTORY
        # ---------------------------------------------

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# 6. HOW IT WORKS
# ============================================================

elif page == "📚 How It Works":

    st.markdown(
        "## 📚 How TruthLens Works"
    )

    st.write(
        """
        TruthLens uses a machine learning pipeline to classify
        news articles as REAL or FAKE.
        """
    )


    # STEP 1

    with st.expander(
        "1️⃣ Enter News Article",
        expanded=True
    ):

        st.write(
            """
            The user enters or pastes a news article into the
            application.
            """
        )


    # STEP 2

    with st.expander(
        "2️⃣ Text Preprocessing"
    ):

        st.write(
            """
            The article is cleaned before being given to the
            machine learning model.

            The preprocessing pipeline includes:

            • Removing non-alphabetic characters
            • Converting text to lowercase
            • Tokenizing words
            • Removing English stopwords
            • Applying Porter stemming
            """
        )


    # STEP 3

    with st.expander(
        "3️⃣ TF-IDF Vectorization"
    ):

        st.write(
            """
            The cleaned text is transformed into numerical
            TF-IDF features using the same vectorizer that was
            fitted during model training.
            """
        )


    # STEP 4

    with st.expander(
        "4️⃣ Linear SVM Classification"
    ):

        st.write(
            """
            The TF-IDF feature vector is passed to the trained
            Linear SVM classifier.

            The classifier predicts:

            0 → REAL

            1 → FAKE
            """
        )


    # STEP 5

    with st.expander(
        "5️⃣ Confidence Score"
    ):

        st.write(
            """
            The SVM decision score is converted into a
            probability-like confidence value for display.

            This value represents how strongly the classifier
            favors one class. It should not be interpreted as
            a guaranteed probability of factual correctness.
            """
        )


    # STEP 6

    with st.expander(
        "6️⃣ Result & Explanation"
    ):

        st.write(
            """
            The application displays:

            • REAL / FAKE prediction
            • Confidence score
            • SVM decision score
            • Article statistics
            • Important learned features
            """
        )


    # PIPELINE

    st.markdown("---")

    st.markdown(
        "### 🔄 Complete Pipeline"
    )

    st.code(
        """
News Article
     ↓
Text Preprocessing
     ↓
Clean Text
     ↓
TF-IDF Vectorization
     ↓
Linear SVM
     ↓
Decision Score
     ↓
REAL / FAKE
     ↓
Confidence + Explanation
        """,
        language="text"
    )


# ============================================================
# 7. FACT-CHECKING DISCLAIMER
# ============================================================

elif page == "⚠️ Disclaimer":

    st.markdown(
        "## ⚠️ Fact-Checking Disclaimer"
    )

    st.markdown(
        """
        <div class="disclaimer">

        <h3>⚠️ Important Notice</h3>

        <p>
        TruthLens is an academic machine learning project
        designed to classify news articles based on textual
        patterns learned from its training dataset.
        </p>

        <p>
        A <b>REAL</b> prediction does not guarantee that the
        information is factually correct, and a <b>FAKE</b>
        prediction does not by itself prove that the information
        is false.
        </p>

        <p>
        The system does not independently verify claims against
        authoritative sources, news agencies, government records,
        or fact-checking databases.
        </p>

        <p>
        Always verify important information using reliable and
        independent sources before accepting or sharing it.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "### 🎓 Project Information"
    )

    st.write(
        """
        **Project:** TruthLens – Fake News Detection System

        **Machine Learning:** Linear Support Vector Machine

        **Feature Extraction:** TF-IDF

        **Text Processing:** NLTK + Porter Stemmer

        **Frontend:** Streamlit

        **Classification:**

        • 0 = REAL

        • 1 = FAKE
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#777; padding:15px;">

    <b>TruthLens</b> | AI-Powered Fake News Detection

    <br>

    Built as a Machine Learning Academic Project

    </div>
    """,
    unsafe_allow_html=True
)
