"""
YouTube Comment Analyzer — Streamlit Dashboard
Uses spaCy for POS tagging, lemmatization, and NER.
Uses TextBlob for sentiment analysis.
Uses scikit-learn for ML classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from textblob import TextBlob
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Comment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, p, h1, h2, h3, h4, h5, h6, span, div, input, textarea, button, label, a {
        font-family: 'Inter', sans-serif !important;
    }

    /* Preserve Material Symbols font for Streamlit icons */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .e1nzilvr5 {
        font-family: 'Material Symbols Rounded' !important;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        text-align: center;
        color: #9ca3af;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102,126,234,0.2);
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102,126,234,0.3);
        color: #e2e8f0;
    }

    .pos-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
        color: white;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid rgba(102,126,234,0.3);
        padding: 0.75rem 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        overflow: hidden;
    }

    .comment-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load spaCy model (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def get_video_id(url: str):
    """Extract video ID from various YouTube URL formats."""
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None


@st.cache_data(show_spinner=False)
def fetch_comments(api_key: str, video_id: str, max_comments: int = 5000):
    """Fetch comments from YouTube Data API v3 with pagination."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
    )
    response = request.execute()

    while request and len(comments) < max_comments:
        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
        if "nextPageToken" in response and len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
            )
            response = request.execute()
        else:
            break
    return comments[:max_comments]


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove URLs, keep only alpha chars."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # remove HTML entities
    text = re.sub(r"&#\d+;", " ", text)  # remove numeric HTML entities
    text = re.sub("[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spacy_process(text: str):
    """Use spaCy for lemmatization, stopword removal, POS tagging, and NER."""
    doc = nlp(text)
    tokens_info = []
    lemmas = []
    for token in doc:
        tokens_info.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "is_stop": token.is_stop,
        })
        if not token.is_stop and not token.is_punct and len(token.text) > 1:
            lemmas.append(token.lemma_)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return " ".join(lemmas), tokens_info, entities


def get_sentiment(text: str):
    """Classify sentiment using TextBlob polarity."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"


# POS tag color mapping
POS_COLORS = {
    "NOUN": "#6366f1",
    "VERB": "#f43f5e",
    "ADJ": "#10b981",
    "ADV": "#f59e0b",
    "PROPN": "#8b5cf6",
    "DET": "#64748b",
    "ADP": "#06b6d4",
    "PRON": "#ec4899",
    "AUX": "#ef4444",
    "CCONJ": "#78716c",
    "SCONJ": "#a3a3a3",
    "PART": "#94a3b8",
    "INTJ": "#fbbf24",
    "NUM": "#14b8a6",
    "PUNCT": "#4b5563",
    "SYM": "#71717a",
    "X": "#9ca3af",
    "SPACE": "#d1d5db",
}


def render_pos_tags(tokens_info: list) -> str:
    """Render tokens as colored POS tag badges."""
    html = ""
    for t in tokens_info:
        color = POS_COLORS.get(t["pos"], "#6b7280")
        html += f'<span class="pos-tag" style="background:{color};" title="{t["tag"]}">{t["text"]} <small>({t["pos"]})</small></span> '
    return html


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Removed API Key input as per user request to store it in secrets
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        api_key = None

    youtube_url = st.text_input(
        "🔗 YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    max_comments = st.slider(
        "📊 Max Comments to Fetch",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
    )

    analyze_btn = st.button("🚀 Analyze Comments", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This dashboard performs **NLP analysis** on YouTube comments using:
    - **spaCy** — POS tagging, lemmatization, NER
    - **TextBlob** — Sentiment analysis
    - **scikit-learn** — ML classification
    - **Plotly** — Interactive visualizations
    """)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🎬 YouTube Comment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">NLP-powered sentiment analysis &amp; POS tagging with spaCy</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────
if analyze_btn:
    if not api_key:
        st.error("❌ YouTube API Key not found in Streamlit secrets.")
        st.info("To fix this, please add your API key to `.streamlit/secrets.toml` locally or in the Streamlit Cloud settings:\n\n`YOUTUBE_API_KEY = \"YOUR_KEY_HERE\"`")
        st.stop()
    if not youtube_url:
        st.error("❌ Please enter a YouTube Video URL in the sidebar.")
        st.stop()

    video_id = get_video_id(youtube_url)
    if not video_id:
        st.error("❌ Could not extract video ID. Please check the URL format.")
        st.stop()

    # ── Fetch comments ──────────────────────────────────────
    with st.spinner("📥 Fetching comments from YouTube..."):
        try:
            raw_comments = fetch_comments(api_key, video_id, max_comments)
        except Exception as e:
            st.error(f"❌ Failed to fetch comments: {e}")
            st.stop()

    if not raw_comments:
        st.warning("⚠️ No comments found for this video.")
        st.stop()

    df = pd.DataFrame(raw_comments, columns=["comment"])

    # ── Text preprocessing & spaCy processing ───────────────
    progress = st.progress(0, text="🔄 Processing comments with spaCy...")
    df["clean_comment"] = df["comment"].apply(clean_text)

    processed_texts = []
    all_tokens_info = []
    all_entities = []
    total = len(df)

    for i, text in enumerate(df["clean_comment"]):
        lemmatized, tokens_info, entities = spacy_process(text)
        processed_texts.append(lemmatized)
        all_tokens_info.append(tokens_info)
        all_entities.append(entities)
        if i % max(1, total // 20) == 0:
            progress.progress(min((i + 1) / total, 1.0), text=f"🔄 Processing comment {i+1}/{total}...")

    df["processed_text"] = processed_texts
    df["tokens_info"] = all_tokens_info
    df["entities"] = all_entities

    # ── Sentiment analysis ──────────────────────────────────
    progress.progress(0.85, text="💭 Analyzing sentiment...")
    df["sentiment"] = df["clean_comment"].apply(get_sentiment)
    df["polarity"] = df["clean_comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["subjectivity"] = df["clean_comment"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # ── POS tag aggregation ─────────────────────────────────
    progress.progress(0.90, text="🏷️ Aggregating POS tags...")
    all_pos_tags = []
    all_words_by_pos = {}
    for info_list in all_tokens_info:
        for t in info_list:
            if not t["is_stop"] and len(t["text"]) > 1:
                all_pos_tags.append(t["pos"])
                if t["pos"] not in all_words_by_pos:
                    all_words_by_pos[t["pos"]] = []
                all_words_by_pos[t["pos"]].append(t["lemma"])

    pos_counts = Counter(all_pos_tags)

    # ── ML model training ───────────────────────────────────
    progress.progress(0.95, text="🤖 Training ML model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["processed_text"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    progress.progress(1.0, text="✅ Analysis complete!")
    progress.empty()

    # Store in session state
    st.session_state["df"] = df
    st.session_state["accuracy"] = accuracy
    st.session_state["report"] = report
    st.session_state["pos_counts"] = pos_counts
    st.session_state["all_words_by_pos"] = all_words_by_pos
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    st.session_state["all_entities"] = all_entities
    st.session_state["all_tokens_info"] = all_tokens_info


# ─────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]
    accuracy = st.session_state["accuracy"]
    report = st.session_state["report"]
    pos_counts = st.session_state["pos_counts"]
    all_words_by_pos = st.session_state["all_words_by_pos"]
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    all_entities = st.session_state["all_entities"]
    all_tokens_info = st.session_state["all_tokens_info"]

    # ── Metric cards ────────────────────────────────────────
    cols = st.columns(4)
    metrics = [
        ("Total Comments", f"{len(df):,}"),
        ("Model Accuracy", f"{accuracy:.1%}"),
        ("Avg Polarity", f"{df['polarity'].mean():.3f}"),
        ("Avg Subjectivity", f"{df['subjectivity'].mean():.3f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Sentiment Analysis",
        "🏷️ POS Tagging",
        "🔍 Named Entities",
        "🤖 ML Metrics",
        "💬 Comment Explorer",
    ])

    # ── Tab 1: Sentiment ────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            sent_counts = df["sentiment"].value_counts()
            fig_pie = px.pie(
                names=sent_counts.index,
                values=sent_counts.values,
                color=sent_counts.index,
                color_discrete_map={
                    "Positive": "#10b981",
                    "Neutral": "#6366f1",
                    "Negative": "#f43f5e",
                },
                hole=0.45,
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                legend=dict(orientation="h", y=-0.1),
                margin=dict(t=20, b=20),
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont_size=13,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = px.bar(
                x=sent_counts.index,
                y=sent_counts.values,
                color=sent_counts.index,
                color_discrete_map={
                    "Positive": "#10b981",
                    "Neutral": "#6366f1",
                    "Negative": "#f43f5e",
                },
                labels={"x": "Sentiment", "y": "Count"},
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                showlegend=False,
                margin=dict(t=20, b=20),
                xaxis=dict(title=""),
                yaxis=dict(title="Number of Comments"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Polarity & Subjectivity scatter
        st.markdown('<div class="section-title">Polarity vs Subjectivity</div>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            df,
            x="polarity",
            y="subjectivity",
            color="sentiment",
            color_discrete_map={
                "Positive": "#10b981",
                "Neutral": "#6366f1",
                "Negative": "#f43f5e",
            },
            opacity=0.5,
            labels={"polarity": "Polarity", "subjectivity": "Subjectivity"},
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            margin=dict(t=20),
            height=450,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Polarity distribution histogram
        st.markdown('<div class="section-title">Polarity Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df,
            x="polarity",
            nbins=50,
            color="sentiment",
            color_discrete_map={
                "Positive": "#10b981",
                "Neutral": "#6366f1",
                "Negative": "#f43f5e",
            },
            barmode="overlay",
            opacity=0.7,
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            margin=dict(t=20),
            xaxis_title="Polarity Score",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Word clouds
        st.markdown('<div class="section-title">Word Clouds by Sentiment</div>', unsafe_allow_html=True)
        wc_cols = st.columns(3)
        sentiments = ["Positive", "Neutral", "Negative"]
        wc_colors = ["Greens", "Purples", "Reds"]

        for col_wc, sent, cmap in zip(wc_cols, sentiments, wc_colors):
            with col_wc:
                st.markdown(f"**{sent}**")
                text = " ".join(df[df["sentiment"] == sent]["processed_text"].dropna())
                if text.strip():
                    wc = WordCloud(
                        width=400,
                        height=300,
                        background_color="rgba(0,0,0,0)",
                        mode="RGBA",
                        colormap=cmap,
                        max_words=80,
                    ).generate(text)
                    fig_wc, ax_wc = plt.subplots(figsize=(5, 3.75))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    fig_wc.patch.set_alpha(0)
                    st.pyplot(fig_wc, use_container_width=True)
                    plt.close(fig_wc)
                else:
                    st.info("No text available")

    # ── Tab 2: POS Tagging ──────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Part-of-Speech Distribution (spaCy)</div>', unsafe_allow_html=True)

        # POS distribution bar chart
        pos_df = pd.DataFrame(
            sorted(pos_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["POS", "Count"],
        )

        fig_pos = px.bar(
            pos_df,
            x="POS",
            y="Count",
            color="POS",
            color_discrete_map=POS_COLORS,
        )
        fig_pos.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            showlegend=False,
            margin=dict(t=20),
            xaxis_title="Part of Speech",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_pos, use_container_width=True)

        # Top words by POS
        st.markdown('<div class="section-title">Top Words by POS Category</div>', unsafe_allow_html=True)

        pos_selection = st.multiselect(
            "Select POS tags to explore",
            options=sorted(all_words_by_pos.keys()),
            default=["NOUN", "VERB", "ADJ"],
        )

        if pos_selection:
            pos_cols = st.columns(len(pos_selection))
            for col_pos, pos_tag in zip(pos_cols, pos_selection):
                with col_pos:
                    color = POS_COLORS.get(pos_tag, "#6b7280")
                    st.markdown(f'<span class="pos-tag" style="background:{color}; font-size: 1rem; padding: 6px 16px;">{pos_tag}</span>', unsafe_allow_html=True)
                    word_freq = Counter(all_words_by_pos.get(pos_tag, []))
                    top_words = word_freq.most_common(15)
                    if top_words:
                        tw_df = pd.DataFrame(top_words, columns=["Word", "Count"])
                        fig_tw = px.bar(
                            tw_df,
                            x="Count",
                            y="Word",
                            orientation="h",
                            color_discrete_sequence=[color],
                        )
                        fig_tw.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", size=11),
                            showlegend=False,
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=400,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_tw, use_container_width=True)

        # Interactive POS tagger
        st.markdown('<div class="section-title">🔬 Interactive POS Tagger</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            "Enter text to analyze POS tags:",
            value="This video explains machine learning concepts beautifully.",
            height=80,
        )
        if user_text:
            doc = nlp(user_text)
            tokens = [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_, "is_stop": t.is_stop} for t in doc]
            html_tags = render_pos_tags(tokens)
            st.markdown(html_tags, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            tag_df = pd.DataFrame(tokens)
            st.dataframe(tag_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Named Entities ───────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Named Entity Recognition (spaCy NER)</div>', unsafe_allow_html=True)

        flat_entities = []
        for ent_list in all_entities:
            flat_entities.extend(ent_list)

        if flat_entities:
            ent_df = pd.DataFrame(flat_entities, columns=["Entity", "Label"])
            ent_counts = ent_df["Label"].value_counts()

            col_e1, col_e2 = st.columns(2)

            with col_e1:
                fig_ent = px.bar(
                    x=ent_counts.index,
                    y=ent_counts.values,
                    labels={"x": "Entity Type", "y": "Count"},
                    color=ent_counts.index,
                )
                fig_ent.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    showlegend=False,
                    margin=dict(t=20),
                )
                st.plotly_chart(fig_ent, use_container_width=True)

            with col_e2:
                # Top entities
                top_entities = ent_df.groupby(["Entity", "Label"]).size().reset_index(name="Count")
                top_entities = top_entities.sort_values("Count", ascending=False).head(20)
                fig_top_ent = px.bar(
                    top_entities,
                    x="Count",
                    y="Entity",
                    color="Label",
                    orientation="h",
                )
                fig_top_ent.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    margin=dict(t=20),
                    height=500,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_top_ent, use_container_width=True)
        else:
            st.info("No named entities detected in the comments.")

    # ── Tab 4: ML Metrics ───────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            # Classification report table
            report_df = pd.DataFrame(report).T
            report_df = report_df.round(3)
            st.dataframe(report_df, use_container_width=True)

        with col_m2:
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
            fig_cm = px.imshow(
                cm,
                x=["Negative", "Neutral", "Positive"],
                y=["Negative", "Neutral", "Positive"],
                text_auto=True,
                color_continuous_scale="Purples",
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                margin=dict(t=20),
                title="Confusion Matrix",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # F1 score comparison
        st.markdown('<div class="section-title">F1 Score Comparison</div>', unsafe_allow_html=True)
        f1_data = {k: v["f1-score"] for k, v in report.items() if k in ["Negative", "Neutral", "Positive"]}
        fig_f1 = px.bar(
            x=list(f1_data.keys()),
            y=list(f1_data.values()),
            color=list(f1_data.keys()),
            color_discrete_map={
                "Positive": "#10b981",
                "Neutral": "#6366f1",
                "Negative": "#f43f5e",
            },
            labels={"x": "Sentiment", "y": "F1 Score"},
        )
        fig_f1.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            showlegend=False,
            margin=dict(t=20),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── Tab 5: Comment Explorer ─────────────────────────────
    with tab5:
        st.markdown('<div class="section-title">Browse & Analyze Comments</div>', unsafe_allow_html=True)

        filter_sentiment = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Neutral", "Negative"],
        )

        filtered_df = df if filter_sentiment == "All" else df[df["sentiment"] == filter_sentiment]

        st.markdown(f"Showing **{len(filtered_df):,}** comments")

        # Show top comments
        for idx, row in filtered_df.head(20).iterrows():
            sentiment_color = {"Positive": "#10b981", "Neutral": "#6366f1", "Negative": "#f43f5e"}.get(row["sentiment"], "#6b7280")
            with st.expander(f"💬 Comment #{idx+1}  |  Sentiment: {row['sentiment']}  |  Polarity: {row['polarity']:.3f}"):
                st.markdown(f'<div class="comment-box">{row["comment"]}</div>', unsafe_allow_html=True)

                # Show POS tags for this comment
                st.markdown("**POS Tags:**")
                if isinstance(row["tokens_info"], list) and row["tokens_info"]:
                    html = render_pos_tags(row["tokens_info"])
                    st.markdown(html, unsafe_allow_html=True)

                # Show entities
                if isinstance(row["entities"], list) and row["entities"]:
                    st.markdown("**Named Entities:**")
                    for ent_text, ent_label in row["entities"]:
                        st.markdown(f"- `{ent_text}` → **{ent_label}**")

        # Download option
        st.markdown("---")
        csv_data = filtered_df[["comment", "clean_comment", "processed_text", "sentiment", "polarity", "subjectivity"]].to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="youtube_comment_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    # ── Empty state ─────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
        <h2 style="color: #9ca3af; font-weight: 400;">Enter a YouTube URL to begin analysis</h2>
        <p style="color: #6b7280; max-width: 500px; margin: 0 auto;">
            This tool fetches YouTube comments, performs NLP analysis using spaCy for POS tagging
            and TextBlob for sentiment analysis, then trains a machine learning classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)
