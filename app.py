import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration (Wide layout, Dark/Light friendly clean title)
st.set_page_config(
    page_title="IMDB Sentiment Analysis Dashboard",
    page_icon="🎬",
    layout="wide"
)

# 2. Custom CSS Injection for Modern UI Styling (Dark Theme & Neon Accents)
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stApp {
        background-color: #0b0f19;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cache Data Loading for High Performance
@st.cache_data
def load_data():
    # Load dataset (make sure IMDB Dataset.csv is in the same folder)
    df = pd.read_csv('IMDB Dataset.csv')
    # Clean/Precompute basic columns if needed
    df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))
    return df

with st.spinner('🚀 Loading IMDB Dataset & Preparing Dashboard... Please wait!'):
    df = load_data()

# 4. Sidebar Controls & Filters
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

sentiment_filter = st.sidebar.selectbox(
    "Filter by Sentiment:",
    ["All Sentiments", "positive", "negative"]
)

search_keyword = st.sidebar.text_input("🔍 Search in Reviews:", "")

max_words = st.sidebar.slider(
    "Max Word Count Filter:", 
    int(df['word_count'].min()), 
    int(df['word_count'].max()), 
    int(df['word_count'].max())
)

# Apply Filters
filtered_df = df.copy()
if sentiment_filter != "All Sentiments":
    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]

if search_keyword:
    filtered_df = filtered_df[filtered_df['review'].str.contains(search_keyword, case=False, na=False)]

filtered_df = filtered_df[filtered_df['word_count'] <= max_words]

# 5. Main Header Section
st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🎬 IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 18px;'>An interactive Big Data analytics dashboard exploring 50,000 cinematic reviews using Streamlit and Pandas.</p>", unsafe_allow_html=True)
st.markdown("---")

# 6. Top Metrics Row
total_reviews = len(filtered_df)
pos_count = len(filtered_df[filtered_df['sentiment'] == 'positive'])
neg_count = len(filtered_df[filtered_df['sentiment'] == 'negative'])
avg_words = int(filtered_df['word_count'].mean()) if total_reviews > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Reviews</div>
            <div class="metric-value" style="color: #38bdf8;">{total_reviews:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Positive Reviews</div>
            <div class="metric-value" style="color: #4ade80;">{pos_count:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Negative Reviews</div>
            <div class="metric-value" style="color: #f87171;">{neg_count:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Word Count</div>
            <div class="metric-value" style="color: #fbbf24;">{avg_words}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Visualizations Section (Charts)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### 📊 Sentiment Proportion")
    if total_reviews > 0:
        sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        fig_pie = px.pie(
            sentiment_counts, 
            names='sentiment', 
            values='count', 
            hole=0.5,
            color='sentiment',
            color_discrete_map={'positive': '#4ade80', 'negative': '#f87171'}
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

with chart_col2:
    st.markdown("### 📈 Review Length Distribution")
    if total_reviews > 0:
        fig_hist = px.histogram(
            filtered_df, 
            x='word_count', 
            nbins=30,
            color_discrete_sequence=['#38bdf8']
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            xaxis_title="Word Count",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

st.markdown("---")

# 8. Interactive Review Inspector Table & Explorer
st.markdown("### 🔎 Interactive Review Inspector")
if total_reviews > 0:
    # Display dataframe preview with styled container
    st.dataframe(
        filtered_df[['review', 'sentiment', 'word_count']].head(100),
        use_container_width=True,
        height=350
    )
    
    # Single Review Inspector Tool
    st.markdown("#### 📖 Inspect Specific Review Text")
    review_index = st.number_input("Select Review Index:", min_value=0, max_value=max(0, len(filtered_df)-1), value=0)
    
    selected_row = filtered_df.iloc[review_index]
    sentiment_color = "#4ade80" if selected_row['sentiment'] == 'positive' else "#f87171"
    
    st.markdown(f"""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid {sentiment_color};">
            <p><b>Sentiment:</b> <span style="color: {sentiment_color}; text-transform: uppercase;">{selected_row['sentiment']}</span> | <b>Word Count:</b> {selected_row['word_count']}</p>
            <p style="color: #cbd5e1; font-style: italic;">"{selected_row['review']}"</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Adjust your sidebar filters to view records.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>IMDB Sentiment Analysis Dashboard | Big Data Project</p>", unsafe_allow_html=True)