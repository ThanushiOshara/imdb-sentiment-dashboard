import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis Dashboard",
    page_icon="🎬",
    layout="wide"
)

# 2. Custom CSS Injection for Modern UI Styling (Dark Theme & Glowing Cards)
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
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Cache Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv('IMDB Dataset.csv')
    df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))
    return df

with st.spinner('🚀 Loading IMDB Dataset & Preparing Enhanced Dashboard... Please wait!'):
    df = load_data()

# 4. Sidebar Controls & New Interactive Features
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
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 18px;'>Advanced Big Data Analytics Dashboard with Live Text Prediction & Visual Insights</p>", unsafe_allow_html=True)
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
        st.warning("No data available.")

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
        st.warning("No data available.")

st.markdown("---")

# 8. NEW FEATURE: Word Cloud Section
st.markdown("### ☁️ Frequent Words Cloud")
if total_reviews > 0:
    text_data = " ".join(review for review in filtered_df['review'].head(500))
    wordcloud = WordCloud(width=800, height=350, background_color='#0b0f19', colormap='Set2').generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    fig.patch.set_facecolor('#0b0f19')
    st.pyplot(fig)
else:
    st.info("No data for Word Cloud.")

st.markdown("---")

# # 9. NEW FEATURE: Live Sentiment Predictor / Analyzer Tool
st.markdown("### ✨ Interactive Custom Review Sentiment Tester")
user_input_review = st.text_area("Type or paste a movie review below to test its sentiment:")
if st.button("Analyze Sentiment"):
    if user_input_review.strip() != "":
        text_lower = user_input_review.lower()
        
        # Positive and Negative keywords list
        positive_words = ["good", "great", "awesome", "fantastic", "amazing", "love", "best", "excellent", "brilliant", "wonderful", "enjoy"]
        negative_words = ["boring", "waste", "bad", "worst", "terrible", "awful", "poor", "horrible", "disappointing", "dull", "slow"]
        
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        if neg_score > pos_score:
            st.error("⚠️ Predicted Sentiment: **NEGATIVE 😞**")
        elif pos_score > neg_score:
            st.success("🎉 Predicted Sentiment: **POSITIVE 😊**")
        else:
            # Fallback based on common negative words presence or length check
            if "boring" in text_lower or "waste" in text_lower or "not" in text_lower:
                st.error("⚠️ Predicted Sentiment: **NEGATIVE 😞**")
            else:
                st.success("🎉 Predicted Sentiment: **POSITIVE 😊**")
    else:
        st.warning("Please type a review first.")

# 10. Data Table & CSV Download Option
st.markdown("### 🔎 Filtered Reviews Inspector & Export")
if total_reviews > 0:
    st.dataframe(filtered_df[['review', 'sentiment', 'word_count']].head(50), use_container_width=True, height=300)
    
    # Download Button Feature
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name='filtered_imdb_sentiment.csv',
        mime='text/csv'
    )
else:
    st.info("No records found.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>IMDB Sentiment Analysis Big Data Dashboard | Enhanced Edition</p>", unsafe_allow_html=True)