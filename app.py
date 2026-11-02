import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="IMDB Sentiment Analysis Dashboard", page_icon="🎬", layout="wide")

# Custom Title
st.title("🎬 IMDB Movie Reviews Sentiment Analysis Dashboard")
st.markdown("**Big Data Analytics & Interactive UI with Python & Streamlit**")
st.write("---")

# Load Data efficiently using Pandas
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df['length'] = df['review'].astype(str).apply(len)
    return df

with st.spinner("Loading dataset... Please wait."):
    df = load_data()

# Sidebar Controls
st.sidebar.header("🔍 Dashboard Controls")
sentiment_filter = st.sidebar.selectbox("Select Sentiment Filter", ["All", "positive", "negative"])
keyword_search = st.sidebar.text_input("Search Keyword in Reviews", "")

# Apply Filters
filtered_df = df.copy()
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]

if keyword_search:
    filtered_df = filtered_df[filtered_df['review'].str.contains(keyword_search, case=False, na=False)]

# Top Level Summary Metrics
st.subheader("📊 Dataset Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

total_count = len(filtered_df)
pos_count = len(filtered_df[filtered_df['sentiment'] == 'positive'])
neg_count = len(filtered_df[filtered_df['sentiment'] == 'negative'])
avg_len = int(filtered_df['length'].mean()) if total_count > 0 else 0

col1.metric("Total Reviews", f"{total_count:,}")
col2.metric("Positive Reviews", f"{pos_count:,}")
col3.metric("Negative Reviews", f"{neg_count:,}")
col4.metric("Avg Review Length", f"{avg_len} chars")

st.write("---")

# Visualizations Section
st.subheader("📈 Visual Analytics")
v_col1, v_col2 = st.columns(2)

with v_col1:
    st.markdown("##### 🍩 Sentiment Ratio")
    if total_count > 0:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ['#2ecc71' if s == 'positive' else '#e74c3c' for s in sentiment_counts.index]
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4))
        st.pyplot(fig)
    else:
        st.warning("No data for chart.")

with v_col2:
    st.markdown("##### 📊 Review Length Distribution")
    if total_count > 0:
        st.bar_chart(filtered_df['length'].head(500))
    else:
        st.warning("No data for chart.")

st.write("---")

# Filtered Data Table & Download Option
st.subheader("📋 Dataset Preview & Export")
col_check, col_dl = st.columns([2, 1])

with col_check:
    show_table = st.checkbox("Show Raw Data Table Preview (First 20 Rows)")
with col_dl:
    # CSV Download Button
    csv_data = filtered_df.head(1000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name="filtered_imdb_reviews.csv",
        mime="text/csv"
    )

if show_table:
    st.dataframe(filtered_df[['sentiment', 'length', 'review']].head(20), use_container_width=True)

# Random Review Inspector Feature
if total_count > 0:
    st.write("---")
    st.subheader("🔍 Review Inspector")
    sample_review = filtered_df.sample(1).iloc[0]
    sentiment_color = "🟢" if sample_review['sentiment'] == 'positive' else "🔴"
    st.markdown(f"**Sample Review ({sentiment_color} {sample_review['sentiment'].capitalize()}):**")
    st.info(f'"{sample_review["review"]}"')