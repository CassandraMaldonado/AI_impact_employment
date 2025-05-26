import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Placeholder data (simulate from your processed analysis)
# ------------------------------
data = {
    'industry': ['Healthcare', 'Finance', 'Education', 'Retail', 'Legal', 'Media'],
    'sentiment': ['Positive', 'Mixed', 'Mixed', 'Positive', 'Negative', 'Positive'],
    'top_topics': [
        'AI diagnostics, triage, MedPaLM',
        'Fraud detection, compliance tools, GPT-4',
        'AI tutors, plagiarism, ChatGPT',
        'Recommendation engines, inventory bots',
        'Contract review, compliance AI',
        'Content generation, translation, Midjourney'
    ],
    'tech_mentions': [
        'MedPaLM, LLMs',
        'GPT-4, ML, rule-based systems',
        'ChatGPT, auto-grading tools',
        'GenAI, CRM AI',
        'Legal LLMs, compliance bots',
        'GenAI, Midjourney, DALL·E'
    ],
    'recommendations': [
        "Use AI to help doctors with diagnosis; test in real hospitals before scaling.",
        "Automate fraud screening and explainable risk modeling. Prioritize transparency.",
        "Pilot AI tutors with oversight; create clear cheating policies.",
        "Adopt AI for inventory and customer recommendations. Monitor personalization risks.",
        "Use AI for document triage only. Invest in explainability to meet legal standards.",
        "Leverage GenAI to create content faster, but watch for bias and misinformation."
    ]
}
df = pd.DataFrame(data)

# Simulated trends data
trend_data = pd.DataFrame({
    'month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'GPT-4': [10, 15, 18, 25, 40, 60, 70, 65, 60, 55, 50, 45],
    'AI tutors': [5, 8, 10, 12, 15, 18, 20, 22, 21, 20, 18, 17],
    'Compliance AI': [3, 5, 6, 7, 10, 12, 15, 17, 15, 14, 13, 12]
}).set_index('month')

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Readiness Navigator", layout="wide")
st.title("🧠 AI Readiness Navigator")
st.markdown("Explore how AI is impacting different industries — and what strategic actions you should take.")

# Sidebar Navigation
page = st.sidebar.radio("Choose a View", ["📊 Industry Dashboard", "🧭 Recommendation Engine", "📈 Trend Explorer"])

# ------------------------------
# 📊 Industry Dashboard
# ------------------------------
if page == "📊 Industry Dashboard":
    st.header("📊 Industry Dashboard")
    selected_industry = st.selectbox("Select Industry", df['industry'].unique())

    row = df[df['industry'] == selected_industry].iloc[0]

    st.subheader(f"Overview for {selected_industry}")
    st.metric("Sentiment Toward AI", row['sentiment'])
    st.markdown(f"**🧵 Top Topics in Coverage**: {row['top_topics']}")
    st.markdown(f"**🔧 Common Tech Mentions**: {row['tech_mentions']}")

# ------------------------------
# 🧭 Recommendation Engine
# ------------------------------
elif page == "🧭 Recommendation Engine":
    st.header("🧭 AI Strategy Recommendations by Industry")
    selected_industry = st.selectbox("Select Industry", df['industry'].unique(), key="reco")

    row = df[df['industry'] == selected_industry].iloc[0]

    st.markdown(f"### ✅ Strategic Guidance for {selected_industry}")
    st.write(row['recommendations'])

    st.markdown("These recommendations are based on analysis of news coverage tone, tech mentions, and role-specific AI adoption patterns.")

# ------------------------------
# 📈 Trend Explorer
# ------------------------------
elif page == "📈 Trend Explorer":
    st.header("📈 AI Trend Explorer")
    selected_topic = st.selectbox("Choose a Technology or Theme", trend_data.columns)

    fig, ax = plt.subplots()
    trend_data[selected_topic].plot(kind='line', marker='o', ax=ax)
    ax.set_title(f"Mentions Over Time: {selected_topic}")
    ax.set_ylabel("Mentions (simulated)")
    ax.set_xlabel("Date")
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("Use this chart to spot attention spikes and adoption signals for different technologies.")
