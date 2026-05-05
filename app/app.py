import streamlit as st
import pandas as pd
import plotly.express as px

# ==============================
# CONFIG PAGE
# ==============================
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide",
    page_icon="🛒"
)

# ==============================
# CUSTOM CSS (MODERN UI)
# ==============================
st.markdown("""
<style>
body {
    background-color: #0F172A;
    color: #E2E8F0;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-card h4 {
    color: #94A3B8;
    margin-bottom: 10px;
}

.metric-card h2 {
    color: #E2E8F0;
}

section.main > div {
    background-color: #0F172A;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("""
# 🛒 Customer Intelligence Dashboard  
<span style='color:#94A3B8'>
Interactive segmentation & recommendation system using RFM Analysis
</span>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data/data.csv", encoding='latin1')

# ==============================
# PREPROCESSING
# ==============================
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ==============================
# RFM CALCULATION
# ==============================
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'Revenue': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# ==============================
# RFM SCORING (FIXED)
# ==============================
rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

rfm['RFM_score'] = (
    rfm['R_score'].astype(int) +
    rfm['F_score'].astype(int) +
    rfm['M_score'].astype(int)
)

# ==============================
# SEGMENTATION (IMPROVED)
# ==============================
def segment(row):
    if row['RFM_score'] >= 10:
        return "💎 High Value"
    elif row['RFM_score'] >= 6:
        return "🟢 Regular"
    else:
        return "⚠️ At Risk"

rfm['Segment'] = rfm.apply(segment, axis=1)

# ==============================
# INPUT
# ==============================
customer_list = sorted(rfm.index.astype(int))
customer_id = st.selectbox("🔍 Select Customer ID", customer_list)

customer_data = rfm.loc[customer_id]

# ==============================
# METRICS
# ==============================
st.markdown("## 📊 Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"""
<div class="metric-card">
<h4>Recency</h4>
<h2>{int(customer_data['Recency'])} days</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="metric-card">
<h4>Frequency</h4>
<h2>{int(customer_data['Frequency'])}</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="metric-card">
<h4>Monetary</h4>
<h2>${round(customer_data['Monetary'],2)}</h2>
</div>
""", unsafe_allow_html=True)

col4.markdown(f"""
<div class="metric-card">
<h4>Segment</h4>
<h2>{customer_data['Segment']}</h2>
</div>
""", unsafe_allow_html=True)

# ==============================
# RECOMMENDATION
# ==============================
st.markdown("## 🛍️ Recommended Products")

top_products = df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

colA, colB = st.columns([2,1])

with colA:
    st.dataframe(top_products)

with colB:
    fig = px.bar(
        top_products.sort_values(),
        orientation='h',
        title="Top Products by Revenue",
        labels={'value': 'Revenue', 'index': 'Product'}
    )

    fig.update_layout(
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#E2E8F0"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# SEGMENT DISTRIBUTION
# ==============================
st.markdown("## 📈 Customer Segment Distribution")

segment_counts = rfm['Segment'].value_counts()

col1, col2 = st.columns([2,1])

with col1:
    fig2 = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        hole=0.6,
        title="Customer Segments",
        color=segment_counts.index,
        color_discrete_map={
            "💎 High Value": "#6366F1",
            "🟢 Regular": "#22C55E",
            "⚠️ At Risk": "#F59E0B"
        }
    )

    fig2.update_layout(
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#E2E8F0"
    )

    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.dataframe(segment_counts)

# ==============================
# CUSTOMER BEHAVIOR MAP
# ==============================
st.markdown("## 🧠 Customer Behavior Map")

fig3 = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="Segment",
    size="Monetary",
    hover_data=["Recency"],
    title="Customer Segmentation Map",
    color_discrete_map={
        "💎 High Value": "#6366F1",
        "🟢 Regular": "#22C55E",
        "⚠️ At Risk": "#F59E0B"
    }
)

fig3.update_layout(
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="#E2E8F0"
)

st.plotly_chart(fig3, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.divider()
st.caption("Built with ❤️ | M. Wildan Nabila — Customer Analytics")