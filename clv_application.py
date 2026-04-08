# ============================================================
# CLV PREDICTION — DYNAMIC STREAMLIT APP
# Works with ANY customer dataset — auto-detects columns
# Tested with: Online Retail II, Customer Flight Activity,
#              and any generic transaction / loyalty CSV
# Run: streamlit run clv_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #181c27; border-right: 1px solid #2a2f45; }
[data-testid="stMetric"] {
    background-color: #181c27; border: 1px solid #2a2f45;
    border-radius: 10px; padding: 16px;
}
[data-testid="stMetricLabel"] { color: #6b7194 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e8eaf0 !important; }
h1,h2,h3 { color: #e8eaf0 !important; }
p, li { color: #9ca3c0; }
hr { border-color: #2a2f45; }
</style>
""", unsafe_allow_html=True)

SEG_COLORS = {
    "Champions":         "#3ecf8e",
    "Loyal Customers":   "#4f8ef7",
    "At-Risk Customers": "#f5a623",
    "Lost/Inactive":     "#f46b6b",
}
PLOT_BG    = "#181c27"
PLOT_PAPER = "#0f1117"
TEMPLATE   = "plotly_dark"


# ════════════════════════════════════════════════════════════
# COLUMN AUTO-DETECTION
# ════════════════════════════════════════════════════════════

def detect_column(df, candidates):
    cols_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in cols_lower:
            return cols_lower[key]
    return None


def auto_detect_schema(df):
    return {
        "customer_id":      detect_column(df, ["customerid","customer id","customer_id","custid","cust id","clientid","client id","userid","user id","memberid","member id"]),
        "date":             detect_column(df, ["invoicedate","invoice date","orderdate","order date","date","transactiondate","purchasedate","purchase date"]),
        "invoice":          detect_column(df, ["invoice","invoiceno","invoice no","orderid","order id","transactionid","transaction id","receiptid"]),
        "quantity":         detect_column(df, ["quantity","qty","units","flightsbooked","flights booked","totalflights","total flights","numorders"]),
        "price":            detect_column(df, ["price","unitprice","unit price","amount","rate","fare","ticketprice","cost"]),
        "monetary_direct":  detect_column(df, ["revenue","sales","total","totalspend","total spend","totalrevenue","spend","monetary","dollarcostpointsredeemed","dollar cost points redeemed"]),
        "frequency_direct": detect_column(df, ["frequency","numtransactions","num transactions","orderscount"]),
        "distance":         detect_column(df, ["distance","miles","km","kilometers"]),
        "points":           detect_column(df, ["pointsaccumulated","points accumulated","points","loyaltypoints","loyalty points","rewardpoints"]),
        "year":             detect_column(df, ["year"]),
        "month":            detect_column(df, ["month"]),
    }


def determine_mode(schema):
    if schema["invoice"] and schema["quantity"] and schema["price"]:
        return "transaction"
    if schema["monetary_direct"] or schema["frequency_direct"]:
        return "aggregated"
    if schema["year"] and (schema["distance"] or schema["points"] or schema["quantity"]):
        return "activity"
    return "unknown"


# ════════════════════════════════════════════════════════════
# RFM BUILDERS
# ════════════════════════════════════════════════════════════

def build_rfm_transaction(df, schema):
    cid, date, inv = schema["customer_id"], schema["date"], schema["invoice"]
    qty, price     = schema["quantity"], schema["price"]
    df = df.copy()
    df[date]  = pd.to_datetime(df[date], errors="coerce")
    df[qty]   = pd.to_numeric(df[qty],   errors="coerce").fillna(0)
    df[price] = pd.to_numeric(df[price], errors="coerce").fillna(0)
    df = df.dropna(subset=[cid, date])
    df = df[(df[qty] > 0) & (df[price] > 0)]
    df["_rev"] = df[qty] * df[price]
    ref = df[date].max() + pd.Timedelta(days=1)
    rfm = df.groupby(cid).agg(
        Recency   = (date,    lambda x: (ref - x.max()).days),
        Frequency = (inv,     "nunique"),
        Monetary  = ("_rev",  "sum"),
    ).reset_index().rename(columns={cid: "Customer ID"})
    spans = df.groupby(cid)[date].agg(["min","max"])
    spans["Lifespan_Months"] = ((spans["max"] - spans["min"]).dt.days / 30).clip(lower=1)
    rfm = rfm.merge(spans[["Lifespan_Months"]], left_on="Customer ID", right_index=True)
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["CLV"] = (rfm["AOV"] * rfm["Frequency"] * rfm["Lifespan_Months"]).round(2)
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm, "Revenue (£)"


def build_rfm_activity(df, schema):
    cid, year, month = schema["customer_id"], schema["year"], schema["month"]
    df = df.copy()
    df[cid] = pd.to_numeric(df[cid], errors="coerce")
    df = df.dropna(subset=[cid])
    df[cid] = df[cid].astype(int)
    if year and month:
        df["_period"] = pd.to_numeric(df[year], errors="coerce") * 12 + pd.to_numeric(df[month], errors="coerce")
    elif year:
        df["_period"] = pd.to_numeric(df[year], errors="coerce") * 12
    else:
        df["_period"] = range(len(df))
    df["_period"] = df["_period"].fillna(0)

    # Monetary proxy
    monetary_label = "Activity Score"
    for key in ["distance", "points", "monetary_direct"]:
        col = schema.get(key)
        if col and col in df.columns:
            df["_monetary"] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
            monetary_label  = col
            break
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        excl = [c for c in [cid, year, month, "_period"] if c]
        sum_cols = [c for c in num_cols if c not in excl]
        df["_monetary"] = df[sum_cols].clip(lower=0).sum(axis=1)

    # Frequency proxy
    freq_col = schema.get("quantity")
    if freq_col and freq_col in df.columns:
        df["_freq"] = pd.to_numeric(df[freq_col], errors="coerce").fillna(0).clip(lower=0)
    else:
        df["_freq"] = 1

    ref = df["_period"].max()
    rfm = df.groupby(cid).agg(
        Recency   = ("_period",   lambda x: ref - x.max()),
        Frequency = ("_freq",     "sum"),
        Monetary  = ("_monetary", "sum"),
        _min_p    = ("_period",   "min"),
        _max_p    = ("_period",   "max"),
    ).reset_index().rename(columns={cid: "Customer ID"})
    rfm["Lifespan_Months"] = (rfm["_max_p"] - rfm["_min_p"]).clip(lower=1)
    rfm = rfm.drop(columns=["_min_p","_max_p"])
    rfm["Monetary"]  = rfm["Monetary"].round(2)
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    rfm["AOV"]       = (rfm["Monetary"] / rfm["Frequency"]).round(2)
    rfm["CLV"]       = (rfm["AOV"] * rfm["Frequency"] * rfm["Lifespan_Months"]).round(2)
    return rfm, monetary_label


def build_rfm_aggregated(df, schema):
    cid = schema["customer_id"]
    df  = df.copy()
    df[cid] = pd.to_numeric(df[cid], errors="coerce")
    df = df.dropna(subset=[cid])
    df[cid] = df[cid].astype(int)
    mon_col  = schema["monetary_direct"]
    freq_col = schema["frequency_direct"] or schema["quantity"]
    agg = {cid: "first"}
    if mon_col:  agg["_mon"]  = (mon_col,  "sum")
    if freq_col: agg["_freq"] = (freq_col, "sum")
    rfm = df.groupby(cid).agg(
        **({} if not mon_col  else {"_mon":  (mon_col,  "sum")}),
        **({} if not freq_col else {"_freq": (freq_col, "sum")}),
    ).reset_index().rename(columns={cid: "Customer ID"})
    rfm["Monetary"]  = pd.to_numeric(rfm.get("_mon",  0), errors="coerce").fillna(0).round(2)
    rfm["Frequency"] = pd.to_numeric(rfm.get("_freq", 1), errors="coerce").fillna(1).clip(lower=1)
    rfm["Recency"]         = 0
    rfm["Lifespan_Months"] = 12
    rfm["AOV"] = (rfm["Monetary"] / rfm["Frequency"]).round(2)
    rfm["CLV"] = (rfm["AOV"] * rfm["Frequency"] * rfm["Lifespan_Months"]).round(2)
    return rfm, mon_col or "Monetary"


def build_rfm_unknown(df, schema):
    cid = schema["customer_id"] or df.columns[0]
    df  = df.copy()
    df[cid] = pd.to_numeric(df[cid], errors="coerce")
    df = df.dropna(subset=[cid])
    df[cid] = df[cid].astype(int)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != cid]
    rfm = df.groupby(cid)[num_cols].sum().reset_index().rename(columns={cid: "Customer ID"})
    rfm["Monetary"]        = rfm[num_cols].clip(lower=0).sum(axis=1).round(2)
    rfm["Frequency"]       = 1
    rfm["Recency"]         = 0
    rfm["Lifespan_Months"] = 12
    rfm["AOV"] = rfm["Monetary"]
    rfm["CLV"] = rfm["Monetary"]
    return rfm, "Aggregated Numerics"


# ════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_regression(rfm_json):
    rfm = pd.read_json(rfm_json)
    cap = rfm["CLV"].quantile(0.99)
    rfm_m = rfm[rfm["CLV"] <= cap].copy()
    features = [f for f in ["Recency","Frequency","Monetary","Lifespan_Months"] if f in rfm_m.columns]
    X, y = rfm_m[features], rfm_m["CLV"]
    if len(X) < 10:
        rfm["Predicted_CLV"] = rfm["CLV"]
        return rfm.to_json(), 0.0, 0.0, features, [], []
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    model  = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pr   = model.predict(X_te)
    r2     = float(r2_score(y_te, y_pr))
    rmse   = float(np.sqrt(mean_squared_error(y_te, y_pr)))
    rfm    = rfm.copy()
    rfm["Predicted_CLV"] = np.round(model.predict(scaler.transform(rfm[features])), 2)
    return rfm.to_json(), r2, rmse, features, list(y_te), list(y_pr)


@st.cache_data(show_spinner=False)
def run_clustering(rfm_json, k):
    rfm = pd.read_json(rfm_json)
    cf  = [f for f in ["Recency","Frequency","Monetary"] if f in rfm.columns]
    sc  = StandardScaler().fit_transform(rfm[cf].fillna(0))
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = km.fit_predict(sc)
    rank      = rfm.groupby("Cluster")["Predicted_CLV"].mean().sort_values(ascending=False).index.tolist()
    seg_names = ["Champions","Loyal Customers","At-Risk Customers","Lost/Inactive"]
    labels    = (seg_names + [f"Segment {i+1}" for i in range(10)])[:k]
    rfm["Segment"] = rfm["Cluster"].map({c: labels[i] for i, c in enumerate(rank)})
    return rfm.to_json()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 CLV Dashboard")
    st.markdown("Customer Lifetime Value Prediction")
    st.divider()
    uploaded = st.file_uploader(
        "Upload any customer CSV", type=["csv"],
        help="Works with retail, airline, loyalty, e-commerce, or any transaction CSV"
    )
    st.divider()
    st.markdown("#### ⚙️ Model Settings")
    n_clusters  = st.slider("K-Means clusters",   2, 6,   4)
    clv_cap_pct = st.slider("CLV cap percentile", 90, 100, 99)
    st.divider()
    st.markdown("PGDM E-Business · WeSchool Mumbai  \nAI & ML Assignment — Trimester III")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

st.title("Customer Lifetime Value Prediction")
st.markdown("Regression + K-Means Segmentation · Python · Scikit-learn")

if uploaded is None:
    st.info("👈 Upload any customer CSV from the sidebar. The app auto-detects your columns.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ✅ Supported formats")
        st.markdown("Online Retail, Flight Activity, Loyalty programs, E-commerce orders, any transaction log")
    with c2:
        st.markdown("#### 🧮 Auto RFM")
        st.markdown("Recency, Frequency & Monetary auto-computed from whatever columns exist")
    with c3:
        st.markdown("#### 🎯 Segments")
        st.markdown("K-Means clusters customers into Champions, Loyal, At-Risk, Lost")
    st.stop()

# ── Load CSV ─────────────────────────────────────────────────
with st.spinner("Reading file..."):
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

with st.expander("📄 Raw file preview & detected columns", expanded=False):
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns  |  Columns: {list(df_raw.columns)}")

schema = auto_detect_schema(df_raw)
mode   = determine_mode(schema)

mode_labels = {
    "transaction": "🛒 Transaction dataset (Invoice × Quantity × Price)",
    "activity":    "✈️ Activity dataset (Year/Month activity rows per customer)",
    "aggregated":  "📦 Aggregated dataset (pre-computed monetary/frequency columns)",
    "unknown":     "🔍 Unknown format — summing all numeric columns as monetary proxy",
}
st.info(f"**Detected mode:** {mode_labels[mode]}")

# ── Build RFM ────────────────────────────────────────────────
with st.spinner("Computing RFM features..."):
    try:
        if mode == "transaction":
            rfm, monetary_label = build_rfm_transaction(df_raw, schema)
        elif mode == "activity":
            rfm, monetary_label = build_rfm_activity(df_raw, schema)
        elif mode == "aggregated":
            rfm, monetary_label = build_rfm_aggregated(df_raw, schema)
        else:
            rfm, monetary_label = build_rfm_unknown(df_raw, schema)
    except Exception as e:
        st.error(f"RFM computation error: {e}")
        st.stop()

if rfm is None or len(rfm) < 5:
    st.error("Not enough customers found. Check your file has a customer ID column.")
    st.stop()

# ── Regression ───────────────────────────────────────────────
with st.spinner("Running regression..."):
    rfm_json, r2, rmse, features, y_test_list, y_pred_list = run_regression(rfm.to_json())
    rfm = pd.read_json(rfm_json)

# ── Clustering ───────────────────────────────────────────────
with st.spinner(f"K-Means clustering (k={n_clusters})..."):
    rfm_json = run_clustering(rfm.to_json(), n_clusters)
    rfm = pd.read_json(rfm_json)

st.success(
    f"✅ {df_raw.shape[0]:,} rows → **{len(rfm):,} customers**  |  "
    f"Mode: **{mode}**  |  Monetary: **{monetary_label}**  |  R²: **{r2:.4f}**"
)

# ════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Customers",      f"{len(rfm):,}")
m2.metric("Total Monetary", f"{rfm['Monetary'].sum():,.0f}")
m3.metric("Avg CLV",        f"{rfm['CLV'].mean():,.1f}")
m4.metric("Avg Frequency",  f"{rfm['Frequency'].mean():,.1f}")
m5.metric("Model R²",       f"{r2:.4f}" if r2 > 0 else "N/A")

# ════════════════════════════════════════════════════════════
# REGRESSION CHARTS
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("📈 Regression Model")

col_r1, col_r2 = st.columns([1.6, 1])
with col_r1:
    if len(y_test_list) > 0:
        cap_val = float(rfm["CLV"].quantile(clv_cap_pct / 100))
        yt = np.array(y_test_list)
        yp = np.array(y_pred_list)
        mask = (yt <= cap_val) & (yp <= cap_val)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yt[mask], y=yp[mask], mode="markers",
            marker=dict(color="#4f8ef7", size=4, opacity=0.5), name="Customers"))
        mn, mx = float(yt[mask].min()), float(yt[mask].max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="#f46b6b", dash="dash", width=1.5), name="Ideal fit"))
        fig.update_layout(title="Actual vs Predicted CLV", xaxis_title="Actual CLV",
            yaxis_title="Predicted CLV", template=TEMPLATE,
            paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, height=340)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for regression plot.")

with col_r2:
    st.markdown("#### Performance")
    st.metric("R² Score", f"{r2:.4f}" if r2 > 0 else "N/A")
    st.metric("RMSE",     f"{rmse:,.2f}" if rmse > 0 else "N/A")
    st.metric("Test set", f"{len(y_test_list):,}")
    st.markdown("#### Features used")
    for f in features:
        st.markdown(f"- `{f}`")

cap_val    = rfm["CLV"].quantile(clv_cap_pct / 100)
fig_hist   = px.histogram(rfm[rfm["CLV"] <= cap_val], x="CLV", nbins=40,
    color_discrete_sequence=["#4f8ef7"], template=TEMPLATE,
    title=f"CLV Distribution (capped at {clv_cap_pct}th percentile)")
fig_hist.update_layout(paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, height=230, bargap=0.05, showlegend=False)
st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════
# SEGMENTATION
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("🎯 Customer Segmentation — K-Means")

agg_args = {"Customers": ("Customer ID","count"), "Avg_CLV": ("Predicted_CLV","mean"),
            "Avg_Frequency": ("Frequency","mean"), "Avg_Monetary": ("Monetary","mean")}
if "Recency" in rfm.columns:
    agg_args["Avg_Recency"] = ("Recency","mean")
seg_summary = rfm.groupby("Segment").agg(**agg_args).round(2).reset_index().sort_values("Avg_CLV", ascending=False)

seg_actions = {
    "Champions":         "Reward & retain — offer loyalty perks",
    "Loyal Customers":   "Upsell higher-value products",
    "At-Risk Customers": "Re-engagement campaign",
    "Lost/Inactive":     "Win-back offer or suppress",
}

cols = st.columns(len(seg_summary))
for idx, (_, row) in enumerate(seg_summary.iterrows()):
    seg   = row["Segment"]
    color = SEG_COLORS.get(seg, "#aaaaaa")
    with cols[idx]:
        st.markdown(f"<div style='border-left:3px solid {color};padding-left:10px'>", unsafe_allow_html=True)
        st.markdown(f"**{seg}**")
        st.metric("Customers",  f"{int(row['Customers']):,}")
        st.metric("Avg CLV",    f"{row['Avg_CLV']:,.1f}")
        st.metric("Avg Freq",   f"{row['Avg_Frequency']:,.1f}")
        st.caption(seg_actions.get(seg, "Analyse further"))
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    cap98 = rfm["Monetary"].quantile(0.98)
    fig   = px.scatter(rfm[rfm["Monetary"] <= cap98], x="Frequency", y="Monetary",
        color="Segment", color_discrete_map=SEG_COLORS, opacity=0.5, template=TEMPLATE,
        title="Frequency vs Monetary by Segment",
        labels={"Monetary": monetary_label, "Frequency": "Frequency"})
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, height=360)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure(go.Pie(
        labels=seg_summary["Segment"], values=seg_summary["Customers"], hole=0.62,
        marker=dict(colors=[SEG_COLORS.get(s,"#aaa") for s in seg_summary["Segment"]]),
        textinfo="label+percent", textfont=dict(size=11),
    ))
    fig.update_layout(title="Segment Split", paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG, height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

cap97 = rfm["Predicted_CLV"].quantile(0.97)
fig   = px.box(rfm[rfm["Predicted_CLV"] <= cap97], x="Segment", y="Predicted_CLV",
    color="Segment", color_discrete_map=SEG_COLORS, template=TEMPLATE,
    title="Predicted CLV by Segment",
    labels={"Predicted_CLV":"Predicted CLV","Segment":""},
    category_orders={"Segment": list(seg_summary["Segment"])})
fig.update_layout(paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, height=300, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()
for _, row in seg_summary.iterrows():
    seg = row["Segment"]
    x_labels = ["Frequency", "Monetary (/100)"]
    y_vals   = [row["Avg_Frequency"], row["Avg_Monetary"] / 100]
    if "Avg_Recency" in seg_summary.columns:
        x_labels.append("Recency")
        y_vals.append(row["Avg_Recency"])
    fig.add_trace(go.Bar(name=seg, x=x_labels, y=y_vals,
        marker_color=SEG_COLORS.get(seg,"#aaa"), opacity=0.85))
fig.update_layout(barmode="group", title="RFM Averages by Segment",
    template=TEMPLATE, paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, height=300)
st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TABLES
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("📋 Data Tables")

show_cols = [c for c in ["Customer ID","Recency","Frequency","Monetary",
                          "Lifespan_Months","CLV","Predicted_CLV","Segment"] if c in rfm.columns]
tab1, tab2, tab3 = st.tabs(["Top Customers", "Segment Summary", "Full RFM Table"])
with tab1:
    st.dataframe(rfm.sort_values("Predicted_CLV", ascending=False).head(20)[show_cols].reset_index(drop=True), use_container_width=True)
with tab2:
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)
with tab3:
    st.dataframe(rfm[show_cols], use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# DOWNLOAD
# ════════════════════════════════════════════════════════════
st.divider()
st.subheader("⬇️ Export")
st.download_button(
    label="Download CLV Predictions CSV",
    data=rfm[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="CLV_Predictions_Output.csv",
    mime="text/csv",
)
