import streamlit as st
import pandas as pd
import numpy as np


EDUCATION_MAP = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4,
}

SPEND_COLUMNS = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
CHANNEL_COLUMNS = ["Web", "Catalog", "Store", "Discount Purchases", "Total Promo"]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("notebooks/data/clustered_data.csv")
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df["Education"] = df["Education"].replace(EDUCATION_MAP)
    df = df.fillna(df.mean(numeric_only=True))
    df["will_buy"] = (df["Total_Spending"] > df["Total_Spending"].median()).astype(int)
    X = df.drop(columns=["cluster", "will_buy"])
    y = df["will_buy"]
    return X, y


def get_model_options():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    return {
        "Logistic Regression": LogisticRegression(max_iter=300, solver="newton-cg"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


class SpendingThresholdModel:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def predict(self, X: pd.DataFrame):
        return (X["Total_Spending"].to_numpy(dtype=float) > self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame):
        spend = X["Total_Spending"].to_numpy(dtype=float)
        scale = max(abs(self.threshold) * 0.25, 1.0)
        prob = 1 / (1 + np.exp(-(spend - self.threshold) / scale))
        return np.column_stack([1 - prob, prob])


def build_fallback_model(X: pd.DataFrame):
    return SpendingThresholdModel(float(X["Total_Spending"].median())), X.columns.tolist()


@st.cache_data
def train_and_evaluate(model_name: str, X: pd.DataFrame, y: pd.Series, test_size: float):
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    model = clone(get_model_options()[model_name])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    final_model = clone(model)
    final_model.fit(X, y)

    return final_model, accuracy, report


def inject_theme():
    st.markdown(
        """
        <style>
        :root {
            --ink: #17202a;
            --muted: #64748b;
            --line: rgba(15, 23, 42, 0.12);
            --surface: #ffffff;
            --soft: #f5f7fb;
            --teal: #0f766e;
            --blue: #2563eb;
            --coral: #e85d55;
            --amber: #f59e0b;
            --violet: #7c3aed;
        }

        .stApp {
            background:
                linear-gradient(rgba(15, 23, 42, 0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(15, 23, 42, 0.035) 1px, transparent 1px),
                linear-gradient(135deg, #f8fafc 0%, #eef6f4 44%, #fff6ed 100%);
            background-size: 32px 32px, 32px 32px, auto;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent 14rem),
                #111827;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stMetric {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 0.8rem;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #ffffff;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        .hero {
            background:
                linear-gradient(115deg, rgba(11, 75, 72, 0.98), rgba(31, 90, 153, 0.95) 58%, rgba(124, 58, 237, 0.88));
            color: white;
            border-radius: 8px;
            padding: 2.2rem;
            box-shadow: 0 22px 54px rgba(15, 23, 42, 0.22);
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.22);
        }

        .hero:after {
            content: "";
            position: absolute;
            right: -6rem;
            top: -5rem;
            width: 20rem;
            height: 20rem;
            background: repeating-linear-gradient(
                45deg,
                rgba(255, 255, 255, 0.18),
                rgba(255, 255, 255, 0.18) 1px,
                transparent 1px,
                transparent 12px
            );
            transform: rotate(8deg);
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .hero-topline {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-bottom: 1rem;
        }

        .hero-pill {
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 800;
            padding: 0.38rem 0.68rem;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0 0 0.55rem 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.02;
            letter-spacing: 0;
        }

        .hero p {
            color: rgba(255, 255, 255, 0.88);
            font-size: 1.05rem;
            max-width: 46rem;
            margin: 0;
        }

        .hero-stats {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 1.35rem;
            max-width: 48rem;
        }

        .hero-stat {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            padding: 0.78rem 0.9rem;
        }

        .hero-stat span {
            color: rgba(255, 255, 255, 0.72);
            display: block;
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        .hero-stat strong {
            color: #ffffff;
            display: block;
            font-size: 1.18rem;
            margin-top: 0.15rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #17202a;
            margin: 0.35rem 0 0.5rem;
        }

        .metric-card,
        .prediction-card,
        .insight-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }

        .metric-card {
            min-height: 112px;
            padding: 1rem;
            position: relative;
            overflow: hidden;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .metric-card:before {
            background: linear-gradient(180deg, var(--teal), var(--amber));
            content: "";
            height: 100%;
            left: 0;
            position: absolute;
            top: 0;
            width: 5px;
        }

        .metric-card:hover {
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.12);
            transform: translateY(-2px);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.7rem;
            font-weight: 850;
            margin-top: 0.25rem;
        }

        .metric-help {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }

        .prediction-card {
            padding: 1.15rem 1.2rem;
            min-height: 246px;
            overflow: hidden;
            position: relative;
        }

        .prediction-card:before {
            background: linear-gradient(90deg, var(--teal), var(--blue), var(--violet));
            content: "";
            height: 6px;
            left: 0;
            position: absolute;
            right: 0;
            top: 0;
        }

        .prediction-body {
            align-items: center;
            display: grid;
            gap: 1rem;
            grid-template-columns: 1fr auto;
            margin-top: 0.35rem;
        }

        .prediction-eyebrow {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .prediction-title {
            font-size: 2rem;
            line-height: 1.08;
            font-weight: 850;
            margin-top: 0.45rem;
        }

        .prediction-title.buy {
            color: var(--teal);
        }

        .prediction-title.no-buy {
            color: var(--coral);
        }

        .probability {
            color: var(--ink);
            font-size: 2.4rem;
            font-weight: 850;
            margin: 0.4rem 0 0.1rem;
        }

        .prediction-copy {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }

        .score-ring {
            --ring: var(--teal);
            --score: 0;
            align-items: center;
            aspect-ratio: 1;
            background: conic-gradient(var(--ring) var(--score), #e2e8f0 0);
            border-radius: 50%;
            display: grid;
            justify-items: center;
            width: 126px;
        }

        .score-ring-inner {
            align-items: center;
            background: #ffffff;
            border-radius: 50%;
            color: var(--ink);
            display: grid;
            font-size: 1.5rem;
            font-weight: 850;
            height: 94px;
            justify-items: center;
            width: 94px;
        }

        .decision-strip {
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 1rem;
        }

        .decision-chip {
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 750;
            padding: 0.55rem;
            text-align: center;
        }

        .recommend-grid {
            display: grid;
            gap: 0.8rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.5rem 0 1rem;
        }

        .recommend-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
            padding: 0.9rem;
        }

        .recommend-label {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .recommend-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 850;
            margin-top: 0.22rem;
        }

        .recommend-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }

        .story-grid {
            display: grid;
            gap: 0.8rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0.5rem 0 1rem;
        }

        .story-step {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
            min-height: 118px;
            padding: 0.95rem;
            position: relative;
            overflow: hidden;
        }

        .story-step:before {
            background: linear-gradient(180deg, var(--blue), var(--teal));
            content: "";
            height: 100%;
            left: 0;
            position: absolute;
            top: 0;
            width: 4px;
        }

        .story-number {
            color: var(--teal);
            font-size: 0.75rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .story-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 850;
            margin-top: 0.25rem;
        }

        .story-copy {
            color: var(--muted);
            font-size: 0.83rem;
            margin-top: 0.24rem;
        }

        .visual-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            padding: 1rem;
        }

        .donut-layout {
            align-items: center;
            display: grid;
            gap: 1.2rem;
            grid-template-columns: auto 1fr;
            min-height: 320px;
        }

        .segment-donut {
            align-items: center;
            aspect-ratio: 1;
            border-radius: 50%;
            display: grid;
            justify-items: center;
            width: min(260px, 68vw);
        }

        .segment-svg {
            height: auto;
            max-width: 280px;
            width: 100%;
        }

        .segment-slice {
            cursor: pointer;
            filter: drop-shadow(0 10px 12px rgba(15, 23, 42, 0.12));
            transition: opacity 160ms ease, stroke-width 160ms ease;
        }

        .segment-slice:hover {
            opacity: 1;
            stroke-width: 45;
        }

        .donut-help {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.4rem;
            text-align: center;
        }

        .donut-legend {
            display: grid;
            gap: 0.65rem;
        }

        .legend-link {
            text-decoration: none;
        }

        .legend-row {
            align-items: center;
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 8px;
            display: grid;
            gap: 0.65rem;
            grid-template-columns: 12px 1fr auto;
            padding: 0.65rem 0.75rem;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        .legend-row:hover,
        .legend-row.selected {
            border-color: rgba(15, 118, 110, 0.32);
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
            transform: translateY(-1px);
        }

        .legend-row.selected {
            background: #e6f4f1;
        }

        .legend-dot {
            border-radius: 50%;
            height: 12px;
            width: 12px;
        }

        .legend-label {
            color: var(--ink);
            font-weight: 800;
        }

        .legend-value {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 750;
        }

        .selected-segment-card {
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.1), rgba(37, 99, 235, 0.09));
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 8px;
            margin-bottom: 0.8rem;
            padding: 0.9rem;
        }

        .selected-segment-label {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .selected-segment-title {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 850;
            margin-top: 0.15rem;
        }

        .selected-segment-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.2rem;
        }

        .insight-card {
            padding: 1rem;
            min-height: 246px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1rem 0.6rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stForm"] label p {
            color: #334155;
            font-weight: 700;
        }

        label p {
            color: #334155;
            font-weight: 700;
        }

        div[data-testid="stNumberInput"] input:disabled {
            -webkit-text-fill-color: #0f766e;
            background: #e6f4f1;
            border-color: rgba(15, 118, 110, 0.2);
            color: #0f766e;
            font-weight: 850;
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.55rem 0.85rem;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #e6f4f1;
            border-color: rgba(15, 118, 110, 0.28);
            color: var(--teal);
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 8px;
            border: 0;
            background: linear-gradient(135deg, var(--teal), var(--blue));
            color: white;
            font-weight: 800;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.2);
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border: 0;
            color: white;
            filter: brightness(1.04);
        }

        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }

        @media (max-width: 780px) {
            .hero {
                padding: 1.35rem;
            }

            .hero h1 {
                font-size: 2rem;
            }

            .metric-card,
            .prediction-card,
            .insight-card {
                min-height: auto;
            }

            .hero-stats,
            .prediction-body,
            .decision-strip,
            .recommend-grid,
            .story-grid,
            .donut-layout {
                grid-template-columns: 1fr;
            }

            .score-ring {
                width: 108px;
            }

            .score-ring-inner {
                height: 80px;
                width: 80px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_delta(value: float) -> str:
    return f"{value:+,.0f}"


def metric_card(label: str, value: str, help_text: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("Customer Lens")
    st.sidebar.caption("Dataset profile")
    st.sidebar.metric("Customers", format_number(len(df)))
    st.sidebar.metric("Median spending", format_currency(df["Total_Spending"].median()))
    st.sidebar.metric("Median income", format_currency(df["Income"].median()))
    st.sidebar.metric("Segments", format_number(df["cluster"].nunique()))


def number_input_from_series(
    label: str,
    X: pd.DataFrame,
    column: str,
    *,
    is_float: bool = False,
    help_text: str | None = None,
):
    median = X[column].median()
    min_value = 0.0 if is_float else 0
    value = float(median) if is_float else int(median)
    step = 100.0 if is_float and column in {"Income", "Total_Spending"} else 1.0
    if not is_float:
        step = 1

    return st.number_input(
        label,
        min_value=min_value,
        value=value,
        step=step,
        help=help_text,
    )


def customer_input_form(X: pd.DataFrame):
    input_data = {}

    with st.container(border=True):
        profile_tab, spending_tab, channel_tab = st.tabs(
            ["Customer Profile", "Spending Mix", "Channels"]
        )

        with profile_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                education = st.selectbox("Education", list(EDUCATION_MAP.keys()), index=2)
                input_data["Education"] = EDUCATION_MAP[education]
                input_data["Age"] = number_input_from_series("Age", X, "Age")
            with col2:
                input_data["Marital Status"] = st.selectbox(
                    "Marital Status", sorted(X["Marital Status"].unique().tolist())
                )
                input_data["Parental Status"] = st.selectbox(
                    "Parental Status", sorted(X["Parental Status"].unique().tolist())
                )
            with col3:
                input_data["Children"] = number_input_from_series("Children", X, "Children")
                input_data["Days_as_Customer"] = number_input_from_series(
                    "Days as Customer", X, "Days_as_Customer"
                )

            col4, col5 = st.columns(2)
            with col4:
                input_data["Income"] = number_input_from_series(
                    "Income", X, "Income", is_float=True
                )
            with col5:
                input_data["Recency"] = number_input_from_series(
                    "Recency", X, "Recency", help_text="Days since last purchase"
                )

        with spending_tab:
            col1, col2, col3 = st.columns(3)
            spend_layout = [col1, col2, col3, col1, col2, col3]
            for container, column in zip(spend_layout, SPEND_COLUMNS):
                with container:
                    input_data[column] = number_input_from_series(
                        f"{column} spending", X, column, is_float=True
                    )

            input_data["Total_Spending"] = float(
                sum(float(input_data[column]) for column in SPEND_COLUMNS)
            )
            st.number_input(
                "Total Spending",
                min_value=0.0,
                value=input_data["Total_Spending"],
                step=100.0,
                disabled=True,
                help="Automatically calculated from Wines, Fruits, Meat, Fish, Sweets, and Gold spending.",
            )

        with channel_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                input_data["Web"] = number_input_from_series("Web purchases", X, "Web")
                input_data["NumWebVisitsMonth"] = number_input_from_series(
                    "Website visits per month", X, "NumWebVisitsMonth"
                )
            with col2:
                input_data["Catalog"] = number_input_from_series(
                    "Catalog purchases", X, "Catalog"
                )
                input_data["Discount Purchases"] = number_input_from_series(
                    "Discount purchases", X, "Discount Purchases"
                )
            with col3:
                input_data["Store"] = number_input_from_series("Store purchases", X, "Store")
                input_data["Total Promo"] = number_input_from_series(
                    "Total promo used", X, "Total Promo"
                )

        submitted = st.button("Predict Purchase Intent", type="primary")

    return pd.DataFrame([input_data]), submitted


def render_hero(df: pd.DataFrame):
    high_value_share = (df["Total_Spending"] > df["Total_Spending"].median()).mean()
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-content">
                <div class="hero-topline">
                    <div class="hero-pill">ML purchase intent studio</div>
                    <div class="hero-pill">Live customer scoring</div>
                </div>
                <h1>Customer Categorization</h1>
                <p>Score purchase intent from customer profile, spend mix, and buying channels with a campaign-ready view.</p>
                <div class="hero-stats">
                    <div class="hero-stat">
                        <span>Customer records</span>
                        <strong>{format_number(len(df))}</strong>
                    </div>
                    <div class="hero-stat">
                        <span>Median spend</span>
                        <strong>{format_currency(df["Total_Spending"].median())}</strong>
                    </div>
                    <div class="hero-stat">
                        <span>High intent base</span>
                        <strong>{high_value_share:.0%}</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_metrics(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Customers", format_number(len(df)), "Available records")
    with col2:
        metric_card(
            "Median Spend",
            format_currency(df["Total_Spending"].median()),
            "Purchase threshold",
        )
    with col3:
        metric_card("Median Income", format_currency(df["Income"].median()), "Customer base")
    with col4:
        high_value_share = (df["Total_Spending"] > df["Total_Spending"].median()).mean()
        metric_card("High Intent", f"{high_value_share:.0%}", "Above median spend")


def predict_customer(model, feature_cols, input_df: pd.DataFrame):
    try:
        input_ready = input_df[feature_cols]
    except Exception:
        input_ready = input_df.reindex(columns=feature_cols, fill_value=0)

    pred_bin = model.predict(input_ready)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_ready)[0, 1]

    return pred_bin, prob


def render_prediction(pred_bin: int | None, prob: float | None):
    if pred_bin is None:
        st.markdown(
            """
            <div class="prediction-card">
                <div class="prediction-eyebrow">Prediction</div>
                <div class="prediction-body">
                    <div>
                        <div class="prediction-title">Ready to score</div>
                        <div class="prediction-copy">
                            Complete the customer fields and run the model to see purchase intent.
                        </div>
                    </div>
                    <div class="score-ring" style="--score: 0%; --ring: #94a3b8;">
                        <div class="score-ring-inner">Run</div>
                    </div>
                </div>
                <div class="decision-strip">
                    <div class="decision-chip">Profile</div>
                    <div class="decision-chip">Spending</div>
                    <div class="decision-chip">Channels</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    will_buy = pred_bin == 1
    title = "Will Buy" if will_buy else "Will Not Buy"
    class_name = "buy" if will_buy else "no-buy"
    probability_text = f"{prob:.0%}" if prob is not None else "N/A"
    ring_score = int((prob or 0) * 100)
    ring_color = "#0f766e" if will_buy else "#e85d55"
    intent_label = "Priority lead" if will_buy else "Nurture lead"
    action_label = "Direct offer" if will_buy else "Warm-up campaign"
    confidence_label = "High confidence" if (prob or 0) >= 0.75 else "Review profile"
    copy = (
        "Strong candidate for a purchase-focused campaign."
        if will_buy
        else "Better suited for nurture, retention, or incentive testing."
    )

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-eyebrow">Prediction</div>
            <div class="prediction-body">
                <div>
                    <div class="prediction-title {class_name}">{title}</div>
                    <div class="probability">{probability_text}</div>
                    <div class="prediction-copy">{copy}</div>
                </div>
                <div class="score-ring" style="--score: {ring_score}%; --ring: {ring_color};">
                    <div class="score-ring-inner">{probability_text}</div>
                </div>
            </div>
            <div class="decision-strip">
                <div class="decision-chip">{intent_label}</div>
                <div class="decision-chip">{action_label}</div>
                <div class="decision-chip">{confidence_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prob is not None:
        st.progress(float(prob))


def render_campaign_recommendations(
    input_df: pd.DataFrame, pred_bin: int | None, prob: float | None
):
    row = input_df.iloc[0]
    spend_mix = row[SPEND_COLUMNS].astype(float)
    purchase_channels = row[["Web", "Catalog", "Store"]].astype(float)
    best_category = spend_mix.idxmax()
    best_channel = purchase_channels.idxmax()

    if pred_bin is None:
        priority = "Awaiting score"
        note = "Run prediction to set campaign priority."
    elif pred_bin == 1:
        priority = "Conversion campaign"
        note = f"Use {best_category.lower()} affinity and {best_channel.lower()} activity."
    else:
        priority = "Nurture campaign"
        note = "Start with a lower-friction offer and watch recency."

    confidence = f"{prob:.0%}" if prob is not None else "Pending"

    st.markdown(
        f"""
        <div class="recommend-grid">
            <div class="recommend-card">
                <div class="recommend-label">Best category signal</div>
                <div class="recommend-value">{best_category}</div>
                <div class="recommend-note">Highest product spend for this profile</div>
            </div>
            <div class="recommend-card">
                <div class="recommend-label">Preferred channel</div>
                <div class="recommend-value">{best_channel}</div>
                <div class="recommend-note">Strongest purchase channel input</div>
            </div>
            <div class="recommend-card">
                <div class="recommend-label">Recommended action</div>
                <div class="recommend-value">{priority}</div>
                <div class="recommend-note">{note} Confidence: {confidence}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_snapshot(input_df: pd.DataFrame, X: pd.DataFrame):
    row = input_df.iloc[0]
    spend_total = float(row["Total_Spending"])
    income = float(row["Income"])
    median_spend = float(X["Total_Spending"].median())
    median_income = float(X["Income"].median())

    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Customer Snapshot</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric(
            "Spend vs median",
            format_currency(spend_total),
            format_delta(spend_total - median_spend),
        )
        col2.metric(
            "Income vs median",
            format_currency(income),
            format_delta(income - median_income),
        )

        spend_mix = row[SPEND_COLUMNS].astype(float).sort_values(ascending=False)
        channel_mix = row[CHANNEL_COLUMNS].astype(float)
        st.caption("Spend mix")
        st.bar_chart(spend_mix, height=170)
        st.caption("Channel activity")
        st.bar_chart(channel_mix, height=170)

def render_story_flow(input_df: pd.DataFrame, pred_bin: int | None, prob: float | None):
    row = input_df.iloc[0]
    prediction = "Pending" if pred_bin is None else ("Will Buy" if pred_bin == 1 else "Will Not Buy")
    confidence = "Run model" if prob is None else f"{prob:.0%} confidence"

    st.markdown(
        f"""
        <div class="story-grid">
            <div class="story-step">
                <div class="story-number">Step 01</div>
                <div class="story-title">Customer profile</div>
                <div class="story-copy">Age {int(row["Age"])}, income {format_currency(float(row["Income"]))}, recency {int(row["Recency"])} days.</div>
            </div>
            <div class="story-step">
                <div class="story-number">Step 02</div>
                <div class="story-title">Spend calculated</div>
                <div class="story-copy">Total spending is auto-built from six product categories: {format_currency(float(row["Total_Spending"]))}.</div>
            </div>
            <div class="story-step">
                <div class="story-number">Step 03</div>
                <div class="story-title">Model decision</div>
                <div class="story-copy">{prediction} with {confidence}. The score updates from the current customer inputs.</div>
            </div>
            <div class="story-step">
                <div class="story-number">Step 04</div>
                <div class="story-title">Campaign action</div>
                <div class="story-copy">Use the strongest category and channel signals to explain the recommendation clearly.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_selected_segment(df: pd.DataFrame) -> int:
    segments = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    if not segments:
        return 0

    raw_segment = st.query_params.get("segment", segments[0])
    if isinstance(raw_segment, list):
        raw_segment = raw_segment[0]

    try:
        selected_segment = int(raw_segment)
    except (TypeError, ValueError):
        selected_segment = segments[0]

    return selected_segment if selected_segment in segments else segments[0]


def render_segment_distribution_diagram(df: pd.DataFrame, selected_segment: int):
    counts = df["cluster"].value_counts().sort_index()
    colors = ["#0f766e", "#2563eb", "#f59e0b", "#7c3aed", "#e85d55"]
    total = int(counts.sum())
    circumference = 2 * np.pi * 82
    offset = 0.0
    svg_slices = []
    legend_rows = []

    for index, (segment, count) in enumerate(counts.items()):
        segment = int(segment)
        percentage = float(count / total * 100)
        color = colors[index % len(colors)]
        dash_length = max(circumference * percentage / 100 - 2, 0)
        selected = segment == selected_segment
        opacity = "1" if selected else "0.58"
        stroke_width = "44" if selected else "36"
        svg_slices.append(
            f'<a href="?segment={segment}" target="_self">'
            f'<circle class="segment-slice" cx="120" cy="120" r="82" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" '
            f'stroke-dasharray="{dash_length:.2f} {circumference - dash_length:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 120 120)">'
            f'<title>Segment {segment}: {int(count)} customers</title>'
            "</circle></a>"
        )
        offset += circumference * percentage / 100
        selected_class = " selected" if selected else ""
        legend_rows.append(
            f'<a class="legend-link" href="?segment={segment}" target="_self">'
            f'<div class="legend-row{selected_class}">'
            f'<div class="legend-dot" style="background: {color};"></div>'
            f'<div class="legend-label">Segment {segment}</div>'
            f'<div class="legend-value">{int(count)} customers | {percentage:.0f}%</div>'
            "</div></a>"
        )

    svg_html = "".join(svg_slices)
    legend_html = "".join(legend_rows)
    st.markdown(
        f"""
        <div class="donut-layout">
            <div>
                <svg class="segment-svg" viewBox="0 0 240 240" role="img" aria-label="Clickable customer segment donut">
                    {svg_html}
                    <circle cx="120" cy="120" r="56" fill="#ffffff" stroke="rgba(15, 23, 42, 0.12)" stroke-width="1"></circle>
                    <text x="120" y="111" text-anchor="middle" fill="#64748b" font-size="13" font-weight="800">Selected</text>
                    <text x="120" y="135" text-anchor="middle" fill="#17202a" font-size="24" font-weight="850">Segment {selected_segment}</text>
                </svg>
                <div class="donut-help">Click a slice or legend row to inspect a segment.</div>
            </div>
            <div class="donut-legend">{legend_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_segment_summary(df: pd.DataFrame, selected_segment: int):
    selected_df = df[df["cluster"].astype(int) == selected_segment]
    total_customers = max(len(df), 1)
    share = len(selected_df) / total_customers
    median_spend = float(selected_df["Total_Spending"].median())
    median_income = float(selected_df["Income"].median())
    median_recency = float(selected_df["Recency"].median())

    st.markdown(
        f"""
        <div class="selected-segment-card">
            <div class="selected-segment-label">Selected customer group</div>
            <div class="selected-segment-title">Segment {selected_segment}</div>
            <div class="selected-segment-note">{len(selected_df)} customers, {share:.0%} of the dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", format_number(len(selected_df)))
    col2.metric("Median Spend", format_currency(median_spend))
    col3.metric("Median Recency", f"{median_recency:.0f} days")

    st.metric("Median Income", format_currency(median_income))

    spend_profile = (
        selected_df[SPEND_COLUMNS]
        .median()
        .astype(float)
        .sort_values(ascending=False)
        .rename("Median Spending")
    )
    st.caption("Product spending profile for selected segment")
    st.bar_chart(spend_profile, height=220)


def spending_comparison_data(input_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    row = input_df.iloc[0]
    customer = row[SPEND_COLUMNS].astype(float)
    median = X[SPEND_COLUMNS].median().astype(float)
    return pd.DataFrame(
        {
            "Category": SPEND_COLUMNS,
            "Selected Customer": customer.values,
            "Dataset Median": median.values,
        }
    ).set_index("Category")


def channel_comparison_data(input_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    row = input_df.iloc[0]
    customer = row[CHANNEL_COLUMNS].astype(float)
    median = X[CHANNEL_COLUMNS].median().astype(float)
    return pd.DataFrame(
        {
            "Channel": CHANNEL_COLUMNS,
            "Selected Customer": customer.values,
            "Dataset Median": median.values,
        }
    ).set_index("Channel")


def intent_map_data(input_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    sample = X[["Recency", "Total_Spending"]].sample(
        min(700, len(X)), random_state=42
    )
    sample = sample.copy()
    median_spend = float(X["Total_Spending"].median())
    sample["Intent Zone"] = np.where(
        sample["Total_Spending"] > median_spend,
        "Above median spend",
        "Below median spend",
    )
    sample["Point Size"] = 30

    selected = pd.DataFrame(
        {
            "Recency": [float(input_df.iloc[0]["Recency"])],
            "Total_Spending": [float(input_df.iloc[0]["Total_Spending"])],
            "Intent Zone": ["Selected customer"],
            "Point Size": [180],
        }
    )
    return pd.concat([sample, selected], ignore_index=True)


def render_visual_insights(
    df: pd.DataFrame,
    X: pd.DataFrame,
    input_df: pd.DataFrame,
    pred_bin: int | None,
    prob: float | None,
):
    st.markdown('<div class="section-title">Visual Insights</div>', unsafe_allow_html=True)
    render_story_flow(input_df, pred_bin, prob)

    segment_tab, spending_tab, channel_tab, intent_tab = st.tabs(
        ["Segment Donut", "Spend Comparison", "Channel Radar", "Intent Map"]
    )

    with segment_tab:
        selected_segment = get_selected_segment(df)
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container(border=True):
                st.markdown(
                    '<div class="section-title">Customer Segment Distribution</div>',
                    unsafe_allow_html=True,
                )
                render_segment_distribution_diagram(df, selected_segment)
        with col2:
            with st.container(border=True):
                st.markdown(
                    '<div class="section-title">Interactive Segment Summary</div>',
                    unsafe_allow_html=True,
                )
                render_selected_segment_summary(df, selected_segment)
                segment_summary = (
                    df.groupby("cluster")
                    .agg(
                        Customers=("cluster", "size"),
                        Median_Spend=("Total_Spending", "median"),
                        Median_Income=("Income", "median"),
                    )
                    .reset_index()
                )
                segment_summary["cluster"] = segment_summary["cluster"].map(
                    lambda value: f"Segment {int(value)}"
                )
                st.dataframe(
                    segment_summary,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "cluster": "Segment",
                        "Median_Spend": st.column_config.NumberColumn(
                            "Median Spend", format="$%.0f"
                        ),
                        "Median_Income": st.column_config.NumberColumn(
                            "Median Income", format="$%.0f"
                        ),
                    },
                )

    with spending_tab:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Selected Customer vs Dataset Median</div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(spending_comparison_data(input_df, X), height=420)

    with channel_tab:
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container(border=True):
                st.markdown(
                    '<div class="section-title">Channel Strength Diagram</div>',
                    unsafe_allow_html=True,
                )
                st.bar_chart(channel_comparison_data(input_df, X), height=360)
        with col2:
            with st.container(border=True):
                st.markdown(
                    '<div class="section-title">Channel Inputs</div>',
                    unsafe_allow_html=True,
                )
                channel_table = (
                    input_df.iloc[0][CHANNEL_COLUMNS]
                    .astype(float)
                    .rename("Value")
                    .reset_index()
                    .rename(columns={"index": "Channel"})
                )
                st.dataframe(channel_table, hide_index=True, use_container_width=True)

    with intent_tab:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Recency vs Spending Intent Map</div>',
                unsafe_allow_html=True,
            )
            st.scatter_chart(
                intent_map_data(input_df, X),
                x="Recency",
                y="Total_Spending",
                color="Intent Zone",
                size="Point Size",
                height=460,
            )


def main():
    st.set_page_config(
        page_title="Customer Categorization",
        layout="wide",
    )
    inject_theme()

    df = load_data()
    X, y = preprocess_data(df)
    model, feature_cols = build_fallback_model(X)

    render_sidebar(df)
    render_hero(df)
    render_dataset_metrics(df)

    st.markdown('<div class="section-title">Prediction Workspace</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        input_df, submitted = customer_input_form(X)

    pred_bin, prob = (None, None)
    if submitted:
        pred_bin, prob = predict_customer(model, feature_cols, input_df)

    with right:
        render_prediction(pred_bin, prob)
        render_customer_snapshot(input_df, X)

    render_visual_insights(df, X, input_df, pred_bin, prob)

    st.markdown('<div class="section-title">Campaign Guidance</div>', unsafe_allow_html=True)
    render_campaign_recommendations(input_df, pred_bin, prob)


if __name__ == "__main__":
    main()
