from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
METRICS_PATH = ROOT / "reports" / "churn_model_metrics.json"

st.set_page_config(page_title="ChurnScope", page_icon="📊", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).copy()
    return data


@st.cache_data
def load_metrics() -> dict:
    return pd.read_json(METRICS_PATH, typ="series").to_dict()


df = load_data()
metrics = load_metrics()

st.title("ChurnScope")
st.caption("Analyse interactive du risque de résiliation client")

with st.sidebar:
    st.header("Filtres")
    contracts = st.multiselect(
        "Type de contrat",
        sorted(df["Contract"].unique()),
        default=sorted(df["Contract"].unique()),
    )
    churn_filter = st.multiselect(
        "Statut churn",
        ["No", "Yes"],
        default=["No", "Yes"],
        format_func=lambda value: "Resté" if value == "No" else "Churn",
    )

filtered = df[df["Contract"].isin(contracts) & df["Churn"].isin(churn_filter)]
churn_rate = (filtered["Churn"].eq("Yes").mean() * 100) if len(filtered) else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Clients affichés", f"{len(filtered):,}".replace(",", " "))
metric_cols[1].metric("Taux de churn", f"{churn_rate:.1f} %")
metric_cols[2].metric("Rappel churn", f"{float(metrics['recall']) * 100:.1f} %")
metric_cols[3].metric("ROC-AUC", f"{float(metrics['roc_auc']):.3f}")

left, right = st.columns(2)
with left:
    churn_fig = px.pie(
        filtered,
        names="Churn",
        hole=0.55,
        title="Répartition du churn",
        labels={"Churn": "Statut"},
        color="Churn",
        color_discrete_map={"No": "#2a9d8f", "Yes": "#e76f51"},
    )
    churn_fig.update_traces(hovertemplate="%{label}: %{percent}<extra></extra>")
    st.plotly_chart(churn_fig, use_container_width=True)

with right:
    contract_fig = px.histogram(
        filtered,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Clients par type de contrat",
        labels={"Contract": "Contrat", "count": "Clients", "Churn": "Statut"},
        color_discrete_map={"No": "#2a9d8f", "Yes": "#e76f51"},
    )
    st.plotly_chart(contract_fig, use_container_width=True)

left, right = st.columns(2)
with left:
    tenure_fig = px.histogram(
        filtered,
        x="tenure",
        color="Churn",
        marginal="box",
        nbins=30,
        title="Ancienneté et churn",
        labels={"tenure": "Ancienneté (mois)", "Churn": "Statut"},
        color_discrete_map={"No": "#2a9d8f", "Yes": "#e76f51"},
    )
    st.plotly_chart(tenure_fig, use_container_width=True)

with right:
    charges_fig = px.box(
        filtered,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Frais mensuels selon le statut",
        labels={"MonthlyCharges": "Frais mensuels", "Churn": "Statut"},
        color_discrete_map={"No": "#2a9d8f", "Yes": "#e76f51"},
    )
    st.plotly_chart(charges_fig, use_container_width=True)

st.subheader("Performance du modèle final")
performance = pd.DataFrame(
    {
        "Indicateur": ["Accuracy", "Précision", "Rappel churn", "F1-score", "ROC-AUC"],
        "Valeur": [
            float(metrics["accuracy"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            float(metrics["f1"]),
            float(metrics["roc_auc"]),
        ],
    }
)
performance["Pourcentage"] = performance["Valeur"] * 100
st.dataframe(performance[["Indicateur", "Pourcentage"]].style.format({"Pourcentage": "{:.1f} %"}), hide_index=True, use_container_width=True)
st.caption(f"Seuil de décision : {float(metrics['decision_threshold']):.2f} · Meilleur paramètre : C = {metrics['best_params']['model__C']}")