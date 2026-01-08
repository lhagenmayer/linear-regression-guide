"""
Streamlit-Anwendung - 100% Plattform-Agnostisch.

Nutzt dieselbe API-Schicht wie externe Frontends (Next.js, Vite, etc.).
Dies stellt eine konsistente Logik über alle Benutzeroberflächen sicher.

Architektur:
    Streamlit App → API Layer → Core Pipeline
    
    Identisch zu:
    Next.js App → HTTP → API Layer → Core Pipeline
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional

from ...config import get_logger
from ...api import RegressionAPI, ContentAPI, AIInterpretationAPI
from ...core.domain.value_objects import SplitConfig

logger = get_logger(__name__)


def run_streamlit_app():
    """Haupt-Einstiegspunkt für die Streamlit-Anwendung."""
    # Seiten-Konfiguration
    st.set_page_config(
        page_title="Regression & Klassifikation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS für modernes UI/UX Design
    from .styles import inject_custom_css, render_hero
    inject_custom_css()
    
    # Initalisierung der APIs
    regression_api = RegressionAPI()
    content_api = ContentAPI()
    ai_api = AIInterpretationAPI()
    
    # Sidebar (Seitenleiste)
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <div class="api-badge">API-GESTÜTZT</div>
            <h1 style="font-size: 1.5rem; margin: 0;">RegAnalysis</h1>
            <p style="color: #94a3b8; font-size: 0.9rem;">Interaktive Lernplattform</p>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_type = st.radio(
            "Analysetyp",
            ["Einfache Regression", "Multiple Regression", "Binäre Klassifikation"],
            key="analysis_type",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Datensatz-Auswahl
        datasets_response = regression_api.get_datasets()
        st.markdown("### 📊 Datensatz")
        
        # Standard-Parameter
        method = "logistic"
        k_neighbors = 3
        
        if analysis_type == "Einfache Regression":
            dataset_options = {d["name"]: d["id"] for d in datasets_response["data"]["simple"]}
            dataset_name = st.selectbox("Wähle Datensatz:", list(dataset_options.keys()), key="dataset_simple")
            dataset_id = dataset_options[dataset_name]
            n_points = st.slider("Stichprobengröße", 20, 200, 50, key="n_simple")
            
        elif analysis_type == "Multiple Regression":
            dataset_options = {d["name"]: d["id"] for d in datasets_response["data"]["multiple"]}
            dataset_name = st.selectbox("Wähle Datensatz:", list(dataset_options.keys()), key="dataset_multiple")
            dataset_id = dataset_options[dataset_name]
            n_points = st.slider("Stichprobengröße", 30, 200, 75, key="n_multiple")
            
        else: # Binäre Klassifikation
            # Auswahl an Datensätzen für die Klassifikation
            cls_options = {
                "🍎 Früchte (2D)": "fruits",
                "🔢 Ziffern (64D)": "digits",
                "📱 Elektronik (Binär-Verschneidung)": "binary_electronics",
                "🏠 Immobilien (Binär-Verschneidung)": "binary_housing",
                "🏥 WHO Health (Extern)": "who_health",
                "🏦 Weltbank (Extern)": "world_bank",
            }
            dataset_name = st.selectbox("Wähle Datensatz:", list(cls_options.keys()), key="dataset_cls")
            dataset_id = cls_options[dataset_name]
            n_points = st.slider("Stichprobengröße", 50, 500, 100, step=10, key="n_cls")
            
            st.markdown("### 🧠 Modell")
            method_display = st.selectbox("Methode", ["Logistische Regression", "K-Nearest Neighbors"], key="method_select")
            method = "logistic" if "Logistische" in method_display else "knn"
            
            if method == "knn":
                k_neighbors = st.slider("Nachbarn (k)", 1, 25, 3, key="k_knn")
                
            # Konfiguration des Data-Splits (Nur bei Klassifikation relevant)
            with st.expander("Data Split & Stratification", expanded=True):
                st.markdown("Konfiguriere das Verhältnis von Trainings- zu Testdaten.")
                
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.slider(
                        "Trainings-Anteil", 
                        min_value=0.1, 
                        max_value=0.9, 
                        value=0.8, 
                        step=0.05,
                        key="train_size_slider"
                    )
                with col2:
                    stratify = st.checkbox(
                        "Stratifizierung", 
                        value=False,
                        key="stratify_checkbox",
                        help="Hält die Klassenverhältnisse in beiden Splits gleich."
                    )
                    
                # Live-Vorschau der Klassenverteilung via API
                try:
                    preview_noise_val = 0.2
                    preview_seed_val = 42
                    
                    preview = content_api.get_split_preview(
                        dataset=dataset_id, 
                        train_size=train_size,
                        stratify=stratify, 
                        seed=preview_seed_val,
                        n=n_points,
                        noise=preview_noise_val
                    )
                    
                    if preview["success"]:
                        stats = preview["stats"]
                        import pandas as pd
                        import plotly.express as px
                        
                        dist_data = []
                        for k, v in stats["train_distribution"].items():
                            dist_data.append({"Klasse": str(k), "Anzahl": v, "Set": "Train"})
                        for k, v in stats["test_distribution"].items():
                            dist_data.append({"Klasse": str(k), "Anzahl": v, "Set": "Test"})
                            
                        df_dist = pd.DataFrame(dist_data)
                        fig_dist = px.bar(
                            df_dist, x="Set", y="Anzahl", color="Klasse", barmode="group",
                            height=150, title=None
                        )
                        fig_dist.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                        st.plotly_chart(fig_dist, use_container_width=True)
                except Exception:
                    pass
        
        st.markdown("### ⚙️ Parameter")
        noise = st.slider("Rausch-Niveau", 0.0, 2.0, 0.2 if analysis_type == "Binäre Klassifikation" else 0.4, 0.1, key="noise")
        seed = st.number_input("Zufalls-Seed", 1, 9999, 42, key="seed")
        
        st.markdown("---")
        
        # API Statusanzeige
        status = ai_api.get_status()
        if status["status"]["configured"]:
            st.success("✅ KI verbunden")
        else:
            st.warning("⚠️ KI Fallback")
            
    # Hauptinhalt
    if "hero_shown" not in st.session_state:
        st.session_state.hero_shown = True
        
    # Tabs für Analyse vs. Datenexplorer
    tab_analysis, tab_data = st.tabs(["📊 Analyse", "🗃️ Daten-Explorer"])
    
    with tab_analysis:
        if analysis_type == "Einfache Regression":
            render_hero("Einfache Regression", "Erkunde Zusammenhänge, analysiere Residuen und meistere statistische Modellierung.")
            render_simple_regression(content_api, ai_api, dataset_id, n_points, noise, seed)
        elif analysis_type == "Multiple Regression":
            render_hero("Multiple Regression", "Multivariate Analyse mit interaktiven 3D-Visualisierungen.")
            render_multiple_regression(content_api, ai_api, dataset_id, n_points, noise, seed)
        else:
            render_hero("Machine Learning", "Von der logistischen Regression bis hin zur KNN-Klassifikation.")
            render_classification(content_api, ai_api, dataset_id, n_points, noise, seed, method, k_neighbors, train_size, stratify)

    with tab_data:
        st.markdown(f"### 🗃️ Rohdaten: {dataset_name}")
        st.markdown(f"**ID:** `{dataset_id}` | **Samples:** {n_points}")
        
        try:
             # Rohdaten via API laden
             raw_resp = content_api.get_dataset_raw(dataset_id)
             if raw_resp.get("success"):
                 data = raw_resp["data"]["data"]
                 columns = raw_resp["data"]["columns"]
                 
                 import pandas as pd
                 df = pd.DataFrame(data)
                 # Spalten sortieren, Target nach hinten
                 if "Target" in columns and "Target" in df.columns:
                      cols = [c for c in df.columns if c != "Target"] + ["Target"]
                      df = df[cols]
                 
                 st.dataframe(df, use_container_width=True)
                 
                 csv = df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     "📥 CSV Herunterladen",
                     csv,
                     f"{dataset_id}.csv",
                     "text/csv",
                     key='download-csv'
                 )
             else:
                 st.error(f"Daten konnten nicht geladen werden: {raw_resp.get('error')}")
        except Exception as e:
            st.error(f"Fehler im Daten-Explorer: {e}")


def render_simple_regression(
    content_api: ContentAPI,
    ai_api: AIInterpretationAPI,
    dataset: str,
    n_points: int,
    noise: float,
    seed: int
):
    """Rendert die einfache Regressionsanalyse unter Nutzung der API."""
    
    # API-Aufruf (identisch zu einem externen Frontend)
    with st.spinner("📊 Lade Daten via API..."):
        response = content_api.get_simple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
    
    if not response["success"]:
        st.error(f"API Fehler: {response.get('error', 'Unbekannter Fehler')}")
        return
    
    # Daten aus dem API-Response extrahieren
    content = response["content"]
    plots = response["plots"]
    stats = response["stats"]
    data = response["data"]
    
    # Statistiken für den Renderer aufbereiten
    stats_dict = _flatten_stats(stats, data)
    
    # Inhalte mittels StreamlitContentRenderer visualisieren
    from ..renderers import StreamlitContentRenderer
    
    renderer = StreamlitContentRenderer(
        plots={},  # Interaktive Plots werden on-demand generiert
        data={
            "x": np.array(data["x"]),
            "y": np.array(data["y"]),
            "x_label": data["x_label"],
            "y_label": data["y_label"],
        },
        stats=stats_dict
    )
    
    # Inhaltsstruktur aus der API rekonstruieren
    from ...content import SimpleRegressionContent
    content_builder = SimpleRegressionContent(stats_dict, {})
    content_obj = content_builder.build()
    
    # Rendern des edukativen Inhalts
    renderer.render(content_obj)
    
    # KI-Interpretation am Ende hinzufügen
    _render_ai_interpretation(ai_api, stats_dict)


def render_multiple_regression(
    content_api: ContentAPI,
    ai_api: AIInterpretationAPI,
    dataset: str,
    n_points: int,
    noise: float,
    seed: int
):
    """Rendert die multiple Regressionsanalyse."""
    
    # API-Aufruf
    with st.spinner("📊 Lade Daten via API..."):
        response = content_api.get_multiple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
    
    if not response["success"]:
        st.error(f"API Fehler: {response.get('error', 'Unbekannter Fehler')}")
        return
    
    # Daten extrahieren
    content = response["content"]
    plots = response["plots"]
    stats = response["stats"]
    data = response["data"]
    
    # Statistiken flach klopfen
    stats_dict = _flatten_multiple_stats(stats, data)
    
    # Rendering
    from ..renderers import StreamlitContentRenderer
    from ...content import MultipleRegressionContent
    
    renderer = StreamlitContentRenderer(
        plots={},
        data={
            "x1": np.array(data["x1"]),
            "x2": np.array(data["x2"]),
            "y": np.array(data["y"]),
            "x1_label": data["x1_label"],
            "x2_label": data["x2_label"],
            "y_label": data["y_label"],
        },
        stats=stats_dict
    )
    
    content_builder = MultipleRegressionContent(stats_dict, {})
    content_obj = content_builder.build()
    
    renderer.render(content_obj)
    
    # KI-Interpretation
    _render_ai_interpretation(ai_api, stats_dict)


def _render_ai_interpretation(ai_api: AIInterpretationAPI, stats_dict: Dict[str, Any]):
    """Rendert die KI-gestützte Interpretation der Statistiken."""
    
    st.subheader("🤖 KI-Interpretation des R-Outputs")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                padding: 1.5rem; border-radius: 1rem; color: white; margin: 1rem 0;">
        <p style="margin: 0; opacity: 0.9;">
            Lass dir alle statistischen Werte gesamtheitlich von Perplexity AI erklären.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statusabfrage über die API
    status = ai_api.get_status()
    if status["status"]["configured"]:
        st.success("✅ Perplexity API verbunden")
    else:
        st.warning("⚠️ Kein API-Key - Fallback-Interpretation wird verwendet")
        st.caption("Setze `PERPLEXITY_API_KEY` als Umgebungsvariable")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Klicke auf **"R-Output interpretieren"** für eine vollständige Erklärung:
        - Zusammenfassung des Modells
        - Interpretation der Koeffizienten
        - Bewertung der Modellgüte
        - Signifikanz-Erklärung
        - Praktische Bedeutung
        """)
    
    with col2:
        interpret_clicked = st.button(
            "🔍 Interpretieren",
            type="primary",
            use_container_width=True,
            key="ai_interpret_btn"
        )
    
    # R-Output anzeigen (Summary Statistik)
    with st.expander("📄 R-Output anzeigen"):
        r_output_response = ai_api.get_r_output(stats_dict)
        if r_output_response["success"]:
            st.code(r_output_response["r_output"], language="r")
    
    # Management des Session States für die API-Antwort
    if "ai_interpretation_result" not in st.session_state:
        st.session_state.ai_interpretation_result = None
    
    if interpret_clicked:
        with st.spinner("🤖 AI analysiert via API..."):
            # API-Aufruf (identisch zum HTMX-Vorgehen im Flask-Frontend)
            response = ai_api.interpret(stats=stats_dict, use_cache=True)
            st.session_state.ai_interpretation_result = response
    
    # Anzeige der Interpretation
    if st.session_state.ai_interpretation_result:
        response = st.session_state.ai_interpretation_result
        
        st.markdown("### 📊 Interpretation")
        
        # Hauptinhalt (Markdown von der KI)
        interpretation = response.get("interpretation", {})
        st.markdown(interpretation.get("content", "Keine Interpretation verfügbar."))
        
        # Metadaten zur Anfrage
        if response.get("success"):
            meta_cols = st.columns(4)
            meta_cols[0].caption(f"📡 {interpretation.get('model', 'N/A')}")
            meta_cols[1].caption(f"⏱️ {interpretation.get('latency_ms', 0):.0f}ms")
            
            usage = response.get("usage", {})
            if usage:
                meta_cols[2].caption(f"📝 {usage.get('total_tokens', 'N/A')} Tokens")
            
            meta_cols[3].caption(f"💾 {'Cached' if interpretation.get('cached') else 'Live'}")
        
        # Quellenangaben/Zitate
        citations = response.get("citations", [])
        if citations:
            with st.expander("📚 Quellen"):
                for i, citation in enumerate(citations, 1):
                    st.markdown(f"{i}. [{citation}]({citation})")


    renderer.render(content_obj)
    
    # AI Interpretation
    _render_ai_interpretation(ai_api, stats_dict)


def render_classification(
    content_api: ContentAPI,
    ai_api: AIInterpretationAPI,
    dataset: str,
    n_points: int,
    noise: float,
    seed: int,
    method: str,
    k_neighbors: int,
    train_size: float,
    stratify: bool
):
    """Rendert die Klassifikationsanalyse (Machine Learning)."""
    
    with st.spinner(f"🧠 Trainiere {method.upper()} Modell via API..."):
        response = content_api.get_classification_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed,
            method=method,
            k=k_neighbors,
            train_size=train_size,
            stratify=stratify
        )
        
    if not response["success"]:
        st.error(f"ML API Fehler: {response.get('error')}")
        return
        
    # Extraktion der Daten
    content_dict = response["content"]
    plots_dict = response["plots"]
    stats_dict = response["stats"]
    data_dict = response["data"]
    results_dict = response.get("results", {})
    test_metrics = results_dict.get("test_metrics")
    
    # Anzeige der Performance-Metriken (Train vs Test)
    st.markdown("### 📉 Modell-Performance")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    metrics = stats_dict # Trainings-Metriken
    
    with m_col1:
        st.metric("Genauigkeit (Train)", f"{metrics.get('accuracy',0):.2%}")
        if test_metrics:
            st.metric("Genauigkeit (Test)", f"{test_metrics.get('accuracy',0):.2%}", 
                     delta=f"{test_metrics.get('accuracy',0) - metrics.get('accuracy',0):.2%}")
            
    with m_col2:
        st.metric("Präzision (Train)", f"{metrics.get('precision',0):.2f}")
        if test_metrics:
            st.metric("Präzision (Test)", f"{test_metrics.get('precision',0):.2f}")

    with m_col3:
        st.metric("Recall (Train)", f"{metrics.get('recall',0):.2f}")
        if test_metrics:
            st.metric("Recall (Test)", f"{test_metrics.get('recall',0):.2f}")
            
    with m_col4:
        st.metric("F1 Score (Train)", f"{metrics.get('f1',0):.2f}")
        if test_metrics:
            st.metric("F1 Score (Test)", f"{test_metrics.get('f1',0):.2f}")
            
    st.markdown("---")
    
    # Rekonstruktion des Inhalts-Objekts
    from ...infrastructure.content.structure import EducationalContent
    try:
        content_obj = EducationalContent.from_dict(content_dict)
    except Exception as e:
        st.error(f"Fehler beim Laden des Inhalts: {e}")
        return
    
    # Rekonstruktion der Plots (Umwandlung von JSON in Plotly-Objekte)
    import plotly.graph_objects as go
    renderer_plots = {}
    
    if plots_dict:
        # Standard-Plots aus der PlotCollection
        for key in ["scatter", "residuals", "diagnostics"]:
           if plots_dict.get(key):
               renderer_plots[key] = go.Figure(plots_dict[key])
        
        # Zusätzliche Plots
        if plots_dict.get("extra"):
            for k, v in plots_dict["extra"].items():
                if v:
                    renderer_plots[k] = go.Figure(v)
                
    # Initialisierung des Renderers
    from ..renderers import StreamlitContentRenderer
    
    # Vorbereitung der Daten für interaktive Plots
    renderer_data = {
        "x": np.array(data_dict.get("X", [])),
        "y": np.array(data_dict.get("y", [])),
        "target_names": data_dict.get("target_names", []),
        "feature_names": stats_dict.get("feature_names", [])
    }
    
    renderer = StreamlitContentRenderer(
        plots=renderer_plots,
        data=renderer_data,
        stats=stats_dict
    )
    
    # Rendern des Inhalts und der KI-Interpretation
    renderer.render(content_obj)
    _render_ai_interpretation(ai_api, stats_dict)


if __name__ == "__main__":
    run_streamlit_app()
