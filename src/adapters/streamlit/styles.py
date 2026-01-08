import streamlit as st

def inject_custom_css():
    """Inject custom CSS for the 'Midnight' state-of-the-art theme."""
    
    st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* === GLOBAL VARIABLES === */
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #ec4899;
            --accent: #06b6d4;
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }
        
        /* === APP CONTAINER === */
        .stApp {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }
        
        /* === TYPOGRAPHY === */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: var(--text-primary) !important;
        }
        
        p, div, span, label {
            font-family: 'Inter', sans-serif;
        }
        
        code {
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* === SIDEBAR === */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.8) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        div[data-testid="stSidebarNav"] {
            display: none; /* Hide default nav if we handle it differently */
        }
        
        /* === WIDGETS === */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: rgba(30, 41, 59, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white !important;
        }
        
        .stSlider div[data-baseweb="slider"] {
            /* Custom slider styles depend on internals, keep simple for now */
        }
        
        /* === BUTTONS === */
        div.stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white !important;
            border: none !important;
            border-radius: 0.75rem !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            font-family: 'Outfit', sans-serif !important;
            box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
            transition: all 0.2s ease !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6);
            filter: brightness(110%);
        }
        
        div.stButton > button:active {
            transform: translateY(1px);
        }
        
        /* Secondary Button (Outline) - Hacky via :nth-child if needed, or specific keys */
        
        /* === METRICS === */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            padding: 1rem;
            backdrop-filter: blur(10px);
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
        }
        
        /* === CARDS / CONTAINERS === */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* === CUSTOM COMPONENT CLASSES === */
        .api-badge {
            background: rgba(99, 102, 241, 0.2);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.4);
            padding: 0.25rem 0.75rem;
            border-radius: 2rem;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
        }
        
        /* === PLOTLY OVERRIDES === */
        .js-plotly-plot .plotly .modebar {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_hero(title: str, subtitle: str):
    """Render a hero section."""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
