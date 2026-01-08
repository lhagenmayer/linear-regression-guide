import streamlit as st

def inject_custom_css():
    """No-op: Custom CSS removed for native look."""
    pass

def render_hero(title: str, subtitle: str):
    """Render a hero section using native components."""
    st.title(title)
    st.caption(subtitle)
