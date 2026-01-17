from __future__ import annotations

import streamlit as st


def apply_light_theme_css() -> None:
    """
    Apply strong light-theme overrides (no dark/black UI).
    """
    st.markdown(
        """
        <style>
        /* Force light theme and fix visibility */
        [data-testid="stAppViewContainer"] { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #f8f9fa; }
        [data-testid="stHeader"] { background-color: #ffffff; }

        /* Headings/text */
        .stMarkdown, p, span, div, label { color: #000000 !important; }
        h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }

        /* Main header */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #0066cc;
            text-align: center;
            margin-bottom: 2rem;
        }
        .role-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #006400;
            margin-top: 1rem;
        }

        /* Info boxes */
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1976d2;
            margin: 1rem 0;
            color: #000000;
        }
        .alert-box {
            background-color: #fff9e6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #ff9800;
            margin: 1rem 0;
            color: #000000;
        }
        .warning-box {
            background-color: #ffebee;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #d32f2f;
            margin: 1rem 0;
            color: #000000;
        }

        /* Buttons (light blue) */
        .stButton > button {
            background-color: #64b5f6 !important;
            color: #000000 !important;
            border: 1px solid #42a5f5 !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: #42a5f5 !important;
            border: 1px solid #2196f3 !important;
        }

        /* Selectbox and dropdown menu */
        .stSelectbox > div > div { background-color: #ffffff !important; color: #000000 !important; }
        div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #000000 !important; }
        div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #ffffff !important; color: #000000 !important; }
        ul[role="listbox"] { background-color: #ffffff !important; }
        li[role="option"] { background-color: #ffffff !important; color: #000000 !important; }
        li[role="option"]:hover { background-color: #e3f2fd !important; color: #000000 !important; }
        div[data-baseweb="select"] span { color: #000000 !important; }

        /* Number input (including +/- stepper buttons) */
        .stNumberInput input { background-color: #ffffff !important; color: #000000 !important; }
        div[data-baseweb="input"] { background-color: #ffffff !important; color: #000000 !important; border-color: #bbdefb !important; }
        div[data-baseweb="input"] input { background-color: #ffffff !important; color: #000000 !important; }
        div[data-baseweb="button-group"] button,
        button[aria-label="Increment"],
        button[aria-label="Decrement"] {
            background-color: #bbdefb !important;
            color: #000000 !important;
            border: 1px solid #90caf9 !important;
        }
        div[data-baseweb="button-group"] button:hover,
        button[aria-label="Increment"]:hover,
        button[aria-label="Decrement"]:hover {
            background-color: #90caf9 !important;
        }
        div[data-baseweb="button-group"] svg,
        button[aria-label="Increment"] svg,
        button[aria-label="Decrement"] svg {
            color: #000000 !important;
            fill: #000000 !important;
        }

        /* Streamlit chart fullscreen button */
        button[title="Fullscreen"], button[aria-label="Fullscreen"] {
            background-color: #64b5f6 !important;
            color: #000000 !important;
            border: 1px solid #42a5f5 !important;
        }
        button[title="Fullscreen"] svg, button[aria-label="Fullscreen"] svg {
            color: #000000 !important;
            fill: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

