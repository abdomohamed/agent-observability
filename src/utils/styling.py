import streamlit as st

def apply_custom_styling():
    """Apply custom CSS styling for the app"""
    st.markdown("""
    <style>
        /* Main app styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Chat message styling */
        .user-message {
            background-color: #e6f7ff;
            border-left: 5px solid #1890ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .assistant-message {
            background-color: #f6f8fa;
            border-left: 5px solid #52c41a;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .system-message {
            background-color: #fff3cd;
            border-left: 5px solid #faad14;
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        
        /* Headers styling */
        h1 {
            color: #1890ff;
            font-weight: 700;
        }
        
        h2 {
            color: #333;
            font-weight: 600;
            margin-top: 1.5rem;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f6f8fa;
            border-radius: 5px 5px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e6f7ff;
            border-bottom: 2px solid #1890ff;
        }
    </style>
    """, unsafe_allow_html=True)
