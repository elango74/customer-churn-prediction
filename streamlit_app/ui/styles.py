import streamlit as st

def load_background():
    st.markdown(
        """
        <style>
        body {
            background: radial-gradient(circle at top, #0f2027, #203a43, #2c5364);
            overflow-x: hidden;
        }

        .particle {
            position: fixed;
            width: 6px;
            height: 6px;
            background: rgba(255,255,255,0.18);
            border-radius: 50%;
            animation: float 25s linear infinite;
            z-index: -1;
        }

        @keyframes float {
            from { transform: translateY(110vh); }
            to   { transform: translateY(-120vh); }
        }
        </style>

        <div class="particle" style="left:5%; animation-duration:22s;"></div>
        <div class="particle" style="left:15%; animation-duration:30s;"></div>
        <div class="particle" style="left:35%; animation-duration:26s;"></div>
        <div class="particle" style="left:55%; animation-duration:34s;"></div>
        <div class="particle" style="left:75%; animation-duration:28s;"></div>
        <div class="particle" style="left:90%; animation-duration:24s;"></div>
        """,
        unsafe_allow_html=True
    )
