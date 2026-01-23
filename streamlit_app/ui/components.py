import streamlit as st

def header():
    st.markdown(
        """
        <div style="padding:30px; border-radius:20px;
                    background:rgba(255,255,255,0.12);
                    box-shadow:0 10px 30px rgba(0,0,0,0.4);">
            <h1 style="color:white;">Customer Churn Prediction</h1>
            <p style="color:#e0e0e0; font-size:18px;">
                Predict churn probability and get intelligent retention action
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def footer():
    st.markdown(
        """
        <hr style="margin-top:50px;">
        <p style="text-align:center; color:#cccccc;">
            Customer Churn Prediction • Streamlit ML App <br>
            Built using Machine Learning <br>
            <b>By Elango E – ClaySYS Project</b>
        </p>
        """,
        unsafe_allow_html=True
    )
