"""
app.py
Streamlit entrypoint for the Advanced Nixtla App Template.
Provides a landing page with navigation to Forecasting and Portfolio Optimizer pages.
"""

import streamlit as st


def main():
    st.set_page_config(page_title="Advanced Nixtla Forecasting & RL Portfolio", layout="wide")
    # Remove practical upload limits by setting a very high max upload size (in MB).
    try:
        st.set_option("server.maxUploadSize", 5000)
    except Exception:
        pass
    st.title("Advanced Nixtla Time Series & RL Portfolio App")
    st.markdown(
        """
        **Welcome!** Use the sidebar to switch between:
        - **Forecasting**: Upload a CSV, choose date/target columns, run Nixtla models (statsforecast, mlforecast, neuralforecast), backtest, and visualize.
        - **Portfolio Optimizer**: Pull price data, generate Nixtla forecasts per asset, and train a lightweight RL agent to propose portfolio weights.
        """
    )
    st.info("Navigate using the Pages section on the left to get started.")


if __name__ == "__main__":
    # Running via `python app.py` does not create a Streamlit script context.
    # Redirect to the proper entrypoint so users don't see ScriptRunContext warnings.
    try:
        import sys
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    except Exception:
        # Fallback: still allow main to run (in bare mode it may warn, but won't crash).
        main()
