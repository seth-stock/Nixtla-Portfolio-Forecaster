"""
Optional Streamlit entrypoint for the Nixtla Portfolio Forecaster.

The packaged Windows build still launches from app.py. This file exists so
source users can run the Streamlit UI without conflicting with the desktop
launcher.
"""

import streamlit as st


def main():
    st.set_page_config(page_title="Nixtla Portfolio Forecaster (Streamlit)", layout="wide")
    try:
        st.set_option("server.maxUploadSize", 5000)
    except Exception:
        pass

    st.title("Nixtla Portfolio Forecaster")
    st.markdown(
        """
        Choose a runtime:

        - **Streamlit source app**: use the pages sidebar for Forecasting and Portfolio Optimizer.
        - **Windows desktop app**: launch `python app.py` or use the installed desktop shortcut.
        """
    )
    st.info("This is the optional Streamlit interface. The packaged Windows build remains the default non-developer runtime.")


if __name__ == "__main__":
    try:
        import sys
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    except Exception:
        main()
