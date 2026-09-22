import streamlit as st
from config import PAGES

pg = st.navigation(
    [
        st.Page(p["path"], title=p["name"], icon=p["icon"], default=(p["name"] == "Home"))
        for p in PAGES
    ]
)
pg.run()
