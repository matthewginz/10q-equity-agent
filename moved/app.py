# Redirect stub deployed on the old subdomain (10q-equity-agent-lu7wjqswdyw7catzwyzq9j)
# so résumés already sent with that link still reach the app.
import streamlit as st

NEW_URL = "https://10q-equity-agent.streamlit.app"

st.set_page_config(page_title="10-Q Equity Research Agent has moved", page_icon="\U0001F4C8")
st.title("This app has moved")
st.write(f"The 10-Q Equity Research Agent now lives at **{NEW_URL}**")
st.link_button("Open the app", NEW_URL, type="primary")
