import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("HackStart", ["Problem statement 1", 'Problem statement 2'],
                           icons=['activity', 'activity'], menu_icon="code-slash", default_index=1)

if selected == "Problem statement 1":
    st.subheader("Hackstart Competition")
    st.info('''### Problem statement 1
The first problem statement consists of two parts:
1. Using data analytics **create a framework** to match the startups of Tamil Nadu sector-wise.
2. Given such data of Startups With Startup TN which in years will be scaled exponentially, **propose an architecture and implementable solution** that can be used to:
    - Maintain a digital snapshot of the Startup Database
    - Track the progress of Startup TN startups''')
    st.write("## Solution")
