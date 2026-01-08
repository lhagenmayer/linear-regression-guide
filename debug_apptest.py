
from streamlit.testing.v1 import AppTest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))

at = AppTest.from_file("run.py")
at.run()
print(f"TITLE: '{at.title[0].value}'")
print(f"SUBHEADERS: {[h.value for h in at.subheader]}")
print(f"RADIO: {at.sidebar.radio[0].value}")
print(f"SELECTBOXES: {[s.label for s in at.selectbox]}")
