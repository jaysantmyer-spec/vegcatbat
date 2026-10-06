"""Executes app.py top-to-bottom against a stub Streamlit (catches data/logic errors in the UI code)."""
import datetime as dt
import os
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRESS = set(os.getenv("PRESS", "").split(","))  # labels of buttons to 'click'


class D:
    def __getattr__(self, k): return D()
    def __call__(self, *a, **k): return D()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([])
    def __bool__(self): return False


def make_st():
    st = types.ModuleType("streamlit")
    d = D()
    for name in ("markdown", "write", "title", "subheader", "info", "warning", "error", "success", "caption",
                 "metric", "dataframe", "line_chart", "bar_chart", "divider", "download_button", "set_page_config"):
        setattr(st, name, lambda *a, **k: None)
    st.sidebar = d
    st.spinner = lambda *a, **k: D()
    st.expander = lambda *a, **k: D()
    st.progress = lambda *a, **k: D()
    def cache(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
    cache.clear = lambda: None
    st.cache_data = cache
    st.cache_resource = cache
    class Col:
        def __getattr__(self, k): return getattr(st, k)
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda spec, **k: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [D() for _ in labels]
    st.selectbox = lambda label, options, index=0, **k: list(options)[index] if len(list(options)) else None
    st.radio = lambda label, options, **k: list(options)[0]
    st.select_slider = lambda label, options=(), value=None, **k: value
    st.slider = lambda label, lo, hi, value, *a, **k: value
    st.checkbox = lambda label, value=False, **k: value
    st.text_input = lambda label, value="", **k: value
    st.json = lambda *a, **k: None
    st.toast = lambda *a, **k: None
    st.multiselect = lambda label, options=(), **k: []
    st.rerun = lambda: None
    st.text_area = lambda label, value="", **k: value
    st.date_input = lambda label, value=None, **k: value
    st.number_input = lambda label, lo=None, hi=None, value=0, **k: value
    st.file_uploader = lambda *a, **k: None
    st.button = lambda label, **k: label in PRESS
    st.session_state = {}
    return st


sys.modules["streamlit"] = make_st()
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
print("app.py executed OK, root:", os.environ.get("HB_ORACLE_ROOT", "repo"))
import resource
print("peak memory MB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
