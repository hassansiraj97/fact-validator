import os
import sys
import tempfile
from pathlib import Path

# Isolated, offline test environment - must be set before `app` is imported
_tmp = tempfile.mkdtemp(prefix="fv_test_")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["FACT_VALIDATOR_DATA_DIR"] = _tmp
os.environ["FACT_VALIDATOR_DB"] = str(Path(_tmp) / "test.db")
os.environ["FACT_VALIDATOR_DOMAIN_CACHE"] = str(Path(_tmp) / "domain_cache.json")
os.environ["FACT_VALIDATOR_SEARCH_CACHE"] = str(Path(_tmp) / "search_cache")
os.environ["SERPAPI_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
