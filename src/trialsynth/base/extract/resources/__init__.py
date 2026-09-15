import json
from pathlib import Path

HERE = Path(__file__).parent
TRIAL_RESULT_SCHEMA_JSON = HERE / "trial_result_schema_anchor.json"
PROMPT_PATH = HERE / "prompt.txt"
TRIAL_RESULT_SCHEMA_ANCHOR: dict = json.loads(TRIAL_RESULT_SCHEMA_JSON.read_text())
PROMPT = PROMPT_PATH.read_text().strip()

CTGOV_RESULT_SCHEMA_JSON = HERE / "ctgov_result_schema_anchor.json"
CTGOV_PROMPT_PATH = HERE / "ctgov_prompt.txt"
CTGOV_RESULT_SCHEMA_ANCHOR: dict = json.loads(CTGOV_RESULT_SCHEMA_JSON.read_text())
CTGOV_PROMPT = CTGOV_PROMPT_PATH.read_text().strip()
