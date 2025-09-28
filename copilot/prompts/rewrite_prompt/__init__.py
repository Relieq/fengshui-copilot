from pathlib import Path
from copilot.prompts import load_prompt

package_path = Path(__file__).resolve().parent
REWRITE_SYSTEM_PROMPT = load_prompt(package_path, "rewrite_system_prompt.txt")
REWRITE_HUMAN_PROMPT = load_prompt(package_path, "rewrite_human_prompt.txt")