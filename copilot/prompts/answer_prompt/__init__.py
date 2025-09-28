from pathlib import Path
from copilot.prompts import load_prompt

package_path = Path(__file__).resolve().parent
ANSWER_SYSTEM_PROMPT = load_prompt(package_path, "answer_system_prompt.txt")
ANSWER_HUMAN_PROMPT = load_prompt(package_path, "answer_human_prompt.txt")