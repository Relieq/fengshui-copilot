from pathlib import Path
from copilot.prompts import load_prompt

package_path = Path(__file__).resolve().parent
JUDGE_SYSTEM_PROMPT = load_prompt(package_path, "judge_system_prompt.txt")
JUDGE_HUMAN_PROMPT = load_prompt(package_path, "judge_human_prompt.txt")