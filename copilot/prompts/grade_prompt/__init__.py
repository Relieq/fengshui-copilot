from pathlib import Path
from copilot.prompts import load_prompt

package_path = Path(__file__).resolve().parent
GRADE_SYSTEM_PROMPT = load_prompt(package_path, "grade_system_prompt.txt")
GRADE_HUMAN_PROMPT = load_prompt(package_path, "grade_human_prompt.txt")