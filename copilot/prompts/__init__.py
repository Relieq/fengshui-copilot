from pathlib import Path

def load_prompt(package_path: Path, name: str) -> str:
    p = package_path / name
    return p.read_text(encoding="utf-8")
