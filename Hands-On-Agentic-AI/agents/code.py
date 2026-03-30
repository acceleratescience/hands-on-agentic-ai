from pathlib import Path
import subprocess

SANDBOX_DIR = Path("./sandbox").resolve()

def write_file(filename: str, content: str) -> str:
    path = (SANDBOX_DIR / filename).resolve()

    if SANDBOX_DIR not in path.parents:
        raise ValueError("Refusing to write outside sandbox")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return str(path.relative_to(SANDBOX_DIR))


def run_command(command: str, timeout: int = 60) -> dict:
    result = subprocess.run(
        command,
        cwd=SANDBOX_DIR,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    return {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
