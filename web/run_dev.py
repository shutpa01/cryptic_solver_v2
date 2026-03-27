"""Development server launcher."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import web` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web import create_app

app = create_app("development")

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
