import os
import platform
import logging
import subprocess
from web.app import app, preflight


def main():
    """Run API in production mode with appropriate WSGI server."""
    if not preflight():
        logging.error("Preflight checks failed! Aborting startup.")
        return

    port = int(os.getenv("PORT", 5001))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    logging.info(f"Launching API on port {port} (debug={debug_mode})")

    system = platform.system().lower()

    # use waitress on windows
    if "windows" in system:
        try:
            from waitress import serve
        except ImportError:
            logging.error("Waitress is not installed. Run: uv add waitress")
            return

        logging.info("Using Waitress WSGI server (Windows mode)")
        serve(app, host="0.0.0.0", port=port)

    # use gunicorn on linux
    else:
        logging.info("Using Gunicorn WSGI server (Unix mode)")
        cmd = [
            "gunicorn",
            "--bind", f"0.0.0.0:{port}",
            "--workers", str(os.getenv("GUNICORN_WORKERS", 2)),
            "--threads", str(os.getenv("GUNICORN_THREADS", 4)),
            "web.app:app",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
