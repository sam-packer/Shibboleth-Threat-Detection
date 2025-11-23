import os
import platform
import logging
import subprocess
from helpers.globals import cfg
from web.app import app, preflight


def main():
    """Run API in production mode with appropriate WSGI server."""
    if not preflight():
        logging.error("Preflight checks failed! Aborting startup.")
        return


    host = cfg("api.host")
    port = cfg("api.port")
    workers = cfg("api.workers", 2)
    threads = cfg("api.threats", 4)
    log_level = cfg("api.log_level", "info")

    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logging.basicConfig(
        level=log_level.upper(),
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

    logging.info(
        f"Launching API on {host}:{port} (debug={debug_mode}) "
        f"workers={workers} threads={threads}"
    )

    system = platform.system().lower()

    # use waitress on windows
    if "windows" in system:
        try:
            from waitress import serve
        except ImportError:
            logging.error("Waitress is not installed. Run: uv add waitress")
            return

        logging.info("Using Waitress WSGI server (Windows mode)")
        serve(app, host=host, port=port)

    # use gunicorn on linux
    else:
        logging.info("Using Gunicorn WSGI server (Unix mode)")
        cmd = [
            "gunicorn",
            "--bind", f"{host}:{port}",
            "--workers", str(workers),
            "--threads", str(threads),
            "web.app:app",
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
