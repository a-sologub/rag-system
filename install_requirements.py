"""Dieses Modul stellt Funktionalität zur Installation erforderlicher Pakete für ein Projekt bereit."""

import argparse
import subprocess
import sys


def install_packages(dev: bool = False) -> None:
    """Installiert die erforderlichen Pakete für das Projekt.

    Args:
        dev (bool): Wenn True, werden auch Entwicklungspakete installiert.
    """
    # Definition der benutzerdefinierten Paketquellen
    index_url = "https://pypi.org/simple"
    extra_index_urls = [
        "https://download.pytorch.org/whl/cu124",
        "https://abetlen.github.io/llama-cpp-python/whl/cu125",
    ]
    # Definition der Hauptpakete zur Installation
    packages = [
        "PyJWT==2.8.0",
        "PyMuPDF==1.24.7",
        "Werkzeug==3.0.3",
        "aiofiles==24.1.0",
        "flask-cors==5.0.0",
        "flask[async]==3.0.3",
        "langchain-core==0.3.32",
        "llama_cpp_python==0.3.7",
        "mkdocs-glightbox==0.4.0",
        "mkdocs-material==9.5.40",
        "mkdocs==1.6.1",
        "mkdocstrings-python==1.12.2",
        "mkdocstrings==0.26.2",
        "numpy==2.2.2",
        "pymongo==4.8.0",
        "scikit-learn==1.5.1",
        "sentence-transformers==3.0.1",
        "spacy==3.8.3",
        "tiktoken==0.7.0",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm==4.66.4",
    ]
    # Definition der Entwicklungspakete zur Installation
    dev_packages = [
        "jupyter==1.0.0",
        "pytest==8.2.2",
        "ruff==0.7.0",
    ]
    # Erstellen des pip-Installationsbefehls
    command = [sys.executable, "-m", "pip", "install", "--index-url", index_url]
    for url in extra_index_urls:
        command.extend(["--extra-index-url", url])
    command.extend(packages)
    # Hinzufügen der Entwicklungspakete, falls dev-Flag gesetzt ist
    if dev:
        command.extend(dev_packages)
    # Ausführen des pip-Installationsbefehls und Ausgabe in Echtzeit
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
        # Prüfen auf Fehler
        rc = process.poll()
        if rc != 0:
            print("Error installing packages.")
            for error in process.stderr:
                print(error.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install required packages.")
    parser.add_argument(
        "--dev", action="store_true", help="Include development packages."
    )
    args = parser.parse_args()
    install_packages(dev=args.dev)
