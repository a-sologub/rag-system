"""
Dieses Modul definiert eine einfache Begrüßungsroute für eine Flask-basierte API. Es dient hauptsächlich als
Gesundheitscheck-Endpunkt und für Pipeline-Tests.
"""

from flask import Blueprint, Response, jsonify, current_app as app

bp = Blueprint("greeting", __name__)


@bp.route("/greeting", methods=["GET"])
def get_greetings() -> Response:
    """
    Liefert eine einfache Begrüßungsantwort.

    Diese Route wird hauptsächlich für Pipeline-Tests verwendet und dient als grundlegender
    Gesundheitscheck-Endpunkt für die Anwendung.

    Route: /greeting

    Methode: GET

    Returns:
        Response: Eine JSON-Antwort, die eine Begrüßungsnachricht enthält.

    Erwartete JSON-Nutzlast:
    ```json
    {
    "greeting": "Hello world"
    }
    ```

    Verhalten:
        1. Protokolliert den Empfang einer Begrüßungsanfrage.
        2. Erstellt eine JSON-Antwort mit einer statischen Begrüßungsnachricht.
        3. Protokolliert den Inhalt der zu sendenden Antwort.
        4. Gibt die Antwort zurück.

    Hinweise:
        - Dieser Endpunkt erfordert keine Authentifizierung.
        - Er gibt immer die gleiche statische Begrüßung zurück, was ihn nützlich macht für
          grundlegende API-Verfügbarkeitsprüfungen und Pipeline-Tests.
        - Die Protokollierung des Antworteninhalts ermöglicht eine einfache Überprüfung
          des Endpunktverhaltens in den Logs.
    """
    app.logger.debug("Greeting request received")
    response = jsonify({"greeting": "Hello world"})
    app.logger.debug("Greeting response sent: %s", response.get_data(as_text=True))
    return response
