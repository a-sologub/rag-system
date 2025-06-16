"""
Dieses Modul definiert die Login-Route für eine Flask-basierte API und behandelt die Benutzerauthentifizierung.
"""

from typing import Union, Dict, Tuple

from flask import Blueprint, Response, request, current_app as app

import source.api.auth as auth

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["POST"])
def login() -> Union[Response, Tuple[Dict, int]]:
    """
    Behandelt Benutzer-Login-Anfragen.

    Diese Funktion ist verantwortlich für die Verarbeitung von POST-Anfragen an den /login-Endpunkt.
    Sie validiert die eingehenden Authentifizierungsdaten, versucht den Benutzer zu authentifizieren
    und gibt eine entsprechende Antwort zurück.

    Route: /login

    Methode: POST

    Erwartete JSON-Nutzlast:
    ```json
    {
        "username": "Benutzername des Benutzers",
        "password": "Passwort des Benutzers"
    }
    ```

    Returns:
        Union[Response, Tuple[Dict, int]]: Ein Flask-Antwortobjekt oder ein Tupel, das Folgendes enthält:

            - Bei erfolgreichem Login: Eine JSON-Nutzlast mit einem Token und einem 200-Statuscode.
            - Bei fehlgeschlagenem Login: Eine JSON-Nutzlast mit einer Fehlermeldung und einem 401-Statuscode.

    Verhalten:
        1. Protokolliert den Eingang eines Login-Versuchs.
        2. Ruft die JSON-Nutzlast aus der Anfrage ab und validiert sie.
        3. Falls die Nutzlast fehlt oder unvollständig ist, wird ein 401-Fehler zurückgegeben.
        4. Versucht, den Benutzer mit den bereitgestellten Anmeldedaten zu authentifizieren.
        5. Protokolliert das Ergebnis des Authentifizierungsversuchs.
        6. Gibt die Antwort des Authentifizierungsprozesses zurück.

    Hinweis:
        Diese Funktion verwendet die authenticate_user-Funktion aus dem auth-Modul
        zur Durchführung der eigentlichen Authentifizierungslogik.
    """
    app.logger.debug("Login attempt received")
    auth_data = request.get_json()

    if not auth_data or not auth_data.get("username") or not auth_data.get("password"):
        app.logger.warning("Login attempt with no data")
        return {"message": "No authentication data provided"}, 401

    app.logger.debug(f"Login attempt for user: {auth_data.get('username', 'unknown')}")

    response = auth.authenticate_user(auth_data)

    if isinstance(response, tuple):
        status_code = response[1]
    else:
        status_code = response.status_code

    if status_code == 200:
        app.logger.debug(
            f"Successful login for user: {auth_data.get('username', 'unknown')}"
        )
    else:
        app.logger.warning(
            f"Failed login attempt for user: {auth_data.get('username', 'unknown')}"
        )

    return response
