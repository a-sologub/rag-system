"""
Dieses Modul enthält Funktionen zur Handhabung der Benutzerauthentifizierung und Token-Verwaltung für eine
Flask-basierte API.
"""

import datetime
from datetime import timezone
from typing import Union, Tuple, Callable

import jwt
from flask import Response, jsonify, request, current_app as app


def get_settings() -> Tuple[str, str, str, int]:
    """
    Ruft Authentifizierungseinstellungen aus der Anwendungskonfiguration ab.

    Diese Funktion greift auf das 'settings'-Wörterbuch aus der Flask-Anwendungskonfiguration zu
    und extrahiert spezifische authentifizierungsbezogene Einstellungen.

    Returns:
        tuple: Ein Tupel, das (secret_key, username, password, token_hours_lifetime) enthält.

    Logs:
        Protokolliert eine Debug-Nachricht, wenn die Authentifizierungseinstellungen abgerufen wurden.
    """
    settings = app.config["settings"]
    secret_key = settings.get("authorization", "secret")
    username = settings.get("authorization", "username")
    password = settings.get("authorization", "password")
    token_hours_lifetime = settings.get("authorization", "tokenHoursLifetime")
    app.logger.debug("Authentication settings retrieved")
    return secret_key, username, password, token_hours_lifetime


def generate_token(username) -> str:
    """
    Generiert ein JWT (JSON Web Token) für den angegebenen Benutzernamen.

    Diese Funktion erstellt ein Token mit dem Benutzernamen und einer Ablaufzeit,
    signiert es mit dem geheimen Schlüssel aus den Anwendungseinstellungen.

    Args:
        username (str): Der Benutzername, der im Token kodiert werden soll.

    Returns:
        str: Das generierte JWT-Token.

    Logs:
        Protokolliert eine Debug-Nachricht mit dem Benutzernamen, für den das Token generiert wurde.
    """
    secret_key, _, _, token_hours_lifetime = get_settings()
    token = jwt.encode(
        {
            "username": username,
            "exp": datetime.datetime.now(timezone.utc)
            + datetime.timedelta(hours=token_hours_lifetime),
        },
        secret_key,
        algorithm="HS256",
    )
    app.logger.debug(f"Token generated for user: {username}")
    return token


def token_required(f) -> Callable:
    """
    Ein Dekorator, der auf ein gültiges JWT im Request-Header prüft.

    Diese Dekorator-Funktion umhüllt andere Funktionen, um sicherzustellen, dass sie nur
    mit einem gültigen Token zugänglich sind. Sie prüft auf das Vorhandensein eines Tokens
    im Authorization-Header, verifiziert dessen Gültigkeit und erlaubt der
    umhüllten Funktion fortzufahren, wenn das Token gültig ist.

    Args:
        f (function): Die zu umhüllende Funktion.

    Returns:
        function: Eine dekorierte Funktion, die die Token-Verifizierung einschließt.

    Logs:
        - Warnung, wenn das Token fehlt oder ungültig ist.
        - Debug-Nachricht, wenn das Token erfolgreich dekodiert wurde.
    """

    def decorator(*args, **kwargs) -> Union[Response, Tuple[Response, int]]:
        secret_key, _, _, _ = get_settings()
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            app.logger.warning("Token missing in request")
            return jsonify({"message": "Token is missing!"}), 401
        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, secret_key, algorithms=["HS256"])
            app.logger.debug("Token successfully decoded")
        except jwt.ExpiredSignatureError:
            app.logger.warning("Expired token used in request")
            return jsonify({"message": "Token has expired!"}), 401
        except jwt.InvalidTokenError:
            app.logger.warning("Invalid token used in request")
            return jsonify({"message": "Invalid token!"}), 401
        return f(*args, **kwargs)

    return decorator


def authenticate_user(auth) -> Union[Response, Tuple[Response, int]]:
    """
    Authentifiziert einen Benutzer basierend auf den bereitgestellten Anmeldedaten.

    Diese Funktion überprüft den bereitgestellten Benutzernamen und das Passwort gegen die
    gespeicherten Anmeldedaten. Bei erfolgreicher Authentifizierung generiert sie
    ein neues JWT-Token und gibt es zurück.

    Args:
        auth (dict): Ein Wörterbuch, das 'username' und 'password' Schlüssel enthält.

    Returns:
        tuple: Eine JSON-Antwort und ein HTTP-Statuscode.
            - Bei erfolgreicher Authentifizierung wird ein JSON-Objekt mit einem Token zurückgegeben.
            - Bei fehlgeschlagener Authentifizierung wird eine Fehlermeldung und der Statuscode 401 zurückgegeben.

    Logs:
        - Warnung bei unvollständigen Authentifizierungsdaten oder fehlgeschlagenen Versuchen.
        - Debug-Nachricht bei erfolgreicher Authentifizierung.
    """
    _, username, password, _ = get_settings()
    if not auth or not auth.get("username") or not auth.get("password"):
        app.logger.warning("Incomplete authentication data provided")
        return jsonify({"message": "Could not verify"}), 401
    if auth["username"] == username and auth["password"] == password:
        token = generate_token(auth["username"])
        app.logger.debug(f"User {auth['username']} successfully authenticated")
        return jsonify({"token": token})
    app.logger.warning(
        f"Failed authentication attempt for user: {auth.get('username')}"
    )
    return jsonify({"message": "Invalid credentials"}), 401
