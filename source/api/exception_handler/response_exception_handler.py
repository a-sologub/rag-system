"""Dieses Modul enthält eine zentrale Funktion zur Behandlung von Ausnahmen in einer Flask-basierten API."""

import json
from typing import Union

from flask import make_response, current_app as app
from werkzeug.exceptions import HTTPException


def response_exception(e) -> Union[make_response, tuple]:
    """
    Behandelt Ausnahmen und gibt entsprechende JSON-Antworten zurück.

    Diese Funktion dient als globaler Ausnahmehandler, verarbeitet sowohl HTTP-
    als auch Nicht-HTTP-Ausnahmen und formatiert sie in JSON-Antworten. Sie
    protokolliert auch die Ausnahmedetails für Debugging-Zwecke.

    Args:
        e (Exception): Die aufgetretene Ausnahme, die behandelt werden muss.

    Returns:
        Union[make_response, tuple]: Ein Flask-Antwortobjekt, das eine JSON-Nutzlast
        mit Fehlerdetails und einem angemessenen HTTP-Statuscode enthält.

    Verhalten:
        1. Für HTTPExceptions (z.B. 404 Not Found, 403 Forbidden):
           - Erstellt eine JSON-Nutzlast mit dem Ausnahmenamen.
           - Setzt den Content-Type der Antwort auf 'application/json'.
           - Protokolliert eine Warnung mit dem Ausnahmenamen und -code.
           - Protokolliert Debug-Informationen mit vollständigen Ausnahmedetails.
           - Gibt eine Antwort mit dem ursprünglichen HTTP-Statuscode zurück.
        2. Für alle anderen Ausnahmen:
           - Erstellt eine JSON-Nutzlast mit einer generischen Fehlermeldung und Ausnahmedetails.
           - Protokolliert einen Fehler mit dem Ausnahmetyp.
           - Protokolliert den vollständigen Ausnahme-Traceback auf Fehlerebene.
           - Gibt eine Antwort mit einem 500 Internal Server Error Statuscode zurück.

    Hinweis:
        Diese Funktion geht davon aus, dass 'app' (Flask-Anwendungsinstanz) im
        globalen Bereich für Logging-Zwecke verfügbar ist.
    """

    if isinstance(e, HTTPException):
        # HTTP-Fehler behandeln
        response = e.get_response()
        payload = json.dumps({"message": e.name})
        response.data = f"{payload}\n"
        response.content_type = "application/json"

        app.logger.warning(f"HTTP exception occurred: {e.name} (code: {e.code})")
        app.logger.debug(f"Exception details: {str(e)}")

        return make_response(response, e.code)
    else:
        # Nicht-HTTP-Ausnahmen behandeln
        response_body = {"message": "unhandled exception occurred", "details": str(e)}

        app.logger.error(f"Unhandled exception occurred: {type(e).__name__}")
        app.logger.exception("Exception details:", exc_info=e)

        return make_response(response_body, 500)
