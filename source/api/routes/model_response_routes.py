"""Dieses Modul definiert eine Flask-Route für die Generierung von Antworten basierend auf Benutzeranfragen."""
import uuid
from typing import ClassVar

from flask import Blueprint, Response, jsonify, request, current_app as app
from flask.views import MethodView

from source.api import auth
from source.model.agent import Agent

bp = Blueprint("generate-response", __name__)


class GenerateResponseView(MethodView):
    """Eine Klasse zur Verarbeitung von POST-Anfragen für die Generierung von Antworten.

    Diese Klasse verwendet einen Agenten und ein KI-Modell, um Antworten auf Benutzeranfragen zu generieren.
    Sie implementiert eine asynchrone POST-Methode und verwendet Streaming für die Antwortgenerierung.

    Attribute:
        decorators (list): Eine Liste von Dekoratoren, die auf die post-Methode angewendet werden.

    Methoden:
        post: Verarbeitet POST-Anfragen zur Antwortgenerierung.
    """

    decorators: ClassVar = [auth.token_required]

    @staticmethod
    async def post() -> Response | tuple[Response, int]:
        """Verarbeitet POST-Anfragen zur Generierung von Antworten.

        Diese Methode empfängt eine Benutzeranfrage, verarbeitet sie mithilfe eines Agenten und eines KI-Modells
        und gibt die generierte Antwort als Stream zurück.

        Returns:
            Response: Ein Streaming-Response-Objekt mit der generierten Antwort.

        Raises:
            KeyError: Wenn die erforderliche Eigenschaft "query" oder "sessionId" fehlt.
            ValueError: Wenn eine leere "query" oder "sessionId" angegeben wird.
            Exception: Bei allgemeinen Fehlern während der Verarbeitung.
        """
        try:
            data: dict[str, any] = request.get_json()
            if not data:
                raise ValueError("Empty JSON request body")

            for key in ["sessionId", "query"]:
                value = data.get(key)
                if value is None:
                    raise KeyError(key)
                if not value:
                    raise ValueError(f"No {key} provided")

        except KeyError as e:
            missing_key = e.args[0]  # Den fehlenden Schlüssel aus der Ausnahme herausziehen
            app.logger.warning(f"Request missing required property: '{missing_key}'")
            return jsonify({"message": f"Missing required property: '{missing_key}'"}), 422

        except ValueError as e:
            app.logger.warning(str(e))
            return jsonify({"message": str(e)}), 422

        except Exception as e:
            app.logger.error(f"Unexpected error while processing request: {e!r}")
            return jsonify({"message": "Unsupported Media Type"}), 415

        try: # Überprüfen, ob die SessionID-Struktur dem Format einer UUID entspricht
            uuid_obj = uuid.UUID(data.get("sessionId"), version=4)
            if not str(uuid_obj) == data.get("sessionId"):
                raise ValueError("Invalid session ID format")

        except ValueError as e:
            app.logger.error(f"Session ID validation failed: {e!r}")
            return jsonify({"message": str(e)}), 422

        user_agent = request.headers.get("User-Agent", "").lower()
        referer = request.headers.get("Referer", "").lower()

        # Überprüft, ob der Benutzer eine Anfrage per Chat sendet oder nicht. (Relevant für Langsmith)
        if ("mozilla" in user_agent or "chrome" in user_agent or "safari" in user_agent) and ("/chat" in referer):
            langsmith_client_name = "ChatUI"
        else:
            langsmith_client_name = "APIRequest"

        agent = Agent(
            use_langsmith=bool(app.config["settings"].get("langSmithSettings", "useLangsmithTestEnvironment")),
            langsmith_client_name=langsmith_client_name,
        )
        app.logger.info("Received request to generate response")
        response = Response(
            await agent(session_id=data.get("sessionId"), message=data.get("query")),
            mimetype="text/event-stream",
        )

        app.logger.info("Sending response stream")
        return response


bp.add_url_rule(
    "/generate-response",
    view_func=GenerateResponseView.as_view("generate_response"),
    methods=["POST"],
    endpoint="generate-response",
)
