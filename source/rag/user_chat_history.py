import threading
from typing import TYPE_CHECKING

from flask import Flask

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage
from langchain_core.messages import AnyMessage

class UserChatHistory:
    def __init__(self, flask_app: Flask):
        """
        Initialisiert eine Instanz der Klasse UserChatHistory.

        Args:
            flask_app (Flask): Die Flask-Anwendung, deren Logger verwendet wird.
        """
        self.chat_history: dict[str: list[AnyMessage]] = dict()
        self.lock = threading.Lock()
        self.logger = flask_app.logger

    def add_message(self, session_id: str, message: AnyMessage):
        """
        Fügt eine Nachricht zum Chat-Verlauf für eine bestimmte Sitzung hinzu.

        Args:
            session_id (str): Die eindeutige ID der Chatsitzung.
            message (str): Die hinzuzufügende Nachricht.

        Returns:
            None
        """
        with self.lock:
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []

            self.chat_history[session_id].append(message)

    def get_messages(self, session_id):
        """
        Gibt die Nachrichten für einen bestimmten Benutzer zurück.

        Args:
            session_id (str): Die eindeutige ID der Chatsitzung.

        Returns:
            list: Eine Liste mit den Nachrichten des Benutzers.
        """
        with self.lock:
            return self.chat_history.get(session_id, [])