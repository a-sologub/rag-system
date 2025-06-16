"""Hauptanwendungsdatei für die Flask-Server.

Diese Datei richtet die Flask-Anwendung ein und konfiguriert sie, einschließlich:

- Logging-Konfiguration
- Datenbankverbindung
- Dienstinitialisierung (Textvorverarbeitung, Keywords-generator, Vektorerstellung, Modellladung)
- Routenregistrierung
- Fehlerbehandlung

Die Anwendung verwendet eine modulare Struktur mit Blueprints für verschiedene Routenkategorien.

Funktionen:
    - setup_logging: Konfiguriert das Logging für die Flask-Anwendung.
    - create_app: Erstellt und konfiguriert die Flask-Anwendung.
    - shutdown_session: Räumt Ressourcen auf, wenn die Anwendung heruntergefahren wird.
    - index: Rendert die Startseite der Anwendung.
    - chat: Rendert die Chat-Seite der Anwendung.

Die Anwendung ist so konzipiert, dass sie als eigenständiger Server ausgeführt oder
importiert und mit einem WSGI-Server verwendet werden kann.

Verwendung:
    python -m source.app

Umgebungsvariablen:
    PORT: Die Portnummer, auf der die Flask-App laufen soll (Standard: 11891)

Abhängigkeiten:
    - Flask
    - MongoDB (über MongoDBManager)
    - benutzerdefinierte Module für Textvorverarbeitung, Keywords-generator, Vektorerstellung und Modellladung
"""

import atexit
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

import argparse
from flask import Flask, render_template, current_app

from source.api.exception_handler import response_exception
from source.api.routes import model_response_routes, auth_routes, greeting_routes, knowledgebase_routes
from source.db.mongodb_manager import MongoDBManager
from source.model.model_loader import ModelLoader
from source.model.system_prompt_loader import SystemPrompt
from source.preprocess.keywords_generator import KeywordsGenerator
from source.preprocess.text_preprocessor import TextPreprocessor
from source.preprocess.vector_creator import VectorCreator
from source.rag.user_chat_history import UserChatHistory
from source.settings_loader import SettingsLoader
from source.test_environment.automated_question_testing import run_automated_tests_in_langsmith


def setup_logging(flask_app: Flask) -> None:
    """Richtet das Logging für die Flask-Anwendung ein.

    Args:
        flask_app (Flask): Die Flask-Anwendungsinstanz.

    Diese Funktion konfiguriert einen RotatingFileHandler für das Logging und richtet
    formatierte Lognachrichten ein, die Zeitstempel, Loglevel, Thread-Info,
    Funktionsname, Zeilennummer und die Lognachricht selbst enthalten.
    """
    if not flask_app.debug:
        flask_app.logger.setLevel(logging.INFO)

    # Überprüft, ob der "logs" Ordner existiert und erstellt ihn, falls nicht
    log_dir = "logs"
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)

    # Erstellt einen File-Handler für das Logging
    file_handler = RotatingFileHandler(Path(log_dir) / "log.log", maxBytes=10240, backupCount=10)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(levelname)s\t(TID %(thread)d %(threadName)s)\t%(funcName)s:%(lineno)d\t%(message)s")
    )
    if not flask_app.debug:
        file_handler.setLevel(logging.INFO)
    # Fügt den File-Handler zum Logger der App hinzu
    flask_app.logger.addHandler(file_handler)


def create_app() -> Flask:
    """Erstellt und konfiguriert die Flask-Anwendung.

    Returns:
        Flask: Die konfigurierte Flask-Anwendungsinstanz.

    Diese Funktion führt die folgenden Schritte aus:
        1. Erstellt eine neue Flask-Anwendungsinstanz
        2. Richtet das Logging ein
        3. Lädt die Anwendungseinstellungen
        4. Initialisiert Dienste (Datenbank, Textvorverarbeitung, Vektorerstellung, Modell)
        5. Registriert Blueprints für verschiedene Routenkategorien
        6. Richtet einen globalen Fehlerhandler ein
        7. Speichert Dienstinstanzen in der Anwendungskonfiguration

    Protokolliert den Fortschritt jedes wichtigen Schritts im Anwendungseinrichtungsprozess.
    """
    flask_app = Flask(__name__)

    # Logging einrichten
    setup_logging(flask_app)

    flask_app.logger.info("Starting to create Flask app")

    # Einstellungen laden
    settings = SettingsLoader(flask_app=flask_app)
    flask_app.logger.debug("Settings loaded")

    if (
            bool(settings.get("langSmithSettings", "useLangsmithTestEnvironment"))
            and str(settings.get("langSmithSettings", "langchainApiKey")) == ""
    ):
        flask_app.logger.error("LangSmith tool is enabled, but langchain_api_key variable is not set")
        raise RuntimeError("LangSmith tool is enabled, but langchain_api_key variable is not set")

    # Dienste initialisieren
    flask_app.logger.debug("Initializing services")
    text_preprocessor = TextPreprocessor(
        flask_app=flask_app,
        stop_words_file_path=settings.get("documentKeywordExtractionSettings", "stopWordsFilePath"),
    )

    db_manager = MongoDBManager(
        flask_app=flask_app,
        client_uri=settings.get("mongodbConnectionSettings", "client"),
        username=settings.get("mongodbConnectionSettings", "username"),
        password=settings.get("mongodbConnectionSettings", "password"),
        db_name=settings.get("mongodbConnectionSettings", "database"),
        knowledge_collection_name=settings.get("mongodbConnectionSettings", "collectionKnowledgebase"),
        vector_collection_name=settings.get("mongodbConnectionSettings", "collectionVector"),
    )

    vector_creator = VectorCreator(
        flask_app=flask_app,
        model_name=settings.get("documentRetrievalSettings", "textToVectorTransformerModel"),
    )

    model = ModelLoader(
        flask_app=flask_app,
        model_path=settings.get("generativeModelSettings", "modelPath"),
        n_gpu_layers=settings.get("generativeModelSettings", "nGpuLayers"),
        n_ctx=settings.get("generativeModelSettings", "maxLengthContext"),
        flash_attn=settings.get("generativeModelSettings", "flashAttention"),
        verbose=settings.get("generativeModelSettings", "verbose"),
        repetition_penalty=settings.get("generativeModelSettings", "repetitionPenalty"),
        temperature=settings.get("generativeModelSettings", "temperature"),
        top_k=settings.get("generativeModelSettings", "topK"),
        top_p=settings.get("generativeModelSettings", "topP"),
    )

    system_prompts = SystemPrompt(
        flask_app=flask_app,
        llm_model=model,
        max_system_prompt_length=int(settings.get("generativeModelSettings", "maxSystemPromptLength")),
        max_chat_history_length=int(settings.get("generativeModelSettings", "maxChatHistoryLength")),
        rag_prompt_path=str(
            Path(settings.get("generativeModelSettings", "systemPromptsFolderPath"))
            / settings.get("generativeModelSettings", "rag_prompt")
        ),
        compare_prompt_path=str(
            Path(settings.get("generativeModelSettings", "systemPromptsFolderPath"))
            / settings.get("generativeModelSettings", "promptCompareQuestionAndContext")
        ),
    )

    keyword_generator = KeywordsGenerator(
        flask_app=flask_app,
        top_n_keywords_per_chunk=settings.get("documentKeywordExtractionSettings", "topNKeywordsPerChunk"),
        db_manager=db_manager,
        text_preprocessor=text_preprocessor,
    )

    user_chat_history = UserChatHistory(flask_app=flask_app)

    flask_app.logger.debug("Services initialized")

    # Blueprints registrieren
    flask_app.logger.debug("Registering blueprints")
    flask_app.register_blueprint(model_response_routes.bp)
    flask_app.register_blueprint(auth_routes.bp)
    flask_app.register_blueprint(greeting_routes.bp)
    flask_app.register_blueprint(knowledgebase_routes.bp)
    flask_app.logger.debug("Blueprints registered")

    @flask_app.errorhandler(Exception)
    def error_handler(e):
        """Globaler Fehlerhandler."""
        current_app.logger.error(f"Unhandled exception: {e!r}", exc_info=True)
        return response_exception(e)

    # Store references to the initialized services in the Flask app context
    flask_app.config["settings"] = settings
    flask_app.config["db_manager"] = db_manager
    flask_app.config["text_preprocessor"] = text_preprocessor
    flask_app.config["vector_creator"] = vector_creator
    flask_app.config["model"] = model
    flask_app.config["system_prompts"] = system_prompts
    flask_app.config["keywords"] = keyword_generator
    flask_app.config["user_chat_history"] = user_chat_history

    # LangSmith Einstellungen
    os.environ["LANGCHAIN_ENDPOINT"] = settings.get("langSmithSettings", "langchainEndpoint")
    os.environ["LANGCHAIN_API_KEY"] = settings.get("langSmithSettings", "langchainApiKey")

    flask_app.logger.info("Flask app creation completed")

    return flask_app


@atexit.register
def shutdown_session() -> None:
    """Räumt Ressourcen auf, wenn die Anwendung heruntergefahren wird.

    Diese Funktion ist registriert, um beim Beenden ausgeführt zu werden. Sie schließt die
    Datenbankverbindung, falls eine existiert. Sie verwendet den Anwendungskontext, um
    den korrekten Zugriff auf die Flask-Anwendungsinstanz und ihren Logger zu gewährleisten.
    """
    from flask import (
        current_app,
    )  # Notwendigkeit für die korrekte Funktionsweise des Loggers

    if "app" in globals():
        with app.app_context():
            current_app.logger.info("Shutting down and cleaning up resources...")
            db_manager = current_app.config.get("db_manager")
            if db_manager:
                current_app.logger.info("Shutting down database connection...")
                db_manager.close()
                current_app.logger.info("Database connection closed.")
            model = current_app.config.get("model")
            if model:
                current_app.logger.info("Unloading the LLM model...")
                model.delete_model()
                current_app.logger.info("LLM model unloaded.")


app = create_app()


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/chat")
def chat() -> str:
    return render_template("chat.html")


@app.route("/knowledgebase")
def knowledgebase() -> str:
    return render_template("knowledgebase_editor.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tests", action="store_true", help="Start automated tests")
    args = parser.parse_args()
    if args.run_tests:
        run_automated_tests_in_langsmith(app)
    else:
        port = int(os.environ.get("PORT", 11892))
        app.logger.info(f"Starting Flask app on port {port}")
        app.run(debug=False, port=port)
