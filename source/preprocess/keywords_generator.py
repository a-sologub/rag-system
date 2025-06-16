"""Dieses Dienstprogramm extrahiert Schlüsselwörter aus allen Dokumenten in der Datenbank."""

import asyncio
from collections import Counter

from flask import Flask

from source.db.mongodb_manager import MongoDBManager
from source.preprocess.text_preprocessor import TextPreprocessor


class KeywordsGenerator:
    """Eine Klasse zum Generieren von Schlüsselwörtern aus allen TextChunks in der MongoDB.

    Attribute:
        - logger: Flask-Anwendungslogger für Logging-Operationen.
        - keyword_set (Set[str]): Eine Menge von generierten Schlüsselwörtern.
        - top_n_keywords_per_chunk (int): Anzahl der Top-Schlüsselwörter pro TextChunk.

    Methoden:
        - __init__: Initialisiert den KeywordsGenerator mit spezifischen Einstellungen.
        - generate_keywords: Führt die Schlüsselwort-Generierung durch.
    """

    def __init__(
            self,
            flask_app: Flask,
            top_n_keywords_per_chunk: int,
            db_manager: MongoDBManager,
            text_preprocessor: TextPreprocessor,
            language: str = "german",
    ) -> None:
        """Initialisiert den KeywordsGenerator mit den angegebenen Parametern.

        Args:
            flask_app: Die Flask-Anwendungsinstanz für das Logging.
            top_n_keywords_per_chunk: Anzahl der Top-Schlüsselwörter, die pro TextChunk generiert werden sollen.
            db_manager: Datenbankmanager für den Zugriff auf die Datenbank.
            text_preprocessor: TextPreprocessor mit sprachspezifischen Einstellungen.
            language: Die gewählte Sprache für die Schlüsselwortgenerierung.
        """
        self.logger = flask_app.logger
        self.top_n_keywords_per_chunk = top_n_keywords_per_chunk
        self.logger.debug(f"Initializing KeywordsGenerator with language: {language}")
        self.keyword_set = asyncio.run(self.async_generate_keywords(db_manager, text_preprocessor))

    async def async_generate_keywords(self, db_manager: MongoDBManager, text_preprocessor: TextPreprocessor) -> set[str]:
        """Lädt alle TextChunks aus der Datenbank und erstellt ein Set aus den Top-N-Schlüsselwörtern.

        Args:
            db_manager: Datenbankmanager zum Aufbau einer Verbindung mit der Datenbank.
            text_preprocessor: TextPreprocessor mit sprachspezifischen Einstellungen.

        Returns:
            Eine Menge von generierten Schlüsselwörtern.
        """
        all_docs = db_manager.knowledge_collection.find()
        keywords = set()

        self.logger.debug("Starting keywords preprocessing")
        for document in all_docs:
            if not document["revised_text"]:
                counter = Counter(await text_preprocessor.preprocess(document["origin_text"])).most_common(
                    self.top_n_keywords_per_chunk)
            else:
                counter = Counter(await text_preprocessor.preprocess(document["revised_text"])).most_common(
                    self.top_n_keywords_per_chunk)

            for word, count in counter:
                keywords.add(word)

        self.logger.debug(f"KeywordsGenerator keywords: {keywords}")
        return keywords
