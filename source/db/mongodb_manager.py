"""
Dieses Modul definiert die `MongoDBManager`-Klasse, die für die Verwaltung von MongoDB-Verbindungen und den
Zugriff auf spezifische Sammlungen zuständig ist.
"""

from flask import Flask
from pymongo import MongoClient
from pymongo.errors import OperationFailure


class MongoDBManager:
    """
    Eine Klasse zur Verwaltung von MongoDB-Verbindungen und zum Zugriff auf spezifische Sammlungen.

    Diese Klasse behandelt die Initialisierung von MongoDB-Verbindungen, bietet Methoden zum Testen
    der Verbindung und stellt sicher, dass die Verbindung ordnungsgemäß geschlossen wird, wenn sie nicht mehr benötigt wird.

    Attribute:
        - logger: Flask-Anwendungslogger für Logging-Operationen.
        - client (MongoClient): Die MongoDB-Client-Verbindung.
        - db (Database): Die MongoDB-Datenbankinstanz.
        - knowledge_collection (Collection): Die Sammlung zur Speicherung von Wissensdaten.
        - vector_collection (Collection): Die Sammlung zur Speicherung von Vektordaten.

    Methoden:
        - __init__: Initialisiert die MongoDB-Verbindung und Sammlungen.
        - test_connection: Testet die Datenbankverbindung und die Existenz der Sammlungen.
        - close: Schließt die MongoDB-Verbindung.
    """

    def __init__(
        self,
        flask_app: Flask,
        client_uri: str,
        username: str,
        password: str,
        db_name: str,
        knowledge_collection_name: str,
        vector_collection_name: str,
    ):
        """
        Initialisiert die MongoDBManager-Instanz.

        Diese Methode stellt eine Verbindung zum MongoDB-Client her und greift auf die angegebene
        Datenbank und Sammlungen basierend auf den bereitgestellten Parametern zu.

        Args:
            flask_app (Flask): Die Flask-Anwendungsinstanz für das Logging.
            client_uri (str): Die MongoDB-Verbindungs-URI.
            username (str): Benutzername für Datenbankzugang.
            password (str): Passwort für Datenbankzugang.
            db_name (str): Der Name der Datenbank, mit der verbunden werden soll.
            knowledge_collection_name (str): Der Name der Wissenssammlung.
            vector_collection_name (str): Der Name der Vektorsammlung.

        Raises:
            ConnectionError: Wenn die Verbindung zur Datenbank fehlschlägt.
            ValueError: Wenn die angegebene Datenbank oder Sammlungen nicht existieren.
        """
        self.logger = flask_app.logger
        flask_app.logger.info(f"Initializing MongoDBManager with database: {db_name}")
        try:
            self.client = MongoClient(client_uri, username=username, password=password, authSource=db_name)
            self.logger.info(f"Authenticated successfully as {username}")
        except OperationFailure as e:
            self.logger.error(f"Authentication failed: {e}")
            raise
        self.db = self.client[db_name]
        self.knowledge_collection = self.db[knowledge_collection_name]
        self.vector_collection = self.db[vector_collection_name]
        self.logger.debug(
            f"Collections initialized: {knowledge_collection_name}, {vector_collection_name}"
        )

        # Check if the database exists and collections are valid
        self.test_connection()

    def test_connection(self) -> None:
        """
        Testet die Verbindung zur Datenbank und prüft, ob die angegebenen Sammlungen existieren.

        Diese Methode überprüft, dass:
        1. Die Verbindung zum MongoDB-Server gültig ist.
        2. Die angegebene Datenbank existiert.
        3. Sowohl die Wissens- als auch die Vektorsammlungen in der Datenbank existieren.

        Raises:
            ValueError: Wenn die Datenbank oder eine der Sammlungen nicht existiert.
            ConnectionError: Wenn es Probleme bei der Verbindung zur Datenbank gibt.
        """
        self.logger.debug("Testing MongoDB connection")
        try:
            # Versucht, die Datenbanknamen aufzulisten, um sicherzustellen, dass die Verbindung gültig ist
            if self.db.name not in self.client.list_database_names():
                self.logger.error(f"Database '{self.db.name}' does not exist")
                raise ValueError(f"Database '{self.db.name}' does not exist")

            # Versucht, die Datenbanknamen aufzulisten, um sicherzustellen, dass die Verbindung gültig ist
            collections = self.db.list_collection_names()
            if self.knowledge_collection.name not in collections:
                self.logger.error(
                    f"Knowledge collection '{self.knowledge_collection.name}' does not exist"
                )
                raise ValueError(
                    f"Knowledge collection '{self.knowledge_collection.name}' does not exist"
                )
            if self.vector_collection.name not in collections:
                self.logger.error(
                    f"Vector collection '{self.vector_collection.name}' does not exist"
                )
                raise ValueError(
                    f"Vector collection '{self.vector_collection.name}' does not exist"
                )
            self.logger.debug("MongoDB connection test successful")
        except Exception as e:
            self.logger.exception(f"Failed to connect to the database: {e}")
            raise ConnectionError(f"Failed to connect to the database: {e}") from e

    def close(self) -> None:
        """
        Schließt die MongoDB-Verbindung.

        Diese Methode sollte aufgerufen werden, wenn MongoDB-Operationen abgeschlossen sind, um sicherzustellen,
        dass die Verbindung ordnungsgemäß geschlossen und Ressourcen freigegeben werden.
        """
        self.logger.debug("Closing MongoDB connection")
        self.client.close()
        self.logger.debug("MongoDB connection closed")
