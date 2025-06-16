"""Dieses Modul definiert die `VectorCreator`-Klasse, die für die Erstellung von Vektoreinbettungen verantwortlich ist."""

from flask import Flask
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor


class VectorCreator:
    """Eine Klasse zur Erstellung von Vektoreinbettungen aus vorverarbeitetem Text.

    Diese Klasse verwendet ein SentenceTransformer-Modell, um Vektoreinbettungen
    für Eingabetexte zu generieren. Sie ist darauf ausgelegt, mit vorverarbeitetem Text
    in Form einer Liste von Tokens zu arbeiten.

    Attribute:
        - logger: Flask-Anwendungslogger für Logging-Operationen.
        - model (SentenceTransformer): Das geladene SentenceTransformer-Modell.

    Methoden:
        - __init__: Initialisiert den VectorCreator mit einem spezifizierten Modell.
        - get_embedding: Generiert Einbettungen für vorverarbeiteten Text.
    """

    def __init__(self, flask_app: Flask, model_name: str) -> None:
        """Initialisiert den VectorCreator mit einem spezifizierten SentenceTransformer-Modell.

        Diese Methode lädt das angegebene SentenceTransformer-Modell und richtet das Logging ein.

        Args:
            flask_app: Die Flask-Anwendungsinstanz für das Logging.
            model_name: Der Name des zu ladenden SentenceTransformer-Modells.

        Raises:
            Exception: Wenn beim Laden des spezifizierten Modells ein Fehler auftritt.
        """
        self.logger = flask_app.logger
        self.logger.debug(f"Initializing VectorCreator with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.debug(f"SentenceTransformer model '{model_name}' loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e!s}")
            raise

    def get_embedding(self, preprocessed_text: list[str]) -> list[Tensor] | ndarray | Tensor | None:
        """Generiert Einbettungen für den vorverarbeiteten Text.

        Diese Methode nimmt eine Liste von vorverarbeiteten Tokens und verwendet das geladene
        SentenceTransformer-Modell, um eine Vektoreinbettung zu generieren.

        Args:
            preprocessed_text: Eine Liste von vorverarbeiteten Tokens.

        Returns:
            Der Einbettungsvektor für den Eingabetext.

        Raises:
            Exception: Wenn während des Einbettungsgenerierungsprozesses ein Fehler auftritt.
        """
        self.logger.debug("Generating embedding for preprocessed text")

        if preprocessed_text is not None:
            try:
                embedding = self.model.encode(preprocessed_text)
                self.logger.debug(f"Embedding generated successfully. Shape: {embedding.shape}")
                return embedding
            except Exception as e:
                self.logger.error(f"Failed to generate embedding: {e!s}")
                raise
