"""Dieses Modul lädt Systemprompts in den Arbeitsspeicher der Anwendung."""

from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask

from source.model.model_loader import ModelLoader

if TYPE_CHECKING:
    from logging import Logger


class SystemPrompt:

    """Eine Klasse zur Verwaltung von Systemprompts für ein Sprachmodell.

    Diese Klasse initialisiert mit einer Flask-App, einem LLM-Modell und Pfaden zu Prompt-Vorlagen.
    Sie lädt diese Vorlagen und stellt sicher, dass sie nicht die angegebene Token-Länge überschreiten.

    Attributes:
        logger (Logger): Das Logger-Objekt der Flask-App.
        llm_model (ModelLoader): Das LLM-Modellloader-Objekt.
        max_system_prompt_length (int): Die maximale erlaubte Token-Länge für Systemprompts.
        max_chat_history_length (int): Die maximale erlaubte Token-Länge für Chat-Verlauf.
        rag_prompt (str): Der Inhalt der rag-Prompt-Vorlage.
        compare_prompt (str): Der Inhalt der Compare-Prompt-Vorlage.
    """
    def __init__(
            self,
            flask_app: Flask,
            llm_model: ModelLoader,
            max_system_prompt_length: int,
            max_chat_history_length: int,
            rag_prompt_path: str,
            compare_prompt_path: str,
    ) -> None:
        """Initialisiert die SystemPrompt-Klasse.

        Args:
            flask_app (Flask): Die Flask-Anwendungsinstanz.
            llm_model (ModelLoader): Das LLM-Modellloader-Objekt.
            max_system_prompt_length (int): Die maximale erlaubte Tokenlänge für Systemprompts.
            max_chat_history_length (int): Die maximale erlaubte Token-Länge für Chat-Verlauf.
            rag_prompt_path (str): Der Dateipfad zur rag-Prompt-Vorlage.
            compare_prompt_path (str): Der Dateipfad zur Compare-Prompt-Vorlage.
        """
        self.logger: Logger = flask_app.logger
        self.llm_model: ModelLoader = llm_model
        self.max_system_prompt_length = max_system_prompt_length
        self.max_chat_history_length = max_chat_history_length
        self.rag_prompt: str = self._load_prompt_template(rag_prompt_path)
        self.compare_prompt: str = self._load_prompt_template(compare_prompt_path)

    def _load_prompt_template(self, prompt_path: str) -> str:
        """Lädt den Inhalt einer Prompt-Vorlage.

        Args:
            prompt_path (str): Der Pfad zur Prompt-Datei.

        Returns:
            str: Der Inhalt der Prompt-Vorlage.

        Raises:
            Exception: Wenn der Prompt zu lang ist.
            OSError: Wenn beim Laden der Prompt-Vorlage ein Fehler auftritt.
        """
        self.logger.debug(f"Loading prompt template from {prompt_path}")
        try:
            with Path(prompt_path).open(encoding="utf-8") as prompt:
                template = prompt.read()
            system_prompt_tokens_length = len(self.llm_model.tokenizer.encode(template))
            if system_prompt_tokens_length > self.max_system_prompt_length:
                raise Exception(
                    f"Prompt with path: {prompt_path} is too long and has {system_prompt_tokens_length} tokens. "
                    f"Must be a maximum of {self.max_system_prompt_length} tokens."
                )
            self.logger.debug(f"Prompt template {prompt_path} loaded successfully")
            return template
        except OSError as e:
            self.logger.error(f"Error loading prompt template {prompt_path}: {e}")
            raise
