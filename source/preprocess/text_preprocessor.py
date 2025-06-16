"""Dieses Modul definiert die `TextPreprocessor`-Klasse, die für die Vorverarbeitung von Textdaten verantwortlich ist."""

import re
from pathlib import Path

import spacy
from flask import Flask


class TextPreprocessor:
    """Eine Klasse zur Vorverarbeitung von Textdaten.

    Diese Klasse bietet Methoden zum Bereinigen, Tokenisieren, Entfernen von Stoppwörtern
    und Lemmatisieren von Texteingaben. Sie ist für die Arbeit mit einer bestimmten Sprache konzipiert,
    standardmäßig Deutsch.

    Attribute:
        - logger: Flask-Anwendungslogger für Logging-Operationen.
        - stop_words (Set[str]): Eine Menge von Stoppwörtern für die angegebene Sprache.
        - nlp (Language): Laden Sie ein SpaCy-Modell aus einem installierten Paket.

    Methoden:
        - __init__: Initialisiert den TextPreprocessor mit sprachspezifischen Einstellungen.
        - delete_sensitive_data: Entfernt sensible Daten aus dem Text.
        - preprocess: Führt die Textvorverarbeitung für den Eingabetext durch.
        - process: Kombiniert die Entfernung sensibler Daten und die Vorverarbeitung.
    """

    def __init__(self, flask_app: Flask, stop_words_file_path: str, language: str = "german") -> None:
        """Initialisiert den TextPreprocessor.

        Diese Methode richtet den Preprocessor mit sprachspezifischen Stoppwörtern
        und einem Lemmatisierer ein. Sie verwendet NLTK's Ressourcen für diese Aufgaben.

        Args:
            flask_app: Die Flask-Anwendungsinstanz für das Logging.
            stop_words_file_path: Der Pfad zur Datei mit Stoppwörtern.
            language: Die Sprache für Stoppwörter. Standardmäßig "german".
        """
        self.logger = flask_app.logger
        self.logger.debug(f"Initializing TextPreprocessor with language: {language}")
        with Path(stop_words_file_path).open(encoding="utf-8") as file:
            self.stop_words = file.read().splitlines()
        self.nlp = spacy.load("de_core_news_lg")
        self.logger.debug(f"Loaded {len(self.stop_words)} stop words")

    def delete_sensitive_data(self, text: str) -> str:
        """Entfernt sensible Daten wie E-Mail-Adressen und Telefonnummern aus dem Text.

        Args:
            text: Der Eingabetext, aus dem sensible Daten entfernt werden sollen.

        Returns:
            Der Text ohne sensible Daten.
        """
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        main_phone_pattern = r"\+\d{1,3}\s?\d{2,3}\s?\d{3,6}[-\s]?\d{0,4}"
        remaining_digits_pattern = r"\[TELEFONNUMMER ENTFERNT\]\s*\d+"

        email_count = len(re.findall(email_pattern, text))
        phone_count = len(re.findall(main_phone_pattern, text))

        text = re.sub(email_pattern, "[EMAIL ENTFERNT]", text)
        text = re.sub(main_phone_pattern, "[TELEFONNUMMER ENTFERNT]", text)
        text = re.sub(remaining_digits_pattern, "[TELEFONNUMMER ENTFERNT]", text)

        self.logger.debug(f"Removed {email_count} email addresses and at least {phone_count} phone numbers")

        return text

    async def preprocess(self, text: str, remove_stop_words: bool = True) -> str | list[str]:
        """Vorverarbeitet den Eingabetext.

        Diese Methode führt die folgenden Schritte am Eingabetext durch:
            1. Einzelne Bindestriche oder nicht zwischen Wörtern stehende Bindestriche entfernen
            2. Entfernung nicht-alphabetischer Zeichen
            3. Vorverarbeitung von Text mit der Spacy-Bibliothek
            4. Lemmatisierung und Umwandlung in Kleinbuchstaben
            5. Entfernung von Stoppwörtern
            6. Lemmatisierung

        Args:
            text: Der zu verarbeitende Eingabetext.
            remove_stop_words: bool to toggle if stopwords should be removed

        Returns:
            Eine Liste der vorverarbeiteten Tokens.
        """
        self.logger.debug("Starting text preprocessing")
        self.logger.debug(f"Input text length: {len(text)}")

        # Einzelne Bindestriche oder nicht zwischen Wörtern stehende Bindestriche entfernen
        text = re.sub(r"(?<!\w)-|-(?!\w)", "", text)
        # Entfernung nicht-alphabetischer Zeichen
        text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s-]", "", text)
        processed_text = self.nlp(text)
        self.logger.debug("Text cleaned")

        lemmatized_tokens = [token.lemma_.lower() for token in processed_text]

        if not remove_stop_words:
            processed_tokens = " ".join(lemmatized_tokens)
            return processed_tokens

        processed_tokens = [token for token in lemmatized_tokens if token not in self.stop_words]
        return processed_tokens

    async def process(self, text: str) -> list[str]:
        """Kombiniert die Entfernung sensibler Daten und die Vorverarbeitung des Textes.

        Diese Methode führt zuerst die Entfernung sensibler Daten durch und
        wendet dann die Vorverarbeitung auf den bereinigten Text an.

        Args:
            text: Der zu verarbeitende Eingabetext.

        Returns:
            Eine Liste der vorverarbeiteten Tokens ohne sensible Daten.
        """
        return await self.preprocess(self.delete_sensitive_data(text))
