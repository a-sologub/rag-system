import json
import os
from typing import Any

from flask import Flask


class SettingsLoader:
    """
    Eine Klasse zum Laden und Zugreifen auf Einstellungen aus einer JSON-Datei.

    Diese Klasse bietet Methoden zum Laden von Einstellungen aus einer JSON-Datei
    und zum Abrufen von Werten mithilfe verschachtelter Schlüssel.

    Attribute:
        - logger (logging.Logger): Logger-Einstellungen für die aktuelle App.
        - settings_path (str): Der Pfad zur Einstellungsdatei.
        - settings (dict): Die geladenen Einstellungen.
    """

    def __init__(self, flask_app: Flask, settings_file: str ="settings.json") -> None:
        """
        Initialisiert den SettingsLoader.

        Args:
            flask_app: Die Flask-Anwendungsinstanz.
            settings_file: Der Name der Einstellungsdatei. Standardmäßig "settings.json".
        """
        self.logger = flask_app.logger
        self.logger.debug(f"Initializing SettingsLoader with file: {settings_file}")
        self.settings_path = os.path.join("source", settings_file)
        self.logger.debug(f"Full settings path: {self.settings_path}")
        self.settings = self.load_settings()

    def load_settings(self) -> dict:
        """
        Lädt Einstellungen aus der JSON-Datei.

        Returns:
            settings: Die geladenen Einstellungen oder ein leeres Dict, wenn die Datei nicht gefunden oder ungültig ist.
        """
        self.logger.debug(f"Attempting to load settings from: {self.settings_path}")
        try:
            with open(self.settings_path) as f:
                settings = json.load(f)
            self.logger.debug("Settings loaded successfully")
            self.logger.debug(f"Loaded {len(settings)} top-level keys")
            return settings
        except FileNotFoundError:
            self.logger.error(f"Settings file not found: {self.settings_path}")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in settings file: {self.settings_path}")
            return {}

    def get(self, *keys) -> Any:
        """
        Ruft einen Wert aus den Einstellungen ab, unter Verwendung verschachtelter Schlüssel.

        Args:
            keys (Any): Variable Anzahl von Argumenten für Schlüssel zum Zugriff auf verschachtelte Dictionaries.

        Returns:
            Der Wert, der mit den gegebenen Schlüsseln verknüpft ist, oder None, wenn nicht gefunden.
        """
        self.logger.debug(f"Attempting to retrieve setting with keys: {keys}")
        value = self.settings
        for key in keys:
            value = value.get(key, {})

        if value == {}:
            self.logger.warning(f"Setting not found for keys: {keys}")
            return None
        else:
            self.logger.debug(f"Successfully retrieved setting for keys: {keys}")
            return value
