"""Dieses Modul definiert die `ModelLoader`-Klasse, die für das Laden und die Verwaltung eines AI-Modells zuständig ist.

Es bietet eine Streaming-basierte Textgenerierung mit konfigurierbaren Parametern wie Sampling,
Temperatur und Token-Länge, sowie detailliertes Token-Tracking für Input und Output. Die Klasse
integriert sich in das Flask-Logging-System für umfassende Operationsprotokolle.
"""

from collections.abc import Generator
from typing import Any

from flask import Flask
from llama_cpp import Llama, LlamaTokenizer


class ModelLoader:
    """ModelLoader ist verantwortlich für das Laden und Verwalten eines Llama-CPP-Generative-AI-Modells.

    Es behandelt Modellinitialisierung, Tokenisierung und Textgenerierung.

    Diese Klasse bietet Funktionalität zum:
        1. Laden eines Llama-CPP-Modells von einem angegebenen Pfad
        2. Einrichten eines Tokenizers für das Modell
        3. Konfigurieren von Generierungsparametern (wie max. Länge, Temperatur, etc.)
        4. Generieren von Text basierend auf Eingabeabfragen und Kontexten

    Die Klasse verwendet die llama-cpp-library für Modelloperationen und integriert
    sich in das Logging-System von Flask für detaillierte Operationsprotokolle.

    Attribute:
        - logger: Flask-App-Logger für Logging-Operationen
        - model_path (str): Pfad zur Llama-CPP-Modelldatei
        - model (Llama): Geladenes Llama-CPP-Modell zur Textgenerierung
        - tokenizer (LlamaTokenizer): Tokenizer für das Modell
        - top_p (float): Top-p-Sampling-Parameter
        - top_k (int): Top-k-Sampling-Parameter
        - temperature (float): Temperatur für die Textgenerierung
        - repetition_penalty (float): Strafe für Token-Wiederholung

    Methoden:
        - generate(query, context): Generiert Text basierend auf Eingabeabfrage und Kontext
        - delete_model: Löscht das geladene Modell.

    Die generate-Methode ist ein Generator, der Tokens ausgibt, während sie generiert werden,
    was Streaming-Output in Anwendungen ermöglicht.
    """

    def __init__(  # noqa: PLR0913 (Warnung über zu viele Argumente in der Funktionsdefinition ignoriert)
        self,
        flask_app: Flask,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        flash_attn: bool,
        verbose: bool,
        repetition_penalty: float,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> None:
        """Initialisiert den ModelLoader mit den angegebenen Parametern.

        Args:
            flask_app (Flask): Die Flask-Anwendungsinstanz für das Logging.
            model_path (str): Pfad zur ONNX-Modelldatei.
            n_gpu_layers (int): Anzahl der GPU-Layer zur Verarbeitung.
            n_ctx (int): Kontextgröße für die Modellverarbeitung.
            flash_attn (bool): Aktiviert Flash-Attention zur Optimierung.
            verbose (bool): Ob detaillierte Log-Ausgaben aktiviert werden sollen.
            repetition_penalty (float): Strafe für Token-Wiederholung.
            temperature (float): Temperatur für die Textgenerierung.
            top_k (int): Top-k-Sampling-Parameter.
            top_p (float): Top-p-Sampling-Parameter.
        """
        self.logger = flask_app.logger  # Verwendet den Logger der Flask-App
        self.logger.debug("Initializing ModelLoader")
        self.model_path = model_path
        self.logger.debug(f"Loading model from {self.model_path}")
        self.model: Llama = Llama(
            model_path=self.model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, flash_attn=flash_attn, verbose=verbose
        )
        self.logger.debug("Model loaded successfully")
        self.tokenizer: LlamaTokenizer = self.model.tokenizer()
        self.logger.debug("Tokenizer created")
        self.repetition_penalty: float = repetition_penalty
        self.temperature: float = temperature
        self.top_k: int = top_k
        self.top_p: float = top_p

    def generate(self, prompt: str) -> Generator[str | tuple[list[str], list[str]], Any, str | tuple[list[str], list[str]]]:
        """Generiert Text basierend auf dem Eingabe-Prompt und gibt Listen von decodierten Wörtern zurück.

        Diese Methode tokenisiert den Eingabe-Prompt, initialisiert einen Generator mit den
        konfigurierten Suchoptionen und generiert Text-Tokens. Sie gibt entweder eine Fehlermeldung
        zurück, wenn der Prompt leer ist, oder einen Generator, der einzelne Tokens während der
        Generierung ausgibt und am Ende die decodierten Eingabe- und Ausgabe-Wörter zurückgibt.

        Args:
            prompt: System-Prompt für das Language Model.

        Yields:
            Einzelne generierte Text-Tokens während der Generierung.

        Returns:
            Ein Tupel bestehend aus (decoded_input_words, decoded_output_words),
            wobei beide Listen von Strings die decodierten Wörter für Input und Output sind.

        Raises:
            ValueError: Wenn der Prompt leer ist.
        """
        if not prompt:
            self.logger.error("Error, input cannot be empty")
            return "Error, input cannot be empty"

        self.logger.debug("Starting generation process")

        input_tokens = self.tokenizer.encode(prompt)
        self.logger.debug(f"Input tokens count: {len(input_tokens)}")

        output_tokens = []
        self.logger.debug("Starting generation loop")

        for token in self.model.generate(
            tokens=input_tokens, top_k=self.top_k, top_p=self.top_p, temp=self.temperature, repeat_penalty=self.repetition_penalty
        ):
            if token == 100265:
                break  # Generator wird angehalten, wenn Token "<|im_end|>" erzeugt wird.
            output_tokens.append(token)
            token_str = self.tokenizer.decode([token])
            yield token_str

        self.logger.debug("Generation completed")

        decoded_input_words = [self.tokenizer.decode([token]).strip() for token in input_tokens if token != 0]
        decoded_output_words = [self.tokenizer.decode([token]).strip() for token in output_tokens if token != 0]

        yield decoded_input_words, decoded_output_words
        return decoded_input_words, decoded_output_words

    def delete_model(self) -> None:
        """Löscht das geladene Modell.

        Diese Methode wird verwendet, um den Speicher freizugeben, der vom geladenen Modell belegt wird.
        """
        del self.model
