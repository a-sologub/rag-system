"""Dieses Modul implementiert einen erweiterten KI-Agenten für das Retrieval-Augmented Generation (RAG) System.

Es enthält eine Agent-Klasse zur Verarbeitung von Benutzeranfragen mit integriertem LangSmith-Tracing,
Dokumentenverarbeitung und Antwortgenerierung. Der Agent unterstützt Streaming-Responses, Token-Tracking
und kontextbasierte Antwortvalidierung durch eine vollständige RAG-Pipeline.
"""

from collections.abc import Generator
from io import StringIO
from typing import TYPE_CHECKING

from flask import current_app as app
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from source.model.prompt_builder_for_rag import create_prompt_for_rag
from source.model.prompt_builder_for_search_answer_in_context_agent import create_prompt_for_search_answer_in_context
from source.rag.document_retrieval import search_similar_texts_in_db
from source.test_environment.langsmith_client import LangSmithClient

if TYPE_CHECKING:
    from logging import Logger

    from source.model.model_loader import ModelLoader
    from source.model.system_prompt_loader import SystemPrompt
    from source.preprocess.keywords_generator import KeywordsGenerator
    from source.preprocess.text_preprocessor import TextPreprocessor
    from source.rag.user_chat_history import UserChatHistory
    from source.settings_loader import SettingsLoader


class Agent:
    """Ein KI-Agent für die Verarbeitung von Benutzeranfragen und die Generierung von Antworten.

    Diese Klasse implementiert einen RAG-Pipeline-Prozess, der Dokumente abruft,
    Kontexte verarbeitet und Antworten basierend auf Benutzeranfragen generiert.

    Attributes:
        use_langsmith (bool): Flag zur Aktivierung von LangSmith.
        langsmith_client (LangSmithClient): Client für LangSmith-Tracing.
        llm_model (ModelLoader): Das zu verwendende Sprachmodell.
        system_prompts (SystemPrompt): Die SystemPrompts für das LLM.
        settings (SettingsLoader): Die Anwendungseinstellungen.
        text_preprocessor (TextPreprocessor): TextPreprocessor mit sprachspezifischen Einstellungen.
        keywords (KeywordsGenerator): KeywordsGenerator mit sprachspezifischen Einstellungen.
        logger (Logger): Der Logger für diese Klasse.
        user_chat_history (UserChatHistory): Chatverlauf für eine bestimmte session_id.
        input_tokens (List[str]): Liste der Eingabe-Tokens.
        output_tokens (List[str]): Liste der Ausgabe-Tokens.
        full_response (StringIO): Der vollständige Antworttext.
        session_id (str): Chat-ID im Browser-Tab.
    """

    def __init__(self, use_langsmith: bool, langsmith_client_name: str) -> None:
        """Initialisiert den Agenten mit dem Sprachmodell und Logger.

        Args:
            use_langsmith: Flag zur Aktivierung von LangSmith.
            langsmith_client_name: Name des LangSmith-Clients.
        """
        self.use_langsmith: bool = use_langsmith
        self.langsmith_client: LangSmithClient = LangSmithClient(use_langsmith, langsmith_client_name)
        self.llm_model: ModelLoader = app.config["model"]
        self.system_prompts: SystemPrompt = app.config["system_prompts"]
        self.settings: SettingsLoader = app.config["settings"]
        self.text_preprocessor: TextPreprocessor = app.config["text_preprocessor"]
        self.keywords: KeywordsGenerator = app.config["keywords"]
        self.logger: Logger = app.logger
        self.user_chat_history: UserChatHistory = app.config["user_chat_history"]
        self.input_tokens: list[str] = []
        self.output_tokens: list[str] = []
        self.full_response: StringIO = StringIO()
        self.session_id: str = ""

    @LangSmithClient.trace_call
    async def __call__(self, session_id: str, message: str) -> Generator[str, None, str] | str:
        """Verarbeitet eine Benutzeranfrage und gibt eine Antwort zurück.

        Args:
            session_id: Session-ID zum Speichern und Abrufen des Chatverlaufs.
            message: Die Benutzeranfrage.

        Returns:
            Die generierte Antwort oder eine Fehlermeldung.
        """
        self.session_id = session_id
        try:
            result = await self._async_run_rag_pipeline(message)
            return self.generate_stream(result)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "I'm sorry, but I encountered an error while processing your request. Please try again later."

    @LangSmithClient.trace_pipeline
    async def _async_run_rag_pipeline(self, message: str) -> str:
        """Führt die RAG-Pipeline aus.

        Diese Methode verarbeitet die Benutzeranfrage, sucht relevante Dokumente,
        erstellt einen Kontext und generiert eine Antwort.

        Args:
            message: Die Benutzeranfrage.

        Returns:
            Die generierte Antwort.
        """
        preprocessed_message = await self.message_processing(message)

        if await self.is_message_match_knowledge_context(preprocessed_message, self.keywords.keyword_set):
            documents_from_db = await self.retrieve_documents(message)
            if await self._is_answer_in_context(message, documents_from_db):
                return await self._handle_response(message, documents_from_db)

        return await self._handle_response(message, context="[Retrieved Documents]: [NO DATA]\n")

    @LangSmithClient.trace_message_processing
    async def message_processing(self, text: str) -> list[str]:
        """Löscht die sensiblen Daten und bearbeitet den Text der Nachricht.

        Args:
            text: Text der Nachricht.

        Returns:
            Eine Liste der vorverarbeiteten Tokens ohne sensible Daten.
        """
        return await self.text_preprocessor.process(text)

    @LangSmithClient.trace_message_match_knowledge_context
    async def is_message_match_knowledge_context(self, word_list: list[str], word_set: set[str]) -> bool:
        """Überprüft, ob eine Nachricht mit dem Wissenskontext übereinstimmt.

        Args:
            word_list: Liste von Wörtern aus der Nachricht.
            word_set: Set von Schlüsselwörtern aus dem Wissenskontext.

        Returns:
            True, wenn es eine Übereinstimmung gibt, sonst False.
        """
        return bool(set(word_list) & word_set)

    @LangSmithClient.trace_handle_response
    async def _handle_response(self, message: str, context: str | None = None) -> str:
        """Verarbeitet die Antwort basierend auf dem gegebenen Prompt-Schlüssel.

        Args:
            message: Die Benutzeranfrage.
            context: Optional. Der Kontext für die Antwortgenerierung.

        Returns:
            Die generierte Antwort.
        """
        if self.session_id in self.user_chat_history.chat_history:
            self.user_chat_history.add_message(
                self.session_id, HumanMessage(content=message, token_count=len(self.llm_model.tokenizer.encode(message)))
            )

            return await create_prompt_for_rag(
                ChatPromptTemplate.from_messages(self.user_chat_history.get_messages(self.session_id)), context
            )

        return await self._create_prompt_and_generate_response(self.system_prompts.rag_prompt, message, context)

    async def _create_prompt_and_generate_response(self, system_prompt: str, message: str, context: str | None = None) -> str:
        """Erstellt einen Prompt und generiert eine Antwort.

        Args:
            system_prompt: Der Systemprompt.
            message: Die Benutzeranfrage.
            context: Optional. Der Kontext für die Antwortgenerierung.

        Returns:
            Die generierte Antwort.
        """
        self.user_chat_history.add_message(
            self.session_id, SystemMessage(content=system_prompt, token_count=len(self.llm_model.tokenizer.encode(system_prompt)))
        )
        self.user_chat_history.add_message(
            self.session_id, HumanMessage(content=message, token_count=len(self.llm_model.tokenizer.encode(message)))
        )
        return await create_prompt_for_rag(
            ChatPromptTemplate.from_messages(self.user_chat_history.get_messages(self.session_id)), context
        )

    @LangSmithClient.trace_retrieve_documents
    async def retrieve_documents(self, message: str) -> str:
        """Sucht nach relevanten Dokumenten in der Datenbank.

        Args:
            message: Die Benutzeranfrage.

        Returns:
            Die Dokumente im Volltext.
        """
        documents_from_db = await search_similar_texts_in_db(
            query=str(message),
            top_k=int(self.settings.get("documentRetrievalSettings", "topK")),
            full_text_content=bool(self.settings.get("documentRetrievalSettings", "returnFullTextContent")),
        )
        self.logger.debug(f"Retrieved {len(documents_from_db)} documents from database")
        return "".join("".join(doc["full_text"]) for doc in documents_from_db)

    @LangSmithClient.trace_search_answer_in_context
    async def _is_answer_in_context(self, message: str, context: str) -> bool:
        """Überprüft, ob eine Antwort auf die Frage im gegebenen Kontext vorhanden ist.

        Args:
            message: Die Benutzeranfrage.
            context: Der Kontext, in dem nach einer Antwort gesucht wird.

        Returns:
            True, wenn eine Antwort im Kontext gefunden wurde, sonst False.
        """
        prompt = await create_prompt_for_search_answer_in_context(
            ChatPromptTemplate.from_messages(
                [SystemMessage(content=self.system_prompts.compare_prompt), HumanMessage(content=message)]
            ),
            context=context,
        )

        generator = self.llm_model.generate(prompt=prompt)

        result = ""
        for item in generator:
            if isinstance(item, str):
                result += item
            elif isinstance(item, tuple):
                break

        self.logger.debug(f"Response from comparison: {result}")
        return "ja" in result.strip().lower()

    @LangSmithClient.trace_stream_generator
    def generate_stream(self, prompt: str) -> Generator[str, None, str]:
        """Generiert einen Stream von Tokens aus dem KI-Modell und speichert decodierte Input- und Output-Wortlisten.

        Diese Methode erzeugt einen Generator, der Tokens streamt und am Ende die vollständige
        Antwort zurückgibt. Sie verarbeitet auch das abschließende Tupel mit Input- und Output-Wortlisten.

        Args:
            prompt: Der zu verarbeitende Prompt für das KI-Modell.

        Yields:
            Einzelne Tokens der generierten Antwort, wie sie vom Modell produziert werden.

        Returns:
            Die vollständige generierte Antwort als zusammengesetzter String.

        Raises:
            Exception: Wenn ein Fehler während der Generierung auftritt.
        """
        try:
            generator = self.llm_model.generate(prompt=prompt)
            for item in generator:
                if isinstance(item, str):
                    self.full_response.write(item)
                    yield item
                elif isinstance(item, tuple):
                    self.input_tokens, self.output_tokens = item
                    self.logger.debug(f"Input tokens count: {len(self.input_tokens)}")
                    self.logger.debug(f"Output tokens count: {len(self.output_tokens)}")
                    break
            self.user_chat_history.add_message(
                self.session_id,
                AIMessage(
                    content=self.full_response.getvalue(),
                    token_count=len(self.llm_model.tokenizer.encode(self.full_response.getvalue())),
                ),
            )
            self.logger.info(f"Generated response stream with {self.get_output_token_count()} tokens")
            self.logger.debug(f"Full response: {self.full_response.getvalue()}")
            return self.get_full_response()
        except Exception as e:
            self.logger.error(f"Error in stream generation: {e!s}")
            self.reset_response_data()
            raise

    @LangSmithClient.trace_response
    def get_full_response(self) -> str:
        """Gibt die vollständige generierte Antwort zurück.

        Returns:
            Die vollständige generierte Antwort.
        """
        return self.full_response.getvalue()

    def get_input_token_count(self) -> int:
        """Gibt die Anzahl der Eingabe-Tokens zurück.

        Returns:
            Die Anzahl der Eingabe-Tokens.
        """
        return len(self.input_tokens)

    def get_output_token_count(self) -> int:
        """Gibt die Anzahl der generierten Ausgabe-Tokens zurück.

        Returns:
            Die Anzahl der generierten Ausgabe-Tokens.
        """
        return len(self.output_tokens)

    def reset_response_data(self) -> None:
        """Setzt die Antwortdaten zurück.

        Diese Methode leert den vollständigen Antworttext und die Token-Listen.
        """
        self.full_response = StringIO()
        self.input_tokens = []
        self.output_tokens = []
