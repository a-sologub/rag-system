"""
Dieses Modul implementiert den LangSmithClient für das Tracing und die Protokollierung von Modellaufrufen.
"""

import datetime
import time
import traceback
from functools import wraps
from typing import Optional, List, Set, Dict, Generator, Callable, Any, TYPE_CHECKING

from flask import current_app as app
from langsmith.client import Client, RUN_TYPE_T
from langsmith.run_trees import RunTree

if TYPE_CHECKING:
    from logging import Logger

    from source.rag.user_chat_history import UserChatHistory


class LangSmithClient:
    """
    Eine Klasse zur Verwaltung von LangSmith-Tracing für Modellaufrufe und Verarbeitungsschritte.

    Diese Klasse bietet Funktionalitäten zum Erstellen von Projekten, Tracen von Läufen
    und Dekorieren von Funktionen für detailliertes Logging und Fehlerbehandlung.

    Attributes:
        use_langsmith (bool): Flag zur Aktivierung des LangSmith-Tracings.
        run_stack (RunStack): Ein Stack zur Verwaltung von verschachtelten Läufen.
        logger (Logger): Der Logger für diese Klasse.
        agent_name (str): Name des Agenten für die Projektbenennung.
        test_run (bool): Flag zur Kennzeichnung von Testläufen.
        test_id (Optional[str]): ID für Testfragen/Testchats.
        user_chat_history (UserChatHistory): Chatverlauf für eine bestimmte session_id.
        client (Optional[Client]): LangSmith-Client-Instanz.
    """

    def __init__(self, use_langsmith: bool, agent_name: str, test_run: bool = False):
        """
        Initialisiert den LangSmithClient.

        Args:
            use_langsmith: Ob LangSmith-Tracing verwendet werden soll.
            agent_name: Name des Agenten für die Projektbenennung.
            test_run: Ob es sich um einen Testlauf handelt.
        """
        self.use_langsmith: bool = use_langsmith
        self.run_stack: RunStack = RunStack()
        self.logger: Logger = app.logger
        self.agent_name = agent_name
        self.test_run: bool = test_run
        self.test_id: Optional[str] = None
        self.user_chat_history: UserChatHistory = app.config["user_chat_history"]
        if self.use_langsmith:
            self.client: Client = Client()

    def create_project(self, agent_name: str, session_id: str, test_id: str) -> str:
        """
        Erstellt ein neues LangSmith-Projekt.

        Args:
            agent_name: Name des Agenten für die Projektbenennung.
            session_id (str): Chat-ID im Browser-Tab.
            test_id (str): ID für Testfragen/Testchats.

        Returns:
            Der erstellte Projektname.
        """
        if test_id and test_id.startswith("Q"):  # Sammeln von Testfragen in einem LangSmith-Projekt
            project_name = agent_name
        else:
            project_name = f"{agent_name} {test_id if self.use_langsmith and test_id else ""} ID: {session_id[:8]}"
        if session_id not in self.user_chat_history.chat_history and not self.client.has_project(project_name):
            try:
                self.client.create_project(project_name)
            except Exception as e:
                self.logger.error(f"Error during project creation in LangSmith: {e!s}")
                raise
        return project_name

    def trace_run(self, name: str, run_type: RUN_TYPE_T, inputs: Dict[str, Any]) -> RunTree:
        """
        Erstellt einen neuen Trace-Run als Kind des aktuellen Runs.

        Args:
            name: Name des Runs.
            run_type: Typ des Runs.
            inputs: Eingabedaten für den Run.

        Returns:
            Der erstellte Run oder None, wenn kein Tracing aktiv ist.
        """
        run = self.run_stack.peek().create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
        )
        run.post()
        # 1 Millisekunde warten, um die POST-Anfragen in der richtigen Reihenfolge in Langsmith zu speichern.
        time.sleep(0.001)
        # Einige Funktionen in der PAG-Pipeline werden so schnell ausgeführt, dass POST-Anfragen fast gleichzeitig gesendet
        # werden, was zu einer Unordnung der ausgeführten Tools in der LangSmith-Benutzeroberfläche führt. Daher lohnt es
        # sich, 1 Millisekunde zu warten, bis die POST-Anfrage in LangSmith korrekt angezeigt wird.
        return run

    @staticmethod
    def handle_error(run: RunTree, error: Exception) -> None:
        """
        Behandelt Fehler in einem Trace-Run.

        Args:
            run: Der Run, in dem der Fehler aufgetreten ist.
            error: Die aufgetretene Ausnahme.
        """
        error_details = {"error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()}
        run.end(error=str(error_details))
        run.patch()
        app.logger.error(f"Error in traced function: {error_details}")

    @staticmethod
    def trace_call(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen von Funktionsaufrufen in LangSmith.

        Dieser Dekorator umschließt die dekorierte Funktion mit LangSmith-Tracing-Logik.
        Er erstellt einen neuen Run, führt die Funktion aus und protokolliert das Ergebnis oder Fehler.

        Args:
            func: Die zu dekorierende Funktion.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität.

        Hinweis:
            Diese Funktion sollte nur verwendet werden, wenn LangSmith-Tracing aktiviert ist.
        """

        @wraps(func)
        async def wrapper(self, session_id: str, message: str) -> Any:
            if self.use_langsmith:
                run_name = (
                    f"Test question: {self.langsmith_client.test_id}" if self.langsmith_client.test_run else "Generation chain"
                )
                run = RunTree(
                    project_name=self.langsmith_client.create_project(
                        self.langsmith_client.agent_name, session_id, self.langsmith_client.test_id
                    ),
                    name=run_name,
                    run_type="chain",
                    inputs={"role": "user", "content": message},
                )
                run.post()
                self.langsmith_client.run_stack.push(run)
                try:
                    result = await func(self, session_id, message)
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                return result
            return await func(self, session_id, message)

        return wrapper

    @staticmethod
    def trace_pipeline(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen der RAG-Pipeline in LangSmith.

        Dieser Dekorator umschließt die Pipeline-Funktion mit LangSmith-Tracing-Logik.
        Er erstellt einen neuen Run für die gesamte Pipeline, führt sie aus und
        protokolliert das Ergebnis oder Fehler.

        Args:
            func: Die zu dekorierende Pipeline-Funktion.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Pipeline-Tracing-Funktionalität.

        Hinweis:
            Dieser Dekorator sollte auf die Hauptfunktion der RAG-Pipeline angewendet werden.
        """

        @wraps(func)
        async def wrapper(self, message: str) -> Any:
            if self.langsmith_client.use_langsmith:
                run = self.langsmith_client.trace_run("RAG Pipeline", "chain", {"role": "user", "content": message})
                self.langsmith_client.run_stack.push(run)
                try:
                    result = await func(self, message)
                    run.end(outputs={"result": result})
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
                    self.langsmith_client.run_stack.pop()
                return result
            return await func(self, message)

        return wrapper

    @staticmethod
    def trace_message_processing(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen der Textvorverarbeitung in LangSmith.

        Dieser Dekorator umschließt die Textvorverarbeitungsfunktion mit Tracing-Logik.
        Er erstellt einen Run für den Vorverarbeitungsschritt, führt ihn aus und
        protokolliert das Ergebnis oder Fehler.

        Args:
            func: Die zu dekorierende Textvorverarbeitungsfunktion.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für die Textvorverarbeitung.

        Hinweis:
            Dieser Dekorator ist speziell für Funktionen gedacht, die Texteingaben vorverarbeiten.
        """

        @wraps(func)
        async def wrapper(self, text: str) -> List[str]:
            if self.langsmith_client.use_langsmith:
                run = self.langsmith_client.trace_run("Textbearbeitung", "tool", {"Nachricht": text})
                try:
                    result = await func(self, text)
                    run.end(outputs={"Liste der Schlüsselwörter aus der Nachricht": result})
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
                return result
            return await func(self, text)

        return wrapper

    @staticmethod
    def trace_message_match_knowledge_context(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen der Schlüsselwort-Übereinstimmungsprüfung in LangSmith.

        Dieser Dekorator umschließt die Funktion, die prüft, ob die Nachricht mit dem
        Wissenskontext übereinstimmt. Er erstellt einen Run für diesen Prüfschritt,
        führt ihn aus und protokolliert das Ergebnis oder Fehler.

        Args:
            func: Die zu dekorierende Funktion für die Schlüsselwort-Übereinstimmungsprüfung.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für die Schlüsselwort-Übereinstimmungsprüfung.

        Hinweis:
            Dieser Dekorator ist speziell für Funktionen gedacht, die Wortlisten mit einem
            Wissenskontext vergleichen.
        """

        @wraps(func)
        async def wrapper(self, word_list: List[str], word_set: Set[str]) -> bool:
            if self.langsmith_client.use_langsmith:
                run = self.langsmith_client.trace_run(
                    "Schlüsselwort-Preprocessings", "tool", {"Liste der Schlüsselwörter aus der Nachricht": word_list}
                )
                try:
                    result = await func(self, word_list, word_set)
                    output = (
                        "Schlüsselwort(e) ist/sind in der Knowledgebase vorhanden."
                        if result
                        else "Schlüsselwort(e) ist/sind in der Knowledgebase nicht vorhanden."
                    )
                    run.end(outputs={"Result": output})
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
                return result
            return await func(self, word_list, word_set)

        return wrapper

    @staticmethod
    def trace_retrieve_documents(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen des Dokumentenabrufs in LangSmith.

        Dieser Dekorator umschließt die Funktion zum Abrufen relevanter Dokumente.
        Er erstellt einen Run für den Abrufprozess, führt ihn aus und protokolliert
        die abgerufenen Dokumente oder Fehler.

        Args:
            func: Die zu dekorierende Funktion für den Dokumentenabruf.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für den Dokumentenabruf.

        Hinweis:
            Dieser Dekorator sollte auf Funktionen angewendet werden, die Dokumente
            basierend auf einer Nachricht und Einstellungen abrufen.
        """

        @wraps(func)
        async def wrapper(self, message: str) -> List[Dict]:
            if self.langsmith_client.use_langsmith:
                run = self.langsmith_client.trace_run(
                    "Dokumente abrufen",
                    "retriever",
                    {
                        "Nachricht": message,
                        "Einstellungen": {
                            "top_k": int(app.config["settings"].get("documentRetrievalSettings", "topK")),
                            "return_full_text_content": bool(
                                app.config["settings"].get("documentRetrievalSettings", "returnFullTextContent")
                            ),
                        },
                    },
                )
                try:
                    result = await func(self, message)
                    run.end(outputs={"Dokumente": result})
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
                return result
            return await func(self, message)

        return wrapper

    @staticmethod
    def trace_search_answer_in_context(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen der Antwortsuche im Kontext in LangSmith.

        Dieser Dekorator umschließt die Funktion, die nach einer Antwort im gegebenen
        Kontext sucht. Er erstellt einen Run für diesen Suchprozess, führt ihn aus und
        protokolliert das Ergebnis oder Fehler.

        Args:
            func: Die zu dekorierende Funktion für die Antwortsuche im Kontext.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für die Antwortsuche im Kontext.

        Hinweis:
            Dieser Dekorator ist für Funktionen gedacht, die prüfen, ob eine Antwort
            auf eine Frage im gegebenen Kontext vorhanden ist.
        """

        @wraps(func)
        async def wrapper(self, message: str, context: str) -> bool:
            if self.use_langsmith:
                run = self.langsmith_client.trace_run(
                    "Suche die Antwort im Kontext", "llm", {"Nachricht": message, "Kontext": context}
                )
                try:
                    result = await func(self, message, context)
                    run.end(outputs={"Result": "Antwort gefunden" if result else "Antwort nicht gefunden"})
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
                return result
            return await func(self, message, context)

        return wrapper

    @staticmethod
    def trace_handle_response(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen der Antwortverarbeitung in LangSmith.

        Dieser Dekorator umschließt die Funktion zur Erstellung und Verarbeitung von Antworten.
        Er erstellt einen Run für den Antwortverarbeitungsprozess, führt ihn aus und
        protokolliert die erstellte Antwort oder Fehler.

        Args:
            func: Die zu dekorierende Funktion für die Antwortverarbeitung.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für die Antwortverarbeitung.

        Hinweis:
            Dieser Dekorator ist für Funktionen gedacht, die Antworten basierend auf
            einem Prompt-Schlüssel, einer Nachricht und optional einem Kontext erstellen.
        """

        @wraps(func)
        async def wrapper(self, message: str, context: Optional[str] = None) -> str:
            if self.use_langsmith:
                run = self.langsmith_client.trace_run(
                    "Erstellung des Prompts", "tool", {
                        "Benutzer Nachricht": message,
                        "Kontext": context if context else "Kein Kontext in der Knowledgebase vorhanden"
                    }
                )
                try:
                    result = await func(self, message, context)
                    run.end(outputs={"Prompt": result})
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
                return result
            return await func(self, message, context)

        return wrapper

    @staticmethod
    def trace_stream_generator(
        func: Callable[[Any, str], Generator[str, None, str]]
    ) -> Callable[[Any, str], Generator[str, None, str]]:
        """
        Ein Dekorator zum Tracen des Stream-Generators in LangSmith.

        Dieser Dekorator umschließt den Stream-Generator mit Tracing-Logik.
        Er erstellt einen Run für den Generierungsprozess, verfolgt die Token-Generierung
        und protokolliert das Endergebnis oder Fehler.

        Args:
            func: Die zu dekorierende Stream-Generator-Funktion.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für die Stream-Generierung.

        Hinweis:
            Dieser Dekorator ist speziell für Generator-Funktionen konzipiert, die
            Text-Tokens streamen.
        """

        @wraps(func)
        def wrapper(self, prompt: str) -> Generator[str, None, str]:
            if self.use_langsmith:
                run = self.langsmith_client.trace_run("Stream Generator", "llm", {"Prompt": prompt})
                run.add_event({"name": "new_token", "time": datetime.datetime.now(datetime.timezone.utc).isoformat()})
                try:
                    yield from func(self, prompt)
                    run.end(
                        outputs={
                            "Generierte Antwort": self.output_tokens,
                            "usage": {
                                "prompt_tokens": len(self.input_tokens),
                                "completion_tokens": len(self.output_tokens),
                                "total_tokens": len(self.input_tokens) + len(self.output_tokens),
                            },
                        }
                    )
                except Exception as e:
                    LangSmithClient.handle_error(run, e)
                    raise
                finally:
                    run.patch()
            else:
                yield from func(self, prompt)

        return wrapper

    @staticmethod
    def trace_response(func: Callable) -> Callable:
        """
        Ein Dekorator zum Tracen der finalen Antwortgenerierung in LangSmith.

        Dieser Dekorator umschließt die Funktion, die die endgültige Antwort generiert.
        Er beendet den aktuellen Run, protokolliert die generierte Antwort oder Fehler
        und aktualisiert den LangSmith-Trace.

        Args:
            func: Die zu dekorierende Funktion für die finale Antwortgenerierung.

        Returns:
            Die dekorierte Funktion mit hinzugefügter Tracing-Funktionalität für die finale Antwortgenerierung.

        Hinweis:
            Dieser Dekorator sollte auf die Funktion angewendet werden, die die
            endgültige Antwort des Systems zurückgibt.
        """

        @wraps(func)
        def wrapper(self) -> Any:
            if self.use_langsmith:
                try:
                    result = func(self)
                    run = self.langsmith_client.run_stack.pop()
                    run.end(outputs={"role": "assistant", "content": result})
                    run.patch()
                    return result
                except Exception as e:
                    if not self.langsmith_client.run_stack.is_empty():
                        run = self.langsmith_client.run_stack.pop()
                        LangSmithClient.handle_error(run, e)
                    raise
            else:
                return func(self)

        return wrapper


class RunStack:
    """
    Eine Klasse zur Verwaltung eines Stacks von RunTree-Objekten.

    Diese Klasse implementiert grundlegende Stack-Operationen für RunTree-Objekte,
    die für das Tracing von verschachtelten Läufen verwendet werden.

    Attributes:
        stack: Eine Liste zur Speicherung von RunTree-Objekten.
    """

    def __init__(self) -> None:
        self.stack: List[RunTree] = []

    def push(self, run: RunTree) -> None:
        self.stack.append(run)

    def pop(self) -> RunTree:
        if self.stack:
            return self.stack.pop()
        raise IndexError("Run stack is empty")

    def peek(self) -> RunTree:
        if self.stack:
            return self.stack[-1]
        raise IndexError("Run stack is empty")

    def is_empty(self) -> bool:
        return len(self.stack) == 0
