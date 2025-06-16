"""
Dieses Modul implementiert automatisierte Tests für einen KI-Agenten unter Verwendung von LangSmith.

Es enthält Funktionen zum Laden von Testfragen, Durchführen von Tests und Starten des Testprozesses.
"""

import asyncio
import json
import uuid
from typing import List, Dict

from flask import current_app as app, Flask

from source.model.agent import Agent


async def load_questions(file_path: str) -> List[Dict[str, str]] | Dict[str, dict]:
    """
    Liest Testfragen aus einer JSONL-Datei oder Testchats aus einer JSON-Datei.

    Args:
        file_path: Der Pfad zur JSONL- oder JSON Datei mit den Testfragen/Testchats.

    Returns:
        Eine Liste von Dictionaries, wobei jedes Dictionary eine Testfrage repräsentiert.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in file]
        elif file_path.endswith('.json'):
            return json.load(file)


async def run_tests(questions: List[Dict[str, str]] = None, test_chats: Dict[str, dict] = None) -> None:
    """
    Führt Tests für eine Liste von Fragen mit einem KI-Agenten durch.

    Diese Funktion initialisiert einen Agenten, verarbeitet jede Frage und protokolliert die Ergebnisse.

    Args:
        questions: Eine Liste von Dictionaries, die die Testfragen enthalten.
        test_chats: Die Testchats, die die Testfragen enthalten.
    """
    agent = Agent(
        use_langsmith=bool(app.config["settings"].get("langSmithSettings", "useLangsmithTestEnvironment")),
        langsmith_client_name="TestRun",
    )
    if questions:
        # Sammeln von Testfragen in einem LangSmith-Projekt
        agent.langsmith_client.agent_name += f" Questions ID: {uuid.uuid4().hex[:8]}"
        for question in questions:
            if not question["id"].startswith("Q"):
                app.logger.warning(f"Question {question["id"]} skipped. Question Id must begin with the letter \"Q\"")
                continue  # Diese Frage überspringen
            try:
                agent.langsmith_client.test_run = True
                agent.langsmith_client.test_id = question["id"]
                app.logger.info(f"Running test for question {question["id"]}")
                test_session_id = uuid.uuid4().hex[:8]
                "".join(await agent(session_id=test_session_id, message=question["question"]))
                agent.reset_response_data()
                app.logger.info(f"Test for question {question["id"]} completed")
            except Exception as e:
                app.logger.error(f"Error processing question {question['id']}: {e}")
    elif test_chats:
        for chat in test_chats:
            if not test_chats[chat]["id"].startswith("C"):
                app.logger.warning(f"Chat {test_chats[chat]["id"]} skipped. Chat Id must begin with the letter \"C\"")
                continue  # Dieser Chat überspringen
            try:
                agent.langsmith_client.test_run = True
                agent.langsmith_client.test_id = test_chats[chat]["id"]
                app.logger.info(f"Running test for chat {test_chats[chat]["id"]}")
                test_session_id = uuid.uuid4().hex[:8]
                for message in test_chats[chat]["messages"]:
                    "".join(await agent(session_id=test_session_id, message=message))
                    agent.reset_response_data()
                app.logger.info(f"Test for chat {test_chats[chat]["id"]} completed")
            except Exception as e:
                app.logger.error(f"Error processing chat {test_chats[chat]['id']}: {e}")


async def start_tests(current_app: Flask) -> None:
    """
    Startet den Testprozess innerhalb des Flask-Anwendungskontexts.

    Diese Funktion lädt die Testfragen und führt die Tests aus.

    Args:
        current_app: Die aktuelle Flask-Anwendungsinstanz.
    """
    with current_app.app_context():
        questions = await load_questions(app.config["settings"].get("langSmithSettings", "questionsFilePath"))
        await run_tests(questions=questions)
        test_chats = await load_questions(app.config["settings"].get("langSmithSettings", "testChatsFilePath"))
        await run_tests(test_chats=test_chats)
        app.logger.info("All tests completed!")


def run_automated_tests_in_langsmith(current_app: Flask) -> None:
    """
    Führt automatisierte Tests in LangSmith aus.

    Diese Funktion ist der Haupteinstiegspunkt für die Ausführung der automatisierten Tests.
    Sie verwendet asyncio, um die asynchronen Testfunktionen auszuführen.

    Args:
        current_app: Die aktuelle Flask-Anwendungsinstanz.
    """
    asyncio.run(start_tests(current_app))
