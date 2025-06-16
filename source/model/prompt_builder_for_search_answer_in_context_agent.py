"""Erstellt einen Prompt, der prüft, ob die Antwort auf eine Frage im gegebenen Kontext vorhanden ist."""

from langchain_core.prompts import ChatPromptTemplate

from source.model.prompt_builder_for_phi_4 import create_prompt_for_phi_4


async def create_prompt_for_search_answer_in_context(chat_template: ChatPromptTemplate, context: str) -> str:
    """Erstellt einen Prompt, der prüft, ob die Antwort auf eine Frage im gegebenen Kontext vorhanden ist.

    Diese Funktion generiert einen formatierten Prompt, der eine Frage und einen gegebenen Kontext verarbeitet.
    Das rag-Sprachmodell soll basierend auf der Eingabe entscheiden, ob die Antwort auf die Frage im Kontext
    enthalten ist. Das Modell gibt "JA" zurück, wenn die Antwort im Kontext gefunden wird, und "NEIN",
    wenn sie nicht vorhanden ist.

    Args:
        chat_template: Die Chat-Vorlage, die alle Nachrichten enthält und die Struktur des Prompts definiert.
        context: Ein Kontext, der in den Prompt integriert werden soll.

    Returns:
        Der erstellte Prompt als formatierter String, bereit für die Eingabe in das rag-Sprachmodell.
    """
    return await create_prompt_for_phi_4(chat_template=chat_template.messages, context=context)
