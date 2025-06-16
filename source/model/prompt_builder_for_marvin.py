"""Erstellt einen Prompt für das rag-Sprachmodell basierend auf einer Chat-Vorlage und optionalem Kontext."""

from langchain_core.prompts import ChatPromptTemplate

from source.model.prompt_builder_for_phi_4 import limit_chat_history, create_prompt_for_phi_4


async def create_prompt_for_rag(chat_template: ChatPromptTemplate, context: str | None = None) -> str:
    """Erstellt einen Prompt für das rag-Sprachmodell basierend auf einer Chat-Vorlage und optionalem Kontext.

    Diese Funktion verarbeitet eine Chat-Vorlage, begrenzt den Chatverlauf basierend auf einem Tokenlimit
    und erstellt anschließend einen formatierten Prompt-String für das rag-Sprachmodell.
    Dabei wird optional ein Kontext berücksichtigt, der in den Prompt eingefügt werden kann.

    Args:
        chat_template: Die Chat-Vorlage, die alle Nachrichten enthält und die Struktur des Prompts definiert.
        context: Ein optionaler Kontext, der in den Prompt integriert werden kann. Standardmäßig None.

    Returns:
        Der erstellte Prompt als formatierter String, bereit für die Eingabe in das rag-Sprachmodell.
    """
    # Bereinigen des Chatverlaufs unter Berücksichtigung des Tokenlimits
    filtered_messages = await limit_chat_history(chat_template)
    return await create_prompt_for_phi_4(chat_template=filtered_messages, context=context)
