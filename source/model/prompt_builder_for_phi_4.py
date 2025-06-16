"""Dieses Modul erstellt einen Prompt für das Sprachmodell."""

from flask import current_app as app
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


async def create_prompt_for_phi_4(chat_template: ChatPromptTemplate | AnyMessage, context: str | None = None) -> str:
    """Erstellt einen Prompt für das Phi-4 Sprachmodell basierend auf einer Chat-Vorlage und optionalem Kontext.

    Diese Funktion generiert einen formatierten Prompt-String für das Phi-4 Sprachmodell.
    Sie verarbeitet verschiedene Nachrichtentypen (System, Human, AI) aus der Chat-Vorlage
    und fügt bei Bedarf einen Kontext hinzu.

    Args:
        chat_template: Die Chat-Vorlage, die die Struktur und den Inhalt des Prompts definiert.
        context: Der optionale Kontext, der in den Prompt eingefügt werden soll. Standardmäßig None.

    Returns:
        Der erstellte Prompt als formatierter String, bereit für die Eingabe in das Phi-4 Sprachmodell.

    Examples:
        >>> template = ChatPromptTemplate.from_messages(
        ...     [
        ...         SystemMessage(content="Du bist ein hilfreicher Assistent."),
        ...         HumanMessage(content="Was ist die Hauptstadt von Frankreich?"),
        ...     ]
        ... )
        >>> knowledge_context = "Frankreich ist ein Land in Westeuropa. Die Hauptstadt von Frankreich ist Paris."
        >>> prompt = create_prompt_for_phi_4(template, knowledge_context)
        >>> print(prompt)
        <|im_start|>system<|im_sep|>
        Du bist ein hilfreicher Assistent.<|im_end|>
        <|im_start|>user<|im_sep|>
        Was ist die Hauptstadt von Frankreich?

        Kontext: Frankreich ist ein Land in Westeuropa. Die Hauptstadt von Frankreich ist Paris.<|im_end|>
        <|im_start|>assistant<|im_sep|>

    """
    prompt_parts = {
        "system": "<|im_start|>system<|im_sep|>\n",
        "user": "<|im_start|>user<|im_sep|>\n",
        "assistant": "<|im_start|>assistant<|im_sep|>\n",
        "suffix": "<|im_end|>\n",
        "context": "\n\nKontext: ",
    }
    prompt_str = ""

    for message in chat_template:
        if isinstance(message, SystemMessage):
            prompt_str += f"{prompt_parts['system']}{message.content}{prompt_parts['suffix']}"
        elif isinstance(message, HumanMessage):
            prompt_str += f"{prompt_parts['user']}{message.content}"
            if context:
                prompt_str += f"{prompt_parts['context']}{context}"
            prompt_str += f"{prompt_parts['suffix']}{prompt_parts['assistant']}"
        elif isinstance(message, AIMessage):
            prompt_str += f"{message.content}{prompt_parts['suffix']}"

    return prompt_str


async def limit_chat_history(chat_template: ChatPromptTemplate) -> list[AnyMessage]:
    """Begrenzt den Chatverlauf basierend auf der Tokenanzahl.

    Args:
        chat_template: Die Chat-Vorlage, die alle Nachrichten enthält.

    Returns:
        Eine gefilterte Liste des Chatverlaufs, die das Tokenlimit berücksichtigt.
        Die neuesten Nachrichten werden priorisiert, und die Liste wird in
        chronologischer Reihenfolge zurückgegeben.

    Raises:
        AttributeError: Wenn eine Nachricht in der Vorlage kein Attribut 'token_count' hat.
    """
    max_tokens = app.config["system_prompts"].max_chat_history_length
    filtered_messages = []
    total_tokens = 0

    # Nachrichten in umgekehrter Reihenfolge durchgehen, um die neuesten zu priorisieren
    for message in reversed(chat_template.messages):
        # SystemMessage immer einfügen, ohne das Tokenlimit zu prüfen
        if isinstance(message, SystemMessage):
            filtered_messages.append(message)
            continue

        if isinstance(message, (HumanMessage, AIMessage)):
            if not hasattr(message, "token_count"):
                error_msg = f"Message of type {type(message).__name__} is missing the 'token_count' attribute"
                app.logger.error(error_msg)
                raise AttributeError(error_msg)

            if total_tokens + message.token_count > max_tokens:
                break

            filtered_messages.append(message)
            total_tokens += message.token_count

    # Die Liste umkehren, da wir in umgekehrter Reihenfolge hinzugefügt haben
    return filtered_messages[::-1]
