"""Turn the shared transcript into model input (used by both chat and voice)."""

from __future__ import annotations

from app.agent.state import Message
from app.llm.base import AssistantText, Item, UserText

MAX_HISTORY_MESSAGES = 20


def transcript_items(messages: list[Message], limit: int = MAX_HISTORY_MESSAGES) -> list[Item]:
    """Prior turns as plain text. Tool calls from earlier turns are NOT replayed; their
    durable results live in the session facts block instead. That keeps the context small,
    and it avoids replaying provider-specific tool-call metadata across turns and modalities."""
    items: list[Item] = []
    for m in messages[-limit:]:
        text = m.text
        if m.role == "assistant" and m.meta.get("interrupted"):
            text += " [the customer interrupted here and did not hear the rest]"
        if m.modality == "voice":
            text = f"(spoken) {text}"
        items.append(UserText(text) if m.role == "user" else AssistantText(text))
    # Gemini requires the conversation to start with a user turn.
    while items and not isinstance(items[0], UserText):
        items.pop(0)
    return items
