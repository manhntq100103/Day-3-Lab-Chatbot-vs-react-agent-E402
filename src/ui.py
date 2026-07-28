"""Gradio interface for the gift-recommendation ReAct agent."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any

import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import run_react_agent
from providers import get_llm_provider


PROVIDER = get_llm_provider()
MAX_HISTORY_TURNS = 6


def _conversation_context(history: list[dict[str, str]], message: str) -> str:
    """Convert the recent browser-session history into a provider prompt."""
    recent_messages = history[-MAX_HISTORY_TURNS * 2 :]
    turns = [
        f"{item['role'].capitalize()}: {item['content']}"
        for item in recent_messages
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    turns.append(f"User: {message}")
    return "Conversation history:\n" + "\n".join(turns)


def chat(message: str, history: list[dict[str, str]] | None) -> tuple[list[dict[str, str]], str, str]:
    """Handle one message and retain history in the active Gradio browser session."""
    history = history or []
    message = message.strip()
    if not message:
        return history, "", ""

    trace_stream = io.StringIO()
    with contextlib.redirect_stdout(trace_stream):
        answer = run_react_agent(_conversation_context(history, message), PROVIDER)

    updated_history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, "", trace_stream.getvalue().strip()


with gr.Blocks(title="Gift Recommendation Agent") as demo:
    gr.Markdown(
        "# Gift Recommendation Agent\n"
        "Your messages are remembered for this browser session. "
        "Open **Agent trace** to inspect Thought → Action → Observation."
    )
    chatbot = gr.Chatbot(label="Conversation", height=480)
    with gr.Row():
        message = gr.Textbox(
            label="Message",
            placeholder="Describe the recipient, occasion, interests, and budget...",
            lines=2,
            scale=8,
        )
        send = gr.Button("Send", variant="primary", scale=1)
    clear = gr.Button("Clear conversation")
    with gr.Accordion("Agent trace", open=False):
        trace = gr.Textbox(label="Thought → Action → Observation", lines=14, interactive=False)

    message.submit(chat, inputs=[message, chatbot], outputs=[chatbot, message, trace])
    send.click(chat, inputs=[message, chatbot], outputs=[chatbot, message, trace])
    clear.click(lambda: ([], ""), outputs=[chatbot, trace])


if __name__ == "__main__":
    demo.launch()
