/**
 * AMPL Automotive Chat Widget
 *
 * Embeddable chat widget that communicates with the AMPL Chatbot API.
 * Usage: include this script and call AMPLChat.init({ apiUrl: "..." })
 */
(function () {
  "use strict";

  const DEFAULT_API = "http://localhost:8000/api/v1";

  // ── State ──────────────────────────────────────────
  let conversationId = null;
  let isOpen = false;
  let isSending = false;

  // ── Config ─────────────────────────────────────────
  const config = {
    apiUrl: DEFAULT_API,
    apiKey: null,
  };

  // ── DOM references ─────────────────────────────────
  const $ = (id) => document.getElementById(id);

  function init(opts) {
    Object.assign(config, opts || {});
    bind();
  }

  function bind() {
    $("ampl-chat-toggle").addEventListener("click", toggleChat);
    $("chat-send-btn").addEventListener("click", sendMessage);
    $("chat-clear-btn").addEventListener("click", clearChat);
    $("chat-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Suggested action buttons
    document.querySelectorAll(".action-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const msg = btn.dataset.msg;
        if (msg) {
          $("chat-input").value = msg;
          sendMessage();
        }
      });
    });
  }

  // ── Toggle chat ────────────────────────────────────
  function toggleChat() {
    isOpen = !isOpen;
    $("ampl-chat-window").classList.toggle("hidden", !isOpen);
    $("chat-icon-open").style.display = isOpen ? "none" : "block";
    $("chat-icon-close").style.display = isOpen ? "block" : "none";
    if (isOpen) $("chat-input").focus();
  }

  // ── Send message ───────────────────────────────────
  async function sendMessage() {
    const input = $("chat-input");
    const text = input.value.trim();
    if (!text || isSending) return;

    input.value = "";
    isSending = true;
    $("chat-send-btn").disabled = true;

    // Hide actions after first user message
    $("chat-actions").style.display = "none";

    appendMessage("user", text);
    showTyping();

    try {
      const resp = await fetch(`${config.apiUrl}/chat`, {
        method: "POST",
        headers: buildHeaders(),
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId,
          include_sources: true,
          max_chunks: 5,
        }),
      });

      removeTyping();

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      conversationId = data.conversation_id;

      appendBotMessage(data);
      updateActions(data.suggested_actions || []);
    } catch (err) {
      removeTyping();
      appendMessage(
        "bot",
        "Sorry, I'm having trouble connecting. Please try again in a moment."
      );
      console.error("AMPL Chat error:", err);
    } finally {
      isSending = false;
      $("chat-send-btn").disabled = false;
      input.focus();
    }
  }

  // ── Clear chat ─────────────────────────────────────
  async function clearChat() {
    if (conversationId) {
      try {
        await fetch(`${config.apiUrl}/chat/${conversationId}`, {
          method: "DELETE",
          headers: buildHeaders(),
        });
      } catch (_) {}
    }

    conversationId = null;
    const msgs = $("chat-messages");
    msgs.innerHTML = "";
    appendMessage(
      "bot",
      "Chat cleared! How can I help you today?"
    );
    $("chat-actions").style.display = "flex";
  }

  // ── DOM helpers ────────────────────────────────────
  function appendMessage(role, text) {
    const msgs = $("chat-messages");
    const wrap = document.createElement("div");
    wrap.className = `message ${role}`;

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;

    wrap.appendChild(content);
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function appendBotMessage(data) {
    const msgs = $("chat-messages");
    const wrap = document.createElement("div");
    wrap.className = "message bot";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = data.response;

    // Lead score badge (only for non-info intents)
    if (data.lead_score && data.lead_priority && data.lead_priority !== "cold") {
      const badge = document.createElement("div");
      badge.className = `lead-badge ${data.lead_priority}`;
      badge.textContent = `${data.lead_priority.toUpperCase()} LEAD (${data.lead_score})`;
      content.appendChild(badge);
    }

    // Sources
    if (data.sources && data.sources.length > 0) {
      const src = document.createElement("div");
      src.className = "sources";
      src.textContent =
        "Sources: " + data.sources.map((s) => s.source).join(", ");
      content.appendChild(src);
    }

    wrap.appendChild(content);
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showTyping() {
    const msgs = $("chat-messages");
    const wrap = document.createElement("div");
    wrap.className = "message bot";
    wrap.id = "typing-msg";

    const indicator = document.createElement("div");
    indicator.className = "typing-indicator";
    indicator.innerHTML = "<span></span><span></span><span></span>";

    wrap.appendChild(indicator);
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function removeTyping() {
    const el = $("typing-msg");
    if (el) el.remove();
  }

  function updateActions(actions) {
    const container = $("chat-actions");
    if (!actions.length) {
      container.style.display = "none";
      return;
    }
    container.innerHTML = "";
    actions.forEach((label) => {
      const btn = document.createElement("button");
      btn.className = "action-btn";
      btn.textContent = label;
      btn.dataset.msg = label;
      btn.addEventListener("click", () => {
        $("chat-input").value = label;
        sendMessage();
      });
      container.appendChild(btn);
    });
    container.style.display = "flex";
  }

  function buildHeaders() {
    const h = { "Content-Type": "application/json" };
    if (config.apiKey) h["X-API-Key"] = config.apiKey;
    return h;
  }

  // ── Expose global API ──────────────────────────────
  window.AMPLChat = { init };

  // Auto-init if no manual call
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init());
  } else {
    init();
  }
})();
