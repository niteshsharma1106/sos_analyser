import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  askQuestion,
  ChatMessage,
  ChatSettings,
  fetchBootstrap,
} from "./api";
import "./App.css";

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "## Start an investigation\nAsk about a host, VM, port, reboot, or an error in your SOS reports. Use a quick prompt below to get started.",
};

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [clusterBanner, setClusterBanner] = useState("");
  const [llmStatus, setLlmStatus] = useState("Loading…");
  const [examples, setExamples] = useState<{ label: string; prompt: string }[]>(
    [],
  );
  const [bootError, setBootError] = useState<string | null>(null);
  const [settings, setSettings] = useState<ChatSettings>({
    db_path: "sos_analysis.duckdb",
    offline: false,
    model: "",
    focus_entity: "",
    include_graph: true,
    answer_style: "Concise RCA",
    show_observability: false,
  });

  const listRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let cancelled = false;
    fetchBootstrap()
      .then((data) => {
        if (cancelled) return;
        setClusterBanner(data.cluster_banner || "");
        setLlmStatus(data.llm_status || "");
        setSettings(data.defaults);
        setExamples(data.examples || []);
      })
      .catch((err: Error) => {
        if (!cancelled) setBootError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, pending]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  }, [draft]);

  async function sendPrompt(raw: string) {
    const text = raw.trim();
    if (!text || pending) return;

    const userMsg: ChatMessage = { id: newId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setDraft("");
    setPending(true);

    try {
      const { answer } = await askQuestion(text, settings);
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "assistant", content: answer },
      ]);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: `Investigation failed: ${detail}`,
        },
      ]);
    } finally {
      setPending(false);
      textareaRef.current?.focus();
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void sendPrompt(draft);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendPrompt(draft);
    }
  }

  if (bootError) {
    return (
      <div className="boot-error">
        Could not start the chat UI: {bootError}
      </div>
    );
  }

  return (
    <div className="shell">
      <header className="brand">
        <div className="brand-row">
          <p className="mark">
            OSP <span>SOS</span>
          </p>
          {clusterBanner ? <p className="cluster">{clusterBanner}</p> : null}
        </div>
        <p className="sub">
          Investigate OpenStack SOS reports — reboots, ports, and cross-node
          failures.
        </p>
      </header>

      <section className="chat" aria-label="Conversation">
        <div className="messages" ref={listRef}>
          {messages.map((msg) => (
            <div key={msg.id} className={`message-row ${msg.role}`}>
              <div className={`bubble ${msg.role}`}>
                {msg.role === "assistant" ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          ))}
          {pending ? (
            <div className="message-row assistant">
              <div className="bubble assistant pending">Investigating…</div>
            </div>
          ) : null}
        </div>
      </section>

      <div className="suggestions">
        {examples.map((ex) => (
          <button
            key={ex.label}
            type="button"
            disabled={pending}
            onClick={() => void sendPrompt(ex.prompt)}
          >
            {ex.label}
          </button>
        ))}
      </div>

      <form className="composer-wrap" onSubmit={onSubmit}>
        <textarea
          ref={textareaRef}
          className="composer"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about a reboot, port, or host…"
          rows={1}
          disabled={pending}
          aria-label="Message"
        />
        <button
          className="send"
          type="submit"
          disabled={pending || !draft.trim()}
          aria-label="Send"
        >
          →
        </button>
      </form>

      <details className="settings">
        <summary>Settings</summary>
        <div className="settings-body">
          <p className="llm-status">
            <strong>LLM config</strong> (from `.env` only — not editable here)
            <br />
            {llmStatus}
          </p>

          <div className="field">
            <label htmlFor="db_path">DuckDB path</label>
            <input
              id="db_path"
              type="text"
              value={settings.db_path}
              onChange={(e) =>
                setSettings((s) => ({ ...s, db_path: e.target.value }))
              }
            />
          </div>

          <div className="field checkbox">
            <input
              id="offline"
              type="checkbox"
              checked={settings.offline}
              onChange={(e) =>
                setSettings((s) => ({ ...s, offline: e.target.checked }))
              }
            />
            <label htmlFor="offline">Offline mode (no LLM)</label>
          </div>

          <div className="field">
            <label htmlFor="focus_entity">Focused entity</label>
            <input
              id="focus_entity"
              type="text"
              placeholder="VM / port / volume / req-id / hostname"
              value={settings.focus_entity}
              onChange={(e) =>
                setSettings((s) => ({ ...s, focus_entity: e.target.value }))
              }
            />
          </div>

          <div className="field checkbox">
            <input
              id="include_graph"
              type="checkbox"
              checked={settings.include_graph}
              onChange={(e) =>
                setSettings((s) => ({
                  ...s,
                  include_graph: e.target.checked,
                }))
              }
            />
            <label htmlFor="include_graph">Use relationship graph</label>
          </div>

          <div className="field">
            <label htmlFor="answer_style">Answer style</label>
            <select
              id="answer_style"
              value={settings.answer_style}
              onChange={(e) =>
                setSettings((s) => ({ ...s, answer_style: e.target.value }))
              }
            >
              <option>Concise RCA</option>
              <option>Evidence-heavy</option>
              <option>Operation path first</option>
            </select>
          </div>

          <div className="field checkbox">
            <input
              id="show_observability"
              type="checkbox"
              checked={settings.show_observability}
              onChange={(e) =>
                setSettings((s) => ({
                  ...s,
                  show_observability: e.target.checked,
                }))
              }
            />
            <label htmlFor="show_observability">
              Show agent observability timeline
            </label>
          </div>
        </div>
      </details>
    </div>
  );
}
