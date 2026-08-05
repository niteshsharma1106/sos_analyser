export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

export type ChatSettings = {
  db_path: string;
  offline: boolean;
  model: string;
  focus_entity: string;
  include_graph: boolean;
  answer_style: string;
  show_observability: boolean;
};

export type BootstrapResponse = {
  cluster_banner: string;
  llm_status: string;
  defaults: ChatSettings;
  examples: { label: string; prompt: string }[];
};

export async function fetchBootstrap(): Promise<BootstrapResponse> {
  const res = await fetch("/api/bootstrap");
  if (!res.ok) {
    throw new Error(`Failed to load UI bootstrap (${res.status})`);
  }
  return res.json();
}

export async function askQuestion(
  message: string,
  settings: ChatSettings,
): Promise<{ answer: string }> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...settings }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}
