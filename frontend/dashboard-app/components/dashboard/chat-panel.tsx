"use client";

import { useState } from "react";
import { MessageCircle, SendHorizontal, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { sendChatMessage } from "@/lib/api";
import { useDashboardStore } from "@/store/dashboard-store";

interface ChatItem {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const QUICK_PROMPTS = [
  "Explain Layer 1 verdict",
  "Is this a known deepfake?",
  "What are the spread risks?",
];

export function ChatPanel() {
  const result = useDashboardStore((state) => state.result);
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [messages, setMessages] = useState<ChatItem[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hi! I'm your Forensic AI. Ask me anything about the authenticity, source matches, or risks associated with this media.",
    },
  ]);

  const onSend = async (forcedMessage?: string) => {
    const message = (forcedMessage || input).trim();
    if (!message || isSending) {
      return;
    }

    const nextUser: ChatItem = { id: `${Date.now()}-u`, role: "user", content: message };
    setMessages((prev) => [...prev, nextUser]);
    setInput("");
    setIsSending(true);

    try {
      const response = await sendChatMessage({
        message,
        analysis_id: result?.analysis_id,
        layer1: result?.layer1 as Record<string, unknown> | undefined,
        layer2: result?.layer2 as Record<string, unknown> | undefined,
        layer3: result?.layer3 as Record<string, unknown> | undefined,
        history: messages.map(m => ({ role: m.role, content: m.content })),
      });
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-a`,
          role: "assistant",
          content: response.reply,
        },
      ]);
    } catch (error) {
      const fallback = error instanceof Error ? error.message : "Chat failed.";
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-e`,
          role: "assistant",
          content: fallback,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <Card className="border-white/10 bg-white/[0.02] backdrop-blur-md">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-[#4facfe] to-[#00f2fe] shadow-[0_0_15px_rgba(79,172,254,0.4)]">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <CardTitle className="text-lg font-bold tracking-tight text-white">Forensic Intelligence Agent</CardTitle>
              <CardDescription className="text-blue-200/60">Powered by Gemini 2.0 Flash</CardDescription>
            </div>
          </div>
          <Button 
            variant="ghost" 
            className="text-blue-300 hover:bg-white/5 hover:text-white"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "Minimize" : "Ask for Insight"}
          </Button>
        </div>
      </CardHeader>
      {open && (
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {QUICK_PROMPTS.map((p) => (
              <button
                key={p}
                onClick={() => onSend(p)}
                disabled={isSending}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-blue-100/70 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-50"
              >
                {p}
              </button>
            ))}
          </div>

          <div className="max-h-[350px] space-y-4 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-white/10">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "assistant" ? "justify-start" : "justify-end"}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-lg ${
                    m.role === "assistant" 
                      ? "border border-white/10 bg-white/5 text-blue-50" 
                      : "bg-blue-600/80 text-white"
                  }`}
                >
                  {m.role === "assistant" ? (
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    m.content
                  )}
                </div>
              </div>
            ))}
            {isSending && (
              <div className="flex justify-start">
                <div className="animate-pulse rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-blue-200/60">
                  Analyzing forensic context...
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 pt-2">
            <input
              type="text"
              className="h-11 w-full rounded-2xl border border-white/10 bg-black/40 px-4 text-sm text-white transition-all outline-none placeholder:text-blue-100/30 focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50"
              placeholder="Ask a forensic question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  onSend();
                }
              }}
            />
            <Button 
              size="icon"
              onClick={() => onSend()} 
              disabled={isSending || !input.trim()} 
              className="h-11 w-11 shrink-0 rounded-xl bg-blue-600 shadow-[0_4px_10px_rgba(37,99,235,0.3)] transition-transform hover:scale-105 active:scale-95"
            >
              <SendHorizontal className="h-5 w-5" />
            </Button>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
