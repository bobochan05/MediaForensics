"use client";

import { useState } from "react";
import { MessageCircle, SendHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { sendChatMessage } from "@/lib/api";
import { useDashboardStore } from "@/store/dashboard-store";

interface ChatItem {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export function ChatPanel() {
  const result = useDashboardStore((state) => state.result);
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [messages, setMessages] = useState<ChatItem[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Ask about authenticity, detection results, risk meaning, or source tracing.",
    },
  ]);

  const onSend = async () => {
    const message = input.trim();
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
        layer1: result?.layer1 as Record<string, unknown> | undefined,
        layer2: result?.layer2 as Record<string, unknown> | undefined,
        layer3: result?.layer3 as Record<string, unknown> | undefined,
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
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <MessageCircle className="h-4 w-4 text-[#83beff]" />
              Intelligence Chat
            </CardTitle>
            <CardDescription>Assistant for media authenticity analysis and tracing results.</CardDescription>
          </div>
          <Button variant="outline" onClick={() => setOpen((v) => !v)}>
            {open ? "Hide" : "Open"}
          </Button>
        </div>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          <div className="max-h-64 space-y-2 overflow-auto rounded-xl border border-white/10 bg-white/[0.02] p-3">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`rounded-lg px-3 py-2 text-sm ${m.role === "assistant" ? "bg-white/[0.05] text-[#dbe8ff]" : "bg-[#2b5aa9]/50 text-white"}`}
              >
                {m.content}
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              className="h-10 w-full rounded-xl border border-white/15 bg-black/25 px-3 text-sm text-[#e5edff] outline-none placeholder:text-[#7f95ba] focus:border-[#5ca7ff]"
              placeholder="Ask about this media analysis..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  onSend();
                }
              }}
            />
            <Button onClick={onSend} disabled={isSending || !input.trim()} className="gap-2">
              <SendHorizontal className="h-4 w-4" />
              Send
            </Button>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
