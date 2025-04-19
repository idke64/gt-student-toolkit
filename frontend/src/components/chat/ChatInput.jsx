import { useState } from "react";

export default function ChatInput({ handleSendMessage }) {
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setSending(true);
    if (!message.trim()) return;

    handleSendMessage(message);
    setSending(false);
    setMessage("");
  };

  return (
    <form
      className="p-4 border-t border-gray-200 bg-white"
      onSubmit={handleSubmit}
    >
      <div className="flex relative">
        <input
          type="text"
          placeholder="What's on your mind?"
          className="text-input w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-100"
          required
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          disabled={sending}
        />
        <button
          type="submit"
          className="absolute right-0 bg-primary-100 hover:bg-yellow-500 text-white font-semibold px-4 py-2 rounded-r-lg transition-colors"
          disabled={sending || !message.trim()}
        >
          {sending ? "Sending..." : "Send"}
        </button>
      </div>
    </form>
  );
}
