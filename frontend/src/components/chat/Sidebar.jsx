import { useStore } from "@nanostores/react";
import { user } from "../../firebase/auth";

export default function Sidebar({ chats, id }) {
  const $user = useStore(user);

  const API_URL = import.meta.env.PUBLIC_API_URL;

  const handleCreateChat = async () => {
    try {
      const response = await fetch(`${API_URL}/api/users/${$user.id}/chats`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: "New Chat" }),
      });

      if (!response.ok) {
        throw new Error("Failed to create chat");
      }

      const data = await response.json();

      window.location.href = `/chat/${data.chat.id}`;

      return data.chat;
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <aside className="w-64 bg-white border-r border-gray-200 p-4">
      <div className="mb-4">
        <button
          onClick={handleCreateChat}
          type="button"
          className="btn-primary w-full py-2 block text-center"
        >
          New Chat +
        </button>
      </div>

      <hr className="border-gray-200 mb-4" />

      <nav className="space-y-2">
        {chats.map((chat) => (
          <a
            key={chat.id}
            href={`/chat/${chat.id}`}
            className={`flex items-center gap-3 p-2 rounded duration-200 text-sm ${
              id === chat.id
                ? "bg-yellow-100 text-yellow-800"
                : "hover:bg-gray-100 border-transparent"
            }`}
          >
            <span>{chat.title}</span>
          </a>
        ))}
      </nav>
    </aside>
  );
}
