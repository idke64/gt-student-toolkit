import { useEffect, useState } from "react";
import { useStore } from "@nanostores/react";
import { isAuth, user } from "../../firebase/auth";
import Sidebar from "./Sidebar.jsx";
import Chat from "./Chat.jsx";

export default function ChatContainer({ id }) {
  const [chats, setChats] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const $isAuth = useStore(isAuth);
  const $user = useStore(user);

  const API_URL = import.meta.env.PUBLIC_API_URL;

  useEffect(() => {
    const fetchChats = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/users/${$user.id}/chats`);

        if (!response.ok) {
          throw new Error("Failed to fetch chats");
        }

        const data = await response.json();

        setChats(data.chats);
      } catch (error) {
        setErrorMsg(error.message);
      } finally {
        setLoading(false);
      }
    };

    if (!$isAuth) return;
    fetchChats();
  }, [$isAuth]);

  return (
    <div className="flex h-[calc(100vh-68px)] mt-[68px] bg-gray-50">
      <Sidebar id={id} chats={chats} />
      <Chat id={id} />
    </div>
  );
}
