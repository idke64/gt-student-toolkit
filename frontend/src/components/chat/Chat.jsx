import Response from "./Response";
import ChatInput from "./ChatInput";
import Message from "./Message";
import { useStore } from "@nanostores/react";
import { isAuth, user } from "../../firebase/auth";
import { useEffect, useState, useRef } from "react";

export default function Chat({ id }) {
  const [messages, setMessages] = useState([]);

  const [chat, setChat] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const [connected, setConnected] = useState(false);

  const socketRef = useRef(null);
  const messageContainerRef = useRef(null);

  const API_URL = import.meta.env.PUBLIC_API_URL;
  const LLM_URL = import.meta.env.PUBLIC_LLM_URL;

  const $isAuth = useStore(isAuth);
  const $user = useStore(user);

  useEffect(() => {
    if (messageContainerRef.current) {
      messageContainerRef.current.scrollTop =
        messageContainerRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (!$isAuth || !$user) return;

    const ws = new WebSocket(LLM_URL);
    socketRef.current = ws;

    ws.onopen = () => {
      console.log("Connected");
      const roomId = id;
      const clientId = $user.id;

      const initialMessage = {
        room_id: roomId,
        client_id: clientId,
      };
      ws.send(JSON.stringify(initialMessage));
    };

    ws.onclose = (event) => {
      console.log("Disconnected", event);
      setConnected(false);
    };

    ws.onerror = (error) => {
      console.error("Error:", error);
      setErrorMsg("Connection error");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("Received message:", data);

        if (data.type === "system" && data.message.includes("Welcome")) {
          setConnected(true);
        }

        if (data.type === "answer") {
          const newMessage = {
            content: data.data.answer,
            sender: "assistant",
            senderType: "assistant",
            chatId: id,
            createdAt: new Date().toISOString(),
            sources: data.data.sources,
          };

          setMessages((prev) => [...prev, newMessage]);

          saveMessage(newMessage);
        }
      } catch (err) {
        setErrorMsg("Failed to parse message");
      }
    };

    return () => {
      socketRef.current.close();
    };
  }, [$isAuth, $user, id]);

  useEffect(() => {
    const fetchMessages = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/chats/${id}/messages`);

        if (!response.ok) {
          throw new Error("Failed to fetch messages");
        }

        const data = await response.json();
        setMessages(data.messages);
        setChat(data.chat);
      } catch (error) {
        setErrorMsg(error.message);
      } finally {
        setLoading(false);
      }
    };

    if (!$isAuth) return;
    fetchMessages();
  }, [$isAuth]);

  const saveMessage = async (message) => {
    try {
      const response = await fetch(`${API_URL}/api/chats/${id}/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [message],
        }),
      });

      if (!response.ok) {
        console.error("Failed to save message");
      }
    } catch (error) {
      setErrorMsg("Failed to save message to database");
    }
  };

  const handleSendMessage = (content) => {
    if (!connected) {
      setErrorMsg("WebSocket not connected");
      return;
    }

    const newMessage = {
      content: content,
      sender: $user?.id,
      senderType: "user",
      chat: id,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, newMessage]);

    saveMessage(newMessage);

    socketRef.current.send(
      JSON.stringify({
        query: content,
      })
    );
  };

  return (
    <div className="flex-1 flex flex-col">
      <header className="p-4 border-b border-gray-200 bg-white">
        <h1 className="text-xl font-bold">{chat?.title}</h1>
      </header>

      <div
        className="flex flex-col h-full overflow-y-auto p-4 w-full gap-y-4"
        ref={messageContainerRef}
      >
        {messages.length == 0 ? (
          <div className="text-center py-4 text-gray-500">
            Start a new conversation
          </div>
        ) : (
          messages.map((message, index) =>
            message.senderType === "user" ? (
              <Message key={index} content={message.content} />
            ) : (
              <Response key={index} content={message.content} />
            )
          )
        )}
      </div>

      <ChatInput handleSendMessage={handleSendMessage} />
    </div>
  );
}
