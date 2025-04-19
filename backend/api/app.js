const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

mongoose
  .connect(process.env.MONGODB_URI || "mongodb://localhost:27017/gt-toolkit")
  .then(() => console.log("MongoDB connected successfully"))
  .catch((err) => console.error("MongoDB connection error:", err));

const messageSchema = new mongoose.Schema({
  content: {
    type: String,
    required: true,
  },
  sender: {
    type: String,
    ref: "User",
    required: true,
  },
  senderType: {
    type: String,
    enum: ["user", "assistant"],
    required: true,
  },
  chat: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Chat",
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

messageSchema.index({ chat: 1, createdAt: 1 });
messageSchema.index({ sender: 1 });
messageSchema.index({ senderType: 1, chat: 1 });

const chatSchema = new mongoose.Schema({
  title: {
    type: String,
    default: "New Conversation",
  },
  owner: {
    type: String,
    ref: "User",
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

chatSchema.index({ owner: 1 });
chatSchema.index({ updatedAt: -1 });
chatSchema.index({ title: "text" });

const userSchema = new mongoose.Schema(
  {
    _id: {
      type: String,
    },
    email: {
      type: String,
      required: true,
      lowercase: true,
      unique: true,
    },
    name: {
      type: String,
      required: true,
    },
    createdAt: {
      type: Date,
      default: () => Date.now(),
    },
    updatedAt: {
      type: Date,
      default: () => Date.now(),
    },
  },
  {
    _id: false,
  }
);

userSchema.index({ name: 1 });
chatSchema.index({ owner: 1, createdAt: -1 });

const Message = mongoose.model("Message", messageSchema);
const Chat = mongoose.model("Chat", chatSchema);
const User = mongoose.model("User", userSchema);

app.post("/api/users", async (req, res) => {
  try {
    const { email, name, userId } = req.body;

    if (!userId) {
      return res.status(400).json({
        error: "Firebase user ID (userId) is required",
      });
    }

    const existingUser = await User.findOne({
      $or: [{ _id: userId }, { email }],
    });

    if (existingUser) {
      return res.status(409).json({
        error: "User already exists",
        user: existingUser,
      });
    }

    const user = new User({
      _id: userId,
      email,
      name,
    });

    await user.save();

    res.status(201).json(user);
  } catch (error) {
    console.error("Error creating user:", error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/chats/:chatId/messages", async (req, res) => {
  try {
    const chatId = req.params.chatId;
    const { messages } = req.body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: "Request body must contain an array of messages",
      });
    }

    const chat = await Chat.findById(chatId);
    if (!chat) {
      return res.status(404).json({ error: "Chat not found" });
    }

    for (const message of messages) {
      if (!message.content || !message.sender || !message.senderType) {
        return res.status(400).json({
          error: "Each message must have content, sender, and senderType",
        });
      }
    }

    const messagesToInsert = messages.map((message) => ({
      content: message.content,
      sender: message.sender,
      senderType: message.senderType,
      chat: chatId,
      createdAt: message.createdAt || new Date(),
      updatedAt: message.updatedAt || new Date(),
    }));

    const savedMessages = await Message.insertMany(messagesToInsert);

    await Chat.findByIdAndUpdate(chatId, { updatedAt: new Date() });

    res.status(201).json({
      message: `Successfully created ${savedMessages.length} messages in chat ${chatId}`,
      messages: savedMessages,
    });
  } catch (error) {
    console.error("Error creating messages in batch:", error);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/chats/:chatId/messages", async (req, res) => {
  try {
    const chatId = req.params.chatId;

    const chat = await Chat.findById(chatId);
    if (!chat) {
      return res.status(404).json({ error: "Chat not found" });
    }

    const sort = req.query.sort || "asc";
    const sortDirection = sort.toLowerCase() === "desc" ? -1 : 1;

    const messages = await Message.find({ chat: chatId })
      .sort({ createdAt: sortDirection })
      .populate("sender", "name email");

    res.json({
      chat: {
        id: chat._id,
        title: chat.title,
        owner: chat.owner,
      },
      messages,
    });
  } catch (error) {
    console.error("Error fetching messages:", error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/users/:userId/chats", async (req, res) => {
  try {
    const { userId } = req.params;
    const { title } = req.body;

    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const chat = new Chat({
      title: title || "New Conversation",
      owner: userId,
      createdAt: new Date(),
      updatedAt: new Date(),
    });

    await chat.save();

    res.status(201).json({
      message: "Chat created successfully",
      chat: {
        id: chat._id,
        title: chat.title,
        owner: chat.owner,
        createdAt: chat.createdAt,
        updatedAt: chat.updatedAt,
      },
    });
  } catch (error) {
    console.error("Error creating chat:", error);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/users/:userId/chats", async (req, res) => {
  try {
    const { userId } = req.params;

    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const sort = req.query.sort || "recent";
    const search = req.query.search || "";

    let sortOptions = {};
    if (sort === "recent") {
      sortOptions = { updatedAt: -1 };
    } else if (sort === "title") {
      sortOptions = { title: 1 };
    } else if (sort === "created") {
      sortOptions = { createdAt: -1 };
    }

    let query = { owner: userId };

    if (search) {
      query.title = { $regex: search, $options: "i" };
    }

    const chats = await Chat.find(query).sort(sortOptions).lean();

    const formattedChats = chats.map((chat) => ({
      id: chat._id,
      title: chat.title,
      owner: chat.owner,
      createdAt: chat.createdAt,
      updatedAt: chat.updatedAt,
    }));

    res.json({
      userId,
      chats: formattedChats,
    });
  } catch (error) {
    console.error("Error fetching user's chats:", error);
    res.status(500).json({ error: error.message });
  }
});

app.delete("/api/users/:userId/chats/:chatId", async (req, res) => {
  try {
    const { userId, chatId } = req.params;

    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const chat = await Chat.findOne({ _id: chatId, owner: userId });
    if (!chat) {
      return res.status(404).json({
        error: "Chat not found or you don't have permission to delete it",
      });
    }

    await Message.deleteMany({ chat: chatId });

    await Chat.findByIdAndDelete(chatId);

    res.json({
      message: "Chat and all its messages deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting chat:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
