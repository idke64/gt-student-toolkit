# GT Student Toolkit

## Overview

There are three main components to the app:

1. **LLM WebSocket Server** - Handles real-time AI interactions and natural language processing
2. **Express Backend** - Manages API endpoints, data storage, and business logic
3. **Astro Frontend** - Provides a fast, responsive user interface for seamless interaction

## Setup

```bash
git clone https://github.com/yourusername/gt-student-toolkit.git
cd gt-student-toolkit
```

### LLM WebSocket Server
The WebSocket server handles real-time LLM interactions.

```bash
cd backend/ai
pip install -r requirements.txt
python server.py
```

### Express Backend Server
The Express server provides API endpoints for the application.

```bash
cd backend/api
node app.js
```

### Astro Frontend
The frontend is built with Astro for fast, modern web experiences.

```bash
cd frontend
npm run dev
```

## Environment Variables

Create a `.env` file in `backend/api` and `frontend` based on `.env.example`