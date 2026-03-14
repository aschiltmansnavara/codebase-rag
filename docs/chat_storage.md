# Chat History Storage

Chat histories are persisted in SQLite. Messages are automatically saved and restored between sessions.

## Configuration

```
CHAT_STORAGE_PATH=./data/chat_history.db
```

No separate service required. The database file is created automatically.

## Implementation

- **`SqliteChatStorage`** (`database/sqlite_storage.py`): direct SQLite interaction
- **`ChatHistoryManager`** (`database/chat_storage.py`): facade for saving/loading chats
- **`Components`** module: integrates persistence with the Streamlit UI

## Database Schema

Single `chats` table:

| Column | Type | Description |
|--------|------|-------------|
| chat_id | TEXT (PK) | Unique identifier for the chat |
| messages | TEXT (JSON) | JSON-serialized list of messages |
| title | TEXT | Chat title |
| user_id | TEXT | User identifier |
| start_time | TEXT | ISO timestamp of chat creation |
| last_updated | TEXT | ISO timestamp of last update |
| message_count | INTEGER | Number of messages in the chat |

## Troubleshooting

1. Check that the data directory exists and is writable
2. Check logs: `streamlit run src/codebase_rag/app/main.py --log_level=DEBUG`
3. Inspect directly: `sqlite3 ./data/chat_history.db ".tables"`
