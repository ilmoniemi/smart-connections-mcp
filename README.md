# Smart Connections MCP Server

Exposes the vector index of the Obsidian [Smart Connections](https://smartconnections.app/) plugin to Claude Code (and other MCP clients), so an assistant can search a vault by meaning.

**About this fork.** The upstream repository has not been updated since October 2025. It reads only the old `.smart-env/multi/` layout, which Smart Connections stopped updating in version 4.7, so upstream keeps answering from a frozen index without any error. This fork reads the current layout and starts fast enough for Claude's fixed 30-second connect limit. The "Upstream documentation" part further down describes the original version and is partly out of date: it used sentence-transformers and torch, and it read `multi/`.

## What this fork changes

- **Smart Environment v3 layout (Smart Connections 4.7 and later).** Reads `smart_sources/*.ajson` and the binary vector files in `smart_sources/` and `smart_blocks/`. Falls back to `multi/` for older plugin versions.
- **Thin block entries (Smart Connections 4.3 and later).** The plugin no longer stores a section's path or text, so the path comes from the entry key and the text is read from the note on disk.
- **No torch.** Queries are embedded with ONNX Runtime and the `tokenizers` library. The server connects in under 2 seconds and the virtual environment stays small.
- **`get_stats` tool.** Shows which layout is in use, vector counts, the model, and whether the index on disk changed since it was loaded. The index reloads by itself when the plugin rewrites it.
- **Notes without a vector are skipped** (the plugin has not embedded them yet) instead of crashing the server.

## Setup

You need Python 3.12 and Obsidian with Smart Connections installed and indexed, using the plugin's default local model `TaylorAI/bge-micro-v2`. The server embeds your queries with the same model, so an index built with another model will not work.

macOS or Linux:

```bash
git clone https://github.com/ilmoniemi/smart-connections-mcp.git ~/tools/smart-connections-mcp
cd ~/tools/smart-connections-mcp
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\python`.

`requirements.txt` keeps the MCP Python SDK below version 2, because version 2 removed the server API this code uses.

The first start downloads the ONNX model (about 70 MB) into `models/` next to the server. Do it once by hand, so the first Claude session does not wait for the download:

```bash
OBSIDIAN_VAULT_PATH="/path/to/your/vault" .venv/bin/python test_handshake.py "a test query"
```

It prints how long the handshake takes, the index layout and vector count, and the top three results for the query.

### Configure Claude Code

Add the server to `.mcp.json` in the vault folder (or to the user-level `~/.claude.json` to make it available everywhere). Use absolute paths:

```json
{
  "mcpServers": {
    "smart-connections": {
      "command": "/Users/you/tools/smart-connections-mcp/.venv/bin/python",
      "args": ["/Users/you/tools/smart-connections-mcp/server.py"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/Users/you/obsidian/Your Vault"
      }
    }
  }
}
```

**Windows:** the `env` block replaces the whole process environment. Also pass `PATH`, `SystemRoot`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `TEMP` and `TMP`, or the server fails at startup with `WinError 10106`.

Start a new Claude Code session and approve the server. The tools are `semantic_search`, `find_related`, `get_context_blocks` and `get_stats`.

### Check that it works

Call `get_stats`. `layout` should be `smart-environment-v3` and `vectors_total` should be in the thousands for a vault of any size. Then search for something you wrote today. If recent notes are missing, check `index_changed_on_disk` and that Obsidian is running, since the plugin only indexes while Obsidian is open.

Run the same check after every Smart Connections plugin update. The plugin changed its storage format twice in 2026, and both times the symptom was old or empty results rather than an error.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `layout` is `legacy-multi` on a current plugin | The v3 files were not found under `.smart-env/smart_sources/`. Check `OBSIDIAN_VAULT_PATH` |
| Results have empty paths or previews | The plugin moved to a storage format this server does not know yet |
| `connection timed out after 30000ms` | Something slow is imported at startup. `python -c "import server"` from the venv should take about 2 seconds |
| First search after startup takes a few seconds | The index loads in the background after the handshake. Later searches are fast |

Native libraries are imported before the MCP transport starts on purpose: importing them afterwards deadlocks on Windows. See the docstring at the top of `server.py`.

---

# Upstream documentation

Exposes your Obsidian Smart Connections vector database to Claude Code via Model Context Protocol (MCP).

## What This Does

Instead of using text-based `Grep`, Claude Code can now perform **semantic search** across your vault:

- **semantic_search**: Find notes by meaning, not keywords
- **find_related**: Get related notes (like Smart Connections sidebar)
- **get_context_blocks**: Get best context for RAG queries

## Architecture

```
Smart Connections Plugin
    ↓ (creates)
.smart-env/multi/*.ajson
    ↓ (reads)
This MCP Server
    ↓ (exposes via)
MCP Protocol
    ↓ (consumed by)
Claude Code
```

## Installation

### Quick Install (Recommended)

```bash
cd ~/smart-connections-mcp
./install.sh
```

The script will:
- ✅ Install UV package manager (if needed)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Auto-detect your Obsidian vault
- ✅ Configure `~/.mcp.json`
- ✅ Verify installation

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

#### 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create Virtual Environment and Install Dependencies

```bash
cd ~/smart-connections-mcp
uv venv
uv pip install -r requirements.txt
```

**Important dependencies:**
- `mcp>=1.0.0` - Official Model Context Protocol SDK
- `sentence-transformers>=2.2.0` - For semantic search
- `numpy<2.0.0` - Version 1.x required (2.x breaks compatibility)
- `torch>=2.0.0` and `transformers>=4.30.0` - ML dependencies

#### 3. Configure Claude Code

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "smart-connections": {
      "command": "/Users/YOUR_USERNAME/smart-connections-mcp/.venv/bin/python",
      "args": ["/Users/YOUR_USERNAME/smart-connections-mcp/server.py"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/your/obsidian/vault"
      }
    }
  }
}
```

**Note:** Use the virtual environment Python, not system Python!

#### 4. Verify Installation

```bash
claude mcp list
```

Expected output:
```
smart-connections: .venv/bin/python server.py - ✓ Connected
```

</details>

### Migration to New Machine

**See [DEPLOYMENT.md](DEPLOYMENT.md)** for detailed migration guide.

Quick migration:
```bash
# On new machine
git clone https://github.com/dan6684/smart-connections-mcp.git ~/smart-connections-mcp
cd ~/smart-connections-mcp
./install.sh
```

**Important:** Keep this MCP server in a **separate repository** from your Obsidian vault. See [DEPLOYMENT.md](DEPLOYMENT.md) for rationale and best practices.

### Troubleshooting

If you see timeout issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Usage Examples

### Semantic Search

**Old way (Grep):**
```
Grep pattern: "self-compassion"
→ Only finds notes with exact word "self-compassion"
```

**New way (Semantic Search):**
```
semantic_search(query: "recognizing self-worth and releasing shame")
→ Finds: Ann Shulgin note ("I am a treasure")
        BM playa note ("I am beautiful, playa saved me")
        Therapy notes (related concepts)
```

### Find Related Notes

**Like Smart Connections sidebar:**
```
find_related(file_path: "DailyNotes/2025-10-25.md")
→ Returns top 10 semantically similar notes
```

### Get Context for RAG

**Build context for complex queries:**
```
get_context_blocks(query: "transformation through embodiment")
→ Returns actual text blocks most relevant to query
→ Claude can use these for grounded answers
```

## How It Works

1. **Reads existing embeddings** from `.smart-env/multi/*.ajson`
2. **No re-indexing needed** - uses Smart Connections' work
3. **Same model** (BGE-micro-v2) for query encoding
4. **Cosine similarity** to rank results
5. **Returns JSON** with file paths, similarity scores, metadata

## Tools Provided

### `semantic_search`
```python
semantic_search(
    query: str,           # Natural language query
    limit: int = 10,      # Max results
    min_similarity: float = 0.3  # Threshold
)
```

Returns:
```json
{
  "query": "self-compassion",
  "results_count": 5,
  "results": [
    {
      "path": "DailyNotes/2025-08-29.md",
      "similarity": 0.87,
      "key": "smart_sources:DailyNotes/2025-08-29.md",
      "metadata": {"tags": ["#Dream", "#grateful"]}
    }
  ]
}
```

### `find_related`
```python
find_related(
    file_path: str,      # e.g., "DailyNotes/2025-10-25.md"
    limit: int = 10
)
```

### `get_context_blocks`
```python
get_context_blocks(
    query: str,
    max_blocks: int = 5
)
```

Returns actual text content (not just paths) for RAG.

## Performance

- **Initial load:** ~2-3 seconds (loads 3,249 embeddings)
- **Query time:** ~100-200ms (cosine similarity across all embeddings)
- **Memory:** ~50MB (cached embeddings)

## Troubleshooting

**See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed debugging guide.**

### Common Issues

#### Server Timeout on `claude mcp list`
**Symptoms:** Connection hangs, no response after 30+ seconds

**Fixes:**
1. Ensure using virtual environment Python (not system Python)
2. Verify NumPy version is <2.0.0: `uv pip list | grep numpy`
3. Check server starts manually:
   ```bash
   OBSIDIAN_VAULT_PATH="/path/to/vault" .venv/bin/python server.py
   ```

#### Import Errors
**Error:** `ImportError: numpy.core.multiarray failed to import`

**Fix:** Reinstall with NumPy 1.x:
```bash
uv pip install "numpy<2.0.0" --force-reinstall
```

#### No Results Returned
- Check `.smart-env/multi/` has .ajson files
- Verify Smart Connections is enabled in Obsidian
- Lower `min_similarity` threshold (try 0.2 instead of 0.3)

#### Wrong Results
- Smart Connections may need to re-index
- Check embedding model matches (BGE-micro-v2)
- Restart server to reload embeddings

## Development

**Update embeddings:**
- Smart Connections auto-updates `.smart-env/`
- MCP server reads on startup (restart to refresh)
- Future: Add file watcher for auto-reload

**Add new tools:**
Edit `handle_request()` in `server.py`

## License

MIT - Use freely for personal PKM workflows
