#!/usr/bin/env python3
"""
Smart Connections MCP Server
Exposes the Smart Connections vector database of an Obsidian vault to Claude
Code via the MCP protocol.

Two .smart-env layouts are supported:

  * Smart Environment v3 (Smart Connections 4.7 and later). Source metadata
    is an append log of JSON lines in smart_sources/smart_sources*.ajson,
    replayed in numeric shard order (last occurrence wins). Vectors live in
    binary float32 files (mf_*) in smart_sources/ (one row per note) and
    smart_blocks/ (one row per block). Each entry points to its row through
    embedding.default.<mf name>.file_i. Block metadata is nested inside the
    source entry under blocks_data.

  * Legacy (Smart Connections 2.x / 3.x). One .ajson per note under multi/
    with vectors stored inline.

The v3 layout is preferred; multi/ is only read when v3 yields nothing.

Startup: torch + transformers take about 8 to 10 seconds to import, and the
model and index are loaded before the stdio transport starts. This is
deliberate. Importing heavy native libraries after the transport is listening
deadlocks on Windows: scipy's bundled OpenBLAS waited on a C-runtime lock that
the stdin reader thread held while blocked in a read. Verified 2026-09-17,
both in a background thread and in the main thread.

sentence-transformers was dropped on 2026-09-21: its import alone took 27.5 s
(scikit-learn, scipy, training utilities) against the desktop app's fixed
30 s connect timeout, which the "timeout" field in .mcp.json does not extend.
The model (bge-micro-v2) uses mean pooling + L2 normalisation, so the direct
transformers path below reproduces SentenceTransformer.encode() exactly
(cosine 1.0 on test queries).
"""

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch  # slow imports; must happen before stdio starts, see module docstring
from transformers import AutoModel, AutoTokenizer

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger("smart-connections-mcp")

DEFAULT_MODEL = 'TaylorAI/bge-micro-v2'
DEFAULT_DIMS = 384

# Matches one line of an .ajson append log:  "key": <json>,
_AJSON_LINE = re.compile(r'^\s*"((?:[^"\\]|\\.)*)"\s*:\s*(.*?),?\s*$')


class SmartConnectionsDatabase:
    """Read-only interface to a vault's .smart-env vector database."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.smart_env_path = self.vault_path / ".smart-env"
        self.multi_path = self.smart_env_path / "multi"
        self.sources_path = self.smart_env_path / "smart_sources"
        self.blocks_path = self.smart_env_path / "smart_blocks"

        self.model = None
        self.tokenizer = None
        self.model_name, self.dims = self._read_model_config()

        # key -> {'path','vector','text','key','lines','metadata'}
        self.embeddings_cache: Dict[str, Dict] = {}
        self._keys: List[str] = []
        self._matrix: Optional[np.ndarray] = None   # normalized, (N, dims)
        self._is_block: Optional[np.ndarray] = None
        self._loaded_signature = None
        self.layout: Optional[str] = None
        self.last_load_seconds: float = 0.0
        self.last_load_at: Optional[float] = None
        self.warm_up_error: Optional[str] = None

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _read_model_config(self) -> Tuple[str, int]:
        """Model key from smart_env.json, dims from embedding_models.ajson."""
        model_name = DEFAULT_MODEL
        dims = DEFAULT_DIMS
        try:
            cfg = json.loads((self.smart_env_path / "smart_env.json").read_text(encoding='utf-8'))
            em = cfg.get('smart_sources', {}).get('embed_model', {})
            adapter = em.get('adapter')
            key = (em.get(adapter) or {}).get('model_key') if adapter else None
            if key:
                model_name = key
        except Exception:
            pass
        try:
            models_file = self.smart_env_path / "embedding_models" / "embedding_models.ajson"
            for _, entry in self._iter_ajson(models_file):
                if isinstance(entry, dict) and entry.get('model_key') == model_name and entry.get('dims'):
                    dims = int(entry['dims'])
                    break
        except Exception:
            pass
        return model_name, dims

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def ensure_model_loaded(self):
        """Load the embedding model once."""
        with self._lock:
            if self.model is None:
                t0 = time.time()
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).eval()
                logger.info("Model %s loaded in %.1fs", self.model_name, time.time() - t0)

    def warm_up(self):
        """Load the model and run one throwaway encode so every native kernel
        torch needs is initialised. Runs before the stdio transport starts.
        The index itself is plain file IO + numpy and is loaded afterwards
        (see main), so it does not count against the connect timeout."""
        try:
            self.ensure_model_loaded()
            self._encode("warm-up")
        except Exception as e:  # keep the server alive, report on first call
            self.warm_up_error = f"{type(e).__name__}: {e}"
            logger.exception("Warm-up failed")

    def load_index_in_background(self):
        """Called once the transport is listening. Safe on Windows because no
        new native modules are imported here; a concurrent query simply waits
        on self._lock until the load finishes."""
        try:
            t0 = time.time()
            self.load_embeddings()
            logger.info("Index ready in %.1fs", time.time() - t0)
        except Exception as e:
            self.warm_up_error = f"{type(e).__name__}: {e}"
            logger.exception("Index load failed")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_ajson(path: Path):
        """Yield (key, value) per line. value is None for deletions / bad JSON."""
        if not path.exists():
            return
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                m = _AJSON_LINE.match(line)
                if not m:
                    continue
                key, raw = m.group(1), m.group(2)
                if raw == 'null':
                    yield key, None
                    continue
                try:
                    yield key, json.loads(raw)
                except Exception:
                    yield key, None

    def _shard_files(self) -> List[Path]:
        """smart_sources.ajson first, then _1, _2, ... in numeric order."""
        files = list(self.sources_path.glob("smart_sources*.ajson"))

        def order(p: Path):
            m = re.match(r'smart_sources(?:_(\d+))?\.ajson$', p.name)
            return int(m.group(1)) if (m and m.group(1)) else 0
        return sorted(files, key=order)

    def _signature(self):
        """Cheap fingerprint of the on-disk index, used to detect changes."""
        sig = []
        for d in (self.sources_path, self.blocks_path):
            if d.exists():
                for p in sorted(d.iterdir()):
                    try:
                        st = p.stat()
                        sig.append((str(p), st.st_mtime_ns, st.st_size))
                    except OSError:
                        pass
        if not sig and self.multi_path.exists():
            try:
                st = self.multi_path.stat()
                sig.append((str(self.multi_path), st.st_mtime_ns, len(os.listdir(self.multi_path))))
            except OSError:
                pass
        return tuple(sig)

    @staticmethod
    def _mf_ref(embedding) -> Optional[dict]:
        """Return the {'file','file_i'} reference from an entry's embedding."""
        default = (embedding or {}).get('default') or {}
        for v in default.values():
            if isinstance(v, dict) and 'file_i' in v:
                return v
        return None

    def _load_v3(self) -> Dict[str, Dict]:
        shards = self._shard_files()
        if not shards:
            return {}

        state: Dict[str, dict] = {}
        for shard in shards:
            for key, val in self._iter_ajson(shard):
                if val is None:
                    state.pop(key, None)
                elif isinstance(val, dict):
                    state[key] = val

        mf_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

        def get_mf(directory: Path, name: str) -> Optional[np.ndarray]:
            ck = (str(directory), name)
            if ck not in mf_cache:
                p = directory / name
                mat = None
                if name and p.exists():
                    raw = np.fromfile(p, dtype=np.float32)
                    if raw.size % self.dims == 0:
                        mat = raw.reshape(-1, self.dims)
                    else:
                        logger.warning("Vector file %s size %d not divisible by dims %d", p, raw.size, self.dims)
                mf_cache[ck] = mat
            return mf_cache[ck]

        cache: Dict[str, Dict] = {}
        for key, entry in state.items():
            if not key.startswith('smart_sources:'):
                continue
            path = entry.get('path') or key.split(':', 1)[1]

            ref = self._mf_ref(entry.get('embedding'))
            if ref:
                mat = get_mf(self.sources_path, ref.get('file', ''))
                i = ref.get('file_i')
                if mat is not None and isinstance(i, int) and 0 <= i < len(mat):
                    cache[key] = {
                        'path': path, 'vector': mat[i], 'text': '',
                        'key': key, 'lines': [], 'metadata': {},
                    }

            for sub, block in (entry.get('blocks_data') or {}).items():
                if not isinstance(block, dict):
                    continue
                ref = self._mf_ref(block.get('embedding'))
                if not ref:
                    continue
                mat = get_mf(self.blocks_path, ref.get('file', ''))
                i = ref.get('file_i')
                if mat is None or not isinstance(i, int) or not (0 <= i < len(mat)):
                    continue
                bkey = 'smart_blocks:' + (block.get('key') or (path + sub))
                cache[bkey] = {
                    'path': path, 'vector': mat[i], 'text': '',
                    'key': bkey, 'lines': block.get('lines') or [], 'metadata': {},
                }
        return cache

    @staticmethod
    def _resolve_path_from_key(key: str) -> Optional[str]:
        if not key or ':' not in key:
            return None
        return key.split(':', 1)[1].split('#', 1)[0]

    def _load_legacy(self) -> Dict[str, Dict]:
        cache: Dict[str, Dict] = {}
        if not self.multi_path.exists():
            return cache
        for ajson_file in self.multi_path.glob("*.ajson"):
            try:
                content = ajson_file.read_text(encoding='utf-8').strip()
                if content and not content.startswith('{'):
                    content = '{' + content.rstrip(',').strip() + '}'
                data = json.loads(content)
            except Exception:
                continue
            for key, item in data.items():
                if not isinstance(item, dict):
                    continue
                emb = (item.get('embeddings') or {}).get(self.model_name) or {}
                vec = emb.get('vec')
                if not vec or len(vec) != self.dims:
                    continue
                cache[key] = {
                    'path': item.get('path') or self._resolve_path_from_key(key),
                    'vector': np.asarray(vec, dtype=np.float32),
                    'text': item.get('text') or '',
                    'key': key,
                    'lines': item.get('lines', []),
                    'metadata': item.get('metadata', {}),
                }
        return cache

    def load_embeddings(self, force: bool = False):
        """(Re)load the index if it changed on disk since the last load."""
        with self._lock:
            sig = self._signature()
            if not force and self._matrix is not None and sig == self._loaded_signature:
                return
            t0 = time.time()
            cache = self._load_v3()
            layout = 'smart-environment-v3'
            if not cache:
                cache = self._load_legacy()
                layout = 'legacy-multi'

            keys = list(cache.keys())
            if keys:
                mat = np.stack([cache[k]['vector'] for k in keys]).astype(np.float32, copy=False)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                mat = mat / norms
                is_block = np.fromiter((k.startswith('smart_blocks:') for k in keys), dtype=bool, count=len(keys))
            else:
                mat = np.zeros((0, self.dims), dtype=np.float32)
                is_block = np.zeros(0, dtype=bool)

            self.embeddings_cache = cache
            self._keys = keys
            self._matrix = mat
            self._is_block = is_block
            self._loaded_signature = sig
            self.layout = layout
            self.last_load_seconds = time.time() - t0
            self.last_load_at = time.time()
            logger.info("Loaded %d vectors (%d blocks) from %s layout in %.1fs",
                        len(keys), int(is_block.sum()), layout, self.last_load_seconds)

    # ------------------------------------------------------------------
    # Text hydration (v3 stores no text; read the vault file on demand)
    # ------------------------------------------------------------------

    def _read_block_text(self, source_path: str, lines) -> str:
        if not source_path or not lines or len(lines) < 2:
            return ''
        try:
            file_path = self.vault_path / source_path
            if not file_path.exists():
                return ''
            start, end = int(lines[0]), int(lines[1])
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
            return ''.join(all_lines[max(0, start - 1):min(len(all_lines), end)])
        except Exception:
            return ''

    def _read_source_head(self, source_path: str, n: int = 400) -> str:
        try:
            file_path = self.vault_path / source_path
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read(n)
        except Exception:
            pass
        return ''

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _prepare(self):
        if self.warm_up_error:
            raise RuntimeError(f"Server warm-up failed: {self.warm_up_error}")
        self.ensure_model_loaded()
        self.load_embeddings()

    def _encode(self, text: str) -> np.ndarray:
        """Mean-pooled, L2-normalised embedding; equals SentenceTransformer.encode() for bge-micro-v2."""
        with torch.no_grad():
            batch = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        q = pooled[0].numpy().astype(np.float32)
        n = np.linalg.norm(q)
        return q / n if n else q

    def semantic_search(self, query: str, limit: int = 10, min_similarity: float = 0.3) -> List[Dict]:
        with self._lock:
            self._prepare()
            if self._matrix is None or len(self._keys) == 0:
                return []
            sims = self._matrix @ self._encode(query)
            order = np.argsort(-sims)
            results = []
            for idx in order:
                s = float(sims[idx])
                if s < min_similarity or len(results) >= limit:
                    break
                item = self.embeddings_cache[self._keys[idx]]
                results.append({
                    'key': item['key'],
                    'path': item['path'],
                    'similarity': s,
                    'lines': item.get('lines'),
                    'metadata': item.get('metadata', {}),
                    'text_preview': (item.get('text') or '')[:200],
                })
        for r in results:
            if r['text_preview']:
                continue
            if r['key'].startswith('smart_blocks:'):
                r['text_preview'] = self._read_block_text(r['path'], r.get('lines'))[:200]
            elif r['path']:
                r['text_preview'] = self._read_source_head(r['path'])[:200]
        return results

    def find_related(self, file_path: str, limit: int = 10) -> List[Dict]:
        with self._lock:
            if self.warm_up_error:
                raise RuntimeError(f"Server warm-up failed: {self.warm_up_error}")
            self.load_embeddings()
            target_key = f"smart_sources:{file_path}"
            if target_key not in self.embeddings_cache or self._matrix is None:
                return []
            ti = self._keys.index(target_key)
            sims = self._matrix @ self._matrix[ti]
            sims = np.where(self._is_block, -np.inf, sims)
            sims[ti] = -np.inf
            order = np.argsort(-sims)[:limit]
            out = []
            for idx in order:
                if not np.isfinite(sims[idx]):
                    break
                item = self.embeddings_cache[self._keys[idx]]
                out.append({
                    'key': item['key'],
                    'path': item['path'],
                    'similarity': float(sims[idx]),
                    'metadata': item.get('metadata', {}),
                })
            return out

    def get_context_blocks(self, query: str, max_blocks: int = 5, min_similarity: float = 0.4) -> List[Dict]:
        with self._lock:
            self._prepare()
            if self._matrix is None or len(self._keys) == 0:
                return []
            sims = self._matrix @ self._encode(query)
            sims = np.where(self._is_block, sims, -np.inf)
            order = np.argsort(-sims)
            results = []
            for idx in order:
                s = float(sims[idx])
                if not np.isfinite(s) or s <= min_similarity or len(results) >= max_blocks:
                    break
                item = self.embeddings_cache[self._keys[idx]]
                results.append({
                    'key': item['key'],
                    'path': item['path'],
                    'similarity': s,
                    'lines': item.get('lines'),
                    'text': item.get('text') or '',
                })
        for r in results:
            if not r['text']:
                r['text'] = self._read_block_text(r['path'], r.get('lines'))
        return results

    def stats(self) -> Dict:
        with self._lock:
            n = len(self._keys)
            blocks = int(self._is_block.sum()) if self._is_block is not None else 0
            return {
                'vault_path': str(self.vault_path),
                'layout': self.layout,
                'model': self.model_name,
                'dims': self.dims,
                'model_loaded': self.model is not None,
                'vectors_total': n,
                'vectors_sources': n - blocks,
                'vectors_blocks': blocks,
                'last_load_seconds': round(self.last_load_seconds, 2),
                'last_load_at': self.last_load_at,
                'index_changed_on_disk': self._signature() != self._loaded_signature,
                'warm_up_error': self.warm_up_error,
            }


# ----------------------------------------------------------------------
# MCP server
# ----------------------------------------------------------------------

TOOLS = [
    types.Tool(
        name="semantic_search",
        description="Search vault using semantic similarity (not keyword matching). Finds notes related to query meaning, not just exact words.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query describing what to search for"},
                "limit": {"type": "integer", "description": "Maximum number of results (default: 10)", "default": 10},
                "min_similarity": {"type": "number", "description": "Minimum similarity threshold 0-1 (default: 0.3)", "default": 0.3},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="find_related",
        description="Find notes related to a specific file path. Like Smart Connections sidebar in Obsidian.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path relative to vault root (e.g., 'DailyNotes/2025-10-25.md')"},
                "limit": {"type": "integer", "description": "Maximum number of results (default: 10)", "default": 10},
            },
            "required": ["file_path"],
        },
    ),
    types.Tool(
        name="get_context_blocks",
        description="Get best text blocks for a query (for RAG/context building). Returns actual text content.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query to find relevant context for"},
                "max_blocks": {"type": "integer", "description": "Maximum number of blocks (default: 5)", "default": 5},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="get_stats",
        description="Index status: storage layout in use, vector counts, model, whether the on-disk index changed since the last load.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting smart-connections-mcp server...")

    vault_path = os.getenv('OBSIDIAN_VAULT_PATH')
    if not vault_path:
        raise ValueError("OBSIDIAN_VAULT_PATH environment variable not set")
    logger.info("Vault path: %s", vault_path)

    db = SmartConnectionsDatabase(vault_path)
    logger.info("Model %s, dims %d", db.model_name, db.dims)

    server = Server("smart-connections-mcp")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return TOOLS

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        arguments = arguments or {}
        try:
            if name == "semantic_search":
                results = await asyncio.to_thread(
                    db.semantic_search, arguments['query'],
                    arguments.get('limit', 10), arguments.get('min_similarity', 0.3))
                payload = {"query": arguments['query'], "results_count": len(results), "results": results}
            elif name == "find_related":
                results = await asyncio.to_thread(
                    db.find_related, arguments['file_path'], arguments.get('limit', 10))
                payload = {"source_file": arguments['file_path'], "related_count": len(results), "related_files": results}
            elif name == "get_context_blocks":
                results = await asyncio.to_thread(
                    db.get_context_blocks, arguments['query'], arguments.get('max_blocks', 5))
                payload = {"query": arguments['query'], "blocks_count": len(results), "blocks": results}
            elif name == "get_stats":
                payload = await asyncio.to_thread(db.stats)
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            raise RuntimeError(f"Tool execution error: {e}")
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]

    # Load the model BEFORE the transport starts listening. Importing native
    # libraries afterwards, in any thread, deadlocks on Windows (see module
    # docstring). The index is only file IO and loads after the handshake.
    t0 = time.time()
    db.warm_up()
    logger.info("Warm-up done in %.1fs", time.time() - t0)

    logger.info("Starting stdio server...")
    async with stdio_server() as (read_stream, write_stream):
        index_task = asyncio.create_task(asyncio.to_thread(db.load_index_in_background))
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="smart-connections-mcp",
                server_version="0.2.0",
                capabilities=types.ServerCapabilities(tools=types.ToolsCapability()),
            ),
        )
        index_task.cancel()
        logger.info("MCP server finished")


if __name__ == "__main__":
    asyncio.run(main())
