"""Drive the server over real MCP stdio and time each step.

Usage (with the venv's python, which is also used to start the server):

    OBSIDIAN_VAULT_PATH=/path/to/vault .venv/bin/python test_handshake.py ["optional search query"]

Prints the time to initialize, list tools and get_stats (which waits for the
index), the index layout and vector count, and, given a query, the top three
semantic_search results. Running server.py bare in a terminal proves little:
a startup hang only shows with a client attached.
"""
import asyncio, json, os, sys, time
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    if not os.environ.get("OBSIDIAN_VAULT_PATH"):
        sys.exit("Set OBSIDIAN_VAULT_PATH to the vault folder first")
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("server.py"))],
        env=dict(os.environ),
    )
    t0 = time.time()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print(f"initialize done in {time.time()-t0:.1f}s", flush=True)
            tools = await s.list_tools()
            print(f"list_tools done in {time.time()-t0:.1f}s ({len(tools.tools)} tools)", flush=True)
            res = await s.call_tool("get_stats", {})
            stats = json.loads(res.content[0].text)
            print(f"get_stats done in {time.time()-t0:.1f}s: layout {stats.get('layout')}, "
                  f"{stats.get('vectors_total')} vectors", flush=True)
            if len(sys.argv) > 1:
                res = await s.call_tool("semantic_search", {"query": sys.argv[1], "limit": 3})
                data = json.loads(res.content[0].text)
                print(f"semantic_search done in {time.time()-t0:.1f}s, {data['results_count']} results", flush=True)
                for r in data["results"]:
                    print(f"  {r['similarity']:.3f}  {r['path']}", flush=True)

asyncio.run(main())
