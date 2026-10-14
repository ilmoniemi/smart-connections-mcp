"""The --http transport must refuse requests that a web page could send.

    .venv/bin/python -m unittest test_http

Builds the HTTP app around a bare MCP server, so neither a vault nor the
model is needed.
"""
import unittest

from mcp.server import Server
from starlette.testclient import TestClient

from server import http_app

PORT = 37499
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "test", "version": "0"}},
}


class HttpSecurityTests(unittest.TestCase):
    def setUp(self):
        server = Server("test")

        @server.list_tools()
        async def list_tools():
            return []

        self.client = TestClient(http_app(server, PORT), base_url=f"http://127.0.0.1:{PORT}")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def initialize(self, **headers):
        headers = {"Accept": "application/json, text/event-stream", **headers}
        return self.client.post("/mcp", json=INITIALIZE, headers=headers)

    def test_local_requests_are_served(self):
        self.assertEqual(self.initialize().status_code, 200)
        self.assertEqual(self.initialize(Host=f"localhost:{PORT}",
                                         Origin=f"http://localhost:{PORT}").status_code, 200)
        self.assertEqual(self.initialize(Origin=f"http://127.0.0.1:{PORT}").status_code, 200)

    def test_foreign_host_is_refused(self):
        self.assertEqual(self.initialize(Host="evil.example").status_code, 421)
        self.assertEqual(self.initialize(Host="127.0.0.1:1234").status_code, 421)

    def test_web_page_origin_is_refused(self):
        response = self.initialize(Origin="https://evil.example")
        self.assertGreaterEqual(response.status_code, 400)
        self.assertLess(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
