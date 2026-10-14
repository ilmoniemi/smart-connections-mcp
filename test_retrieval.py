"""Regression tests for index loading and result retrieval.

Run with the venv's python:

    .venv/bin/python -m unittest test_retrieval

Every test builds a synthetic vault in a temporary folder. The ONNX model is
never loaded: the encoder is replaced with a fake that returns a fixed vector.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from server import DEFAULT_DIMS, DEFAULT_MODEL, SmartConnectionsDatabase

DIMS = DEFAULT_DIMS


def fake_encoder(db):
    """Skip the model: mark it loaded and encode every query as all ones."""
    db.model = object()
    db._encode = lambda text: np.ones(DIMS, dtype=np.float32) / np.sqrt(DIMS)


class LegacyLayoutTests(unittest.TestCase):
    """Old layout: one .ajson per note under .smart-env/multi/, vectors inline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.multi = self.root / '.smart-env' / 'multi'
        self.multi.mkdir(parents=True)
        (self.root / 'note.md').write_text('Title\nRelevant passage\nEnd\n')
        self.db = SmartConnectionsDatabase(str(self.root))
        fake_encoder(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def write_index(self, entries, name='test.ajson'):
        (self.multi / name).write_text(json.dumps(entries)[1:-1] + ',')

    @staticmethod
    def entry(path, vec):
        return {'path': path, 'embeddings': {DEFAULT_MODEL: {'vec': vec}}}

    def test_non_numeric_vector_skips_only_that_entry(self):
        self.write_index({
            'smart_sources:note.md': self.entry('note.md', [1.0] * DIMS),
            'smart_sources:bad.md': self.entry('bad.md', ['x'] * DIMS),
        })
        self.db.load_embeddings()
        self.assertEqual(list(self.db.embeddings_cache), ['smart_sources:note.md'])

    def test_file_rewritten_in_place_is_reloaded(self):
        self.write_index({'smart_sources:note.md': self.entry('note.md', [1.0] * DIMS)})
        self.db.load_embeddings()
        self.assertEqual(len(self.db.embeddings_cache), 1)
        # Same file name, so the folder's own mtime and file count do not change.
        (self.multi / 'test.ajson').write_text('{}')
        self.db.load_embeddings()
        self.assertEqual(self.db.embeddings_cache, {})


if __name__ == '__main__':
    unittest.main()
