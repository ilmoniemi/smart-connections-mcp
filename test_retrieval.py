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


class V3LayoutTests(unittest.TestCase):
    """Smart Environment v3: .ajson append log plus float32 vector files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'vault'
        self.env = self.root / '.smart-env'
        for d in ('smart_sources', 'smart_blocks'):
            (self.env / d).mkdir(parents=True)
        self.note = self.root / 'note.md'
        self.note.write_text('Title\nRelevant passage\nEnd\n')
        self.db = SmartConnectionsDatabase(str(self.root))
        fake_encoder(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def ref(i):
        return {'default': {'mf_x': {'file': 'mf_x', 'file_i': i}}}

    def source(self, path, row, blocks=None):
        return {
            'path': path,
            'embedding': self.ref(row),
            'last_import': {'mtime': (self.root / path).stat().st_mtime_ns // 1_000_000},
            'blocks_data': blocks or {},
        }

    def write_index(self, entries, source_rows=1, block_rows=1, extra_floats=0):
        lines = [f'{json.dumps(k)}: {json.dumps(v)},' for k, v in entries.items()]
        (self.env / 'smart_sources' / 'smart_sources.ajson').write_text('\n'.join(lines) + '\n')
        np.ones(source_rows * DIMS + extra_floats, dtype='<f4').tofile(self.env / 'smart_sources' / 'mf_x')
        np.ones(block_rows * DIMS, dtype='<f4').tofile(self.env / 'smart_blocks' / 'mf_x')

    def test_partial_trailing_row_keeps_complete_rows(self):
        (self.root / 'other.md').write_text('Other\n')
        self.write_index({
            'smart_sources:note.md': self.source('note.md', 0),
            'smart_sources:other.md': self.source('other.md', 1),
        }, source_rows=2, extra_floats=DIMS // 2)  # the plugin is mid-way through a third row
        self.db.load_embeddings()
        self.assertEqual(sorted(self.db.embeddings_cache),
                         ['smart_sources:note.md', 'smart_sources:other.md'])


if __name__ == '__main__':
    unittest.main()
