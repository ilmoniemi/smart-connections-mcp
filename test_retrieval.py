"""Regression tests for index loading and result retrieval.

Run with the venv's python:

    .venv/bin/python -m unittest test_retrieval

Every test builds a synthetic vault in a temporary folder. The ONNX model is
never loaded: the encoder is replaced with a fake that returns a fixed vector.
"""
import json
import os
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


def rows(n):
    """n vectors, each less similar to the all-ones query than the one before."""
    m = np.ones((n, DIMS), dtype='<f4')
    for i in range(n):
        m[i, :i * DIMS // 8] = 0
    return m


def change_after_indexing(path, text):
    """Rewrite a note and move its mtime past the one the index recorded."""
    old = path.stat().st_mtime
    path.write_text(text)
    os.utime(path, (old + 2, old + 2))


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

    def test_passage_of_a_changed_note_is_dropped(self):
        source = dict(self.entry('note.md', [1.0] * DIMS),
                      last_import={'mtime': self.root.joinpath('note.md').stat().st_mtime_ns // 1_000_000})
        block = {'embeddings': {DEFAULT_MODEL: {'vec': [1.0] * DIMS}}, 'lines': [2, 2]}
        self.write_index({'smart_sources:note.md': source, 'smart_blocks:note.md#Title': block})
        self.assertEqual([b['text'] for b in self.db.get_context_blocks('passage')], ['Relevant passage\n'])
        change_after_indexing(self.root / 'note.md', 'Entirely changed content\n')
        self.assertEqual(self.db.get_context_blocks('passage'), [])


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
        src = np.concatenate([rows(source_rows).ravel(), np.ones(extra_floats, dtype='<f4')])
        src.tofile(self.env / 'smart_sources' / 'mf_x')
        rows(block_rows).tofile(self.env / 'smart_blocks' / 'mf_x')

    def test_partial_trailing_row_keeps_complete_rows(self):
        (self.root / 'other.md').write_text('Other\n')
        self.write_index({
            'smart_sources:note.md': self.source('note.md', 0),
            'smart_sources:other.md': self.source('other.md', 1),
        }, source_rows=2, extra_floats=DIMS // 2)  # the plugin is mid-way through a third row
        self.db.load_embeddings()
        self.assertEqual(sorted(self.db.embeddings_cache),
                         ['smart_sources:note.md', 'smart_sources:other.md'])

    def test_paths_outside_the_vault_are_not_read(self):
        outside = self.root.parent / 'outside.md'
        outside.write_text('Outside secret\nOutside secret\n')
        (self.root / 'link.md').symlink_to(outside)
        block = {'key': '../outside.md#Top', 'lines': [1, 2], 'embedding': self.ref(0)}
        entries = {
            'smart_sources:note.md': self.source('note.md', 0),
            'smart_sources:../outside.md': dict(self.source('../outside.md', 0), blocks_data={'#Top': block}),
            'smart_sources:link.md': self.source('link.md', 0),
            'smart_sources:' + str(outside): self.source(str(outside), 0),
        }
        self.write_index(entries)
        hits = self.db.semantic_search('passage', limit=10) + self.db.get_context_blocks('passage')
        self.assertTrue(any('Relevant passage' in r.get('text_preview', '') for r in hits))
        for r in hits:
            self.assertNotIn('Outside secret', r.get('text_preview', '') + r.get('text', ''))

    def test_deleted_note_is_skipped_and_limit_still_filled(self):
        for name in ('other.md', 'third.md'):
            (self.root / name).write_text(name + '\n')
        self.write_index({
            'smart_sources:note.md': self.source('note.md', 0),
            'smart_sources:other.md': self.source('other.md', 1),
            'smart_sources:third.md': self.source('third.md', 2),
        }, source_rows=3)
        self.db.load_embeddings()
        self.note.unlink()
        self.assertEqual([r['path'] for r in self.db.semantic_search('passage', limit=1)], ['other.md'])
        self.assertEqual([r['path'] for r in self.db.find_related('third.md', limit=1)], ['other.md'])

    def test_changed_note_drops_its_passages_and_flags_the_note(self):
        (self.root / 'other.md').write_text('Other\nOther passage\n')
        blocks = {'#Title': {'key': 'note.md#Title', 'lines': [2, 2], 'embedding': self.ref(0)}}
        other_blocks = {'#Other': {'key': 'other.md#Other', 'lines': [2, 2], 'embedding': self.ref(1)}}
        self.write_index({
            'smart_sources:note.md': self.source('note.md', 0, blocks),
            'smart_sources:other.md': self.source('other.md', 1, other_blocks),
        }, source_rows=2, block_rows=2)
        fresh = self.db.semantic_search('passage', limit=10)
        self.assertEqual(len(fresh), 4)
        self.assertFalse(any(r['index_stale'] for r in fresh))

        change_after_indexing(self.note, 'Entirely changed content\n')
        blocks = self.db.get_context_blocks('passage', max_blocks=1)
        self.assertEqual([(b['key'], b['text']) for b in blocks],
                         [('smart_blocks:other.md#Other', 'Other passage\n')])
        hits = {r['key']: r for r in self.db.semantic_search('passage', limit=10)}
        self.assertNotIn('smart_blocks:note.md#Title', hits)
        self.assertTrue(hits['smart_sources:note.md']['index_stale'])
        self.assertFalse(hits['smart_sources:other.md']['index_stale'])


if __name__ == '__main__':
    unittest.main()
