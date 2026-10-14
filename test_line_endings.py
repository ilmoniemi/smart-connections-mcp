"""Shell scripts must use LF line endings: with CRLF the kernel reads the
shebang as "/bin/bash\r" and fails with "bad interpreter".

    .venv/bin/python -m unittest test_line_endings
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class LineEndingTests(unittest.TestCase):
    def test_shell_scripts_have_no_carriage_returns(self):
        scripts = sorted(ROOT.glob('*.sh'))
        self.assertTrue(scripts)
        for script in scripts:
            self.assertNotIn(b'\r', script.read_bytes(), script.name)


if __name__ == '__main__':
    unittest.main()
