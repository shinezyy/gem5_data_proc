import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import batch
from utils.common import xs_get_stats
from utils.derived_metrics import apply_derived_metrics
from simpoint_cpt.compute_weighted import compute_weighted_metrics


class RtlMissingStatsTest(unittest.TestCase):
    def parse(self, text):
        targets = {
            'committedInsts': r'commitInstr, (\d+)',
            'cycles': r'cycles, (\d+)',
            'inst_spec': r'inst_spec, (\d+)',
        }
        diagnostics = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'simulator_err.txt'
            path.write_text(text)
            stats = xs_get_stats(
                str(path), targets, required_keys={'committedInsts', 'cycles'},
                diagnostics=diagnostics,
            )
        return stats, list(diagnostics.values())

    def test_optional_missing_keeps_ipc_and_nan(self):
        stats, messages = self.parse('commitInstr, 20\ncommitInstr, 30\ncycles, 10\n')
        self.assertEqual(stats['ipc'], 3)
        self.assertTrue(math.isnan(stats['inst_spec']))
        df = apply_derived_metrics(pd.DataFrame([stats]), {'derived': 'inst_spec / cycles'})
        self.assertTrue(math.isnan(df.loc[0, 'derived']))
        self.assertIn('missing optional stats: inst_spec', messages[0])

    def test_missing_cycles_rejects_slice(self):
        stats, messages = self.parse('commitInstr, 20\ncommitInstr, 30\n')
        self.assertIsNone(stats)
        self.assertIn('missing required stats: cycles', messages[0])

    def test_single_dump_rejects_unfinished_slice(self):
        stats, messages = self.parse('commitInstr, 20\ncycles, 10\n')
        self.assertIsNone(stats)
        self.assertIn('expected 2', messages[0])

    def test_zero_cycles_rejects_slice(self):
        stats, messages = self.parse('commitInstr, 20\ncommitInstr, 30\ncycles, 0\n')
        self.assertIsNone(stats)
        self.assertIn('non-positive', messages[0])

    def test_weighting_rejects_empty_or_malformed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            for text, message in [
                ('', 'empty input CSV'),
                (',derived\n', 'no valid rows'),
                (',cycles\npoint1,10\n', 'missing required columns'),
            ]:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaisesRegex(SystemExit, message):
                        compute_weighted_metrics(str(path), 'unused.json', None,
                                                 SimpleNamespace(spec_version='06'))

    def test_batch_groups_warnings_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for point in (10, 20):
                log = root / f'bzip2_program_{point}' / 'simulator_err.txt'
                log.parent.mkdir()
                log.write_text(
                    '[PERF ][time=1] SimTop.rob: commitInstr, 20\n'
                    '[PERF ][time=2] SimTop.rob: commitInstr, 30\n'
                    '[PERF ][time=2] SimTop.rob: clock_cycle, 10\n'
                )
            output = root / 'output.csv'
            stdout = io.StringIO()
            argv = ['batch.py', '-s', str(root), '-X', '-g', 'basic,intel_topdown',
                    '--jobs', '1', '-o', str(output)]
            with patch('sys.argv', argv), \
                    redirect_stdout(stdout):
                batch.main()
            df = pd.read_csv(output)
            self.assertEqual(len(df), 2)
            self.assertTrue((df['ipc'] == 3).all())
            self.assertTrue(df['inst_spec'].isna().all())
            self.assertEqual(stdout.getvalue().count('missing optional stats:'), 1)
            self.assertIn('2 file(s)', stdout.getvalue())
            self.assertNotIn('obtained stats', stdout.getvalue())

    def test_json_filter_selects_weighted_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for weight, cycles in [('0.25', 10), ('0.75', 20)]:
                log = root / f'namd_10_{weight}' / 'simulator_err.txt'
                log.parent.mkdir()
                log.write_text(
                    '[PERF ][time=1] SimTop.rob: commitInstr, 20\n'
                    '[PERF ][time=2] SimTop.rob: commitInstr, 30\n'
                    f'[PERF ][time=2] SimTop.rob: clock_cycle, {cycles}\n'
                )
            profile = root / 'profile.json'
            profile.write_text(json.dumps({'namd': {'points': {'10': '0.25'}}}))
            output = root / 'output.csv'
            argv = ['batch.py', '-s', str(root), '-X', '-g', 'basic',
                    '--json-filter', str(profile), '--jobs', '1', '-o', str(output)]
            with patch('sys.argv', argv), \
                    redirect_stdout(io.StringIO()):
                batch.main()
            df = pd.read_csv(output)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, 'cycles'], 10)
