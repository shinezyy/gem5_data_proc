import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis_inputs import filter_profile_paths, rtl_window, selected_logs

REPO = Path(__file__).resolve().parents[1]


def rtl_text(cycles=40, instructions=100):
    return ("[PERF ][time=1] SimTop.rob: clock_cycle, 999\n"
            "[PERF ][time=1] SimTop.rob: commitInstr, 200\n"
            "[PERF ][time=1] SimTop.l2cache.topDown: l2prefetchSentBOP, 7\n"
            f"[PERF ][time=2] SimTop.rob: clock_cycle, {cycles}\n"
            f"[PERF ][time=2] SimTop.rob: commitInstr, {instructions}\n"
            "[PERF ][time=2] SimTop.frontend: customCounter, 10\n"
            "[PERF ][time=2] SimTop.l2cache.topDown: l2prefetchSentBOP, 9\n")


class InputTests(unittest.TestCase):
    def test_profile_selection_accepts_weight_formats_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / 'profile.json'
            data = {'gcc_input': {'points': {'1': '0.123456789', '2': '0.25', '3': '0.5'}}}
            profile.write_text(json.dumps(data))
            names = ['gcc_input_1_0.123457', 'gcc_input_1_0.999999', 'gcc_input_2_0.25', 'gcc_input_3', 'other_1_0.123457']
            for name in names:
                (root / name).mkdir()
                (root / name / 'simulator_err.txt').touch()
            logs = selected_logs(root, profile)
            self.assertEqual([p.parent.name for _, p in logs], [names[0], names[2], names[3]])
            paths = [('gcc_input_1', root / names[0] / 'simulator_err.txt'),
                     ('gcc_input_1', root / names[1] / 'simulator_err.txt')]
            self.assertEqual(filter_profile_paths(paths, root, data), paths[:1])
            (root / 'gcc_input_1').mkdir()
            (root / 'gcc_input_1/simulator_err.txt').touch()
            with self.assertRaisesRegex(ValueError, '多个目录'):
                selected_logs(root, profile)

    def test_final_epoch_raw_and_original_parser_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'simulator_err.txt'
            path.write_text(rtl_text() + '[PERF ][time=2] SimTop.frontend: customCounter, 12\n')
            text, raw = rtl_window(path)
            self.assertIn('time=1] SimTop.rob: commitInstr, 200', text)
            self.assertIn('999', text)
            self.assertEqual(raw['SimTop.rob:clock_cycle'], 40)
            self.assertIsNone(raw['SimTop.frontend:customCounter'])
            for text in [rtl_text().replace('time=1', 'time=2'),
                         rtl_text().split('time=2')[0],
                         rtl_text() + '[PERF ][time=2] SimTop.core1.rob: commitInstr, 100\n']:
                path.write_text(text)
                self.assertIsNone(rtl_window(path))


class PipelineTests(unittest.TestCase):
    def cli(self, action, request, root):
        path = root / f'{action}.json'
        path.write_text(json.dumps(request))
        result = subprocess.run([sys.executable, str(REPO / 'analysis_pipeline.py'), action, str(path)],
                                cwd=root, capture_output=True, text=True,
                                env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-3000:])

    def test_direct_cli_prepares_both_backends_and_queries_missing_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / 'profile.json'
            profile.write_text(json.dumps({'gcc': {'insts': '1000', 'points': {'1': '.25', '2': '.75', '3': '.1'}}}))
            snapshots = []
            for backend in ('gem5', 'xs'):
                source = root / backend
                for point, cycles in [(1, 100), (2, 200), (3, 300)]:
                    if backend == 'gem5':
                        folder = source / f'spec_all/gcc_{point}'
                        (folder / 'm5out').mkdir(parents=True)
                        block = (f'---------- Begin Simulation Statistics ----------\n'
                                 f'system.cpu.numCycles {cycles} # fixture\n'
                                 'system.cpu.committedInsts 100 # fixture\n'
                                 f'system.cpu.totalIpc {100 / cycles} # fixture\n'
                                 f'system.cpu.customCounter {point * 10} # fixture\n'
                                 '---------- End Simulation Statistics ----------\n')
                        (folder / 'm5out/stats.txt').write_text(block + block)
                        if point != 3:
                            (folder / 'completed').touch()
                    else:
                        suffix = {1: '0.250000', 2: '0.750000', 3: '0.100000'}[point]
                        folder = source / f'gcc_{point}_{suffix}'
                        folder.mkdir(parents=True)
                        (folder / 'simulator_err.txt').write_text(rtl_text(cycles))
                        (folder / 'simulator_out.txt').write_text('EXCEEDING CYCLE/INSTR LIMIT' if point != 3 else 'HIT BAD TRAP')
                output = root / f'{backend}-snapshot'
                request = {'source': str(source), 'profile_path': str(profile), 'output': str(output),
                           'backend': backend, 'spec': '06', 'extract_jobs': 1}
                self.cli('prepare', request, root)
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['completed_points'], 2)
                self.assertEqual(manifest['expected_points'], 3)
                self.assertEqual(len(manifest['excluded']), 1)
                self.assertEqual(len(list((output / 'raw').glob('*.gz'))), 2)
                snapshots.append(str(output))
            output = root / 'comparison'
            output.mkdir()
            self.cli('compare', {'snapshots': snapshots, 'metrics': ['cpi', 'ipc', 'raw:system.cpu.customCounter',
                      'raw:SimTop.frontend:customCounter', 'raw:absent', 'l2_bop_issued'], 'output': str(output)}, root)
            a, b = json.loads((output / 'values.json').read_text())
            for result in (a, b):
                row = next(r for r in result['benchmark'] if r['name'] == 'gcc')
                self.assertAlmostEqual(row['cpi'], 1.75)
                self.assertAlmostEqual(row['ipc'], 1 / 1.75)
                self.assertIsNone(row['raw:absent'])
            row_a = next(r for r in a['benchmark'] if r['name'] == 'gcc')
            row_b = next(r for r in b['benchmark'] if r['name'] == 'gcc')
            self.assertAlmostEqual(row_a['raw:system.cpu.customCounter'], 17.5)
            self.assertIsNone(row_b['raw:system.cpu.customCounter'])
            self.assertEqual(row_b['raw:SimTop.frontend:customCounter'], 10)
            # The original parser sums the last two BOP matches, even across epochs.
            self.assertEqual(row_b['l2_bop_issued'], 16)
            with gzip.open(Path(snapshots[1]) / 'raw/gcc_1.json.gz', 'rt') as stream:
                self.assertEqual(json.load(stream)['SimTop.rob:clock_cycle'], 100)


if __name__ == '__main__':
    unittest.main()
