import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from optimization_logs.round5.run_matrix import load_completed, preserve_unrecorded


class ResumeTests(unittest.TestCase):
    def test_resume_requires_matching_valid_prefix_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root/'longer_predictive_seed211'
            artifact.mkdir()
            trace = artifact/'stream_arrivals.json'
            trace.write_text('[]')
            row = dict(case='longer', mode='predictive', seed='211', valid='True', return_code='0',
                       log='longer_predictive_seed211.log', model_hash='model',
                       trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest())
            with (root/'results.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            schedule = [('longer', 'predictive', 211), ('longer', 'predictive_neural', 211)]
            rows, paired = load_completed(root, True, schedule)
            self.assertEqual(len(rows), 1)
            self.assertEqual(paired[('longer', 211)][0], 'model')
            with self.assertRaises(ValueError):
                load_completed(root, False, schedule)
            with self.assertRaises(ValueError):
                load_completed(root, True, list(reversed(schedule)))
            trace.write_text('[1]')
            with self.assertRaises(ValueError):
                load_completed(root, True, schedule)

    def test_orphaned_outputs_are_preserved_before_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = 'longer_certified_seed307'
            work = root/'work'/name
            work.mkdir(parents=True)
            (work/'stream_metrics.json').write_text(json.dumps({'requests': 6000}))
            (root/(name+'.log')).write_text('original log')
            artifact = root/name
            artifact.mkdir()
            (artifact/'trace.json').write_text('original trace')
            preserve_unrecorded(root, name)
            saved = root/'interrupted'/name/'attempt001'
            self.assertEqual((saved/(name+'.log')).read_text(), 'original log')
            self.assertEqual((saved/'artifacts'/'trace.json').read_text(), 'original trace')
            self.assertEqual(json.loads((saved/'work'/'stream_metrics.json').read_text())['requests'], 6000)
            self.assertFalse(work.exists())
            self.assertFalse((root/(name+'.log')).exists())


if __name__ == '__main__':
    unittest.main()
