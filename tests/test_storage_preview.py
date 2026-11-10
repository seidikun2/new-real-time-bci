import csv
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from pylsl import local_clock

from bci_core.config_models import load_config
from bci_core.storage import PROJECT_ROOT, external_path, prepare_data_root
from bci_core.signal_preview import signal_snapshot, display_envelope
from bci_core.receive_data_log import make_log_dir, BlockRecorder
from bci_core.acquisition import set_source
from test_acquisition_report import eeg_info


class StorageAndPreviewTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(PROJECT_ROOT / 'config.yaml'))

    def test_data_root_must_be_outside_project_and_symlinks_are_resolved(self):
        for location in [PROJECT_ROOT, PROJECT_ROOT / 'Dados', '   ']:
            with self.assertRaises(ValueError):
                external_path(location)
        with tempfile.TemporaryDirectory() as folder:
            nested = Path(folder) / 'dados'
            self.assertEqual(prepare_data_root(nested), str(nested.resolve()))
            link = Path(folder) / 'project_link'
            try:
                link.symlink_to(PROJECT_ROOT, target_is_directory=True)
            except OSError:
                return  # Windows pode exigir permissão adicional para criar symlink.
            with self.assertRaises(ValueError):
                external_path(link / 'Dados')

    def test_session_path_cannot_fall_back_inside_project(self):
        self.cfg.experiment.log_root = str(PROJECT_ROOT.parent)
        self.cfg.experiment.subject_id = '../' + PROJECT_ROOT.name
        with self.assertRaises(ValueError):
            make_log_dir(self.cfg, 'train')
        with tempfile.TemporaryDirectory() as folder:
            self.cfg.experiment.log_root = folder
            self.cfg.experiment.subject_id = 'P001'
            self.assertEqual(
                make_log_dir(self.cfg, 'train'),
                Path(folder) / 'TEST' / 'P001' / 'S1' / 'IM_treino' / 'train',
            )

    def test_snapshot_is_last_five_seconds_and_envelope_keeps_spikes(self):
        t = np.arange(2000)/100
        X = np.zeros((2000, 2))
        X[-100, 0], X[-150, 1] = 100., -80.
        snapshot = signal_snapshot(t, X, ['Cz', 'Fz'])
        self.assertGreaterEqual(snapshot['times'][0], t[-1]-5.)
        self.assertEqual(snapshot['channels'].tolist(), ['Cz', 'Fz'])
        tt, envelope = display_envelope(snapshot['times'], snapshot['values'], points=50)
        self.assertLessEqual(len(tt), 50)
        self.assertEqual(envelope[:, 0].max(), 100.)
        self.assertEqual(envelope[:, 1].min(), -80.)
        self.assertEqual(X[-100, 0], 100.)

    def test_recorder_preview_is_bounded_and_csv_keeps_all_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            set_source(self.cfg, 'eego', 500, 32)
            self.cfg._raw_config['eego'].update(receive_channels=[2, 1])
            self.cfg.experiment.log_root = folder
            recorder = BlockRecorder(self.cfg, 'train')
            stop = threading.Event()
            samples = np.zeros((3000, 32))
            samples[:, 0], samples[:, 1] = 1e-6, 2e-6
            times = local_clock()-6+np.arange(3000)/500
            inlet = Mock()
            inlet.info.return_value = eeg_info(32, 500, unit='V')
            audit = Mock()
            audit.time_correction.return_value = 0.
            audit.pull_chunk.return_value = ([], [])
            def chunk(**kwargs):
                stop.set()
                return samples.tolist(), times.tolist()
            inlet.pull_chunk.side_effect = chunk
            try:
                with patch('bci_core.receive_data_log.resolve_signal', side_effect=[inlet, audit]):
                    recorder.record(stop)
                snapshot = recorder.snapshot()
                self.assertEqual(snapshot['values'].shape, (2500, 2))
                self.assertEqual(snapshot['channels'].tolist(), ['Ch1', 'Ch0'])
                np.testing.assert_allclose(snapshot['values'][-1], [2., 1.])
                with recorder.signal_path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3000)
                self.assertEqual(float(rows[-1]['Ch1']), 2.)
            finally:
                recorder.close()


if __name__ == '__main__':
    unittest.main()
