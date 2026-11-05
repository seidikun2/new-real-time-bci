import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from pylsl import local_clock, proc_clocksync, proc_dejitter, proc_monotonize

from bci_core.acquisition import set_source, recording_config
from bci_core.config_models import load_config
from bci_core.data import read_signal_csv
from bci_core.decoder_calibration import run_calibration
from bci_core.lsl_timing import timing_options, signal_processing_flags, TimestampAudit, audit_training_error
from bci_core.receive_data_log import BlockRecorder
from test_acquisition_report import eeg_info, ROOT


class LSLTimingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.cfg.experiment.log_root = self.folder.name

    def test_regularization_is_external_only_and_large_discontinuities_remain_blocked(self):
        set_source(self.cfg, 'simulator')
        self.assertEqual(signal_processing_flags(self.cfg), proc_clocksync)
        set_source(self.cfg, 'eego')
        self.assertEqual(signal_processing_flags(self.cfg), proc_clocksync | proc_dejitter | proc_monotonize)
        source, synced, working = (TimestampAudit() for _ in range(3))
        for audit in [source, synced]:
            audit.update([1., 1.002, 1.001, 1.003])
        working.update([1., 1.002, 1.004, 1.006])
        self.assertIsNone(audit_training_error(source, synced, working, timing_options(self.cfg)))
        source.update([.7, .702])
        self.assertIn('treino impedido', audit_training_error(source, synced, working, timing_options(self.cfg)))
        self.assertEqual(source.summary()['backward_pairs'], 2)
        self.assertAlmostEqual(source.max_backward_s, .303)
        for audit in [source, synced]:
            audit.previous = None
            audit.backwards = 0
            audit.max_backward_s = 0
        working.max_abs_age_s = 1.
        self.assertIn('relógio', audit_training_error(source, synced, working, timing_options(self.cfg)))

    def test_raw_copy_keeps_source_timestamps_and_clock_correction(self):
        set_source(self.cfg, 'eego', 500, 3)
        self.cfg._raw_config['eego']['receive_channels'] = [3, 1]
        recorder = BlockRecorder(self.cfg, 'train')
        self.addCleanup(recorder.close)
        stop = threading.Event()
        now = local_clock()
        original = np.array([now-.006, now-.004, now-.005, now-.003])-123.
        working = np.array([now-.006, now-.004, now-.002, now])
        values = [[1e-6, 2e-6, 3e-6]]*4
        inlet, raw = Mock(), Mock()
        inlet.info.return_value = eeg_info(3, 500, unit='V')
        raw.time_correction.return_value = 123.
        raw.pull_chunk.return_value = (values, original.tolist())
        def chunk(**kwargs):
            stop.set()
            return values, working.tolist()
        inlet.pull_chunk.side_effect = chunk
        with patch('bci_core.receive_data_log.resolve_signal', side_effect=[inlet, raw]):
            recorder.record(stop)
        with recorder.raw_path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        np.testing.assert_array_equal([float(r['source_lsl_time_s']) for r in rows], original)
        np.testing.assert_allclose([float(r['lsl_time_s']) for r in rows], original+123., rtol=0, atol=1e-10)
        self.assertEqual(float(rows[0]['clock_correction_s']), 123.)
        self.assertEqual(float(rows[0]['Ch1']), 2e-6)
        times, X, names = read_signal_csv(recorder.signal_path)
        np.testing.assert_array_equal(times, working)
        self.assertEqual(names, ['Ch2', 'Ch0'])
        np.testing.assert_allclose(X[0], [3., 1.])
        self.assertEqual(recorder.timing_summary()['source']['backward_pairs'], 1)
        self.assertIsNone(recorder.timing_summary()['training_error'])
        with self.assertRaisesRegex(ValueError, 'cópia de auditoria'):
            run_calibration(self.cfg, str(recorder.marker_path), str(recorder.raw_path))

    def test_timestamp_round_trip_does_not_create_duplicates(self):
        set_source(self.cfg, 'simulator', 500, 2)
        recorder = BlockRecorder(self.cfg, 'train')
        self.addCleanup(recorder.close)
        times = np.array([12345., np.nextafter(12345., np.inf)])
        stop = threading.Event()
        inlet = Mock()
        inlet.info.return_value = eeg_info(2, 500)
        def chunk(**kwargs):
            stop.set()
            return [[1., 2.], [3., 4.]], times.tolist()
        inlet.pull_chunk.side_effect = chunk
        with patch('bci_core.receive_data_log.resolve_signal', return_value=inlet):
            recorder.record(stop)
        actual, _, _ = read_signal_csv(recorder.signal_path)
        np.testing.assert_array_equal(actual, times)
        self.assertGreater(actual[1], actual[0])

    def test_slow_disk_does_not_pause_reception_and_queue_overflow_is_explicit(self):
        for capacity, overflow in [(128, False), (1, True)]:
            with self.subTest(capacity=capacity):
                set_source(self.cfg, 'simulator', 500, 2)
                self.cfg._raw_config['acquisition_timing']['queue_batches'] = capacity
                recorder = BlockRecorder(self.cfg, 'train')
                stop, disk_started, release, source_finished = (threading.Event() for _ in range(4))
                inlet = Mock()
                inlet.info.return_value = eeg_info(2, 500)
                count = 0
                start = local_clock()
                def chunk(**kwargs):
                    nonlocal count
                    times = (start+np.arange(count*10, count*10+10)/500).tolist()
                    count += 1
                    if count == 10:
                        stop.set()
                        source_finished.set()
                    return [[1., 2.]]*10, times
                inlet.pull_chunk.side_effect = chunk
                real_writer = csv.writer
                def slow_writer(stream):
                    writer = real_writer(stream)
                    def row(values):
                        if values[0] != 'iso_time':
                            disk_started.set()
                            if not release.wait(3.):
                                raise TimeoutError('Disco simulado não foi liberado.')
                        writer.writerow(values)
                    return Mock(writerow=row)
                errors = []
                def record():
                    try:
                        recorder.record(stop)
                    except Exception as exc:
                        errors.append(exc)
                worker = threading.Thread(target=record)
                try:
                    with patch('bci_core.receive_data_log.resolve_signal', return_value=inlet), patch('bci_core.receive_data_log.csv.writer', side_effect=slow_writer):
                        worker.start()
                        self.assertTrue(disk_started.wait(2.))
                        if overflow:
                            deadline = time.monotonic()+2
                            while not recorder.unwritten_samples and time.monotonic()<deadline:
                                threading.Event().wait(.01)
                            self.assertEqual(recorder.unwritten_samples, 10)
                        else:
                            self.assertTrue(source_finished.wait(2.))
                            self.assertGreaterEqual(recorder.received_samples, 90)
                            self.assertEqual(recorder.samples, 0)
                            self.assertIsNotNone(recorder.last_sample_at)
                        release.set()
                        worker.join(3.)
                        self.assertFalse(worker.is_alive())
                    if overflow:
                        self.assertIn('Fila de gravação cheia', str(errors[0]))
                    else:
                        self.assertFalse(errors)
                        self.assertEqual(recorder.samples, 100)
                finally:
                    release.set()
                    stop.set()
                    worker.join(3.)
                    recorder.close()

    def test_retraining_cannot_ignore_failed_audit(self):
        recorder = BlockRecorder(self.cfg, 'train')
        recorder.save_metadata({'stream': {'fs_hz':500., 'recorded_channels':['a','b'], 'source':'eego'},
                                'timing': {'training_error': 'Descontinuidade original; treino impedido.'}})
        recorder.close()
        restored = recording_config(self.cfg, recorder.signal_path)
        self.assertIn('training_error', restored._timing_audit)
        with self.assertRaisesRegex(ValueError, 'Descontinuidade original'):
            run_calibration(self.cfg, str(recorder.marker_path), str(recorder.signal_path))
