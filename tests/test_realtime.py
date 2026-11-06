import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from pylsl import local_clock
from scipy import signal

from bci_core.config_models import load_config
from bci_core.decoder_calibration import fit_final_model
from bci_core.intention_control import run_intention_controller
from bci_core.online_inference import run_realtime_decoder, window_to_feature
from bci_core.realtime import LatestWindow, RealtimeMonitor, fresh_timestamp
from bci_core.realtime_report import realtime_report

ROOT = Path(__file__).resolve().parents[1]


class RealtimeTests(unittest.TestCase):
    def make_buffer(self, win=20):
        monitor = RealtimeMonitor()
        sos = signal.butter(3, [5, 40], fs=100, btype='bandpass', output='sos')
        return LatestWindow(win, 2, sos, 100, .5, monitor), monitor

    def test_latest_window_preserves_causal_filter_without_pending_window_queue(self):
        buffer, monitor = self.make_buffer()
        stamps = np.arange(100)/100
        values = np.random.default_rng(1).normal(size=(100, 2))
        for start in range(0, 100, 10):
            buffer.ingest(values[start:start+10], stamps[start:start+10], stamps[start+9]+.01)
        current, stamp, sequence = buffer.latest(20)
        expected = signal.sosfilt(buffer.sos, values, axis=0)
        np.testing.assert_allclose(current, expected[-20:])
        self.assertEqual(sequence, 100)
        self.assertEqual(stamp, .99)
        self.assertIsNone(buffer.latest(101))
        self.assertEqual(monitor.snapshot()['discarded_input_samples'], 0)

    def test_stale_input_is_discarded_and_a_full_new_window_is_required(self):
        buffer, monitor = self.make_buffer()
        buffer.ingest(np.ones((20, 2)), np.arange(20)/100, .2)
        buffer.ingest(np.ones((20, 2)), .2+np.arange(20)/100, 2.)
        self.assertIsNone(buffer.latest(0))
        buffer.ingest(np.ones((10, 2)), 2.+np.arange(10)/100, 2.1)
        self.assertIsNone(buffer.latest(0))
        buffer.ingest(np.ones((10, 2)), 2.1+np.arange(10)/100, 2.2)
        self.assertIsNotNone(buffer.latest(0))
        self.assertEqual(monitor.snapshot()['discarded_input_samples'], 20)

    def test_gap_inside_batch_resets_filter_and_never_bridges_missing_eeg(self):
        buffer, monitor = self.make_buffer()
        stamps = np.r_[np.arange(20)/100, .5+np.arange(10)/100]
        buffer.ingest(np.ones((30, 2)), stamps, .6)
        self.assertIsNone(buffer.latest(0))
        buffer.ingest(np.ones((10, 2)), .6+np.arange(10)/100, .7)
        values, stamp, _ = buffer.latest(0)
        expected = signal.sosfilt(buffer.sos, np.ones((20, 2)), axis=0)
        np.testing.assert_allclose(values, expected)
        self.assertAlmostEqual(stamp, .69)
        self.assertEqual(monitor.snapshot()['gap_resets'], 1)

    def test_invalid_values_and_timestamps_are_not_made_usable_by_drop_policy(self):
        for stamps in [[1., 1.], [1., .99], [1., np.nan]]:
            buffer, _ = self.make_buffer()
            with self.subTest(stamps=stamps), self.assertRaises(ValueError):
                buffer.ingest(np.ones((2, 2)), stamps, 1.)
        buffer, _ = self.make_buffer()
        with self.assertRaises(ValueError):
            buffer.ingest(np.full((2, 2), np.nan), [1., 1.01], 1.02)

    def test_slow_classifier_skips_windows_rejects_old_results_and_recovers(self):
        cfg = load_config(str(ROOT/'config.yaml'))
        cfg._raw_config['realtime']['max_output_age_s'] = .12
        reference, pca, clf, _ = fit_final_model(np.random.default_rng(1).normal(size=(12, 2, 20)), np.repeat([0, 1], 6))
        meta = dict(classifier_input='tangent_space', channels_selected=['A', 'B'], signal_unit='native',
                    filter={'band_hz': [5., 40.], 'order': 3}, window_s=.2, step_s=.05, fs_hz=100.,
                    classes_map={'LEFT_MI_STIM': 0, 'REST_STIM': 1})
        stop = threading.Event()
        monitor = RealtimeMonitor()
        origin = local_clock()
        count = 0
        def pull(**kwargs):
            nonlocal count
            stop.wait(.01)
            stamp = origin+count/100
            count += 1
            return [[np.sin(count), np.cos(count)]], [stamp]
        inlet = Mock()
        inlet.pull_chunk.side_effect = pull
        inlet.samples_available.return_value = 0
        output = []
        outlet = Mock()
        outlet.push_sample.side_effect = lambda vector, timestamp: output.append((timestamp, local_clock()))
        calls = 0
        def slow_once(window, mean):
            nonlocal calls
            calls += 1
            if calls == 2:
                time.sleep(.2)
            if calls >= 6:
                stop.set()
            return window_to_feature(window, mean)
        with tempfile.TemporaryDirectory() as folder:
            cfg.experiment.log_root = folder
            timer = threading.Timer(4., stop.set)
            timer.start()
            try:
                with patch('bci_core.online_inference.resolve_signal', return_value=inlet), \
                     patch('bci_core.online_inference.stream_layout', return_value=([0, 1], np.ones(2), {'fs_hz':100., 'unit':'native', 'recorded_channels':['A', 'B']})), \
                     patch('bci_core.online_inference.load_artifacts', return_value=(reference, pca, clf)), \
                     patch('bci_core.online_inference.read_model_meta', return_value=meta), \
                     patch('bci_core.online_inference.make_outlet_unified', return_value=outlet), \
                     patch('bci_core.online_inference.window_to_feature', side_effect=slow_once):
                    run_realtime_decoder(cfg, model_prefix='chosen', stop_event=stop, monitor=monitor)
            finally:
                timer.cancel()
            stats = monitor.snapshot()
            self.assertGreaterEqual(calls, 6)
            self.assertGreater(stats['rejected_results'], 0)
            self.assertGreater(stats['skipped_windows'], 0)
            self.assertGreater(len(output), 2)
            self.assertTrue(all(0 <= now-stamp <= .12 for stamp, now in output))
            self.assertEqual(stats['discarded_input_samples'], 0)
            self.assertGreater(stats['received_samples'], len(output)*5)
            report = realtime_report(stats, Path(folder))
            self.assertIn('resultados vencidos', report)
            self.assertTrue((Path(folder)/'decoder_timing.csv').exists())
            self.assertTrue((Path(folder)/'decoder_timing.png').exists())
        inlet.close_stream.assert_called_once()

    def test_controller_rejects_old_eeg_even_when_prediction_just_arrived(self):
        cfg = load_config(str(ROOT/'config.yaml'))
        cfg._raw_config['control']['enter_persist_s'] = 0.
        stop = threading.Event()
        inlet, outlet = Mock(), Mock()
        # LEFT com alta confiança, mas produzido com EEG de dois segundos atrás.
        inlet.pull_chunk.return_value = ([[0., 0., 1., 0., 0., 1., 1., 0.]], [local_clock()-2.])
        outputs = []
        def push(vector, **kwargs):
            outputs.append(vector)
            stop.set()
        outlet.push_sample.side_effect = push
        with tempfile.TemporaryDirectory() as folder:
            cfg.experiment.log_root = folder
            with patch('bci_core.intention_control.resolve_byprop', return_value=[Mock()]), \
                 patch('bci_core.intention_control.StreamInlet', return_value=inlet), \
                 patch('bci_core.intention_control.StreamOutlet', return_value=outlet):
                run_intention_controller(cfg, stop_event=stop)
            with next(Path(folder).rglob('*_control_*.csv')).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['fresh'], '0')
            self.assertGreater(float(rows[0]['eeg_age_s']), 1.)
        self.assertTrue(all(vector[0] == vector[1] == 0. and vector[2] == 1. for vector in outputs))
        self.assertFalse(fresh_timestamp(100., 102., .5))
        self.assertFalse(fresh_timestamp(102., 100., .5))
        self.assertTrue(fresh_timestamp(100., 100.1, .5))
