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
from bci_core.lsl_timing import timing_options, signal_processing_flags, TimestampAudit, TimingMonitor, audit_training_error, audit_warnings
from bci_core.receive_data_log import BlockRecorder
from test_acquisition_report import eeg_info, ROOT


class LSLTimingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.cfg.experiment.log_root = self.folder.name

    def test_original_irregularities_warn_by_default_and_strict_mode_is_explicit(self):
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
        options = timing_options(self.cfg)
        self.assertIsNone(audit_training_error(source, synced, working, options))
        self.assertTrue(audit_warnings(source, synced, options))
        self.assertIn('treino impedido', audit_training_error(source, synced, working, {**options, 'strict_raw_timestamps': True}))
        self.assertEqual(source.summary()['backward_pairs'], 2)
        self.assertAlmostEqual(source.max_backward_s, .303)
        for audit in [source, synced]:
            audit.previous = None
            audit.backwards = 0
            audit.max_backward_s = 0
        working.max_abs_age_s = 1.
        self.assertIn('relógio', audit_training_error(source, synced, working, timing_options(self.cfg)))

    def test_reported_54ms_source_reversal_does_not_abort_healthy_working_stream(self):
        set_source(self.cfg, 'eego')
        source, synced, working = (TimestampAudit() for _ in range(3))
        for audit in [source, synced]:
            audit.samples = 1752
            audit.backwards = 14
            audit.max_backward_s = .0538404
            audit.max_gap_s = .061709
        synced.observe_age(.103482359, 9.8394)
        working.samples = 1752
        working.max_gap_s = .0020533
        working.observe_age(.09758693, 9.8348)
        options = timing_options(self.cfg)
        self.assertIsNone(audit_training_error(source, synced, working, options))
        self.assertIn('53.84', audit_warnings(source, synced, options)[0])
        self.assertIsNotNone(audit_training_error(source, synced, working, {**options, 'strict_raw_timestamps': True}))
        synced.observe_age(.5, 10.)
        self.assertIn('relógio', audit_training_error(source, synced, working, options))
        synced.max_abs_age_s = .1
        source.nonfinite = 1
        self.assertIn('NaN/Inf', audit_training_error(source, synced, working, options))

    def test_source_jitter_warning_does_not_reset_recent_stability(self):
        set_source(self.cfg, 'eego')
        monitor = TimingMonitor(timing_options(self.cfg))
        for index in range(30):
            now = 100.+index*.1
            stamps = now-.02-np.arange(49, -1, -1)/500
            monitor.observe('working', stamps, now, index*.1)
            row = monitor.observe('raw', stamps+(.028 if index%2 else -.028), now, index*.1)
        result = monitor.summary(now, 2.9)
        self.assertTrue(result['current_health']['ready'])
        self.assertIn('fonte com alertas', result['current_health']['reason'])
        self.assertIsNone(result['training_error'])
        self.assertTrue(result['warnings'])
        self.assertEqual(row['batch_error'], '')

    def test_source_warning_policy_keeps_working_stream_failures_blocking(self):
        set_source(self.cfg, 'eego')
        options = timing_options(self.cfg)
        source, synced = TimestampAudit(), TimestampAudit()
        for audit in [source, synced]:
            audit.update([1., 1.002, .948, .950])
        for stamps in [[1., np.nan], [1., 1.], [1., .999], [1., 1.101]]:
            with self.subTest(stamps=stamps):
                working = TimestampAudit()
                working.update(stamps)
                self.assertIsNotNone(audit_training_error(source, synced, working, options))

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
                def slow_writer(stream, *args, **kwargs):
                    writer = real_writer(stream, *args, **kwargs)
                    if Path(stream.name) != recorder.signal_path:
                        return writer
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

    def test_cold_start_peak_does_not_block_stable_protocol_but_later_failures_do(self):
        set_source(self.cfg, 'eego')
        monitor = TimingMonitor(timing_options(self.cfg))
        for kind in ['working', 'raw']:
            monitor.observe(kind, [100., 100.002], 101.055, 0.)
        for index in range(1, 41):
            now = 101.055+index*.1
            stamps = now-.02-np.arange(49, -1, -1)/500
            for kind in ['working', 'raw']:
                monitor.observe(kind, stamps, now, index*.1)
        self.assertTrue(monitor.current_health(now, 4.)['ready'])
        self.assertAlmostEqual(monitor.audits['working'].max_abs_age_s, 1.053)
        monitor.begin()
        now += .1
        stamps = now-.02-np.arange(49, -1, -1)/500
        for kind in ['working', 'raw']:
            monitor.observe(kind, stamps, now, 4.1)
        result = monitor.summary(now, 4.1)
        self.assertIsNone(result['training_error'])
        self.assertAlmostEqual(result['active']['working']['max_abs_age_s'], .02)
        # Depois do início, atraso real continua sendo uma falha, sem reiniciar o histórico.
        monitor.observe('working', stamps+.1, now+1., 5.1)
        self.assertIn('atrasada', monitor.summary(now+1., 5.1)['training_error'])

    def test_future_timestamp_direction_and_stale_stream_are_visible(self):
        set_source(self.cfg, 'simulator')
        monitor = TimingMonitor(timing_options(self.cfg))
        monitor.observe('working', [100., 100.002], 99., 0.)
        result = monitor.summary(99., 0.)
        self.assertIn('adiantada', result['training_error'])
        self.assertAlmostEqual(result['working']['min_age_s'], -1.002)
        monitor.observe('working', [100.004, 100.006], 100.026, 1.)
        self.assertTrue(monitor.current_health(100.026, 1.)['ready'])
        self.assertFalse(monitor.current_health(101., 2.)['ready'])
        self.assertIn('atrasada',monitor.current_health(101.1,2.1)['reason'])

    def test_resuming_after_a_pause_requires_a_new_stable_interval(self):
        set_source(self.cfg, 'simulator')
        monitor = TimingMonitor({**timing_options(self.cfg), 'stable_s':.5})
        last = 100.
        elapsed = 0.
        for _ in range(7):
            stamps=last+np.arange(1,51)/500
            last=stamps[-1]
            monitor.observe('working',stamps,last+.02,elapsed)
            elapsed+=.1
        self.assertTrue(monitor.current_health(last+.02,elapsed-.1)['ready'])
        self.assertFalse(monitor.current_health(last+.4,elapsed+.3)['ready'])
        # O lote represado tem timestamps contínuos, mas a recepção ficou parada.
        stamps=last+np.arange(1,201)/500
        monitor.observe('working',stamps,stamps[-1]+.02,elapsed+.3)
        health=monitor.current_health(stamps[-1]+.02,elapsed+.3)
        self.assertFalse(health['ready'])
        self.assertEqual(health['stable_for_s'],0.)
