"""Recuperação do dejitter sem modificar os horários originais."""

import unittest
from unittest.mock import Mock, patch

from bci_core.acquisition import set_source
from bci_core.config_models import load_config
from bci_core.lsl_recovery import DejitterRecovery
from bci_core.lsl_timing import signal_processing_flags
from test_acquisition_report import ROOT


class DejitterRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))
        set_source(self.cfg, 'eego')
        self.recovery = DejitterRecovery(self.cfg)

    def test_reported_pauses_request_resets_but_isolated_282ms_spike_does_not(self):
        # Idades e pausas extraídas da telemetria de 16/11, sem EEG/identificadores.
        rows = [
            (93.72, 0.02),
            (93.826, 0.282),
            (93.929, 0.02),
            (119.487533, -0.007501),
            (120.593056, 0.926209),
            (120.900424, 0.958502),
            (123.106107, 2.830514),
        ]
        requested = []
        for elapsed, age in rows:
            stamps = [1000.0 + elapsed - age]
            original = list(stamps)
            event = self.recovery.observe(stamps, 1000.0 + elapsed, elapsed)
            if event:
                requested.append(event['elapsed_s'])
            self.assertEqual(stamps, original)
        self.assertEqual(requested, [120.593056, 123.106107])

    def test_large_pause_with_fresh_eeg_does_not_reopen(self):
        self.recovery.observe([100.0], 100.02, 0.0)
        self.assertIsNone(self.recovery.observe([102.2], 102.22, 2.2))

    def test_constant_old_data_without_a_pause_is_not_hidden_by_resets(self):
        for index in range(50):
            elapsed = index * 0.1
            self.assertIsNone(self.recovery.observe([100.0 + elapsed], 102.0 + elapsed, elapsed))

    def test_reopen_uses_a_new_sdk_inlet_and_reports_discarded_pending_samples(self):
        self.recovery.observe([100.0], 100.02, 0.0)
        event = self.recovery.observe([100.1], 101.2, 1.18)
        inlet, info, replacement = Mock(), Mock(), Mock()
        resolved = Mock()
        inlet._resolved_info = resolved
        inlet.samples_available.return_value = 17
        resolved.source_id.return_value = ''
        with patch('bci_core.lsl_recovery.StreamInlet', return_value=replacement) as factory:
            self.assertIs(self.recovery.reopen(inlet, info, event), replacement)
        factory.assert_called_once_with(
            resolved, recover=False, processing_flags=signal_processing_flags(self.cfg)
        )
        self.assertIs(replacement._resolved_info, resolved)
        inlet.close_stream.assert_called_once()
        self.assertEqual(self.recovery.summary()['discarded_buffer_samples'], 17)
        self.assertEqual(self.recovery.summary()['resets'], 1)

    def test_failed_reopen_is_explicit_and_does_not_close_the_original_inlet(self):
        self.recovery.observe([100.0], 100.0, 0.0)
        event = self.recovery.observe([100.1], 101.2, 1.2)
        inlet = Mock()
        inlet.samples_available.return_value = 0
        with patch('bci_core.lsl_recovery.StreamInlet', side_effect=RuntimeError('source lost')):
            with self.assertRaisesRegex(RuntimeError, 'source lost'):
                self.recovery.reopen(inlet, Mock(), event)
        inlet.close_stream.assert_not_called()
        self.assertEqual(self.recovery.summary()['events'][0]['status'], 'failed')
        self.assertEqual(self.recovery.summary()['resets'], 0)

    def test_clocksync_and_explicit_disable_keep_original_policy(self):
        for source, enabled in [('simulator', True), ('eego', False)]:
            set_source(self.cfg, source)
            self.cfg._raw_config['acquisition_timing']['reset_dejitter_on_gap'] = enabled
            recovery = DejitterRecovery(self.cfg)
            recovery.observe([100.0], 100.0, 0.0)
            self.assertIsNone(recovery.observe([100.1], 102.0, 2.0))
