import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pylsl import StreamInfo

from bci_core.acquisition import set_source, stream_layout, recording_config
from bci_core.check_data import inspect_quality
from bci_core.config_models import load_config
from bci_core.report import write_report

ROOT = Path(__file__).resolve().parents[1]


def eeg_info(channels=64, rate=500., unit='microvolts'):
    info = StreamInfo('EEGO_TEST_ONLY', 'EEG', channels, rate, 'float32', 'test-source')
    rows = info.desc().append_child('channels')
    for i in range(channels):
        row = rows.append_child('channel')
        row.append_child_value('label', f'Ch{i}')
        row.append_child_value('unit', unit)
    return info


class AcquisitionAndReportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))
        set_source(self.cfg, 'simulator')

    def test_eego_64_channels_500hz_and_programmable_profile(self):
        set_source(self.cfg, 'eego', 500, 64)
        self.assertEqual(self.cfg.fs_hz, 500.)
        self.assertEqual(self.cfg.n_channels, 64)
        self.assertFalse(self.cfg.runtime.simulate_signal)
        indices, scale, metadata = stream_layout(self.cfg, eeg_info())
        self.assertEqual(len(indices), 64)
        self.assertEqual(metadata['recorded_channels'][0], 'Ch0')
        np.testing.assert_allclose(scale, 1.)
        set_source(self.cfg, 'eego', 1000, 32, 'custom-eego')
        self.assertEqual(self.cfg.lsl.signal_name, 'custom-eego')
        self.assertEqual(len(stream_layout(self.cfg, eeg_info(32, 1000))[0]), 32)

    def test_receive_subset_preserves_labels_order_and_voltage_conversion(self):
        set_source(self.cfg, 'eego')
        self.cfg._raw_config['eego'].update(receive_by='name', receive_channels=['Ch63', 'Ch0'])
        indices, scale, metadata = stream_layout(self.cfg, eeg_info(unit='V'))
        self.assertEqual(indices, [63, 0])
        self.assertEqual(metadata['recorded_channels'], ['Ch63', 'Ch0'])
        np.testing.assert_allclose(scale, [1e6, 1e6])
        self.cfg._raw_config['eego'].update(receive_by='index', receive_channels=[64, 1])
        self.assertEqual(stream_layout(self.cfg, eeg_info())[0], [63, 0])

    def test_external_layout_adopts_stream_rate_and_channels(self):
        set_source(self.cfg, 'eego')
        indices, scales, metadata = stream_layout(self.cfg, eeg_info(channels=32, rate=512))
        self.assertEqual(self.cfg.fs_hz, 512.)
        self.assertEqual(self.cfg.n_channels, 32)
        self.assertEqual(len(indices), 32)
        self.assertTrue(metadata['observations'])
        self.cfg._raw_config['eego'].update(receive_channels=[1, 1])
        with self.assertRaises(ValueError):
            stream_layout(self.cfg, eeg_info())

    def test_missing_unit_preserves_native_samples_without_blocking_recording(self):
        set_source(self.cfg, 'eego')
        indices, scales, metadata = stream_layout(self.cfg, eeg_info(unit=''))
        self.assertEqual(metadata['unit'], 'native')
        self.assertIsNone(metadata['scale_to_uV'])
        np.testing.assert_allclose(scales, 1.)
        self.assertEqual(len(indices), 64)
        self.assertTrue(metadata['observations'])

    def test_eego_name_is_preset_and_zero_nominal_rate_uses_profile(self):
        from bci_core.acquisition import EEGO_STREAM_NAME
        set_source(self.cfg, 'eego')
        self.assertEqual(self.cfg.lsl.signal_name, EEGO_STREAM_NAME)
        self.assertEqual(stream_layout(self.cfg, eeg_info(rate=0))[2]['fs_hz'], 500.)

    def test_native_unit_does_not_apply_microvolt_amplitude_limits(self):
        self.cfg.runtime.source = 'eego'
        self.cfg._recorded_unit = 'native'
        self.cfg._recorded_channel_count = 2
        t = np.arange(2048)/self.cfg.fs_hz
        X = np.random.default_rng(4).normal(size=(len(t), 2)) * 1e-6
        result, *_ = inspect_quality(t, X, ['a','b'], np.array([0.,7.]), ['BASELINE','BLOCK_END'], self.cfg)
        self.assertNotIn('canal plano', result['channels'][0]['reasons'])
        self.assertEqual(result['unit'], 'native')

    def test_stream_without_source_id_is_read_without_requesting_auto_recovery(self):
        import threading
        from unittest.mock import Mock, patch
        from bci_core.receive_data_log import resolve_signal
        set_source(self.cfg, 'eego')
        info = Mock()
        info.type.return_value = 'EEG'
        info.name.return_value = self.cfg.lsl.signal_name
        info.source_id.return_value = ''
        info.uid.return_value = 'current-provider'
        with patch('bci_core.receive_data_log.resolve_byprop', return_value=[info]), patch('bci_core.receive_data_log.StreamInlet') as inlet:
            resolve_signal(self.cfg, threading.Event())
        self.assertFalse(inlet.call_args.kwargs['recover'])
        self.assertEqual(self.cfg._signal_uid, 'current-provider')

    def test_recording_metadata_restores_actual_rate_for_analysis(self):
        with tempfile.TemporaryDirectory() as root:
            signal = Path(root) / 'signal.csv'
            signal.with_suffix('.json').write_text(json.dumps({'stream': dict(fs_hz=500., source='eego', recorded_channels=['Cz', 'Fz']), 'effective_config': {'trial_duration_s': .4}, 'config': {'protocol': {'trials_per_class': 2}}}))
            restored = recording_config(self.cfg, signal)
            self.assertEqual(restored.fs_hz, 500.)
            self.assertEqual(restored.n_channels, 2)
            self.assertEqual(restored.trial_duration_s, .4)
            self.assertEqual(restored._raw_config['protocol']['trials_per_class'], 2)
            self.assertEqual(self.cfg.fs_hz, 256.)

    def test_simulation_spectral_metrics_are_informative_but_nan_still_warns(self):
        fs = self.cfg.fs_hz
        t = np.arange(2048) / fs
        X = np.random.default_rng(4).normal(size=(len(t), 16))
        report, *_ = inspect_quality(t, X, [f'ch{i}' for i in range(16)], np.array([0., 7.9]), ['BASELINE', 'BLOCK_END'], self.cfg)
        self.assertTrue(report['simulated'])
        self.assertTrue(all(not row['reasons'] for row in report['channels']))
        self.assertGreater(report['channels'][0]['high_frequency_ratio'], .35)
        X[0, 0] = np.nan
        report, *_ = inspect_quality(t, X, [f'ch{i}' for i in range(16)], np.array([0., 7.9]), ['BASELINE', 'BLOCK_END'], self.cfg)
        self.assertEqual(report['channels'][0]['invalid_samples'], 1)
        self.assertTrue(report['channels'][0]['reasons'])

    def test_report_has_pca_confusion_figures_and_channel_table_at_bottom(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots()
            axis.scatter([0, 1], [1, 0])
            pca = folder / 'source_pca.png'
            fig.savefig(pca)
            plt.close(fig)
            t = np.arange(2048) / self.cfg.fs_hz
            X = np.random.default_rng(0).normal(size=(len(t), 16))
            report, *_ = inspect_quality(t, X, [f'ch{i}' for i in range(16)], np.array([0., 7.9]), ['BASELINE', 'BLOCK_END'], self.cfg)
            evaluation = dict(pca_check_png=str(pca), classes_map={'REST_STIM': 0, 'RIGHT_MI_STIM': 1},
                              cv=dict(confusion_matrix=[[8, 2], [1, 9]], class_order=[0, 1], acc_mean=.85, balanced_accuracy=.85, trial_accuracy=.9))
            path = write_report(self.cfg, report, evaluation, 'recording.csv', folder)
            text = path.read_text()
            self.assertIn('confusion_matrix.png', text)
            self.assertIn('pca_check.png', text)
            self.assertGreater(text.index('<h2>Check dos canais'), text.index('<h2>EEG, espectro'))
            self.assertNotIn('{{', text)
            self.assertTrue((folder / 'confusion_matrix.png').exists())
            self.assertTrue((folder / 'pca_check.png').exists())


if __name__ == '__main__':
    unittest.main()
