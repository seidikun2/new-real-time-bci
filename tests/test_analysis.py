"""Regressões de custo, isolamento da CV e cobertura dos checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from scipy.signal import welch
from sklearn.model_selection import StratifiedGroupKFold

from bci_core.analysis import analysis_options, training_step
from bci_core.check_data import channel_spectra, inspect_quality
from bci_core.config_models import load_config
from bci_core.data import read_signal_csv
from bci_core.decoder_calibration import cross_validate, mean_covariance
from main import analyze_block

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))

    def test_covariance_reuse_never_uses_validation_trials_in_the_reference(self):
        rng = np.random.default_rng(3)
        windows = rng.normal(size=(24, 4, 60))
        windows[12:, 0] *= 4
        labels = np.repeat([0, 1], 12)
        groups = np.repeat(np.arange(12), 2)
        covariances = Covariances('oas').transform(windows)
        expected = list(
            StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
                covariances, labels, groups
            )
        )
        seen = []

        def reference(values):
            seen.append(values.copy())
            return mean_covariance(values)

        with patch('bci_core.decoder_calibration.mean_covariance', side_effect=reference):
            result = cross_validate(windows, labels, groups, 1.0, 3)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['windows'], 24)
        self.assertEqual(len(seen), 3)
        for values, (train, test) in zip(seen, expected):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            np.testing.assert_array_equal(values, covariances[train])

    def test_spectra_in_channel_batches_match_welch_on_the_entire_signal(self):
        values = np.random.default_rng(4).normal(size=(5001, 17))
        expected_f, expected_p = welch(values, fs=500.0, nperseg=1000, axis=0)
        frequencies, power = channel_spectra(values, 500.0)
        np.testing.assert_array_equal(frequencies, expected_f)
        np.testing.assert_allclose(power, expected_p, rtol=1e-13, atol=1e-15)

    def test_duplicate_channel_confirmation_checks_samples_outside_the_preview(self):
        values = np.random.default_rng(5).normal(size=(1000, 4))
        values[:, 2] = values[:, 0]
        values[:, 3] = values[:, 0]
        values[1, 3] += 1.0  # Fora dos pontos da prévia de comparação.
        report, *_ = inspect_quality(
            np.arange(1000) / 100.0,
            values,
            ['A', 'B', 'Cópia', 'Diferente'],
            np.array([0.0, 9.99]),
            ['BASELINE', 'BLOCK_END'],
            self.cfg,
        )
        self.assertIn('Canais idênticos: A e Cópia.', report['issues'])
        self.assertFalse(
            any('idênticos' in issue and 'Diferente' in issue for issue in report['issues'])
        )

    def test_training_step_can_be_restored_without_changing_online_configuration(self):
        original = self.cfg.step_s
        self.assertEqual(training_step(self.cfg), 0.25)
        self.assertEqual(self.cfg.step_s, original)
        self.assertEqual(self.cfg.decoder.step_s, original)
        self.cfg._raw_config['analysis']['training_step_s'] = None
        self.assertEqual(training_step(self.cfg), original)
        for value in [0.0, -1.0, float('nan'), float('inf')]:
            self.cfg._raw_config['analysis']['training_step_s'] = value
            with self.assertRaises(ValueError):
                analysis_options(self.cfg)

    def test_block_reads_once_trains_once_and_preserves_full_quality_coverage(self):
        cfg = self.cfg
        cfg.fs_hz, cfg.n_channels = 100.0, 4
        cfg.window_s, cfg.step_s, cfg.trial_duration_s = 0.2, 0.05, 1.0
        cfg.protocol.stage, cfg.protocol.training_mode = 'two_legs', 'left_right_vs_rest'
        cfg.cv_splits = 2
        cfg._raw_config['analysis']['training_step_s'] = 0.1
        cfg._raw_config['representation_feedback']['grid_size'] = 30
        cfg._raw_config['protocol'].update(
            stage='two_legs', training_mode='left_right_vs_rest', trials_per_class=2
        )
        with tempfile.TemporaryDirectory() as destination:
            cfg.experiment.log_root = destination
            signal = Path(destination) / 'P_TEST_S1_IM_treino_train_signal_20261113_120000.csv'
            markers = signal.with_name(signal.name.replace('_signal_', '_markers_'))
            times = np.arange(1300) / 100.0
            values = np.random.default_rng(7).normal(size=(1300, 4))
            values[:, 3] = 0.0  # O check de canal plano deve continuar presente.
            frame = pd.DataFrame(values, columns=['C3', 'Cz', 'C4', 'Trigger'])
            frame.insert(0, 'lsl_time_s', times)
            frame.to_csv(signal, index=False)
            rows = [(0.0, 'BASELINE')]
            conditions = ['LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM'] * 2
            for index, label in enumerate(conditions):
                stamp = index * 2.0 + 1.0
                rows.extend([(stamp - 0.2, label), (stamp, 'ATTEMPT'), (stamp + 1.0, 'REST')])
            rows.append((12.9, 'BLOCK_END'))
            pd.DataFrame(rows, columns=['lsl_time_s', 'label']).to_csv(markers, index=False)
            signal.with_suffix('.json').write_text(
                json.dumps(
                    {
                        'complete': True,
                        'mode': 'train',
                        'started_at': '2026-11-13T12:00:00+00:00',
                        'ended_at': '2026-11-13T12:00:13+00:00',
                    }
                )
            )
            original = signal.read_bytes()
            progress, covariance_calls = [], []
            transform = Covariances.transform

            def covariances(estimator, windows):
                covariance_calls.append(windows.shape)
                return transform(estimator, windows)

            with patch.object(Covariances, 'transform', covariances), patch(
                'main.read_signal_csv', wraps=read_signal_csv
            ) as read_once, patch(
                'bci_core.decoder_calibration.read_signal_csv'
            ) as train_read, patch(
                'bci_core.check_data.read_signal_csv'
            ) as quality_read:
                qc, evaluation = analyze_block(
                    cfg,
                    dict(
                        complete=True,
                        mode='train',
                        signal_file=str(signal),
                        markers_file=str(markers),
                    ),
                    progress=progress.append,
                )
            train_read.assert_not_called()
            quality_read.assert_not_called()
            read_once.assert_called_once()
            self.assertEqual(len(covariance_calls), 1)
            self.assertEqual(evaluation['feature_dimensions'], 10)
            self.assertEqual(evaluation['step_s'], 0.05)
            self.assertEqual(evaluation['training_step_s'], 0.1)
            self.assertEqual(evaluation['cv']['trials'], 6)
            self.assertEqual(qc['samples'], 1300)
            self.assertEqual(len(qc['channels']), 4)
            self.assertIn('canal plano', qc['channels'][3]['reasons'])
            self.assertEqual(signal.read_bytes(), original)
            self.assertTrue(any('2/2' in stage for stage in progress))
            timing = json.loads(Path(qc['analysis_timing_file']).read_text())
            self.assertEqual(timing['status'], 'completed')
            self.assertEqual(timing['windows'], evaluation['cv']['windows'])
            self.assertGreater(timing['total_s'], 0.0)
            self.assertIn('analysis_timing.json', Path(qc['report_path']).read_text())
            self.assertTrue(Path(evaluation['pca_check_png']).is_file())
            self.assertTrue(Path(qc['report_path']).with_name('confusion_matrix.png').is_file())


if __name__ == '__main__':
    unittest.main()
