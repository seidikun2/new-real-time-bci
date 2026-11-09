import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from bci_core.config_models import load_config
from bci_core.acquisition import set_source
from bci_core.protocol import build_schedule
from bci_core.decoder_calibration import fit_final_model, cross_validate, run_calibration
from bci_core.data import epoch_trials_sliding, find_marker_signal_pairs, attempts_by_class, timestamp_integrity
from bci_core.check_data import inspect_quality, run_check_data
from bci_core.online_inference import window_to_feature
from bci_core.intention_control import IntentionStateMachine, approach
from bci_core.feedback_contract import signal_vector
from main import validate_model, with_session

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))
        set_source(self.cfg, 'simulator')

    def test_protocol_balanced_repeatable_and_timing(self):
        self.cfg._raw_config['protocol']['trials_per_class'] = 15
        schedule = build_schedule(self.cfg)
        labels = [event.label for event in schedule]
        for label in ['LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM']:
            self.assertEqual(labels.count(label), 15)
        self.assertEqual(labels.count('ATTEMPT'), 45)
        self.assertEqual(schedule[0].duration, 3.)
        self.assertEqual(schedule[1].duration, 2.)
        self.assertEqual(schedule[2].duration, 1.25)
        self.assertEqual(schedule[3].duration, 3.75)
        self.assertEqual(schedule[4].duration, 2.)
        self.assertEqual(labels[-1], 'BLOCK_END')
        self.assertEqual(schedule, build_schedule(self.cfg))
        self.assertNotEqual(schedule, build_schedule(self.cfg, 1))
        self.cfg._raw_config['protocol']['cue_s'] = -1
        with self.assertRaises(ValueError):
            build_schedule(self.cfg)

    def test_full_dimensional_classifier_separate_visual_pca(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(18, 4, 80))
        y = np.repeat([0, 1, 2], 6)
        X[y == 1, 0] *= 4
        X[y == 2, 3] *= 4
        reference, pca, classifier, visual = fit_final_model(X, y)
        self.assertEqual(classifier.n_features_in_, 10)
        self.assertEqual(visual.shape, (18, 2))
        online = window_to_feature(X[0], reference)
        self.assertEqual(online.shape, (10,))
        np.testing.assert_allclose(pca.transform(online[None])[0], visual[0])
        self.assertEqual(classifier.predict(online[None]).shape, (1,))

    def test_cv_uses_full_features_and_disjoint_trials(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(12), 2)
        y = np.repeat(np.repeat([0, 1], 6), 2)
        X = rng.normal(size=(24, 4, 60))
        X[y == 1, 0] *= 5
        splits_seen = []
        original = StratifiedGroupKFold.split
        def split(splitter, features, labels, groups):
            for train, test in original(splitter, features, labels, groups):
                self.assertFalse(set(groups[train]) & set(groups[test]))
                splits_seen.append((train, test))
                yield train, test
        original_fit = SVC.fit
        def fit(clf, features, labels, *args, **kwargs):
            self.assertEqual(features.shape[1], 10)
            return original_fit(clf, features, labels, *args, **kwargs)
        with patch.object(StratifiedGroupKFold, 'split', split), patch.object(SVC, 'fit', fit):
            cv = cross_validate(X, y, groups, 1., 3)
        self.assertEqual(cv['status'], 'ok')
        self.assertEqual(len(splits_seen), 3)
        self.assertEqual(cv['windows'], 24)
        self.assertEqual(cv['trials'], 12)
        tiny = cross_validate(X[:2], np.array([0, 1]), np.array([0, 1]), 1., 5)
        self.assertIsNone(tiny['acc_mean'])
        self.assertEqual(tiny['splits'], 0)

    def test_identical_channels_and_one_flat_still_train_in_full_space(self):
        X = np.repeat(np.random.default_rng(9).normal(size=(12, 1, 250)), 4, axis=1)
        X[:, 3] = 0.
        y = np.repeat([0, 1], 6)
        reference, pca, classifier, visual = fit_final_model(X, y)
        self.assertEqual(classifier.n_features_in_, 10)
        self.assertTrue(np.isfinite(visual).all())
        self.assertEqual(cross_validate(X, y, np.arange(12), 1., 2)['status'], 'ok')

    def test_timestamp_diagnostic_and_training_report_exact_failures(self):
        times = np.array([0., .002, .002, .001, np.nan, np.inf])
        original = times.copy()
        result = timestamp_integrity(times)
        self.assertEqual(result['nonfinite_samples'], 2)
        self.assertEqual(result['duplicate_pairs'], 1)
        self.assertEqual(result['backward_pairs'], 1)
        self.assertAlmostEqual(result['largest_backward_s'], .001)
        self.assertEqual(result['first_bad_csv_row'], 4)
        np.testing.assert_array_equal(times, original)
        self.assertIsNone(timestamp_integrity([1., 1.002, 1.05])['error'])
        with tempfile.TemporaryDirectory() as folder:
            markers = Path(folder) / 'markers.csv'
            signal = Path(folder) / 'signal.csv'
            markers.write_text('lsl_time_s,label\n0,BASELINE\n')
            text = 'lsl_time_s,ch1,ch2\n0,1,2\n.002,2,3\n.002,3,4\n.001,4,5\nnan,5,6\ninf,6,7\n'
            signal.write_text(text)
            with self.assertRaises(ValueError) as error:
                run_calibration(self.cfg, str(markers), str(signal))
            self.assertEqual(str(error.exception), result['error'])
            self.assertEqual(signal.read_text(), text)

    def test_windows_do_not_bridge_gaps_or_exceed_attempt(self):
        fs = 100
        times = np.arange(300) / fs
        times[150:] += 1.
        X = np.random.default_rng(0).normal(size=(300, 2))
        epochs, y, groups, _ = epoch_trials_sliding(times, X, {'REST_STIM': [1.]}, {'REST_STIM': 0}, fs, .4, .1, 1., 0.)
        self.assertEqual(epochs.shape[0], 2)
        self.assertTrue(np.all(groups == 0))
        with self.assertRaises(ValueError):
            epoch_trials_sliding(times, X, {'REST_STIM': [5.]}, {'REST_STIM': 0}, fs, .4, .1, 1., 0.)

    def test_confuser_does_not_reuse_previous_label(self):
        events = attempts_by_class([1, 2, 3, 4], ['LEFT_MI_STIM', 'ATTEMPT', 'RIGHT_MI_STIM', 'ATTEMPT'], ['LEFT_MI_STIM'])
        self.assertEqual(events, {'LEFT_MI_STIM': [2.]})

    def test_quality_detects_flat_nan_line_noise_and_time_gaps(self):
        set_source(self.cfg, 'hiamp')
        fs = self.cfg.fs_hz
        times = np.arange(2048) / fs
        X = np.column_stack([np.zeros(len(times)), np.sin(2*np.pi*50*times), np.random.default_rng(0).normal(size=len(times))])
        X[3, 2] = np.nan
        times[100:] += .5
        report, _, _, _, bad = inspect_quality(times, X, ['ch1', 'ch2', 'ch3'], np.array([0., .1]), ['BASELINE', 'ATTEMPT'], self.cfg)
        self.assertEqual(report['gaps'], 1)
        self.assertIn('canal plano', report['channels'][0]['reasons'])
        self.assertIn('rede 50/60 Hz', report['channels'][1]['reasons'])
        self.assertEqual(report['channels'][2]['invalid_samples'], 1)
        self.assertTrue(bad[0, 2])
        self.assertTrue(any('BLOCK_END' in issue for issue in report['issues']))

    def test_controller_ignores_pca_and_rest_never_moves_both(self):
        machine = IntentionStateMachine()
        active = {'LEFT_MI_STIM': 1, 'REST_STIM': 1, 'RIGHT_MI_STIM': 1}
        probabilities = {'LEFT_MI_STIM': .9, 'REST_STIM': .05, 'RIGHT_MI_STIM': .05}
        machine.update(signal_vector(0, 0, probabilities, active), 1.)
        state, _ = machine.update(signal_vector(1e9, -1e9, probabilities, active), 1.2)
        self.assertEqual(state, 'LEFT_MI_STIM')
        probabilities = {'LEFT_MI_STIM': .01, 'REST_STIM': .98, 'RIGHT_MI_STIM': .01}
        machine.update(signal_vector(9, 9, probabilities, active), 2.)
        state, _ = machine.update(signal_vector(-9, -9, probabilities, active), 2.2)
        self.assertEqual(state, 'REST_STIM')
        self.assertLess(approach(.5, 0, .1, 1., .8), .5)
        machine.reset()
        self.assertEqual(machine.state, 'REST_STIM')

    def test_data_pairs_never_guess_an_unrelated_signal(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / 'a_markers_1.csv').touch()
            (folder / 'a_signal_2.csv').touch()
            self.assertEqual(find_marker_signal_pairs(folder), [])
            (folder / 'a_signal_1.csv').touch()
            self.assertEqual(len(find_marker_signal_pairs(folder)), 1)

    def test_legacy_pca_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'model_meta.json').write_text('{}')
            (Path(folder) / 'classifier.pkl').touch()
            with self.assertRaisesRegex(ValueError, 'PCA'):
                validate_model(self.cfg, folder)

    def test_stage_copies_preserve_active_configuration(self):
        stage = with_session(self.cfg, 'IM_online_x')
        self.assertEqual(self.cfg.experiment.session_type, 'IM_treino')
        self.assertEqual(stage.experiment.session_type, 'IM_online_x')

    def test_all_phases_analyzed_and_online_model_stays_fixed(self):
        from main import run_experiment
        from unittest.mock import Mock
        operator = Mock()
        operator.closing = False
        operator.prepare_block.return_value = True
        operator.review_block.return_value = 'next'
        operator.run_job.side_effect = lambda title, function, *args: function(*args)
        cfg = self.cfg
        cfg.protocol.start_phase = 'auto'
        fake = {'complete': True, 'markers_file': 'm', 'signal_file': 's'}
        with patch('main.choose_model', return_value='/chosen/model'), patch('main.run_block', return_value=fake) as run, patch('main.analyze_block', return_value=({}, {'cv': {'acc_mean': .8}})) as analyze:
            run_experiment(cfg, operator)
        self.assertEqual(run.call_count, 5)
        self.assertEqual(analyze.call_count, 5)
        operator.confirm.assert_not_called()
        operator.notify.assert_not_called()
        self.assertEqual([call.args[1] for call in run.call_args_list], ['resting', 'train', 'train', 'online', 'online'])
        self.assertEqual([call.args[2] for call in run.call_args_list], [None, None, None, '/chosen/model', '/chosen/model'])

    def test_acquisition_failure_is_incomplete_and_preserves_diagnostics(self):
        from main import run_block
        with tempfile.TemporaryDirectory() as folder:
            self.cfg.experiment.log_root = folder
            self.cfg._signal_uid = 'old-block-uid'
            self.cfg._signal_source_id = 'old-block-source'
            with patch('main.MarkerPublisher') as publisher, patch('main.simulate_signal', side_effect=RuntimeError('device failed')), patch('main.BlockRecorder.record'):
                result = run_block(self.cfg)
            self.assertFalse(result['complete'])
            self.assertFalse(hasattr(self.cfg, '_signal_uid'))
            self.assertFalse(hasattr(self.cfg, '_signal_source_id'))
            self.assertIn('device failed', result['reason'])
            self.assertTrue(Path(result['markers_file']).exists())
            self.assertTrue(Path(result['signal_file']).with_suffix('.json').exists())
            sent = [call.args[0] for call in publisher.return_value.send.call_args_list]
            self.assertEqual(sent[-2:], ['REST', 'BLOCK_END'])

    def test_hardware_reference_channel_count_and_transform_preserved(self):
        from bci_core.config_models import expected_signal_channels
        from bci_core.input_hiamp import transform_reference
        set_source(self.cfg, 'hiamp')
        self.assertEqual(expected_signal_channels(self.cfg), 15)
        X = np.random.default_rng(0).normal(size=(10, 16))
        output = transform_reference(self.cfg, X)
        self.assertEqual(output.shape, (10, 15))
        np.testing.assert_allclose(output.mean(axis=1), 0., atol=1e-6)

    def test_report_generated_for_empty_recording(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / 'x_markers_1.csv'
            eeg = Path(root) / 'x_signal_1.csv'
            marker.write_text('lsl_time_s,label\n')
            eeg.write_text('lsl_time_s,ch1,ch2\n')
            result = run_check_data(self.cfg, markers_file=str(marker), signal_file=str(eeg))
            self.assertEqual(result['status'], 'error')
            self.assertTrue(Path(result['report_path']).exists())


if __name__ == '__main__':
    unittest.main()
