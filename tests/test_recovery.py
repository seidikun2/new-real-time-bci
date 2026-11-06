import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bci_core.config_models import load_config
from bci_core.decoder_calibration import run_calibration
from bci_core.data import epoch_trials_sliding
from bci_core.recovery import RecoveryAudit
from bci_core.usable_data import filter_segments

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_reported_369_and_308ms_spikes_can_recover_without_latching_old_peak(self):
        audit = RecoveryAudit(3.)
        audit.observe('working', 'Atraso 369.4 ms', [100., 100.056], 49.864, 'protocol')
        audit.observe('working', 'Atraso 308.4 ms', [100.058, 100.22], 49.968, 'protocol')
        self.assertIsNone(audit.check(49.972, True))
        self.assertEqual(len(audit.summary(49.972)['excluded_intervals']), 2)
        audit.observe('working', None, [100.222, 100.5], 50.3, 'protocol')
        self.assertIsNone(audit.check(100., True))
        event = audit.summary(100.)['episodes'][0]
        self.assertAlmostEqual(event['duration_s'], .436)
        self.assertEqual(event['batches'], 2)

    def test_persistent_problem_stops_even_with_continuous_reception(self):
        audit = RecoveryAudit(3.)
        for index in range(31):
            audit.observe('working', 'EEG atrasado', [100.+index*.1], index*.1, 'protocol')
        self.assertIn('persistiu por 3.00 s', audit.check(3., True))
        audit.observe('working', None, [104.], 3.1, 'protocol')
        self.assertIsNotNone(audit.check(3.1, True))

    def test_rare_episodes_do_not_accumulate_as_a_single_persistent_failure(self):
        audit = RecoveryAudit(3.)
        for index in range(10):
            audit.observe('working', 'Pico', [100.+index], index*5., 'protocol')
            audit.observe('working', None, [100.5+index], index*5.+.2, 'protocol')
        self.assertIsNone(audit.check(50., True))
        summary = audit.summary(50.)
        self.assertEqual(summary['bad_batches']['working'], 10)
        self.assertEqual(summary['batches']['working'], 20)
        self.assertEqual(len(summary['episodes']), 10)

    def test_warmup_can_recover_and_strict_policy_is_explicit(self):
        audit = RecoveryAudit(0.)
        audit.observe('working', 'Acomodação', [1.], 0., 'warmup')
        self.assertIsNone(audit.check(5., False))
        audit.observe('working', None, [2.], 5.1, 'warmup')
        audit.observe('working', 'Pico', [3.], 6., 'protocol')
        self.assertIsNotNone(audit.check(6., True))
        strict = RecoveryAudit(3.)
        strict.observe('raw', 'Fonte estrita', [3.], 0., 'protocol', strict=True)
        self.assertEqual(strict.fatal, 'Fonte estrita')

    def test_excluded_regions_reset_filter_and_epochs_never_cross_them(self):
        times = np.arange(500)/100
        values = np.random.default_rng(1).normal(size=(500, 2))
        intervals = [{'start_lsl_s':2., 'end_lsl_s':2.2}]
        filtered, segments, usable = filter_segments(times, values, 100, 3, [5., 40.], intervals)
        self.assertEqual(usable['excluded_samples'], 21)
        self.assertTrue(np.isnan(filtered[200:221]).all())
        windows, labels, groups, centers = epoch_trials_sliding(times, filtered, {'REST_STIM':[1.]},
            {'REST_STIM':0}, 100, .4, .1, 3., 0., segments=segments)
        self.assertTrue(np.isfinite(windows).all())
        self.assertTrue(np.all((centers+.2 <= 1.) | (centers-.2 >= 1.21)))
        self.assertEqual(set(groups), {0})

    def test_model_trains_after_startup_duplicates_and_marked_spike_without_editing_csv(self):
        cfg = load_config(str(ROOT/'config.yaml'))
        cfg.protocol.online_target = 'left'
        cfg.window_s, cfg.step_s, cfg.trial_duration_s, cfg.cv_splits = .2, .1, .6, 2
        cfg._raw_config['representation_feedback']['grid_size'] = 30
        with tempfile.TemporaryDirectory() as folder:
            cfg.experiment.log_root = folder
            signal = Path(folder)/'test_signal_1.csv'
            markers = Path(folder)/'test_markers_1.csv'
            times = np.arange(1300)/100
            times[5] = times[4]  # Duplicata anterior às tentativas.
            values = np.random.default_rng(2).normal(size=(1300, 2))
            pd.DataFrame({'lsl_time_s':times, 'A':values[:, 0], 'B':values[:, 1]}).to_csv(signal, index=False)
            original = signal.read_bytes()
            rows = [(0., 'BASELINE')]
            for index, event in enumerate([1., 3., 5., 7., 9., 11.]):
                rows.extend([(event-.1, 'LEFT_MI_STIM' if index%2 == 0 else 'REST_STIM'),
                             (event, 'ATTEMPT'), (event+.6, 'REST')])
            rows.append((12., 'BLOCK_END'))
            pd.DataFrame(rows, columns=['lsl_time_s', 'label']).to_csv(markers, index=False)
            audit = {'training_error':None, 'recovery':{'policy':'exclude_segments',
                     'max_bad_duration_s':3., 'excluded_intervals':[{'start_lsl_s':5.15, 'end_lsl_s':5.25}]}}
            metadata = {'stream':{'fs_hz':100., 'recorded_channels':['A', 'B'], 'source':'eego', 'unit':'native'},
                        'timing':audit, 'complete':True}
            signal.with_suffix('.json').write_text(json.dumps(metadata))
            result = run_calibration(cfg, str(markers), str(signal))['meta']
            self.assertEqual(signal.read_bytes(), original)
            self.assertGreater(result['usable_data']['excluded_samples'], 10)
            self.assertEqual(result['usable_data']['trials_used'], 6)
            self.assertEqual(result['feature_dimensions'], 3)
            self.assertEqual(result['cv']['status'], 'ok')
            self.assertTrue(result['warnings'])
