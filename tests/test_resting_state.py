import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from bci_core.acquisition import set_source
from bci_core.config_models import load_config
from bci_core.decoder_calibration import run_calibration
from bci_core.resting_state import RESTING_SESSION, resting_options, resting_schedule, run_resting
from main import analyze_block, protocol_blocks

ROOT = Path(__file__).resolve().parents[1]


class RestingStateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT/'config.yaml'))

    def test_resting_is_first_in_full_protocol_and_can_be_skipped_when_resuming(self):
        self.assertEqual(protocol_blocks(self.cfg)[0][2], 'resting')
        for phase in ['imagery', 'online']:
            self.cfg.protocol.start_phase = phase
            self.assertNotIn('resting', [mode for _, _, mode in protocol_blocks(self.cfg)])
        self.cfg.protocol.start_phase = 'execution'
        self.cfg._raw_config['resting_state']['enabled'] = False
        self.assertEqual(protocol_blocks(self.cfg)[0][2], 'train')

    def test_45_seconds_default_and_30_seconds_configurable(self):
        self.assertEqual([event.duration for event in resting_schedule(self.cfg)], [45., 45., 0.])
        self.cfg._raw_config['resting_state']['duration_s'] = 30.
        self.assertEqual([event.duration for event in resting_schedule(self.cfg)], [30., 30., 0.])
        for duration in [0, -1, float('nan')]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.cfg._raw_config['resting_state']['duration_s'] = duration
                resting_options(self.cfg)

    def test_manual_transition_is_not_counted_in_either_45_second_condition(self):
        now = [100.]
        recorder, operator = Mock(), Mock()
        operator.resting_ready = False
        def wait(duration):
            now[0] += duration
            if now[0] >= 152.:
                operator.resting_ready = True
            return True
        with patch('bci_core.resting_state.time.monotonic', side_effect=lambda: now[0]), \
             patch('bci_core.resting_state.local_clock', side_effect=lambda: now[0]):
            self.assertTrue(run_resting(self.cfg, recorder, wait, operator))
        markers = [call.args for call in recorder.marker.call_args_list]
        self.assertEqual([m[0] for m in markers], [10, 11, 12, 13, 99])
        self.assertAlmostEqual(markers[1][2]-markers[0][2], 45.)
        self.assertGreaterEqual(markers[2][2]-markers[1][2], 7.)
        self.assertAlmostEqual(markers[3][2]-markers[2][2], 45.)
        operator.prepare_closed_eyes.assert_called_once_with(45.)
        operator.finish_resting.assert_called_once()

    def test_interruption_during_transition_does_not_record_closed_eyes(self):
        self.cfg._raw_config['resting_state']['duration_s'] = 1.
        now = [0.]
        operator, recorder = Mock(), Mock()
        operator.resting_ready = False
        def wait(duration):
            if now[0] >= 1.:
                return False
            now[0] += duration
            return True
        with patch('bci_core.resting_state.time.monotonic', side_effect=lambda: now[0]), \
             patch('bci_core.resting_state.local_clock', side_effect=lambda: now[0]):
            self.assertFalse(run_resting(self.cfg, recorder, wait, operator))
        self.assertEqual([call.args[1] for call in recorder.marker.call_args_list], ['EYES_OPEN', 'EYES_OPEN_END'])
        operator.finish_resting.assert_not_called()

    def test_report_uses_original_duration_without_motor_trial_errors_or_training(self):
        set_source(self.cfg, 'simulator')
        with tempfile.TemporaryDirectory() as folder:
            self.cfg.experiment.log_root = folder
            signal = Path(folder)/'test_signal_eyes.csv'
            markers = Path(folder)/'test_markers_eyes.csv'
            times = np.arange(7400)/100
            values = np.random.default_rng(1).normal(size=(len(times), 2))
            pd.DataFrame({'lsl_time_s': times, 'A': values[:, 0], 'B': values[:, 1]}).to_csv(signal, index=False)
            labels = ['EYES_OPEN', 'EYES_OPEN_END', 'EYES_CLOSED', 'EYES_CLOSED_END', 'BLOCK_END']
            pd.DataFrame({'lsl_time_s':[2., 32., 40., 70., 70.], 'label':labels}).to_csv(markers, index=False)
            metadata = {'mode':'resting', 'complete':True,
                        'config':{'resting_state':{'duration_s':30.}, 'experiment':{'session_type':RESTING_SESSION}},
                        'stream':{'fs_hz':100., 'recorded_channels':['A', 'B'], 'source':'simulator', 'unit':'native'}}
            signal.with_suffix('.json').write_text(json.dumps(metadata))
            block = {'mode':'resting', 'complete':True, 'signal_file':str(signal), 'markers_file':str(markers)}
            with patch('main.run_calibration') as train:
                qc, evaluation = analyze_block(self.cfg, block)
            train.assert_not_called()
            self.assertTrue(evaluation['not_applicable'])
            self.assertEqual([row['duration_s'] for row in qc['resting_state']], [30., 30.])
            self.assertEqual(qc['issues'], [])
            page = Path(qc['report_path']).read_text()
            self.assertIn('Olhos abertos', page)
            self.assertIn('planejado: 30 s', page)
            self.assertNotIn('<h2>Mapa PCA', page)
            self.assertNotIn('<h2>Matriz de confusão', page)
            self.assertNotIn('Acurácia média', page)
            with self.assertRaisesRegex(ValueError, 'não é treino de imagética'):
                run_calibration(self.cfg, str(markers), str(signal))
            metadata['complete'] = False
            metadata['reason'] = 'KeyboardInterrupt: Interrompido pelo experimentador.'
            signal.with_suffix('.json').write_text(json.dumps(metadata))
            pd.DataFrame({'lsl_time_s':[2., 32., 40.], 'label':['EYES_OPEN', 'EYES_OPEN_END', 'BLOCK_END']}).to_csv(markers, index=False)
            block.update(complete=False, reason=metadata['reason'])
            qc, evaluation = analyze_block(self.cfg, block)
            self.assertIsNone(qc['resting_state'][1]['duration_s'])
            self.assertIn('INTERROMPIDO', Path(qc['report_path']).read_text())
