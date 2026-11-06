import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from bci_core.config_models import load_config
from bci_core.acquisition import set_source
from bci_core.lsl_timing import TimingMonitor, timing_options
from bci_core.timing_report import timing_report

ROOT = Path(__file__).resolve().parents[1]


class TimingVisualTests(unittest.TestCase):
    def test_report_preserves_cold_start_peak_and_shows_protocol_separately(self):
        cfg = load_config(str(ROOT/'config.yaml'))
        set_source(cfg, 'eego')
        options = timing_options(cfg)
        monitor = TimingMonitor(options)
        rows = []
        def batch(kind, stamps, now, elapsed):
            row = monitor.observe(kind, stamps, now, elapsed)
            row.update(receive_gap_s=.1, write_lag_s=.01, queue_batches=0)
            rows.append(row)
        for kind in ['working','raw']:
            batch(kind, [100.,100.002],101.055,0.)
        for index in range(1,41):
            now=101.055+index*.1
            stamps=now-.02-np.arange(49,-1,-1)/500
            for kind in ['working','raw']:
                batch(kind,stamps,now,index*.1)
        with patch('bci_core.lsl_timing.time.monotonic',return_value=monitor.origin+4.):
            monitor.begin()
        for index in range(41,46):
            now=101.055+index*.1
            stamps=now-.02-np.arange(49,-1,-1)/500
            for kind in ['working','raw']:
                batch(kind,stamps,now,index*.1)
        audit=monitor.summary(now,4.5)
        audit.update(method='dejitter',options=options)
        self.assertIsNone(audit['training_error'])
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            trace=folder/'source_timing.csv'
            pd.DataFrame(rows).to_csv(trace,index=False)
            audit['trace_file']=str(trace)
            result=timing_report(audit,folder)
            self.assertIn('1053.0',result)
            self.assertIn('20.0',result)
            self.assertIn('picos da acomodação',result)
            self.assertIn('Durante o protocolo/teste',result)
            self.assertIn('Negativo: horário no futuro',result)
            self.assertTrue((folder/'timing_overview.png').is_file())
            self.assertEqual(len(pd.read_csv(folder/'timing_batches.csv')),len(rows))

    def test_old_absolute_only_report_does_not_invent_a_direction_or_curve(self):
        with tempfile.TemporaryDirectory() as temp:
            audit={'method':'dejitter','working':{'samples':3476,'max_abs_age_s':1.053,'backward_pairs':0},
                   'training_error':'Distância temporal.'}
            text=timing_report(audit,Path(temp))
            self.assertIn('1053.0',text)
            self.assertIn('Não registrado',text)
            self.assertIn('Série temporal indisponível',text)
            self.assertFalse((Path(temp)/'timing_overview.png').exists())
