"""Testes com Tk real; executar com DISPLAY ou Xvfb."""
import os
import time
import unittest
from pathlib import Path

from bci_core.config_models import load_config

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Requer display gráfico ou Xvfb')
class PersistentOperatorTests(unittest.TestCase):
    def setUp(self):
        from bci_core.operator import Operator
        self.op = Operator()
        self.addCleanup(self.op.close)
        self.cfg = load_config(str(ROOT / 'config.yaml'))

    def assert_one_window(self):
        def children(widget):
            for child in widget.winfo_children():
                yield child
                yield from children(child)
        self.assertFalse(any(w.winfo_class() == 'Toplevel' for w in children(self.op.root)))

    def test_one_window_for_configuration_collection_analysis_and_results(self):
        def prepare():
            self.op.fields['source'].set('ANT Neuro · eego mylab')
            self.op.action.set('prepare')
        self.op.root.after(50, prepare)
        self.assertTrue(self.op.configure(self.cfg))
        self.assertIn('EE225-020034-000905', self.cfg.lsl.signal_name)
        self.assert_one_window()
        self.op.root.after(50, lambda: self.op.action.set('start'))
        self.assertTrue(self.op.prepare_block(self.cfg, 'Imagética motora'))
        self.op.run_job('Preparando aquisição', lambda: None)
        self.op.begin_block()
        self.assertEqual(self.op.title.get(), 'Imagética motora')
        self.op.set_event('ATTEMPT')
        self.assertTrue(self.op.poll_block('10 amostras · 500 Hz · 64 canais'))
        self.op.cancelled = True
        self.assertFalse(self.op.poll_block('Interrompendo'))
        self.op.end_block()
        self.assert_one_window()
        ticks = []
        def tick():
            ticks.append(True)
            if len(ticks) < 5:
                self.op.root.after(20, tick)
        self.op.root.after(20, tick)
        value = self.op.run_job('Analisando', lambda: (time.sleep(.2), 42)[1])
        self.assertEqual(value, 42)
        self.assertGreaterEqual(len(ticks), 4)
        self.op.root.after(50, lambda: self.op.action.set('next'))
        result = self.op.review_block('Imagética', {'complete': True}, {'issues': []}, {'cv': {'acc_mean': .75}})
        self.assertEqual(result, 'next')
        self.op.root.geometry('960x660')
        self.op.root.update_idletasks()
        self.assertLessEqual(self.op.buttons.winfo_y()+self.op.buttons.winfo_height(), self.op.root.winfo_height())
        self.assert_one_window()

    def test_notifications_and_errors_do_not_open_modal_windows(self):
        self.op.notify('Aviso não bloqueante')
        self.assertEqual(self.op.note.get(), 'Aviso não bloqueante')
        with self.assertRaisesRegex(ValueError, 'falha de análise'):
            self.op.run_job('Analisando', lambda: (_ for _ in ()).throw(ValueError('falha de análise')))
        self.assert_one_window()
