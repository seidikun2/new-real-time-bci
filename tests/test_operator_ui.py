"""Testes com Tk real; executar com DISPLAY ou Xvfb."""
import os
import time
import tempfile
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
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        from unittest.mock import patch

        settings = patch.dict(
            os.environ, {'BCI_SETTINGS_PATH': str(Path(self.output.name) / 'preferences.json')}
        )
        settings.start()
        self.addCleanup(settings.stop)
        self.cfg.experiment.log_root = self.output.name

    def assert_one_window(self):
        def children(widget):
            for child in widget.winfo_children():
                yield child
                yield from children(child)
        self.assertFalse(any(w.winfo_class() == 'Toplevel' for w in children(self.op.root)))

    def test_registered_session_reloads_after_restart_and_laterality_selects_control_leg(self):
        from bci_core.catalog import save_session, save_preferences
        from bci_core.operator import Operator

        record = dict(
            group='LM',
            subject='P001',
            session='S2',
            cap='M / 64 canais',
            age='',
            sex='',
            laterality='left',
            collection_date='2026-11-10',
            experimenter='Ana',
        )
        save_session(self.output.name, record)
        save_preferences(self.output.name, 'LM', 'P001', 2)
        failures = []

        def prepare():
            try:
                self.assertEqual(self.op.fields['group'].get(), 'LM')
                self.assertEqual(self.op.fields['subject'].get(), 'P001')
                self.assertEqual(self.op.fields['cap'].get(), 'M / 64 canais')
                self.assertEqual(self.op.fields['experimenter'].get(), 'Ana')
                self.assertEqual(self.op.fields['age'].get(), '')
                self.op.fields['source'].set('Simulação')
                self.op.action.set('prepare')
            except Exception as exc:
                failures.append(exc)
                self.op.action.set('stop')

        self.op.root.after(50, prepare)
        self.assertTrue(self.op.configure(self.cfg), failures)
        self.assertEqual(self.cfg.protocol.online_target, 'left')
        self.assertEqual(self.cfg.experiment.group, 'LM')
        self.assertEqual(self.cfg.experiment.session_id, 2)
        self.op.close()
        reopened = Operator()
        self.addCleanup(reopened.close)
        captured = {}

        def inspect():
            captured.update(
                {
                    key: reopened.fields[key].get()
                    for key in ['group', 'subject', 'session', 'cap', 'experimenter', 'laterality']
                }
            )
            reopened.action.set('stop')

        reopened.root.after(50, inspect)
        self.assertFalse(reopened.configure(self.cfg))
        self.assertEqual(captured['subject'], 'P001')
        self.assertEqual(captured['experimenter'], 'Ana')
        self.assertEqual(captured['laterality'], 'Esquerda')

    def test_new_session_prefills_profile_and_keeps_previous_session_unchanged(self):
        from bci_core.catalog import save_session, save_preferences, session_defaults

        save_session(
            self.output.name,
            dict(
                group='CTRL',
                subject='P002',
                session='S1',
                cap='G',
                age='40',
                sex='F',
                laterality='right',
                collection_date='2026-10-01',
                experimenter='Ana',
            ),
        )
        save_preferences(self.output.name, 'CTRL', 'P002', 1)
        failures = []

        def prepare():
            try:
                self.op.fields['session'].set('2')
                self.op.load_registration()
                self.assertEqual(self.op.fields['cap'].get(), 'G')
                self.assertEqual(self.op.fields['age'].get(), '40')
                self.op.fields['age'].set('')
                self.op.fields['experimenter'].set('Bia')
                self.op.action.set('prepare')
            except Exception as exc:
                failures.append(exc)
                self.op.action.set('stop')

        self.op.root.after(50, prepare)
        self.assertTrue(self.op.configure(self.cfg), failures)
        self.assertEqual(self.cfg.experiment.session_id, 2)
        self.assertEqual(self.cfg.experiment.age, '')
        self.assertEqual(
            session_defaults(self.output.name, 'CTRL', 'P002', 1)['experimenter'], 'Ana'
        )
        self.assertEqual(
            session_defaults(self.output.name, 'CTRL', 'P002', 2)['experimenter'], 'Bia'
        )

    def test_optional_block_comment_is_saved_before_next_and_absent_for_eyes(self):
        from bci_core.receive_data_log import BlockRecorder
        from bci_core.catalog import read_json, read_table
        from bci_core.resting_state import RESTING_SESSION

        self.op.cfg = self.cfg
        recorder = BlockRecorder(self.cfg, 'train')
        recorder.close()
        recorder.signal_path.write_text('lsl_time_s,C3,C4\n1,2,3\n1.01,4,5\n')
        block = dict(complete=True, signal_file=str(recorder.signal_path))

        def annotate():
            self.op.comment_widget.insert(
                '1.0', 'Ruído breve em Cz.\nParticipante ajustou a postura.'
            )
            self.op.action.set('next')

        self.op.root.after(50, annotate)
        self.assertEqual(self.op.review_block('Imagética', block, {'issues': []}, {}), 'next')
        text = read_json(recorder.signal_path.with_suffix('.json'))['comment']
        self.assertIn('ajustou a postura', text)
        self.assertEqual(
            read_table(Path(self.output.name) / 'dataset_index.csv')[1][0]['comment'], text
        )
        self.cfg.experiment.session_type = RESTING_SESSION
        captured = []

        def inspect():
            captured.append(self.op.comment_widget)
            self.op.action.set('next')

        self.op.root.after(50, inspect)
        self.op.review_block('Olhos', block, {'issues': []}, {'not_applicable': True})
        self.assertEqual(captured, [None])

    def test_resting_countdown_and_manual_closed_eyes_use_same_window(self):
        from bci_core.resting_state import RESTING_SESSION
        self.cfg.experiment.session_type = RESTING_SESSION
        self.op.root.after(50, lambda: self.op.action.set('start'))
        self.assertTrue(self.op.prepare_block(self.cfg, 'Olhos abertos e fechados'))
        self.op.begin_block()
        self.op.update_resting_phase('EYES_OPEN', 45., 45.)
        self.assertIn('GRAVANDO · olhos abertos', self.op.event.get())
        self.assertIn('45 s', self.op.resting_countdown.get())
        self.op.prepare_closed_eyes(45.)
        self.assertFalse(self.op.resting_ready)
        self.assertIn('Peça para fechar os olhos', self.op.event.get())
        self.op.resting_start.invoke()
        self.assertTrue(self.op.resting_ready)
        self.op.update_resting_phase('EYES_CLOSED', 30., 45.)
        self.assertIn('olhos fechados', self.op.event.get())
        self.op.finish_resting()
        self.assertIn('Concluído', self.op.resting_countdown.get())
        self.op.root.geometry('960x660')
        self.op.root.update_idletasks()
        self.assertLessEqual(self.op.buttons.winfo_y()+self.op.buttons.winfo_height(), self.op.root.winfo_height())
        self.assert_one_window()
        self.op.end_block()
        self.op.root.after(50, lambda: self.op.action.set('next'))
        self.assertEqual(self.op.review_block('Repouso', {'complete':True}, {'issues':[]}, {'not_applicable':True}), 'next')
        tabs = next(w for w in self.op.body.winfo_children() if w.winfo_class() == 'TNotebook')
        self.assertEqual([tabs.tab(tab, 'text') for tab in tabs.tabs()], ['Snapshot do sinal', 'Sincronização'])

    def test_resting_duration_can_be_changed_to_30_seconds_in_parameters(self):
        self.op.cfg = self.cfg
        self.op.configure_parameters()
        window = next(w for w in self.op.root.winfo_children() if w.winfo_class() == 'Toplevel')
        frame = window.winfo_children()[0]
        choice = next(w for w in frame.winfo_children() if w.winfo_class() == 'TCombobox' and tuple(w.cget('values')) == ('30', '45'))
        choice.set('30')
        next(w for w in frame.winfo_children() if w.winfo_class() == 'TButton' and w.cget('text') == 'Salvar parâmetros').invoke()
        self.assertEqual(self.cfg._raw_config['resting_state']['duration_s'], 30.)
        self.assert_one_window()

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

    def test_report_link_large_figures_and_scrollable_64_channel_snapshot(self):
        from unittest.mock import patch
        import numpy as np
        from PIL import Image
        self.op.cfg = self.cfg
        folder = Path(self.output.name)
        report = folder / 'report.html'
        report.write_text('<html>Relatório completo</html>')
        pca = folder / 'pca.png'
        Image.new('RGB', (2000, 1200), 'white').save(pca)
        snapshot = folder / 'signal_snapshot.npz'
        times = np.arange(1000)/200
        values = np.random.default_rng(1).normal(size=(1000, 64))
        np.savez(snapshot, times=times, values=values, channels=[f'Ch{i}' for i in range(64)])
        assertions = []
        def inspect():
            try:
                self.op.root.update_idletasks()
                self.assertTrue(self.op.report_link.winfo_ismapped())
                canvas = self.op.signal_preview.canvas
                self.assertEqual(len(canvas.find_withtag('channel_label')), 64)
                self.assertEqual({canvas.itemcget(item, 'fill') for item in canvas.find_withtag('signal_trace')}, {'#334155'})
                self.assertGreater(float(canvas.cget('scrollregion').split()[-1]), canvas.winfo_height())
                canvas.yview_moveto(1.)
                self.assertAlmostEqual(canvas.yview()[1], 1., places=2)
                self.op.report_link.invoke()
                tabs = next(widget for widget in self.op.body.winfo_children() if widget.winfo_class() == 'TNotebook')
                tabs.select(1)
                self.op.root.update()
                image_labels = self.op.root.nametowidget(tabs.tabs()[1]).winfo_children()
                self.assertGreater(image_labels[0].image.width(), 380)
                assertions.append(True)
            except Exception as exc:
                assertions.append(exc)
            finally:
                self.op.action.set('next')
        self.op.root.after(180, inspect)
        with patch('webbrowser.open') as browser:
            self.op.review_block('Imagética', {'complete': True},
                                 {'issues': [], 'report_path': str(report), 'snapshot_file': str(snapshot)},
                                 {'cv': {'acc_mean': .8}, 'pca_check_png': str(pca)})
            browser.assert_called_once_with(report.as_uri())
        self.assertEqual(assertions, [True])
        self.op.root.after(50, lambda: self.op.action.set('start'))
        self.op.prepare_block(self.cfg, 'Online')
        self.assertTrue(self.op.report_link.winfo_ismapped())
        self.assertEqual(self.op.report_path, str(report))
        self.assert_one_window()

    def test_notifications_and_errors_do_not_open_modal_windows(self):
        self.op.notify('Aviso não bloqueante')
        self.assertEqual(self.op.note.get(), 'Aviso não bloqueante')
        with self.assertRaisesRegex(ValueError, 'falha de análise'):
            self.op.run_job('Analisando', lambda: (_ for _ in ()).throw(ValueError('falha de análise')))
        self.assert_one_window()

    def test_acquisition_check_has_no_model_or_stage_selection(self):
        def prepare():
            self.assertNotIn('stage', self.op.fields)
            self.assertNotIn('target', self.op.fields)
            self.op.fields['source'].set('Simulação')
            self.op.action.set('prepare')
        self.op.root.after(50, prepare)
        self.assertTrue(self.op.configure(self.cfg, acquisition_duration=600.))
        self.cfg._raw_config['protocol']['acquisition_test_s'] = 600.
        self.op.root.after(50, lambda: self.op.action.set('start'))
        self.assertTrue(self.op.prepare_block(self.cfg, 'Teste de aquisição · 10 min'))
        self.op.begin_block()
        self.assertIn('auditando', self.op.subtitle.get())
        self.assert_one_window()
        self.op.end_block()
        self.op.root.after(50, lambda: self.op.action.set('next'))
        self.assertEqual(self.op.review_block('Teste de aquisição', {'complete': True}, {'issues': []}, {}), 'next')
        tabs = next(w for w in self.op.body.winfo_children() if w.winfo_class() == 'TNotebook')
        self.assertEqual([tabs.tab(tab, 'text') for tab in tabs.tabs()], ['Snapshot do sinal', 'Sincronização'])
        self.op.update_timing_preview({'rows': [{'stream':'working','age_s':-.5,'elapsed_s':3.,'write_lag_s':.01,'queue_batches':0}],
                                       'options':{'max_age_s':.25}, 'phase':'diagnostic', 'health':{'reason':'Teste temporal'}})
        self.assertIn('no futuro', self.op.timing_preview.metrics.get())
        from bci_core.realtime import RealtimeMonitor
        monitor = RealtimeMonitor()
        monitor.update(compute_s=.07, output_age_s=.08, window_s=1., step_s=.05,
                       enter_persist_s=.1, movement_rise_s=1., control_fresh=False, control_source_age_s=.8)
        self.op.update_timing_preview({'rows': [], 'options': {'max_age_s': .5}, 'phase': 'protocol',
                                      'health': {'reason': 'Online'}, 'decoder': monitor.snapshot()})
        self.assertIn('70.0 ms', self.op.timing_preview.metrics.get())
        self.assertIn('repouso por falta de previsão recente', self.op.timing_preview.metrics.get())
        self.op.update_timing_preview({'rows': [], 'options': {'max_age_s': .25}, 'phase': 'protocol',
                                      'health': {'reason': 'Aguardando novo lote.', 'waiting_for_batch': True}})
        self.assertIn('não invalida os dados recebidos', self.op.timing_preview.status.get())
        from bci_core.recovery import RecoveryAudit
        recovery = RecoveryAudit(3.)
        recovery.observe('working', 'Pico de atraso', [1.], 1., 'protocol')
        self.op.update_timing_preview({'rows': [], 'options': {'max_age_s': .25}, 'phase': 'protocol',
                                      'health': {'reason': 'Recuperando aquisição.'}, 'recovery': recovery.summary(1.1)})
        self.assertIn('interrompe se persistir por 3 s', self.op.timing_preview.metrics.get())
        self.assert_one_window()
