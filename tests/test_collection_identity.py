"""O cabeçalho identifica a coleta original e nunca confunde fim com interrupção."""
import json
import tempfile
import unittest
from pathlib import Path

from bci_core.check_data import run_check_data
from bci_core.config_models import load_config
from bci_core.report import collection_info, display_time, write_report
from bci_core.receive_data_log import BlockRecorder

ROOT = Path(__file__).resolve().parents[1]


class CollectionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(str(ROOT / 'config.yaml'))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.signal = self.folder / 'participant_signal_20261103_100000.csv'

    def save_metadata(self, complete):
        metadata = {'complete': complete, 'reason': 'KeyboardInterrupt: Interrompido pelo experimentador.',
                    'started_at': '2026-11-03T10:00:00-03:00', 'ended_at': '2026-11-03T10:02:15-03:00',
                    'requested_at': '2026-11-03T09:59:58-03:00', 'mode': 'train',
                    'config': {'experiment': {'subject_id': 'P007', 'session_id': 4, 'session_type': 'EM_treino'},
                               'runtime': {'source': 'eego'}, 'protocol': {}}}
        self.signal.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
        return metadata

    def report(self, issues=None):
        report = {'issues': issues or [], 'channels': [], 'estimated_fs_hz': 500., 'samples': 10,
                  'thresholds': {}, 'unit': 'uV', 'source': 'simulator'}
        return write_report(self.cfg, report, {}, self.signal, self.folder).read_text(encoding='utf-8')

    def test_header_uses_original_time_subject_and_block_even_after_config_changes(self):
        self.save_metadata(True)
        self.cfg.experiment.subject_id = 'OTHER'
        self.cfg.experiment.session_type = 'IM_online_other'
        page = self.report(['Canal suspeito'])
        self.assertIn('<h1>Execução motora</h1>', page)
        self.assertIn('Sujeito P007', page)
        self.assertIn('Sessão 4', page)
        self.assertIn('ANT Neuro', page)
        self.assertNotIn('Sujeito OTHER', page)
        self.assertIn('03/11/2026 às 10:00:00 (UTC-03:00)', page)
        self.assertIn('03/11/2026 às 10:02:15 (UTC-03:00)', page)
        self.assertIn('>Concluído</span>', page)
        self.assertIn('>Revisar sinal</span>', page)
        self.assertLess(page.index('Início da coleta'), page.index('<h2>Mapa PCA'))
        self.assertLess(page.index('Arquivo: participant_signal'), page.index('<h2>Mapa PCA'))

    def test_interrupted_remains_prominent_even_with_no_quality_alerts(self):
        self.save_metadata(False)
        page = self.report()
        self.assertIn('class="badge interrupted">INTERROMPIDO', page)
        self.assertIn('Os dados são parciais', page)
        self.assertIn('Interrompido pelo experimentador.', page)
        self.assertNotIn('>Concluído</span>', page)
        self.assertLess(page.index('Os dados são parciais'), page.index('Início da coleta'))

    def test_empty_interrupted_block_has_same_identification_and_no_broken_signal_image(self):
        metadata = self.save_metadata(False)
        metadata.update(started_at=None, ended_at=None, reason='TimeoutError: EEG não encontrado.')
        self.signal.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
        marker = self.folder / 'participant_markers_20261103_100000.csv'
        # BLOCK_END também é emitido na interrupção; não prova conclusão normal.
        marker.write_text('lsl_time_s,label\n1,BLOCK_END\n', encoding='utf-8')
        result = run_check_data(self.cfg, markers_file=str(marker), signal_file=str(self.signal))
        page = Path(result['report_path']).read_text(encoding='utf-8')
        self.assertFalse(result['collection']['complete'])
        self.assertIn('class="badge interrupted">INTERROMPIDO', page)
        self.assertIn('Nenhuma amostra EEG foi recebida', page)
        self.assertIn('03/11/2026 às 09:59:58 (UTC-03:00)', page)
        self.assertIn('EEG não encontrado.', page)
        self.assertNotIn('<img src="overview.png"', page)
        saved = json.loads(Path(result['report_path']).with_name('quality.json').read_text())
        self.assertEqual(saved['collection']['block_type'], 'EM_treino')

    def test_legacy_csv_uses_collection_time_and_marks_unknown_completion_and_timezone(self):
        self.signal.write_text('iso_time,lsl_time_s,ch1,ch2\n2026-01-02T08:15:00,1,0,0\n2026-01-02T08:17:00,2,0,0\n', encoding='utf-8')
        identity = collection_info(self.cfg, self.signal)
        self.assertEqual(identity['started_at'], '2026-01-02T08:15:00')
        self.assertEqual(identity['ended_at'], '2026-01-02T08:17:00')
        self.assertIsNone(identity['complete'])
        self.assertIn('02/01/2026 às 08:15:00 (fuso não registrado)', self.report())
        self.assertIn('Conclusão não registrada', self.report())

    def test_online_label_is_recovered_from_original_protocol(self):
        metadata = self.save_metadata(True)
        metadata['config']['experiment']['session_type'] = 'CUSTOM_ONLINE'
        metadata['config']['protocol']['online_modes_two_legs'] = [
            {'session_type': 'CUSTOM_ONLINE', 'label': 'Online duas pernas — bloco 2'}]
        self.signal.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
        self.assertEqual(collection_info(self.cfg, self.signal)['block_label'], 'Online duas pernas — bloco 2')
        self.assertEqual(display_time(None), 'Não registrado')

    def test_unfinished_recording_already_has_persistent_identity(self):
        self.cfg.experiment.log_root = str(self.folder)
        recorder = BlockRecorder(self.cfg, 'train')
        self.assertIn('iso_time,lsl_time_s', recorder.marker_path.read_text())
        recorder.close()  # Sem encerramento normal do run_block.
        identity = collection_info(self.cfg, recorder.signal_path)
        self.assertFalse(identity['complete'])
        self.assertEqual(identity['block_label'], 'Imagética motora')
        self.assertIsNotNone(identity['requested_at'])
        self.assertIn('encerramento normal', identity['reason'])


if __name__ == '__main__':
    unittest.main()
