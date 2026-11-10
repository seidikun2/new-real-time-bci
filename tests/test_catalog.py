import csv
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from bci_core.catalog import (
    INDEX_FIELDS,
    context_for_signal,
    index_recording,
    read_json,
    read_sessions,
    read_table,
    refresh_catalog,
    save_block_comment,
    save_session,
    session_defaults,
    write_json,
    write_table,
)
from bci_core.config_models import load_config
from bci_core.receive_data_log import BlockRecorder, make_log_dir
from bci_core.report import write_report
from bci_core.storage import PROJECT_ROOT


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_config(str(PROJECT_ROOT / 'config.yaml'))
        self.cfg.experiment.log_root = str(self.root)
        self.cfg.experiment.group = 'CTRL'
        self.cfg.experiment.subject_id = 'P001'

    def profile(self, **updates):
        row = dict(
            group='CTRL',
            subject='P001',
            session='S1',
            cap='M / 64 canais',
            age='',
            sex='',
            laterality='left',
            collection_date='2026-11-10',
            experimenter='Ana',
        )
        row.update(updates)
        return row

    def recording(self, block='IM_online_target_com_feedback', complete=True):
        folder = self.root / 'CTRL' / 'P001' / 'S1' / block / 'online'
        folder.mkdir(parents=True, exist_ok=True)
        signal = folder / f'P001_BCI_S1_{block}_online_signal_20261110_100000.csv'
        marker = signal.with_name(signal.name.replace('_signal_', '_markers_'))
        signal.write_text('lsl_time_s,C3,C4\n1,2,3\n1.01,4,5\n')
        marker.write_text('lsl_time_s,label\n1,BASELINE\n1.01,BLOCK_END\n')
        experiment = asdict(self.cfg.experiment)
        experiment.update(session_type=block)
        write_json(
            signal.with_suffix('.json'),
            {
                'complete': complete,
                'mode': 'online',
                'reason': 'Parada manual' if complete is False else 'completed',
                'config': {
                    'experiment': experiment,
                    'protocol': {'stage': 'single_target', 'online_target': 'left'},
                },
                'stream': {
                    'fs_hz': 100.0,
                    'recorded_channels': ['C3', 'C4'],
                    'source': 'eego',
                    'unit': 'native',
                },
            },
        )
        return signal

    def test_new_paths_group_subject_session_and_clear_feedback_names(self):
        self.cfg.experiment.session_type = 'IM_online_target_PCA'
        path = make_log_dir(self.cfg, 'online')
        self.assertEqual(
            path, self.root / 'CTRL' / 'P001' / 'S1' / 'IM_online_target_com_feedback' / 'online'
        )
        self.assertNotIn('PCA', path.as_posix())
        self.cfg.experiment.group = 'UNASSIGNED'
        with self.assertRaises(ValueError):
            make_log_dir(self.cfg, 'train')

    def test_registration_survives_reload_and_new_session_inherits_without_changing_old(self):
        save_session(self.root, self.profile())
        loaded = session_defaults(self.root, 'CTRL', 'P001', 1)
        self.assertEqual(loaded['cap'], 'M / 64 canais')
        self.assertEqual(loaded['age'], '')
        self.assertEqual(loaded['laterality'], 'left')
        second = session_defaults(self.root, 'CTRL', 'P001', 2)
        self.assertEqual(second['cap'], loaded['cap'])
        second.update(experimenter='Bia', age='42')
        save_session(self.root, second)
        self.assertEqual(len(read_sessions(self.root)), 2)
        self.assertEqual(session_defaults(self.root, 'CTRL', 'P001', 1)['experimenter'], 'Ana')
        self.assertEqual(session_defaults(self.root, 'CTRL', 'P001', 2)['experimenter'], 'Bia')

    def test_registration_json_survives_locked_csv_and_rebuild_exports_it(self):
        with patch('bci_core.catalog.write_table', side_effect=PermissionError('Planilha em uso')):
            result = save_session(self.root, self.profile())
        self.assertIn('_warning', result)
        self.assertEqual(session_defaults(self.root, 'CTRL', 'P001', 1)['experimenter'], 'Ana')
        refresh_catalog(self.root)
        self.assertEqual(read_table(self.root / 'sessions.csv')[1][0]['experimenter'], 'Ana')

    def test_same_identifier_in_another_group_does_not_change_known_recording_group(self):
        signal = self.recording()
        save_session(self.root, self.profile(group='LM', cap='L', laterality='right'))
        context = context_for_signal(signal, self.root)
        self.assertEqual(context['group'], 'CTRL')
        self.assertNotEqual(context.get('cap'), 'L')

    def test_legacy_import_preserves_curated_columns_and_can_assign_group_without_moving_data(self):
        folder = self.root / 'P002' / 'S3' / 'IM_treino' / 'train'
        folder.mkdir(parents=True)
        signal = folder / 'P002_BCI_S3_IM_treino_train_signal_20261028_143728.csv'
        signal.write_text('lsl_time_s,Cz\n1,2\n')
        signal.with_name(signal.name.replace('_signal_', '_markers_')).write_text(
            'lsl_time_s,label\n1,BASELINE\n'
        )
        relative = signal.relative_to(self.root).as_posix()
        curated = dict(
            run_id='legacy_01',
            include='0',
            group='UNASSIGNED',
            subject='P002',
            session='S3',
            signal_path=relative,
            bad_channels='Cz',
            exclude_trials='2,4',
            reason='Análise manual',
            train_run_id='chosen_03',
            custom_field='preservar',
        )
        write_table(self.root / 'dataset_index.csv', INDEX_FIELDS, [curated])
        rows = refresh_catalog(self.root)
        self.assertEqual(len(rows), 1)
        for key in [
            'run_id',
            'include',
            'bad_channels',
            'exclude_trials',
            'reason',
            'train_run_id',
            'custom_field',
        ]:
            self.assertEqual(rows[0][key], curated[key])
        self.assertEqual(rows[0]['group'], 'UNASSIGNED')
        self.assertEqual(rows[0]['complete'], '')
        self.assertEqual(rows[0]['collection_date'], '2026-10-28')
        save_session(self.root, self.profile(group='LM', subject='P002', session='S3'))
        self.assertEqual(read_table(self.root / 'dataset_index.csv')[1][0]['group'], 'LM')
        rows = refresh_catalog(self.root)
        self.assertEqual(rows[0]['group'], 'LM')
        self.assertEqual(rows[0]['signal_path'], relative)
        self.assertTrue(signal.exists())
        self.assertEqual(len(read_sessions(self.root)), 1)

    def test_explicit_partial_status_and_file_sizes_do_not_depend_on_guessing(self):
        signal = self.recording(complete=False)
        row = index_recording(signal, self.root)
        self.assertEqual(row['complete'], '0')
        self.assertEqual(row['status'], 'partial')
        self.assertEqual(int(row['signal_bytes']), signal.stat().st_size)
        self.assertEqual(row['interrupt_reason'], 'Parada manual')
        self.assertEqual(row['block_type'], 'online_com_feedback')
        fields, rows = read_table(self.root / 'dataset_index.csv')
        rows[0]['include'] = '0'
        rows[0]['reason'] = 'Manter fora da análise'
        write_table(self.root / 'dataset_index.csv', fields, rows)
        metadata = read_json(signal.with_suffix('.json'))
        metadata.update(complete=True, reason='completed')
        write_json(signal.with_suffix('.json'), metadata)
        row = index_recording(signal, self.root)
        self.assertEqual(row['complete'], '1')
        self.assertEqual(row['include'], '0')
        self.assertEqual(row['reason'], 'Manter fora da análise')

    def test_multiline_comment_is_preserved_in_index_metadata_report_and_after_copy(self):
        save_session(self.root, self.profile())
        signal = self.recording()
        original = signal.read_bytes()
        index_recording(signal, self.root)
        folder = signal.parent / '_qc' / '20261110_100000'
        folder.mkdir(parents=True)
        report = {
            'issues': [],
            'channels': [],
            'estimated_fs_hz': 100.0,
            'samples': 2,
            'thresholds': {},
            'unit': 'native',
        }
        write_report(self.cfg, report, {}, signal, folder)
        comment = 'Tapping forte, ruído em Cz.\n<observação> & segunda linha'
        self.assertEqual(save_block_comment(signal, comment, self.root), '')
        self.assertEqual(read_json(signal.with_suffix('.json'))['comment'], comment)
        self.assertEqual(read_table(self.root / 'dataset_index.csv')[1][0]['comment'], comment)
        page = (folder / 'report.html').read_text()
        self.assertIn('&lt;observação&gt; &amp; segunda linha', page)
        self.assertEqual(signal.read_bytes(), original)
        with tempfile.TemporaryDirectory() as destination:
            copied_root = Path(destination) / 'outro_laboratorio'
            shutil.copytree(self.root, copied_root)
            copied = copied_root / signal.relative_to(self.root)
            refresh_catalog(copied_root)
            context = context_for_signal(copied, copied_root)
            self.assertEqual(context['comment'], comment)
            self.assertEqual(context['cap'], 'M / 64 canais')
            self.assertEqual(context['laterality'], 'left')
            self.assertTrue(
                all(
                    not Path(r['signal_path']).is_absolute()
                    for r in read_table(copied_root / 'dataset_index.csv')[1]
                )
            )

    def test_comment_survives_locked_index_and_is_recovered_from_json(self):
        signal = self.recording()
        index_recording(signal, self.root)
        with patch(
            'bci_core.catalog.index_recording', side_effect=PermissionError('Tabela em uso')
        ):
            warning = save_block_comment(signal, 'Pico breve.', self.root)
        self.assertIn('salvo', warning)
        refresh_catalog(self.root)
        self.assertEqual(
            read_table(self.root / 'dataset_index.csv')[1][0]['comment'], 'Pico breve.'
        )

    def test_comment_never_replaces_corrupt_recording_metadata(self):
        signal = self.recording()
        sidecar = signal.with_suffix('.json')
        sidecar.write_text('{ json incompleto')
        with self.assertRaises(ValueError):
            save_block_comment(signal, 'Anotação', self.root)
        self.assertEqual(sidecar.read_text(), '{ json incompleto')

    def test_online_index_links_the_selected_training_recording(self):
        training = self.recording('IM_treino')
        trained = index_recording(training, self.root)
        model = training.parent / '_model' / 'chosen'
        write_json(model / 'model_meta.json', {'signal_file': str(training)})
        online = self.recording()
        metadata = read_json(online.with_suffix('.json'))
        metadata['active_model'] = str(model)
        write_json(online.with_suffix('.json'), metadata)
        row = index_recording(online, self.root)
        self.assertEqual(row['train_run_id'], trained['run_id'])
