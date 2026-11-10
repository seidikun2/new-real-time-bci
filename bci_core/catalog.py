"""Cadastro por sessão e índice portátil dos blocos, dentro da pasta de dados."""

import csv
import json
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from .storage import GROUPS, canonical_block, external_path, feedback_condition, session_path

SESSION_FIELDS = [
    'group',
    'subject',
    'session',
    'cap',
    'age',
    'sex',
    'laterality',
    'collection_date',
    'experimenter',
    'registered',
    'updated_at',
]
INDEX_FIELDS = [
    'run_id',
    'include',
    'group',
    'subject',
    'session',
    'block',
    'block_type',
    'stage',
    'stamp',
    'signal_path',
    'marker_path',
    'available',
    'bad_channels',
    'exclude_trials',
    'reason',
    'train_run_id',
    'cap',
    'age',
    'sex',
    'laterality',
    'collection_date',
    'experimenter',
    'comment',
    'comment_updated_at',
    'feedback_condition',
    'complete',
    'status',
    'interrupt_reason',
    'signal_bytes',
    'signal_mb',
    'marker_bytes',
    'started_at',
    'ended_at',
    'report_path',
]
LOCK = threading.RLock()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_table(path):
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, strict=True)
            return list(reader.fieldnames or []), list(reader)
    except FileNotFoundError:
        return [], []


def write_table(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(
        dict.fromkeys([*fields, *(key for row in rows for key in row if key is not None)])
    )
    with NamedTemporaryFile(
        'w', encoding='utf-8-sig', newline='', dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def session_id(value):
    return f"S{int(str(value).strip().lstrip('Ss'))}"


def session_key(row):
    return row.get('group') or 'UNASSIGNED', row.get('subject', ''), session_id(row['session'])


def read_sessions(data_root):
    root = external_path(data_root)
    _, rows = read_table(root / 'sessions.csv')
    records = {session_key(row): row for row in rows if row.get('subject') and row.get('session')}
    for path in root.glob('*/*/S*/session.json'):
        if path.resolve().is_relative_to(root):
            row = read_json(path)
            if row.get('subject') and row.get('session'):
                records[session_key(row)] = row
    return list(records.values())


def session_defaults(data_root, group, subject, session):
    rows = read_sessions(data_root)
    related = [
        r for r in rows if r['subject'] == subject and r.get('group') in {group, '', 'UNASSIGNED'}
    ]
    exact = [r for r in related if session_id(r['session']) == session_id(session)]
    registered = sorted(
        (r for r in related if r.get('registered') == '1'),
        key=lambda r: int(session_id(r['session'])[1:]),
        reverse=True,
    )
    row = dict(registered[0]) if registered else {}
    if exact:
        row = (
            dict(exact[0])
            if exact[0].get('registered') == '1'
            else {**row, **{key: value for key, value in exact[0].items() if value}}
        )
    if not exact:
        row.update(collection_date=date.today().isoformat(), registered='0')
    row.update(group=group, subject=subject, session=session_id(session))
    return row


def save_session(data_root, record):
    root = external_path(data_root)
    record = {key: str(record.get(key, '')).strip() for key in SESSION_FIELDS}
    record['session'] = session_id(record['session'])
    if record['laterality'] not in {'left', 'right'}:
        raise ValueError('Escolha a lateralidade: Esquerda ou Direita.')
    record.update(registered='1', updated_at=datetime.now().astimezone().isoformat())
    path = (
        session_path(root, record['group'], record['subject'], record['session'][1:])
        / 'session.json'
    )
    with LOCK:
        write_json(path, record)  # O cadastro persiste mesmo se a planilha estiver aberta no Excel.
        rows = [
            r
            for r in read_sessions(root)
            if session_key(r) != session_key(record)
            and not (
                r.get('group') in {'', 'UNASSIGNED'}
                and r['subject'] == record['subject']
                and session_id(r['session']) == record['session']
            )
        ]
        rows.append(record)
        try:
            write_table(root / 'sessions.csv', SESSION_FIELDS, rows)
            fields, recordings = read_table(root / 'dataset_index.csv')
            for row in recordings:
                if (
                    row.get('subject') == record['subject']
                    and row.get('session') == record['session']
                    and row.get('group') in {record['group'], '', 'UNASSIGNED'}
                ):
                    row.update(
                        {
                            key: record[key]
                            for key in SESSION_FIELDS
                            if key not in {'registered', 'updated_at'}
                        }
                    )
            if recordings:
                write_table(root / 'dataset_index.csv', [*fields, *INDEX_FIELDS], recordings)
        except (OSError, csv.Error) as exc:
            return dict(record, _warning=f'Cadastro salvo no JSON; tabela pendente: {exc}')
    return record


def settings_path():
    override = os.environ.get('BCI_SETTINGS_PATH')
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    return base / 'new-real-time-bci' / 'preferences.json'


def save_preferences(data_root, group, subject, session):
    write_json(
        settings_path(),
        dict(
            data_root=str(external_path(data_root)),
            group=group,
            subject=subject,
            session=session_id(session),
        ),
    )


def path_identity(signal, root):
    try:
        parts = Path(signal).relative_to(root).parts
    except ValueError:
        return {}
    for i, part in enumerate(parts):
        if re.fullmatch(r'S\d+', part) and i >= 1:
            return dict(
                subject=parts[i - 1],
                session=part,
                group=parts[i - 2] if i == 2 and parts[0] in GROUPS else 'UNASSIGNED',
            )
    return {}


def catalog_root(signal, hint=None):
    path = Path(signal).resolve()
    if hint:
        root = external_path(hint)
        if path.is_relative_to(root):
            return root
    for parent in path.parents:
        if (parent / 'sessions.csv').exists() or (parent / 'dataset_index.csv').exists():
            return parent
    for parent in path.parents:
        if re.fullmatch(r'S\d+', parent.name):
            candidate = parent.parent.parent
            return candidate.parent if candidate.name in GROUPS else candidate
    return path.parent


def relative_path(path, root):
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def context_for_signal(signal, data_root=None, sessions=None, index_row=None):
    signal = Path(signal).resolve()
    root = catalog_root(signal, data_root)
    metadata = read_json(signal.with_suffix('.json'))
    original = metadata.get('effective_config', {}).get('experiment') or metadata.get(
        'config', {}
    ).get('experiment', {})
    identity = path_identity(signal, root)
    context = dict(identity)
    context.update(
        {
            key: original[key]
            for key in [
                'group',
                'cap',
                'age',
                'sex',
                'laterality',
                'collection_date',
                'experimenter',
            ]
            if key in original
        }
    )
    context['subject'] = original.get('subject_id', context.get('subject', ''))
    context['session'] = session_id(original.get('session_id', context.get('session', 1)))
    if index_row is None:
        _, rows = read_table(root / 'dataset_index.csv')
        index_row = next(
            (
                r
                for r in rows
                if r.get('signal_path', '').replace('\\', '/') == relative_path(signal, root)
            ),
            {},
        )
    if not context['subject']:
        context['subject'] = index_row.get('subject', '')
    if 'session_id' not in original and 'session' not in identity and index_row.get('session'):
        context['session'] = session_id(index_row['session'])
    if context.get('group') not in GROUPS and index_row.get('group') in GROUPS:
        context['group'] = index_row['group']
    if sessions is None:
        sessions = (
            read_sessions(root) if signal.exists() or signal.with_suffix('.json').exists() else []
        )
    candidates = [
        r
        for r in sessions
        if r['subject'] == context['subject'] and session_id(r['session']) == context['session']
    ]
    exact = [r for r in candidates if r.get('group') == context.get('group')]
    chosen = exact or (
        candidates if context.get('group') not in GROUPS and len(candidates) == 1 else []
    )
    if chosen:
        context.update(
            {
                key: chosen[0].get(key, '')
                for key in SESSION_FIELDS
                if key not in {'subject', 'session'}
            }
        )
    context.setdefault(
        'laterality', metadata.get('config', {}).get('protocol', {}).get('online_target', '')
    )
    file_stamp = metadata.get('comment_updated_at', '')
    table_stamp = index_row.get('comment_updated_at', '') or ''
    context['comment'] = (
        metadata.get('comment', '')
        if file_stamp > table_stamp or 'comment' not in index_row
        else index_row.get('comment', '')
    )
    context['comment_updated_at'] = max(file_stamp, table_stamp)
    match = re.search(r'_S\d+_(.+)_(train|online|resting|diagnostic)_signal_', signal.stem)
    context['block'] = canonical_block(original.get('session_type') or (match[1] if match else ''))
    context['mode'] = metadata.get('mode') or (match[2] if match else '')
    context['stage'] = metadata.get('config', {}).get('protocol', {}).get('stage') or index_row.get(
        'stage', ''
    )
    return context


def recording_row(signal, root, previous, sessions):
    signal = Path(signal)
    row = dict(previous)
    context = context_for_signal(signal, root, sessions, previous)
    metadata = read_json(signal.with_suffix('.json'))
    original = metadata.get('effective_config', {}).get('experiment') or metadata.get(
        'config', {}
    ).get('experiment', {})
    match = re.search(
        r'_S\d+_(.+)_(train|online|resting|diagnostic)_signal_(\d{8}_\d{6}(?:_\d{6})?)$',
        signal.stem,
    )
    block = canonical_block(
        original.get('session_type') or (match[1] if match else row.get('block', ''))
    )
    mode = metadata.get('mode') or (match[2] if match else '')
    stamp = match[3] if match else row.get('stamp', '')
    condition = feedback_condition(block)
    block_type = (
        'online_' + condition
        if condition
        else 'online_nao_informado' if mode == 'online' else block or row.get('block_type', '')
    )
    marker = signal.with_name(signal.name.replace('_signal_', '_markers_'))
    complete = metadata.get('complete')
    size = signal.stat().st_size if signal.is_file() else None
    available = signal.is_file() and marker.is_file()
    report = signal.parent / '_qc' / signal.stem.rsplit('_signal_', 1)[-1] / 'report.html'
    row.update(
        {
            key: context.get(key, row.get(key, ''))
            for key in SESSION_FIELDS
            if key not in {'registered', 'updated_at'}
        }
    )
    row.update(
        comment=context['comment'],
        comment_updated_at=context['comment_updated_at'],
        block=f'{block}_{mode}' if mode else block,
        block_type=block_type,
        feedback_condition=condition,
        stamp=stamp,
        signal_path=relative_path(signal, root),
        marker_path=relative_path(marker, root),
        available='1' if available else '0',
        complete='1' if complete is True else '0' if complete is False else '',
        status='complete' if complete is True else 'partial' if complete is False else 'unknown',
        interrupt_reason=metadata.get('reason', '') if complete is False else '',
        signal_bytes=str(size) if size is not None else '',
        signal_mb=f'{size/(1024**2):.3f}' if size is not None else '',
        marker_bytes=str(marker.stat().st_size) if marker.is_file() else '',
        started_at=metadata.get('started_at', ''),
        ended_at=metadata.get('ended_at', ''),
        report_path=relative_path(report, root) if report.is_file() else row.get('report_path', ''),
    )
    row['group'] = row.get('group') or 'UNASSIGNED'
    row.setdefault(
        'run_id', f"{row['group']}_{context['subject']}_{context['session']}_{block}_{stamp}"
    )
    row.setdefault(
        'include', '1'
    )  # Curadoria é preservada; complete/status informam a integridade.
    row['stage'] = metadata.get('config', {}).get('protocol', {}).get('stage') or row.get(
        'stage', ''
    )
    row['collection_date'] = row.get('collection_date') or (
        f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}' if stamp else ''
    )
    return row


def index_recording(signal, data_root=None):
    root = external_path(catalog_root(signal, data_root))
    with LOCK:
        fields, rows = read_table(root / 'dataset_index.csv')
        key = relative_path(signal, root)
        previous = next((r for r in rows if r.get('signal_path', '').replace('\\', '/') == key), {})
        row = recording_row(signal, root, previous, read_sessions(root))
        if not row.get('train_run_id'):
            active = read_json(Path(signal).with_suffix('.json')).get('active_model')
            if active and Path(active).resolve().is_relative_to(root):
                source = read_json(Path(active) / 'model_meta.json').get('signal_file', '')
                candidates = [
                    r
                    for r in rows
                    if source and Path(r.get('signal_path', '')).name == Path(source).name
                ]
                if len(candidates) == 1:
                    row['train_run_id'] = candidates[0]['run_id']
        rows = [r for r in rows if r is not previous]
        rows.append(row)
        write_table(root / 'dataset_index.csv', [*fields, *INDEX_FIELDS], rows)
    return row


def refresh_catalog(data_root):
    root = external_path(data_root)
    with LOCK:
        fields, rows = read_table(root / 'dataset_index.csv')
        sessions = read_sessions(root)
        by_path = {r.get('signal_path', '').replace('\\', '/'): r for r in rows}
        paths = {
            relative_path(p, root): p
            for p in root.rglob('*_signal_*.csv')
            if p.resolve().is_relative_to(root) and not {'_qc', '_model'}.intersection(p.parts)
        }
        for key in by_path:
            path = Path(key)
            candidate = path if path.is_absolute() else root / path
            if candidate.resolve().is_relative_to(root):
                paths.setdefault(key, candidate)
        updated = [
            recording_row(path, root, by_path.get(key, {}), sessions) for key, path in paths.items()
        ]
        updated.extend(
            dict(row, available='0', status='missing')
            for key, row in by_path.items()
            if key not in paths
        )
        known = {session_key(r): r for r in sessions}
        for row in updated:
            if row.get('subject') and row.get('session') and session_key(row) not in known:
                known[session_key(row)] = {key: row.get(key, '') for key in SESSION_FIELDS}
                known[session_key(row)]['registered'] = '0'
        write_table(root / 'sessions.csv', SESSION_FIELDS, list(known.values()))
        write_table(root / 'dataset_index.csv', [*fields, *INDEX_FIELDS], updated)
    return updated


def ensure_session(cfg):
    exp = cfg.experiment
    key = (exp.group, exp.subject_id, session_id(exp.session_id))
    if any(session_key(r) == key for r in read_sessions(exp.log_root)):
        return
    save_session(
        exp.log_root,
        dict(
            group=exp.group,
            subject=exp.subject_id,
            session=session_id(exp.session_id),
            cap=exp.cap,
            age=exp.age,
            sex=exp.sex,
            laterality=cfg.protocol.online_target,
            collection_date=exp.collection_date or date.today().isoformat(),
            experimenter=exp.experimenter,
        ),
    )


def save_block_comment(signal, comment, data_root=None):
    signal = external_path(signal)
    with LOCK:
        sidecar = signal.with_suffix('.json')
        metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError('Metadados inválidos; comentário não substituiu o arquivo original.')
        metadata.update(comment=comment, comment_updated_at=datetime.now().astimezone().isoformat())
        write_json(signal.with_suffix('.json'), metadata)
        warning = ''
        try:
            index_recording(signal, data_root)
        except (OSError, ValueError, csv.Error) as exc:
            warning = f'Comentário salvo nos dados; tabela não atualizada: {exc}'
        folder = signal.parent / '_qc' / signal.stem.rsplit('_signal_', 1)[-1]
        try:
            quality = folder / 'quality.json'
            if quality.exists():
                report = read_json(quality)
                report['comment'] = comment
                write_json(quality, report)
            page = folder / 'report.html'
            if page.exists():
                from .report import comment_section

                text = page.read_text(encoding='utf-8')
                text = re.sub(
                    r'<!-- BLOCK_COMMENT_START -->.*?<!-- BLOCK_COMMENT_END -->',
                    lambda _: comment_section(comment),
                    text,
                    flags=re.S,
                )
                page.write_text(text, encoding='utf-8')
        except OSError as exc:
            warning = f'Comentário salvo; atualização do relatório pendente: {exc}'
    return warning
