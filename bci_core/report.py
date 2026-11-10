"""Relatório HTML offline com visão do modelo antes dos detalhes de canais."""
import html
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .class_schema import display_name
from .acquisition import SOURCES
from .timing_report import timing_report
from .realtime_report import realtime_report
from .resting_state import RESTING_LABEL
from .catalog import context_for_signal
from .storage import canonical_block, feedback_condition

def collection_info(cfg, signal_file):
    """Identificação original da coleta, independente da configuração ao reabrir."""
    path = Path(signal_file)
    sidecar = path.with_suffix('.json')
    try:
        metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    raw = metadata.get('config', {})
    context = context_for_signal(signal_file, cfg.experiment.log_root)
    original = metadata.get('effective_config', {}).get('experiment') or raw.get('experiment', {})
    session = canonical_block(
        original.get('session_type') or context.get('block') or cfg.experiment.session_type
    )
    protocol = raw.get('protocol', {})
    label = session
    if protocol.get('acquisition_test_s'):
        label = f"Teste de aquisição · {protocol['acquisition_test_s']/60:g} min"
    if session == protocol.get('motor_session_type', 'EM_treino'):
        label = 'Execução motora'
    elif session == protocol.get('imagery_session_type', 'IM_treino'):
        label = 'Imagética motora'
    elif metadata.get('mode') == 'online' or session.startswith('IM_online'):
        label = 'Online'
    for key, modes in protocol.items():
        if key.startswith('online_modes') and isinstance(modes, list):
            for mode in modes:
                if mode.get('session_type') == session:
                    label = mode.get('label', label)
    if metadata.get('mode') == 'resting':
        label = RESTING_LABEL
    condition = feedback_condition(session)
    if condition:
        label = 'Online · ' + ('sem feedback' if condition == 'sem_feedback' else 'com feedback')
    started, ended = metadata.get('started_at'), metadata.get('ended_at')
    # Arquivos anteriores não tinham esses campos. Use o horário do CSV, nunca
    # mtime do arquivo ou o momento em que o relatório foi gerado.
    if (not started or not ended) and path.exists():
        with path.open(newline='', encoding='utf-8') as stream:
            first, last = None, None
            for row in csv.DictReader(stream):
                value = row.get('iso_time')
                try:
                    datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    continue
                first = first or value
                last = value
            started, ended = started or first, ended or last
    complete = metadata.get('complete')
    runtime = metadata.get('effective_config', {}).get('runtime') or raw.get('runtime', {})
    source = (metadata.get('stream') or {}).get('source') or runtime.get('source')
    if not source and 'simulate_signal' in runtime:
        source = 'simulator' if runtime['simulate_signal'] else 'hiamp'
    return {
        'block_type': session,
        'block_label': label,
        'subject': original.get('subject_id')
        or context.get('subject')
        or cfg.experiment.subject_id,
        'session_id': original.get('session_id')
        or int(context.get('session', f'S{cfg.experiment.session_id}')[1:]),
        'started_at': started,
        'ended_at': ended,
        'requested_at': metadata.get('requested_at'),
        'complete': complete if isinstance(complete, bool) else None,
        'reason': metadata.get('reason'),
        'filename': path.name,
        'source': source,
        'mode': context.get('mode'),
        'participant': context,
        'comment': context.get('comment', ''),
    }


def comment_section(comment):
    text = html.escape(comment) if comment else 'Sem anotações para este bloco.'
    return (
        '<!-- BLOCK_COMMENT_START --><section class="card" id="block-comment"><h2>Anotações do bloco</h2>'
        f'<p style="white-space:pre-wrap">{text}</p></section><!-- BLOCK_COMMENT_END -->'
    )


def display_time(value):
    if not value:
        return 'Não registrado'
    try:
        stamp = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return 'Não registrado'
    text = stamp.strftime('%d/%m/%Y às %H:%M:%S')
    offset = stamp.strftime('%z')
    return text + (f' (UTC{offset[:3]}:{offset[3:]})' if offset else ' (fuso não registrado)')


def confusion_figure(cv, class_map, path):
    matrix = np.asarray(cv.get('confusion_matrix', []))
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return False
    inverse = {value: display_name(key) for key, value in class_map.items()}
    names = [inverse.get(value, str(value)) for value in cv.get('class_order', range(len(matrix)))]
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros(matrix.shape, float), where=totals != 0)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
        for axis, values, title, fmt in zip(axes, [matrix, normalized], ['Contagem de janelas', 'Proporção por classe real'], ['d', '.0%']):
            ConfusionMatrixDisplay(values, display_labels=names).plot(ax=axis, colorbar=False, cmap='Blues', values_format=fmt)
            axis.set(title=title, xlabel='Classe predita', ylabel='Classe real')
        fig.savefig(path, dpi=145, facecolor='white')
        plt.close(fig)
    return True


def write_report(cfg, report, evaluation, signal_file, folder):
    escape = lambda value: html.escape(str(value))
    evaluation = evaluation or {}
    cv = evaluation.get('cv', {})
    collection = report.get('collection') or collection_info(cfg, signal_file)
    complete = collection['complete']
    if complete is False:
        outcome, outcome_class = 'INTERROMPIDO', 'interrupted'
        message = ('Registro interrompido. Os períodos de olhos abertos e fechados podem estar incompletos.'
                   if collection.get('mode') == 'resting' else
                   'Coleta interrompida. Os dados são parciais e este bloco não foi usado no treino automático.')
        if collection.get('reason'):
            message += ' Motivo: ' + collection['reason']
    elif complete is True:
        outcome, outcome_class = 'Concluído', 'good'
        message = ''
    else:
        outcome, outcome_class = 'Conclusão não registrada', 'warn'
        message = 'Não foi possível confirmar como este bloco terminou. Consulte os registros originais.'
    plot_matrix = confusion_figure(cv, evaluation.get('classes_map', {}), folder / 'confusion_matrix.png')
    pca = evaluation.get('pca_check_png') or evaluation.get('pca_map_png')
    pca_available = bool(pca and Path(pca).exists())
    if pca_available:
        if Path(pca).resolve() != (folder / 'pca_check.png').resolve():
            shutil.copyfile(pca, folder / 'pca_check.png')
    def metric(value):
        return '—' if value is None else f'{value:.1%}'
    def figure(available, name, text):
        return f'<img src="{name}" alt="{escape(text)}">' if available else f'<div class="empty">{escape(text)}</div>'
    alerts = ''.join(f'<li>{escape(value)}</li>' for value in report['issues'])
    if not alerts:
        alerts = '<li>Nenhum alerta nos critérios aplicados a este bloco.</li>'
    notes = ''.join(f'<p>{escape(value)}</p>' for value in report.get('observations', []))
    if report.get('resting_state') is not None:
        notes += '<p>Condições conduzidas pelo experimentador; a espera entre elas não faz parte dos períodos medidos.</p>'
        for condition in report['resting_state']:
            duration = condition['duration_s']
            actual = f'{duration:.1f} s' if duration is not None else 'incompleto'
            notes += f"<p><strong>{escape(condition['label'])}:</strong> {actual} (planejado: {condition['planned_s']:g} s).</p>"
    usable = evaluation.get('usable_data', {})
    if usable:
        notes += (f"<p>Treino: {usable['usable_samples']}/{usable['total_samples']} amostras utilizáveis; "
                  f"{usable['excluded_samples']} excluídas. {usable['windows_used']} janelas de "
                  f"{usable['trials_used']}/{usable['trials_total']} tentativas. Os arquivos de coleta estão preservados. "
                  'A acurácia refere-se às janelas utilizadas.</p>')
    rows = []
    for row in report['channels']:
        suspect = bool(row['reasons'])
        values = [row['channel'], f"{row['std']:.3g}", f"{row['peak_to_peak']:.3g}",
                  f"{row['bad_window_fraction']:.1%}", f"{row['line_noise_ratio']:.1%}",
                  f"{row['high_frequency_ratio']:.1%}", ' · '.join(row['reasons']) or 'Sem alerta']
        rows.append('<tr class="'+('suspect' if suspect else '')+'">'+''.join(f'<td>{escape(value)}</td>' for value in values)+'</tr>')
    status = 'Revisar sinal' if report['issues'] else ('Simulação íntegra' if report.get('simulated') else 'Sem alertas')
    available = cv.get('acc_mean') is not None
    summary = ('Validação cruzada por tentativa; as janelas do mesmo trial permanecem no mesmo fold.' if available else
               evaluation.get('error', evaluation.get('note', cv.get('reason', 'Aguardando treino após a inspeção dos dados.'))))
    fs = report['estimated_fs_hz']
    request_note = ''
    if not collection['started_at'] and collection.get('requested_at'):
        request_note = 'Tentativa de coleta iniciada em ' + display_time(collection['requested_at']) + '. '
        request_note += 'Nenhuma amostra EEG foi recebida.' if report['samples'] == 0 else 'Horário das amostras não registrado.'
    fields = {
        'timing': timing_report(report.get('timing_audit', {}), folder),
        'realtime': realtime_report(report.get('realtime', {}), folder),
        'session': escape(collection['block_type']),
        'block_label': escape(collection['block_label']),
        'subject': escape(collection['subject']),
        'session_id': escape(collection['session_id']),
        'source': escape(SOURCES.get(collection.get('source') or report.get('source'), 'EEG')),
        'filename': escape(Path(signal_file).name),
        'status': status,
        'status_class': 'warn' if report['issues'] else 'good',
        'outcome': outcome,
        'outcome_class': outcome_class,
        'outcome_message': (
            f'<div class="collection-alert {outcome_class}">{escape(message)}</div>'
            if message
            else ''
        ),
        'started_at': escape(display_time(collection['started_at'])),
        'ended_at': escape(display_time(collection['ended_at'])),
        'request_note': escape(request_note),
        'samples': f"{report['samples']:,}".replace(',', '.'),
        'fs': f'{fs:.1f}' if fs else '—',
        'channel_count': str(len(report['channels'])),
        'accuracy': metric(cv.get('acc_mean')),
        'balanced': metric(cv.get('balanced_accuracy')),
        'trial_accuracy': metric(cv.get('trial_accuracy')),
        'summary': escape(summary),
        'notes': notes,
        'alerts': alerts,
        'channel_rows': ''.join(rows),
        'performance_note': (
            '<p>O resultado é uma estimativa offline deste bloco; não substitui o modelo ativo no online.</p>'
            if available
            else ''
        ),
        'pca': figure(
            pca_available, 'pca_check.png', 'Mapa PCA disponível após o treino deste bloco.'
        ),
        'matrix': figure(
            plot_matrix,
            'confusion_matrix.png',
            'Matriz indisponível até concluir a validação cruzada.',
        ),
        'thresholds': escape(json.dumps(report['thresholds'], ensure_ascii=False)),
        'unit': escape(
            'escala original (unidade não informada)'
            if report['unit'] == 'native'
            else report['unit']
        ),
        'overview': figure(
            report['samples'] > 1 and (folder / 'overview.png').exists(),
            'overview.png',
            'Sem sinal suficiente para gerar os gráficos.',
        ),
        'channels_download': (
            ' &nbsp;·&nbsp; <a href="channels.csv">Tabela CSV</a>'
            if (folder / 'channels.csv').exists()
            else ''
        ),
        'block_comment': (
            ''
            if collection.get('mode') == 'resting'
            else comment_section(report.get('comment', collection.get('comment', '')))
        ),
    }
    participant = collection.get('participant', {})
    fields['registration'] = ' &nbsp;·&nbsp; '.join(
        escape(f'{label}: {participant[key]}')
        for key, label in [
            ('group', 'Grupo'),
            ('cap', 'Touca'),
            ('age', 'Idade'),
            ('sex', 'Sexo'),
            ('laterality', 'Lateralidade'),
            ('collection_date', 'Data da sessão'),
            ('experimenter', 'Experimentador/a'),
        ]
        if participant.get(key)
    )
    cards = [('Acurácia média · CV', fields['accuracy'], 'Predições fora do fold de treino'),
             ('Acurácia balanceada', fields['balanced'], 'Mesmo peso para cada classe'),
             ('Voto por tentativa', fields['trial_accuracy'], 'Decisão agregada no trial'),
             ('Aquisição registrada', fields['fs']+' <small>Hz</small>',
              f"{fields['channel_count']} canais · {fields['samples']} amostras")]
    if collection.get('mode') == 'resting':
        conditions = report.get('resting_state') or []
        cards = [(row['label'], f"{row['duration_s']:.1f} s" if row['duration_s'] is not None else '—',
                  f"Planejado: {row['planned_s']:g} s") for row in conditions] + cards[-1:]
    fields['metrics'] = '<div class="metrics">'+''.join(
        f'<div class="metric"><span>{escape(label)}</span><strong>{value}</strong><small>{escape(caption)}</small></div>'
        for label, value, caption in cards)+'</div>'
    fields['model_figures'] = '' if collection.get('mode') == 'resting' else (
        '<div class="grid"><section class="card"><div class="eyebrow">Representação visual</div><h2>Mapa PCA</h2>'
        '<p class="muted">Check do bloco no espaço de duas dimensões. A classificação usa o espaço completo.</p>'
        +fields['pca']+'</section><section class="card"><div class="eyebrow">Desempenho fora do treino</div>'
        '<h2>Matriz de confusão</h2><p class="muted">Linhas: classe real. Colunas: classe predita. '
        'À direita, porcentagem dentro de cada classe real.</p>'+fields['matrix']+'</section></div>')
    template = (Path(__file__).parent / 'report_template.html').read_text(encoding='utf-8')
    template = re.sub(r'\{\{(\w+)\}\}', lambda match: fields[match.group(1)], template)
    path = folder / 'report.html'
    path.write_text(template, encoding='utf-8')
    return path
