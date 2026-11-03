"""Relatório HTML offline com visão do modelo antes dos detalhes de canais."""
import html
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .class_schema import display_name
from .acquisition import SOURCES


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
    rows = []
    for row in report['channels']:
        suspect = bool(row['reasons'])
        values = [row['channel'], f"{row['std']:.3g}", f"{row['peak_to_peak']:.3g}",
                  f"{row['bad_window_fraction']:.1%}", f"{row['line_noise_ratio']:.1%}",
                  f"{row['high_frequency_ratio']:.1%}", ' · '.join(row['reasons']) or 'Sem alerta']
        rows.append('<tr class="'+('suspect' if suspect else '')+'">'+''.join(f'<td>{escape(value)}</td>' for value in values)+'</tr>')
    status = 'Revisar bloco' if report['issues'] else ('Simulação íntegra' if report.get('simulated') else 'Sem alertas')
    available = cv.get('acc_mean') is not None
    summary = 'Validação cruzada por tentativa; as janelas do mesmo trial permanecem no mesmo fold.' if available else evaluation.get('error', cv.get('reason', 'Aguardando treino após a inspeção dos dados.'))
    fs = report['estimated_fs_hz']
    fields = {
        'session': escape(cfg.experiment.session_type), 'subject': escape(cfg.experiment.subject_id),
        'session_id': escape(cfg.experiment.session_id), 'source': escape(SOURCES.get(report.get('source'), 'EEG')),
        'filename': escape(Path(signal_file).name), 'status': status,
        'status_class': 'warn' if report['issues'] else 'good',
        'samples': f"{report['samples']:,}".replace(',', '.'), 'fs': f'{fs:.1f}' if fs else '—',
        'channel_count': str(len(report['channels'])), 'accuracy': metric(cv.get('acc_mean')),
        'balanced': metric(cv.get('balanced_accuracy')), 'trial_accuracy': metric(cv.get('trial_accuracy')),
        'summary': escape(summary), 'notes': notes, 'alerts': alerts, 'channel_rows': ''.join(rows),
        'pca': figure(pca_available, 'pca_check.png', 'Mapa PCA disponível após o treino deste bloco.'),
        'matrix': figure(plot_matrix, 'confusion_matrix.png', 'Matriz indisponível até concluir a validação cruzada.'),
        'thresholds': escape(json.dumps(report['thresholds'], ensure_ascii=False)),
        'unit': escape(report['unit']),
    }
    template = (Path(__file__).parent / 'report_template.html').read_text(encoding='utf-8')
    template = re.sub(r'\{\{(\w+)\}\}', lambda match: fields[match.group(1)], template)
    path = folder / 'report.html'
    path.write_text(template, encoding='utf-8')
    return path
