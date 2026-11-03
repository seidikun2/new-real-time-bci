"""Relatório por bloco: integridade, canais, espectro e artefatos por segundo.

Os limites são heurísticos e dependem da unidade do amplificador. O relatório
não remove canais/tentativas e não substitui inspeção do sinal bruto.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .config_models import expected_signal_channels
from .acquisition import source_name, recording_config
from .report import collection_info, write_report
from .data import read_signal_csv, read_markers_csv


def inspect_quality(times, X, names, marker_times, labels, cfg):
    q = cfg._raw_config.get('quality', {})
    issues = []
    observations = list(getattr(cfg, '_stream_observations', []))
    simulated = source_name(cfg) == 'simulator'
    if simulated:
        observations.append('Sinal simulado: ruído branco e bursts sintéticos não reproduzem um EEG. Métricas espectrais são informativas, sem alertas fisiológicos.')
    unit = 'a.u.' if simulated else getattr(cfg, '_recorded_unit', q.get('unit', 'uV'))
    absolute = unit == 'uV'
    peak_limit = q.get('peak_to_peak_limit', 200.) if absolute else float('inf')
    flat_limit = q.get('flat_std_limit', .01) if absolute else 0.
    fs = cfg.fs_hz
    if len(times) < 2:
        raise ValueError('Gravação sem amostras suficientes para análise.')
    differences = np.diff(times)
    good_dt = differences[np.isfinite(differences) & (differences > 0)]
    estimated_fs = float(1 / np.median(good_dt)) if len(good_dt) else None
    gaps = int(np.sum(differences > 1.5 / fs))
    missing = int(np.sum(np.maximum(np.round(good_dt * fs) - 1, 0)))
    if not np.isfinite(times).all() or np.any(differences <= 0):
        issues.append('ERRO: timestamps EEG não finitos, duplicados ou fora de ordem.')
    if gaps:
        issues.append(f'Descontinuidade/jitter: {gaps} intervalos longos (~{missing} amostras potencialmente ausentes; estimativa por timestamps).')
    if estimated_fs is None or abs(estimated_fs-fs) / fs > .05:
        issues.append(f'Taxa EEG estimada {estimated_fs} Hz difere de {fs} Hz.')
    expected_channels = getattr(cfg, '_recorded_channel_count', expected_signal_channels(cfg))
    if len(names) != expected_channels:
        issues.append(f'ERRO: {len(names)} canais; configuração espera {expected_channels}.')
    if len(marker_times) == 0 or not np.isfinite(marker_times).all() or np.any(np.diff(marker_times) < 0):
        issues.append('ERRO: marcadores ausentes ou timestamps inválidos.')
    counts = Counter(labels)
    if counts.get('ATTEMPT', 0) != sum(counts.get(label, 0) for label in ['LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM']):
        issues.append('Número de ATTEMPTs diferente do número de cues.')
    if counts.get('REST', 0) != counts.get('ATTEMPT', 0) or counts.get('BASELINE', 0) != 1:
        issues.append('Sequência incompleta: confira BASELINE e pausas REST.')
    if counts.get('BLOCK_END', 0) != 1:
        issues.append('Bloco incompleto ou duplicado: esperado um BLOCK_END.')
    required = ['LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM']
    expected = cfg._raw_config.get('protocol', {}).get('trials_per_class', 15)
    if any(counts[label] != expected for label in required):
        issues.append(f'Quantidade de cues diferente do plano ({expected} por classe).')
    last_cue = None
    attempts = []
    for i, (ts, label) in enumerate(zip(marker_times, labels)):
        if label in required:
            if last_cue is not None:
                issues.append('Cue sem ATTEMPT antes do próximo cue.')
            last_cue = label
        elif label == 'ATTEMPT':
            if last_cue is None:
                issues.append('ATTEMPT sem cue correspondente.')
            attempts.append(float(ts))
            if i + 1 == len(labels) or labels[i+1] != 'REST' or abs(marker_times[i+1] - ts - cfg.trial_duration_s) > .15:
                issues.append('ATTEMPT com duração incorreta ou sem REST seguinte.')
            last_cue = None
        elif label not in cfg.codes.code_map.values():
            issues.append(f'Marcador desconhecido: {label}.')
    if last_cue is not None:
        issues.append('Último cue sem ATTEMPT.')
    finite_times = times[np.isfinite(times)]
    if len(finite_times) and any(t < finite_times[0] or t + cfg.trial_duration_s > finite_times[-1] + 1/fs for t in attempts):
        issues.append('Tentativa fora da cobertura EEG: início/fim truncado.')
    clean = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    frequencies, psd = signal.welch(clean, fs=fs, nperseg=min(len(clean), max(8, round(fs*2))), axis=0)
    power = np.maximum(psd.sum(axis=0), np.finfo(float).tiny)
    line_mask = (abs(frequencies-50) <= 2) | (abs(frequencies-60) <= 2)
    line_ratio = psd[line_mask].sum(axis=0) / power
    high_ratio = psd[frequencies >= 40].sum(axis=0) / power
    med = np.median(clean, axis=0)
    robust_scale = 1.4826 * np.median(abs(clean-med), axis=0)
    win = max(2, round(fs))
    windows = [X[i:i+win] for i in range(0, len(X), win) if len(X[i:i+win]) >= 2]
    # Cada janela é marcada por amplitude, flatline, pico robusto ou NaN/Inf.
    bad = []
    for window in windows:
        finite = np.isfinite(window).all(axis=0)
        safe = np.nan_to_num(window, nan=0., posinf=0., neginf=0.)
        flag = (~finite | (np.ptp(safe, axis=0) > peak_limit) |
                (np.std(safe, axis=0) <= flat_limit) |
                (np.max(abs(safe-med), axis=0) > q.get('max_robust_z', 8.) * np.maximum(robust_scale, np.finfo(float).tiny)))
        bad.append(flag)
    bad = np.asarray(bad, bool)
    rows = []
    for i, name in enumerate(names):
        reasons = []
        std = float(np.std(clean[:, i]))
        invalid = int(np.sum(~np.isfinite(X[:, i])))
        bad_fraction = float(np.mean(bad[:, i]))
        if invalid:
            reasons.append(f'{invalid} NaN/Inf')
        if std <= flat_limit:
            reasons.append('canal plano')
        if bad_fraction > q.get('max_bad_window_fraction', .2):
            reasons.append('muitas janelas suspeitas')
        if not simulated and line_ratio[i] > q.get('max_line_noise_ratio', .25):
            reasons.append('rede 50/60 Hz')
        if not simulated and high_ratio[i] > q.get('max_high_frequency_ratio', .35):
            reasons.append('alta frequência / possível EMG')
        # Longos patamares podem indicar desconexão ou saturação do conversor.
        flat_fraction = float(np.mean(np.diff(clean[:, i]) == 0))
        if flat_fraction > .1:
            reasons.append('patamares/repetição de amostras')
        rows.append(dict(channel=name, std=std, peak_to_peak=float(np.ptp(clean[:, i])),
                         invalid_samples=invalid, bad_window_fraction=bad_fraction,
                         line_noise_ratio=float(line_ratio[i]), high_frequency_ratio=float(high_ratio[i]),
                         repeated_fraction=flat_fraction, reasons=reasons))
    noisy = [row['channel'] for row in rows if row['reasons']]
    if noisy:
        issues.append('Inspecione os canais: ' + ', '.join(noisy))
    for i in range(len(names)):
        for j in range(i):
            if np.array_equal(clean[:, i], clean[:, j]):
                issues.append(f'Canais idênticos: {names[j]} e {names[i]}.')
    report = dict(status='attention' if issues else 'ok', issues=list(dict.fromkeys(issues)),
                  estimated_fs_hz=estimated_fs, samples=len(times), gaps=gaps, estimated_missing_samples=missing,
                  marker_counts=dict(counts), channels=rows, unit=unit,
                  source=source_name(cfg), simulated=simulated, observations=observations,
                  thresholds=q, notes='Heurísticas: sem exclusão automática; confira unidade/ganho. CV não comprova ausência de artefatos.')
    return report, clean, frequencies, psd, bad


def run_check_data(cfg, mode='train', markers_file=None, signal_file=None, save_png=True, evaluation=None):
    cfg = recording_config(cfg, signal_file) if signal_file else cfg
    if markers_file is None or signal_file is None:
        raise ValueError('Selecione explicitamente os CSVs do bloco.')
    if Path(signal_file).exists():
        times, X, names = read_signal_csv(signal_file)
    else:
        times, X, names = np.array([]), np.empty((0, 0)), []
    tm, labels = read_markers_csv(markers_file, cfg.codes.code_map)
    folder = Path(signal_file).parent / '_qc' / Path(signal_file).stem.rsplit('_signal_', 1)[-1]
    folder.mkdir(parents=True, exist_ok=True)
    collection = collection_info(cfg, signal_file)
    if len(times) < 2:
        report = {'status': 'error', 'issues': ['EEG vazio ou insuficiente; confira aquisição antes de repetir.'],
                  'evaluation': evaluation, 'collection': collection, 'samples': len(times),
                  'estimated_fs_hz': None, 'channels': [], 'source': source_name(cfg),
                  'thresholds': cfg._raw_config.get('quality', {}), 'unit': 'uV'}
        (folder / 'quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        path = write_report(cfg, report, evaluation, signal_file, folder)
        return {'report_path': str(path.resolve()), **report}
    report, clean, freq, psd, bad = inspect_quality(times, X, names, tm, labels, cfg)
    report['evaluation'] = evaluation
    report['issues'].extend((evaluation or {}).get('warnings', []))
    if report['issues']:
        report['status'] = 'attention'
    report['collection'] = collection
    (folder / 'quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(3, 1, figsize=(13, max(12, len(names)*.34)), layout='constrained')
    origin = times[np.isfinite(times)][0] if np.isfinite(times).any() else 0.
    step = max(1, len(times) // 15000)
    scales = np.maximum(1.4826 * np.median(abs(clean-np.median(clean, axis=0)), axis=0), 1e-9)
    normalized = np.clip((clean-np.median(clean, axis=0))/scales, -6, 6)
    for i, name in enumerate(names):
        axes[0].plot(times[::step]-origin, normalized[::step, i]+i*10, lw=.4)
    axes[0].set_yticks(np.arange(len(names))*10, names)
    axes[0].set_title('EEG bruto: escala robusta por canal, picos recortados em ±6 para facilitar leitura')
    for ts, label in zip(tm, labels):
        if label in {'LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM'}:
            axes[0].axvline(ts-origin, alpha=.25, color={'LEFT_MI_STIM':'blue','RIGHT_MI_STIM':'red','REST_STIM':'gray'}[label])
    axes[0].set_xlabel('Tempo desde primeira amostra (s); azul LEFT, vermelho RIGHT, cinza REST_STIM')
    for i, name in enumerate(names):
        axes[1].semilogy(freq, np.maximum(psd[:, i], 1e-16), label=name)
    for line in (50, 60):
        axes[1].axvline(line, color='gray', ls='--')
    axes[1].set(xlabel='Frequência (Hz)', ylabel=f"PSD ({'escala original' if report['unit'] == 'native' else report['unit']}²/Hz)", title='Espectro bruto por canal · referências de 50/60 Hz')
    axes[1].legend(ncol=min(len(names), 8), fontsize=7)
    axes[2].imshow(bad.T, aspect='auto', interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
    axes[2].set_yticks(range(len(names)), names)
    axes[2].set(xlabel='Janela de ~1 s (índice de amostras; consulte lacunas)', title='Artefatos: vermelho = amplitude, flatline, pico robusto ou NaN/Inf')
    fig.savefig(folder / 'overview.png', dpi=130)
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(report['channels']).to_csv(folder / 'channels.csv', index=False)
    path = write_report(cfg, report, evaluation, signal_file, folder)
    print(f'[qualidade] {report["status"]}: {path}')
    return {'report_path': str(path.resolve()), **report}
