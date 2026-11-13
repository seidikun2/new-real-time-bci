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
from .storage import external_path
from .signal_preview import signal_snapshot, display_envelope
from .data import read_signal_csv, read_markers_csv, timestamp_integrity
from .resting_state import inspect_resting


def inspect_quality(times, X, names, marker_times, labels, cfg, progress=None):
    progress = progress or (lambda _: None)
    q = cfg._raw_config.get('quality', {})
    issues = []
    observations = list(getattr(cfg, '_stream_observations', []))
    audit = getattr(cfg, '_timing_audit', {})
    if audit.get('training_error'):
        issues.append(audit['training_error'])
    issues.extend(audit.get('warnings', []))
    if audit:
        summaries = []
        for label, key in [('Publicador', 'source'), ('Sincronizado', 'clock_synced'), ('Trabalho', 'working')]:
            if key in audit:
                item = audit[key]
                summaries.append(f"{label}: {item['backward_pairs']} recuos (até {item['largest_backward_s']*1000:.3f} ms)")
        observations.append(f"Horários: {audit.get('method', 'não registrado')}. " + '; '.join(summaries) + '.')
        if audit.get('active'):
            if audit.get('recovery'):
                observations.append('A acomodação é mostrada separadamente. Episódios pontuais marcam trechos excluídos; '
                                    'a interrupção depende da persistência durante o protocolo/teste.')
            else:
                observations.append('Os máximos desde o início incluem a acomodação. A decisão temporal do treino considera o período do protocolo/teste, mostrado separadamente abaixo.')
        observations.append(f"Maior pausa de recepção: {audit.get('max_receive_pause_s', 0):.3f} s; "
                            f"maior atraso de gravação: {audit.get('max_write_lag_s', 0):.3f} s. "
                            f"Arquivo de trabalho: {audit.get('received_samples', 0)} amostras recebidas, "
                            f"{audit.get('written_samples', 0)} gravadas. "
                            'A auditoria original usa um inlet independente, sem pareamento por índice.')
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
    timing = timestamp_integrity(times)
    if timing['error']:
        prefix = ('Trechos com horários inválidos são excluídos do treino; CSV original preservado. '
                  if audit.get('recovery', {}).get('policy') == 'exclude_segments' else 'ERRO: ')
        issues.append(prefix + timing['error'])
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
    conditions = None
    if getattr(cfg, '_block_mode', None) == 'resting':
        resting_issues, conditions = inspect_resting(marker_times, labels, times, cfg)
        issues.extend(resting_issues)
    else:
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
    progress('Qualidade · espectro de todos os canais')
    clean = X if np.isfinite(X).all() else np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    frequencies, psd = channel_spectra(clean, fs)
    power = np.maximum(psd.sum(axis=0), np.finfo(float).tiny)
    line_mask = (abs(frequencies-50) <= 2) | (abs(frequencies-60) <= 2)
    line_ratio = psd[line_mask].sum(axis=0) / power
    high_ratio = psd[frequencies >= 40].sum(axis=0) / power
    progress('Qualidade · amplitude, artefatos e canais planos')
    med = np.median(clean, axis=0)
    robust_scale = 1.4826 * np.median(abs(clean-med), axis=0)
    win = max(2, round(fs))
    windows = [X[i:i+win] for i in range(0, len(X), win) if len(X[i:i+win]) >= 2]
    # Cada janela é marcada por amplitude, flatline, pico robusto ou NaN/Inf.
    bad = []
    for window in windows:
        finite = np.isfinite(window).all(axis=0)
        safe = np.nan_to_num(window, nan=0., posinf=0., neginf=0.)
        span = np.ptp(safe, axis=0)
        flag = (~finite | (span > peak_limit) |
                (span == 0) | (np.std(safe, axis=0) <= flat_limit) |
                (np.max(abs(safe-med), axis=0) > q.get('max_robust_z', 8.) * np.maximum(robust_scale, np.finfo(float).tiny)))
        bad.append(flag)
    bad = np.asarray(bad, bool)
    rows = []
    for i, name in enumerate(names):
        reasons = []
        span = float(np.ptp(clean[:, i]))
        # Uma constante não nula pode ter std residual por arredondamento.
        std = float(np.std(clean[:, i])) if span else 0.
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
        rows.append(
            dict(
                channel=name,
                std=std,
                peak_to_peak=span,
                median=float(med[i]),
                robust_scale=float(robust_scale[i]),
                invalid_samples=invalid,
                bad_window_fraction=bad_fraction,
                line_noise_ratio=float(line_ratio[i]),
                high_frequency_ratio=float(high_ratio[i]),
                repeated_fraction=flat_fraction,
                reasons=reasons,
            )
        )
    noisy = [row['channel'] for row in rows if row['reasons']]
    if noisy:
        issues.append('Inspecione os canais: ' + ', '.join(noisy))
    progress('Qualidade · canais idênticos')
    probe = clean[:: max(1, len(clean) // 128)]
    for i in range(len(names)):
        for j in range(i):
            # A prévia só descarta pares diferentes; a confirmação usa todo o sinal.
            if np.array_equal(probe[:, i], probe[:, j]) and np.array_equal(
                clean[:, i], clean[:, j]
            ):
                issues.append(f'Canais idênticos: {names[j]} e {names[i]}.')
    report = dict(status='attention' if issues else 'ok', issues=list(dict.fromkeys(issues)),
                  timestamp_integrity=timing,
                  timing_audit=audit,
                  realtime=getattr(cfg, '_realtime_audit', {}), resting_state=conditions,
                  estimated_fs_hz=estimated_fs, samples=len(times), gaps=gaps, estimated_missing_samples=missing,
                  marker_counts=dict(counts), channels=rows, unit=unit,
                  source=source_name(cfg), simulated=simulated, observations=observations,
                  thresholds=q, notes='Heurísticas de amplitude/espectro não excluem canais automaticamente. '
                  'Trechos temporalmente inválidos ou não finitos são excluídos conforme a política registrada. '
                  'Confira unidade/ganho; CV não comprova ausência de artefatos.')
    return report, clean, frequencies, psd, bad


def channel_spectra(values, fs):
    """Welch em lotes pequenos: mesmas amostras, menor pico de memória."""
    spectra = []
    for first in range(0, values.shape[1], 8):
        frequencies, power = signal.welch(
            values[:, first : first + 8],
            fs=fs,
            nperseg=min(len(values), max(8, round(fs * 2))),
            axis=0,
        )
        spectra.append(power)
    return frequencies, np.concatenate(spectra, axis=1)


def run_check_data(
    cfg,
    mode='train',
    markers_file=None,
    signal_file=None,
    save_png=True,
    evaluation=None,
    recording=None,
    progress=None,
):
    progress = progress or (lambda _: None)
    cfg = recording_config(cfg, signal_file) if signal_file else cfg
    if markers_file is None or signal_file is None:
        raise ValueError('Selecione explicitamente os CSVs do bloco.')
    if recording is None:
        progress('Lendo EEG e marcadores para os checks')
        samples = (
            read_signal_csv(signal_file)
            if Path(signal_file).exists()
            else (np.array([]), np.empty((0, 0)), [])
        )
        recording = (*samples, *read_markers_csv(markers_file, cfg.codes.code_map))
    times, X, names, tm, labels = recording
    folder = external_path(signal_file).parent / '_qc' / Path(signal_file).stem.rsplit('_signal_', 1)[-1]
    folder = external_path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    collection = collection_info(cfg, signal_file)
    if len(times) < 2:
        report = {
            'status': 'error',
            'issues': ['EEG vazio ou insuficiente; confira aquisição antes de repetir.'],
            'evaluation': evaluation,
            'collection': collection,
            'samples': len(times),
            'participant': collection.get('participant', {}),
            'comment': collection.get('comment', ''),
            'realtime': getattr(cfg, '_realtime_audit', {}),
            'estimated_fs_hz': None,
            'channels': [],
            'source': source_name(cfg),
            'thresholds': cfg._raw_config.get('quality', {}),
            'unit': 'uV',
            'analysis_timing_file': getattr(cfg, '_analysis_timing_file', None),
        }
        (folder / 'quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        path = write_report(cfg, report, evaluation, signal_file, folder)
        return {'report_path': str(path.resolve()), **report}
    report, clean, freq, psd, bad = inspect_quality(times, X, names, tm, labels, cfg, progress)
    report['evaluation'] = evaluation
    report['issues'].extend((evaluation or {}).get('warnings', []))
    report['issues'] = list(dict.fromkeys(report['issues']))
    if report['issues']:
        report['status'] = 'attention'
    report['collection'] = collection
    report['participant'] = collection.get('participant', {})
    report['comment'] = collection.get('comment', '')
    report['analysis_timing_file'] = getattr(cfg, '_analysis_timing_file', None)
    progress('Gerando o snapshot e as métricas do bloco')
    snapshot = signal_snapshot(times, X, names)
    if snapshot is not None:
        snapshot_file = folder / 'signal_snapshot.npz'
        np.savez_compressed(snapshot_file, **snapshot)
        report['snapshot_file'] = str(snapshot_file.resolve())
    (folder / 'quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    progress('Gerando as figuras de sinal, espectro e artefatos')
    plt.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(3, 1, figsize=(13, max(12, len(names)*.34)), layout='constrained')
    origin = times[np.isfinite(times)][0] if np.isfinite(times).any() else 0.
    center = np.array([row['median'] for row in report['channels']])
    scales = np.maximum([row['robust_scale'] for row in report['channels']], 1e-9)
    display_times, display_values = display_envelope(times, clean, points=3000)
    normalized = np.clip((display_values - center) / scales, -6, 6)
    for i, name in enumerate(names):
        axes[0].plot(display_times - origin, normalized[:, i] + i * 10, lw=0.4, color='#334155')
    axes[0].set_yticks(np.arange(len(names))*10, names)
    axes[0].set_title('EEG bruto: envelope mínimo/máximo, escala robusta por canal, picos recortados em ±6')
    if report.get('resting_state') is not None:
        for condition, color in zip(report['resting_state'], ['#b9d9f0', '#d9dee6']):
            if condition['start_lsl_s'] is not None and condition['end_lsl_s'] is not None:
                axes[0].axvspan(condition['start_lsl_s']-origin, condition['end_lsl_s']-origin,
                               color=color, alpha=.45, label=condition['label'])
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(loc='upper right')
        axes[0].set_xlabel('Tempo desde primeira amostra (s); faixas indicam olhos abertos e fechados')
    else:
        for ts, label in zip(tm, labels):
            if label in {'LEFT_MI_STIM', 'RIGHT_MI_STIM', 'REST_STIM'}:
                axes[0].axvline(ts-origin, alpha=.25, color={'LEFT_MI_STIM':'blue','RIGHT_MI_STIM':'red','REST_STIM':'gray'}[label])
        axes[0].set_xlabel('Tempo desde primeira amostra (s); azul LEFT, vermelho RIGHT, cinza REST_STIM')
    peak = np.maximum(psd.max(axis=0), np.finfo(float).tiny)
    relative = 10*np.log10(np.maximum(psd/peak, np.finfo(float).tiny))
    image = axes[1].imshow(relative.T, aspect='auto', origin='lower', cmap='Greys', vmin=-60, vmax=0,
                           extent=[float(freq[0]), float(freq[-1]), -.5, len(names)-.5])
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set(xlabel='Frequência (Hz)', ylabel='Canais', title='Espectro por canal · dB relativos ao pico de cada canal')
    for line in (50, 60):
        axes[1].axvline(line, color='#64748b', ls='--', lw=.7)
    fig.colorbar(image, ax=axes[1], label='dB relativos')
    axes[2].imshow(bad.T, aspect='auto', interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
    axes[2].set_yticks(range(len(names)), names)
    axes[2].set(xlabel='Janela de ~1 s (índice de amostras; consulte lacunas)', title='Artefatos: vermelho = amplitude, flatline, pico robusto ou NaN/Inf')
    fig.savefig(folder / 'overview.png', dpi=130)
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(report['channels']).to_csv(folder / 'channels.csv', index=False)
    progress('Montando o relatório, PCA e matriz de confusão')
    path = write_report(cfg, report, evaluation, signal_file, folder)
    print(f'[qualidade] {report["status"]}: {path}')
    return {'report_path': str(path.resolve()), **report}
