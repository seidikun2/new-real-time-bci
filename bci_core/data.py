"""Leitura, pareamento e janelas compartilhados pelo treino e pelo relatório."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .class_schema import STIM_CLASS_ORDER


def find_marker_signal_pairs(folder):
    pairs = []
    for marker in Path(folder).glob('*_markers_*.csv'):
        signal_path = marker.with_name(marker.name.replace('_markers_', '_signal_'))
        if signal_path.exists():
            pairs.append((str(marker.resolve()), str(signal_path.resolve())))
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)


def read_markers_csv(path, code_map):
    frame = pd.read_csv(path)
    times = pd.to_numeric(frame['lsl_time_s'], errors='coerce').to_numpy(float)
    labels = frame['label'].fillna('').str.strip().str.upper().tolist() if 'label' in frame else [code_map.get(int(c), 'UNKNOWN') for c in frame['code']]
    return times, labels


def read_signal_csv(path):
    frame = pd.read_csv(path)
    channels = [str(c) for c in frame if c not in {'iso_time', 'lsl_time_s', 'local_recv_s'}]
    if not channels:
        raise ValueError('CSV sem canais ch1..chN.')
    return (pd.to_numeric(frame['lsl_time_s'], errors='coerce').to_numpy(float),
            frame[channels].apply(pd.to_numeric, errors='coerce').to_numpy(float), channels)


def select_channel_indices(select_by, selection, names, index_base=1):
    if not selection:
        return list(range(len(names)))
    if select_by == 'name':
        lookup = {name.lower(): i for i, name in enumerate(names)}
        try:
            indices = [lookup[str(name).lower()] for name in selection]
        except KeyError as exc:
            raise ValueError(f'Canal ausente: {exc}') from exc
    else:
        indices = [int(i) - index_base for i in selection]
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(names) for i in indices):
        raise ValueError(f'Seleção de canais inválida: {selection}')
    return indices


def bandpass_causal(data, fs, order, band):
    sos = signal.butter(order, band, btype='bandpass', fs=fs, output='sos')
    return signal.sosfilt(sos, data, axis=0)


def attempts_by_class(times, labels, model_labels):
    attempts = {label: [] for label in model_labels}
    last = None
    for time, label in zip(times, labels):
        if label in STIM_CLASS_ORDER:
            last = label if label in attempts else None
        elif label == 'ATTEMPT':
            if last is not None:
                attempts[last].append(float(time))
            last = None
        elif label in {'REST', 'BASELINE', 'BLOCK_END'}:
            last = None
    return attempts


def epoch_trials_sliding(t_sig, X, attempts, label_map, fs, window_s, step_s,
                         trial_duration_s, trial_offset_s, seed=42):
    """Nunca atravessa lacunas, fim de arquivo ou limite de uma tentativa."""
    win = round(window_s * fs)
    hop = max(1, round(step_s * fs))
    epochs, targets, groups, centers = [], [], [], []
    trial = 0
    for label, events in attempts.items():
        for event in events:
            start_time = event + trial_offset_s
            end_time = event + trial_duration_s
            start = int(np.searchsorted(t_sig, start_time))
            stop = int(np.searchsorted(t_sig, end_time))
            for i in range(start, min(stop, len(t_sig)) - win + 1, hop):
                times = t_sig[i:i+win]
                if times[-1] > end_time or np.any(np.diff(times) > max(.02, 1.5 / fs)):
                    continue
                epochs.append(X[i:i+win].T)
                targets.append(label_map[label])
                groups.append(trial)
                centers.append(float(times[win // 2] - event))
            trial += 1
    if not epochs:
        raise ValueError('Nenhuma janela válida dentro dos ATTEMPTs. Confira tempos e perdas de EEG.')
    return np.stack(epochs), np.array(targets), np.array(groups), np.array(centers)


def lsl_channel_names(info):
    channel = info.desc().child('channels').child('channel')
    names = []
    for i in range(info.channel_count()):
        names.append(channel.child_value('label') or f'ch{i+1}')
        channel = channel.next_sibling()
    if len(set(name.lower() for name in names)) != len(names):
        raise ValueError('Nomes de canais LSL duplicados.')
    return names
