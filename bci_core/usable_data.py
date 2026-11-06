"""Segmentos utilizáveis sem alterar os arquivos ou reconstruir timestamps."""
import numpy as np

from .data import bandpass_causal


def valid_segments(times, values, fs, intervals=()):
    times = np.asarray(times, float)
    if not len(times):
        return []
    valid = np.isfinite(times) & np.isfinite(values).all(axis=1)
    # Conserva a ordem original; não utiliza duplicatas ou trechos em recuo.
    peak = np.maximum.accumulate(np.where(np.isfinite(times), times, -np.inf))
    valid &= times > np.r_[-np.inf, peak[:-1]]
    for interval in intervals:
        valid &= ~((times >= interval['start_lsl_s']) & (times <= interval['end_lsl_s']))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(times[indices]) > max(.02, 1.5/fs)))+1
    return [(int(chunk[0]), int(chunk[-1])+1) for chunk in np.split(indices, cuts) if len(chunk)]


def filter_segments(times, values, fs, order, band, intervals):
    segments = valid_segments(times, values, fs, intervals)
    filtered = np.full(values.shape, np.nan, dtype=float)
    for start, stop in segments:
        filtered[start:stop] = bandpass_causal(values[start:stop], fs, order, band)
    usable = sum(stop-start for start, stop in segments)
    return filtered, segments, dict(total_samples=len(times), usable_samples=usable,
                                    excluded_samples=len(times)-usable, segments=len(segments))
