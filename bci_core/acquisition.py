"""Fontes de EEG: simulador, g.HIamp local ou streamer LSL externo (eego)."""
import copy
from dataclasses import asdict

import numpy as np
from pylsl import resolve_streams

from .data import lsl_channel_names, select_channel_indices

SOURCES = {'simulator': 'Simulação', 'hiamp': 'g.HIamp', 'eego': 'ANT Neuro · eego mylab', 'lsl': 'Outro stream LSL'}


def source_name(cfg):
    return cfg.runtime.source or ('simulator' if cfg.runtime.simulate_signal else 'hiamp')


def set_source(cfg, source, fs_hz=None, channels=None, stream_name=None):
    if source not in SOURCES:
        raise ValueError('Fonte de EEG desconhecida.')
    cfg.__dict__.pop('_signal_source_id', None)
    cfg.runtime.source = source
    cfg.runtime.simulate_signal = source == 'simulator'
    options = cfg._raw_config.get(source, {})
    if source in {'eego', 'lsl'}:
        cfg.fs_hz = float(fs_hz if fs_hz is not None else options.get('fs_hz', 500. if source == 'eego' else cfg.fs_hz))
        cfg.n_channels = int(channels if channels is not None else options.get('n_channels', 64 if source == 'eego' else cfg.n_channels))
        cfg.lsl.signal_name = stream_name if stream_name is not None else (options.get('signal_name') or '')
        cfg.lsl.signal_type = options.get('signal_type', 'EEG')
    else:
        cfg.fs_hz = float(fs_hz if fs_hz is not None else cfg._raw_config.get('fs_hz', 256.))
        cfg.n_channels = int(channels if channels is not None else cfg._raw_config.get('n_channels', 16))
        cfg.lsl.signal_name = cfg._raw_config.get('lsl', {}).get('signal_name', 'gHIamp_EEG')
        cfg.sim_signal.fs = cfg.fs_hz
        cfg.sim_signal.channels = cfg.n_channels
    if not np.isfinite(cfg.fs_hz) or cfg.fs_hz <= 0 or cfg.n_channels < 2:
        raise ValueError('Informe uma frequência positiva e pelo menos dois canais.')
    if not 0 < cfg.bp_band_hz[0] < cfg.bp_band_hz[1] < cfg.fs_hz / 2:
        raise ValueError('A banda do filtro deve estar abaixo de metade da frequência de aquisição.')
    cfg._raw_config['runtime'] = asdict(cfg.runtime)
    if source in {'eego', 'lsl'}:
        cfg._raw_config.setdefault(source, {}).update(fs_hz=cfg.fs_hz, n_channels=cfg.n_channels, signal_name=cfg.lsl.signal_name)


def receive_indices(cfg, names):
    options = cfg._raw_config.get(source_name(cfg), {})
    return select_channel_indices(options.get('receive_by', 'index'), options.get('receive_channels', []), names, options.get('index_base', 1))


def stream_layout(cfg, info):
    """Valida a configuração real do transmissor; nunca reamostra silenciosamente."""
    names = lsl_channel_names(info)
    fs = float(info.nominal_srate())
    if fs <= 0 or abs(fs-cfg.fs_hz) > .01:
        raise ValueError(f'O stream anuncia {fs:g} Hz; esperado {cfg.fs_hz:g} Hz. Ajuste o transmissor ou a frequência na tela inicial.')
    from .config_models import expected_signal_channels
    expected = expected_signal_channels(cfg)
    if len(names) != expected:
        raise ValueError(f'O stream anuncia {len(names)} canais; esperado {expected}. Confira a tela inicial e o transmissor.')
    indices = receive_indices(cfg, names)
    if len(indices) < 2:
        raise ValueError('Selecione pelo menos dois canais para a pipeline.')
    options = cfg._raw_config.get(source_name(cfg), {})
    channel = info.desc().child('channels').child('channel')
    units = []
    for i in range(len(names)):
        units.append(channel.child_value('unit') or '')
        channel = channel.next_sibling()
    declared = options.get('unit', 'auto')
    factors = []
    for i in indices:
        unit = units[i] if declared == 'auto' else declared
        normalized = unit.strip().lower().replace('µ', 'u').replace('μ', 'u')
        scales = {'uv': 1., 'microvolts': 1., 'microvolt': 1., 'mv': 1e3, 'millivolts': 1e3, 'v': 1e6, 'volts': 1e6, 'volt': 1e6}
        if source_name(cfg) in {'eego', 'lsl'} and normalized not in scales:
            raise ValueError(f'Unidade desconhecida em {names[i]}: {unit!r}. Configure unit: uV, mV ou V na fonte após conferir o transmissor.')
        factors.append(scales.get(normalized, 1.))
    metadata = {'name': info.name(), 'type': info.type(), 'source_id': info.source_id(),
                'fs_hz': fs, 'source_channels': len(names), 'source_channel_names': names,
                'recorded_channels': [names[i] for i in indices], 'source_units': units,
                'scale_to_uV': factors, 'unit': 'uV', 'source': source_name(cfg)}
    return indices, np.asarray(factors), metadata


def discover_signals(timeout=1.):
    streams = resolve_streams(wait_time=timeout)
    return [dict(name=s.name(), type=s.type(), fs_hz=s.nominal_srate(), n_channels=s.channel_count(), source_id=s.source_id())
            for s in streams if s.type().lower() == 'eeg']


def recording_config(cfg, signal_file):
    """Relatórios/retreino usam a frequência e a origem do bloco selecionado."""
    import json
    from pathlib import Path
    sidecar = Path(signal_file).with_suffix('.json')
    if not sidecar.exists():
        return cfg
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    stream = meta.get('stream') or {}
    if not stream:
        return cfg
    result = copy.deepcopy(cfg)
    result.fs_hz = stream['fs_hz']
    result.n_channels = len(stream['recorded_channels'])
    result.runtime.source = stream['source']
    result.runtime.simulate_signal = stream['source'] == 'simulator'
    result._raw_config.setdefault(stream['source'], {})['receive_channels'] = []
    result._recorded_channel_count = len(stream['recorded_channels'])
    recorded = meta.get('config', {})
    effective = meta.get('effective_config', {})
    result.trial_duration_s = effective.get('trial_duration_s', recorded.get('trial_duration_s', result.trial_duration_s))
    original_protocol = recorded.get('protocol', {})
    for key in ['trials_per_class', 'baseline_s', 'attention_s', 'cue_s', 'rest_s']:
        if key in original_protocol:
            result._raw_config.setdefault('protocol', {})[key] = original_protocol[key]
    return result
