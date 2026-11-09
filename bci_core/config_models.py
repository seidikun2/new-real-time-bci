from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import yaml
from .storage import DEFAULT_DATA_ROOT


@dataclass
class ExperimentConfig:
    subject_id: str
    session_id: int
    log_root: str
    exp_name: str
    session_type: str


@dataclass
class LSLConfig:
    marker_name: str
    marker_type: str
    signal_name: str
    signal_type: str
    decoder_name: str = "Signal"
    decoder_type: str = "BCI"


@dataclass
class RuntimeConfig:
    simulate_signal: bool
    source: Optional[str] = None


@dataclass
class ProtocolConfig:
    stage: str = "single_target"
    start_phase: str = "auto"
    training_mode: str = "target_vs_rest"
    online_target: str = "left"
    block_end_code: int = 99
    block_end_label: str = "BLOCK_END"
    motor_session_type: str = "EM_treino"
    imagery_session_type: str = "IM_treino"
    min_cv_accuracy: Optional[float] = None
    online_modes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SimSignalConfig:
    fs: float
    channels: int
    chunk: int
    noise_std: float


@dataclass
class SimBurstConfig:
    freq: float
    amp: float
    dur: float
    taper_frac: float


@dataclass
class SimProfilesConfig:
    left_mi: Tuple[float, float]
    both_mi: Tuple[float, float]
    right_mi: Tuple[float, float]


@dataclass
class CodesConfig:
    left_mi: int
    both_mi: int
    right_mi: int
    rest_stim: int
    attempt: int
    block_end: int
    code_map: Dict[int, str]


@dataclass
class DecoderConfig:
    epoch_s: float
    step_s: float
    band_hz: List[float]
    filter_order: int
    outlet_name: str
    outlet_type: str
    lsl_rate_hz: float


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    lsl: LSLConfig
    runtime: RuntimeConfig
    protocol: ProtocolConfig
    sim_signal: SimSignalConfig
    sim_burst: SimBurstConfig
    sim_profiles: SimProfilesConfig
    codes: CodesConfig
    decoder: DecoderConfig

    fs_hz: float
    n_channels: int
    trial_duration_s: float
    trial_offset_s: float
    window_s: float
    step_s: float
    filter_mode: str
    bp_order: int
    bp_band_hz: List[float]
    select_by: str
    index_base: int
    select_channels: List[Any]
    feature_space: str
    classifier: str
    pca_dim: int
    svc_c: float
    cv_splits: int
    rng_seed: int

    _raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Configuração única; os nomes básicos do YAML continuam iguais."""
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    values = {
        "fs_hz": 256.0, "n_channels": 16, "trial_duration_s": 3.75,
        "trial_offset_s": 0.0, "window_s": 1.0, "step_s": 0.05,
        "filter_mode": "causal", "bp_order": 3, "bp_band_hz": [5.0, 40.0],
        "select_by": "index", "index_base": 1, "select_channels": [],
        "feature_space": "riemann", "classifier": "svm_linear", "pca_dim": 2,
        "svc_c": 1.0, "cv_splits": 5, "rng_seed": 42,
    }
    values.update({key: raw[key] for key in values if key in raw})
    if values["filter_mode"] != "causal" or values["feature_space"] != "riemann":
        raise ValueError("Use filter_mode: causal e feature_space: riemann (PCA é apenas visual).")
    if values["classifier"] != "svm_linear":
        raise ValueError("O classificador suportado é svm_linear.")
    fs, window, step = values["fs_hz"], values["window_s"], values["step_s"]
    low, high = values["bp_band_hz"]
    if not (0 < low < high < fs / 2 and window > 0 and step > 0):
        raise ValueError("Verifique frequência, banda, janela e passo.")
    if values["trial_offset_s"] < 0 or values["trial_offset_s"] + window > values["trial_duration_s"]:
        raise ValueError("A janela precisa caber dentro do ATTEMPT.")
    if values["n_channels"] < 2 or values["pca_dim"] != 2 or values["cv_splits"] < 2:
        raise ValueError("Use >=2 canais, pca_dim: 2 e cv_splits >=2.")
    exp_values = dict(subject_id="TEST", session_id=1, log_root=DEFAULT_DATA_ROOT, exp_name="TEST", session_type="IM_treino")
    exp_values.update(raw.get("experiment", {}))
    exp = ExperimentConfig(**exp_values)
    for name in (exp.subject_id, exp.exp_name, exp.session_type):
        if not name or any(c in name for c in '/\\') or name in {".", ".."}:
            raise ValueError("Identificadores devem ser nomes simples, sem separadores de pasta.")
    p = raw.get("protocol", {})
    stage = p.get("stage", "single_target")
    if stage not in {"single_target", "two_legs"}:
        raise ValueError("Etapa deve ser single_target ou two_legs.")
    target = p.get("online_target", "right")
    if target not in {"left", "right"}:
        raise ValueError("online_target deve ser left ou right.")
    protocol_values = {key: value for key, value in p.items() if key in ProtocolConfig.__dataclass_fields__}
    protocol_values["training_mode"] = "target_vs_rest" if stage == "single_target" else "left_right_vs_rest"
    protocol_values["online_target"] = target
    protocol = ProtocolConfig(**protocol_values)
    lsl = LSLConfig("GrazMI_Markers", "Markers", "gHIamp_EEG", "EEG")
    for key, value in raw.get("lsl", {}).items():
        if key not in LSLConfig.__dataclass_fields__:
            raise ValueError(f"Campo LSL desconhecido: {key}")
        setattr(lsl, key, value)
    cmap = {1: "BASELINE", 2: "ATTENTION", 3: "LEFT_MI_STIM", 4: "RIGHT_MI_STIM", 5: "ATTEMPT", 6: "REST", 7: "BOTH_MI_STIM", 8: "REST_STIM", 99: "BLOCK_END"}
    from .resting_state import RESTING_CODES
    cmap.update({code: label for label, code in RESTING_CODES.items()})
    cmap.update({int(k): v for k, v in raw.get("codes", {}).get("map", {}).items()})
    cr = raw.get("codes", {})
    codes = CodesConfig(
        left_mi=int(cr.get("left_mi", 3)), right_mi=int(cr.get("right_mi", 4)),
        both_mi=7, rest_stim=int(cr.get("rest_stim", 8)), attempt=int(cr.get("attempt", 5)),
        block_end=int(cr.get("block_end", 99)), code_map=cmap,
    )
    if cmap.get(protocol.block_end_code) != protocol.block_end_label:
        raise ValueError("Código/label BLOCK_END inconsistente.")
    cfg = AppConfig(
        experiment=exp, lsl=lsl, runtime=RuntimeConfig(**raw.get("runtime", {"simulate_signal": True})), protocol=protocol,
        sim_signal=SimSignalConfig(fs, values["n_channels"], raw.get("sim_chunk", 10), raw.get("sim_noise_std", 0.8)),
        sim_burst=SimBurstConfig(raw.get("sim_burst_freq_hz", 10.0), raw.get("sim_burst_amp", 1.5), values["trial_duration_s"], raw.get("sim_taper_frac", 0.2)),
        sim_profiles=SimProfilesConfig(tuple(raw.get("sim_profile_left_mi", [1.0, .3])), (1., 1.), tuple(raw.get("sim_profile_right_mi", [.3, 1.0]))),
        codes=codes,
        decoder=DecoderConfig(window, step, values["bp_band_hz"], values["bp_order"], raw.get("decoder_outlet_name", lsl.decoder_name), raw.get("decoder_outlet_type", lsl.decoder_type), raw.get("decoder_lsl_rate_hz", 64.)),
        **values, _raw_config=raw,
    )

    if cfg.runtime.source:
        from .acquisition import set_source
        set_source(cfg, cfg.runtime.source)
    return cfg


def expected_signal_channels(cfg):
    from .acquisition import source_name
    if source_name(cfg) in {'eego', 'lsl'}:
        return cfg.n_channels
    if source_name(cfg) == 'simulator':
        return cfg.sim_signal.channels
    from .input_hiamp import get_hiamp_acquisition_channels, get_output_indices
    return len(get_output_indices(cfg, get_hiamp_acquisition_channels(cfg)))
