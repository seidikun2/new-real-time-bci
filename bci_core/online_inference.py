# -*- coding: utf-8 -*-
"""Inferência causal contínua: SVM no espaço completo; PCA somente no dashboard."""

import os, csv, json, time, threading, pickle, datetime as dt
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
from scipy import signal
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pylsl import StreamInfo, StreamOutlet, local_clock

from .config_models import AppConfig
from .acquisition import stream_layout
from .receive_data_log import resolve_signal, make_log_dir
from .data import select_channel_indices as select_channels
from .representation_feedback import transform_online_representation
from .feedback_contract import SIGNAL_CHANNELS, classifier_probabilities, signal_vector


def log(msg: str) -> None:
    print(f"[decoder] {msg}")


def _artifact_paths(model_ref: str) -> tuple[str, str, str, str]:
    if os.path.isdir(model_ref) and os.path.exists(os.path.join(model_ref, "classifier.pkl")):
        return (
            os.path.join(model_ref, "riemann_mean.pkl"),
            os.path.join(model_ref, "pca.pkl"),
            os.path.join(model_ref, "classifier.pkl"),
            os.path.join(model_ref, "model_meta.json"),
        )
    return (
        model_ref + "_best_c_mean.pkl",
        model_ref + "_dim_red.pkl",
        model_ref + "_classifier.pkl",
        model_ref + "_meta.json",
    )


def load_artifacts(prefix_or_dir: str):
    cmean_p, pca_p, clf_p, _ = _artifact_paths(prefix_or_dir)
    for path in (cmean_p, pca_p, clf_p):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefato não encontrado: {path}")

    with open(cmean_p, "rb") as f: cmean = pickle.load(f)
    with open(pca_p, "rb") as f: pca = pickle.load(f)
    with open(clf_p, "rb") as f: clf = pickle.load(f)

    log("Artefatos carregados:")
    log(f"  cmean: {cmean_p}")
    log(f"  pca  : {pca_p}")
    log(f"  clf  : {clf_p}")
    return cmean, pca, clf


def read_model_meta(prefix_or_dir: str) -> Dict[str, Any]:
    _, _, _, meta_p = _artifact_paths(prefix_or_dir)
    try:
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if os.path.isdir(prefix_or_dir):
            meta["pca_map_json"] = os.path.abspath(os.path.join(prefix_or_dir, "pca_map.json"))
            meta["pca_map_png"] = os.path.abspath(os.path.join(prefix_or_dir, "pca_map.png"))
        return meta
    except Exception:
        return {}


def make_outlet_unified(
    name: str,
    stype: str = "BCI",
    srate: float = 0.0,
    representation_meta: Optional[Dict[str, Any]] = None,
    feedback_mode: str = "pca",
) -> StreamOutlet:
    """Cria o contrato fixo de 8 canais do feedback contínuo."""
    info = StreamInfo(name, stype, len(SIGNAL_CHANNELS), srate, "float32")
    root = info.desc()

    desc = root.append_child("channels")
    for lab in SIGNAL_CHANNELS:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lab)
        if lab.startswith("p_"):
            ch.append_child_value("unit", "probability")
            ch.append_child_value("type", "BCIProbability")
        elif lab.startswith("active_"):
            ch.append_child_value("unit", "binary")
            ch.append_child_value("type", "BCIClassActive")
        else:
            ch.append_child_value("unit", "a.u.")
            ch.append_child_value("type", "BCIRepresentation")

    rep = representation_meta or {}
    if isinstance(rep, dict):
        if rep.get("representation_id"):
            root.append_child_value("representation_id", str(rep["representation_id"]))
        if rep.get("representation_name"):
            root.append_child_value("representation_name", str(rep["representation_name"]))
        if rep.get("transform_type"):
            root.append_child_value("representation_transform", str(rep["transform_type"]))

    root.append_child_value("decoder_semantics", ",".join(SIGNAL_CHANNELS))
    root.append_child_value("feedback_mode", str(feedback_mode))
    root.append_child_value("feedback_enabled", "false")
    root.append_child_value("representation_audience", "experimenter")

    return StreamOutlet(info)


def design_bandpass(fs: float, order: int, band) -> np.ndarray:
    low, high = float(band[0]), float(band[1])
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Banda inválida {band} para fs={fs}")
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def window_to_feature(X_win_CxT: np.ndarray, cmean) -> np.ndarray:
    cov = Covariances("oas").transform(X_win_CxT[None, ...])
    ts  = tangent_space(cov, cmean)
    return ts[0]


def timestamp_iso_from_lsl(lsl_time_s: float, unix_offset: float) -> str:
    return dt.datetime.fromtimestamp(lsl_time_s + unix_offset).isoformat(timespec="microseconds")


def open_csv_logger(cfg: AppConfig, mode: str, feedback_mode: str = "pca"):
    out_dir = make_log_dir(cfg, mode)
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    core = f"{cfg.experiment.subject_id}_{cfg.experiment.exp_name}_S{cfg.experiment.session_id}_{cfg.experiment.session_type}_decoder"
    path = os.path.join(out_dir, f"{core}_{stamp}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    # pca1_raw/pca2_raw ficam no log para diagnóstico e retrocompatibilidade.
    # rep1/rep2 são exatamente as coordenadas enviadas ao Unity.
    w.writerow([
        "iso_time", "lsl_time_s", "recv_time_s",
        "pca1_raw", "pca2_raw", "rep1", "rep2",
        "p_left", "p_rest", "p_right",
        "active_left", "active_rest", "active_right",
        "feedback_mode",
    ])
    log(f"Log de inferência: {path}")
    return f, w, path


def class_outputs(clf, feat: np.ndarray, model_meta: Dict[str, Any]):
    """Probabilidades semânticas diretamente de predict_proba + classes ativas."""
    return classifier_probabilities(clf, feat, model_meta)


def run_realtime_decoder(cfg: AppConfig, mode: str = "online", model_prefix: Optional[str] = None, stop_event: Optional[threading.Event] = None, feedback_mode: str = "none", ready_event=None):
    if stop_event is None:
        stop_event = threading.Event()

    decfg = cfg.decoder
    mcfg  = cfg

    if not model_prefix:
        raise ValueError("Escolha explicitamente um modelo antes do online.")
    prefix = model_prefix
    cmean, pca, clf = load_artifacts(prefix)
    meta = read_model_meta(prefix)

    if meta.get("classifier_input") != "tangent_space":
        raise ValueError("Modelo antigo classifica em PCA. Retreine os dados para usar o espaço completo.")
    inlet = resolve_signal(cfg, stop_event)
    info = inlet.info()
    source_indices, source_scales, stream_meta = stream_layout(cfg, info)
    fs = stream_meta['fs_hz']
    if meta.get('signal_unit', 'uV') != stream_meta['unit']:
        raise ValueError('O modelo e o stream usam escalas diferentes. Retreine os dados recebidos nesta sessão.')
    ch_all = stream_meta['recorded_channels']
    log(f"Fs stream={fs:.2f} Hz | canais={ch_all}")

    train_ch_names = meta["channels_selected"]
    sel_idx = select_channels("name", train_ch_names, ch_all)
    ch_sel = [ch_all[i] for i in sel_idx]
    log(f"Canais usados: {ch_sel}")

    filt_meta = meta.get("filter", {}) if isinstance(meta, dict) else {}
    band_hz   = filt_meta.get("band_hz", getattr(mcfg, "bp_band_hz", decfg.band_hz))
    order     = int(filt_meta.get("order", getattr(mcfg, "bp_order", decfg.filter_order)))
    window_s  = float(meta.get("window_s", meta.get("epoch_s", getattr(mcfg, "window_s", decfg.epoch_s))))
    step_s    = float(meta.get("step_s", getattr(decfg, "step_s", 0.05)))
    train_fs  = float(meta.get("fs_hz", fs)) if isinstance(meta, dict) else fs
    if abs(fs - train_fs) > 1e-3:
        raise ValueError(f"Fs stream={fs:.2f} Hz diferente do modelo={train_fs:.2f} Hz.")

    sos = design_bandpass(fs, order, band_hz)
    zf = np.zeros((sos.shape[0], 2, len(sel_idx)))

    win_n = int(round(window_s * fs))
    hop_n = max(1, int(round(step_s * fs)))
    if win_n < 8:
        raise ValueError("Janela muito curta.")
    log(f"Online contínuo: janela={window_s:.2f}s ({win_n} samples), step={step_s:.3f}s ({hop_n} samples), filtro causal")

    rep_meta = meta.get("representation", {}) if isinstance(meta, dict) else {}
    outlet = make_outlet_unified(
        name=decfg.outlet_name,
        stype=decfg.outlet_type,
        srate=1.0 / step_s,
        representation_meta=rep_meta,
        feedback_mode=feedback_mode,
    )

    if rep_meta:
        log(
            "Representação feedback: "
            f"id={rep_meta.get('representation_id', 'sem-id')} | "
            f"nome={rep_meta.get('representation_name', 'representation')} | "
            f"transform={rep_meta.get('transform_type', 'identity')}"
        )
    else:
        log("Modelo antigo sem bloco 'representation': usando PCA identity nos canais rep1/rep2.")
    log("Classificação no espaço tangente completo; PCA apenas para o experimentador.")
    fcsv, wcsv, _ = open_csv_logger(cfg, mode, feedback_mode=feedback_mode)
    unix_offset = time.time() - local_clock()

    buf_X = deque(maxlen=win_n)
    buf_t = deque(maxlen=win_n)
    n_samples = 0
    next_compute = win_n
    previous_stamp = None

    try:
        while not stop_event.is_set():
            data, ts = inlet.pull_chunk(timeout=0.2, max_samples=max(32, 4 * hop_n))
            if not ts:
                continue

            X = np.asarray(data, dtype=float)[:, source_indices] * source_scales
            if X.ndim != 2 or X.shape[1] <= max(sel_idx):
                continue
            if not np.isfinite(X).all():
                raise ValueError("EEG contém NaN/Inf. Bloco interrompido; inspecione os canais.")
            if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                raise ValueError("Timestamps EEG inválidos durante inferência.")
            if (previous_stamp is not None and ts[0] - previous_stamp > max(.1, 1.5/fs)) or np.any(np.diff(ts) > max(.1, 1.5/fs)):
                raise ValueError("Lacuna no EEG durante inferência. Inspecione aquisição antes de continuar.")
            previous_stamp = ts[-1]
            X_sel = X[:, sel_idx]

            # Mesmo tipo de filtro do treino: causal contínuo, preservando estado.
            Xf, zf = signal.sosfilt(sos, X_sel, axis=0, zi=zf)

            for x_f, t_lsl in zip(Xf, ts):
                buf_X.append(x_f)
                buf_t.append(float(t_lsl))
                n_samples += 1

                if n_samples >= next_compute and len(buf_X) >= win_n:
                    X_win = np.vstack(buf_X)  # (T,C), já filtrado
                    feat = window_to_feature(X_win.T, cmean)
                    visual = pca.transform(feat[None, :])[0]
                    pca_dim = int(getattr(pca, "n_components_", getattr(pca, "n_components", 2)) or 2)
                    p1_raw = float(visual[0]) if pca_dim >= 1 else 0.0
                    p2_raw = float(visual[1]) if pca_dim >= 2 else 0.0

                    # MESMO transform usado para gerar as densidades do treino.
                    # Coordenadas servem apenas ao dashboard do experimentador.
                    rep = transform_online_representation(visual, meta)
                    rep1 = float(rep[0])
                    rep2 = float(rep[1])

                    probs, active = class_outputs(clf, feat, meta)
                    vec = signal_vector(rep1, rep2, probs, active)
                    outlet.push_sample(vec, timestamp=float(buf_t[-1]))
                    if ready_event is not None:
                        ready_event.set()

                    t_out = float(buf_t[-1])
                    recv_time_s = time.time()
                    iso = timestamp_iso_from_lsl(t_out, unix_offset)
                    wcsv.writerow([
                        iso,
                        f"{t_out:.9f}",
                        f"{recv_time_s:.6f}",
                        f"{p1_raw:.6f}",
                        f"{p2_raw:.6f}",
                        f"{rep1:.6f}",
                        f"{rep2:.6f}",
                        f"{probs['LEFT_MI_STIM']:.6f}",
                        f"{probs['REST_STIM']:.6f}",
                        f"{probs['RIGHT_MI_STIM']:.6f}",
                        f"{active['LEFT_MI_STIM']:.0f}",
                        f"{active['REST_STIM']:.0f}",
                        f"{active['RIGHT_MI_STIM']:.0f}",
                        str(feedback_mode),
                    ])
                    next_compute += hop_n
    except KeyboardInterrupt:
        log("Ctrl+C recebido.")
    finally:
        inlet.close_stream()
        try:
            fcsv.close()
        except Exception:
            pass
        log("Decoder encerrado.")
