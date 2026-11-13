"""OAS → espaço tangente completo → SVM linear. PCA é somente visual."""

import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import tangent_space
from pyriemann.utils.mean import mean_covariance
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .acquisition import recording_config
from .analysis import analysis_options, training_step
from .storage import external_path, canonical_block
from .class_schema import class_details, training_label_map
from .data import (
    read_signal_csv,
    read_markers_csv,
    select_channel_indices,
    bandpass_causal,
    attempts_by_class,
    epoch_trials_sliding,
    timestamp_integrity,
)
from .representation_feedback import prepare_training_representation
from .usable_data import filter_segments


def safe_cv_splits(y, groups, requested):
    counts = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    return min(int(requested), min(counts)) if len(counts) >= 2 else 0


def cross_validate(Xw, y, groups, svc_c, n_splits, seed=42):
    """Todos os parâmetros aprendidos são ajustados somente no fold de treino.

    Acurácia por janela e por tentativa (voto majoritário) são separadas.
    As janelas sobrepostas de um trial nunca entram nos dois lados do fold.
    """
    covariances = Covariances('oas').transform(Xw)
    return _cross_validate_covariances(covariances, y, groups, svc_c, n_splits, seed)


def _cross_validate_covariances(covariances, y, groups, svc_c, n_splits, seed=42, progress=None):
    splits = safe_cv_splits(y, groups, n_splits)
    if splits < 2:
        return {
            'status': 'insufficient_trials',
            'reason': 'Mínimo de 2 tentativas por classe para CV.',
            'splits': 0,
            'acc_mean': None,
            'accs': [],
        }
    predictions = np.full(len(y), -1)
    scores = []
    splitter = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(covariances, y, groups), 1):
        if progress:
            progress(f'Validação por tentativa · divisão {fold}/{splits}')
        if len(np.unique(y[train])) != len(np.unique(y)):
            return {
                'status': 'insufficient_trials',
                'reason': 'Fold de treino sem todas as classes.',
                'splits': 0,
                'acc_mean': None,
                'accs': [],
            }

        cov = covariances[train]
        reference = mean_covariance(cov)
        clf = SVC(kernel='linear', C=svc_c).fit(tangent_space(cov, reference), y[train])
        features = tangent_space(covariances[test], reference)
        predictions[test] = clf.predict(features)
        scores.append(float(accuracy_score(y[test], predictions[test])))
    trial_true, trial_pred = [], []
    for group in np.unique(groups):
        mask = groups == group
        labels, counts = np.unique(predictions[mask], return_counts=True)
        trial_true.append(int(y[mask][0]))
        trial_pred.append(int(labels[counts.argmax()]))
    return {
        'status': 'ok',
        'splits': splits,
        'acc_mean': float(np.mean(scores)),
        'acc_std': float(np.std(scores)),
        'accs': scores,
        'window_accuracy': float(accuracy_score(y, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
        'trial_accuracy': float(accuracy_score(trial_true, trial_pred)),
        'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        'trial_confusion_matrix': confusion_matrix(
            trial_true, trial_pred, labels=np.unique(y)
        ).tolist(),
        'class_order': np.unique(y).tolist(),
        'trials': len(trial_true),
        'windows': len(y),
    }


def fit_final_model(Xw, y, pca_dim=2, svc_c=1.0, seed=42):
    cov = Covariances('oas').transform(Xw)
    return _fit_covariances(cov, y, pca_dim, svc_c, seed)


def _fit_covariances(cov, y, pca_dim=2, svc_c=1.0, seed=42, progress=None):
    if progress:
        progress('Modelo final · referência Riemanniana e características completas')
    reference = mean_covariance(cov)
    features = tangent_space(cov, reference)
    if progress:
        progress('Modelo final · SVM e probabilidades')
    classifier = SVC(kernel='linear', C=svc_c, probability=True, random_state=seed).fit(features, y)
    if progress:
        progress('PCA · projeção visual do experimentador')
    pca = PCA(n_components=min(pca_dim, *features.shape), random_state=seed).fit(features)
    return reference, pca, classifier, pca.transform(features)


def run_calibration(cfg, markers_file, signal_file, recording=None, progress=None):
    with threadpool_limits(limits=analysis_options(cfg)['blas_threads']):
        return _run_calibration(
            cfg, markers_file, signal_file, recording, progress or (lambda _: None)
        )


def _run_calibration(cfg, markers_file, signal_file, recording, progress):
    cfg = recording_config(cfg, signal_file)
    if getattr(cfg, '_block_mode', None) == 'resting':
        raise ValueError(
            'Registro de olhos não é treino de imagética motora. Selecione dados de EM/IM ou online.'
        )
    audit = getattr(cfg, '_timing_audit', {})
    if getattr(cfg, '_audit_only', False):
        raise ValueError(
            'Este CSV é a cópia de auditoria. Selecione o CSV de trabalho *_signal_*.csv.'
        )
    if audit.get('training_error'):
        raise ValueError(audit['training_error'])
    if recording is None:
        progress('Lendo EEG e marcadores para o treino')
        recording = (
            *read_signal_csv(signal_file),
            *read_markers_csv(markers_file, cfg.codes.code_map),
        )
    times, X, names, tm, labels = recording
    if len(times) < 2:
        raise ValueError('EEG sem amostras suficientes para treinar.')
    timing = timestamp_integrity(times)
    recovery = audit.get('recovery', {})
    exclude_segments = recovery.get('policy') == 'exclude_segments'
    if timing['error'] and not exclude_segments:
        raise ValueError(timing['error'])
    if not np.isfinite(X).all() and not exclude_segments:
        raise ValueError('EEG contém NaN/Inf: consulte o relatório antes de treinar.')
    if not np.isfinite(tm).all() or np.any(np.diff(tm) < 0):
        raise ValueError('Timestamps de marcadores inválidos.')
    fs = cfg.fs_hz
    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if not len(intervals) or abs(1 / np.median(intervals) - fs) / fs > 0.05:
        raise ValueError('Taxa de amostragem não corresponde à configuração.')
    indices = select_channel_indices(cfg.select_by, cfg.select_channels, names, cfg.index_base)
    selected = [names[i] for i in indices]
    segments, usable = None, {}
    progress('Filtrando o sinal e separando tentativas válidas')
    if exclude_segments:
        Xf, segments, usable = filter_segments(
            times,
            X[:, indices],
            fs,
            cfg.bp_order,
            cfg.bp_band_hz,
            recovery.get('excluded_intervals', []),
        )
    else:
        Xf = bandpass_causal(X[:, indices], fs, cfg.bp_order, cfg.bp_band_hz)
    label_map = training_label_map(labels, cfg.protocol.training_mode, cfg.protocol.online_target)
    attempts = attempts_by_class(tm, labels, list(label_map))
    step = training_step(cfg)
    Xw, y, groups, _ = epoch_trials_sliding(
        times,
        Xf,
        attempts,
        label_map,
        fs,
        cfg.window_s,
        step,
        cfg.trial_duration_s,
        cfg.trial_offset_s,
        segments=segments,
    )
    if set(y) != set(label_map.values()):
        raise ValueError('Uma classe ficou sem janelas válidas; modelo não foi treinado.')
    progress(f'Covariâncias OAS · {len(y)} janelas, {len(selected)} canais')
    covariances = Covariances('oas').transform(Xw)
    # OAS é calculado por janela, sem aprender parâmetros entre tentativas.
    # Cada fold ainda aprende sua referência e seu SVM somente no treino.
    del Xw, Xf
    cv = _cross_validate_covariances(
        covariances,
        y,
        groups,
        cfg.svc_c,
        cfg.cv_splits,
        cfg.rng_seed,
        progress,
    )
    reference, pca, classifier, projected = _fit_covariances(
        covariances, y, cfg.pca_dim, cfg.svc_c, cfg.rng_seed, progress
    )
    signal_path = external_path(signal_file)
    run_id = signal_path.stem.rsplit('_signal_', 1)[-1]
    model_dir = (
        signal_path.parent
        / '_model'
        / f'{run_id}_{cfg.protocol.stage}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    )
    model_dir = external_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    details = class_details(label_map, cfg.codes.code_map)
    progress('PCA · mapa e contornos de densidade')
    representation = prepare_training_representation(
        projected,
        y,
        cfg,
        str(model_dir),
        class_labels={d['model_class']: d['display_label'] for d in details},
        class_details=details,
    )
    progress('Salvando o modelo e os resultados da validação')
    for filename, artifact in [
        ('riemann_mean.pkl', reference),
        ('pca.pkl', pca),
        ('classifier.pkl', classifier),
    ]:
        with (model_dir / filename).open('wb') as stream:
            pickle.dump(artifact, stream)
    source = signal_path.with_suffix('.json')
    source_config = (
        json.loads(source.read_text(encoding='utf-8')).get('config', {}) if source.exists() else {}
    )
    warnings = list(audit.get('warnings', []))
    if usable:
        usable.update(
            windows_used=len(y),
            trials_used=len(np.unique(groups)),
            trials_total=sum(len(events) for events in attempts.values()),
        )
        if usable['excluded_samples'] or usable['segments'] > 1:
            warnings.append(
                f"Treino: {usable['excluded_samples']}/{usable['total_samples']} amostras excluídas; "
                f"{usable['segments']} segmentos, {usable['windows_used']} janelas de "
                f"{usable['trials_used']}/{usable['trials_total']} tentativas utilizáveis. "
                'Filtro reiniciado entre segmentos; janelas não atravessam trechos excluídos.'
            )
    meta = {
        'schema_version': 2,
        'feature_space': 'riemann',
        'classifier_input': 'tangent_space',
        'feature_dimensions': int(classifier.n_features_in_),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'markers_file': str(Path(markers_file).resolve()),
        'signal_file': str(signal_path),
        'protocol_stage': cfg.protocol.stage,
        'training_mode': cfg.protocol.training_mode,
        'online_target': cfg.protocol.online_target,
        'session_type': canonical_block(
            source_config.get('experiment', {}).get('session_type', cfg.experiment.session_type)
        ),
        'channels_selected': selected,
        'channel_indices': indices,
        'fs_hz': fs,
        'signal_unit': getattr(cfg, '_recorded_unit', 'uV'),
        'participant': asdict(cfg.experiment),
        'timing_warnings': list(audit.get('warnings', [])),
        'warnings': warnings,
        'usable_data': usable,
        'timing_policy': {
            'method': audit.get('method'),
            'validation_scope': audit.get('validation_scope'),
            'strict_raw_timestamps': audit.get('options', {}).get('strict_raw_timestamps'),
            'recovery_policy': recovery.get('policy'),
            'max_bad_duration_s': recovery.get('max_bad_duration_s'),
        },
        'filter': {'band_hz': cfg.bp_band_hz, 'order': cfg.bp_order},
        'window_s': cfg.window_s,
        'step_s': cfg.step_s,
        'training_step_s': step,
        'training_windows': len(y),
        'trial_duration_s': cfg.trial_duration_s,
        'classes_map': label_map,
        'classes': details,
        'cv': cv,
        'representation': representation['meta'],
        'pca_check_png': str(model_dir / 'pca_check.png'),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'pca_map_json': str(model_dir / 'pca_map.json'),
        'pca_map_png': str(model_dir / 'pca_map.png'),
        'pca_train_xlim': [representation['meta']['x_min'], representation['meta']['x_max']],
        'pca_train_ylim': [representation['meta']['y_min'], representation['meta']['y_max']],
        'probabilities_note': 'SVC probability=True; CV report evaluates predictions, not probability calibration.',
    }
    (model_dir / 'model_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[treino] {selected}: {classifier.n_features_in_} dimensões; CV: {cv}')
    return {'model_dir': str(model_dir), 'meta': meta, 'accs_cv': cv['accs']}
