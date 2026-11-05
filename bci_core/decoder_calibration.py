"""OAS → espaço tangente completo → SVM linear. PCA é somente visual."""
import json
import pickle
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

from .acquisition import recording_config
from .storage import external_path
from .class_schema import class_details, training_label_map
from .data import (read_signal_csv, read_markers_csv, select_channel_indices,
                   bandpass_causal, attempts_by_class, epoch_trials_sliding, timestamp_integrity)
from .representation_feedback import prepare_training_representation


def safe_cv_splits(y, groups, requested):
    counts = [len(np.unique(groups[y == c])) for c in np.unique(y)]
    return min(int(requested), min(counts)) if len(counts) >= 2 else 0


def cross_validate(Xw, y, groups, svc_c, n_splits, seed=42):
    """Todos os parâmetros aprendidos são ajustados somente no fold de treino.

    Acurácia por janela e por tentativa (voto majoritário) são separadas.
    As janelas sobrepostas de um trial nunca entram nos dois lados do fold.
    """
    splits = safe_cv_splits(y, groups, n_splits)
    if splits < 2:
        return {'status': 'insufficient_trials', 'reason': 'Mínimo de 2 tentativas por classe para CV.',
                'splits': 0, 'acc_mean': None, 'accs': []}
    predictions = np.full(len(y), -1)
    scores = []
    splitter = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(Xw, y, groups):
        if len(np.unique(y[train])) != len(np.unique(y)):
            return {'status': 'insufficient_trials', 'reason': 'Fold de treino sem todas as classes.',
                    'splits': 0, 'acc_mean': None, 'accs': []}
        cov = Covariances('oas').transform(Xw[train])
        reference = mean_covariance(cov)
        clf = SVC(kernel='linear', C=svc_c).fit(tangent_space(cov, reference), y[train])
        features = tangent_space(Covariances('oas').transform(Xw[test]), reference)
        predictions[test] = clf.predict(features)
        scores.append(float(accuracy_score(y[test], predictions[test])))
    trial_true, trial_pred = [], []
    for group in np.unique(groups):
        mask = groups == group
        labels, counts = np.unique(predictions[mask], return_counts=True)
        trial_true.append(int(y[mask][0]))
        trial_pred.append(int(labels[counts.argmax()]))
    return {'status': 'ok', 'splits': splits, 'acc_mean': float(np.mean(scores)),
            'acc_std': float(np.std(scores)), 'accs': scores,
            'window_accuracy': float(accuracy_score(y, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
            'trial_accuracy': float(accuracy_score(trial_true, trial_pred)),
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
            'trial_confusion_matrix': confusion_matrix(trial_true, trial_pred, labels=np.unique(y)).tolist(),
            'class_order': np.unique(y).tolist(), 'trials': len(trial_true), 'windows': len(y)}


def fit_final_model(Xw, y, pca_dim=2, svc_c=1., seed=42):
    cov = Covariances('oas').transform(Xw)
    reference = mean_covariance(cov)
    features = tangent_space(cov, reference)
    classifier = SVC(kernel='linear', C=svc_c, probability=True, random_state=seed).fit(features, y)
    pca = PCA(n_components=min(pca_dim, *features.shape), random_state=seed).fit(features)
    return reference, pca, classifier, pca.transform(features)


def run_calibration(cfg, markers_file, signal_file):
    cfg = recording_config(cfg, signal_file)
    times, X, names = read_signal_csv(signal_file)
    tm, labels = read_markers_csv(markers_file, cfg.codes.code_map)
    if len(times) < 2:
        raise ValueError('EEG sem amostras suficientes para treinar.')
    timing = timestamp_integrity(times)
    if timing['error']:
        raise ValueError(timing['error'])
    if not np.isfinite(X).all():
        raise ValueError('EEG contém NaN/Inf: consulte o relatório antes de treinar.')
    if not np.isfinite(tm).all() or np.any(np.diff(tm) < 0):
        raise ValueError('Timestamps de marcadores inválidos.')
    fs = cfg.fs_hz
    if abs(1 / np.median(np.diff(times)) - fs) / fs > .05:
        raise ValueError('Taxa de amostragem não corresponde à configuração.')
    indices = select_channel_indices(cfg.select_by, cfg.select_channels, names, cfg.index_base)
    selected = [names[i] for i in indices]
    Xf = bandpass_causal(X[:, indices], fs, cfg.bp_order, cfg.bp_band_hz)
    label_map = training_label_map(labels, cfg.protocol.training_mode, cfg.protocol.online_target)
    attempts = attempts_by_class(tm, labels, list(label_map))
    Xw, y, groups, _ = epoch_trials_sliding(times, Xf, attempts, label_map, fs, cfg.window_s,
                                           cfg.step_s, cfg.trial_duration_s, cfg.trial_offset_s)
    if set(y) != set(label_map.values()):
        raise ValueError('Uma classe ficou sem janelas válidas; modelo não foi treinado.')
    cv = cross_validate(Xw, y, groups, cfg.svc_c, cfg.cv_splits, cfg.rng_seed)
    reference, pca, classifier, projected = fit_final_model(Xw, y, cfg.pca_dim, cfg.svc_c, cfg.rng_seed)
    signal_path = external_path(signal_file)
    run_id = signal_path.stem.rsplit('_signal_', 1)[-1]
    model_dir = signal_path.parent / '_model' / f'{run_id}_{cfg.protocol.stage}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    model_dir = external_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    details = class_details(label_map, cfg.codes.code_map)
    representation = prepare_training_representation(
        projected, y, cfg, str(model_dir),
        class_labels={d['model_class']: d['display_label'] for d in details}, class_details=details)
    for filename, artifact in [('riemann_mean.pkl', reference), ('pca.pkl', pca), ('classifier.pkl', classifier)]:
        with (model_dir / filename).open('wb') as stream:
            pickle.dump(artifact, stream)
    source = signal_path.with_suffix('.json')
    source_config = json.loads(source.read_text(encoding='utf-8')).get('config', {}) if source.exists() else {}
    meta = {
        'schema_version': 2, 'feature_space': 'riemann', 'classifier_input': 'tangent_space',
        'feature_dimensions': int(classifier.n_features_in_),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'markers_file': str(Path(markers_file).resolve()), 'signal_file': str(signal_path),
        'protocol_stage': cfg.protocol.stage, 'training_mode': cfg.protocol.training_mode,
        'online_target': cfg.protocol.online_target, 'session_type': source_config.get('experiment', {}).get('session_type', cfg.experiment.session_type),
        'channels_selected': selected, 'channel_indices': indices, 'fs_hz': fs,
        'signal_unit': getattr(cfg, '_recorded_unit', 'uV'),
        'filter': {'band_hz': cfg.bp_band_hz, 'order': cfg.bp_order},
        'window_s': cfg.window_s, 'step_s': cfg.step_s, 'trial_duration_s': cfg.trial_duration_s,
        'classes_map': label_map, 'classes': details, 'cv': cv,
        'representation': representation['meta'],
        'pca_check_png': str(model_dir / 'pca_check.png'),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'pca_map_json': str(model_dir / 'pca_map.json'), 'pca_map_png': str(model_dir / 'pca_map.png'),
        'pca_train_xlim': [representation['meta']['x_min'], representation['meta']['x_max']],
        'pca_train_ylim': [representation['meta']['y_min'], representation['meta']['y_max']],
        'probabilities_note': 'SVC probability=True; CV report evaluates predictions, not probability calibration.',
    }
    (model_dir / 'model_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[treino] {selected}: {classifier.n_features_in_} dimensões; CV: {cv}')
    return {'model_dir': str(model_dir), 'meta': meta, 'accs_cv': cv['accs']}
