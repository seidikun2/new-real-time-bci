"""Mapa PCA estático do experimentador; não participa da classificação/controle."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def prepare_training_representation(X_base, y, cfg, out_dir, class_labels=None, class_details=None):
    X = np.asarray(X_base, float)
    if X.shape[1] < 2:
        X = np.column_stack([X[:, 0], np.zeros(len(X))])
    X = X[:, :2]
    options = cfg._raw_config.get('representation_feedback', {})
    pad = float(options.get('pad_frac', .1))
    limits = []
    for column in X.T:
        low, high = float(column.min()), float(column.max())
        margin = max(high-low, 1.) * pad
        limits.append([low-margin, high+margin])
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    size = max(30, min(300, int(options.get('grid_size', 180))))
    xx, yy = np.meshgrid(np.linspace(*limits[0], size), np.linspace(*limits[1], size))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    masses = options.get('density_masses', [.5, .8, .95])
    labels = class_labels or {}
    details = {row['model_class']: row for row in class_details or []}
    regions = []
    for i, cls in enumerate(np.unique(y)):
        points = X[y == cls]
        color = plt.get_cmap('tab10')(i)
        ax.scatter(*points.T, s=4, alpha=.12, color=color)
        contours = []
        try:
            kde = gaussian_kde(points.T)
            density = kde(grid).reshape(xx.shape)
            sorted_density = np.sort(density.ravel())[::-1]
            cumulative = np.cumsum(sorted_density) / sorted_density.sum()
            for mass in masses:
                index = min(np.searchsorted(cumulative, mass), len(sorted_density)-1)
                level = sorted_density[index]
                if not density.min() < level < density.max():
                    continue
                contour = ax.contour(xx, yy, density, levels=[level], colors=[color], linewidths=.8)
                polygons = [segment[::max(1, len(segment)//160)].tolist() for segment in contour.allsegs[0] if len(segment) >= 3]
                contours.append({'mass': float(mass), 'polygons': polygons})
        except (ValueError, np.linalg.LinAlgError):
            # Com poucos pontos ou covariância singular, o scatter continua válido.
            pass
        center = np.mean(points, axis=0)
        if options.get('show_labels', True):
            ax.text(*center, labels.get(int(cls), str(cls)), color=color, weight='bold', ha='center')
        row = details.get(int(cls), {})
        regions.append({'event_label': row.get('event_label', str(cls)),
                        'display_label': labels.get(int(cls), str(cls)), 'center': center.tolist(), 'regions': contours})
    ax.set(xlim=limits[0], ylim=limits[1])
    fig.subplots_adjust(left=.13, right=.96, bottom=.12, top=.9)
    ax.set(xlabel='PC1', ylabel='PC2', title='Mapa PCA · verificação do bloco')
    ax.grid(alpha=.15)
    fig.savefig(folder / 'pca_check.png')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    png = folder / 'pca_map.png'
    fig.savefig(png)
    plt.close(fig)
    meta = {'representation_id': hashlib.sha256(X.tobytes()).hexdigest()[:12],
            'representation_name': 'PCA', 'dimensions': 2, 'audience': 'experimenter',
            'x_min': limits[0][0], 'x_max': limits[0][1], 'y_min': limits[1][0], 'y_max': limits[1][1],
            'map_image': png.name, 'transform_type': 'identity'}
    manifest = {**meta, 'purpose': 'experimenter visualization only', 'density_regions': regions,
                'event_labels': [d['event_label'] for d in class_details or []]}
    path = folder / 'pca_map.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'X_rep': X, 'meta': meta, 'map_png': str(png), 'manifest_json': str(path)}


def transform_online_representation(projected, model_meta=None):
    result = np.zeros(2)
    values = np.asarray(projected).ravel()
    result[:min(2, len(values))] = values[:2]
    return result
