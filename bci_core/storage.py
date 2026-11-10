"""Todos os dados e artefatos ficam fora do checkout do projeto."""
from pathlib import Path
from tempfile import TemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = '~/BCI_Dados'
GROUPS = ('TEST', 'CTRL', 'LM', 'LION')


def external_path(value):
    if not value or not str(value).strip():
        raise ValueError('Escolha uma pasta de dados fora da pasta do projeto.')
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(PROJECT_ROOT):
        raise ValueError('A pasta de dados deve ficar fora do projeto/GitHub. Escolha outra pasta.')
    return path


def prepare_data_root(value):
    path = external_path(value)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with TemporaryFile(dir=path) as probe:
            probe.write(b'bci')
    except OSError as exc:
        raise ValueError(f'Não foi possível gravar na pasta de dados {path}: {exc}') from exc
    return str(path)


def session_path(data_root, group, subject, session):
    if group not in GROUPS:
        raise ValueError('Escolha o grupo TEST, CTRL, LM ou LION.')
    if (
        not subject
        or subject in {'.', '..'}
        or any(c in subject for c in '/\\')
        or int(session) < 1
    ):
        raise ValueError('Informe participante sem barras e sessão inteira positiva.')
    return external_path(external_path(data_root) / group / subject / f'S{int(session)}')


def session_roots(cfg):
    current = session_path(
        cfg.experiment.log_root,
        cfg.experiment.group,
        cfg.experiment.subject_id,
        cfg.experiment.session_id,
    )
    legacy = (
        external_path(cfg.experiment.log_root)
        / cfg.experiment.subject_id
        / f'S{cfg.experiment.session_id}'
    )
    return [current, legacy] if legacy.is_dir() else [current]


def canonical_block(name):
    return str(name).replace('_PCA', '_com_feedback')


def feedback_condition(name):
    value = canonical_block(name).lower()
    if 'sem_feedback' in value:
        return 'sem_feedback'
    if 'com_feedback' in value:
        return 'com_feedback'
    return ''
