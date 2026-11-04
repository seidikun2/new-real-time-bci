"""Todos os dados e artefatos ficam fora do checkout do projeto."""
from pathlib import Path
from tempfile import TemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = '~/BCI_Dados'


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
