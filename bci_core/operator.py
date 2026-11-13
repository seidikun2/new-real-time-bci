"""Uma janela para preparar, coletar, revisar e avançar o protocolo."""
import copy
import csv
import json
import math
import threading
import time
import webbrowser
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from queue import Queue, Empty

from .acquisition import SOURCES, source_name, set_source, discover_signals
from .storage import prepare_data_root, external_path, session_path, GROUPS, DEFAULT_DATA_ROOT
from .catalog import (
    read_json,
    settings_path,
    read_sessions,
    session_defaults,
    save_session,
    save_preferences,
    refresh_catalog,
    save_block_comment,
    context_for_signal,
)
from .signal_preview import SignalPreview
from .timing_preview import TimingPreview
from .resting_state import RESTING_SESSION, resting_options
from .theme import PALETTE

STAGES = {'single_target': 'Etapa 1 · alvo e repouso', 'two_legs': 'Etapa 2 · duas pernas e repouso'}
PHASES = {'auto': 'Protocolo completo', 'execution': 'Execução motora', 'imagery': 'Imagética motora', 'online': 'Online'}


class Operator:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, filedialog, font
        self.tk, self.ttk, self.filedialog = tk, ttk, filedialog
        self.root = tk.Tk()
        self.root.title('BCI · condução experimental')
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f'{min(1240, width-80)}x{min(900, height-100)}')
        self.root.minsize(min(900, width-40), min(620, height-80))
        family = 'Segoe UI' if 'Segoe UI' in font.families(self.root) else font.nametofont('TkDefaultFont').actual('family')
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=(family, 11))
        style.configure('TFrame', background=PALETTE['background'])
        style.configure('TLabel', background=PALETTE['background'], foreground=PALETTE['ink'])
        available_fonts = {name.lower() for name in font.families(self.root)}
        serif = next(
            (
                name
                for name in [
                    'Georgia',
                    'DejaVu Serif',
                    'Liberation Serif',
                    'Latin Modern Roman',
                    'Times',
                ]
                if name.lower() in available_fonts
            ),
            family,
        )
        style.configure('Title.TLabel', font=(serif, 24, 'bold'))
        style.configure('Muted.TLabel', foreground=PALETTE['muted'])
        style.configure('Resting.TLabel', font=(family, 17, 'bold'), foreground=PALETTE['accent'])
        style.configure('Countdown.TLabel', font=(family, 30, 'bold'))
        style.configure('TButton', padding=(16, 10))
        style.map('Primary.TButton', background=[('active', PALETTE['accent_hover'])])
        style.configure('TEntry', padding=8)
        style.configure('TCombobox', padding=7)
        style.map('TCombobox', fieldbackground=[('readonly', PALETTE['paper'])])
        style.configure('Treeview', rowheight=34)
        style.configure('TNotebook', background=PALETTE['background'], borderwidth=0)
        style.configure(
            'TNotebook.Tab',
            padding=(16, 9),
            background=PALETTE['sand'],
            foreground=PALETTE['muted'],
        )
        style.map(
            'TNotebook.Tab',
            background=[('selected', PALETTE['paper'])],
            foreground=[('selected', PALETTE['accent'])],
        )
        style.configure(
            'Horizontal.TProgressbar', background=PALETTE['accent'], troughcolor=PALETTE['sand']
        )
        style.configure(
            'Treeview',
            background=PALETTE['paper'],
            fieldbackground=PALETTE['paper'],
            foreground=PALETTE['ink'],
        )
        style.configure('Treeview.Heading', background=PALETTE['sand'], foreground=PALETTE['ink'])
        style.configure(
            'TButton',
            background=PALETTE['paper'],
            foreground=PALETTE['ink'],
            bordercolor=PALETTE['border'],
        )
        style.configure('Primary.TButton', background=PALETTE['accent'], foreground='white')
        style.configure('TEntry', fieldbackground=PALETTE['paper'], foreground=PALETTE['ink'])
        style.configure(
            'TCombobox',
            foreground=PALETTE['ink'],
            selectbackground=PALETTE['sand'],
            selectforeground=PALETTE['ink'],
        )
        style.map(
            'Treeview',
            background=[('selected', PALETTE['sand'])],
            foreground=[('selected', PALETTE['ink'])],
        )
        style.configure(
            'Link.TButton',
            foreground=PALETTE['accent'],
            font=(family, 11, 'underline'),
            padding=(0, 4),
            background=PALETTE['background'],
        )
        self.root.configure(background=PALETTE['background'])
        self.shell = ttk.Frame(self.root, padding=24)
        self.shell.pack(fill='both', expand=True)
        header = ttk.Frame(self.shell)
        header.pack(fill='x')
        ttk.Label(header, text='BCI  /  SESSÃO EXPERIMENTAL', style='Muted.TLabel').pack(side='left')
        self.parameters = ttk.Button(header, text='Parâmetros', command=self.configure_parameters)
        self.parameters.pack(side='right')
        self.steps = tk.StringVar()
        self.steps_label = ttk.Label(
            self.shell, textvariable=self.steps, style='Muted.TLabel', wraplength=970
        )
        self.steps_label.pack(anchor='w', pady=(14, 10))
        self.title = tk.StringVar()
        self.title_label = ttk.Label(self.shell, textvariable=self.title, style='Title.TLabel')
        self.title_label.pack(anchor='w')
        self.subtitle = tk.StringVar()
        ttk.Label(self.shell, textvariable=self.subtitle, style='Muted.TLabel', wraplength=970).pack(anchor='w', pady=(6, 12))
        self.note = tk.StringVar()
        self.note_label = ttk.Label(self.shell, textvariable=self.note, wraplength=970)
        self.note_label.pack(anchor='w', pady=(0, 10))
        self.report_link = ttk.Button(self.shell, text='Abrir relatório completo', style='Link.TButton', cursor='hand2', command=self.open_report)
        self.signal_preview = None
        self.timing_preview = None
        self.buttons = ttk.Frame(self.shell)
        self.buttons.pack(side='bottom', fill='x', pady=(14, 0))
        self.body = ttk.Frame(self.shell)
        self.body.pack(fill='both', expand=True)
        self.action = tk.StringVar()
        self.closing = self.cancelled = False
        self.report_path = None
        self.model_ref = None
        self.cfg = None
        self.live_inlet = None
        self.comment_widget = None
        self.comment_after = None
        self.summary = None
        self.root.protocol('WM_DELETE_WINDOW', self.request_close)

    def request_close(self):
        self.flush_comment()
        if self.summary is not None:
            self.summary.flush()
        if getattr(self, '_registration_dirty', False):
            self.save_registration()
        self.closing = self.cancelled = True
        self.note.set('Encerrando a sessão. A gravação será salva; aguarde a operação atual terminar.')
        self.action.set('stop')

    def page(self, title, subtitle=''):
        if self.summary is not None:
            self.summary.flush()
            self.summary = None
        self.comment_widget = None
        self.title.set(title)
        self.subtitle.set(subtitle)
        self.parameters.pack(side='right')
        self.steps_label.pack(anchor='w', pady=(14, 10), before=self.title_label)
        self.note_label.pack(anchor='w', pady=(0, 10), before=self.body)
        for container in (self.body, self.buttons):
            for widget in container.winfo_children():
                widget.destroy()
        self.signal_preview = None
        self.timing_preview = None
        self.parameters.configure(state='normal')
        return self.body

    def button(self, text, value, primary=False, side='right'):
        self.ttk.Button(self.buttons, text=text, style='Primary.TButton' if primary else 'TButton',
                        command=lambda: self.action.set(value)).pack(side=side, padx=(8, 0))

    def wait_action(self):
        if self.closing:
            return 'stop'
        self.action.set('')
        self.root.wait_variable(self.action)
        return self.action.get()

    def notify(self, text):
        self.note.set(text)
        self.root.update_idletasks()

    def configure(self, cfg, acquisition_duration=None):
        self.cfg = cfg
        self._registration_dirty = False
        self._registration_after = None
        self._loaded_identity = None
        self._catalog_root = None
        preferences = read_json(settings_path())
        configured_root = str(Path(cfg.experiment.log_root).expanduser().resolve())
        default_root = str(Path(DEFAULT_DATA_ROOT).expanduser().resolve())
        use_preferences = preferences.get('data_root') and configured_root in {
            default_root,
            preferences['data_root'],
        }
        root_value = preferences['data_root'] if use_preferences else configured_root
        initial_group = (
            preferences.get('group', cfg.experiment.group)
            if use_preferences
            else cfg.experiment.group
        )
        initial_subject = (
            preferences.get('subject', cfg.experiment.subject_id)
            if use_preferences
            else cfg.experiment.subject_id
        )
        initial_session = (
            preferences.get('session', cfg.experiment.session_id)
            if use_preferences
            else cfg.experiment.session_id
        )
        body = self.page(
            'Preparar sessão',
            'Selecione grupo, participante e sessão. O cadastro salvo é carregado automaticamente.',
        )
        if acquisition_duration is not None:
            self.title.set(f'Verificar aquisição · {acquisition_duration/60:g} min')
            self.subtitle.set('Escolha a aquisição e a pasta dos dados. Este teste não exige Unity nem modelo.')
        self.fields = {}
        identity = self.ttk.Frame(body)
        identity.pack(fill='x', pady=(0, 12))

        def field(parent, label, key, value, row, column, choices=None, editable=False):
            cell = self.ttk.Frame(parent)
            cell.grid(row=row, column=column, sticky='ew', padx=(0, 16), pady=7)
            self.ttk.Label(cell, text=label).pack(anchor='w', pady=(0, 5))
            variable = self.tk.StringVar(value=value)
            widget = (
                self.ttk.Combobox(
                    cell,
                    textvariable=variable,
                    values=choices,
                    state='normal' if editable else 'readonly',
                )
                if choices is not None
                else self.ttk.Entry(cell, textvariable=variable)
            )
            widget.pack(fill='x')
            self.fields[key] = variable
            parent.columnconfigure(column, weight=1, uniform='form')
            return widget

        group_widget = field(identity, 'Grupo', 'group', initial_group, 0, 0, list(GROUPS))
        self.subject_selector = field(
            identity, 'Participante', 'subject', initial_subject, 0, 1, [], True
        )
        session_widget = field(
            identity, 'Sessão', 'session', str(initial_session).lstrip('Ss'), 0, 2
        )
        self.configuration_tabs = self.ttk.Notebook(body)
        self.configuration_tabs.pack(fill='both', expand=True)
        collection = self.ttk.Frame(self.configuration_tabs, padding=14)
        registration = self.ttk.Frame(self.configuration_tabs, padding=14)
        self.configuration_tabs.add(collection, text='Coleta')
        self.configuration_tabs.add(registration, text='Cadastro da sessão')
        field(
            collection,
            'Aquisição',
            'source',
            SOURCES[source_name(cfg)],
            0,
            0,
            list(SOURCES.values()),
        )
        if acquisition_duration is None:
            field(
                collection,
                'Etapa',
                'stage',
                STAGES[cfg.protocol.stage],
                0,
                1,
                list(STAGES.values()),
            )
        folder = self.ttk.Frame(collection)
        folder.grid(row=1, column=0, columnspan=2, sticky='ew', pady=15)
        self.ttk.Label(folder, text='Pasta de dados deste laboratório · fora do projeto').pack(
            anchor='w', pady=(0, 6)
        )
        self.fields['data_root'] = self.tk.StringVar(value=root_value)
        self.ttk.Button(
            folder,
            text='Escolher pasta…',
            command=lambda: self.choose_data_root(self.fields['data_root']),
        ).pack(side='right', padx=(8, 0))
        folder_entry = self.ttk.Entry(folder, textvariable=self.fields['data_root'])
        folder_entry.pack(side='left', fill='x', expand=True)
        self.registration_status = self.tk.StringVar(value='Cadastro não carregado.')
        self.ttk.Label(
            collection, textvariable=self.registration_status, wraplength=860, style='Muted.TLabel'
        ).grid(row=2, columnspan=2, sticky='w', pady=12)
        self.ttk.Button(
            collection,
            text='Editar cadastro da sessão',
            command=lambda: self.configuration_tabs.select(registration),
        ).grid(row=3, column=0, sticky='w')
        field(registration, 'Touca', 'cap', '', 0, 0)
        field(
            registration,
            'Lateralidade · perna de controle',
            'laterality',
            'Direita',
            0,
            1,
            ['Esquerda', 'Direita'],
        )
        field(registration, 'Idade (opcional)', 'age', '', 1, 0)
        field(
            registration,
            'Sexo (opcional)',
            'sex',
            'Não informado',
            1,
            1,
            ['Não informado', 'Feminino', 'Masculino', 'Outro'],
        )
        field(registration, 'Data da coleta', 'collection_date', '', 2, 0)
        field(registration, 'Experimentador/a · nome', 'experimenter', '', 2, 1)
        self.ttk.Label(
            registration,
            text='Salvo automaticamente após editar. Em nova sessão, confira data e experimentador/a.\nA lateralidade define diretamente a perna usada na etapa de uma perna.',
            wraplength=850,
            style='Muted.TLabel',
        ).grid(row=3, columnspan=2, sticky='w', pady=12)
        self.ttk.Button(
            registration, text='Salvar cadastro', command=lambda: self.save_registration(force=True)
        ).grid(row=4, column=0, sticky='w')
        self.ttk.Button(
            registration,
            text='Atualizar tabelas com os dados existentes',
            command=lambda: self.load_registration(refresh=True),
        ).grid(row=4, column=1, sticky='e')
        for widget in [group_widget, self.subject_selector, session_widget, folder_entry]:
            widget.bind('<FocusOut>', lambda event: self.load_registration())
            widget.bind('<Return>', lambda event: self.load_registration())
        group_widget.bind('<<ComboboxSelected>>', lambda event: self.load_registration())
        self.subject_selector.bind('<<ComboboxSelected>>', lambda event: self.load_registration())
        self._loading_registration = False
        for key in ['cap', 'age', 'sex', 'laterality', 'collection_date', 'experimenter']:
            self.fields[key].trace_add('write', self.registration_changed)
        self.load_registration(refresh=True)
        self.button('Preparar protocolo', 'prepare', True)
        self.button('Encerrar', 'stop', side='left')
        while self.wait_action() == 'prepare':
            try:
                source = next(key for key, value in SOURCES.items() if value == self.fields['source'].get())
                selected_root = prepare_data_root(self.fields['data_root'].get())
                record = self.save_registration(force=True)
                if record is None:
                    continue
                set_source(cfg, source)
                cfg.experiment.log_root = selected_root
                cfg.experiment.group = record['group']
                cfg.experiment.subject_id = record['subject']
                cfg.experiment.session_id = int(record['session'][1:])
                for key in ['cap', 'age', 'sex', 'laterality', 'collection_date', 'experimenter']:
                    setattr(cfg.experiment, key, record[key])
                cfg.protocol.online_target = record['laterality']
                if 'stage' in self.fields:
                    cfg.protocol.stage = next(key for key, value in STAGES.items() if value == self.fields['stage'].get())
                cfg.protocol.training_mode = 'target_vs_rest' if cfg.protocol.stage == 'single_target' else 'left_right_vs_rest'
                cfg._raw_config['experiment'] = asdict(cfg.experiment)
                cfg._raw_config['protocol'].update(stage=cfg.protocol.stage, online_target=cfg.protocol.online_target, training_mode=cfg.protocol.training_mode)
                self.fields = {}
                self.note.set('')
                return True
            except (ValueError, TypeError, StopIteration, OSError) as exc:
                self.notify(f'Confira os campos: {exc}')
        if self._registration_dirty:
            self.save_registration()
        return False

    def registration_identity(self):
        root = external_path(self.fields['data_root'].get())
        group = self.fields['group'].get()
        subject = self.fields['subject'].get().strip()
        session = int(self.fields['session'].get())
        session_path(root, group, subject, session)
        return str(root), group, subject, session

    def load_registration(self, refresh=False):
        if not getattr(self, 'fields', {}).get('group'):
            return
        try:
            identity = self.registration_identity()
            if identity == self._loaded_identity and not refresh:
                return
            if self._registration_dirty and self._loaded_identity:
                self.save_registration(identity=self._loaded_identity)
            root, group, subject, session = identity
            if Path(root).is_dir() and (refresh or self._catalog_root != root):
                refresh_catalog(root)
                self._catalog_root = root
            rows = read_sessions(root)
            self.subject_selector.configure(
                values=sorted(
                    {r['subject'] for r in rows if r.get('group') in {group, '', 'UNASSIGNED'}}
                )
            )
            record = session_defaults(root, group, subject, session)
            self._loading_registration = True
            for key in ['cap', 'age', 'experimenter']:
                self.fields[key].set(record.get(key, ''))
            self.fields['sex'].set(
                {'F': 'Feminino', 'M': 'Masculino'}.get(
                    record.get('sex'), record.get('sex') or 'Não informado'
                )
            )
            side = record.get('laterality') or self.cfg.protocol.online_target
            self.fields['laterality'].set('Esquerda' if side == 'left' else 'Direita')
            value = record.get('collection_date') or date.today().isoformat()
            self.fields['collection_date'].set(datetime.fromisoformat(value).strftime('%d/%m/%Y'))
            self._loaded_identity = identity
            self._registration_dirty = False
            self.registration_status.set(
                (
                    'Cadastro carregado'
                    if record.get('registered') == '1'
                    else 'Confira o cadastro desta sessão'
                )
                + f" · {group} / {subject} / S{session} · controle: {self.fields['laterality'].get()}"
            )
            if Path(root).is_dir():
                try:
                    save_preferences(root, group, subject, session)
                except OSError:
                    pass
        except (ValueError, OSError, csv.Error) as exc:
            self.registration_status.set(f'Cadastro: {exc}')
        finally:
            self._loading_registration = False

    def registration_changed(self, *_):
        if self._loading_registration:
            return
        self._registration_dirty = True
        if self._registration_after is not None:
            self.root.after_cancel(self._registration_after)
        self._registration_after = self.root.after(600, self.save_registration)

    def save_registration(self, force=False, identity=None):
        if self._registration_after is not None:
            self.root.after_cancel(self._registration_after)
            self._registration_after = None
        if not getattr(self, 'fields', {}).get('group'):
            return None
        try:
            root, group, subject, session = identity or self.registration_identity()
            if not Path(root).is_dir() and not force:
                return None
            prepare_data_root(root)
            value = self.fields['collection_date'].get().strip()
            collected = (
                datetime.strptime(value, '%d/%m/%Y').date()
                if '/' in value
                else date.fromisoformat(value)
            )
            record = dict(
                group=group,
                subject=subject,
                session=f'S{session}',
                cap=self.fields['cap'].get(),
                age=self.fields['age'].get(),
                sex={'Feminino': 'F', 'Masculino': 'M', 'Não informado': ''}.get(
                    self.fields['sex'].get(), self.fields['sex'].get()
                ),
                laterality='left' if self.fields['laterality'].get() == 'Esquerda' else 'right',
                collection_date=collected.isoformat(),
                experimenter=self.fields['experimenter'].get(),
            )
            saved = save_session(root, record)
            try:
                save_preferences(root, group, subject, session)
            except OSError:
                pass  # O cadastro principal já está preservado na pasta de dados.
            self._registration_dirty = False
            self._loaded_identity = (root, group, subject, session)
            self.registration_status.set(
                saved.get('_warning')
                or f'Cadastro salvo · {group} / {subject} / S{session} · controle: {self.fields["laterality"].get()}'
            )
            return record
        except (ValueError, OSError, csv.Error) as exc:
            self.registration_status.set(f'Cadastro não salvo: {exc}')
            return None

    def choose_data_root(self, variable):
        selected = self.filedialog.askdirectory(title='Pasta dos dados da sessão', parent=self.root,
                                                initialdir=variable.get() if Path(variable.get()).is_dir() else str(Path.home()))
        if selected:
            variable.set(selected)
            self.load_registration(refresh=True)

    def open_report(self):
        if self.report_path:
            webbrowser.open(Path(self.report_path).resolve().as_uri())

    def set_report(self, filename, label):
        self.report_path = filename
        if filename:
            self.report_link.configure(text=f'Abrir relatório completo · {label}')
            self.report_link.pack(anchor='w', before=self.body, pady=(0, 8))
        else:
            self.report_link.pack_forget()

    def configure_parameters(self):
        if self.cfg is None:
            return
        source = source_name(self.cfg)
        if getattr(self, 'fields', {}).get('source'):
            source = next(key for key, label in SOURCES.items() if label == self.fields['source'].get())
        candidate = copy.deepcopy(self.cfg)
        set_source(candidate, source)
        window = self.tk.Toplevel(self.root)
        window.title('Parâmetros de aquisição')
        frame = self.ttk.Frame(window, padding=22)
        frame.pack(fill='both', expand=True)
        options = candidate._raw_config.get(source, {})
        variables = {}
        for i, (label, key, value) in enumerate([
            ('Frequência inicial (Hz)', 'fs', str(candidate.fs_hz)), ('Canais esperados', 'channels', str(candidate.n_channels)),
            ('Nome LSL', 'stream', candidate.lsl.signal_name),
            ('Canais a gravar (vazio = todos)', 'receive', ', '.join(map(str, options.get('receive_channels', [])))),
        ]):
            self.ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w', pady=8, padx=(0, 15))
            value_var = self.tk.StringVar(value=value)
            self.ttk.Entry(frame, textvariable=value_var, width=58).grid(row=i, column=1, sticky='ew', pady=8)
            variables[key] = value_var
        phase = self.tk.StringVar(value=PHASES[candidate.protocol.start_phase])
        self.ttk.Label(frame, text='Começar em').grid(row=4, column=0, sticky='w', pady=8)
        self.ttk.Combobox(frame, textvariable=phase, values=list(PHASES.values()), state='readonly').grid(row=4, column=1, sticky='ew')
        eyes_duration = self.tk.StringVar(value=f"{resting_options(candidate)['duration_s']:g}")
        self.ttk.Label(frame, text='Olhos abertos / fechados (s por condição)').grid(row=5, column=0, sticky='w', pady=8)
        self.ttk.Combobox(frame, textvariable=eyes_duration, values=['30', '45'], state='readonly').grid(row=5, column=1, sticky='ew')
        feedback = self.tk.StringVar(value='A taxa e a quantidade reais serão lidas do stream. Não é necessário informar unidade.')
        self.ttk.Label(frame, textvariable=feedback, wraplength=680).grid(row=6, columnspan=2, sticky='w', pady=15)
        def discover():
            try:
                streams = discover_signals()
                names = [stream['name'] for stream in streams]
                feedback.set('Streams encontrados: '+', '.join(names) if names else 'Nenhum EEG encontrado. Inicie o transmissor e tente novamente.')
                if len(streams) == 1:
                    variables['stream'].set(streams[0]['name'])
                    variables['fs'].set(str(streams[0]['fs_hz'] or candidate.fs_hz))
                    variables['channels'].set(str(streams[0]['n_channels']))
            except Exception as exc:
                feedback.set(str(exc))
        def save():
            try:
                set_source(candidate, source, float(variables['fs'].get()), int(variables['channels'].get()), variables['stream'].get().strip())
                received = [value.strip() for value in variables['receive'].get().split(',') if value.strip()]
                indices = all(value.isdigit() for value in received)
                if received and (len(received) < 2 or len(set(received)) != len(received)):
                    raise ValueError('Selecione pelo menos dois canais distintos ou deixe vazio.')
                if source in {'simulator', 'hiamp'}:
                    candidate._raw_config.update(fs_hz=candidate.fs_hz, n_channels=candidate.n_channels)
                candidate._raw_config.setdefault(source, {}).update(receive_channels=[int(v) for v in received] if indices else received,
                                                                   receive_by='index' if indices else 'name', index_base=1)
                candidate.protocol.start_phase = next(key for key, value in PHASES.items() if value == phase.get())
                candidate._raw_config['protocol']['start_phase'] = candidate.protocol.start_phase
                candidate._raw_config.setdefault('resting_state', {})['duration_s'] = float(eyes_duration.get())
                self.cfg.__dict__.update(candidate.__dict__)
                window.destroy()
            except (ValueError, TypeError) as exc:
                feedback.set(str(exc))
        self.ttk.Button(frame, text='Buscar streams', command=discover).grid(row=7, column=0, sticky='w')
        self.ttk.Button(frame, text='Salvar parâmetros', command=save, style='Primary.TButton').grid(row=7, column=1, sticky='e')
        window.grab_set()

    def set_protocol(self, labels):
        self.protocol_labels = labels
        self.steps.set('  |  '.join(labels))

    def prepare_block(self, cfg, label, model_ref=None):
        self.cfg, self.model_ref, self.block_label = cfg, model_ref, label
        if hasattr(self, 'protocol_labels') and label in self.protocol_labels:
            current = self.protocol_labels.index(label)
            self.steps.set('  |  '.join(f'{i+1}. '+value+(' (atual)' if i == current else '') for i, value in enumerate(self.protocol_labels)))
        self.page(label, f'Sujeito {cfg.experiment.subject_id} · sessão {cfg.experiment.session_id} · {SOURCES[source_name(cfg)]}')
        instruction = 'Simulação completa: não precisa de Unity nem equipamento.' if source_name(cfg) == 'simulator' else 'Abra o Unity e prepare o participante. Inicie quando estiver pronto.'
        if cfg._raw_config.get('protocol', {}).get('acquisition_test_s'):
            instruction = 'Teste de aquisição: sem classificação e sem Unity. Mantenha o transmissor EEG ligado.'
        if cfg.experiment.session_type == RESTING_SESSION:
            duration = resting_options(cfg)['duration_s']
            instruction = (f'Peça para manter os olhos abertos. Após o EEG estabilizar, a contagem de {duration:g} s começa. '
                           f'A tela avisará quando pedir para fechar os olhos; um clique inicia os próximos {duration:g} s.')
        self.ttk.Label(self.body, text=instruction, wraplength=920).pack(anchor='w', pady=18)
        self.ttk.Label(self.body, text=f'Stream: {cfg.lsl.signal_name or "EEG disponível"}', style='Muted.TLabel').pack(anchor='w')
        self.ttk.Label(self.body, text=f'Dados em: {cfg.experiment.log_root}', style='Muted.TLabel').pack(anchor='w', pady=8)
        self.button('Iniciar bloco', 'start', True)
        self.button('Encerrar sessão', 'stop', side='left')
        return self.wait_action() == 'start'

    def begin_block(self):
        label = getattr(self, 'block_label', self.title.get())
        self.page(label, 'Gravando EEG e enviando os estímulos para o Unity.')
        resting = self.cfg.experiment.session_type == RESTING_SESSION
        if resting:
            self.subtitle.set('Preparando o EEG. Comandos ao participante pelo experimentador; observe a contagem abaixo.')
        if self.cfg._raw_config.get('protocol', {}).get('acquisition_test_s'):
            self.subtitle.set('Gravando EEG e auditando os horários. Ao terminar, confira o relatório completo.')
        self.parameters.configure(state='disabled')
        self.cancelled = False
        self.status = self.tk.StringVar(value='Conectando à aquisição…')
        self.event = self.tk.StringVar(value='Preparando streams')
        self.ttk.Label(self.body, textvariable=self.event, style='Resting.TLabel' if resting else 'Muted.TLabel',
                       wraplength=970).pack(anchor='w', pady=(4, 6))
        if resting:
            self.resting_ready = False
            self.resting_countdown = self.tk.StringVar(value='Aguardando EEG estável')
            self.ttk.Label(self.body, textvariable=self.resting_countdown, style='Countdown.TLabel').pack(anchor='w')
            self.resting_start = self.ttk.Button(self.buttons, text='Iniciar olhos fechados',
                                                command=self.start_closed_eyes, style='Primary.TButton', state='disabled')
            self.resting_start.pack(side='left')
        self.ttk.Label(self.body, textvariable=self.status).pack(anchor='w')
        self.bar = self.ttk.Progressbar(self.body, mode='indeterminate')
        self.bar.pack(fill='x', pady=(8, 12))
        self.bar.start(25)
        self.ttk.Button(self.buttons, text='Interromper e salvar', command=lambda: setattr(self, 'cancelled', True)).pack(side='right')
        self.preview_tabs = self.ttk.Notebook(self.body)
        self.preview_tabs.pack(fill='both', expand=True)
        signal_tab = self.ttk.Frame(self.preview_tabs)
        self.preview_tabs.add(signal_tab, text='Snapshot do sinal')
        self.signal_preview = SignalPreview(signal_tab)
        timing_tab = self.ttk.Frame(self.preview_tabs)
        self.preview_tabs.add(timing_tab, text='Sincronização')
        self.timing_preview = TimingPreview(timing_tab)
        if self.cfg._raw_config.get('protocol', {}).get('acquisition_test_s'):
            self.preview_tabs.select(timing_tab)
        self.live_canvas = None
        self.last_resolve = 0.
        if self.model_ref and self.cfg._raw_config.get('debug_plot', {}).get('enabled', True):
            from .online_inference import read_model_meta
            meta = read_model_meta(self.model_ref)
            self.live_limits = meta['pca_train_xlim'] + meta['pca_train_ylim']
            pca_tab = self.ttk.Frame(self.preview_tabs)
            self.preview_tabs.add(pca_tab, text='PCA online')
            self.live_background = meta['pca_map_png']
            self.live_canvas = self.tk.Canvas(pca_tab, background='white', highlightthickness=0)
            self.live_canvas.pack(fill='both', expand=True)
            self.live_canvas.bind('<Configure>', lambda event: self.resize_live_pca())
            self.live_dot = self.live_canvas.create_oval(0, 0, 0, 0, fill='#173c72', outline='white', width=2)
            self.live_probabilities = self.tk.StringVar(value='Probabilidades: aguardando decoder')
            self.ttk.Label(pca_tab, text='PCA · experimentador', style='Muted.TLabel').pack()
            self.ttk.Label(pca_tab, textvariable=self.live_probabilities).pack(pady=6)
        self.root.update_idletasks()

    def update_resting_phase(self, label, remaining, duration):
        condition = 'olhos abertos' if label == 'EYES_OPEN' else 'olhos fechados'
        self.subtitle.set('EEG em gravação. O experimentador orienta o participante.')
        self.event.set(f'GRAVANDO · {condition}')
        self.resting_countdown.set(f'{math.ceil(remaining)} s restantes')
        self.bar.stop()
        self.bar.configure(mode='determinate', maximum=duration, value=duration-remaining)

    def prepare_closed_eyes(self, duration):
        self.subtitle.set('O EEG continua sendo gravado. A espera não entra no tempo das condições.')
        self.event.set(f'{duration:g} s de olhos abertos concluídos. Peça para fechar os olhos.')
        self.resting_countdown.set('Aguardando o experimentador')
        self.resting_ready = False
        self.resting_start.configure(state='normal')
        self.resting_start.focus_set()
        self.root.bell()

    def start_closed_eyes(self):
        self.resting_ready = True
        self.resting_start.configure(state='disabled')

    def finish_resting(self):
        self.event.set('Registro concluído · peça para abrir os olhos.')
        self.resting_countdown.set('Concluído')
        self.root.bell()

    def set_event(self, label):
        labels = {'BASELINE': 'Baseline', 'ATTENTION': 'Atenção', 'LEFT_MI_STIM': 'Esquerda', 'RIGHT_MI_STIM': 'Direita',
                  'REST_STIM': 'Repouso · estímulo', 'ATTEMPT': 'Tentativa', 'REST': 'Pausa', 'BLOCK_END': 'Bloco concluído'}
        self.event.set(labels.get(label, label))

    def poll_block(self, text):
        self.status.set(text)
        self.root.update()
        if self.live_canvas is not None:
            try:
                from pylsl import resolve_byprop, StreamInlet
                if self.live_inlet is None and time.monotonic()-self.last_resolve > 1.:
                    self.last_resolve = time.monotonic()
                    streams = resolve_byprop('name', self.cfg.decoder.outlet_name, timeout=.01)
                    if streams:
                        self.live_inlet = StreamInlet(streams[0], recover=False)
                if self.live_inlet is not None:
                    samples, stamps = self.live_inlet.pull_chunk(timeout=0.)
                    if stamps:
                        x, y = samples[-1][:2]
                        left, rest, right = samples[-1][2:5]
                        self.live_probabilities.set(f'Esquerda {left:.0%}    Repouso {rest:.0%}    Direita {right:.0%}')
                        xmin, xmax, ymin, ymax = self.live_limits
                        width, height = self.live_canvas.winfo_width(), self.live_canvas.winfo_height()
                        px = max(0, min(width, width*(x-xmin)/(xmax-xmin)))
                        py = max(0, min(height, height*(ymax-y)/(ymax-ymin)))
                        self.live_canvas.coords(self.live_dot, px-5, py-5, px+5, py+5)
            except Exception:
                pass  # Falha na visualização não pode interromper a gravação.
        return not (self.cancelled or self.closing)

    def update_signal_preview(self, snapshot):
        if self.signal_preview is not None:
            try:
                self.signal_preview.update(snapshot)
            except (ValueError, IndexError, self.tk.TclError):
                self.signal_preview.caption.set('Snapshot indisponível. A gravação continua; confira o relatório.')

    def update_timing_preview(self, data):
        if self.timing_preview is not None:
            try:
                self.timing_preview.update(data)
            except (ValueError, TypeError, KeyError, self.tk.TclError):
                self.timing_preview.status.set('Visualização temporal indisponível; a gravação continua. Consulte o relatório.')

    def resize_live_pca(self):
        from PIL import Image, ImageTk
        width, height = max(1, self.live_canvas.winfo_width()), max(1, self.live_canvas.winfo_height())
        with Image.open(self.live_background) as source:
            self.live_background_image = ImageTk.PhotoImage(source.resize((width, height)), master=self.root)
        self.live_canvas.delete('background')
        self.live_canvas.create_image(0, 0, anchor='nw', image=self.live_background_image, tags='background')
        self.live_canvas.tag_lower('background')

    def end_block(self):
        self.bar.stop()
        if self.live_inlet is not None:
            self.live_inlet.close_stream()
            self.live_inlet = None

    def run_job(self, title, function, *args, report_progress=False, **kwargs):
        self.page(title, 'Operação em andamento. Acompanhe nesta janela.')
        self.parameters.configure(state='disabled')
        progress = self.ttk.Progressbar(self.body, mode='indeterminate')
        progress.pack(fill='x', pady=25)
        progress.start(25)
        messages = Queue()
        self.job_phase = self.tk.StringVar(value='Iniciando…')
        self.job_elapsed = self.tk.StringVar(value='Tempo decorrido: 0 s')
        self.ttk.Label(self.body, textvariable=self.job_phase, wraplength=950).pack(anchor='w')
        self.ttk.Label(self.body, textvariable=self.job_elapsed, style='Muted.TLabel').pack(
            anchor='w', pady=8
        )
        started = time.perf_counter()
        if report_progress:
            kwargs['progress'] = messages.put
        result = Queue()
        finished = self.tk.BooleanVar(value=False)
        def worker():
            try:
                result.put((True, function(*args, **kwargs)))
            except Exception as exc:
                result.put((False, exc))
        def poll():
            while not messages.empty():
                self.job_phase.set(messages.get_nowait())
            self.job_elapsed.set(f'Tempo decorrido: {time.perf_counter()-started:.0f} s')
            try:
                self.job_result = result.get_nowait()
                finished.set(True)
            except Empty:
                self.root.after(80, poll)
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, poll)
        self.root.wait_variable(finished)
        progress.stop()
        success, value = self.job_result
        if not success:
            raise value
        return value

    def add_image(self, parent, filename):
        if not filename or not Path(filename).exists():
            self.ttk.Label(parent, text='Figura indisponível para este bloco.').pack(pady=20)
            return
        from PIL import Image, ImageTk
        with Image.open(filename) as source:
            original = source.convert('RGB')
        label = self.ttk.Label(parent, anchor='center')
        label.pack(fill='both', expand=True)
        def resize(event):
            image = original.copy()
            image.thumbnail((max(1, event.width-16), max(1, event.height-16)))
            photo = ImageTk.PhotoImage(image, master=self.root)
            label.configure(image=photo)
            label.image = photo
        label.bind('<Configure>', resize)

    def review_block(self, label, block, qc, evaluation):
        complete = block['complete']
        self.page(label + (' · concluído' if complete else ' · INTERROMPIDO'), 'Confira o resultado e escolha como continuar.')
        self.set_report(qc.get('report_path'), label)
        cv = evaluation.get('cv', {})
        diagnostic = bool(self.cfg and self.cfg._raw_config.get('protocol', {}).get('acquisition_test_s'))
        resting = bool(self.cfg and self.cfg.experiment.session_type == RESTING_SESSION)
        inspection_only = diagnostic or resting
        accuracy = cv.get('acc_mean')
        text = f'Acurácia CV: {accuracy:.1%}' if accuracy is not None else 'Acurácia não estimada'
        if qc.get('analysis_timing'):
            text += f" · análise em {qc['analysis_timing']['total_s']:.1f} s"
        if diagnostic:
            text = 'Diagnóstico de aquisição · sem treino de modelo'
        if resting:
            text = ('Registro de repouso concluído · peça para abrir os olhos.' if complete else
                    'Registro de repouso interrompido · oriente o participante a abrir os olhos.')
        self.ttk.Label(self.body, text=text, style='Muted.TLabel').pack(anchor='w', pady=4)
        messages = []
        if not complete:
            messages.append(f"Dados parciais salvos. {block.get('reason', '')}")
        if evaluation.get('error'):
            messages.append(evaluation['error'])
        if qc.get('issues'):
            messages.append(f"Qualidade: {len(qc['issues'])} ponto(s) de atenção. Consulte o relatório.")
        self.ttk.Label(self.body, text='\n'.join(messages), wraplength=950).pack(anchor='w', pady=5)
        if not resting and block.get('signal_file'):
            note = self.ttk.Frame(self.body)
            note.pack(fill='x', pady=(4, 10))
            self.ttk.Label(note, text='Anotações do bloco · opcional').pack(anchor='w', pady=(0, 4))
            self.comment_status = self.tk.StringVar(
                value='Salvo automaticamente nos dados e na tabela.'
            )
            self.comment_signal = block['signal_file']
            self.comment_widget = self.tk.Text(
                note,
                height=3,
                wrap='word',
                font='TkDefaultFont',
                background=PALETTE['paper'],
                foreground=PALETTE['ink'],
                insertbackground=PALETTE['accent'],
                relief='solid',
                borderwidth=1,
                padx=10,
                pady=8,
                undo=True,
            )
            self.comment_widget.pack(fill='x')
            self._comment_saved = context_for_signal(
                self.comment_signal, self.cfg.experiment.log_root
            ).get('comment', '')
            self.comment_widget.insert('1.0', self._comment_saved)
            self.comment_widget.edit_modified(False)
            self.comment_widget.bind('<<Modified>>', self.comment_changed)
            self.ttk.Label(note, textvariable=self.comment_status, style='Muted.TLabel').pack(
                anchor='w'
            )
        tabs = self.ttk.Notebook(self.body)
        tabs.pack(fill='both', expand=True)
        signal_tab = self.ttk.Frame(tabs)
        tabs.add(signal_tab, text='Snapshot do sinal')
        self.signal_preview = SignalPreview(signal_tab)
        snapshot_file = qc.get('snapshot_file')
        if snapshot_file and Path(snapshot_file).exists():
            import numpy as np
            try:
                with np.load(snapshot_file, allow_pickle=False) as snapshot:
                    self.update_signal_preview({key: snapshot[key] for key in ['times', 'values', 'channels']})
            except (OSError, ValueError, KeyError):
                self.signal_preview.caption.set('Snapshot indisponível; consulte o relatório completo.')
        else:
            self.signal_preview.caption.set('Sem sinal suficiente para gerar um snapshot deste bloco.')
        figures = [('PCA', evaluation.get('pca_check_png')),
                   ('Matriz de confusão', str(Path(self.report_path).with_name('confusion_matrix.png')) if self.report_path else None)]
        for title, filename in ([] if inspection_only else figures):
            tab = self.ttk.Frame(tabs)
            tabs.add(tab, text=title)
            self.add_image(tab, filename)
        audit = qc.get('timing_audit', {})
        if inspection_only or audit:
            timing_tab = self.ttk.Frame(tabs)
            tabs.add(timing_tab, text='Sincronização')
            self.timing_preview = TimingPreview(timing_tab)
            rows = []
            trace = audit.get('trace_file')
            if trace and Path(trace).is_file():
                import pandas as pd
                rows = pd.read_csv(trace).tail(2000).astype(object).where(lambda f: f.notna(), None).to_dict('records')
            self.update_timing_preview(dict(rows=rows, options=audit.get('options', {'max_age_s':.25}),
                                            phase=audit.get('phase', 'diagnostic'), protocol_started_s=audit.get('protocol_started_s'),
                                            elapsed_s=audit.get('capture_ended_s'),
                                            health=audit.get('current_health', {'reason':'Sem série temporal neste registro.'})))
            if diagnostic:
                tabs.select(timing_tab)
        self.button('Encerrar teste' if diagnostic else 'Próximo bloco', 'next', True)
        self.button('Repetir bloco', 'repeat')
        if self.model_ref:
            self.button('Trocar modelo', 'model')
        self.button('Encerrar sessão', 'stop')
        action = self.wait_action()
        self.flush_comment()
        return action

    def comment_changed(self, *_):
        if self.comment_widget is None or not self.comment_widget.edit_modified():
            return
        self.comment_widget.edit_modified(False)
        self.comment_status.set('Salvando anotações…')
        if self.comment_after is not None:
            self.root.after_cancel(self.comment_after)
        self.comment_after = self.root.after(500, self.flush_comment)

    def flush_comment(self):
        if self.comment_after is not None:
            self.root.after_cancel(self.comment_after)
            self.comment_after = None
        if self.comment_widget is None:
            return
        try:
            text = self.comment_widget.get('1.0', 'end-1c')
            if text == self._comment_saved:
                return
            warning = save_block_comment(self.comment_signal, text, self.cfg.experiment.log_root)
            self._comment_saved = text
            self.comment_status.set(warning or 'Anotações salvas nos dados e na tabela.')
        except (OSError, ValueError, self.tk.TclError) as exc:
            self.comment_status.set(f'Não foi possível salvar as anotações: {exc}')

    def choose(self, title, rows):
        self.page(title, 'Escolha os dados ou o modelo que serão usados no online.')
        listing = self.ttk.Treeview(self.body, columns=('option',), show='', selectmode='browse', height=9)
        listing.column('option', width=1500, stretch=False)
        vertical = self.ttk.Scrollbar(self.body, orient='vertical', command=listing.yview)
        horizontal = self.ttk.Scrollbar(self.body, orient='horizontal', command=listing.xview)
        listing.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        listing.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)
        for i, (label, _) in enumerate(rows):
            listing.insert('', 'end', iid=str(i), values=(label,))
        self.button('Usar seleção', 'select', True)
        self.button('Voltar', 'stop', side='left')
        while self.wait_action() == 'select':
            selected = listing.selection()
            if selected:
                return rows[int(selected[0])][1]
        return None

    def select_file(self, title, initialdir):
        return self.filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=[('Modelo BCI', 'model_meta.json')], parent=self.root)

    def finish(self, text):
        self.flush_comment()
        self.page('Resumo da sessão', text)
        self.parameters.pack_forget()
        self.steps_label.pack_forget()
        if not self.note.get():
            self.note_label.pack_forget()
        self.set_report(None, '')
        if self.cfg is not None:
            from .session_summary import SessionSummary

            try:
                self.summary = SessionSummary(self.body, self.cfg)
            except (OSError, ValueError, csv.Error) as exc:
                self.notify(
                    f'Os dados estão preservados. Não foi possível carregar a tabela: {exc}'
                )
        self.button('Fechar', 'stop', True)
        self.wait_action()
        if self.summary is not None:
            self.summary.flush()

    def close(self):
        try:
            self.flush_comment()
            if self.summary is not None:
                self.summary.flush()
            if getattr(self, '_registration_dirty', False):
                self.save_registration()
            self.root.destroy()
        except self.tk.TclError:
            pass
