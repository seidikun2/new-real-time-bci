"""Revisão dos blocos já salvos, na mesma janela do experimento."""

import math
import os
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from .catalog import save_block_comment, session_recordings
from .storage import external_path, session_roots
from .theme import PALETTE


def percentage(value):
    try:
        number = float(value)
        return f'{number:.1%}' if math.isfinite(number) else '—'
    except (ValueError, TypeError):
        return '—'


def block_label(row):
    name = row.get('block', '')
    if 'sem_feedback' in name:
        return 'Online · sem feedback'
    if 'com_feedback' in name:
        return 'Online · com feedback'
    if name.startswith('Repouso_olhos'):
        return 'Olhos abertos / fechados'
    if name.startswith('EM_treino'):
        return 'Execução motora'
    if name.startswith('IM_treino'):
        return 'Imagética motora'
    return name


def collection_time(row):
    try:
        value = row.get('started_at')
        stamp = (
            datetime.fromisoformat(value)
            if value
            else datetime.strptime(row['stamp'][:15], '%Y%m%d_%H%M%S')
        )
        return stamp.strftime('%d/%m %H:%M:%S')
    except (ValueError, KeyError, TypeError):
        return 'Não registrado'


def open_path(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Caminho indisponível: {path}')
    if sys.platform == 'win32':
        os.startfile(str(path))
    else:
        command = 'open' if sys.platform == 'darwin' else 'xdg-open'
        subprocess.Popen([command, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SessionSummary:
    def __init__(self, parent, cfg):
        self.root = external_path(cfg.experiment.log_root)
        self.rows, warning = session_recordings(cfg)
        self.selected = None
        self.pending = None
        self.saved_comment = ''
        self.loading = False
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill='both', expand=True)
        exp = cfg.experiment
        evaluated = sum(percentage(row.get('cv_accuracy')) != '—' for row in self.rows)
        partial = sum(row.get('status') == 'partial' for row in self.rows)
        ttk.Label(
            self.frame,
            text=f'{exp.group} / {exp.subject_id} / S{exp.session_id} · '
            f'{len(self.rows)} blocos · {evaluated} com CV · {partial} interrompidos',
            style='Muted.TLabel',
        ).pack(anchor='w', pady=(0, 8))
        folder = ttk.Frame(self.frame)
        folder.pack(fill='x', pady=(0, 12))
        ttk.Label(folder, text='Pasta da sessão').pack(side='left', padx=(0, 10))
        self.session_folder = session_roots(cfg)[0]
        self.session_folder_value = tk.StringVar(value=str(self.session_folder))
        self.session_entry = ttk.Entry(
            folder, textvariable=self.session_folder_value, state='readonly'
        )
        self.session_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(folder, text='Abrir pasta', command=lambda: self.open(self.session_folder)).pack(
            side='right', padx=(8, 0)
        )
        self.status = tk.StringVar(
            value=warning
            or 'Selecione um bloco; use a aba Editar anotações para alterar o comentário.'
        )
        ttk.Label(self.frame, textvariable=self.status, style='Muted.TLabel', wraplength=950).pack(
            side='bottom', anchor='w', pady=(4, 0)
        )
        self.tabs = ttk.Notebook(self.frame)
        self.tabs.pack(fill='both', expand=True)
        blocks_page = ttk.Frame(self.tabs)
        notes_page = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(blocks_page, text='Blocos da sessão')
        self.tabs.add(notes_page, text='Editar anotações')
        blocks_page.rowconfigure(0, weight=1)
        blocks_page.columnconfigure(0, weight=1)
        area = ttk.Frame(blocks_page)
        area.grid(row=0, column=0, sticky='nsew')
        columns = [
            ('time', 'Coleta', 125),
            ('block', 'Bloco', 205),
            ('stage', 'Etapa', 80),
            ('status', 'Estado', 120),
            ('cv', 'CV · média ± DP', 145),
            ('size', 'MiB', 70),
            ('comment', 'Comentário', 260),
        ]
        self.table = ttk.Treeview(
            area,
            columns=[key for key, _, _ in columns],
            show='headings',
            selectmode='browse',
            height=5,
        )
        for key, title, width in columns:
            self.table.heading(key, text=title)
            minimum = 190 if key == 'block' else 180 if key == 'comment' else 60
            self.table.column(
                key, width=width, minwidth=minimum, stretch=key in {'block', 'comment'}
            )
        vertical = ttk.Scrollbar(area, orient='vertical', command=self.table.yview)
        horizontal = ttk.Scrollbar(area, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        area.rowconfigure(0, weight=1)
        area.columnconfigure(0, weight=1)
        self.table.tag_configure('partial', foreground='#9b451e')
        for index, row in enumerate(self.rows):
            self.table.insert(
                '', 'end', iid=str(index), values=self.values(row), tags=(row.get('status', ''),)
            )
        self.table.bind('<<TreeviewSelect>>', self.select)
        self.table.bind(
            '<Configure>',
            lambda event: self.table.see(self.selected) if self.selected is not None else None,
        )
        self.detail = tk.StringVar(value='Nenhum bloco registrado nesta sessão.')
        ttk.Label(blocks_page, textvariable=self.detail, style='Muted.TLabel', wraplength=950).grid(
            row=1, column=0, sticky='ew', pady=8
        )
        self.data_folder = tk.StringVar()
        ttk.Entry(blocks_page, textvariable=self.data_folder, state='readonly').grid(
            row=2, column=0, sticky='ew'
        )
        actions = ttk.Frame(blocks_page)
        actions.grid(row=3, column=0, sticky='ew', pady=8)
        self.folder_button = ttk.Button(
            actions,
            text='Pasta do bloco',
            command=lambda: self.open(self.data_folder.get()),
            state='disabled',
        )
        self.folder_button.pack(side='left')
        self.report_button = ttk.Button(
            actions, text='Relatório completo', command=self.open_report, state='disabled'
        )
        self.report_button.pack(side='left', padx=8)
        ttk.Button(
            actions,
            text='Abrir tabela CSV',
            command=lambda: self.open(self.root / 'dataset_index.csv'),
        ).pack(side='left')
        self.note_title = tk.StringVar(value='Selecione um bloco na aba Blocos da sessão.')
        ttk.Label(notes_page, textvariable=self.note_title, wraplength=900).pack(
            anchor='w', pady=(0, 8)
        )
        ttk.Label(
            notes_page,
            text='Campo opcional · salvo automaticamente após editar.',
            style='Muted.TLabel',
        ).pack(anchor='w', pady=(0, 8))
        self.comment = tk.Text(
            notes_page,
            height=5,
            wrap='word',
            font='TkDefaultFont',
            background=PALETTE['paper'],
            foreground=PALETTE['ink'],
            padx=8,
            pady=6,
            relief='solid',
            borderwidth=1,
            undo=True,
            state='disabled',
        )
        self.comment.pack(fill='both', expand=True)
        self.comment.bind('<<Modified>>', self.changed)
        if self.rows:
            last = str(len(self.rows) - 1)
            self.table.selection_set(last)
            self.table.see(last)
            self.select()

    def values(self, row):
        accuracy = percentage(row.get('cv_accuracy'))
        if accuracy != '—' and percentage(row.get('cv_std')) != '—':
            accuracy += ' ± ' + percentage(row['cv_std'])
        if row.get('cv_status') == 'not_applicable':
            accuracy = 'Não se aplica'
        state = {
            'complete': 'Concluído',
            'partial': 'Interrompido',
            'unknown': 'Não confirmado',
        }.get(row.get('status'), 'Indisponível')
        return (
            collection_time(row),
            block_label(row),
            {'single_target': '1 perna', 'two_legs': '2 pernas'}.get(row.get('stage'), '—'),
            state,
            accuracy,
            row.get('signal_mb') or '—',
            ' '.join(row.get('comment', '').split()),
        )

    def select(self, event=None):
        selection = self.table.selection()
        if self.loading or not selection or selection[0] == self.selected:
            return
        if not self.flush():
            self.table.selection_set(self.selected)
            return
        self.selected = selection[0]
        row = self.rows[int(self.selected)]
        self.note_title.set(f'{block_label(row)} · {collection_time(row)}')
        signal = self.root / row['signal_path']
        self.data_folder.set(str(signal.parent))
        self.folder_button.configure(state='normal' if signal.parent.is_dir() else 'disabled')
        report = self.root / row.get('report_path', '')
        self.report_button.configure(state='normal' if report.is_file() else 'disabled')
        text = f"CV do bloco · balanceada: {percentage(row.get('cv_balanced_accuracy'))} · por tentativa: {percentage(row.get('cv_trial_accuracy'))} · divisões: {row.get('cv_splits') or '—'}"
        reason = row.get('evaluation_error') or row.get('interrupt_reason')
        self.detail.set(text + (f' · {reason}' if reason else ''))
        self.loading = True
        self.comment.configure(state='normal')
        self.comment.delete('1.0', 'end')
        self.saved_comment = row.get('comment', '')
        self.comment.insert('1.0', self.saved_comment)
        self.comment.edit_reset()
        self.comment.edit_modified(False)
        if row.get('block', '').startswith('Repouso_olhos') or not signal.is_file():
            self.comment.configure(state='disabled')
            self.note_title.set(
                self.note_title.get() + '\nAnotações indisponíveis para este registro.'
            )
        self.loading = False

    def changed(self, event=None):
        if self.loading or not self.comment.edit_modified():
            return
        self.comment.edit_modified(False)
        if self.pending is not None:
            self.frame.after_cancel(self.pending)
        self.status.set('Salvando anotações…')
        self.pending = self.frame.after(500, self.flush)

    def flush(self):
        if self.pending is not None:
            self.frame.after_cancel(self.pending)
            self.pending = None
        if self.selected is None:
            return True
        text = self.comment.get('1.0', 'end-1c')
        if text == self.saved_comment:
            return True
        row = self.rows[int(self.selected)]
        try:
            warning = save_block_comment(self.root / row['signal_path'], text, self.root)
            row['comment'] = self.saved_comment = text
            self.table.item(self.selected, values=self.values(row))
            self.status.set(warning or 'Anotações salvas nos dados, na tabela e no relatório.')
            return True
        except (OSError, ValueError) as exc:
            self.status.set(f'Não foi possível salvar: {exc}')
            return False

    def open(self, path):
        try:
            open_path(path)
        except OSError as exc:
            self.status.set(str(exc))

    def open_report(self):
        if self.selected is not None:
            self.open(self.root / self.rows[int(self.selected)]['report_path'])
