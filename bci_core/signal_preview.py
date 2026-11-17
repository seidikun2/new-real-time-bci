"""Prévia de EEG com rótulos por canal e redução de pontos que preserva picos."""

import numpy as np


def signal_snapshot(times, values, channels, seconds=5.):
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if len(times) < 2 or values.ndim != 2:
        return None
    first = np.searchsorted(times, times[-1]-seconds) if np.isfinite(times).all() and np.all(np.diff(times) > 0) else max(0, len(times)-2500)
    return {'times': times[first:].copy(), 'values': values[first:].copy(), 'channels': np.asarray(channels, dtype=str)}


def display_envelope(times, values, points=600):
    """Reduz somente a exibição; mínimos/máximos de cada grupo mantêm os picos."""
    step = max(1, int(np.ceil(len(times)/max(1, points//2))))
    if step == 1:
        return times, values
    starts = np.arange(0, len(times), step)
    low, high = np.minimum.reduceat(values, starts, axis=0), np.maximum.reduceat(values, starts, axis=0)
    return np.repeat(times[starts], 2), np.stack([low, high], axis=1).reshape(-1, values.shape[1])


def display_traces(times, values, points=1800):
    """Mínimos e máximos por canal, na ordem em que ocorreram no sinal."""
    step = max(1, int(np.ceil(len(times) / max(1, points // 2))))
    if step == 1:
        return np.broadcast_to(np.asarray(times)[:, None], values.shape), values
    channels = values.shape[1]
    complete = len(values) // step * step
    bins = values[:complete].reshape(-1, step, channels)
    base = np.arange(0, complete, step)[:, None]
    low = base + bins.argmin(axis=1)
    high = base + bins.argmax(axis=1)
    indices = np.sort(np.stack([low, high], axis=1), axis=1).reshape(-1, channels)
    if complete < len(values):
        tail = values[complete:]
        extra = complete + np.sort(np.stack([tail.argmin(axis=0), tail.argmax(axis=0)]), axis=0)
        indices = np.vstack([indices, extra])
    return np.asarray(times)[indices], np.take_along_axis(values, indices, axis=0)


class SignalPreview:

    def __init__(self, parent, expandable=True):
        import tkinter as tk
        from tkinter import ttk
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill='both', expand=True)
        self.caption = tk.StringVar(value='Aguardando sinal…')
        self.expanded = None
        if expandable:
            ttk.Button(self.frame, text='Ampliar sinal', command=self.enlarge).pack(anchor='e')
        ttk.Label(self.frame, textvariable=self.caption, wraplength=950).pack(anchor='w', pady=(4, 8))
        area = ttk.Frame(self.frame)
        area.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(area, background='white', highlightthickness=0, yscrollincrement=24)
        scrollbar = ttk.Scrollbar(area, orient='vertical', command=self.scroll)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.canvas.bind('<Map>', lambda event: self.draw())
        self.canvas.bind(
            '<MouseWheel>', lambda event: self.scroll('scroll', -int(event.delta / 120), 'units')
        )
        self.canvas.bind('<Button-4>', lambda event: self.scroll('scroll', -1, 'units'))
        self.canvas.bind('<Button-5>', lambda event: self.scroll('scroll', 1, 'units'))
        self.snapshot = None

    def scroll(self, *args):
        self.canvas.yview(*args)
        self.draw()

    def enlarge(self):
        import tkinter as tk

        if self.expanded is not None and self.expanded.frame.winfo_exists():
            self.expanded.frame.winfo_toplevel().lift()
            return
        window = tk.Toplevel(self.frame)
        window.title('Sinal EEG · visualização ampliada')
        window.geometry(
            f'{min(1500, window.winfo_screenwidth()-40)}x{window.winfo_screenheight()-80}'
        )
        self.expanded = SignalPreview(window, expandable=False)
        self.expanded.update(self.snapshot)

    def update(self, snapshot):
        if snapshot is None:
            return
        self.snapshot = snapshot
        self.draw()
        if self.expanded is not None:
            if self.expanded.frame.winfo_exists():
                self.expanded.update(snapshot)
            else:
                self.expanded = None

    def draw(self):
        if self.snapshot is None or not self.canvas.winfo_ismapped():
            return
        times, values, channels = (self.snapshot[key] for key in ['times', 'values', 'channels'])
        if len(times) < 2:
            return
        width = max(300, self.canvas.winfo_width())
        height = max(self.canvas.winfo_height(), len(channels) * 36 + 60)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.canvas.delete('all')
        invalid = int(np.sum(~np.isfinite(values)))
        safe = np.where(np.isfinite(values), values, np.nan)
        # Médias/escala servem só à visualização; os CSVs não são modificados.
        center = np.nanmedian(safe, axis=0)
        center = np.nan_to_num(center, nan=0.)
        safe = np.where(np.isfinite(values), values, center)
        scale = 1.4826*np.median(abs(safe-center), axis=0)
        scale = np.where(scale > 0, scale, np.std(safe, axis=0))
        scale = np.where(scale > 0, scale, 1.)
        normalized = np.clip((safe-center)/scale, -3., 3.)
        reliable_time = np.isfinite(times).all() and np.all(np.diff(times) > 0)
        plot_times = times if reliable_time else np.arange(len(times), dtype=float)
        display_t, display_v = display_traces(
            plot_times, normalized, points=max(800, min(1600, width))
        )
        span = max(plot_times[-1]-plot_times[0], 1e-9)
        left = max(90, min(190, max(len(str(name)) for name in channels)*8+12))
        x = left+(display_t-plot_times[0])/span*(width-left-18)
        row = (height-45)/len(channels)
        for i, name in enumerate(channels):
            baseline = (i+.5)*row+10
            self.canvas.create_text(
                left - 10,
                baseline,
                text=str(name),
                font='TkDefaultFont',
                anchor='e',
                fill='#1e3448',
                tags='channel_label',
            )
            self.canvas.create_line(left, baseline, width-18, baseline, fill='#edf0f3')
            if baseline + row < self.canvas.canvasy(0) or baseline - row > self.canvas.canvasy(
                self.canvas.winfo_height()
            ):
                continue
            y = baseline-display_v[:, i]*(row*.43/3)
            points = np.column_stack([x[:, i], y]).ravel().tolist()
            self.canvas.create_line(*points, fill='#334155', width=1, tags='signal_trace')
        for fraction in (0., .25, .5, .75, 1.):
            px = left+fraction*(width-left-18)
            self.canvas.create_text(
                px,
                height - 15,
                font='TkDefaultFont',
                text=(
                    f'{(fraction-1)*span:.1f} s'
                    if reliable_time
                    else f'{(fraction-1)*span:.0f} amostras'
                ),
                fill='#63768a',
            )
        text = (
            (f'Últimos {span:.1f} s' if reliable_time else 'Horários inválidos · eixo por amostras')
            + ' · escala por canal, picos limitados só na exibição, pontos em ordem temporal · role para ver todos os canais.'
        )
        if invalid:
            text += f' {invalid} valores não finitos no sinal.'
        self.caption.set(text)
