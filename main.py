"""Buscaminas con tres modos: humano, red neuronal y aprendizaje por refuerzo."""
import sys
import time

import numpy as np
import pygame

from minas.agents import NN_PATH, RL_PATH, load_agent, step_reward
from minas.features import CHANNELS, K
from minas.game import LOST, PLAYING, WON, Minesweeper

import os

WIN_W, WIN_H = 1200, 780
BOARD_X, BOARD_Y, BOARD_MAX = 20, 70, 690
PANEL_X, PANEL_W = 740, 440

BG = (24, 27, 34)
PANEL = (34, 38, 48)
TXT = (230, 233, 240)
DIM = (140, 148, 165)
ACCENT = (86, 180, 233)
COVER = (72, 84, 108)
COVER_HI = (96, 110, 138)
OPEN = (205, 210, 220)
NUM_COLORS = [None, (25, 90, 220), (30, 140, 50), (210, 40, 40), (20, 30, 140),
              (140, 30, 30), (20, 140, 140), (20, 20, 20), (100, 100, 100)]

SIZES = {"Pequeño 9x9": (9, 9), "Medio 12x12": (12, 12), "Grande 16x16": (16, 16)}
MODES = {"human": "Humano", "nn": "IA · Red neuronal", "rl": "IA · Refuerzo (DQN)"}
SPEEDS = [2.0, 1.0, 0.5, 0.25, 0.1, 0.03]


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat(t):
    """0 = verde (bueno) ... 1 = rojo (malo)."""
    t = float(np.clip(t, 0, 1))
    g, y, r = (60, 190, 90), (235, 200, 60), (225, 60, 60)
    return lerp(g, y, t * 2) if t < 0.5 else lerp(y, r, (t - 0.5) * 2)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Buscaminas · Humano / Red neuronal / Refuerzo")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        mono = "menlo,consolas,dejavusansmono,couriernew"
        self.f_big = pygame.font.SysFont(mono, 26, bold=True)
        self.f = pygame.font.SysFont(mono, 15)
        self.f_s = pygame.font.SysFont(mono, 12)
        self.clock = pygame.time.Clock()
        self.scene = "menu"
        self.mode, self.size_name = "human", "Medio 12x12"
        self.buttons = []
        self.message = ""
        self.stats = {"wins": 0, "games": 0}
        self.speed = 2
        self.paused = False
        self.auto_new = False
        self.explore = False
        self.learn = False
        self.agents = {}

    # ----------------------------------------------------------------- flujo
    def start(self):
        try:
            if self.mode != "human":
                if self.mode not in self.agents:
                    self.agents[self.mode] = load_agent(self.mode)
                self.agent = self.agents[self.mode]
        except FileNotFoundError:
            script = "train_nn.py" if self.mode == "nn" else "train_rl.py"
            self.message = f"Falta el modelo entrenado: ejecuta  python {script}"
            return
        self.message = ""
        self.stats = {"wins": 0, "games": 0}
        self.scene = "game"
        self.new_game()

    def new_game(self):
        h, w = SIZES[self.size_name]
        self.game = Minesweeper(h, w)
        self.cell = max(20, min(52, BOARD_MAX // max(h, w)))
        self.decision = None
        self.phase = "idle"
        self.acc = 0.0
        self.result_line = ""
        self.over_at = None
        self.t_start = None
        self.t_end = None
        self.counted = False

    def is_ai(self):
        return self.mode != "human"

    # -------------------------------------------------------------- IA paso a paso
    def advance(self):
        g = self.game
        if g.over:
            return
        if self.phase == "idle":
            eps = 0.1 if (self.mode == "rl" and self.explore) else 0.0
            self.decision = self.agent.decide(g, epsilon=eps)
            self.phase = "analysis"
            self.result_line = ""
            return
        d = self.decision
        r, c = d.cell
        if self.t_start is None:
            self.t_start = time.time()
        r_before = int(g.revealed.sum())
        reward = step_reward(g, r, c)
        opened = int(g.revealed.sum()) - r_before
        if self.mode == "rl":
            self.agent.observe(d, reward, g)
            if self.learn:
                for _ in range(4):
                    self.agent.train_step()
        if g.state == LOST:
            self.result_line = f"({r + 1},{c + 1}) era una MINA. Recompensa {reward:+.2f}."
        elif g.state == WON:
            self.result_line = f"Tablero resuelto. Recompensa {reward:+.2f}."
        else:
            self.result_line = f"Abrió {opened} celda(s). Recompensa {reward:+.2f}."
        self.phase = "idle"

    def update(self, dt):
        g = self.game
        if g.over:
            if not self.counted:
                self.counted = True
                self.stats["games"] += 1
                self.stats["wins"] += g.state == WON
                self.t_end = time.time()
                self.over_at = time.time()
                if self.mode == "rl":
                    self.agent.episodes += 1
            if self.is_ai() and self.auto_new and time.time() - self.over_at > 1.5:
                self.new_game()
            return
        if self.is_ai() and not self.paused:
            self.acc += dt
            if self.acc >= SPEEDS[self.speed]:
                self.acc = 0.0
                self.advance()

    # ------------------------------------------------------------------- entrada
    def cell_at(self, pos):
        x, y = pos[0] - BOARD_X, pos[1] - BOARD_Y
        r, c = y // self.cell, x // self.cell
        if x >= 0 and y >= 0 and self.game.in_bounds(r, c):
            return int(r), int(c)
        return None

    def on_click(self, ev):
        for rect, action in self.buttons:
            if rect.collidepoint(ev.pos) and ev.button == 1:
                self.do(action)
                return
        if self.scene != "game" or self.is_ai() or self.game.over:
            return
        cell = self.cell_at(ev.pos)
        if not cell:
            return
        g = self.game
        if ev.button == 1:
            if g.revealed[cell]:
                g.chord(*cell)
            else:
                g.reveal(*cell)
            if g.state == PLAYING and self.t_start is None:
                self.t_start = time.time()
        elif ev.button == 3:
            g.toggle_flag(*cell)

    def do(self, action):
        if action.startswith("mode:"):
            self.mode = action[5:]
            self.message = ""
        elif action.startswith("size:"):
            self.size_name = action[5:]
        elif action == "start":
            self.start()
        elif action == "menu":
            self.scene = "menu"
        elif action == "new":
            self.new_game()
        elif action == "pause":
            self.paused = not self.paused
        elif action == "step":
            self.advance()
        elif action == "slower":
            self.speed = max(0, self.speed - 1)
        elif action == "faster":
            self.speed = min(len(SPEEDS) - 1, self.speed + 1)
        elif action == "auto":
            self.auto_new = not self.auto_new
        elif action == "explore":
            self.explore = not self.explore
        elif action == "learn":
            self.learn = not self.learn

    def on_key(self, key):
        if self.scene == "menu":
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.start()
            return
        table = {pygame.K_n: "new", pygame.K_m: "menu", pygame.K_SPACE: "pause",
                 pygame.K_s: "step", pygame.K_RIGHT: "step", pygame.K_MINUS: "slower",
                 pygame.K_PLUS: "faster", pygame.K_EQUALS: "faster", pygame.K_a: "auto",
                 pygame.K_e: "explore", pygame.K_l: "learn", pygame.K_ESCAPE: "menu"}
        if key in table and (self.is_ai() or table[key] in ("new", "menu")):
            if table[key] in ("explore", "learn") and self.mode != "rl":
                return
            self.do(table[key])

    # --------------------------------------------------------------------- dibujo
    def text(self, s, pos, font=None, color=TXT, center=False):
        surf = (font or self.f).render(s, True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def button(self, label, rect, action, active=False, enabled=True):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos())
        col = ACCENT if active else (COVER_HI if hover and enabled else COVER)
        pygame.draw.rect(self.screen, col, rect, border_radius=6)
        self.text(label, rect.center, self.f, (10, 14, 20) if active else TXT, center=True)
        if enabled:
            self.buttons.append((rect, action))

    def wrap(self, s, width, font):
        words, lines, cur = s.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if font.size(t)[0] <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        return lines + ([cur] if cur else [])

    def draw_menu(self):
        self.text("BUSCAMINAS", (WIN_W // 2, 110), self.f_big, center=True)
        self.text("Cada partida sortea cuántas minas hay y dónde están.", (WIN_W // 2, 150),
                  self.f, DIM, center=True)
        self.text("Modo de juego", (WIN_W // 2, 215), self.f, ACCENT, center=True)
        for i, (k, label) in enumerate(MODES.items()):
            self.button(label, (WIN_W // 2 - 330 + i * 225, 240, 215, 46), f"mode:{k}", self.mode == k)
        self.text("Tamaño del tablero", (WIN_W // 2, 335), self.f, ACCENT, center=True)
        for i, label in enumerate(SIZES):
            self.button(label, (WIN_W // 2 - 330 + i * 225, 360, 215, 46), f"size:{label}",
                        self.size_name == label)
        self.button("JUGAR", (WIN_W // 2 - 110, 460, 220, 56), "start")
        if self.message:
            self.text(self.message, (WIN_W // 2, 545), self.f, (230, 100, 100), center=True)
        desc = {
            "human": "Click izq: abrir · Click der: bandera · Click en un número: abrir vecinos.",
            "nn": "Una red neuronal estima la probabilidad de mina de cada celda y abre la más segura.",
            "rl": "Un agente DQN elige la celda con mayor valor Q. Puedes dejar que siga aprendiendo (L).",
        }[self.mode]
        self.text(desc, (WIN_W // 2, 600), self.f, DIM, center=True)

    def draw_board(self):
        g, cs = self.game, self.cell
        ov = self.decision if (self.is_ai() and self.phase == "analysis") else None
        scores = ov.scores if ov is not None else None
        lo = hi = None
        if scores is not None and ov.kind == "q" and np.isfinite(scores).any():
            lo, hi = np.nanmin(scores), np.nanmax(scores)
        tint = pygame.Surface((cs, cs), pygame.SRCALPHA)
        for r in range(g.h):
            for c in range(g.w):
                rect = pygame.Rect(BOARD_X + c * cs, BOARD_Y + r * cs, cs, cs)
                inner = rect.inflate(-2, -2)
                if g.revealed[r, c]:
                    if g.mines[r, c]:
                        pygame.draw.rect(self.screen, (200, 60, 60) if g.exploded == (r, c) else OPEN, inner)
                        pygame.draw.circle(self.screen, (20, 20, 20), inner.center, cs // 4)
                    else:
                        pygame.draw.rect(self.screen, OPEN, inner)
                        n = g.counts[r, c]
                        if n:
                            self.text(str(n), inner.center, self.f_big if cs > 30 else self.f,
                                      NUM_COLORS[n], center=True)
                else:
                    pygame.draw.rect(self.screen, COVER, inner)
                    if g.state == LOST and g.mines[r, c] and not g.flags[r, c]:
                        pygame.draw.circle(self.screen, (20, 20, 20), inner.center, cs // 5)
                    if g.flags[r, c]:
                        x, y = inner.center
                        pygame.draw.polygon(self.screen, (235, 70, 70),
                                            [(x - cs // 5, y - cs // 4), (x + cs // 4, y), (x - cs // 5, y + cs // 8)])
                        pygame.draw.line(self.screen, (240, 240, 240), (x - cs // 5, y - cs // 4),
                                         (x - cs // 5, y + cs // 3), 2)
                    if scores is not None and np.isfinite(scores[r, c]):
                        if ov.kind == "prob":
                            t, label = min(scores[r, c] / 0.4, 1.0), f"{scores[r, c]:.0%}"
                        else:
                            t = 0.5 if hi - lo < 1e-9 else 1 - (scores[r, c] - lo) / (hi - lo)
                            label = f"{scores[r, c]:+.2f}"
                        tint.fill((*heat(t), 175))
                        self.screen.blit(tint, inner.topleft)
                        if cs >= 36:
                            self.text(label, inner.center, self.f_s, (15, 15, 20), center=True)
        if ov is not None:
            r, c = ov.cell
            r0, c0 = max(0, r - 2), max(0, c - 2)
            r1, c1 = min(g.h, r + 3), min(g.w, c + 3)
            pygame.draw.rect(self.screen, (250, 220, 90),
                             (BOARD_X + c0 * cs, BOARD_Y + r0 * cs, (c1 - c0) * cs, (r1 - r0) * cs), 2)
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (BOARD_X + c * cs, BOARD_Y + r * cs, cs, cs), 4)
        elif self.is_ai() and self.decision is not None and self.phase == "idle":
            r, c = self.decision.cell
            pygame.draw.rect(self.screen, ACCENT, (BOARD_X + c * cs, BOARD_Y + r * cs, cs, cs), 3)

    def draw_window(self, x, y):
        d = self.decision
        self.text("Lo que ve la red alrededor de la celda elegida (5x5)", (x, y), self.f_s, DIM)
        if d is None or d.window is None:
            return
        cs = 11
        for ch in range(len(CHANNELS)):
            ox = x + ch * 62
            self.text(CHANNELS[ch], (ox, y + 18), self.f_s, ACCENT)
            for i in range(K):
                for j in range(K):
                    v = float(np.clip(d.window[ch, i, j], 0, 1))
                    col = lerp((30, 33, 42), (250, 215, 90), v)
                    pygame.draw.rect(self.screen, col, (ox + j * (cs + 1), y + 36 + i * (cs + 1), cs, cs))
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (ox + 2 * (cs + 1) - 1, y + 36 + 2 * (cs + 1) - 1, cs + 2, cs + 2), 1)

    def draw_activations(self, x, y):
        d = self.decision
        self.text("Neuronas ocultas (brillo = activación)", (x, y), self.f_s, DIM)
        if d is None or not d.hidden:
            return
        yy = y + 18
        for h in d.hidden:
            m = max(float(h.max()), 1e-6)
            per_row = 64
            for i, a in enumerate(h):
                px, py = x + (i % per_row) * 6, yy + (i // per_row) * 10
                pygame.draw.rect(self.screen, lerp((30, 33, 42), (120, 220, 160), float(a) / m),
                                 (px, py, 5, 8))
            yy += ((len(h) - 1) // per_row + 1) * 10 + 6

    def draw_panel(self):
        g = self.game
        pygame.draw.rect(self.screen, PANEL, (PANEL_X - 10, 20, PANEL_W + 20, WIN_H - 40), border_radius=10)
        x = PANEL_X + 10
        self.text(MODES[self.mode], (x, 32), self.f_big, ACCENT)
        elapsed = 0 if self.t_start is None else int((self.t_end or time.time()) - self.t_start)
        state = {"ready": "Listo", "playing": "Jugando", "won": "GANÓ", "lost": "PERDIÓ"}[g.state]
        self.text(f"Minas: {g.n_mines} ({g.n_mines / (g.h * g.w):.0%})   Tablero: {g.h}x{g.w}",
                  (x, 72))
        extra = f"   Banderas: {g.flags_left}" if not self.is_ai() else ""
        self.text(f"Estado: {state}   Tiempo: {elapsed}s{extra}", (x, 94))
        if self.is_ai():
            s = self.stats
            self.text(f"Sesión: {s['wins']} ganadas de {s['games']}", (x, 116), color=DIM)

        self.button("Nueva (N)", (x, 142, 120, 32), "new")
        self.button("Menú (M)", (x + 130, 142, 110, 32), "menu")
        if not self.is_ai():
            lines = ["Click izq: abrir  ·  Click der: bandera", "Click en un número con sus banderas: abre vecinos.",
                     "El primer click nunca es una mina."]
            for i, s in enumerate(lines):
                self.text(s, (x, 200 + i * 22), color=DIM)
            return

        self.button("Seguir" if self.paused else "Pausa", (x + 250, 142, 90, 32), "pause", self.paused)
        self.button("Paso (S)", (x + 350, 142, 70, 32), "step")
        self.button("-", (x, 184, 36, 30), "slower")
        self.button("+", (x + 44, 184, 36, 30), "faster")
        self.text(f"Velocidad: {SPEEDS[self.speed]:g}s/fase", (x + 92, 190))
        self.button(f"Auto-nueva: {'sí' if self.auto_new else 'no'} (A)", (x, 222, 210, 30), "auto", self.auto_new)
        if self.mode == "rl":
            self.button(f"Explorar: {'sí' if self.explore else 'no'} (E)", (x + 220, 222, 200, 30), "explore", self.explore)
            self.button(f"Aprender en vivo: {'sí' if self.learn else 'no'} (L)", (x, 260, 300, 30), "learn", self.learn)
            self.text(f"ep {self.agent.episodes}", (x + 312, 267), self.f_s, DIM)

        y = 304
        phase = {"idle": "esperando", "analysis": "ANALIZANDO"}[self.phase]
        if g.over:
            phase = "fin de la partida"
        self.text(f"Análisis · fase: {phase}", (x, y), self.f, ACCENT)
        y += 24
        d = self.decision
        shown = (d.lines if d else ["Pulsa Paso o espera: la IA analizará el tablero."]) + \
                ([self.result_line] if self.result_line else [])
        for s in shown:
            for ln in self.wrap(s, PANEL_W - 20, self.f_s):
                self.text(ln, (x, y), self.f_s)
                y += 16
            y += 3
        y = max(y + 6, 500)
        self.draw_window(x, y)
        self.draw_activations(x, y + 106)
        hov = self.cell_at(pygame.mouse.get_pos())
        if hov and d is not None and d.scores is not None and np.isfinite(d.scores[hov]):
            lab = "P(mina)" if d.kind == "prob" else "Q"
            val = f"{d.scores[hov]:.1%}" if d.kind == "prob" else f"{d.scores[hov]:+.3f}"
            self.text(f"Cursor ({hov[0] + 1},{hov[1] + 1}): {lab} = {val}", (x, WIN_H - 70), self.f_s, ACCENT)
        self.text("Espacio pausa · S paso · +/- velocidad · N nueva · M menú", (x, WIN_H - 48), self.f_s, DIM)

    def draw(self):
        self.screen.fill(BG)
        self.buttons = []
        if self.scene == "menu":
            self.draw_menu()
        else:
            self.draw_board()
            self.draw_panel()
            if self.game.over:
                msg = "¡GANASTE!" if self.game.state == WON else "BOOM"
                if self.is_ai():
                    msg = "La IA resolvió el tablero" if self.game.state == WON else "La IA pisó una mina"
                self.text(msg, (BOARD_X, 30), self.f_big, (110, 220, 130) if self.game.state == WON else (235, 90, 90))
            else:
                self.text("Tablero", (BOARD_X, 36), self.f_big, DIM)
        pygame.display.flip()

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(ev)
                if ev.type == pygame.KEYDOWN:
                    self.on_key(ev.key)
            if self.scene == "game":
                self.update(dt)
            self.draw()


if __name__ == "__main__":
    App().run()
