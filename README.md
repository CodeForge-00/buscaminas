# Buscaminas · Humano / Red neuronal / Refuerzo

Cada partida sortea la cantidad de minas (10%–20% del tablero) y sus posiciones
(se colocan tras el primer click, que nunca pierde).

## Uso
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```
Modelos ya entrenados en `models/`. Para reentrenar:
```
.venv/bin/python train_nn.py                 # ~1 min
.venv/bin/python train_rl.py --minutes 15    # DQN; --resume para continuar
```

## Modos
- **Humano**: click izq. abre, click der. bandera, click en un número abre vecinos.
- **IA · Red neuronal**: MLP (numpy) que estima P(mina) por celda a partir de una ventana 5x5; abre la más segura.
- **IA · Refuerzo (DQN)**: red que estima Q(celda) (recompensa esperada); abre la de mayor Q.

## Ver cómo piensa la IA
Cada jugada tiene dos fases: **analizar** (mapa de calor con P(mina) o Q por celda, celda elegida en blanco,
ventana 5x5 en amarillo, texto explicativo, lo que ve la red y activación de neuronas) y **actuar**.

Teclas: `Espacio` pausa · `S` un paso · `+/-` velocidad · `N` nueva · `M` menú · `A` auto-nueva ·
`E` exploración (RL) · `L` aprender en vivo (RL).
