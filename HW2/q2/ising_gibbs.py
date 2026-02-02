#!/usr/bin/env python3
"""
Gibbs sampler for the 2D Ising model.
"""

import numpy as np
import matplotlib.pyplot as plt

N = 20 
BETAS = [0.2, 0.4, 0.6]
N_ITER = 1000
RECORD_EVERY = 10
SAVE_AT_ITERS = [0, 100, 500, 1000]


def energy(grid):
    h = np.roll(grid, -1, axis=1)  
    v = np.roll(grid, -1, axis=0)  
    E = -np.sum(grid * h) - np.sum(grid * v)
    return E


def magnetization(grid):
    return np.mean(grid)


def delta_E_i(grid, i, j):
    """Energy change"""
    s = grid[i, j]
    left = grid[i, (j - 1) % N]
    right = grid[i, (j + 1) % N]
    up = grid[(i - 1) % N, j]
    down = grid[(i + 1) % N, j]
    neighbor_sum = left + right + up + down
    return 2 * s * neighbor_sum


def gibbs_step(grid, beta, rng):
    """One Gibbs sweep"""
    rows = np.arange(N)
    cols = np.arange(N)
    rng.shuffle(rows)
    rng.shuffle(cols)
    for i in rows:
        for j in cols:
            dE = delta_E_i(grid, i, j)
            p_flip = 1.0 / (1.0 + np.exp(beta * dE))
            if rng.random() < p_flip:
                grid[i, j] *= -1


def run_sampler(beta, rng, store_states_at=None):
    """Run the sampler"""
    grid = 2 * (rng.integers(0, 2, size=(N, N))) - 1  
    initial_state = grid.copy()
    states_at = {} if store_states_at else None
    energies = []
    magnetizations = []
    for t in range(N_ITER + 1):
        if t in (store_states_at or []):
            states_at[t] = grid.copy()
        if t % RECORD_EVERY == 0:
            energies.append(energy(grid))
            magnetizations.append(magnetization(grid))
        if t < N_ITER:
            gibbs_step(grid, beta, rng)
    return energies, magnetizations, initial_state, states_at


def main():
    rng = np.random.default_rng(42)
    iterations = np.arange(0, N_ITER + 1, RECORD_EVERY)

    # (b) 
    all_energies = {}
    all_magnetizations = {}
    all_initial_states = {}
    all_states_at = {}

    for beta in BETAS:
        energies, magnetizations, initial_state, states_at = run_sampler(
            beta, rng, store_states_at=SAVE_AT_ITERS
        )
        all_energies[beta] = energies
        all_magnetizations[beta] = magnetizations
        all_initial_states[beta] = initial_state
        all_states_at[beta] = states_at

    # (b) 
    fig_e, ax_e = plt.subplots(1, 1, figsize=(8, 5))
    for beta in BETAS:
        ax_e.plot(iterations, all_energies[beta], label=rf"$\beta$ = {beta}")
    ax_e.set_xlabel("Iteration")
    ax_e.set_ylabel(r"$E(s)$")
    ax_e.set_title("Ising model: Energy vs iteration (Gibbs sampler)")
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)
    fig_e.tight_layout()
    fig_e.savefig("ising_energy_vs_iteration.png", dpi=150)
    plt.close(fig_e)
    print("Saved: ising_energy_vs_iteration.png")

    # (c) Plot M versus iteration number
    fig_m, ax_m = plt.subplots(1, 1, figsize=(8, 5))
    for beta in BETAS:
        ax_m.plot(iterations, all_magnetizations[beta], label=rf"$\beta$ = {beta}")
    ax_m.set_xlabel("Iteration")
    ax_m.set_ylabel(r"$M$")
    ax_m.set_title("M versus iteration number")
    ax_m.legend()
    ax_m.grid(True, alpha=0.3)
    fig_m.tight_layout()
    fig_m.savefig("ising_magnetization_vs_iteration.png", dpi=150)
    plt.close(fig_m)
    print("Saved: ising_magnetization_vs_iteration.png")

    # (c) Create visualizations of the grid state at iterations using plt.imshow, plt.colorbar, plt.savefig
    for beta in BETAS:
        for iter in SAVE_AT_ITERS:
            grid = all_states_at[beta][iter]
            plt.figure(figsize=(5, 5))
            plt.imshow(grid, cmap='binary')
            plt.colorbar()
            plt.savefig(f'state_{beta}_{iter}.png', dpi=150)
            plt.close()
            print(f"Saved: state_{beta}_{iter}.png")

    print("Done.")


if __name__ == "__main__":
    main()
