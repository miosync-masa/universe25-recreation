# ==============================================================
# Λ³ Self‑Evolving Automaton — FULL VERSION (Colab-ready)
# Universe‑25 extensions + SOC + Mobility + Energy Conservation
# Lineage viz + BO 3-type stats + Event-aligned slopes + τ_delay
# Finite-size scaling support
# ==============================================================

# Colab tips:
# - This script saves all outputs under ./outputs
# - Requires: numpy, matplotlib, pandas (preinstalled in Colab)
# - No seaborn; 1 chart per figure; no explicit color choices.

import os, math, json, csv, time, shutil
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# File helpers
# ------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

# ------------------------
# Config
# ------------------------
@dataclass
class SimConfig:
    # Grid / time
    GRID_H: int = 36
    GRID_W: int = 36
    STEPS: int  = 120
    NEIGH_RADIUS: int = 1
    SEED_DENSITY: float = 0.06

    # Genome
    G: int = 24  # 3x8 bits

    # Base Λ thresholds
    BASE_L_REPL: float = 0.95
    BASE_L_EVOL: float = 1.10
    EPS_L_CRIT: float  = 0.03

    # SOC threshold tuning
    SOC_GAMMA: float = 0.10
    SOC_SIGMA0: float = 0.02
    THR_MIN_REPL: float = 0.80
    THR_MAX_REPL: float = 0.98
    THR_MIN_EVOL: float = 1.05
    THR_MAX_EVOL: float = 1.35

    # Mutation
    MUT_LOW: float  = 0.008
    MUT_HIGH: float = 0.08

    # K and V components
    K_BASE: float = 0.05
    K_MISMATCH_SCALE: float = 0.6
    K_GAME_SCALE: float = 0.2
    K_ENERGY_W: float = 0.3
    K_DENSITY_W: float = 0.9  # Universe-25 stress weight

    V_FIT_W: float = 0.50
    V_NEI_W: float = 0.25
    V_GAME_W: float = 0.15
    V_POT_W: float = 0.10

    # Energy & Resources
    INIT_RESOURCE_MEAN: float = 2.0
    INIT_RESOURCE_NOISE: float = 0.2
    INIT_EINT_IF_ALIVE: float = 0.5
    HARVEST_RATE: float = 0.20
    METAB_RATE: float = 0.015
    MOVE_COST: float = 0.02
    RESOURCE_DIFFUSION: float = 0.10
    E_NORM: float = 1.0
    SELFISH_HARVEST: bool = True  # selfish gain for low coop

    # Movement tendency effect scaling inside step (probability uses move_prop and L)
    MOVE_ENABLE: bool = True

    # Universe‑25 density stress
    CRIT_DENSITY: float = 0.65

    # Vanity & Sacrifice (enabled via genome fields)
    # (vanity spend coefficient used inside function)

    # Event detection / τ_delay
    PRE_W: int = 10
    POST_W: int = 20
    SPIKE_Q: int = 80  # percentile for spike detection
    EPS_REP: float = 1e-3
    LCC_THRES: float = 0.30
    K_STABLE: int = 5
    EPS_LOCAL: float = 0.06  # if using energy-based local death (optional view)

    # Output
    OUTDIR: str = "./outputs"
    TAG: str = "U25_FULL"

    # RNG
    BASE_SEED: int = 5000

# ------------------------
# Data structures
# ------------------------
@dataclass
class Phenotype:
    coop: float
    repl_bias: int
    csp1_pref: float
    csp2_len: int
    mut_scale: float
    rule_tft: float
    res_eff: float
    move_prop: float
    sacrifice: float
    vanity: float

@dataclass
class Cell:
    alive: bool
    genome: np.ndarray  # bool bits
    id: int
    parent: int
    last_action: int
    E_int: float
    actions_hist: List[str]

# ------------------------
# RNG helper
# ------------------------
def rng_for(seed):
    return np.random.default_rng(seed)

# ------------------------
# Neighborhood
# ------------------------
def neighbors(cfg: SimConfig, i: int, j: int) -> List[Tuple[int,int]]:
    coords = []
    R = cfg.NEIGH_RADIUS
    for di in range(-R, R+1):
        for dj in range(-R, R+1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i+di, j+dj
            if 0 <= ii < cfg.GRID_H and 0 <= jj < cfg.GRID_W:
                coords.append((ii, jj))
    return coords

# ------------------------
# Genome parsing (opcodes 0..7 with vanity/sacrifice packing)
# ------------------------
def bits_to_int(bits):
    v = 0
    for b in bits: v = (v<<1) | int(b)
    return v

def parse_genome(cfg: SimConfig, genome: np.ndarray) -> Phenotype:
    fields = {
        'coop':0.5, 'repl_bias':0, 'csp1_pref':1.0, 'csp2_len':3,
        'mut_scale':1.0, 'rule_tft':0.0, 'res_eff':1.0, 'move_prop':0.0,
        'sacrifice':0.0, 'vanity':0.0
    }
    n_instr = min(3, len(genome)//8)
    for k in range(n_instr):
        chunk = genome[k*8:(k+1)*8]
        opcode = bits_to_int(chunk[:3])
        arg    = bits_to_int(chunk[3:])
        if opcode == 0:
            fields['coop'] = min(1.0, arg/31.0)
        elif opcode == 1:
            fields['repl_bias'] = int((arg/31.0)*2.99)
        elif opcode == 2:
            fields['csp1_pref'] = 0.5 + 0.5*(arg/31.0)
        elif opcode == 3:
            fields['csp2_len'] = 1 + int((arg/31.0)*5.99)
        elif opcode == 4:
            fields['mut_scale'] = 0.5 + (arg/31.0)
        elif opcode == 5:
            # split: high bits → rule_tft, low bits → vanity
            fields['rule_tft'] = (arg >> 2) / 7.0
            fields['vanity']   = (arg & 0b11)/3.0
        elif opcode == 6:
            fields['res_eff'] = 0.5 + (arg/31.0)
        elif opcode == 7:
            # split: high bits → move_prop, low bits → sacrifice
            fields['move_prop'] = (arg >> 3)/3.0
            fields['sacrifice'] = (arg & 0b111)/7.0
    return Phenotype(**fields)

# ------------------------
# Environment: Multi-constraint CSP
# ------------------------
def target_s1(cfg: SimConfig, i, j, t):
    G = cfg.G
    base = (G/2
            + (G/4)*np.sin(2*np.pi*(i/cfg.GRID_H + 0.02*t))
            + (G/6)*np.cos(2*np.pi*(j/cfg.GRID_W + 0.03*t)))
    return int(np.clip(round(base), 0, G))

def csp1_fitness(cfg: SimConfig, genome, s, pref):
    ones = int(genome.sum())
    return max(0.0, 1.0 - pref*abs(ones - s)/cfg.G)

def csp2_fitness(cfg: SimConfig, genome, L, i, j, t):
    G = cfg.G
    window_len = 8
    start = (i + j + t) % (G - window_len + 1)
    seg = genome[start:start+window_len].astype(int)
    score = 0; run = 1
    for k in range(1, len(seg)):
        if seg[k] == seg[k-1]:
            run += 1
            if run == L:
                score += 1
        else:
            run = 1
    return min(1.0, score / max(1, window_len))

def fitness_multi(cfg: SimConfig, genome, i, j, t, ph: Phenotype):
    s = target_s1(cfg, i, j, t)
    f1 = csp1_fitness(cfg, genome, s, ph.csp1_pref)
    f2 = csp2_fitness(cfg, genome, ph.csp2_len, i, j, t)
    w1 = 0.6 + 0.2*np.sin(2*np.pi*(t/30 + i/cfg.GRID_H))
    w2 = 1.0 - w1
    f = max(0.0, min(1.0, w1*f1 + w2*f2))
    return f, s, f1, f2, w1, w2

# ------------------------
# PD interaction
# ------------------------
R_pay, S_pay, Tm_pay, P_pay = 3, 0, 5, 1

def pd_payoff(my_act, nb_act):
    if my_act == 1 and nb_act == 1: return R_pay
    if my_act == 1 and nb_act == 0: return S_pay
    if my_act == 0 and nb_act == 1: return Tm_pay
    return P_pay

def pd_action(cfg: SimConfig, cell: Cell, ph: Phenotype, avg_nb_last, rng):
    base = ph.coop
    influence = ph.rule_tft * (avg_nb_last - 0.5) * 2.0
    pC = np.clip(base + 0.25*influence, 0.0, 1.0)
    return 1 if rng.random() < pC else 0

def neighbor_game(cfg: SimConfig, grid, i, j, ph: Phenotype, rng):
    nb = neighbors(cfg, i, j)
    acts = []
    for ii,jj in nb:
        c = grid[ii][jj]
        if c.alive: acts.append(c.last_action)
    avg_nb_last = np.mean(acts) if acts else 0.5
    me = grid[i][j]
    my_act = pd_action(cfg, me, ph, avg_nb_last, rng)
    pay = 0.0; cnt = 0
    for ii,jj in nb:
        c = grid[ii][jj]
        if c.alive:
            pay += pd_payoff(my_act, c.last_action); cnt += 1
    score = pay/(cnt*Tm_pay) if cnt>0 else 0.5
    return score, my_act, cnt

# ------------------------
# Init grid & resources
# ------------------------
_global_id_counter = 0
def next_id():
    global _global_id_counter
    _global_id_counter += 1
    return _global_id_counter

def random_genome(cfg: SimConfig, rng):
    return rng.integers(0, 2, size=cfg.G, dtype=np.int8).astype(bool)

def init_grid(cfg: SimConfig, rng):
    R = cfg.INIT_RESOURCE_MEAN + cfg.INIT_RESOURCE_NOISE*(rng.random((cfg.GRID_H, cfg.GRID_W)) - 0.5)
    grid = [[None for _ in range(cfg.GRID_W)] for _ in range(cfg.GRID_H)]
    global _global_id_counter
    _global_id_counter = 0
    for i in range(cfg.GRID_H):
        for j in range(cfg.GRID_W):
            if rng.random() < cfg.SEED_DENSITY:
                gid = next_id()
                grid[i][j] = Cell(True, random_genome(cfg, rng), gid, -1, rng.integers(0,2), E_int=cfg.INIT_EINT_IF_ALIVE, actions_hist=[])
            else:
                grid[i][j] = Cell(False, np.zeros(cfg.G, dtype=bool), -1, -1, 0, E_int=0.0, actions_hist=[])
    return grid, np.clip(R, 0.0, None)

# ------------------------
# Energy bookkeeping
# ------------------------
def diffuse_resource(cfg: SimConfig, R):
    Rnew = R.copy()
    for i in range(cfg.GRID_H):
        for j in range(cfg.GRID_W):
            center = R[i,j]; nsum = 0.0; cnt = 0
            for ii,jj in neighbors(cfg, i, j):
                nsum += R[ii,jj]; cnt += 1
            if cnt>0:
                lap = (nsum/cnt) - center
                Rnew[i,j] += cfg.RESOURCE_DIFFUSION * lap
    return np.clip(Rnew, 0.0, None)

def total_internal_energy(cfg: SimConfig, grid):
    s = 0.0
    for i in range(cfg.GRID_H):
        for j in range(cfg.GRID_W):
            s += grid[i][j].E_int
    return s

# ------------------------
# Universe‑25: density stress & strategy costs
# ------------------------
def density_stress(cfg: SimConfig, grid, i, j):
    nb = neighbors(cfg, i, j)
    if not nb: return 0.0
    occ = sum(1 for ii,jj in nb if grid[ii][jj].alive)
    dens = occ/len(nb)
    return (dens - 0.8)*5.0 if dens>0.8 else 0.0

def child_survival_rate(cfg: SimConfig, density, parent_stress):
    base = 1.0
    if density > 0.7:
        base *= (1.0 - (density-0.7)*2.0)
    if parent_stress > 0.5:
        base *= (1.0 - parent_stress*0.5)
    return max(0.0, base)

def density_dependent_costs(cfg: SimConfig, ph: Phenotype, density, action):
    COOP_BASE_COST = 0.02
    MOVE_BASE_COST = 0.03
    REPL_BASE_COST = 0.08
    costs = {}
    if density > cfg.CRIT_DENSITY:
        df = (density - cfg.CRIT_DENSITY)/(1.0 - cfg.CRIT_DENSITY)
        costs['cooperation'] = ph.coop * COOP_BASE_COST * (1.0 + 0.15*df*10.0)
        costs['movement']    = ph.move_prop * MOVE_BASE_COST * (1.0 + 0.20*df*15.0)
        if action == "replicate":
            failure_prob = 1.0 - child_survival_rate(cfg, density, 0.5)
            costs['reproduction'] = REPL_BASE_COST * (1.0 + 0.50*df*20.0) * (1.0 + failure_prob*2.0)
        else:
            costs['reproduction'] = 0.0
    else:
        costs['cooperation'] = ph.coop * COOP_BASE_COST
        costs['movement']    = ph.move_prop * MOVE_BASE_COST
        costs['reproduction'] = REPL_BASE_COST if action=="replicate" else 0.0
    costs['metabolism'] = 0.0  # 代謝は別処理で落とす
    return costs

def strategy_payoff(cfg: SimConfig, ph: Phenotype, density, action, harvest, gscore):
    income = harvest + gscore * 0.1
    costs = density_dependent_costs(cfg, ph, density, action)
    total_cost = sum(costs.values())
    net_payoff = income - total_cost
    return net_payoff, costs

# ------------------------
# Sacrifice (altruistic transfer) & Vanity (waste)
# ------------------------
def perceived_fairness(cfg: SimConfig, grid, i, j, ph: Phenotype):
    my_E = grid[i][j].E_int
    nb = neighbors(cfg, i, j)
    richer = []
    for ii, jj in nb:
        c = grid[ii][jj]
        if not c.alive: continue
        nbr_E = c.E_int
        ph2 = parse_genome(cfg, c.genome)
        if nbr_E > my_E * 1.3:
            richer.append((nbr_E, ph2.sacrifice, ph2.vanity))
    if not richer: return 1.0
    avg_rich_sac = float(np.mean([x[1] for x in richer]))
    avg_rich_van = float(np.mean([x[2] for x in richer]))
    unfair = 0.0
    if ph.sacrifice > avg_rich_sac + 0.2:
        unfair = (ph.sacrifice - avg_rich_sac) * 2.0
        if avg_rich_van > 0.5:
            unfair *= 1.5
    fairness = 1.0 - min(1.0, unfair)
    return fairness

def sacrifice_willingness_adjusted(base_sacrifice, fairness):
    if fairness > 0.7: return base_sacrifice
    elif fairness > 0.4: return base_sacrifice * 0.5
    else: return base_sacrifice * 0.1

def noblesse_oblige_pressure(cell_E, avg_E, base_sacrifice):
    if cell_E < avg_E: return 0.0
    wealth_ratio = cell_E / max(0.1, avg_E)
    expected = min(0.8, 0.3 + 0.2 * (wealth_ratio - 1.0))
    deficit = expected - base_sacrifice
    if deficit > 0.2: return -0.5 * deficit
    else: return 0.1

def altruistic_transfer(cfg: SimConfig, grid, i, j, ph: Phenotype, rng):
    donor = grid[i][j]
    if donor.E_int < 0.3: return 0, 0.0
    fairness = perceived_fairness(cfg, grid, i, j, ph)
    eff_sac  = sacrifice_willingness_adjusted(ph.sacrifice, fairness)
    if rng.random() > eff_sac: return 0, 0.0
    nb = neighbors(cfg, i, j)
    needy = []
    for ii, jj in nb:
        r = grid[ii][jj]
        if r.alive and r.E_int < 0.2:
            needy.append((ii, jj, r.E_int))
    if not needy: return 0, 0.0
    needy.sort(key=lambda x: x[2])
    ii, jj, _ = needy[0]
    surplus = donor.E_int - 0.3
    transfer = surplus * eff_sac * 0.5
    transfer = min(transfer, surplus * 0.8)
    donor.E_int -= transfer
    grid[ii][jj].E_int += transfer
    return 1, transfer

def vanity_consumption(cfg: SimConfig, cell: Cell, ph: Phenotype, neighbors_can_see):
    if cell.E_int < 0.5: return 0.0
    surplus = cell.E_int - 0.4
    vanity_spend = surplus * ph.vanity * 0.25
    if neighbors_can_see > 0:
        vanity_spend *= (1.0 + 0.4 * (neighbors_can_see / 8.0))
    vanity_spend = min(vanity_spend, cell.E_int)
    cell.E_int -= vanity_spend
    return vanity_spend

# ------------------------
# BO classification (3-type)
# ------------------------
def classify_bo_type(cfg: SimConfig, cell: Cell, ph: Phenotype, density: float, recent_actions: List[str]):
    genome_bo = (ph.coop < 0.4 and ph.move_prop < 0.4)
    idle_count = sum(1 for a in recent_actions[-10:] if a == "idle")
    behavioral_bo = (idle_count >= 4)
    if not (genome_bo and behavioral_bo):
        return "active"
    if cell.E_int > 0.25:   return "bo_strategic"
    elif cell.E_int > 0.10: return "bo_survivor"
    else:                   return "bo_dying"

# ------------------------
# Mutation
# ------------------------
def mutate(cfg: SimConfig, genome, rate, rng):
    mask = rng.random(cfg.G) < rate
    newg = genome.copy()
    newg[mask] = ~newg[mask]
    return newg

# ------------------------
# SOC threshold fields
# ------------------------
def init_thresholds(cfg: SimConfig):
    Lr = np.full((cfg.GRID_H, cfg.GRID_W), cfg.BASE_L_REPL, dtype=float)
    Le = np.full((cfg.GRID_H, cfg.GRID_W), cfg.BASE_L_EVOL, dtype=float)
    return Lr, Le

def local_variance(arr, i, j, neigh_fn):
    vals = [arr[ii,jj] for ii,jj in neigh_fn(i,j)] + [arr[i,j]]
    m = np.mean(vals)
    return float(np.mean([(v-m)**2 for v in vals]))

# ------------------------
# LCC (largest connected component) ratio (alive graph, 8‑neigh)
# ------------------------
def lcc_fraction_alive(cfg: SimConfig, grid):
    H,W = cfg.GRID_H, cfg.GRID_W
    visited = [[False]*W for _ in range(H)]
    def neigh8(i,j):
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                if di==0 and dj==0: continue
                ii,jj = i+di, j+dj
                if 0<=ii<H and 0<=jj<W:
                    yield ii,jj
    alive_total = 0
    best = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j].alive:
                alive_total += 1
    if alive_total == 0:
        return 0.0
    for i in range(H):
        for j in range(W):
            if grid[i][j].alive and not visited[i][j]:
                q = deque([(i,j)])
                visited[i][j] = True
                cnt = 0
                while q:
                    x,y = q.popleft()
                    cnt += 1
                    for ii,jj in neigh8(x,y):
                        if grid[ii][jj].alive and not visited[ii][jj]:
                            visited[ii][jj] = True
                            q.append((ii,jj))
                if cnt > best: best = cnt
    return best / alive_total

# ------------------------
# One simulation step (FULL)
# ------------------------
def step(cfg: SimConfig, grid, resource, Lthr_repl, Lthr_evol, t, rng,
         lineage_edges, critical_log):

    H,W = cfg.GRID_H, cfg.GRID_W
    next_grid = [[Cell(False, np.zeros(cfg.G, dtype=bool), -1, -1, 0, 0.0, []) for _ in range(W)] for _ in range(H)]

    total_fit = 0.0
    total_alive = 0
    total_coop = 0.0
    near_crit = 0
    L_map = np.zeros((H, W))

    pd_actions = np.zeros((H, W), dtype=int)
    harvest_map = np.zeros((H, W))
    phenos = [[None]*W for _ in range(H)]

    # Prepare phenotypes
    for i in range(H):
        for j in range(W):
            if grid[i][j].alive:
                phenos[i][j] = parse_genome(cfg, grid[i][j].genome)

    # Harvest & metabolism
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if not c.alive: continue
            ph = phenos[i][j]
            if cfg.SELFISH_HARVEST:
                selfish_gain = 0.5 + 0.5*(1.0 - ph.coop)
                harvest = cfg.HARVEST_RATE * resource[i,j] * ph.res_eff * selfish_gain
            else:
                harvest = cfg.HARVEST_RATE * resource[i,j] * ph.res_eff * ph.coop
            resource[i,j] = max(0.0, resource[i,j] - harvest)
            c.E_int += harvest
            harvest_map[i,j] = harvest
            dQ = cfg.METAB_RATE * c.E_int
            c.E_int = max(0.0, c.E_int - dQ)
            resource[i,j] += dQ

    # Sacrifice & Vanity
    total_vanity = 0.0
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if not c.alive: continue
            ph = phenos[i][j]
            _n, _tr = altruistic_transfer(cfg, grid, i, j, ph, rng)
            nb = neighbors(cfg, i, j)
            neighbors_can_see = sum(1 for ii,jj in nb if grid[ii][jj].alive)
            total_vanity += vanity_consumption(cfg, c, ph, neighbors_can_see)

    # PD game, fitness, Λ, and decide actions
    actions = [[("empty",0.0,0.0,None,0.0,0.0) for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if not c.alive:
                actions[i][j] = ("empty", 0.0, 0.0, c.genome, 0.0, c.E_int)
                continue
            ph = phenos[i][j]
            f, s, f1, f2, w1, w2 = fitness_multi(cfg, c.genome, i, j, t, ph)

            # neighbor support and density
            nb = neighbors(cfg, i, j)
            nei_acc = 0.0; nei_cnt = 0; occ = 0
            for ii,jj in nb:
                cc = grid[ii][jj]
                if cc.alive:
                    occ += 1
                    ff, *_ = fitness_multi(cfg, cc.genome, i, j, t, parse_genome(cfg, cc.genome))
                    nei_acc += ff; nei_cnt += 1
            local_density = occ/len(nb) if nb else 0.0
            nei = (nei_acc/nei_cnt) if nei_cnt>0 else 0.0

            # PD game
            gscore, my_act, _ = neighbor_game(cfg, grid, i, j, ph, rng)
            pd_actions[i,j] = my_act

            # V, K, Λ
            Eint_norm = min(1.0, c.E_int / cfg.E_NORM)
            Vpot_norm = min(1.0, resource[i,j] / (cfg.INIT_RESOURCE_MEAN + 1e-6))
            dens_st = density_stress(cfg, grid, i, j)

            # noblesse oblige small pressure around average neighbor energy
            nbE = [grid[ii][jj].E_int for ii,jj in nb if grid[ii][jj].alive]
            avgE = float(np.mean(nbE)) if nbE else 0.0
            nob = noblesse_oblige_pressure(c.E_int, avgE, ph.sacrifice)

            V = cfg.V_FIT_W*f + cfg.V_NEI_W*nei + cfg.V_GAME_W*gscore + cfg.V_POT_W*Vpot_norm + nob
            mismatch = 1.0 - f
            K = cfg.K_BASE + cfg.K_MISMATCH_SCALE*mismatch + cfg.K_GAME_SCALE*(1.0 - gscore) + cfg.K_ENERGY_W*Eint_norm + cfg.K_DENSITY_W*dens_st

            L = K / (V + 1e-6)
            L_map[i,j] = L

            total_fit  += f
            total_alive += 1
            total_coop += ph.coop
            if abs(L-1.0) < cfg.EPS_L_CRIT:
                near_crit += 1
                critical_log.append({'t':t,'i':i,'j':j,'Lambda':float(L),
                                     'fitness':float(f),'nei':float(nei),'gscore':float(gscore)})

            # Proposed action by Λ thresholds
            Lr = Lthr_repl[i,j]
            Le = Lthr_evol[i,j]
            if L < Lr:          proposed = "replicate"
            elif L <= Le:       proposed = "evolve"
            else:               proposed = "die"

            # Bounded-rationality: check net payoff; allow idle if negative or high density
            net_pay, costs = strategy_payoff(cfg, ph, local_density, proposed, harvest_map[i,j], gscore)
            if proposed == "replicate":
                if net_pay < 0:
                    proposed = "idle"
                else:
                    p_idle = max(0.0, (local_density - cfg.CRIT_DENSITY)/(1.0 - cfg.CRIT_DENSITY)) if local_density > cfg.CRIT_DENSITY else 0.0
                    p_idle *= 0.5
                    if rng.random() < p_idle:
                        proposed = "idle"

            # Deduct explicit action costs here (metabolism already applied)
            action_cost = 0.0
            for key in ("cooperation","movement","reproduction"):
                action_cost += costs.get(key, 0.0)
            c.E_int = max(0.0, c.E_int - action_cost)

            actions[i][j] = (proposed, L, f, c.genome, ph.move_prop, c.E_int)

    # Apply evolve / die / keep
    for i in range(H):
        for j in range(W):
            act, L, f, g, move_prop, Eint = actions[i][j]
            c = grid[i][j]
            if act == "die":
                next_grid[i][j] = Cell(False, np.zeros(cfg.G, dtype=bool), -1, -1, 0, 0.0, [])
            elif act == "evolve":
                ph = parse_genome(cfg, g)
                newg = mutate(cfg, g, cfg.MUT_HIGH * ph.mut_scale, rng)
                next_grid[i][j] = Cell(True, newg, c.id, c.parent, pd_actions[i,j], Eint, c.actions_hist[-11:]+["evolve"])
            elif act == "replicate":
                # parent keeps (small drift)
                ph = parse_genome(cfg, g)
                newg = mutate(cfg, g, cfg.MUT_LOW * ph.mut_scale, rng)
                next_grid[i][j] = Cell(True, newg, c.id, c.parent, pd_actions[i,j], Eint, c.actions_hist[-11:]+["replicate"])
            elif act == "idle":
                next_grid[i][j] = Cell(True, c.genome, c.id, c.parent, pd_actions[i,j], Eint, c.actions_hist[-11:]+["idle"])
            else:
                next_grid[i][j] = Cell(False, np.zeros(cfg.G, dtype=bool), -1, -1, 0, 0.0, [])

    # Movement (swap to richer site)
    if cfg.MOVE_ENABLE:
        claimed = set()
        for i in range(H):
            for j in range(W):
                if not next_grid[i][j].alive: continue
                act, L, _, _, move_prop, _ = actions[i][j]
                p_move = max(0.0, move_prop * min(1.0, max(0.0, L - 0.8)))
                if np.random.random() < p_move:
                    nb = neighbors(cfg, i, j)
                    empties = [(ii,jj) for (ii,jj) in nb if not next_grid[ii][jj].alive]
                    if empties:
                        empties.sort(key=lambda xy: resource[xy[0], xy[1]], reverse=True)
                        ii,jj = empties[0]
                        if (ii,jj) not in claimed:
                            # movement cost to resource (dissipation)
                            cost = min(cfg.MOVE_COST, next_grid[i][j].E_int)
                            next_grid[i][j].E_int -= cost
                            resource[i,j] += cost*0.5
                            resource[ii,jj] += cost*0.5
                            # swap
                            next_grid[ii][jj] = next_grid[i][j]
                            next_grid[i][j] = Cell(False, np.zeros(cfg.G, dtype=bool), -1, -1, 0, 0.0, [])
                            claimed.add((ii,jj))

    # Replication pass (child creation to empties)
    replicate_attempts = 0; replicate_successes = 0; child_failures = 0
    def neighbor_slots(i, j):
        nb = neighbors(cfg, i, j)
        empties, allies, others = [], [], []
        for ii,jj in nb:
            if not next_grid[ii][jj].alive:
                empties.append((ii,jj))
            else:
                hd = np.sum(next_grid[ii][jj].genome != next_grid[i][j].genome)
                if hd < cfg.G*0.25: allies.append((ii,jj))
                else: others.append((ii,jj))
        return empties, allies, others

    for i in range(H):
        for j in range(W):
            act, L, f, g, move_prop, Ein = actions[i][j]
            if act != "replicate": continue
            ph = parse_genome(cfg, g)
            empties, allies, others = neighbor_slots(i, j)
            choices = []
            if ph.repl_bias == 0:   choices = empties + allies + others
            elif ph.repl_bias == 1: choices = sorted(empties, key=lambda xy: -fitness_multi(cfg, g, xy[0], xy[1], t, ph)[0]) + allies + others
            else:                   choices = allies + empties + others
            nb = neighbors(cfg, i, j)
            density = (sum(1 for ii,jj in nb if next_grid[ii][jj].alive)/len(nb)) if nb else 0.0
            for ii,jj in choices:
                if not next_grid[ii][jj].alive:
                    replicate_attempts += 1
                    if rng.random() > child_survival_rate(cfg, density, 0.0):
                        child_failures += 1
                        break
                    parent = next_grid[i][j]
                    parent_ph = parse_genome(cfg, parent.genome)
                    newg = mutate(cfg, g, cfg.MUT_LOW * parent_ph.mut_scale, rng)
                    child_id = next_id()
                    share = min(0.1, parent.E_int)
                    next_grid[i][j].E_int -= share
                    child_E = share
                    next_grid[ii][jj] = Cell(True, newg, child_id, parent.id, np.random.randint(0,2), child_E, ["idle"])
                    lineage_edges.append((parent.id, child_id, t))
                    replicate_successes += 1
                    break

    # Resource diffusion
    resource = diffuse_resource(cfg, resource)

    # SOC: update thresholds via local variance of Λ
    new_Lr = Lthr_repl.copy()
    new_Le = Lthr_evol.copy()
    get_neigh = lambda i,j: neighbors(cfg, i, j)
    for i in range(H):
        for j in range(W):
            varL = local_variance(L_map, i, j, get_neigh)
            d = cfg.SOC_GAMMA * (varL - cfg.SOC_SIGMA0)
            new_Lr[i,j] = np.clip(Lthr_repl[i,j] + d, cfg.THR_MIN_REPL, cfg.THR_MAX_REPL)
            new_Le[i,j] = np.clip(Lthr_evol[i,j] + d, cfg.THR_MIN_EVOL, cfg.THR_MAX_EVOL)

    # Metrics
    avg_fit = total_fit/total_alive if total_alive>0 else 0.0
    frac_alive = total_alive/(H*W)
    avg_coop = total_coop/total_alive if total_alive>0 else 0.0
    frac_crit = near_crit/(H*W)
    lcc_frac = lcc_fraction_alive(cfg, next_grid)
    rep_rate = (replicate_successes/max(1,total_alive)) if total_alive>0 else 0.0
    child_fail_rate = (child_failures/max(1,replicate_attempts)) if replicate_attempts>0 else 0.0

    return (next_grid, resource, new_Lr, new_Le,
            avg_fit, frac_alive, avg_coop, frac_crit,
            lcc_frac, rep_rate, child_fail_rate, total_vanity)

# ------------------------
# Single simulation
# ------------------------
def simulate_once(cfg: SimConfig, seed: int, save_prefix: str):
    _ensure_dir(cfg.OUTDIR)
    rng = rng_for(seed)
    grid, resource = init_grid(cfg, rng)
    Lthr_repl, Lthr_evol = init_thresholds(cfg)

    # Initial H0
    H0_init = total_internal_energy(cfg, grid) + float(np.sum(resource))

    # Logs
    avg_fits, alive_fracs, avg_coops, crit_fracs = [], [], [], []
    lcc_fracs, rep_rates, child_fail_rates = [], [], []
    bo_strat_fracs, bo_surv_fracs, bo_dy_fracs = [], [], []
    mean_local_density_ts = []
    vanity_total_ts = []
    H0_drifts = []
    lineage_edges = []
    critical_log = []

    for t in range(cfg.STEPS):
        # One step
        grid, resource, Lthr_repl, Lthr_evol, a, z, c, q, lcc, rep, chf, vanity_total = step(
            cfg, grid, resource, Lthr_repl, Lthr_evol, t, rng, lineage_edges, critical_log
        )
        # Density and BO classification counts (after state update)
        sum_local_density = 0.0; alive_cells_for_density = 0
        bo_s, bo_v, bo_d = 0,0,0
        for i in range(cfg.GRID_H):
            for j in range(cfg.GRID_W):
                if not grid[i][j].alive: continue
                ph = parse_genome(cfg, grid[i][j].genome)
                nb = neighbors(cfg, i, j)
                occ = sum(1 for ii,jj in nb if grid[ii][jj].alive)
                local_density = (occ/len(nb)) if nb else 0.0
                sum_local_density += local_density; alive_cells_for_density += 1
                tlabel = classify_bo_type(cfg, grid[i][j], ph, local_density, grid[i][j].actions_hist)
                if tlabel == "bo_strategic": bo_s += 1
                elif tlabel == "bo_survivor": bo_v += 1
                elif tlabel == "bo_dying":    bo_d += 1

        mean_local_density = (sum_local_density/max(1,alive_cells_for_density)) if alive_cells_for_density>0 else 0.0

        avg_fits.append(a); alive_fracs.append(z); avg_coops.append(c); crit_fracs.append(q)
        lcc_fracs.append(lcc); rep_rates.append(rep); child_fail_rates.append(chf)
        bo_strat_fracs.append(bo_s/(cfg.GRID_H*cfg.GRID_W))
        bo_surv_fracs.append( bo_v/(cfg.GRID_H*cfg.GRID_W))
        bo_dy_fracs.append(  bo_d/(cfg.GRID_H*cfg.GRID_W))
        mean_local_density_ts.append(mean_local_density)
        vanity_total_ts.append(vanity_total)
        H0_now = total_internal_energy(cfg, grid) + float(np.sum(resource))
        H0_drifts.append(H0_now - H0_init)

    # Final maps
    final_fit  = np.zeros((cfg.GRID_H, cfg.GRID_W))
    final_coop = np.zeros((cfg.GRID_H, cfg.GRID_W))
    for i in range(cfg.GRID_H):
        for j in range(cfg.GRID_W):
            if grid[i][j].alive:
                ph = parse_genome(cfg, grid[i][j].genome)
                f, *_ = fitness_multi(cfg, grid[i][j].genome, i, j, cfg.STEPS, ph)
                final_fit[i, j] = f
                final_coop[i, j] = ph.coop

    # Save logs
    base = os.path.join(cfg.OUTDIR, f"{cfg.TAG}_{save_prefix}")
    _ensure_dir(base)

    # CSVs
    pd.DataFrame({
        "t": np.arange(cfg.STEPS),
        "alive": alive_fracs,
        "avg_fit": avg_fits,
        "avg_coop": avg_coops,
        "crit_frac": crit_fracs,
        "lcc_frac": lcc_fracs,
        "rep_rate": rep_rates,
        "child_fail": child_fail_rates,
        "bo_strategic": bo_strat_fracs,
        "bo_survivor": bo_surv_fracs,
        "bo_dying": bo_dy_fracs,
        "mean_local_density": mean_local_density_ts,
        "vanity_total": vanity_total_ts,
        "H0_drift": H0_drifts
    }).to_csv(os.path.join(base, "timeseries.csv"), index=False)

    with open(os.path.join(base, "critical_events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t","i","j","Lambda","fitness","nei","gscore"])
        w.writeheader(); [w.writerow(row) for row in critical_log]

    with open(os.path.join(base, "lineage_edges.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["parent_id","child_id","t"])
        for p,c,t in lineage_edges: w.writerow([p,c,t])

    with open(os.path.join(base, "final_grid_meta.json"), "w") as f:
        meta = [[{"id": int(grid[i][j].id), "parent": int(grid[i][j].parent), "alive": bool(grid[i][j].alive)}
                 for j in range(cfg.GRID_W)] for i in range(cfg.GRID_H)]
        json.dump(meta, f)

    # Plots (each one figure; no explicit colors)
    # 1) Core time series
    def plot_ts(y, title, ylabel, filename, xlab="Step"):
        plt.figure(); plt.plot(y); plt.title(title); plt.xlabel(xlab); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
        _savefig(os.path.join(base, filename))

    plot_ts(avg_fits, "Average Fitness vs Time", "Average Fitness", "avg_fitness.pdf")
    plot_ts(alive_fracs, "Alive Fraction vs Time", "Alive Fraction", "alive_fraction.pdf")
    plot_ts(avg_coops, "Average Cooperation vs Time", "Average Cooperation", "avg_coop.pdf")
    plot_ts(crit_fracs, "Near-Critical Fraction (|Λ-1|<ε)", "Near-Critical Fraction", "crit_frac.pdf")
    plot_ts(H0_drifts, "Energy Conservation Drift H0-H0_init", "Drift", "H0_drift.pdf")
    plot_ts(lcc_fracs, "LCC Fraction vs Time", "LCC Fraction", "lcc_frac.pdf")
    plot_ts(rep_rates, "Reproduction Rate vs Time", "Reproduction Rate", "rep_rate.pdf")
    plot_ts(child_fail_rates, "Child Failure Rate vs Time", "Child Failure Rate", "child_fail_rate.pdf")
    plot_ts(bo_strat_fracs, "BO Strategic Fraction vs Time", "BO Strategic", "bo_strategic.pdf")
    plot_ts(bo_surv_fracs,  "BO Survivor Fraction vs Time",  "BO Survivor",  "bo_survivor.pdf")
    plot_ts(bo_dy_fracs,    "BO Dying Fraction vs Time",     "BO Dying",     "bo_dying.pdf")
    plot_ts(mean_local_density_ts, "Mean Local Density vs Time", "Mean Local Density", "mean_local_density.pdf")
    plot_ts(vanity_total_ts, "Vanity Consumption (Total) vs Time", "Vanity Total", "vanity_total.pdf")

    # 2) Heatmaps (final)
    def plot_heatmap(arr, title, filename, xlabel="j", ylabel="i"):
        plt.figure(); plt.imshow(arr, interpolation='nearest'); plt.title(title); plt.colorbar(); plt.xlabel(xlabel); plt.ylabel(ylabel)
        _savefig(os.path.join(base, filename))

    plot_heatmap(final_fit,  "Final Grid Fitness", "final_fitness_heatmap.pdf")
    plot_heatmap(final_coop, "Final Grid Cooperation", "final_coop_heatmap.pdf")
    plot_heatmap(resource,   "Final Resource Field", "final_resource_heatmap.pdf")

    # 3) Lineage tree (time-layered)
    # Build birth times from lineage edges
    birth_time = {}
    children_map = defaultdict(list)
    for p,c,t in lineage_edges:
        if c not in birth_time or t < birth_time[c]:
            birth_time[c] = t
        if p not in birth_time:
            birth_time[p] = 0
        children_map[p].append(c)
    nodes = sorted(birth_time.keys(), key=lambda nid: birth_time[nid])
    x_positions = {}; y_positions = {}; layer_counts = defaultdict(int)
    for nid in nodes:
        x = birth_time[nid]
        layer_counts[x] += 1
        y = layer_counts[x]
        x_positions[nid] = x; y_positions[nid] = y
    plt.figure(figsize=(8,6))
    for p,c,t in lineage_edges:
        if p in x_positions and c in x_positions:
            xs = [x_positions[p], x_positions[c]]
            ys = [y_positions[p], y_positions[c]]
            plt.plot(xs, ys)
    plt.scatter([x_positions[n] for n in nodes], [y_positions[n] for n in nodes])
    plt.title("Lineage Tree (time-layered)"); plt.xlabel("Birth step"); plt.ylabel("Node index per layer")
    _savefig(os.path.join(base, "lineage_tree.png"))

    # Return collected series for higher-level analytics
    return {
        "base_dir": base,
        "series": pd.DataFrame({
            "t": np.arange(cfg.STEPS),
            "alive": alive_fracs,
            "rep_rate": rep_rates,
            "lcc_frac": lcc_fracs,
            "E_local_norm": np.array(vanity_total_ts)*0.0  # placeholder: can recompute if needed
        }),
        "alive": np.array(alive_fracs),
        "rep_rate": np.array(rep_rates),
        "lcc_frac": np.array(lcc_fracs)
    }

# ------------------------
# Event detection & statistics
# ------------------------
def detect_spikes(series: np.ndarray, q: int=80) -> Tuple[List[int], float]:
    thr = np.percentile(series, q)
    idx = []
    for t in range(1, len(series)-1):
        if series[t] >= thr and series[t] > series[t-1] and series[t] >= series[t+1]:
            idx.append(t)
    return idx, float(thr)

def slope_between(y: np.ndarray, t0: int, t1: int) -> float:
    if t1 <= t0: return 0.0
    return float((y[t1] - y[t0]) / (t1 - t0))

def bootstrap_ci(values: List[float], B=5000, q=(2.5,97.5), seed=1234):
    if len(values)==0: return (None,None)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [np.mean(arr[rng.integers(0, len(arr), size=len(arr))]) for _ in range(B)]
    lo, hi = np.percentile(boots, q[0]), np.percentile(boots, q[1])
    return float(lo), float(hi)

def binomial_sign_test(values: List[float]) -> Optional[float]:
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    n = pos + neg
    if n == 0: return None
    # Two-sided p under Binomial(n, 0.5)
    def logcomb(n,k): return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
    def pmf(n,k): return math.exp(logcomb(n,k) - n*math.log(2.0))
    up  = sum(pmf(n,k) for k in range(pos, n+1))
    low = sum(pmf(n,k) for k in range(0, pos+1))
    return float(min(1.0, 2*min(up, low)))

# ------------------------
# τ_delay (social death → extinction)
# ------------------------
def compute_tau_delay(alive: np.ndarray, rep: np.ndarray, lcc: np.ndarray, EPS_REP: float, LCC_THRES: float, K_STABLE: int):
    # t_global: rep<=EPS for K consecutive and LCC<=THRES
    t_global = None; streak = 0
    for t in range(len(alive)):
        if rep[t] <= EPS_REP: streak += 1
        else: streak = 0
        if streak >= K_STABLE and lcc[t] <= LCC_THRES:
            t_global = t - K_STABLE + 1
            break
    # t_local: Alive==0 for K consecutive
    t_local = None; zstreak = 0
    for t in range(len(alive)):
        if alive[t] <= 0.0: zstreak += 1
        else: zstreak = 0
        if zstreak >= K_STABLE:
            t_local = t - K_STABLE + 1
            break
    # Censoring → set to last step
    if t_global is None: t_global = len(alive)-1
    if t_local  is None: t_local  = len(alive)-1
    return int(t_global), int(t_local), int(t_local - t_global)

# ------------------------
# Ensemble runner
# ------------------------
def run_ensemble(cfg: SimConfig, N: int, base_seed: int, do_events: bool=True, tag_suffix: str="Nensemble"):
    all_res = []
    for r in range(N):
        seed = base_seed + r
        res = simulate_once(cfg, seed, save_prefix=f"{tag_suffix}_run{r:02d}")
        all_res.append(res)

    # Aggregate time series (pad with NaN if lengths vary)
    L = cfg.STEPS
    metrics = {
        "alive": np.vstack([np.array(res["alive"]) for res in all_res]),
        "rep_rate": np.vstack([np.array(res["rep_rate"]) for res in all_res]),
        "lcc_frac": np.vstack([np.array(res["lcc_frac"]) for res in all_res]),
    }
    mean_ts = {k: np.nanmean(v, axis=0) for k,v in metrics.items()}
    std_ts  = {k: np.nanstd (v, axis=0) for k,v in metrics.items()}
    sem_ts  = {k: std_ts[k]/np.sqrt(N)  for k in metrics.keys()}

    # τ_delay per run
    tau_rows = []
    for r,res in enumerate(all_res):
        tg, tl, tau = compute_tau_delay(res["alive"], res["rep_rate"], res["lcc_frac"],
                                        cfg.EPS_REP, cfg.LCC_THRES, cfg.K_STABLE)
        tau_rows.append([r, tg, tl, tau])

    out_ens = os.path.join(cfg.OUTDIR, f"{cfg.TAG}_ensemble")
    _ensure_dir(out_ens)
    pd.DataFrame({
        "t": np.arange(L),
        **{f"{k}_mean": mean_ts[k] for k in metrics.keys()},
        **{f"{k}_std":  std_ts[k]  for k in metrics.keys()},
        **{f"{k}_sem":  sem_ts[k]  for k in metrics.keys()},
    }).to_csv(os.path.join(out_ens, f"{tag_suffix}_aggregates.csv"), index=False)
    pd.DataFrame(tau_rows, columns=["run","t_global","t_local","tau_delay"]).to_csv(
        os.path.join(out_ens, f"{tag_suffix}_tau_delay.csv"), index=False
    )

    # Event-aligned around BO_dying & BO_strategic (replay from per-run CSV if needed)
    # For speed, we recompute spikes from per-run bo series CSVs written in simulate_once (timeseries.csv).
    # Align on “bo_dying” & “bo_strategic”; compute pre/post slopes for Alive.
    if do_events:
        ev_tables = []
        for series_name in ["bo_dying","bo_strategic"]:
            diffs = []
            post  = []
            rows  = []
            segments = []
            for r in range(N):
                base_dir = os.path.join(cfg.OUTDIR, f"{cfg.TAG}_{tag_suffix}_run{r:02d}")
                ts = pd.read_csv(os.path.join(base_dir, "timeseries.csv"))
                sig = ts[series_name].to_numpy(dtype=float)
                alive = ts["alive"].to_numpy(dtype=float)
                idx, thr = detect_spikes(sig, q=cfg.SPIKE_Q)
                for t0 in idx:
                    L0 = max(0, t0 - cfg.PRE_W)
                    R0 = min(len(alive)-1, t0 + cfg.POST_W)
                    if R0 <= t0 or t0 - L0 < 2:
                        continue
                    sp = slope_between(alive, L0, t0)
                    so = slope_between(alive, t0, R0)
                    rows.append([r, t0, series_name, float(sig[t0]), float(thr), float(sp), float(so), float(so-sp)])
                    diffs.append(so-sp); post.append(so)
                    # segments for event-aligned mean ± 95% CI
                    Lseg = t0 - cfg.PRE_W; Rseg = t0 + cfg.POST_W + 1
                    if Lseg >= 0 and Rseg <= len(alive):
                        segments.append(alive[Lseg:Rseg])
            ev_df = pd.DataFrame(rows, columns=["run","t_spike","series","value_at_spike","threshold_used","alive_pre_slope","alive_post_slope","alive_slope_diff"])
            ev_df.to_csv(os.path.join(out_ens, f"{tag_suffix}_events_{series_name}.csv"), index=False)

            # Summary
            summary = {
                "series": series_name,
                "N_runs": N,
                "N_events": len(rows),
                "alive_slope_post_mean": (float(np.mean(post)) if post else None),
                "alive_slope_post_CI95": (bootstrap_ci(post) if post else (None,None)),
                "alive_slope_diff_mean": (float(np.mean(diffs)) if diffs else None),
                "alive_slope_diff_CI95": (bootstrap_ci(diffs) if diffs else (None,None)),
                "sign_test_p_two_sided": (binomial_sign_test(diffs) if diffs else None),
            }
            with open(os.path.join(out_ens, f"{tag_suffix}_events_{series_name}_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # Event-aligned figure (Alive)
            if len(segments)>0:
                seg_arr = np.array(segments)
                mean_seg = np.mean(seg_arr, axis=0)
                sem_seg  = np.std(seg_arr, axis=0)/np.sqrt(seg_arr.shape[0])
                ci_seg   = 1.96*sem_seg
                x = np.arange(-cfg.PRE_W, cfg.POST_W+1)
                plt.figure(figsize=(10,4.2))
                plt.plot(x, mean_seg); plt.fill_between(x, mean_seg-ci_seg, mean_seg+ci_seg, alpha=0.25)
                plt.axvline(0, linestyle='--')
                plt.xlabel("Time from spike (steps)"); plt.ylabel("Alive Fraction"); plt.grid(True, alpha=0.3)
                _savefig(os.path.join(out_ens, f"{tag_suffix}_event_aligned_alive_{series_name}.pdf"))

    return out_ens

# ------------------------
# Finite-size scaling (τ_delay vs L)
# ------------------------
def scan_sizes(base_cfg: SimConfig, sizes: List[int], steps_scale: float=3.0, N_per_size: int=5, base_seed: int=7000):
    rows = []
    for k,L in enumerate(sizes):
        cfg = SimConfig(**{**base_cfg.__dict__})
        cfg.GRID_H = L; cfg.GRID_W = L
        cfg.STEPS = int(max(base_cfg.STEPS, steps_scale*L))  # longer run for bigger L
        cfg.TAG = f"{base_cfg.TAG}_L{L}"
        out_dir = run_ensemble(cfg, N=N_per_size, base_seed=base_seed + 1000*k, do_events=False, tag_suffix=f"L{L}_N{N_per_size}")
        # Load tau csv
        tau_csv = os.path.join(out_dir, f"L{L}_N{N_per_size}_tau_delay.csv")
        df = pd.read_csv(tau_csv)
        tau_mean = float(df["tau_delay"].mean())
        tau_std  = float(df["tau_delay"].std(ddof=1)) if len(df)>1 else 0.0
        rows.append([L, cfg.STEPS, N_per_size, tau_mean, tau_std, out_dir])

    result = pd.DataFrame(rows, columns=["L","STEPS","N","tau_delay_mean","tau_delay_std","out_dir"])
    result.to_csv(os.path.join(base_cfg.OUTDIR, f"{base_cfg.TAG}_finite_size_tau.csv"), index=False)

    # Plot log-log
    if len(result) >= 2:
        x = result["L"].to_numpy(dtype=float)
        y = result["tau_delay_mean"].to_numpy(dtype=float)
        logx = np.log(x+1e-9); logy = np.log(y+1e-9)
        A = np.vstack([logx, np.ones_like(logx)]).T
        slope, intercept = np.linalg.lstsq(A, logy, rcond=None)[0]
        plt.figure(); plt.plot(x, y, marker='o'); plt.xscale("log"); plt.yscale("log")
        plt.xlabel("L (grid size)"); plt.ylabel("tau_delay (mean)")
        plt.title(f"Finite-size scaling: slope ≈ {slope:.3f}")
        _savefig(os.path.join(base_cfg.OUTDIR, f"{base_cfg.TAG}_finite_size_loglog.pdf"))
        with open(os.path.join(base_cfg.OUTDIR, f"{base_cfg.TAG}_finite_size_fit.json"), "w") as f:
            json.dump({"slope": float(slope), "intercept": float(intercept)}, f, indent=2)

    return result

# ------------------------
# MAIN (example)
# ------------------------
if __name__ == "__main__":
    # Base config close to Universe‑25 collapse regime
    cfg = SimConfig(
        GRID_H=48, GRID_W=48, STEPS=365, SEED_DENSITY=0.08,
        BASE_L_REPL=0.95, BASE_L_EVOL=1.10,
        SOC_GAMMA=0.06, SOC_SIGMA0=0.04,
        THR_MIN_REPL=0.88, THR_MAX_REPL=0.98,
        THR_MIN_EVOL=1.05, THR_MAX_EVOL=1.30,
        MUT_LOW=0.008, MUT_HIGH=0.08,
        K_BASE=0.05, K_MISMATCH_SCALE=0.5, K_GAME_SCALE=0.2, K_ENERGY_W=0.2, K_DENSITY_W=0.8,
        V_FIT_W=0.50, V_NEI_W=0.20, V_GAME_W=0.15, V_POT_W=0.10,
        INIT_RESOURCE_MEAN=2.0, INIT_RESOURCE_NOISE=0.2, INIT_EINT_IF_ALIVE=0.5,
        HARVEST_RATE=0.20, METAB_RATE=0.015, MOVE_COST=0.02, RESOURCE_DIFFUSION=0.08,
        E_NORM=1.0, SELFISH_HARVEST=True,
        CRIT_DENSITY=0.65,
        PRE_W=10, POST_W=20, SPIKE_Q=80,
        EPS_REP=1e-3, LCC_THRES=0.30, K_STABLE=5,
        OUTDIR="./outputs", TAG="U25_FULL", BASE_SEED=6000
    )

    _ensure_dir(cfg.OUTDIR)

    # 1) Single run (quick check)
    #res = simulate_once(cfg, seed=cfg.BASE_SEED, save_prefix="single")

    # 2) Ensemble (set True to run; adjust N for Colab time)
    RUN_ENSEMBLE = True
    if RUN_ENSEMBLE:
        out_ens = run_ensemble(cfg, N=10, base_seed=cfg.BASE_SEED+100, do_events=True, tag_suffix="N10")
        print("Ensemble outputs →", out_ens)

    # 3) Finite-size scaling (optional; can be time-consuming)
    SCAN_SIZES = False
    if SCAN_SIZES:
        sizes = [28, 36, 48, 64]  # extend as needed
        res_fs = scan_sizes(cfg, sizes=sizes, steps_scale=3.2, N_per_size=5, base_seed=7000)
        print(res_fs)
