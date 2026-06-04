"""
logifast_solver.py
==================
Resuelve el MIP de crossdocking LogiFast usando PuLP + CBC.
Acepta los parámetros i, o, n, r, s y retorna la solución completa.

Referencia: Yu & Egbelu (2008)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import pulp


@dataclass
class SolverResult:
    status: str
    makespan: Optional[float]
    sequence_inbound: list[int]
    sequence_outbound: list[int]
    x: dict          # x[(i,j,k)] = unidades transferidas
    c: dict          # c[i] = llegada inbound i
    F: dict          # F[i] = salida inbound i
    d: dict          # d[j] = llegada outbound j
    L: dict          # L[j] = salida outbound j
    v: dict          # v[(i,j)] = 1 si hay transferencia i->j
    solve_time: float
    message: str = ""


def solve_logifast(
    I: int,
    O: int,
    N: int,
    r: dict,   # r[(i,k)] = unidades del producto k en camión i
    s: dict,   # s[(j,k)] = demanda del producto k en camión j
    V: float = 1.0,
    BigM: float = 100_000,
    time_limit: int = 120,
) -> SolverResult:
    """
    Resuelve el modelo MIP de crossdocking.

    Parámetros
    ----------
    I        : número de camiones de entrada
    O        : número de camiones de salida
    N        : número de tipos de producto
    r        : dict {(i,k): cantidad}, i∈[1..I], k∈[1..N]
    s        : dict {(j,k): cantidad}, j∈[1..O], k∈[1..N]
    V        : tiempo unitario de transferencia (min/unidad)
    BigM     : constante M para restricciones lógicas
    time_limit: límite de tiempo en segundos
    """
    t0 = time.time()

    inbound  = range(1, I + 1)
    outbound = range(1, O + 1)
    products = range(1, N + 1)

    # Rellenar con 0 los valores no especificados
    r_full = {(i, k): r.get((i, k), 0) for i in inbound for k in products}
    s_full = {(j, k): s.get((j, k), 0) for j in outbound for k in products}

    # ── Modelo ────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("LogiFast_Crossdocking", pulp.LpMinimize)

    # Variables continuas
    T_var = pulp.LpVariable("T", lowBound=0)
    c_var = {i: pulp.LpVariable(f"c_{i}", lowBound=0) for i in inbound}
    F_var = {i: pulp.LpVariable(f"F_{i}", lowBound=0) for i in inbound}
    d_var = {j: pulp.LpVariable(f"d_{j}", lowBound=0) for j in outbound}
    L_var = {j: pulp.LpVariable(f"L_{j}", lowBound=0) for j in outbound}

    # Variables enteras
    x_var = {
        (i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", lowBound=0, cat="Integer")
        for i in inbound for j in outbound for k in products
    }

    # Variables binarias
    v_var = {
        (i, j): pulp.LpVariable(f"v_{i}_{j}", cat="Binary")
        for i in inbound for j in outbound
    }
    p_var = {
        (i, i2): pulp.LpVariable(f"p_{i}_{i2}", cat="Binary")
        for i in inbound for i2 in inbound
    }
    q_var = {
        (j, j2): pulp.LpVariable(f"q_{j}_{j2}", cat="Binary")
        for j in outbound for j2 in outbound
    }

    # ── Función objetivo ──────────────────────────────────────────────────────
    prob += T_var, "Minimizar_Makespan"

    # ── Restricciones ─────────────────────────────────────────────────────────

    # (1) T >= L[j] para todo j
    for j in outbound:
        prob += T_var >= L_var[j], f"R01_j{j}"

    # (2) Todo lo de camión i producto k se transfiere
    for i in inbound:
        for k in products:
            prob += pulp.lpSum(x_var[i, j, k] for j in outbound) == r_full[i, k], f"R02_i{i}_k{k}"

    # (3) Demanda del camión j producto k se cumple
    for j in outbound:
        for k in products:
            prob += pulp.lpSum(x_var[i, j, k] for i in inbound) == s_full[j, k], f"R03_j{j}_k{k}"

    # (4) x[i,j,k] <= BigM * v[i,j]
    for i in inbound:
        for j in outbound:
            for k in products:
                prob += x_var[i, j, k] <= BigM * v_var[i, j], f"R04_i{i}_j{j}_k{k}"

    # (5) F[i] >= c[i] + sum_k r[i,k]
    for i in inbound:
        total_r = sum(r_full[i, k] for k in products)
        prob += F_var[i] >= c_var[i] + total_r, f"R05_i{i}"

    # (6-7) Solo un inbound a la vez en muelle recepción
    for i in inbound:
        for i2 in inbound:
            if i != i2:
                prob += c_var[i2] >= F_var[i] - BigM * (1 - p_var[i, i2]), f"R06_i{i}_i2{i2}"
                prob += c_var[i]  >= F_var[i2] - BigM * p_var[i, i2],      f"R07_i{i}_i2{i2}"

    # (8) p[i,i] = 0
    for i in inbound:
        prob += p_var[i, i] == 0, f"R08_i{i}"

    # (9) L[j] >= d[j] + sum_k s[j,k]
    for j in outbound:
        total_s = sum(s_full[j, k] for k in products)
        prob += L_var[j] >= d_var[j] + total_s, f"R09_j{j}"

    # (10-11) Solo un outbound a la vez en muelle despacho
    for j in outbound:
        for j2 in outbound:
            if j != j2:
                prob += d_var[j2] >= L_var[j] - BigM * (1 - q_var[j, j2]), f"R10_j{j}_j2{j2}"
                prob += d_var[j]  >= L_var[j2] - BigM * q_var[j, j2],      f"R11_j{j}_j2{j2}"

    # (12) q[j,j] = 0
    for j in outbound:
        prob += q_var[j, j] == 0, f"R12_j{j}"

    # (13) L[j] >= c[i] + V*sum_k x[i,j,k] - BigM*(1-v[i,j])
    for i in inbound:
        for j in outbound:
            prob += (
                L_var[j] >= c_var[i]
                + V * pulp.lpSum(x_var[i, j, k] for k in products)
                - BigM * (1 - v_var[i, j]),
                f"R13_i{i}_j{j}",
            )

    # ── Resolver ──────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    elapsed = time.time() - t0
    status = pulp.LpStatus[prob.status]

    if prob.status not in (1, -2):  # 1=Optimal, -2=Not solved / infeasible
        return SolverResult(
            status=status,
            makespan=None,
            sequence_inbound=[],
            sequence_outbound=[],
            x={}, c={}, F={}, d={}, L={}, v={},
            solve_time=elapsed,
            message=f"Solver terminó con estado: {status}",
        )

    # ── Extraer resultados ────────────────────────────────────────────────────
    makespan_val = pulp.value(T_var)

    c_vals = {i: pulp.value(c_var[i]) for i in inbound}
    F_vals = {i: pulp.value(F_var[i]) for i in inbound}
    d_vals = {j: pulp.value(d_var[j]) for j in outbound}
    L_vals = {j: pulp.value(L_var[j]) for j in outbound}

    x_vals = {
        (i, j, k): round(pulp.value(x_var[i, j, k]) or 0)
        for i in inbound for j in outbound for k in products
    }
    v_vals = {
        (i, j): round(pulp.value(v_var[i, j]) or 0)
        for i in inbound for j in outbound
    }

    # Orden de atención inbound (por c[i])
    seq_in  = sorted(inbound,  key=lambda i: c_vals.get(i, 0))
    # Orden de atención outbound (por d[j])
    seq_out = sorted(outbound, key=lambda j: d_vals.get(j, 0))

    return SolverResult(
        status=status,
        makespan=makespan_val,
        sequence_inbound=seq_in,
        sequence_outbound=seq_out,
        x=x_vals,
        c=c_vals,
        F=F_vals,
        d=d_vals,
        L=L_vals,
        v=v_vals,
        solve_time=elapsed,
        message="Solución óptima encontrada" if status == "Optimal" else f"Mejor cota: {makespan_val}",
    )


# ── Instancia TS5 predefinida ─────────────────────────────────────────────────
TS5_PARAMS = {
    "I": 5,
    "O": 3,
    "N": 8,
    "r": {
        (1, 1): 170,
        (2, 1): 6,  (2, 2): 6,  (2, 3): 19, (2, 4): 50, (2, 5): 38, (2, 6): 6,  (2, 7): 19, (2, 8): 56,
        (3, 1): 49, (3, 2): 31, (3, 3): 60,              (3, 6): 12, (3, 7): 37, (3, 8): 31,
        (4, 5): 143,(4, 7): 47,
        (5, 4): 58, (5, 5): 36, (5, 7): 72, (5, 8): 14,
    },
    "s": {
        (1, 1): 75, (1, 2): 12, (1, 3): 59,              (1, 6): 9,  (1, 7): 98, (1, 8): 40,
        (2, 1): 150,(2, 5): 217,
        (3, 2): 25, (3, 3): 20, (3, 4): 108,(3, 6): 9,  (3, 7): 77, (3, 8): 61,
    },
}
