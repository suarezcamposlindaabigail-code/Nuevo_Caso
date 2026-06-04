# =============================================================================
# LogiFast — Modelo MIP de Crossdocking
# Basado en Yu & Egbelu (2008)
# Curso II-1122 | UCR Alajuela
# =============================================================================
# ÍNDICES
# i, i2 : camiones de entrada (inbound), i = 1..I
# j, j2 : camiones de salida  (outbound), j = 1..O
# k      : tipo de producto,  k = 1..N

param I integer > 0;   # número de camiones de entrada
param O integer > 0;   # número de camiones de salida
param N integer > 0;   # número de tipos de producto

set INBOUND  := 1..I;
set OUTBOUND := 1..O;
set PRODUCTS := 1..N;

# Parámetros de carga / demanda
param r {INBOUND,  PRODUCTS} >= 0 default 0;   # unidades del producto k en camión i
param s {OUTBOUND, PRODUCTS} >= 0 default 0;   # demanda del producto k en camión j

# Tiempo de transferencia (minutos por unidad) — asumido 1 min/unidad si no se especifica
param V >= 0 default 1;

# Big-M para las restricciones de secuencia
param BigM >= 0 default 100000;

# =============================================================================
# VARIABLES DE DECISIÓN
# =============================================================================
# — Continuas
var T >= 0;                          # makespan (tiempo total de operación)
var c {INBOUND}  >= 0;               # llegada del camión i al muelle de recepción
var F {INBOUND}  >= 0;               # salida  del camión i del muelle de recepción
var d {OUTBOUND} >= 0;               # llegada del camión j al muelle de despacho
var L {OUTBOUND} >= 0;               # salida  del camión j del muelle de despacho

# — Enteras (no negativas)
var x {INBOUND, OUTBOUND, PRODUCTS} integer >= 0;   # unidades del producto k transferidas de i a j

# — Binarias
var v {INBOUND, OUTBOUND}  binary;   # 1 si algún producto va del camión i al camión j
var p {INBOUND, INBOUND}   binary;   # 1 si camión inbound i precede a i2 en muelle recepción
var q {OUTBOUND, OUTBOUND} binary;   # 1 si camión outbound j precede a j2 en muelle despacho

# =============================================================================
# FUNCIÓN OBJETIVO
# =============================================================================
minimize makespan: T;

# =============================================================================
# RESTRICCIONES
# =============================================================================

# (1) Makespan >= salida de cada camión de salida
subject to R01 {j in OUTBOUND}:
    T >= L[j];

# (2) Todo lo cargado en el camión i del producto k debe transferirse exactamente
subject to R02 {i in INBOUND, k in PRODUCTS}:
    sum {j in OUTBOUND} x[i,j,k] = r[i,k];

# (3) Demanda del camión j del producto k se cumple exactamente
subject to R03 {j in OUTBOUND, k in PRODUCTS}:
    sum {i in INBOUND} x[i,j,k] = s[j,k];

# (4) Vincula cantidades x con binaria v (Big-M)
subject to R04 {i in INBOUND, j in OUTBOUND, k in PRODUCTS}:
    x[i,j,k] <= BigM * v[i,j];

# (5) El camión i no puede salir antes de terminar de descargar
subject to R05 {i in INBOUND}:
    F[i] >= c[i] + sum {k in PRODUCTS} r[i,k];

# (6-7) Solo un camión inbound a la vez en muelle de recepción (Big-M con p)
subject to R06 {i in INBOUND, i2 in INBOUND: i <> i2}:
    c[i2] >= F[i] - BigM * (1 - p[i,i2]);

subject to R07 {i in INBOUND, i2 in INBOUND: i <> i2}:
    c[i]  >= F[i2] - BigM * p[i,i2];

# (8) Un camión inbound no puede precederse a sí mismo
subject to R08 {i in INBOUND}:
    p[i,i] = 0;

# (9) El camión j no puede salir antes de completar su carga
subject to R09 {j in OUTBOUND}:
    L[j] >= d[j] + sum {k in PRODUCTS} s[j,k];

# (10-11) Solo un camión outbound a la vez en muelle de despacho (Big-M con q)
subject to R10 {j in OUTBOUND, j2 in OUTBOUND: j <> j2}:
    d[j2] >= L[j] - BigM * (1 - q[j,j2]);

subject to R11 {j in OUTBOUND, j2 in OUTBOUND: j <> j2}:
    d[j]  >= L[j2] - BigM * q[j,j2];

# (12) Un camión outbound no puede precederse a sí mismo
subject to R12 {j in OUTBOUND}:
    q[j,j] = 0;

# (13) Vincula salida de j con llegada de i si hay transferencia (V = tiempo unitario)
subject to R13 {i in INBOUND, j in OUTBOUND}:
    L[j] >= c[i] + V * sum {k in PRODUCTS} x[i,j,k] - BigM * (1 - v[i,j]);
