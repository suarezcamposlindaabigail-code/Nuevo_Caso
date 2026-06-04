# 🚛 LogiFast — Crossdocking MIP Optimizer

**Curso II-1122 · UCR Alajuela · Branch-and-Bound**  
Referencia: Yu & Egbelu (2008)

---

## Descripción del Problema

LogiFast es un centro de crossdocking donde **camiones de entrada (inbound)** descargan productos en el muelle de recepción, y esos productos se transfieren directamente a **camiones de salida (outbound)** en el muelle de despacho. El objetivo es minimizar el **makespan** (tiempo total de operación).

### Variables de Decisión

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `T` | Continua | Makespan (tiempo total) |
| `c[i]` | Continua | Llegada del camión inbound i al muelle de recepción |
| `F[i]` | Continua | Salida del camión inbound i del muelle de recepción |
| `d[j]` | Continua | Llegada del camión outbound j al muelle de despacho |
| `L[j]` | Continua | Salida del camión outbound j del muelle de despacho |
| `x[i,j,k]` | Entera ≥ 0 | Unidades del producto k transferidas de i a j |
| `v[i,j]` | Binaria | = 1 si hay alguna transferencia de i a j |
| `p[i,i']` | Binaria | = 1 si el camión inbound i precede a i' en el muelle |
| `q[j,j']` | Binaria | = 1 si el camión outbound j precede a j' en el muelle |

### Función Objetivo

```
min T
```

### 13 Restricciones Canónicas

| # | Restricción | Descripción |
|---|-------------|-------------|
| (1) | `T ≥ L[j]  ∀j` | Makespan ≥ salida del último camión de despacho |
| (2) | `Σⱼ x[i,j,k] = r[i,k]  ∀i,k` | Todo lo de camión i producto k se transfiere |
| (3) | `Σᵢ x[i,j,k] = s[j,k]  ∀j,k` | Demanda de camión j producto k se cumple |
| (4) | `x[i,j,k] ≤ M·v[i,j]  ∀i,j,k` | Vincula cantidades con binaria v (Big-M) |
| (5) | `F[i] ≥ c[i] + Σk r[i,k]  ∀i` | Inbound no sale hasta terminar descarga |
| (6) | `c[i'] ≥ F[i] - M(1-p[i,i'])  ∀i≠i'` | Secuencia inbound (parte A) |
| (7) | `c[i] ≥ F[i'] - M·p[i,i']  ∀i≠i'` | Secuencia inbound (parte B) |
| (8) | `p[i,i] = 0  ∀i` | Sin auto-precedencia inbound |
| (9) | `L[j] ≥ d[j] + Σk s[j,k]  ∀j` | Outbound no parte hasta completar carga |
| (10) | `d[j'] ≥ L[j] - M(1-q[j,j'])  ∀j≠j'` | Secuencia outbound (parte A) |
| (11) | `d[j] ≥ L[j'] - M·q[j,j']  ∀j≠j'` | Secuencia outbound (parte B) |
| (12) | `q[j,j] = 0  ∀j` | Sin auto-precedencia outbound |
| (13) | `L[j] ≥ c[i] + V·Σk x[i,j,k] - M(1-v[i,j])  ∀i,j` | Vincula salida j con llegada i si hay transferencia |

---

## Instancia TS5

```
i = 5  (camiones de entrada)
o = 3  (camiones de salida)
n = 8  (tipos de producto)
```

---

## Instalación y ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/logifast-mip.git
cd logifast-mip

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

---

## Archivos AMPL

Para resolver con AMPL + solver externo (Gurobi, CPLEX, etc.):

```bash
ampl
model logifast.mod;
data TS5.dat;
option solver gurobi;
solve;
display T, c, F, d, L;
```

---

## Estructura del proyecto

```
logifast/
├── app.py              # Aplicación Streamlit principal
├── logifast_solver.py  # Solver MIP (PuLP + CBC)
├── logifast.mod        # Modelo AMPL
├── TS5.dat             # Datos instancia TS5 (AMPL)
├── requirements.txt    # Dependencias Python
└── README.md           # Este archivo
```

---

## Despliegue en Streamlit Cloud

1. Fork este repositorio en tu GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Selecciona `app.py` como archivo principal
5. ¡Deploy!

---

**UCR Alajuela · II-1122 · 2026**
