"""
app.py — LogiFast Crossdocking Optimizer
Curso II-1122 | UCR Alajuela | Branch-and-Bound
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from logifast_solver import solve_logifast, TS5_PARAMS

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="LogiFast · Crossdocking MIP",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.main { background: #0f1117; }

/* Título hero */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f59e0b, #ef4444, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Tarjetas de métricas */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f59e0b;
    font-family: 'JetBrains Mono', monospace;
}
.metric-unit {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* Badges de restricciones */
.rest-badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #f59e0b;
    margin-right: 0.3rem;
}

/* Sección headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 4px solid #f59e0b;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* Tabla de flujo */
.flow-cell-active {
    background: #065f46 !important;
    color: #34d399 !important;
    font-weight: 600;
}

/* Fórmula matemática */
.math-block {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
    margin: 0.5rem 0;
    white-space: pre-wrap;
}
.math-highlight { color: #f59e0b; }
.math-comment   { color: #64748b; font-style: italic; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: navegación ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚛 **LogiFast**")
    st.markdown('<p style="color:#64748b;font-size:0.8rem;">Optimización de Crossdocking · II-1122</p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Sección",
        ["📊 Instancia TS5", "🔧 Modo Paramétrico", "📐 Formulación MIP"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<p style="color:#475569;font-size:0.72rem;">Referencia: Yu & Egbelu (2008)<br>Solver: CBC via PuLP<br>UCR Alajuela · 2026</p>', unsafe_allow_html=True)

# =============================================================================
# PÁGINA 1 — INSTANCIA TS5
# =============================================================================
if page == "📊 Instancia TS5":

    st.markdown('<p class="hero-title">Instancia TS5</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Solución óptima del modelo MIP con datos reales de LogiFast</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Parámetros TS5 ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📦 Parámetros del archivo TS5</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Camiones Entrada (i)</div>
          <div class="metric-value">5</div>
          <div class="metric-unit">inbound trucks R1–R5</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Camiones Salida (o)</div>
          <div class="metric-value">3</div>
          <div class="metric-unit">outbound trucks S1–S3</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-label">Tipos de Producto (n)</div>
          <div class="metric-value">8</div>
          <div class="metric-unit">productos P1–P8</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Matrices r y s
    col_r, col_s = st.columns(2)
    I, O, N = 5, 3, 8
    r_data = TS5_PARAMS["r"]
    s_data = TS5_PARAMS["s"]

    with col_r:
        st.markdown("**r\[i,k\] — Carga en camiones de ENTRADA** (unidades)")
        r_matrix = [[r_data.get((i, k), 0) for k in range(1, N+1)] for i in range(1, I+1)]
        df_r = pd.DataFrame(r_matrix, index=[f"R{i}" for i in range(1, I+1)],
                            columns=[f"P{k}" for k in range(1, N+1)])
        # totales
        df_r["TOTAL"] = df_r.sum(axis=1)
        st.dataframe(
            df_r.style
                .highlight_max(axis=1, subset=[f"P{k}" for k in range(1, N+1)], color="#065f46")
                .format("{:.0f}")
                .map(lambda v: "background-color:#1e293b; color:#64748b" if v == 0 else ""),
            use_container_width=True
        )

    with col_s:
        st.markdown("**s\[j,k\] — Demanda en camiones de SALIDA** (unidades)")
        s_matrix = [[s_data.get((j, k), 0) for k in range(1, N+1)] for j in range(1, O+1)]
        df_s = pd.DataFrame(s_matrix, index=[f"S{j}" for j in range(1, O+1)],
                            columns=[f"P{k}" for k in range(1, N+1)])
        df_s["TOTAL"] = df_s.sum(axis=1)
        st.dataframe(
            df_s.style
                .highlight_max(axis=1, subset=[f"P{k}" for k in range(1, N+1)], color="#1e3a5f")
                .format("{:.0f}")
                .map(lambda v: "background-color:#1e293b; color:#64748b" if v == 0 else ""),
            use_container_width=True
        )

    # Balance oferta/demanda
    st.markdown("**✅ Balance Oferta = Demanda por producto**")
    balance_data = []
    for k in range(1, N+1):
        supply = sum(r_data.get((i, k), 0) for i in range(1, I+1))
        demand = sum(s_data.get((j, k), 0) for j in range(1, O+1))
        balance_data.append({"Producto": f"P{k}", "Oferta (Σ r)": supply, "Demanda (Σ s)": demand, "Balance": supply - demand})
    df_balance = pd.DataFrame(balance_data)
    st.dataframe(
        df_balance.style.map(
            lambda v: "color: #34d399; font-weight:600" if v == 0 else "color:#ef4444",
            subset=["Balance"]
        ).format({"Oferta (Σ r)": "{:.0f}", "Demanda (Σ s)": "{:.0f}", "Balance": "{:+.0f}"}),
        use_container_width=True, hide_index=True
    )

    # Gráfico de distribución de carga
    st.markdown('<p class="section-header">📊 Distribución de carga por camión y producto</p>', unsafe_allow_html=True)

    fig_heat = go.Figure()
    fig_heat.add_trace(go.Heatmap(
        z=r_matrix, x=[f"P{k}" for k in range(1, N+1)],
        y=[f"R{i}" for i in range(1, I+1)],
        colorscale="YlOrRd", showscale=True,
        hovertemplate="Camión %{y}<br>Producto %{x}<br>Unidades: %{z}<extra></extra>",
        text=[[str(v) if v > 0 else "" for v in row] for row in r_matrix],
        texttemplate="%{text}", textfont={"size": 13},
    ))
    fig_heat.update_layout(
        title="Carga por camión de entrada (r[i,k])",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e2e8f0", family="Space Grotesk"),
        height=300
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Resolución TS5 ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔬 Resolución Óptima — TS5</p>', unsafe_allow_html=True)

    with st.spinner("⚙️ Ejecutando solver CBC (MIP)…"):
        result = solve_logifast(**TS5_PARAMS, time_limit=180)

    if result.makespan is not None:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Makespan Óptimo</div>
              <div class="metric-value">{result.makespan:.0f}</div>
              <div class="metric-unit">minutos</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Estado</div>
              <div class="metric-value" style="font-size:1.2rem;color:#34d399">✓ {result.status}</div>
              <div class="metric-unit">{result.message}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Tiempo de Cómputo</div>
              <div class="metric-value">{result.solve_time:.1f}</div>
              <div class="metric-unit">segundos</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            transfers = sum(1 for (i,j),val in result.v.items() if val > 0.5)
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Pares con Transferencia</div>
              <div class="metric-value">{transfers}</div>
              <div class="metric-unit">de {I*O} combinaciones (i,j)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Secuencias óptimas
        col_seq1, col_seq2 = st.columns(2)
        with col_seq1:
            st.markdown("#### 🟠 Secuencia óptima — Muelle de RECEPCIÓN")
            seq_in_data = []
            for rank, i in enumerate(result.sequence_inbound, 1):
                seq_in_data.append({
                    "Posición": f"#{rank}",
                    "Camión": f"R{i}",
                    "Llegada c[i]": f"{result.c.get(i, 0):.1f} min",
                    "Salida F[i]": f"{result.F.get(i, 0):.1f} min",
                    "Duración": f"{result.F.get(i,0) - result.c.get(i,0):.1f} min",
                })
            st.dataframe(pd.DataFrame(seq_in_data), use_container_width=True, hide_index=True)

        with col_seq2:
            st.markdown("#### 🔵 Secuencia óptima — Muelle de DESPACHO")
            seq_out_data = []
            for rank, j in enumerate(result.sequence_outbound, 1):
                seq_out_data.append({
                    "Posición": f"#{rank}",
                    "Camión": f"S{j}",
                    "Llegada d[j]": f"{result.d.get(j, 0):.1f} min",
                    "Salida L[j]": f"{result.L.get(j, 0):.1f} min",
                    "Duración": f"{result.L.get(j,0) - result.d.get(j,0):.1f} min",
                })
            st.dataframe(pd.DataFrame(seq_out_data), use_container_width=True, hide_index=True)

        # Diagrama de Gantt
        st.markdown('<p class="section-header">📅 Diagrama de Gantt — Operación del Centro</p>', unsafe_allow_html=True)

        gantt_data = []
        for i in range(1, I+1):
            gantt_data.append(dict(
                Task=f"R{i} (entrada)", Start=result.c.get(i, 0), Finish=result.F.get(i, 0),
                Resource="Muelle Recepción", Truck=f"R{i}"
            ))
        for j in range(1, O+1):
            gantt_data.append(dict(
                Task=f"S{j} (salida)", Start=result.d.get(j, 0), Finish=result.L.get(j, 0),
                Resource="Muelle Despacho", Truck=f"S{j}"
            ))

        fig_gantt = go.Figure()
        colors = {"Muelle Recepción": "#f59e0b", "Muelle Despacho": "#3b82f6"}
        for row in gantt_data:
            fig_gantt.add_trace(go.Bar(
                x=[row["Finish"] - row["Start"]],
                y=[row["Task"]],
                base=[row["Start"]],
                orientation="h",
                marker_color=colors[row["Resource"]],
                name=row["Resource"],
                showlegend=False,
                hovertemplate=f"<b>{row['Task']}</b><br>Inicio: {row['Start']:.1f} min<br>Fin: {row['Finish']:.1f} min<br>Duración: {row['Finish']-row['Start']:.1f} min<extra></extra>",
            ))

        # Línea de makespan
        fig_gantt.add_vline(x=result.makespan, line_dash="dash", line_color="#ef4444", line_width=2,
                            annotation_text=f"Makespan = {result.makespan:.0f} min",
                            annotation_font_color="#ef4444")
        fig_gantt.update_layout(
            barmode="overlay",
            paper_bgcolor="#0f1117", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0", family="Space Grotesk"),
            height=350,
            xaxis=dict(title="Tiempo (minutos)", gridcolor="#334155"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

        # Flujo de transferencias x[i,j,k]
        st.markdown('<p class="section-header">🔄 Flujo de Transferencias x[i,j,k]</p>', unsafe_allow_html=True)

        flow_data = []
        for i in range(1, I+1):
            for j in range(1, O+1):
                for k in range(1, N+1):
                    val = result.x.get((i, j, k), 0)
                    if val > 0:
                        flow_data.append({
                            "Inbound": f"R{i}", "Outbound": f"S{j}",
                            "Producto": f"P{k}", "Unidades": val
                        })

        if flow_data:
            df_flow = pd.DataFrame(flow_data)
            # Pivot para visualizar como matriz
            pivot = df_flow.groupby(["Inbound", "Outbound"])["Unidades"].sum().unstack(fill_value=0)
            st.markdown("**Suma de unidades transferidas por par (i → j)**")
            st.dataframe(pivot.style.background_gradient(cmap="YlOrRd").format("{:.0f}"), use_container_width=True)

            st.markdown("**Detalle completo de transferencias activas**")
            st.dataframe(df_flow.sort_values(["Inbound", "Outbound", "Producto"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron transferencias activas.")

    else:
        st.error(f"❌ El solver no encontró solución: {result.message}")

# =============================================================================
# PÁGINA 2 — MODO PARAMÉTRICO
# =============================================================================
elif page == "🔧 Modo Paramétrico":

    st.markdown('<p class="hero-title">Modo Paramétrico</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Define tu propia instancia de crossdocking y resuelve el MIP</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("### ⚙️ Parámetros de la instancia")
        I_p = st.slider("Camiones de entrada (i)", 1, 10, 3)
        O_p = st.slider("Camiones de salida (o)", 1, 8, 2)
        N_p = st.slider("Tipos de producto (n)", 1, 10, 4)
        V_p = st.number_input("Tiempo unitario V (min/unidad)", 0.1, 10.0, 1.0, 0.1)
        BigM_p = st.number_input("Big-M", 1000, 1_000_000, 100_000, step=10_000)
        time_lim = st.slider("Límite de tiempo solver (seg)", 10, 300, 60)

    st.markdown('<p class="section-header">📥 Matriz de carga r[i,k]</p>', unsafe_allow_html=True)
    st.caption("Ingresa las unidades del producto k en el camión de entrada i (0 = no lleva ese producto)")

    r_default = pd.DataFrame(
        np.zeros((I_p, N_p), dtype=int),
        index=[f"R{i}" for i in range(1, I_p+1)],
        columns=[f"P{k}" for k in range(1, N_p+1)],
    )
    df_r_edit = st.data_editor(r_default, use_container_width=True, key="r_editor")

    st.markdown('<p class="section-header">📤 Matriz de demanda s[j,k]</p>', unsafe_allow_html=True)
    st.caption("Ingresa la demanda del producto k en el camión de salida j")

    s_default = pd.DataFrame(
        np.zeros((O_p, N_p), dtype=int),
        index=[f"S{j}" for j in range(1, O_p+1)],
        columns=[f"P{k}" for k in range(1, N_p+1)],
    )
    df_s_edit = st.data_editor(s_default, use_container_width=True, key="s_editor")

    # Verificar balance
    balance_ok = True
    msgs = []
    for k_idx, k_col in enumerate(df_r_edit.columns):
        supply = int(df_r_edit[k_col].sum())
        demand = int(df_s_edit[k_col].sum())
        if supply != demand:
            msgs.append(f"Producto {k_col}: oferta={supply} ≠ demanda={demand}")
            balance_ok = False

    if not balance_ok:
        st.warning("⚠️ **Balance incumplido** — Las restricciones (2) y (3) requieren que Σᵢ r[i,k] = Σⱼ s[j,k] para cada k:\n\n" + "\n".join(msgs))
    else:
        st.success("✅ Balance correcto: oferta = demanda para todos los productos.")

    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run = st.button("🚀 Resolver MIP", use_container_width=True, disabled=not balance_ok)

    if run:
        r_param = {}
        s_param = {}
        for i in range(1, I_p+1):
            for k in range(1, N_p+1):
                val = int(df_r_edit.iloc[i-1, k-1])
                if val > 0:
                    r_param[(i, k)] = val
        for j in range(1, O_p+1):
            for k in range(1, N_p+1):
                val = int(df_s_edit.iloc[j-1, k-1])
                if val > 0:
                    s_param[(j, k)] = val

        with st.spinner("⚙️ Resolviendo…"):
            res = solve_logifast(I_p, O_p, N_p, r_param, s_param, V_p, BigM_p, time_lim)

        if res.makespan is not None:
            st.success(f"🎯 **Makespan óptimo: {res.makespan:.0f} minutos** — {res.message} ({res.solve_time:.2f}s)")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Secuencia Inbound**")
                rows = [{"Posición": f"#{r}", "Camión": f"R{i}",
                         "c[i]": f"{res.c.get(i,0):.1f}", "F[i]": f"{res.F.get(i,0):.1f}"}
                        for r, i in enumerate(res.sequence_inbound, 1)]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**Secuencia Outbound**")
                rows = [{"Posición": f"#{r}", "Camión": f"S{j}",
                         "d[j]": f"{res.d.get(j,0):.1f}", "L[j]": f"{res.L.get(j,0):.1f}"}
                        for r, j in enumerate(res.sequence_outbound, 1)]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Gantt
            gantt_rows = []
            for i in range(1, I_p+1):
                gantt_rows.append(dict(Task=f"R{i}", Start=res.c.get(i,0), Finish=res.F.get(i,0), Dock="Recepción"))
            for j in range(1, O_p+1):
                gantt_rows.append(dict(Task=f"S{j}", Start=res.d.get(j,0), Finish=res.L.get(j,0), Dock="Despacho"))

            fig2 = go.Figure()
            pal = {"Recepción": "#f59e0b", "Despacho": "#3b82f6"}
            for row in gantt_rows:
                fig2.add_trace(go.Bar(
                    x=[row["Finish"] - row["Start"]], y=[row["Task"]],
                    base=[row["Start"]], orientation="h",
                    marker_color=pal[row["Dock"]],
                    hovertemplate=f"<b>{row['Task']}</b><br>{row['Start']:.1f}→{row['Finish']:.1f} min<extra></extra>",
                    showlegend=False
                ))
            fig2.add_vline(x=res.makespan, line_dash="dash", line_color="#ef4444",
                           annotation_text=f"T* = {res.makespan:.0f}", annotation_font_color="#ef4444")
            fig2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e293b",
                               font=dict(color="#e2e8f0"), height=300,
                               xaxis_title="Tiempo (minutos)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error(f"❌ {res.message}")

# =============================================================================
# PÁGINA 3 — FORMULACIÓN MIP
# =============================================================================
elif page == "📐 Formulación MIP":

    st.markdown('<p class="hero-title">Formulación MIP Completa</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Modelo matemático de optimización · Yu & Egbelu (2008) · 13 restricciones canónicas</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Índices y variables
    st.markdown('<p class="section-header">📌 Índices y Conjuntos</p>', unsafe_allow_html=True)
    idx_data = [
        ("i", "1 … I", "Camiones de ENTRADA (inbound)"),
        ("j", "1 … O", "Camiones de SALIDA (outbound)"),
        ("k", "1 … N", "Tipos de producto"),
    ]
    st.dataframe(pd.DataFrame(idx_data, columns=["Símbolo", "Rango", "Descripción"]),
                 hide_index=True, use_container_width=True)

    st.markdown('<p class="section-header">📊 Parámetros</p>', unsafe_allow_html=True)
    param_data = [
        ("r[i,k]", "≥ 0, entero", "Unidades del producto k que lleva el camión de entrada i"),
        ("s[j,k]", "≥ 0, entero", "Unidades del producto k que demanda el camión de salida j"),
        ("V",      "≥ 0, real",   "Tiempo de transferencia por unidad (min/unidad)"),
        ("M",      "≫ 0",         "Constante Big-M para restricciones lógicas"),
    ]
    st.dataframe(pd.DataFrame(param_data, columns=["Símbolo", "Dominio", "Descripción"]),
                 hide_index=True, use_container_width=True)

    st.markdown('<p class="section-header">🔵 Variables de Decisión</p>', unsafe_allow_html=True)
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.markdown("**Continuas**")
        cont_vars = [
            ("T", "Makespan — tiempo total de operación"),
            ("cᵢ", "Llegada del camión i al muelle de recepción"),
            ("Fᵢ", "Salida del camión i del muelle de recepción"),
            ("dⱼ", "Llegada del camión j al muelle de despacho"),
            ("Lⱼ", "Salida del camión j del muelle de despacho"),
        ]
        for sym, desc in cont_vars:
            st.markdown(f'<span class="rest-badge">{sym}</span> {desc}<br>', unsafe_allow_html=True)
    with col_v2:
        st.markdown("**Enteras (≥ 0)**")
        st.markdown('<span class="rest-badge">x[i,j,k]</span> Unidades del producto k transferidas del camión i al camión j', unsafe_allow_html=True)
    with col_v3:
        st.markdown("**Binarias {0,1}**")
        bin_vars = [
            ("v[i,j]", "= 1 si algún producto se transfiere de i a j"),
            ("p[i,i']", "= 1 si el camión inbound i precede a i' en el muelle de recepción"),
            ("q[j,j']", "= 1 si el camión outbound j precede a j' en el muelle de despacho"),
        ]
        for sym, desc in bin_vars:
            st.markdown(f'<span class="rest-badge">{sym}</span> {desc}<br>', unsafe_allow_html=True)

    # FO
    st.markdown('<p class="section-header">🎯 Función Objetivo</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="math-block">
<span class="math-highlight">min  T</span>

<span class="math-comment">-- Minimizar el tiempo total de operación del centro de distribución (makespan)</span>
    </div>
    """, unsafe_allow_html=True)

    # Restricciones
    st.markdown('<p class="section-header">📋 13 Restricciones del Modelo</p>', unsafe_allow_html=True)

    restricciones = [
        ("(1)", "T ≥ Lⱼ  ∀j",
         "El makespan debe ser mayor o igual a la salida de cada camión de despacho.",
         "T ≥ L[j]   para todo j ∈ {1…O}"),
        ("(2)", "Σⱼ x[i,j,k] = r[i,k]  ∀i,k",
         "Todo lo cargado en el camión i del producto k debe transferirse exactamente a los camiones de salida.",
         "sum{j} x[i,j,k] = r[i,k]   para todo i ∈ {1…I}, k ∈ {1…N}"),
        ("(3)", "Σᵢ x[i,j,k] = s[j,k]  ∀j,k",
         "La demanda del camión j del producto k se cumple exactamente con lo que recibe de los inbound.",
         "sum{i} x[i,j,k] = s[j,k]   para todo j ∈ {1…O}, k ∈ {1…N}"),
        ("(4)", "x[i,j,k] ≤ M·v[i,j]  ∀i,j,k",
         "Vincula las cantidades transferidas con la variable binaria v (Big-M). Si v[i,j]=0 no puede haber transferencia.",
         "x[i,j,k] ≤ M · v[i,j]   para todo i,j,k"),
        ("(5)", "Fᵢ ≥ cᵢ + Σk r[i,k]  ∀i",
         "El camión de entrada i no puede salir del muelle de recepción hasta terminar de descargar todos sus productos.",
         "F[i] ≥ c[i] + sum{k} r[i,k]   para todo i"),
        ("(6)", "cᵢ' ≥ Fᵢ − M·(1−p[i,i'])  ∀i≠i'",
         "Secuencia inbound parte A: si p[i,i']=1 (i precede a i'), entonces i' no puede llegar antes de que i salga.",
         "c[i'] ≥ F[i] - M·(1-p[i,i'])   para todo i ≠ i'"),
        ("(7)", "cᵢ ≥ Fᵢ' − M·p[i,i']  ∀i≠i'",
         "Secuencia inbound parte B: complemento de (6). Garantiza que solo un camión use el muelle a la vez.",
         "c[i] ≥ F[i'] - M·p[i,i']   para todo i ≠ i'"),
        ("(8)", "p[i,i] = 0  ∀i",
         "Un camión inbound no puede precederse a sí mismo (elimina auto-precedencia).",
         "p[i,i] = 0   para todo i"),
        ("(9)", "Lⱼ ≥ dⱼ + Σk s[j,k]  ∀j",
         "El camión de salida j no puede partir hasta que haya completado de cargar todos los productos demandados.",
         "L[j] ≥ d[j] + sum{k} s[j,k]   para todo j"),
        ("(10)", "dⱼ' ≥ Lⱼ − M·(1−q[j,j'])  ∀j≠j'",
         "Secuencia outbound parte A: si q[j,j']=1 (j precede a j'), entonces j' no puede llegar antes de que j parta.",
         "d[j'] ≥ L[j] - M·(1-q[j,j'])   para todo j ≠ j'"),
        ("(11)", "dⱼ ≥ Lⱼ' − M·q[j,j']  ∀j≠j'",
         "Secuencia outbound parte B: complemento de (10). Solo un camión usa el muelle de despacho a la vez.",
         "d[j] ≥ L[j'] - M·q[j,j']   para todo j ≠ j'"),
        ("(12)", "q[j,j] = 0  ∀j",
         "Un camión outbound no puede precederse a sí mismo.",
         "q[j,j] = 0   para todo j"),
        ("(13)", "Lⱼ ≥ cᵢ + V·Σk x[i,j,k] − M·(1−v[i,j])  ∀i,j",
         "Vincula la salida del camión j con la llegada del camión i y el tiempo de transferencia, si hay flujo entre ellos.",
         "L[j] ≥ c[i] + V·sum{k}x[i,j,k] - M·(1-v[i,j])   para todo i,j"),
    ]

    group_colors = {
        "(1)": "#f59e0b", "(2)": "#3b82f6", "(3)": "#3b82f6",
        "(4)": "#8b5cf6", "(5)": "#10b981", "(6)": "#ef4444",
        "(7)": "#ef4444", "(8)": "#ef4444", "(9)": "#10b981",
        "(10)": "#f97316", "(11)": "#f97316", "(12)": "#f97316",
        "(13)": "#8b5cf6",
    }
    group_labels = {
        "(1)": "Makespan", "(2)": "Flujo", "(3)": "Flujo",
        "(4)": "Big-M", "(5)": "Inbound", "(6)": "Secuencia R",
        "(7)": "Secuencia R", "(8)": "Secuencia R", "(9)": "Outbound",
        "(10)": "Secuencia S", "(11)": "Secuencia S", "(12)": "Secuencia S",
        "(13)": "Vinculación",
    }

    for num, formula, desc, impl in restricciones:
        color = group_colors.get(num, "#64748b")
        label = group_labels.get(num, "")
        with st.expander(f"{num} **{formula}** — _{label}_"):
            st.markdown(f"**Interpretación:** {desc}")
            st.markdown(f"""
            <div class="math-block">
<span class="math-highlight">{impl}</span>
            </div>
            """, unsafe_allow_html=True)
            # Instancia TS5 concreta
            if num == "(2)":
                st.markdown("**Ejemplo TS5:** Para i=2, k=1: Σⱼ x[2,j,1] = r[2,1] = **6 unidades**")
            elif num == "(3)":
                st.markdown("**Ejemplo TS5:** Para j=1, k=7: Σᵢ x[i,1,7] = s[1,7] = **98 unidades**")
            elif num == "(5)":
                st.markdown("**Ejemplo TS5:** Para i=2: F[2] ≥ c[2] + (6+6+19+50+38+6+19+56) = c[2] + **200 min**")
            elif num == "(9)":
                st.markdown("**Ejemplo TS5:** Para j=2: L[2] ≥ d[2] + (150+217) = d[2] + **367 min**")

    # Conteo de restricciones expandidas para TS5
    st.markdown('<p class="section-header">🔢 Desglose de restricciones en TS5 (i=5, o=3, n=8)</p>', unsafe_allow_html=True)
    I_ts, O_ts, N_ts = 5, 3, 8
    desglose = [
        ("(1)", "T ≥ Lⱼ", "O", O_ts, O_ts),
        ("(2)", "Σⱼ x[i,j,k] = r[i,k]", "I × N", I_ts * N_ts, I_ts * N_ts),
        ("(3)", "Σᵢ x[i,j,k] = s[j,k]", "O × N", O_ts * N_ts, O_ts * N_ts),
        ("(4)", "x[i,j,k] ≤ M·v[i,j]", "I × O × N", I_ts * O_ts * N_ts, I_ts * O_ts * N_ts),
        ("(5)", "Fᵢ ≥ cᵢ + ...", "I", I_ts, I_ts),
        ("(6)", "cᵢ' ≥ Fᵢ − ...", "I×(I−1)", "I(I-1)", I_ts * (I_ts-1)),
        ("(7)", "cᵢ ≥ Fᵢ' − ...", "I×(I−1)", "I(I-1)", I_ts * (I_ts-1)),
        ("(8)", "p[i,i] = 0", "I", I_ts, I_ts),
        ("(9)", "Lⱼ ≥ dⱼ + ...", "O", O_ts, O_ts),
        ("(10)", "dⱼ' ≥ Lⱼ − ...", "O×(O−1)", "O(O-1)", O_ts * (O_ts-1)),
        ("(11)", "dⱼ ≥ Lⱼ' − ...", "O×(O−1)", "O(O-1)", O_ts * (O_ts-1)),
        ("(12)", "q[j,j] = 0", "O", O_ts, O_ts),
        ("(13)", "Lⱼ ≥ cᵢ + ...", "I × O", I_ts * O_ts, I_ts * O_ts),
    ]
    df_desglose = pd.DataFrame(desglose, columns=["Restricción", "Fórmula", "Cardinalidad", "Fórmula TS5", "# en TS5"])
    total_rest = sum(d[4] for d in desglose)
    st.dataframe(df_desglose, hide_index=True, use_container_width=True)
    st.info(f"**Total de restricciones en TS5: {total_rest}** (expandidas desde las 13 canónicas)")

    # Variables en TS5
    st.markdown('<p class="section-header">📐 Dimensión del modelo en TS5</p>', unsafe_allow_html=True)
    var_dim = [
        ("T", "Continua", 1),
        ("cᵢ", "Continua", I_ts),
        ("Fᵢ", "Continua", I_ts),
        ("dⱼ", "Continua", O_ts),
        ("Lⱼ", "Continua", O_ts),
        ("x[i,j,k]", "Entera ≥ 0", I_ts * O_ts * N_ts),
        ("v[i,j]", "Binaria", I_ts * O_ts),
        ("p[i,i']", "Binaria", I_ts * I_ts),
        ("q[j,j']", "Binaria", O_ts * O_ts),
    ]
    df_vars = pd.DataFrame(var_dim, columns=["Variable", "Tipo", "# en TS5"])
    df_vars.loc[len(df_vars)] = ["TOTAL", "—", df_vars["# en TS5"].sum()]
    st.dataframe(df_vars, hide_index=True, use_container_width=True)
