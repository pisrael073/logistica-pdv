import streamlit as st
import pandas as pd
import gspread
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import cdist
from math import radians, cos, sin, asin, sqrt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Ruta Táctica de Cobranza", layout="wide", page_icon="🕵️‍♂️")

# Coordenada Base (Ambato) para iniciar el cálculo de ruta
BODEGA_AMBATO = np.array([[-1.241667, -78.619722]])

# --- MOTORES MATEMÁTICOS (Logística) ---
def ordenar_ruta_cobranza(df_grupo):
    """Ordena los deudores para hacer una ruta lógica y no dar vueltas"""
    if len(df_grupo) < 1: return df_grupo
    
    # Preparamos coordenadas
    coords = df_grupo[['Latitud', 'Longitud']].values
    pendientes = list(range(len(coords)))
    orden_indices = []
    
    # PASO 1: Empezar por el cliente más cercano a la base (Ambato)
    # Opcional: Podríamos empezar por el que debe MÁS dinero, pero gastaríamos mucha gasolina.
    # Usaremos optimización geográfica.
    distancias_bodega = cdist(BODEGA_AMBATO, coords, metric='euclidean')
    idx_primero = np.argmin(distancias_bodega)
    orden_indices.append(idx_primero)
    pendientes.remove(idx_primero)
    
    # PASO 2: Encadenar el resto (Vecino más cercano)
    while pendientes:
        ultimo_idx = orden_indices[-1]
        ultimo_punto = coords[ultimo_idx].reshape(1, -1)
        
        coords_pendientes = coords[pendientes]
        distancias = cdist(ultimo_punto, coords_pendientes, metric='euclidean')
        
        idx_local = np.argmin(distancias)
        idx_real = pendientes[idx_local]
        
        orden_indices.append(idx_real)
        pendientes.remove(idx_real)
        
    return df_grupo.iloc[orden_indices].reset_index(drop=True)

# --- CARGA DE DATOS (Financiera + GPS) ---
@st.cache_data(ttl=600)
def cargar_datos_fusionados():
    try:
        credenciales = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(credenciales)
        sh = gc.open("soluto")

        # Cargar Hojas
        df_cartera = pd.DataFrame(sh.worksheet("CARTERA").get_all_records())
        df_master = pd.DataFrame(sh.worksheet("CLIENTES_MASTER").get_all_records())

        # Limpieza Columnas
        df_cartera.columns = [col.strip() for col in df_cartera.columns]
        df_master.columns = [col.strip() for col in df_master.columns]

        # ID como Texto para el cruce
        col_id_cartera = 'Identificacion' if 'Identificacion' in df_cartera.columns else 'RUC'
        col_id_master = 'CEDULA O RUC'
        df_cartera['ID_CRUCE'] = df_cartera[col_id_cartera].astype(str).str.strip()
        df_master['ID_CRUCE'] = df_master[col_id_master].astype(str).str.strip()

        # GPS
        if 'Ubicacion GPS' in df_master.columns:
            lat_lon = df_master['Ubicacion GPS'].astype(str).str.split(',', expand=True)
            if len(lat_lon.columns) >= 2:
                df_master['Latitud'] = pd.to_numeric(lat_lon[0], errors='coerce')
                df_master['Longitud'] = pd.to_numeric(lat_lon[1], errors='coerce')

        # Limpieza Dinero ($)
        df_cartera['Saldo'] = df_cartera['Saldo'].astype(str).str.replace('$', '').str.replace(',', '')
        df_cartera['Saldo'] = pd.to_numeric(df_cartera['Saldo'], errors='coerce').fillna(0)
        df_cartera['Dias'] = pd.to_numeric(df_cartera['Dias'], errors='coerce').fillna(0)

        # Cruce
        df_final = pd.merge(df_cartera, df_master, on='ID_CRUCE', how='left')
        
        # Filtro básico: Solo gente que debe y tiene GPS
        df_final = df_final[df_final['Saldo'] > 1]
        df_final = df_final.dropna(subset=['Latitud', 'Longitud'])
        
        # Clasificar Riesgo
        def clasificar(dias):
            if dias <= 30: return "🟢 Fresca"
            elif dias <= 90: return "🟡 Alerta"
            else: return "🔴 JUDICIAL"
        df_final['Prioridad'] = df_final['Dias'].apply(clasificar)

        return df_final

    except Exception as e:
        st.error(f"Error de datos: {e}")
        return pd.DataFrame()

# --- INTERFAZ ---
st.title("🚔 Ruta Táctica de Cobranza")
st.markdown("Generador de Hojas de Ruta Optimizadas por Zona")

df = cargar_datos_fusionados()

if not df.empty:
    # --- FILTROS INTELIGENTES ---
    st.sidebar.header("📍 Definir Misión")
    
    # 1. Filtro Ciudad (Si existe la columna, si no usa Vendedor)
    col_zona = 'Ciudad' if 'Ciudad' in df.columns else 'Vendedor'
    
    zonas_disponibles = sorted(df[col_zona].astype(str).unique())
    seleccion_zona = st.sidebar.selectbox(f"Seleccionar {col_zona}:", zonas_disponibles)
    
    # 2. Filtro Monto Mínimo (Para no ir por $5 dólares)
    monto_min = st.sidebar.number_input("Mínimo a Cobrar por Cliente ($)", value=10, step=10)

    # --- GENERADOR DE RUTA ---
    # Filtramos
    df_ruta = df[
        (df[col_zona] == seleccion_zona) & 
        (df['Saldo'] >= monto_min)
    ].copy()

    if not df_ruta.empty:
        # OPTIMIZAMOS LA RUTA (El cerebro del sistema)
        df_optimizada = ordenar_ruta_cobranza(df_ruta)
        df_optimizada['Secuencia'] = range(1, len(df_optimizada) + 1)
        
        # --- KPIS DE LA MISIÓN ---
        total_recuperar = df_optimizada['Saldo'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Objetivo Total", f"${total_recuperar:,.2f}")
        c2.metric("📍 Paradas", len(df_optimizada))
        c3.metric("🎯 Cliente Top", f"${df_optimizada['Saldo'].max():,.2f}")

        # --- MAPA DE RUTA (CONECTADO) ---
        lat_c = df_optimizada['Latitud'].mean()
        lon_c = df_optimizada['Longitud'].mean()

        fig = go.Figure()

        # 1. Línea de Ruta (Camino a seguir)
        fig.add_trace(go.Scattermapbox(
            lat=df_optimizada['Latitud'], lon=df_optimizada['Longitud'],
            mode='lines',
            line=dict(width=4, color='blue'),
            name='Ruta Sugerida'
        ))

        # 2. Puntos de Cobro (Marcadores)
        # Color rojo si es judicial, verde si es normal
        colors = df_optimizada['Prioridad'].map({"🔴 JUDICIAL": "red", "🟡 Alerta": "orange", "🟢 Fresca": "green"})
        
        fig.add_trace(go.Scattermapbox(
            lat=df_optimizada['Latitud'], lon=df_optimizada['Longitud'],
            mode='markers+text',
            marker=dict(size=15, color=colors, opacity=0.9),
            text=df_optimizada['Secuencia'].astype(str), # Número 1, 2, 3...
            textposition='top center',
            textfont=dict(color='black', size=12, family="Arial Black"),
            hovertext=df_optimizada['Razon Social'] + " ($" + df_optimizada['Saldo'].astype(str) + ")",
            name='Deudores'
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            title=f"Ruta de Recuperación: {seleccion_zona}",
            mapbox=dict(center=dict(lat=lat_c, lon=lon_c), zoom=12),
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- HOJA DE RUTA DESCARGABLE ---
        st.subheader("📋 Hoja de Ruta para el Cobrador")
        
        # Tabla limpia para imprimir/ver
        cols_mostrar = ['Secuencia', 'Razon Social', 'Saldo', 'Dias', 'Prioridad', 'Direccion', 'Telefono']
        # Validar que las columnas existan antes de mostrarlas
        cols_finales = [c for c in cols_mostrar if c in df_optimizada.columns]
        
        st.dataframe(df_optimizada[cols_finales], use_container_width=True)

    else:
        st.warning(f"No hay clientes en {seleccion_zona} con deuda mayor a ${monto_min}.")
else:
    st.info("Cargando base de datos...")
