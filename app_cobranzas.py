import streamlit as st
import pandas as pd
import gspread
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from math import radians, cos, sin, asin, sqrt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Ruta Táctica de Cobranza (GPS)", layout="wide", page_icon="🕵️‍♂️")

# Coordenada Base (Ambato)
BODEGA_AMBATO = np.array([[-1.241667, -78.619722]])

# --- MOTORES MATEMÁTICOS ---
def ordenar_ruta_cobranza(df_grupo):
    """Ordena los deudores para hacer una ruta lógica por cercanía"""
    if len(df_grupo) < 1: return df_grupo
    
    coords = df_grupo[['Latitud', 'Longitud']].values
    pendientes = list(range(len(coords)))
    orden_indices = []
    
    # Empezar por el más cercano a Ambato (o al centro del grupo)
    distancias_bodega = cdist(BODEGA_AMBATO, coords, metric='euclidean')
    idx_primero = np.argmin(distancias_bodega)
    orden_indices.append(idx_primero)
    pendientes.remove(idx_primero)
    
    # Encadenar vecinos más cercanos
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

# --- CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_fusionados():
    try:
        credenciales = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(credenciales)
        sh = gc.open("soluto")

        df_cartera = pd.DataFrame(sh.worksheet("CARTERA").get_all_records())
        df_master = pd.DataFrame(sh.worksheet("CLIENTES_MASTER").get_all_records())

        # Limpieza
        df_cartera.columns = [col.strip() for col in df_cartera.columns]
        df_master.columns = [col.strip() for col in df_master.columns]

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

        # Dinero
        df_cartera['Saldo'] = df_cartera['Saldo'].astype(str).str.replace('$', '').str.replace(',', '')
        df_cartera['Saldo'] = pd.to_numeric(df_cartera['Saldo'], errors='coerce').fillna(0)
        df_cartera['Dias'] = pd.to_numeric(df_cartera['Dias'], errors='coerce').fillna(0)

        # Cruce
        df_final = pd.merge(df_cartera, df_master, on='ID_CRUCE', how='left')
        
        # Filtros estrictos: Debe tener deuda y DEBE TENER GPS
        df_final = df_final[df_final['Saldo'] > 1]
        df_final = df_final.dropna(subset=['Latitud', 'Longitud'])
        df_final = df_final[(df_final['Latitud'] != 0) & (df_final['Longitud'] != 0)]
        
        # Clasificar Riesgo
        def clasificar(dias):
            if dias <= 30: return "🟢 Fresca"
            elif dias <= 90: return "🟡 Alerta"
            else: return "🔴 JUDICIAL"
        df_final['Prioridad'] = df_final['Dias'].apply(clasificar)

        return df_final
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# --- INTERFAZ ---
st.title("🚔 Ruta Táctica de Cobranza (GPS Puro)")
st.markdown("Zonificación automática basada en coordenadas reales.")

df = cargar_datos_fusionados()

if not df.empty:
    st.sidebar.header("📍 Configuración de Misión")
    
    # 1. ZONIFICACIÓN IA (Reemplaza al filtro de Ciudad)
    # El usuario decide en cuántas zonas dividir la cartera
    num_zonas = st.sidebar.slider("¿En cuántas zonas dividir la cobranza?", 2, 20, 5)
    monto_min = st.sidebar.number_input("Monto Mínimo a gestionar ($)", value=10)

    # Procesamiento Clustering (IA)
    kmeans = KMeans(n_clusters=num_zonas, random_state=42, n_init=10)
    df['Zona_IA'] = kmeans.fit_predict(df[['Latitud', 'Longitud']])
    
    # Convertir zona numérica a texto amigable "Zona 1", "Zona 2"
    df['Nombre_Zona'] = "Zona Geográfica " + (df['Zona_IA'] + 1).astype(str)
    
    # 2. SELECCIÓN DE ZONA
    zonas_disponibles = sorted(df['Nombre_Zona'].unique())
    seleccion_zona = st.sidebar.selectbox("Seleccionar Zona de Operación:", zonas_disponibles)
    
    # Filtrar datos
    df_ruta = df[
        (df['Nombre_Zona'] == seleccion_zona) & 
        (df['Saldo'] >= monto_min)
    ].copy()

    if not df_ruta.empty:
        # OPTIMIZAR RUTA
        df_optimizada = ordenar_ruta_cobranza(df_ruta)
        df_optimizada['Secuencia'] = range(1, len(df_optimizada) + 1)
        
        # KPIS
        total_recuperar = df_optimizada['Saldo'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Cartera en Zona", f"${total_recuperar:,.2f}")
        c2.metric("📍 Puntos a Visitar", len(df_optimizada))
        c3.metric("🚨 Deuda Máxima", f"${df_optimizada['Saldo'].max():,.2f}")

        # MAPA
        lat_c = df_optimizada['Latitud'].mean()
        lon_c = df_optimizada['Longitud'].mean()

        fig = go.Figure()

        # Línea de Ruta
        fig.add_trace(go.Scattermapbox(
            lat=df_optimizada['Latitud'], lon=df_optimizada['Longitud'],
            mode='lines',
            line=dict(width=4, color='blue'),
            name='Ruta'
        ))

        # Puntos
        colors = df_optimizada['Prioridad'].map({"🔴 JUDICIAL": "red", "🟡 Alerta": "orange", "🟢 Fresca": "green"})
        
        fig.add_trace(go.Scattermapbox(
            lat=df_optimizada['Latitud'], lon=df_optimizada['Longitud'],
            mode='markers+text',
            marker=dict(size=14, color=colors, opacity=0.9),
            text=df_optimizada['Secuencia'].astype(str),
            textposition='top center',
            textfont=dict(color='black', size=11, family="Arial Black"),
            hovertext=df_optimizada['Razon Social'] + " | $" + df_optimizada['Saldo'].astype(str),
            name='Clientes'
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            title=f"Mapa Táctico: {seleccion_zona}",
            mapbox=dict(center=dict(lat=lat_c, lon=lon_c), zoom=12),
            margin={"r":0,"t":30,"l":0,"b":0},
            legend=dict(x=0,y=1, bgcolor='rgba(255,255,255,0.7)')
        )
        st.plotly_chart(fig, use_container_width=True)

        # TABLA
        st.subheader("📋 Lista de Trabajo (Secuencia Lógica)")
        cols_mostrar = ['Secuencia', 'Razon Social', 'Saldo', 'Dias', 'Prioridad', 'Direccion']
        cols_finales = [c for c in cols_mostrar if c in df_optimizada.columns]
        st.dataframe(df_optimizada[cols_finales], use_container_width=True)

    else:
        st.warning("No hay clientes en esta zona con ese monto de deuda.")
else:
    st.info("Conectando y detectando coordenadas...")
