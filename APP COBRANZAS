import streamlit as st
import pandas as pd
import gspread
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Cobranzas 360 - PDV", layout="wide", page_icon="💸")

# --- CONEXIÓN A GOOGLE SHEETS (SECRETS) ---
@st.cache_data(ttl=600)
def cargar_datos_cobranzas():
    try:
        # 1. Conexión Nube
        credenciales = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(credenciales)
        sh = gc.open("soluto")

        # 2. Cargar CARTERA y MASTER
        # Usamos get_all_records con expected_headers para evitar errores de filas vacías
        df_cartera = pd.DataFrame(sh.worksheet("CARTERA").get_all_records())
        df_master = pd.DataFrame(sh.worksheet("CLIENTES_MASTER").get_all_records())

        # 3. Limpieza de Nombres de Columnas
        df_cartera.columns = [col.strip() for col in df_cartera.columns]
        df_master.columns = [col.strip() for col in df_master.columns]

        # 4. Estandarizar IDs para el Cruce (Convertir a texto)
        # Ajusta "Identificacion" o "CEDULA O RUC" según tus encabezados reales
        col_id_cartera = 'Identificacion' if 'Identificacion' in df_cartera.columns else 'RUC'
        col_id_master = 'CEDULA O RUC' 
        
        df_cartera['ID_CRUCE'] = df_cartera[col_id_cartera].astype(str).str.strip()
        df_master['ID_CRUCE'] = df_master[col_id_master].astype(str).str.strip()

        # 5. Procesar GPS del Master
        if 'Ubicacion GPS' in df_master.columns:
            lat_lon = df_master['Ubicacion GPS'].astype(str).str.split(',', expand=True)
            if len(lat_lon.columns) >= 2:
                df_master['Latitud'] = pd.to_numeric(lat_lon[0], errors='coerce')
                df_master['Longitud'] = pd.to_numeric(lat_lon[1], errors='coerce')

        # 6. Procesar Números de Cartera (Saldo y Días)
        # Limpiamos símbolos de moneda ($) o comas si existen
        df_cartera['Saldo'] = df_cartera['Saldo'].astype(str).str.replace('$', '').str.replace(',', '')
        df_cartera['Saldo'] = pd.to_numeric(df_cartera['Saldo'], errors='coerce').fillna(0)
        
        df_cartera['Dias'] = pd.to_numeric(df_cartera['Dias'], errors='coerce').fillna(0)

        # 7. CRUCE: Unir Deuda + Mapa
        # Left Join: Mantenemos toda la cartera, si no tiene mapa, sale sin mapa.
        df_final = pd.merge(df_cartera, df_master[['ID_CRUCE', 'Latitud', 'Longitud']], 
                            on='ID_CRUCE', how='left')

        # 8. Eliminar deuda cero (Clientes que ya pagaron)
        df_final = df_final[df_final['Saldo'] > 1] # Filtramos saldos menores a $1

        # 9. Categorización de Riesgo (Semáforo)
        def clasificar_riesgo(dias):
            if dias <= 30: return "🟢 Corriente (0-30)"
            elif dias <= 90: return "🟡 Vencida (31-90)"
            else: return "🔴 JUDICIAL (+90)"
        
        df_final['Estado'] = df_final['Dias'].apply(clasificar_riesgo)

        return df_final

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# --- INTERFAZ ---
st.title("💸 Radar de Cobranzas - Recuperación Inteligente")
st.markdown("**Objetivo:** Priorizar rutas por monto y antigüedad de la deuda.")

df = cargar_datos_cobranzas()

if not df.empty:
    # --- BARRA LATERAL ---
    st.sidebar.header("🔍 Filtros de Cobro")
    
    # 1. Filtro por Vendedor/Ruta (Conexión AppSheet)
    rutas = ["TODAS"] + sorted(df['Vendedor'].unique().tolist())
    
    # Detectar URL de AppSheet (?ruta=JUAN PEREZ)
    params = st.query_params
    ruta_url = params.get("ruta", None)
    idx_ruta = 0
    if ruta_url and ruta_url in rutas:
        idx_ruta = rutas.index(ruta_url)
        st.success(f"📱 Ruta filtrada desde App: {ruta_url}")

    seleccion_ruta = st.sidebar.selectbox("Vendedor / Ruta:", rutas, index=idx_ruta)

    # 2. Filtro por Estado de Deuda
    tipos_deuda = ["TODAS", "🔴 JUDICIAL (+90)", "🟡 Vencida (31-90)", "🟢 Corriente (0-30)"]
    seleccion_estado = st.sidebar.selectbox("Nivel de Riesgo:", tipos_deuda, index=1) # Por defecto Vencida

    # --- APLICAR FILTROS ---
    df_view = df.copy()
    if seleccion_ruta != "TODAS":
        df_view = df_view[df_view['Vendedor'] == seleccion_ruta]
    
    if seleccion_estado != "TODAS":
        df_view = df_view[df_view['Estado'] == seleccion_estado]

    # --- KPIs FINANCIEROS ---
    total_deuda = df_view['Saldo'].sum()
    clientes_deuda = len(df_view)
    max_deuda = df_view['Saldo'].max() if not df_view.empty else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cartera Total en Ruta", f"${total_deuda:,.2f}")
    c2.metric("Clientes con Deuda", clientes_deuda)
    c3.metric("Deuda Más Alta (Visitar YA)", f"${max_deuda:,.2f}")
    
    # Porcentaje de clientes ubicados
    ubicados = df_view.dropna(subset=['Latitud']).shape[0]
    c4.metric("Georeferenciados", f"{ubicados}/{clientes_deuda}", 
              delta="Faltan coordenadas" if ubicados < clientes_deuda else "Completo")

    # --- MAPA DE CALOR DE DEUDA ---
    st.subheader(f"🗺️ Mapa de Recuperación: {seleccion_ruta}")
    
    # Separar con GPS y sin GPS
    df_mapa = df_view.dropna(subset=['Latitud', 'Longitud'])
    
    if not df_mapa.empty:
        # Calcular centro
        lat_c = df_mapa['Latitud'].mean()
        lon_c = df_mapa['Longitud'].mean()

        fig = go.Figure()

        # Puntos de Deuda
        # El tamaño del punto depende del Saldo (logarítmico para que no tape el mapa)
        sizes = np.log(df_mapa['Saldo'] + 1) * 3 
        
        # Colores según estado
        colors = df_mapa['Estado'].map({
            "🟢 Corriente (0-30)": "green",
            "🟡 Vencida (31-90)": "gold",
            "🔴 JUDICIAL (+90)": "red"
        })

        fig.add_trace(go.Scattermapbox(
            lat=df_mapa['Latitud'], lon=df_mapa['Longitud'],
            mode='markers',
            marker=dict(
                size=sizes, 
                color=colors,
                opacity=0.7
            ),
            text=df_mapa['Razon Social'] + "<br>Deuda: $" + df_mapa['Saldo'].astype(str) + "<br>Días: " + df_mapa['Dias'].astype(str),
            hoverinfo='text',
            name="Deudores"
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            height=600,
            mapbox=dict(center=dict(lat=lat_c, lon=lon_c), zoom=11),
            margin={"r":0,"t":0,"l":0,"b":0},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No hay clientes con coordenadas GPS en esta selección. Revisa el cruce con CLIENTES_MASTER.")

    # --- TABLA DE GESTIÓN ---
    with st.expander("📂 Ver Listado de Cobro (Ordenado por Prioridad)"):
        # Ordenar: Primero los que deben más dinero
        df_table = df_view[['Razon Social', 'Saldo', 'Dias', 'Estado', 'Direccion', 'Telefono']].sort_values(by='Saldo', ascending=False)
        st.dataframe(df_table, use_container_width=True)

else:
    st.info("Conectando con base de datos de Cartera...")
