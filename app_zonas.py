import streamlit as st
import pandas as pd
import gspread
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from math import radians, cos, sin, asin, sqrt

# --- 1. CONFIGURACIÓN E IMAGEN CORPORATIVA ---
st.set_page_config(page_title="SaaS Logística Nacional - PDV", layout="wide", page_icon="🇪🇨")

# Coordenada Fija: BODEGA CENTRAL AMBATO
# Usamos numpy array para cálculos rápidos de distancia
BODEGA_AMBATO = np.array([[-1.241667, -78.619722]])

# --- 2. MOTORES DE CÁLCULO MATEMÁTICO ---

def haversine(lon1, lat1, lon2, lat2):
    """Calcula distancia real en KM usando la curvatura de la Tierra"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * 6371 # Radio Tierra KM

def calcular_distancia_total(df_ruta):
    """Calcula el recorrido: Ambato -> Cliente 1 -> ... -> Cliente N -> Ambato"""
    distancia = 0
    if len(df_ruta) < 1: return 0
    
    # Tramo 1: Salida de Bodega al primer cliente
    p1 = df_ruta.iloc[0]
    distancia += haversine(BODEGA_AMBATO[0][1], BODEGA_AMBATO[0][0], p1['Longitud'], p1['Latitud'])
    
    # Tramos intermedios (Cliente a Cliente)
    for i in range(len(df_ruta) - 1):
        p_act = df_ruta.iloc[i]
        p_sig = df_ruta.iloc[i+1]
        distancia += haversine(p_act['Longitud'], p_act['Latitud'], p_sig['Longitud'], p_sig['Latitud'])
    
    # Retorno a Bodega (Vuelta a casa)
    p_ult = df_ruta.iloc[-1]
    distancia += haversine(p_ult['Longitud'], p_ult['Latitud'], BODEGA_AMBATO[0][1], BODEGA_AMBATO[0][0])
    
    return round(distancia, 2)

def ordenar_ruta_desde_bodega(df_grupo):
    """Algoritmo de optimización (Vecino más cercano) empezando desde Ambato"""
    if len(df_grupo) < 1: return df_grupo
    
    coords = df_grupo[['Latitud', 'Longitud']].values
    pendientes = list(range(len(coords)))
    orden_indices = []
    
    # PASO 1: Encontrar el primer cliente más cercano a la Bodega
    distancias_bodega = cdist(BODEGA_AMBATO, coords, metric='euclidean')
    idx_primero = np.argmin(distancias_bodega)
    orden_indices.append(idx_primero)
    pendientes.remove(idx_primero)
    
    # PASO 2: Encadenar el resto
    while pendientes:
        ultimo_idx = orden_indices[-1]
        ultimo_punto = coords[ultimo_idx].reshape(1, -1)
        # Comparar solo contra los que faltan
        coords_pendientes = coords[pendientes]
        distancias = cdist(ultimo_punto, coords_pendientes, metric='euclidean')
        
        idx_local = np.argmin(distancias)
        idx_real = pendientes[idx_local]
        
        orden_indices.append(idx_real)
        pendientes.remove(idx_real)
        
    return df_grupo.iloc[orden_indices].reset_index(drop=True)

# --- 3. CONEXIÓN Y CARGA DE DATOS ---
@st.cache_data(ttl=600)
def cargar_datos_master():
    try:
        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE ---
        # Conexión usando Secrets (Nube)
        credenciales = dict(st.secrets["gcp_service_account"])
        gc = gspread.service_account_from_dict(credenciales)
        
        sh = gc.open("soluto")
        
        # Leer CLIENTES_MASTER
        df = pd.DataFrame(sh.worksheet("CLIENTES_MASTER").get_all_records())
        
        # Limpieza de nombres de columnas (quita espacios extra)
        df.columns = [col.strip() for col in df.columns]
        
        # Procesamiento de GPS (Separar string "-1.23, -78.56")
        if 'Ubicacion GPS' in df.columns:
            lat_lon = df['Ubicacion GPS'].astype(str).str.split(',', expand=True)
            if len(lat_lon.columns) >= 2:
                df['Latitud'] = pd.to_numeric(lat_lon[0], errors='coerce')
                df['Longitud'] = pd.to_numeric(lat_lon[1], errors='coerce')
                
                # Eliminar filas sin coordenadas válidas (0,0 o vacías)
                df = df.dropna(subset=['Latitud', 'Longitud'])
                df = df[(df['Latitud'] != 0) & (df['Longitud'] != 0)]
                
                # Crear columna 'Total' simulada si no existe (para el Heatmap)
                if 'Total' not in df.columns: df['Total'] = 1 
                
                return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error de conexión o formato: {e}")
        return pd.DataFrame()

# --- 4. INTERFAZ PRINCIPAL (DASHBOARD) ---
st.title("🇪🇨 PDV Sin Límites - Centro de Comando Nacional")
st.markdown(f"**Base Operativa:** Ambato | **Alcance:** Nacional (Optimización IA)")

# Cargar datos
df = cargar_datos_master()

if not df.empty:
    # --- BARRA LATERAL (CONTROLES) ---
    st.sidebar.header("🎛️ Configuración de Flota")
    
    # Slider para simular hasta 130 rutas (25 vendedores * 5 días)
    num_rutas = st.sidebar.slider("Total Rutas a Generar", 5, 130, 25)
    
    st.sidebar.markdown("---")
    st.sidebar.header("👁️ Capas Visuales")
    ver_etiquetas = st.sidebar.checkbox("Ver Nombres (Etiquetas)", value=True)
    ver_bodega = st.sidebar.checkbox("Líneas a Bodega (Ambato)", value=True, help="Muestra la conexión desde Ambato hasta el inicio de cada ruta")
    modo_calor = st.sidebar.checkbox("🔥 Activar Mapa de Calor", value=False)

    # --- PROCESAMIENTO IA (CLUSTERING + TSP) ---
    with st.spinner(f'Calculando logística óptima para {len(df)} clientes...'):
        # 1. Clustering (Agrupación por cercanía)
        kmeans = KMeans(n_clusters=num_rutas, random_state=42, n_init=10)
        df['Zona_ID'] = kmeans.fit_predict(df[['Latitud', 'Longitud']])
        
        # 2. Secuenciación y Detección de Outliers
        df_final = pd.DataFrame()
        rutas_activas = sorted(df['Zona_ID'].unique())
        
        for zona in rutas_activas:
            sub = df[df['Zona_ID'] == zona].copy()
            # Ordenamos la ruta
            sub_ord = ordenar_ruta_desde_bodega(sub)
            
            # Asignamos metadatos
            sub_ord['Orden'] = range(1, len(sub_ord) + 1)
            sub_ord['Nombre_Ruta'] = f"Ruta {zona + 1}"
            
            # Lógica de "Outliers" (Clientes aislados)
            coords_ruta = sub_ord[['Latitud', 'Longitud']].values
            if len(coords_ruta) > 2:
                dists = cdist(coords_ruta, coords_ruta, metric='euclidean')
                np.fill_diagonal(dists, np.inf) # Ignorar distancia a sí mismo
                min_dists = dists.min(axis=1) # Vecino más cercano
                umbral = min_dists.mean() * 2.5 # Si está 2.5 veces más lejos del promedio, es sospechoso
                sub_ord['Es_Outlier'] = min_dists > umbral
            else:
                sub_ord['Es_Outlier'] = False

            df_final = pd.concat([df_final, sub_ord])

    # --- SELECTOR DE VISTA (DRILL-DOWN) ---
    opciones = ["VISTA GLOBAL (TODAS)"] + sorted(df_final['Nombre_Ruta'].unique())
    seleccion = st.selectbox("🔍 Selecciona Zona para Auditar:", opciones)

    # Filtrado de datos para visualización
    if seleccion == "VISTA GLOBAL (TODAS)":
        df_view = df_final
        zoom_auto = 6 # Zoom lejano (todo el país)
        titulo_mapa = "Mapa Nacional de Cobertura"
    else:
        df_view = df_final[df_final['Nombre_Ruta'] == seleccion]
        zoom_auto = 12 # Zoom cercano
        titulo_mapa = f"Detalle Operativo: {seleccion}"

    # --- KPI DASHBOARD (MÉTRICAS) ---
    c1, c2, c3, c4 = st.columns(4)
    
    # Cálculo de distancia
    if seleccion != "VISTA GLOBAL (TODAS)":
        km_est = calcular_distancia_total(df_view)
        txt_km = f"{km_est} km"
    else:
        km_est = 0
        txt_km = "Global"
    
    c1.metric("Clientes Asignados", len(df_view))
    c2.metric("Distancia Total (Ida/Vuelta)", txt_km)
    
    # KPI de Eficiencia (Km por Cliente)
    if isinstance(km_est, (int, float)) and km_est > 0 and len(df_view) > 0:
        km_por_cliente = round(km_est / len(df_view), 2)
        # Rojo si gastas más de 5km de gasolina para visitar a un solo cliente
        delta_color = "normal" if km_por_cliente < 5 else "inverse" 
        c3.metric("Km Promedio por Cliente", f"{km_por_cliente} km", delta_color=delta_color)
    else:
        c3.metric("Eficiencia", "-")

    # KPI Outliers
    outliers_count = df_view['Es_Outlier'].sum()
    c4.metric("⚠️ Clientes Aislados (Costosos)", outliers_count, delta="-Revisar" if outliers_count > 0 else "Ok")

    # --- MAPA INTERACTIVO PRO ---
    fig = go.Figure()

    if modo_calor and seleccion == "VISTA GLOBAL (TODAS)":
        # MODO HEATMAP (Densidad)
        fig.add_trace(go.Densitymapbox(
            lat=df_view['Latitud'], lon=df_view['Longitud'],
            z=df_view['Total'], # Ponderado
            radius=20, opacity=0.7, colorscale="Hot"
        ))
    else:
        # MODO LOGÍSTICO (Rutas y Puntos)
        
        # 1. Dibujar Bodega Central (Ambato)
        fig.add_trace(go.Scattermapbox(
            lat=[BODEGA_AMBATO[0][0]], lon=[BODEGA_AMBATO[0][1]],
            mode='markers+text', 
            marker=dict(size=14, color='black', symbol='star'),
            text=["CENTRAL AMBATO"], textposition="top center",
            name="CENTRAL"
        ))

        if seleccion != "VISTA GLOBAL (TODAS)":
            # --- VISTA DETALLADA ---
            
            # Línea de Conexión (Ambato -> Inicio Ruta)
            # CORRECCIÓN APLICADA AQUÍ: Se eliminó dash='dot'
            if ver_bodega:
                fig.add_trace(go.Scattermapbox(
                    lat=[BODEGA_AMBATO[0][0], df_view.iloc[0]['Latitud']],
                    lon=[BODEGA_AMBATO[0][1], df_view.iloc[0]['Longitud']],
                    mode='lines', 
                    line=dict(width=1, color='grey'), # Línea sólida fina
                    name='Conexión Central'
                ))
            
            # Línea de la Ruta (Cliente -> Cliente)
            fig.add_trace(go.Scattermapbox(
                lat=df_view['Latitud'], lon=df_view['Longitud'],
                mode='lines', 
                line=dict(width=3, color='blue'),
                name='Recorrido'
            ))
            
            # Separar Clientes Normales vs Outliers
            df_norm = df_view[~df_view['Es_Outlier']]
            df_out = df_view[df_view['Es_Outlier']]

            # Clientes Normales (Coloreados por orden de visita)
            fig.add_trace(go.Scattermapbox(
                lat=df_norm['Latitud'], lon=df_norm['Longitud'],
                mode='markers+text' if ver_etiquetas else 'markers',
                marker=dict(size=12, color=df_norm['Orden'], colorscale='Viridis', showscale=True),
                text=df_norm['Orden'].astype(str) + ". " + df_norm['nombre'].str[:15],
                textposition="top right",
                textfont=dict(color='black', size=10),
                name="Clientes"
            ))
            
            # Clientes Outliers (Rojos)
            if not df_out.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=df_out['Latitud'], lon=df_out['Longitud'],
                    mode='markers', 
                    marker=dict(size=15, color='red', opacity=0.8),
                    text="⚠️ DESVÍO: " + df_out['nombre'], 
                    hoverinfo='text',
                    name="⚠️ Desvíos Costosos"
                ))
        else:
            # --- VISTA GLOBAL (Puntos simples) ---
            fig.add_trace(go.Scattermapbox(
                lat=df_view['Latitud'], lon=df_view['Longitud'],
                mode='markers', 
                marker=dict(size=7, color=df_view['Zona_ID'], colorscale='Turbo'),
                text=df_view['nombre'], 
                name="Clientes"
            ))

    # Auto-Centrado Inteligente del Mapa
    if seleccion == "VISTA GLOBAL (TODAS)":
        # Centro aproximado de Ecuador
        center_lat, center_lon = -1.5, -78.5 
        zoom = 6.5
    else:
        # Centrar en la ruta seleccionada
        lats = df_view['Latitud'].tolist()
        lons = df_view['Longitud'].tolist()
        # Añadimos la bodega para que salga en la foto
        lats.append(BODEGA_AMBATO[0][0])
        lons.append(BODEGA_AMBATO[0][1])
        
        center_lat = sum(lats)/len(lats)
        center_lon = sum(lons)/len(lons)
        zoom = 10.5

    fig.update_layout(
        title=titulo_mapa, 
        mapbox_style="open-street-map", 
        height=700,
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        margin={"r":0,"t":40,"l":0,"b":0},
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- GRÁFICO DE BALANCE DE CARGAS ---
    if seleccion == "VISTA GLOBAL (TODAS)":
        st.subheader("⚖️ Balance de Cargas (Clientes por Ruta)")
        conteo = df_final['Nombre_Ruta'].value_counts().reset_index()
        conteo.columns = ['Ruta', 'Clientes']
        
        fig_bar = px.bar(conteo, x='Ruta', y='Clientes', 
                         color='Clientes', title="Distribución de Trabajo",
                         color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- EXPORTAR DATOS ---
    with st.expander("📥 Exportar Planilla de Rutas (Excel/CSV)"):
        st.dataframe(df_view, use_container_width=True)

else:
    st.info("Esperando conexión... Asegúrate de tener 'credenciales.json' y la hoja 'CLIENTES_MASTER' con coordenadas.")
