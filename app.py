import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA (Debe ir lo primero)
# ==========================================
st.set_page_config(
    page_title="DeepDerm | Asistente IA Melanoma", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ARQUITECTURA DEL MODELO (MicroResNetV2 + SE)
# ==========================================
class BloqueAtencionSE(nn.Module):
    def __init__(self, canales, reduccion=16):
        super(BloqueAtencionSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(canales, canales // reduccion, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(canales // reduccion, canales, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BloqueResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BloqueResidual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.atencion_se = BloqueAtencionSE(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.atencion_se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MicroResNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MicroResNetV2, self).__init__()
        self.entrada = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.capa1 = nn.Sequential(BloqueResidual(32, 32, stride=1), BloqueResidual(32, 32, stride=1))
        self.capa2 = nn.Sequential(BloqueResidual(32, 64, stride=2), BloqueResidual(64, 64, stride=1))
        self.capa3 = nn.Sequential(BloqueResidual(64, 128, stride=2), BloqueResidual(128, 128, stride=1))
        self.capa4 = nn.Sequential(BloqueResidual(128, 256, stride=2), BloqueResidual(256, 256, stride=1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.entrada(x)
        x = self.capa1(x)
        x = self.capa2(x)
        x = self.capa3(x)
        x = self.capa4(x)
        x = self.gap(x)
        x = self.clasificador(x)
        return x

# ==========================================
# 3. CARGA DEL MODELO Y CACHÉ
# ==========================================
@st.cache_resource
def cargar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicroResNetV2(num_classes=2)
    # IMPORTANTE: Asegúrate de que tu archivo se llama exactamente así
    model.load_state_dict(torch.load('mejor_modelo_melanomaSE.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

modelo, device = cargar_modelo()

transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# 4. DISEÑO DE LA INTERFAZ (UI/UX)
# ==========================================

# --- PANEL LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063134.png", width=80) # Icono médico
    st.title("Panel Clínico")
    st.markdown("Ajustes del motor de Inferencia IA.")
    
    st.markdown("---")
    st.subheader("⚙️ Umbral de Sensibilidad")
    umbral = st.slider(
        "Límite de Detección (Maligno)", 
        min_value=0, max_value=100, value=50, step=1,
        help="Baja este umbral para volver a la IA más paranoica y evitar falsos negativos."
    ) / 100.0
    
    st.markdown("---")
    st.info("**Modelo:** MicroResNetV2\n\n**Atención:** SE Block\n\n**XAI:** Grad-CAM++")
    
    st.markdown("---")
    st.caption("⚠️ **Aviso Legal:** Esta aplicación es un prototipo académico de Apoyo a la Decisión Clínica (CDSS). No sustituye el diagnóstico de un dermatólogo profesional.")

# --- CUERPO PRINCIPAL ---
st.title("🔬 DeepDerm: Análisis Dermatológico Asistido por IA")
st.markdown("Sube una imagen dermatoscópica o clínica de la lesión para obtener una evaluación probabilística y un mapa de calor de las zonas de riesgo.")

archivo_subido = st.file_uploader("📂 Selecciona la fotografía de la lesión (JPG, PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Mostrar spinner de carga para dar sensación de procesamiento complejo
    with st.spinner('Analizando topología y morfología celular...'):
        
        imagen_pil = Image.open(archivo_subido).convert('RGB')
        tensor_img = transformaciones(imagen_pil).unsqueeze(0).to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = modelo(tensor_img)
            probabilidades = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        prob_benigno = probabilidades[0]
        prob_maligno = probabilidades[1]
        es_maligno = prob_maligno >= umbral

        # Grad-CAM++
        target_layers = [modelo.capa4[-1]]
        cam = GradCAMPlusPlus(model=modelo, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)] 
        grayscale_cam = cam(input_tensor=tensor_img, targets=targets)[0, :]
        
        img_redimensionada = np.array(imagen_pil.resize((224, 224))) / 255.0
        visualizacion_cam = show_cam_on_image(img_redimensionada, grayscale_cam, use_rgb=True)

    # --- RESULTADOS VISUALES ---
    st.markdown("---")
    
    # Cartel gigante con el diagnóstico
    if es_maligno:
        st.error("🚨 **ALERTA CLÍNICA: Patrón Morfológico Compatible con Melanoma Detectado.** Se recomienda biopsia o derivación urgente.")
    else:
        st.success("✅ **ANÁLISIS FAVORABLE: Lesión aparentemente Benigna.** Mantener observación rutinaria.")

    st.markdown("<br>", unsafe_allow_html=True) # Espacio

    # Mostrar fotos e indicadores en columnas
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<h4 style='text-align: center;'>Visión Original</h4>", unsafe_allow_html=True)
        st.image(imagen_pil, use_container_width=True)

    with col2:
        st.markdown("<h4 style='text-align: center;'>Análisis Grad-CAM++</h4>", unsafe_allow_html=True)
        st.image(visualizacion_cam, use_container_width=True, caption="Zonas rojas indican alta atención de la red.")

    with col3:
        st.markdown("<h4 style='text-align: center;'>Métricas de Confianza</h4>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Barras de progreso visuales
        st.write(f"🔴 Riesgo de Malignidad: **{prob_maligno * 100:.1f}%**")
        st.progress(float(prob_maligno))
        
        st.write(f"🟢 Probabilidad Benigna: **{prob_benigno * 100:.1f}%**")
        st.progress(float(prob_benigno))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="Umbral de Alerta Actual", value=f"{umbral*100}%")