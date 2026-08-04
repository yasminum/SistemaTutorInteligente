[README.md](https://github.com/user-attachments/files/30689916/README.md)
# 🧠 Sistema de Tutoría Inteligente Adaptativa (DRL)

**Tesis de Maestría en Inteligencia Artificial - UNI**
Integración de MSKT, NeuralCD y Dyna-PPO para la optimización de la comprensión lectora basada en literatura peruana.

## 🚀 Arquitectura del Sistema
Este repositorio contiene la implementación computacional (`in-silico`) de un Sistema de Tutoría Inteligente (ITS) basado en Deep Reinforcement Learning. El sistema utiliza una arquitectura novedosa que combina:
- **MSKT + SAINT+**: Para el modelado del estudiante y olvido temporal.
- **NeuralCD**: Para el diagnóstico cognitivo multidimensional.
- **Dyna-PPO (MBPO)**: Para la planificación model-based con rollouts ramificados.

## 📊 Resultados Experimentales
La validación demuestra que la planificación *model-based* supera significativamente a los enfoques *model-free*:

| Condición | Modelo | LG_sim (Ganancia) |
|-----------|--------|-------------------|
| B4 | PPO (Model-Free) | 0.239 |
| **B5** | **Dyna-PPO (Model-Based)** | **0.504** ✅ |

**Validación Estadística:**
- Mejora relativa de B5 sobre B4: **94.07%** (Hipótesis HE3 aceptada)
- Tamaño del efecto (d de Cohen): **8.908** (Significancia gigantesca)
- Valor-p: `< 0.00001`

Para el reporte completo con gráficas, ejecuta `python generate_report.py` o lee el archivo `EXPERIMENT_REPORT.md`.

## 📁 Estructura del Repositorio
- `src/`: Código fuente modular (Entorno, Modelos, Agentes).
- `generate_report.py`: Script para regenerar las figuras y estadísticas de la tesis.
- `data/results/`: Checkpoints de los modelos entrenados y logs JSON.

## ⚙️ Reproducibilidad
```bash
pip install -r requirements.txt
python generate_report.py
```
