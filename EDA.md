# 📊 EDA: Análisis Exploratorio de Datos

**Sistema de Tutoría Inteligente Adaptativa (DRL) · Grupo 2 · Maestría en IA · UNI**  
*Curso: Proyecto de Investigación 2 (Prof. Glen Rodríguez)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasminum/SistemaTutorInteligente/blob/eda/notebooks/01_EDA.ipynb)

---

## 📌 Resumen de Entregables del EDA

| Entregable | Ruta en el Repositorio | Descripción |
|---|---|---|
| **Notebook EDA Ejecutado** | [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) | 29 celdas completas ejecutadas con tablas y gráficos |
| **Logs Maestros de Entrenamiento** | [`data/results/ablation_summary_colab.json`](data/results/ablation_summary_colab.json) | 6.4 MB (6 condiciones × 10 semillas × 5,000 episodios) |
| **Muestra de Datos (Ítems IRT)** | [`data/sample/item_bank_seed0.csv`](data/sample/item_bank_seed0.csv) | Parámetros psicométricos $a, b, c$ de 300 ítems |
| **Muestra de Estudiantes** | [`data/sample/estudiantes_simulados.csv`](data/sample/estudiantes_simulados.csv) | Vectores $\theta$ pre-test y cotas teóricas por semilla |
| **Muestra de Logs Episódicos** | [`data/sample/logs_muestra_ppo.csv`](data/sample/logs_muestra_ppo.csv) | Transiciones de ganancia de aprendizaje (LG) por episodio |
| **Resumen por Semilla** | [`data/sample/resumen_por_semilla.csv`](data/sample/resumen_por_semilla.csv) | Métricas agregadas finales y de convergencia |
| **Visualizaciones Vectoriales** | `figures/eda/` | 8 figuras generadas por el pipeline exploratorio |

---

## 1. Fuentes de Datos y Flujo de Calidad

```mermaid
flowchart LR
    classDef syn fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef out fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100;
    classDef eda fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;

    A["Banco de Ítems IRT-3PL<br/>30 textos × 10 ítems<br/>a ~ U(0.8, 2.0), b ~ N(0, 1)"]:::syn
    B["Estudiantes Simulados<br/>theta ~ Beta(2,2)<br/>3 habilidades lectoras"]:::syn
    C["Simulador In-Silico<br/>ReadingTutorEnv + train.py<br/>Horizonte H = 50 pasos"]:::out
    D["ablation_summary_colab.json<br/>300,000 transiciones episódicas"]:::out
    E["01_EDA.ipynb<br/>Auditoría, Distribuciones<br/>y Feature Engineering"]:::eda

    A --> C
    B --> C
    C --> D
    A --> E
    B --> E
    D --> E
