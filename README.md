<div align="center">
  <img src="https://raw.githubusercontent.com/yasminum/SistemaTutorInteligente/main/figures/uni_logo.png" alt="UNI Logo" width="120" style="margin-bottom: 20px;"/>
  
  # 🧠 Sistema de Tutoría Inteligente Adaptativa (DRL)
  
  **Tesis de Maestría en Inteligencia Artificial - Universidad Nacional de Ingeniería (UNI)**
  
  *Integración de Multi-Skill Knowledge Tracing (MSKT), Neural Cognitive Diagnosis (NeuralCD) y Model-Based Policy Optimization (Dyna-PPO) para la optimización de la comprensión lectora basada en literatura peruana.*

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
  [![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-green.svg?style=for-the-badge)](https://gymnasium.farama.org/)
  [![Status](https://img.shields.io/badge/Estado-Investigación_Finalizada-success.svg?style=for-the-badge)]()
</div>

<br>

## 📋 Resumen Ejecutivo
Este proyecto presenta la implementación computacional (*in-silico*) de un **Sistema de Tutoría Inteligente (ITS)** revolucionario. Emplea un enfoque de **Deep Reinforcement Learning (DRL)** con una arquitectura *Model-Based* para generar políticas de enseñanza personalizadas. Superando drásticamente a las heurísticas tradicionales y a los agentes de RL puros (*Model-Free*), el sistema modela de forma autónoma la memoria, la fatiga y el progreso cognitivo de los estudiantes frente a textos de literatura peruana.

---

## 🏗️ Arquitectura del Sistema (Pipeline Avanzado)

La arquitectura ha sido iterada adversarialmente para robustecer la extracción de características y la planificación bajo incertidumbre. Se estructura en cuatro capas de procesamiento tensorial acopladas en un bucle cerrado (Closed-Loop CMDP), superando las limitaciones de los ITS clásicos:

<img width="1754" height="914" alt="image" src="https://github.com/user-attachments/assets/1a056437-b623-447d-bc53-9e67a35853b1" />

```mermaid
flowchart TD
    %% Estilos de alto contraste y nivel académico
    classDef envLayer fill:#f8f9fa,stroke:#ced4da,stroke-width:2px,color:#212529;
    classDef cogLayer fill:#e3f2fd,stroke:#90caf9,stroke-width:2px,color:#0d47a1;
    classDef mbLayer fill:#fff3e0,stroke:#ffb74d,stroke-width:2px,color:#e65100;
    classDef rlLayer fill:#e8f5e9,stroke:#81c784,stroke-width:2px,color:#1b5e20;
    classDef shield fill:#ffebee,stroke:#ef5350,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph L1 ["Capas 1: Entorno Simulado (Generación de Datos)"]
        direction LR
        E1["Simulador IRT\n(θ, Fatiga, Racha)"]:::envLayer
        E2["Banco de Textos\nLiteratura Peruana\n(Parámetros a,b,c)"]:::envLayer
        E1 <-->|Interacción (a, t)| E2
    end

    subgraph L2 ["Capa 2: Diagnóstico y Rastreabilidad Cognitiva"]
        direction LR
        C1["MSKT + SAINT+\n(LSTM, Olvido Temporal)"]:::cogLayer
        C2["NeuralCD\n(Proyección de Maestría KC)"]:::cogLayer
        C3["Feature Fusing\nExtracción Latente x_t"]:::cogLayer
        C1 --> C2 --> C3
    end

    subgraph L3 ["Capa 3: Dinámicas del Mundo (Model-Based)"]
        direction TB
        M1["World Model Neural Network\nf(s,a) → s', r"]:::mbLayer
        M2["Branching Rollouts\n(Simulación de H pasos)"]:::mbLayer
        M1 -->|Genera| M2
        M1 -.-|Loss: MSE(s')| M1
    end

    subgraph L4 ["Capa 4: Optimización de Política (Dyna-PPO)"]
        direction TB
        P1["Actor Network (π)\nMultidiscreta (Texto, Dif, Scaffold)"]:::rlLayer
        P2["Critic Network (V)\nValue Estimation"]:::rlLayer
        P3["PPO Clipping & Advantage\nGAE-λ"]:::rlLayer
        P1 & P2 --> P3
    end
    
    %% Filtro de Seguridad
    S1{{"CMDP Shield\n(Bloqueo de Frustración)"}}:::shield

    %% Conexiones Principales
    L1 == "Vector de Respuesta (R, t)" ==> L2
    C3 == "Estado Continuo (s_t)" ==> M1
    C3 == "Estado Continuo (s_t)" ==> P1
    C3 == "Estado Continuo (s_t)" ==> P2
    
    M2 == "Batch de Datos Sintéticos" ==> P3
    L1 == "Batch de Datos Reales" ==> P3
    
    P1 == "Acción Propuesta (a_t)" ==> S1
    S1 == "Acción Segura Ejecutada" ==> L1
