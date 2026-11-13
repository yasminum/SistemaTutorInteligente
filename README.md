# Sistema de Tutoría Inteligente Adaptativa (DRL) — Avance Parcial (Semana 9)

**Grupo 2 · Maestría en Inteligencia Artificial · UNI (FIIS)**  
*Proyecto de Investigación II — Docente: Dr. Glen Dario Rodríguez Rafael · Asesora: Mg. Ing. María Fernanda Tejada Begazo*  
*Rama de entrega: `feat/hyperparameter-optimization` (consolidación y depuración de evidencia real hasta Semana 9)*

[![Open In Colab (EDA)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasminum/SistemaTutorInteligente/blob/feat/hyperparameter-optimization/notebooks/01_EDA.ipynb)
![Pruebas de fuga](https://img.shields.io/badge/pruebas_de_fuga-5%2F5_PASS-00B3A4)
![Régimen B](https://img.shields.io/badge/D2__clean5__zpd2_LG-%2B0.0203_%C2%B1_0.0167-FF1F7A)
![Oráculo](https://img.shields.io/badge/vs_Or%C3%A1culo_ZPD-49.4%25_(%2B0.0411)-7B2FF7)

---

## Nota de transparencia metodológica (Saneamiento Semana 9)

La arquitectura objetivo de la tesis combina **MSKT** (seguimiento de conocimiento), **NeuralCD** (diagnóstico cognitivo) y **Dyna-PPO** (planificación basada en modelo con búsqueda de hiperparámetros). **Al corte de la Semana 9 estos tres componentes están planificados para las Semanas 10–16 y no cuentan con corridas ejecutadas.**  
Las cifras sintéticas de ablación `B0–B5` y `HPO` (`0.50–0.53`) reportadas en un avance previo fueron retiradas junto con sus archivos, porque no provenían del código versionado. **Todo el código, logs, tablas y figuras de esta rama reflejan exclusivamente los resultados reales ejecutados y verificables del prototipo en Régimen B.**

---

## 1. Resumen cuantitativo verificado (`data/results/summary.json`, `runs_oracle.json`, `sprint2_diagnostics.json`)

Protocolo: `100` estudiantes sintéticos, sesiones de `50` pasos con olvido exponencial en competencias no ejercitadas, `GroupKFold(n_splits=5)` (`fold_seed = 2026`, `bank_seed = 2026`, `seeds = [0, 1]` → `10` corridas pareadas por variante PPO lineal), `train_episodes = 1500`. Huellas SHA-256: `data/item_bank.csv = 07ed76f0…d4706`, `code = ed9d8eac…49b1a`.

| Política / vector de estado-acción | `LG` holdout (media ± DE) | `ΔLG` vs `FS0` [IC 95 % bootstrap] | `p_perm` (signos exacta) | `d_z` | Victorias | Estado |
|---|---:|---:|---:|---:|---:|---|
| **Aleatoria** | `−0.0443` | — | — | — | — | Referencia negativa |
| **Currículo fijo** | `−0.0299` | — | — | — | — | Referencia negativa |
| **`PPOlin_FS0_raw13`** (13 crudas) | `−0.0095 ± 0.0174` | Referencia | — | — | — | Línea base PPO |
| **`PPOlin_FS2_hist6`** (+6 historial) | `−0.0063 ± 0.0162` | `+0.0032 [−0.0051, +0.0122]` | `0.4941` (`Holm 0.4941`) | `0.22` | `5/10` | **Descartada** |
| **`PPOlin_D1_zpd2`** (+2 ZPD sobre crudo) | `+0.0094 ± 0.0159` | `+0.0189 [+0.0107, +0.0278]` | `0.0039` | `1.30` | `9/10` | Ablación intermedia |
| **`PPOlin_FS1_zpd4`** (+4 ZPD sobre crudo) | `+0.0105 ± 0.0150` | `+0.0200 [+0.0130, +0.0274]` | `0.00195` (`Holm 0.0039`) | `1.61` | `10/10` | Adoptada (Sem. 5) |
| **`PPOlin_D2_clean5_zpd2`** (7 limpias) | **`+0.0203 ± 0.0167`** | **`+0.0298 [+0.0225, +0.0372]`** | **`0.00195`** | **`2.40`** | **`10/10`** | **Seleccionada (49.4 % del oráculo)** |
| **Oráculo ZPD** (`argmax zpd_hit` con `θ*`) | `+0.0411` | — | — | — | — | Techo heurístico en este simulador |

* **Contraste de depuración pura (`D2_clean5_zpd2` vs `FS1_zpd4`):** `Δ = +0.0098 [+0.0012, +0.0204]`, `p_perm = 0.0605`, `d_z = 0.60`, `8/10` corridas (tendencia favorable, aún no significativa a `α = 0.05` con `n = 10`).
* **Importancia y colinealidad (`sprint2_diagnostics.json`):** `zpd_hit` (`ΔAUC = +0.0959`, `|SHAP| = 0.0677`, `VIF = 2.070`) y `abs_gap` (`ΔAUC = +0.0140`, `|SHAP| = 0.0127`, `VIF = 2.125`); se podaron `8` columnas sin variación (`n_dead_xt = 8`: `u_acc`, `u_streak`, `tau_1`, `tau_2`, `E_hat`, `sigma_1..3`) y `2` colineales (`zpd_prox`, `signed_gap`).
* **Sensibilidad sin oráculo (`runs_elo.json`, `k = 0.1`, `n = 5` folds):** `D2_elo = +0.0009 ± 0.0203` vs `FS1_elo`; ventaja reportada `+0.0095`, `4/5` folds, `p = 0.25` (no concluyente; el resultado principal depende de observar `θ*`).

---

## 2. Trazabilidad semanal de ramas y entregables

| Semana / Etapa | Rama en GitHub | Artefactos principales en el repositorio | Resultado clave |
|---|---|---|---|
| **Sem. 4 — EDA** | `eda` | [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) · [`figures/eda/`](figures/eda/) | Caracterización de `300` ítems IRT-3PL y auditoría forense de datos |
| **Sem. 5 — FE** | `feature-engineering` | [`src/feature_engineering.py`](src/feature_engineering.py) · [`src/leakage_checks.py`](src/leakage_checks.py) | Creación de señales ZPD y `5/5 PASS` anti-fuga (`L1–L5`) |
| **Sem. 6 — Poda** | `data-centric-audit-and-pruning` | [`notebooks/02_Feature_Engineering.ipynb`](notebooks/02_Feature_Engineering.ipynb) | Eliminación de `8` columnas muertas y reducción de colinealidad |
| **Sem. 7 — Diagnósticos** | `feat/variable-importance-and-consistency-matrix` | [`notebooks/03_Importancia_Variables_y_Matriz_Consistencia.ipynb`](notebooks/03_Importancia_Variables_y_Matriz_Consistencia.ipynb) · [`docs/matriz_consistencia.md`](docs/matriz_consistencia.md) | `ΔAUC`, `\|SHAP\|`, `VIF <= 2.125` y matriz de consistencia |
| **Sem. 8–9 — Saneamiento e Informe** | `feat/hyperparameter-optimization` | [`docs/Informe_Parcial_PI2_UNI_Grupo2_Academica_FINAL.pdf`](docs/Informe_Parcial_PI2_UNI_Grupo2_Academica_FINAL.pdf) · [`data/results/`](data/results/) | Depuración de evidencia real del Régimen B e informe parcial UNI |

*(Nota: la rama final conserva el nombre `feat/hyperparameter-optimization`, pero la búsqueda de hiperparámetros y `Dyna-PPO` quedan explícitamente planificados para las Semanas 10–14.)*

---

## 3. Estructura del repositorio y reproducción

```text
├── data/
│   ├── item_bank.csv & SNAPSHOT.json      # Banco determinista (30 textos × 10 ítems, bank_seed=2026)
│   ├── results/                           # Salidas reales Régimen B: summary.json, runs_oracle.json, runs_elo.json,
│   │                                      # references.json, sprint2_diagnostics.json, tabla_comparativa_ab.csv
│   └── sample/                            # Muestras CSV de EDA (Sem. 4) y features transformadas (Sem. 5–7)
├── docs/
│   ├── Informe_Parcial_PI2_UNI_Grupo2_Academica_FINAL.docx / .pdf
│   └── matriz_consistencia.md
├── figures/
│   ├── eda/                               # 8 gráficos del EDA y auditoría inicial (Sem. 4)
│   ├── fe/                                # 5 gráficos del experimento A/B, SHAP, VIF y calibración (Sem. 5–7)
│   └── informe_parcial/                   # 4 figuras actualizadas del Informe Parcial (Sem. 9)
├── logs/metrics_experimentos.csv          # Log completo de corridas ejecutadas
├── notebooks/                             # 01_EDA.ipynb, 02_Feature_Engineering.ipynb, 03_Importancia_Variables_y_Matriz_Consistencia.ipynb
├── src/                                   # simulator.py, feature_engineering.py, agent_ppo_linear.py,
│                                          # experiment.py, leakage_checks.py, diagnostics.py, report.py, run_ab.py
└── tests/test_leakage_checks.py           # Pruebas unitarias L1–L5 y determinismo
```

### Comandos de verificación rápida

```bash
pip install -r requirements.txt
python -m src.leakage_checks
python -m unittest tests/test_leakage_checks.py -v
python -m src.run_ab --all --robustness --workers 4
```

---

## 4. Planificación pendiente (Semanas 10–16)

1. **Semanas 10–11:** Implementar modelo del mundo y `Dyna-PPO` real en `src/`, con logs por corrida.
2. **Semana 12:** Ejecutar búsqueda de hiperparámetros con espacio prefijado antes de correr.
3. **Semanas 13–14:** Sustituir `θ*` por estimador aprendido (`MSKT` / `NeuralCD`) y ejecutar evaluación confirmatoria sobre particiones y semillas nuevas (`HG` enmendada: IC 95 % sin incluir `0` y `≥ 40 %` del oráculo).
4. **Semanas 15–16:** Informe final y sustentación.
