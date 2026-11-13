# Matriz de Consistencia Metodológica y Operacional (Enmendada — Semana 9)

**Proyecto de Investigación II · Maestría en Inteligencia Artificial · UNI**  
**Título de Tesis:** *Sistema de Tutoría Inteligente Adaptativa para Comprensión Lectora mediante Deep Reinforcement Learning Integrado con MSKT y NeuralCD: Un Enfoque Basado en Literatura Peruana*  
**Autores (Grupo 2):** Flores Rivera, Amit Roy · Urrego Montoya, Yasmin  
**Asesora:** Mg. Ing. María Fernanda Tejada Begazo · **Docente del curso:** Dr. Glen Dario Rodríguez Rafael  
**Rama de entrega:** `feat/hyperparameter-optimization` (evidencia real del prototipo PPO lineal en Régimen B)

> **Enmienda al protocolo (Semana 9):** El umbral original `LG >= 0.50` se retira porque el simulador IRT-3PL de `50` pasos con olvido activo acota la ganancia en `0.3255` (acierto en ZPD en los 50 pasos) y la heurística con oráculo logra `+0.0411`. Las hipótesis se formulan en diferencias pareadas y fracción del oráculo; `Dyna-PPO`, `HPO` y `MSKT/NeuralCD` quedan planificados para las Semanas 10–16.

---

## 1. Matriz de Consistencia (Problema · Objetivos · Hipótesis · Variables · Indicadores)

| Problema | Objetivo | Hipótesis | Variables (VI / VD) | Indicadores y resultado real al corte (`summary.json`, `sprint2_diagnostics.json`) |
|---|---|---|---|---|
| **Problema General (PG):**<br>¿En qué medida una política de aprendizaje por refuerzo, alimentada con variables depuradas de acoplamiento habilidad–dificultad, mejora la ganancia de aprendizaje simulada frente a políticas base en un entorno IRT-3PL? | **Objetivo General (OG):**<br>Desarrollar y evaluar en simulación una política de secuenciación adaptativa de textos basada en aprendizaje por refuerzo que maximice la ganancia de aprendizaje en comprensión lectora. | **Hipótesis General (HG — enmendada):**<br>PPO con el vector depurado de 7 variables (`D2_clean5_zpd2`) supera a PPO con el vector crudo (`FS0_raw13`) y a las políticas aleatoria y de currículo fijo.<br>*Criterio confirmatorio propuesto (Sem. 13–14):* particiones/semillas nuevas, IC 95 % sin incluir `0` y `≥ 40 %` del oráculo. | **VI:** Conjunto de variables de estado-acción (`FS0_raw13`, `FS2_hist6`, `D1_zpd2`, `FS1_zpd4`, `D2_clean5_zpd2`) y tipo de política.<br><br>**VD:** Ganancia de aprendizaje normalizada simulada (`LG` en comprensión literal `θ_1`, inferencial `θ_2` y crítica `θ_3`). | • `Aleatoria`: `−0.0443` · `Currículo fijo`: `−0.0299`<br>• `FS0_raw13`: `−0.0095 ± 0.0174`<br>• `D2_clean5_zpd2`: **`+0.0203 ± 0.0167`** (`49.4 %` del oráculo `+0.0411`)<br>• `Δ(D2 − FS0)` = **`+0.0298 [+0.0225, +0.0372]`**, `p_perm = 0.00195`, `d_z = 2.40`, `10/10` corridas.<br>**Estado:** Respaldada de forma exploratoria. |
| **Problema Específico 1 (PE1):**<br>¿Cuánto aportan las variables de acoplamiento (`zpd_hit`, `abs_gap`) a predecir la transición de aprendizaje y a la ganancia de la política? | **Objetivo Específico 1 (OE1):**<br>Determinar el aporte de las variables de acoplamiento ZPD sobre la predicción de transición de aprendizaje y el desempeño de la política. | **Hipótesis Específica 1 (HE1):**<br>`zpd_hit` y `abs_gap` son las dos variables más informativas del vector, sin colinealidad severa (`VIF < 2.5`). | **VI:** `zpd_hit` (`1[|θ_k − b| < 0.5]`) y `abs_gap` (`|θ_k − b|`).<br><br>**VD:** Transición de aprendizaje y `LG`. | • **Permutación (`ΔAUC`, 5 folds):** `zpd_hit = +0.0959` (`#1`), `abs_gap = +0.0140` (`#2`).<br>• **`\|SHAP\|` medio:** `zpd_hit = 0.0677` (`#1`), `abs_gap = 0.0127` (`#4`, detrás de `θ_1 = 0.0153` y `θ_3 = 0.0148`).<br>• **`VIF` máximo:** `2.125` (`abs_gap`).<br>**Estado:** **Parcialmente confirmada.** |
| **Problema Específico 2 (PE2):**<br>¿Mejora la ganancia al eliminar columnas sin información (`n_dead_xt = 8`) y variables colineales del estado? | **Objetivo Específico 2 (OE2):**<br>Evaluar el efecto de depurar el estado sobre la ganancia y eficiencia muestral frente al vector con 4 variables ZPD (`FS1_zpd4`). | **Hipótesis Específica 2 (HE2):**<br>Depurar el estado mejora la ganancia frente al vector con 4 variables ZPD (`FS1_zpd4`). | **VI:** Estado depurado (`D2_clean5_zpd2`, `7` vars) vs. estado sin depurar (`FS1_zpd4`, `17` vars).<br><br>**VD:** `LG` y episodios hasta línea base. | • `Δ(D2 − FS1)` = `+0.0098 [+0.0012, +0.0204]`, `p_perm = 0.0605`, `d_z = 0.60`, `8/10` corridas.<br>• Episodios hasta `LG` baseline: `275` (`D2`) vs `750` (`FS0`).<br>**Estado:** **No confirmada a `α = 0.05`** (tendencia favorable). |
| **Problema Específico 3 (PE3):**<br>¿Mejora la ganancia y la eficiencia muestral al añadir planificación con modelo (`Dyna-PPO`) y ajuste de hiperparámetros? | **Objetivo Específico 3 (OE3):**<br>Evaluar planificación con modelo (`Dyna-PPO`), ajuste de hiperparámetros y estimadores `MSKT/NeuralCD` (Semanas 10–16). | **Hipótesis Específica 3 (HE3):**<br>`Dyna-PPO` con hiperparámetros ajustados supera a PPO sin modelo. | **VI:** Planificación con modelo del mundo, hiperparámetros y estimador aprendido.<br><br>**VD:** `LG` y eficiencia muestral. | • En sensibilidad sin oráculo (`Elo k = 0.1`, `n = 5`): `+0.0095`, `4/5` folds, `p = 0.25` (no concluyente).<br>• `Dyna-PPO`, `HPO` y `MSKT/NeuralCD` aún no implementados.<br>**Estado:** **No evaluada (Semanas 10–16).** |

---

## 2. Variables evaluadas (`D2_clean5_zpd2` vs. descartadas)

| Variable | Definición operativa | `ΔAUC` Perm. | `\|SHAP\|` | `VIF` (`D2`) | Decisión |
|---|---|:---:|:---:|:---:|---|
| `zpd_hit` | `1[\|θ_k − b\| < 0.5]` | `+0.0959` | `0.0677` | `2.070` | **Retenida** |
| `abs_gap` | `\|θ_k − b\|` | `+0.0140` | `0.0127` | `2.125` | **Retenida** |
| `theta_1` | Habilidad en competencia literal | `−0.0002` | `0.0153` | `1.013` | **Retenida** |
| `theta_3` | Habilidad en competencia crítica | `+0.0047` | `0.0148` | `1.033` | **Retenida** |
| `u_progress` | Progreso normalizado `t / 50` | `−0.0011` | `0.0089` | `1.004` | **Retenida** |
| `b_prev_raw` | Dificultad del texto en `t − 1` | `−0.0003` | `0.0065` | `1.001` | **Retenida** |
| `theta_2` | Habilidad en competencia inferencial | `−0.0025` | `0.0043` | `1.012` | **Retenida** |
| `zpd_prox`, `signed_gap` | Proximidad gaussiana y brecha con signo | — | — | — | **Descartadas por colinealidad** |
| `sigma_1..3`, `u_acc`, `u_streak`, `tau_1`, `tau_2`, `E_hat` | Columnas sin variación (`n_dead_xt = 8`) | — | — | — | **Descartadas** |
