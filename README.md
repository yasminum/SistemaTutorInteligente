# Auditoría de Datos, Selección de Variables (VIF / SHAP / Permutación), Calibración y Experimento A/B

**Sistema de Tutoría Inteligente Adaptativa (DRL) · Grupo 2 · Maestría en IA · UNI**  
*Proyecto de Investigación 2 — Prof. Glen Rodríguez · Rama `feat/variable-importance-and-consistency-matrix` (heredada de `data-centric-audit-and-pruning`)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasminum/SistemaTutorInteligente/blob/feat/variable-importance-and-consistency-matrix/notebooks/02_Feature_Engineering.ipynb)
![Leakage](https://img.shields.io/badge/pruebas_de_fuga-5%2F5_PASS-blue)
![VIF](https://img.shields.io/badge/VIF_m%C3%A1x-2.12_(vs_3.96)-success)
![Brier](https://img.shields.io/badge/Brier_Score-0.070_(vs_0.166)-informational)

> 📄 **Documento dedicado de Matriz de Consistencia:** [`docs/matriz_consistencia.md`](docs/matriz_consistencia.md)

---

## 0. Matriz de Consistencia Metodológica y Operacional

| Problema | Objetivos | Hipótesis | Variables (Independientes y Dependientes) | Indicadores y Evidencia Experimental |
|---|---|---|---|---|
| **Problema General:**<br>¿En qué medida un Sistema de Tutoría Inteligente Adaptativa basado en DRL, alimentado con una representación depurada de acoplamiento habilidad–dificultad, mejora la ganancia de aprendizaje en comprensión lectora frente a políticas de estado crudo sin depurar? | **Objetivo General (OG):**<br>Diseñar, implementar y validar un Sistema de Tutoría Inteligente Adaptativa basado en DRL (PPO / Dyna-PPO), integrando selección de variables sin colinealidad y calibración probabilística, para optimizar la secuencia de textos y maximizar la comprensión lectora. | **Hipótesis General (HG):**<br>El Sistema de Tutoría Inteligente basado en DRL con representación depurada (`Var2: clean5 + 2 vars ZPD`) incrementa significativamente la ganancia de aprendizaje normalizada ($LG_{\text{sim}}$) en estudiantes no vistos respecto al *Baseline* de estado crudo ($p < 0.05$). | **Variable Independiente General (VI):**<br>Representación de datos y política de secuenciación adaptativa DRL (`Baseline FS0_raw13` vs. `Var1 FS1_zpd4` vs. `Var2 D2_clean5_zpd2`).<br><br>**Variable Dependiente General (VD):**<br>Ganancia de aprendizaje en comprensión lectora ($\theta_1$ literal, $\theta_2$ inferencial, $\theta_3$ crítica). | • **Ganancia de Aprendizaje Normalizada de Hake ($LG_{\text{sim}}$):**<br>`Baseline`: $-0.0095 \pm 0.0174$<br>`Var1 (4 vars)`: $+0.0105 \pm 0.0150$<br>**`Var2 (clean5 + 2 vars)`**: **$+0.0203 \pm 0.0167$**<br>• **Mejora en $LG$ (`Var2 − Baseline`):** $+0.0298$, IC95 $[+0.0225, +0.0372]$, $p_{\text{Holm}}=0.006$, $d_z=2.50$ ($10/10$ pares y $5/5$ folds).<br>• **Eventos de aprendizaje/paso:** $0.334$ vs. $0.273$. |
| **Problema Específico 1 (Importancia de Variables, Colinealidad y Calibración):**<br>¿Cuáles son las variables de estado y de interacción alumno–texto que explican la transición de aprendizaje lector sin colinealidad ni sesgo por ruido, y qué discriminación y calibración alcanzan? | **Objetivo Específico 1 (OE1):**<br>Identificar y justificar el subconjunto mínimo de variables mediante importancia en árboles (`Impurity Gain`), importancia por permutación en validación, valores `SHAP`, `VIF` y curvas de calibración (`Brier Score`). | **Hipótesis Específica 1 (HE1):**<br>La poda de las 8 columnas sin señal del estado junto con **2 variables de acoplamiento ZPD (`zpd_hit` y `abs_gap`)** mantiene $\text{VIF} < 2.5$, repite el Top-2 en el $100\%$ de los folds, alcanza $\text{ROC-AUC} \ge 0.85$ a nivel de ítem y reduce el `Brier Score` por debajo de $0.10$. | **var(s) indep(s):**<br>1. **`zpd_hit`**: fracción de preguntas del texto en ventana de aprendizaje ($\lvert\theta_k - b_{adj}\rvert < 0.5$).<br>2. **`abs_gap`**: diferencia absoluta de nivel ($\lvert\theta_k - b_{adj}\rvert$).<br>3. **`clean5`**: $[\theta_1, \theta_2, \theta_3, t/50, b_{t-1}]$.<br><br>**var depend:**<br>Probabilidad calibrada de transición de aprendizaje $P(\Delta\theta_k > 0 \mid s_t, a_t)$. | • **Importancia por Permutación (5 folds):** `zpd_hit` puesto **#1** ($0.0960$) y `abs_gap` puesto **#2** ($0.0142$) en los **$5/5$ folds**.<br>• **Impacto medio `\|SHAP\|`:** `zpd_hit` $= 0.0677$, `abs_gap` $= 0.0127$.<br>• **Colinealidad (`VIF`):** máx. **$2.12$** (baja desde $3.96$).<br>• **`ROC-AUC`:** **$0.9261$** (GBDT con `zpd_hit + abs_gap`) y **$0.9477$** (con `p_item + abs_gap`) vs. $0.4974$ del Baseline.<br>• **`Brier Score`:** **$0.0703 \pm 0.0025$** ($-57.5\%$ vs. Baseline $0.1656$); `ECE` texto $= 0.053$. |
| **Problema Específico 2 (Eficiencia Muestral y Seguridad CMDP):**<br>¿En qué medida la eliminación del ruido amplificado por el escalador y la reducción a 2 variables ZPD aceleran la convergencia del agente PPO sin incrementar la frustración del alumno? | **Objetivo Específico 2 (OE2):**<br>Evaluar mediante curvas de aprendizaje (Train vs. Validación) el impacto de la depuración de datos sobre la velocidad de convergencia en episodios y sobre la tasa de activación del escudo CMDP. | **Hipótesis Específica 2 (HE2):**<br>La variante depurada (`Var2`) corrige el alto sesgo (*underfitting*) del `Baseline`, alcanza el desempeño final del `Baseline` en $\le 500$ episodios y mantiene la tasa del escudo en $\le 0.02$ por paso. | **var(s) indep(s):**<br>Poda de las 8 columnas sin señal del estado (eliminando la amplificación de $34.6\times$ del ruido $\sigma_{1..3}$ en `StandardScaler`).<br><br>**var depend:**<br>Episodios hasta convergencia y tasa de activación del escudo CMDP. | • **Episodios hasta alcanzar el $LG$ final del Baseline (mediana):** **$275$ episodios** en `Var2` (cumple $\le 500$) vs. $350$ en `Var1` y $750$ en `Baseline` (**$-63.3\%$ episodios**).<br>• **Tasa de activación del escudo CMDP (racha $\ge 4$ errores):** **$0.017$ por paso** (sin incremento respecto al Baseline). |
| **Problema Específico 3 (Complejidad Computacional y Robustez sin Oráculo):**<br>¿Cuál es el efecto de la poda de variables sobre el costo de inferencia y sobre la ganancia cuando la habilidad $\hat{\theta}$ se estima en línea? | **Objetivo Específico 3 (OE3):**<br>Cuantificar la reducción de parámetros, la latencia por decisión y la estabilidad del $LG$ al sustituir $\theta^*$ por un estimador Elo observable ($\hat{\theta}_k$). | **Hipótesis Específica 3 (HE3):**<br>El modelo depurado (`Var2`) reduce en $>50\%$ los parámetros entrenables, mantiene latencia $< 1\text{ ms}$ y conserva ganancia superior al `Baseline` con habilidad estimada $\hat{\theta}$. | **var(s) indep(s):**<br>Dimensión $(d_s=5, d_a=2)$ y modo de observación ($\theta^*$ exacta vs. $\hat{\theta}$ Elo online).<br><br>**var depend:**<br>Número de parámetros, latencia (ms) y $LG_{\text{sim}}$ con $\hat{\theta}$. | • **Parámetros entrenables:** **$542$** en `Var2` vs. $1\,264$ en `Var1` (**$-57.1\%$**).<br>• **Latencia mediana:** **$0.118\text{ ms}$** ($-21.3\%$ vs. `Var1`).<br>• **Mejora en $LG$ con $\hat{\theta}$ Elo:** $+0.0110$ sobre Baseline ($4/5$ folds). |

---

## Diagnóstico de ingeniería

El vector de observación crudo $x_t \in \mathbb{R}^{13}$ de `environment.py` presenta dos defectos de acondicionamiento de señal que degradan cualquier política lineal:
1. **Amplificación de ruido blanco ($34.6\times$) y canales muertos:** cinco canales tienen varianza nula ($\text{SD}=0.000$) y tres canales ($\sigma_{1..3} \sim U(0.05, 0.15)$, $\text{SD}=0.0289$) son ruido blanco independiente del alumno. Al estandarizar con `StandardScaler`, las tres columnas de ruido se multiplican por $1/0.0289 = 34.6\times$ hasta varianza unitaria, inyectando ruido en $270$ pesos de la política. En árboles (`RandomForest`, 300 árboles), estos tres canales absorben **$16.0\%$ de ganancia por impureza espuria**, mientras que su `Permutation Importance` en validación es **$\approx 0.000$**.
2. **No-separabilidad lineal del evento de aprendizaje:** la transición $\Delta\theta_k > 0$ ocurre solo cuando $|\theta_k - b_{adj}| < 0.5$ y hay acierto, una función par de la diferencia $(\theta_k - b_{adj})$ que un hiperplano $W x_t$ sobre coordenadas separadas no puede resolver ($\text{AUC}=0.497$, $\text{Brier}=0.1656$, $\text{ECE}=0.258$).

Al podar las **8 columnas sin señal del estado** ($\mathbb{R}^{13} \to \mathbb{R}^5$) y conservar **únicamente 2 variables de acoplamiento ZPD por acción** (`zpd_hit` y `abs_gap`, que ocupan los puestos `#1` y `#2` en `Permutation Importance` y `SHAP` en los **5 de 5 folds**, reduciendo el $\text{VIF}$ máximo de $3.96$ a $2.12$ y los parámetros de la política de $1\,264$ a $542$), el *Learning Gain* en holdout sube de **$-0.0095 \pm 0.0174$** (`Baseline`) y **$+0.0105 \pm 0.0150$** (`Var1` con 4 variables ZPD sobre estado sucio) a **$+0.0203 \pm 0.0167$** (`Var2` adoptada: $\Delta LG = +0.0298$, IC95 $[+0.0225, +0.0372]$, $p_{\text{Holm}}=0.006$, $10/10$ pares ganados y signo positivo en los $5/5$ folds).

---

## 1. Síntesis del EDA sobre los Datos Crudos → Decisiones de Datos

| Código | Hallazgo en los datos crudos (`01_EDA.ipynb` + auditoría de $x_t$) | Evidencia cuantitativa | Transformación sobre los datos |
|---|---|---|---|
| **P1. Desalineación de escalas ($\theta$ vs. $b$)** | $\theta_k \in [0, 1]$ ($\text{Beta}(2,2)$), mientras que $b_i \sim \mathcal{N}(0, 1)$ ($[-3.90, +3.32]$). | El **66.4 %** de los 300 ítems cae fuera de $[0, 1]$ ($51.8\%$ con $b<0$; $14.6\%$ con $b>1$). En nivel 0, el **58.4 %** de los ítems son triviales ($P>0.90$). | Variables de acoplamiento en la escala nativa del generador ($g = \theta_k - b_{adj}$): $-\lvert\theta - b_{adj}\rvert$ alcanza $\text{AUC}=0.925$ frente a $0.640$ del escalado sigmoidal directo $-\lvert\theta - \sigma(b)\rvert$. |
| **P2. Canales constantes y ruido amplificado en $x_t \in \mathbb{R}^{13}$** | 5 entradas constantes (`u_acc`, `u_streak`, `tau_rt`, `tau_lag`, `E`) y 3 entradas de ruido uniforme $\sigma_{1..3} \sim U(0.05, 0.15)$. | $\text{SD}=0.000$ en 5 columnas; `StandardScaler` amplifica $\sigma_{1..3}$ en **$34.6\times$** ($\text{SD}=0.0289 \to 1.0$). | **Poda de estado (`clean5`)**: eliminar las 8 columnas sin señal y retener $[\theta_1, \theta_2, \theta_3, t/H, b_{t-1}] \in \mathbb{R}^5$. |
| **P3. Colinealidad en candidatas ZPD y temporales** | En `FS1_zpd4`, `abs_gap`, `zpd_prox` y `zpd_hit` miden la misma distancia $|\theta_k - b_{adj}|$; en `FS2_hist6`, `last_lvl_0..2` suma 1. | $r(\text{zpd\_prox}, \text{zpd\_hit}) = 0.849$ ($\text{VIF}=3.96$); en `FS2_hist6`, $\text{VIF}=\infty$. | **Poda a 2 variables ZPD (`zpd2`)**: conservar solo `zpd_hit` y `abs_gap` ($\text{VIF}_{\max}=2.12$; estado $\text{VIF} \in [1.00, 1.03]$). |
| **P4. Sobreajuste a un alumno por semilla y sesgo de $\theta_0$** | `train.py` repetía el mismo alumno $5\,000$ episodios y $\rho(\theta_{\text{pre}}, LG) = -0.95$. | Población efectiva original: 1 alumno/semilla. | Cohorte de 100 alumnos con `GroupKFold` (5 folds: 80 train / 20 test) y evaluación pareada. |

---

## 2. Selección y Justificación de Features: VIF, Importancias, SHAP y Estabilidad en 5 Folds

![Importancias, SHAP, VIF y Estabilidad](figures/fe/04_importancias_shap_vif_estabilidad.png)

### 2.1 Control de Colinealidad (VIF) y Poda de Variables

| Variable | Rol | VIF en `FS1_zpd4` (4 vars) | VIF en `D2_clean5_zpd2` (2 vars) | `RF Gain` (media 5 folds) | `Permutation ΔAUC` (media 5 folds) | `Mean \|SHAP\|` (validación) | Ranking en `Fold 0..4` | Decisión |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`zpd_hit`** | Acción: $\overline{\mathbb{1}[\lvert\theta_k - b_{adj}\rvert < 0.5]}$ | 3.96 | **2.07** | 0.1875 | **0.0960** | **0.0677** | **#1, #1, #1, #1, #1** | **CONSERVAR (Var ZPD #1)** |
| **`abs_gap`** | Acción: $\overline{\lvert\theta_k - b_{adj}\rvert}$ | 2.54 | **2.12** | 0.1008 | **0.0142** | **0.0127** | **#2, #2, #2, #2, #2** | **CONSERVAR (Var ZPD #2)** |
| `zpd_prox` | Acción: $\overline{\exp(-g^2/2\sigma_F^2)}$ | 3.84 | — | 0.1297 | — (colineal) | — | — | **DESCARTAR** ($r=0.849$ con `zpd_hit`) |
| `p_hat` | Acción: $\overline{P_{3PL}}$ | 1.41 | — | 0.0728 | 0.0007 | — | — | **DESCARTAR** ($\Delta\text{AUC}\approx 0$) |
| `theta_1..3` | Estado: habilidades por KC | 1.01–1.03 | **1.01–1.03** | 0.051–0.067 | 0.001–0.005 | 0.004–0.015 | #3–#7 | **CONSERVAR (`clean5`)** |
| `u_progress`, `b_prev_raw` | Estado: paso $t/H$ y $b_{t-1}$ | 1.00 | **1.00** | 0.039–0.052 | 0.001 | 0.007–0.009 | #3–#7 | **CONSERVAR (`clean5`)** |
| `sigma_1..3` | Estado: ruido $U(0.05, 0.15)$ | 1.00 | — | **0.1604 (suma)** | **0.0001–0.0015** | — | Ruido | **PODAR** (amplificado $34.6\times$) |
| `u_acc`, `u_streak`, `tau_1..2`, `E` | Estado: 5 constantes fijas | $\infty$ ($\text{SD}=0$) | — | 0.0000 | 0.0000 | — | Constante | **PODAR** (varianza cero) |

---

## 3. Diagnósticos de Aprendizaje (Train vs. Validación) y Calibración (`Brier Score`)

![Learning Curves y Calibración](figures/fe/05_learning_curves_y_calibracion.png)

| Modelo / Representación de Datos | Diagnóstico Learning Curve (Train vs. CV 5-fold) | `Brier Score` (5 folds, ↓ mejor) | `ROC-AUC` (5 folds, ↑ mejor) | `Log-Loss` | `ECE` (error de calibración, ↓ mejor) |
|---|---|:---:|:---:|:---:|:---:|
| **`Baseline (FS0_raw13)`** | **Alto sesgo (*underfitting*):** curvas de train y validación pegadas en $\text{AUC}\approx 0.50$ y $LG \approx -0.010$; añadir más datos no mejora. | $0.1656 \pm 0.0086$ | $0.4974 \pm 0.0199$ | $0.5174$ | $0.2584$ |
| **`FS2_hist6` (Historial sin ZPD)** | **Alto sesgo:** sin el cruce $(\theta_k, b_{adj})$, el historial de aciertos no predice si la acción entrante caerá en la ZPD. | $0.1690 \pm 0.0097$ | $0.5003 \pm 0.0185$ | $0.5259$ | $0.2756$ |
| **`D2 Pre-decisión por Texto` (`clean5 + 2 vars ZPD` + Platt en train)** | **Bajo sesgo y baja varianza:** mejora sostenida desde $N=400$ hasta $N=4\,000$ y convergencia PPO en $275$ episodios. | **$0.1540 \pm 0.0085$** | **$0.6370 \pm 0.0192$** | **$0.4876$** | **$0.0535$ ($-79\%$) |
| **`D2 Transición por Ítem` (`clean5 + 2 vars ZPD` a nivel ítem)** | **Techo de resolución del modelo de mundo:** brecha train-val $<0.025$ desde $N=1\,100$ registros. | **$0.0703 \pm 0.0025$ ($-58\%$) | **$0.9477 \pm 0.0063$** | **$0.2194$** | **$0.0581$** |

---

## 4. Resultados del Experimento A/B en Variantes de LOS DATOS (5 Folds y 10 Pares)

**Protocolo idéntico entre variantes (Checklist Sprint 2):** 100 estudiantes sintéticos $\to$ `GroupKFold` (5 folds: 80 train / 20 test) $\times$ semillas `{0, 1}` $= 10$ pares pareados. Presupuesto fijo: $1\,500$ episodios $\times 50$ pasos $= 75\,000$ pasos/corrida. Evaluación en los 20 alumnos de test con política *greedy* y pesos congelados (`5/5 PASS` en `python -m src.leakage_checks`).

### 4.1 Tabla comparativa estándar

| ID Experimento | Variante de Datos | Dimensión $(d_s, d_a)$ | Parámetros Política | $LG_{\text{sim}}$ holdout (media ± DE) | Acierto | Eventos de aprendizaje/paso | Violaciones (shield)/paso | Episodios a $LG_{\text{base}}$ (mediana) | Latencia mediana (ms) | Decisión |
|---|---|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `FE6-PPOlin_FS0_raw13` | **Baseline** (`FS0_raw13`) | $(13, 0)$ | $1\,260$ | $-0.0095 \pm 0.0174$ | $0.730$ | $0.273$ | $0.017$ | $750$ ($10/10$) | $0.031$ | Referencia |
| `FE6-PPOlin_FS2_hist6` | **Var Historial** (`FS2_hist6`) | $(19, 0)$ | $1\,800$ | $-0.0063 \pm 0.0162$ | $0.737$ | $0.272$ | $0.015$ | $775$ ($8/10$) | $0.034$ | **DESCARTAR** |
| `FE6-PPOlin_FS1_zpd4` | **Var1** (`FS1_zpd4`, 4 vars colineales) | $(13, 4)$ | $1\,264$ | $+0.0105 \pm 0.0150$ | $0.716$ | $0.312$ | $0.016$ | $350$ ($10/10$) | $0.150$ | Superada por `D2` |
| `FE6-PPOlin_D1_zpd2` | **Ablación 2 vars** (`D1_zpd2` en estado crudo) | $(13, 2)$ | $1\,262$ | $+0.0094 \pm 0.0159$ | $0.707$ | $0.313$ | $0.022$ | $325$ ($10/10$) | $0.142$ | Ablación intermedia |
| `FE6-PPOlin_D2_clean5_zpd2` | **Var2 Adoptada** (`clean5 + 2 vars ZPD`) | **$(5, 2)$** | **$542$** | **$+0.0203 \pm 0.0167$** | $0.711$ | **$0.334$** | $0.017$ | **$275$ ($10/10$)** | **$0.118$** | **ADOPTAR** |

### 4.2 Comparación Fold por Fold (`Fold 0` a `Fold 4`) y Efecto Pareado (10 pares)

![Ablación por Fold y Efecto Pareado](figures/fe/06_ablacion_por_fold_y_pareada.png)

| Fold (`GroupKFold`, 20 alumnos test/fold) | `Baseline (FS0_raw13)` | `FS2_hist6` (Historial) | `Var1 (FS1_zpd4, 4 vars)` | `D1_zpd2` (2 vars en estado crudo) | **`Var2 Adoptada (D2: clean5 + 2 vars)`** | **Mejora en $LG$ (`Var2 − Baseline`)** | **Mejora en $LG$ (`Var2 − Var1`)** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | $-0.0262$ | $-0.0067$ | $-0.0024$ | $-0.0014$ | **$+0.0026$** | $+0.0289$ | $+0.0050$ |
| **Fold 1** | $-0.0303$ | $-0.0302$ | $+0.0016$ | $-0.0052$ | **$+0.0071$** | $+0.0375$ | $+0.0055$ |
| **Fold 2** | $+0.0054$ | $+0.0049$ | $+0.0308$ | $+0.0306$ | **$+0.0359$** | $+0.0304$ | $+0.0051$ |
| **Fold 3** | $-0.0049$ | $+0.0020$ | $+0.0014$ | $+0.0068$ | **$+0.0322$** | $+0.0371$ | $+0.0308$ |
| **Fold 4** | $+0.0086$ | $-0.0015$ | $+0.0208$ | $+0.0160$ | **$+0.0226$** | $+0.0140$ | $+0.0018$ |
| **Media 5 Folds (10 pares)** | **$-0.0095$** | **$-0.0063$** | **$+0.0105$** | **$+0.0094$** | **$+0.0203$** | **$+0.0298$** | **$+0.0098$** |

| Contraste pareado ($n=10$ pares `fold × seed`) | $\Delta LG_{\text{sim}}$ | IC95 Bootstrap | $p$ permutación exacta | $p_{\text{Holm}}$ | $d_z$ de Cohen | Pares ganados |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Var2 (D2_clean5_zpd2)` − `Baseline`** | **$+0.0298$** | **$[+0.0225, +0.0372]$** | **$0.0020$** | **$0.0060$** | **$2.50$** | **$10/10$** |
| **`Var1 (FS1_zpd4)` − `Baseline`** | $+0.0200$ | $[+0.0130, +0.0274]$ | $0.0020$ | $0.0060$ | $1.61$ | $10/10$ |
| **`D1 (zpd2)` − `Baseline`** | $+0.0189$ | $[+0.0107, +0.0278]$ | $0.0039$ | $0.0078$ | $1.31$ | $9/10$ |
| **`FS2_hist6` − `Baseline`** | $+0.0032$ | $[-0.0051, +0.0122]$ | $0.4941$ | $0.4941$ | $0.22$ | $5/10$ |
| **`Var2 (D2_clean5_zpd2)` − `Var1 (FS1_zpd4)`** *(efecto puro de podar ruido y colinealidad)* | **$+0.0098$** | **$[+0.0021, +0.0189]$** | **$0.0137$** | **$0.0137$** | **$0.71$** | **$8/10$ ($5/5$ folds)** |

---

## 5. Sensibilidad sin Oráculo ($\hat{\theta}$ Elo Online) y Transición a Aula Real

Al reemplazar la habilidad verdadera $\theta^*$ por el estimador recursivo Elo ($\hat{\theta}_k \leftarrow \text{clip}(\hat{\theta}_k + 0.1(y_t - \hat{P}_{3PL}), 0, 1)$, inicializado en $0.5$), `Var2 (D2_clean5_zpd2)` mantiene el mejor desempeño de todas las variantes también bajo incertidumbre de medición:

| Variante | $LG_{\text{sim}}$ con $\theta^*$ (seed 0) | $LG_{\text{sim}}$ con $\hat{\theta}$ Elo observable (seed 0) | Ventaja sobre Baseline con $\hat{\theta}$ Elo |
|---|:---:|:---:|:---:|
| `PPOlin_FS0_raw13` (Baseline) | $-0.0106$ | $-0.0101$ | — |
| `PPOlin_FS1_zpd4` (4 vars ZPD + estado crudo) | $+0.0136$ | $-0.0006$ | $+0.0095$ ($4/5$ folds) |
| **`PPOlin_D2_clean5_zpd2` (2 vars ZPD + estado limpio)** | **$+0.0176$** | **$+0.0009$** | **$+0.0110$ ($4/5$ folds)** |

### 5.1 Protocolo de despliegue cuando sea real en aula (Tesis II)

| Componente | En el simulador (Tesis I) | En despliegue real con estudiantes (Tesis II) |
|---|---|---|
| **Estado depurado (`clean5`)** | $[\theta_1, \theta_2, \theta_3, t/H, b_{t-1}] \in \mathbb{R}^5$ | $[\hat{\theta}_1, \hat{\theta}_2, \hat{\theta}_3, t/H, b_{t-1}] \in \mathbb{R}^5$ inicializado con pretest corto de 6 ítems (2 por competencia) y actualizado en línea por IRT bayesiano / Elo tras cada respuesta. |
| **Las 2 variables ZPD (`zpd2`)** | `zpd_hit` y `abs_gap` sobre `item_bank.csv` sintético | **Idéntica función `action_features(theta_hat)`**: solo cambia la tabla `item_bank.csv` por los parámetros $b_i$ calibrados en el piloto ($N \ge 60$ alumnos con Elo multimodelo; Pelánek, 2016) y la calibración sigmoidal de Platt sobre las probabilidades de transición. |
| **Seguridad pedagógica (CMDP)** | Shield determinista ante racha $\ge 4$ errores ($\text{nivel} \leftarrow 0$) | Mismo shield en producción más restricción de soporte $\lvert\hat{\theta}_k - b_{adj}\rvert \le 0.6$ para evitar exploración fuera de zona calibrada. |

---

## 6. Reproducibilidad (Checklist Sprint 2)

```bash
pip install -r requirements.txt                       # numpy, pandas, matplotlib, scikit-learn, scipy
python -m src.leakage_checks                          # 5/5 pruebas automaticas de cero fuga
python -m src.run_ab --variant PPOlin_FS0_raw13       # Baseline (estado crudo 13 vars)
python -m src.run_ab --variant PPOlin_FS1_zpd4        # Var1 (estado crudo + 4 vars ZPD)
python -m src.run_ab --variant PPOlin_D2_clean5_zpd2  # Var2 Adoptada (estado limpio 5 vars + 2 vars ZPD)
jupyter notebook notebooks/02_Feature_Engineering.ipynb
```

| Artefacto | Valor verificable |
|---|---|
| Snapshot de datos | `data/item_bank.csv` · SHA-256 `07ed76f0bee0a21d…` · 300 filas (`data/SNAPSHOT.json`) |
| Código experimental | SHA-256 `ed9d8eac50ac6a5d…` (`simulator.py`, `feature_engineering.py`, `agent_ppo_linear.py`, `experiment.py`) |
| Logs y diagnósticos | `logs/metrics_experimentos.csv`, `data/results/runs_oracle.json`, `runs_elo.json`, `sprint2_diagnostics.json`, `tabla_comparativa_ab.csv` |

## Referencias

* Birnbaum, A. (1968). Some latent trait models. En Lord & Novick, *Statistical Theories of Mental Test Scores*. Addison-Wesley.
* Doroudi, S., Aleven, V., & Brunskill, E. (2019). Where's the reward? A review of reinforcement learning for instructional sequencing. *IJAIED, 29*, 568–620. https://doi.org/10.1007/s40593-019-00187-x
* Hake, R. R. (1998). Interactive-engagement versus traditional methods. *American Journal of Physics, 66*(1), 64–74. https://doi.org/10.1119/1.18809
* Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD, 6*(4), 1–21. https://doi.org/10.1145/2382577.2382579
* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS), 30*, 4765–4774.
* Pelánek, R. (2016). Applications of the Elo rating system in adaptive educational systems. *Computers & Education, 98*, 169–179. https://doi.org/10.1016/j.compedu.2016.03.017
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
