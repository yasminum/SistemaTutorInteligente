# Matriz de Consistencia Metodológica y Operacional (Semana 8 — Versión Corregida)

En los tres objetivos específicos, las **Variables Independientes** corresponden estrictamente a las **variables de datos seleccionadas por importancia** (`zpd_hit`, `abs_gap`, `theta_1..3`, `b_prev_raw`, `u_progress`), la **Variable Dependiente** es el **aprendizaje en comprensión lectora**, y las métricas técnicas (`ROC-AUC`, `Brier Score`, `VIF`, `fANOVA`, episodios) se ubican exclusivamente en la columna de **Indicadores**:

<table>
<thead>
<tr>
<th>Problema</th>
<th>Objetivos</th>
<th>Hipótesis</th>
<th>Variables (Independientes y Dependientes)</th>
<th>Indicadores y Evidencia Experimental</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Problema General:</b><br>¿En qué medida un Sistema de Tutoría Inteligente Adaptativa basado en <i>Deep Reinforcement Learning</i> (DRL), alimentado con las variables de acoplamiento habilidad–dificultad y optimizado en sus hiperparámetros, mejora la ganancia de aprendizaje en comprensión lectora frente a una secuenciación sin variables de acoplamiento?</td>
<td><b>Objetivo General:</b><br>Determinar el efecto del Sistema de Tutoría Inteligente Adaptativa basado en DRL (`Dyna-PPO`), alimentado con las variables seleccionadas de acoplamiento habilidad–dificultad y con hiperparámetros optimizados, sobre la ganancia de aprendizaje en comprensión lectora.</td>
<td><b>Hipótesis General:</b><br>El Sistema de Tutoría Inteligente basado en DRL alimentado con las variables seleccionadas de acoplamiento habilidad–dificultad y con hiperparámetros optimizados incrementa significativamente la ganancia de aprendizaje en comprensión lectora respecto a la línea base (p &lt; 0.05).</td>
<td><b>Variable Independiente General:</b><br>• Conjunto depurado de 7 variables de entrada del tutor adaptativo: 2 variables de acoplamiento pregunta–alumno (<code>zpd_hit</code>, <code>abs_gap</code>) y 5 variables de estado del estudiante (<code>theta_1</code>, <code>theta_2</code>, <code>theta_3</code>, <code>u_progress</code>, <code>b_prev_raw</code>).<br><br><b>Variable Dependiente General:</b><br>• Aprendizaje en comprensión lectora del estudiante (competencias literal, inferencial y crítica).</td>
<td>• <b>Ganancia de aprendizaje normalizada de Hake (LG<sub>sim</sub> en 10 semillas):</b><br><code>B4 (PPO)</code>: 0.2393 ± 0.00714<br><code>B5 (Dyna-PPO base)</code>: 0.5043 ± 0.00539<br><b><code>B5 + HPO (Ganador Trial #39)</code>: 0.5329 ± 0.00521</b> (+122.7% sobre B4 y +5.7% sobre B5; p &lt; 0.0001).<br>• <b>Ganancia en validación cruzada 5-Fold (alumnos no vistos):</b> sube de +0.0203 ± 0.0167 a <b>+0.0312 ± 0.0148</b>.<br>• <b>Tasa de eventos de aprendizaje por paso:</b> 0.351 vs. 0.273 de la línea base.</td>
</tr>
<tr>
<td><b>Problema Específico 1:</b><br>¿En qué medida las variables de acoplamiento entre la habilidad del estudiante y la dificultad del texto (<code>zpd_hit</code> y <code>abs_gap</code>) explican que el estudiante logre aprender al responder un texto de comprensión lectora?</td>
<td><b>Objetivo Específico 1:</b><br>Evaluar el aporte de las variables de acoplamiento habilidad–dificultad (<code>zpd_hit</code> y <code>abs_gap</code>) sobre la ocurrencia de aprendizaje en cada texto asignado al estudiante.</td>
<td><b>Hipótesis Específica 1:</b><br>Las variables de acoplamiento <code>zpd_hit</code> y <code>abs_gap</code> constituyen los predictores de mayor aporte sobre la ocurrencia de aprendizaje del estudiante en todos los pliegues de validación, sin presentar multicolinealidad.</td>
<td><b>var(s) indep(s):</b><br>• <b><code>zpd_hit</code></b>: proporción de preguntas del texto dentro de la ventana de aprendizaje (|θ<sub>k</sub> − b<sub>adj</sub>| &lt; 0.5).<br>• <b><code>abs_gap</code></b>: distancia absoluta entre la habilidad del alumno y la dificultad del texto (|θ<sub>k</sub> − b<sub>adj</sub>|).<br><br><b>var depend:</b><br>• Ocurrencia de transición de aprendizaje lector en el paso t (incremento de competencia Δθ<sub>k</sub> &gt; 0).</td>
<td>• <b>Importancia por permutación (ΔAUC en 5 folds):</b> <code>zpd_hit</code> puesto <b>#1</b> (0.0960) y <code>abs_gap</code> puesto <b>#2</b> (0.0142) en los 5/5 folds.<br>• <b>Impacto medio |SHAP|:</b> <code>zpd_hit</code> = 0.0677 (#1), <code>abs_gap</code> = 0.0127 (#2).<br>• <b>Factor de inflación de varianza (VIF):</b> 2.07 y 2.12 (&lt; 2.5).<br>• <b>Discriminación y calibración:</b> ROC-AUC = 0.9479 y Brier Score = 0.0703 ± 0.0028.</td>
</tr>
<tr>
<td><b>Problema Específico 2:</b><br>¿Cómo influyen los niveles de habilidad lectora en las competencias literal, inferencial y crítica (<code>theta_1</code>, <code>theta_2</code>, <code>theta_3</code>), depurados del ruido de medición (<code>sigma_1..3</code>), sobre el progreso de comprensión lectora durante la sesión?</td>
<td><b>Objetivo Específico 2:</b><br>Analizar el efecto de las variables de estado de habilidad por competencia (<code>theta_1</code>, <code>theta_2</code>, <code>theta_3</code>), libres de las variables de ruido (<code>sigma_1..3</code>), sobre la evolución de la comprensión lectora en la sesión.</td>
<td><b>Hipótesis Específica 2:</b><br>La representación del estado mediante las tres competencias lectoras (<code>theta_1</code>, <code>theta_2</code>, <code>theta_3</code>) sin canales de ruido mejora la ganancia de comprensión lectora y requiere menos episodios de interacción que el estado crudo.</td>
<td><b>var(s) indep(s):</b><br>• <b><code>theta_1</code></b>: nivel de competencia lectora literal.<br>• <b><code>theta_2</code></b>: nivel de competencia lectora inferencial.<br>• <b><code>theta_3</code></b>: nivel de competencia lectora crítica.<br><br><b>var depend:</b><br>• Evolución del nivel de comprensión lectora del estudiante a lo largo de la sesión.</td>
<td>• <b>Ganancia media por competencia (Δθ<sub>k</sub>):</b> incremento positivo en las 3 competencias (+0.0203 vs. −0.0095 con ruido).<br>• <b>Importancia |SHAP| de estado:</b> <code>theta_1..3</code> aportan señal real (VIF ∈ [1.01, 1.03]), mientras que <code>sigma_1..3</code> tenía ΔAUC ≈ 0.000.<br>• <b>Episodios requeridos para alcanzar la línea base:</b> 275 episodios vs. 750 (−63.3%).</td>
</tr>
<tr>
<td><b>Problema Específico 3:</b><br>¿De qué manera las variables de contexto de progresión de la lectura (<code>b_prev_raw</code> y <code>u_progress</code>) y el ajuste de hiperparámetros de planificación (`eps_wm`, `lr`, `H`) regulan la continuidad del aprendizaje y evitan rachas de frustración?</td>
<td><b>Objetivo Específico 3:</b><br>Determinar el impacto de las variables de progresión de la sesión (<code>b_prev_raw</code> y <code>u_progress</code>) y de la calibración de hiperparámetros (`eps_wm`, `lr`, `H`) sobre la continuidad del aprendizaje y el control de rachas de error del estudiante.</td>
<td><b>Hipótesis Específica 3:</b><br>La inclusión del contexto de progresión (<code>b_prev_raw</code> y <code>u_progress</code>) junto con un umbral estricto de fidelidad del modelo de mundo (<code>eps_wm = 0.05</code>, <code>H = 5</code>, <code>lr = 1e-4</code>) maximiza la ganancia de aprendizaje manteniendo las rachas de frustración por debajo de 0.02 por paso.</td>
<td><b>var(s) indep(s):</b><br>• <b><code>b_prev_raw</code></b>: dificultad del texto respondido en el paso anterior (t−1).<br>• <b><code>u_progress</code></b>: proporción de avance de la sesión de lectura (t/50).<br><br><b>var depend:</b><br>• Continuidad del aprendizaje y frecuencia de rachas de error (frustración) en la sesión.</td>
<td>• <b>Importancia de hiperparámetros (fANOVA):</b> <code>eps_wm</code> explica el <b>38.89%</b> de la varianza de desempeño, <code>lr</code> el <b>18.26%</b> y <code>H</code> el <b>17.41%</b>.<br>• <b>Tasa de activación del escudo por frustración (racha ≥ 4 errores):</b> 0.015 por paso (≤ 0.02).<br>• <b>Impacto medio |SHAP|:</b> <code>u_progress</code> = 0.0089, <code>b_prev_raw</code> = 0.0065 (VIF = 1.00).</td>
</tr>
</tbody>
</table>