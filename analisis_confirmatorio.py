# -*- coding: utf-8 -*-
"""
Analisis confirmatorio compatible con el esquema real de runs_oracle.json / runs_virgen.json.
Fixes respecto a analisis_confirmatorio.py original:
  1. Lee ganancia anidada en r["final"]["lg"] cuando no existe en el nivel raiz.
  2. Soporta prefijo "PPOlin_" en variant ("PPOlin_FS0_raw13", "PPOlin_FS1_zpd4", "PPOlin_D2_clean5_zpd2").
  3. Soporta fold_seed si existen multiples corridas por (fold, seed) en runs_virgen.json (30 corridas).
  4. Para n > 20 usa permutacion Monte Carlo vectorizada (100,000 iteraciones en <0.1s) en vez de 2^30 = 1.07e9 iteraciones en Python.
"""
import argparse
import itertools
import json
import math
import sys
import numpy as np

CAMPOS = {
    "variante": ["variant", "config", "name", "model", "variante", "feature_set"],
    "particion": ["fold", "partition", "particion", "k"],
    "semilla": ["seed", "semilla", "train_seed"],
    "ganancia": ["lg", "lg_sim", "gain", "lg_val", "val_lg", "lg_mean", "hake"],
}

B_BOOT = 10_000
SEMILLA_BOOT = 2026


def cargar_registros(ruta):
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    if isinstance(datos, dict):
        for v in datos.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        registros = [{"variante": k, **v} for k, v in datos.items() if isinstance(v, dict)]
        if registros:
            return registros
    elif isinstance(datos, list) and datos and isinstance(datos[0], dict):
        return datos
    sys.exit("No se reconoce la estructura del JSON.")


def detectar_campo(registro, concepto):
    claves = {k.lower(): k for k in registro}
    for cand in CAMPOS[concepto]:
        if cand in claves:
            return claves[cand]
    return None


def extraer_lg(r, c_lg):
    if c_lg and c_lg in r:
        return float(r[c_lg])
    if isinstance(r.get("final"), dict):
        for cand in CAMPOS["ganancia"]:
            if cand in r["final"]:
                return float(r["final"][cand])
    return None


def normalizar_variante(v):
    v = str(v)
    return v.replace("PPOlin_", "")


def tabla_lg(registros):
    c_var = detectar_campo(registros[0], "variante")
    c_par = detectar_campo(registros[0], "particion")
    c_sem = detectar_campo(registros[0], "semilla")
    c_lg = detectar_campo(registros[0], "ganancia")
    if not all([c_var, c_par, c_sem]) or extraer_lg(registros[0], c_lg) is None:
        print("Estructura del primer registro:", json.dumps(registros[0], indent=2)[:500])
        sys.exit("No se pudieron mapear variante/particion/semilla/lg.")
    tabla = {}
    for idx, r in enumerate(registros):
        lg = extraer_lg(r, c_lg)
        if lg is None:
            continue
        f_seed = r.get("fold_seed", 0)
        clave = (int(f_seed), int(r[c_par]), int(r[c_sem]))
        tabla.setdefault(normalizar_variante(r[c_var]), {})[clave] = lg
    return tabla


def permutacion_signos(dif, rng):
    n = len(dif)
    obs = abs(dif.mean())
    if n <= 20:
        cuenta = 0
        for signos in itertools.product((1, -1), repeat=n):
            if abs(np.dot(signos, dif) / n) >= obs - 1e-15:
                cuenta += 1
        return cuenta / (2 ** n), 2 / (2 ** n)
    # Monte Carlo vectorizado para n > 20 (evita 2^30 = 1.07e9 en Python)
    M = 100_000
    signos = rng.choice((-1, 1), size=(M, n))
    medias = np.abs((signos @ dif) / n)
    p = (np.sum(medias >= obs - 1e-15) + 1) / (M + 1)
    return float(p), 2 / (2 ** n)


def ic_bootstrap(dif, rng):
    n = len(dif)
    idx = rng.integers(0, n, size=(B_BOOT, n))
    medias = dif[idx].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def contraste(tabla, variante_a, variante_b, rng):
    a, b = tabla.get(variante_a), tabla.get(variante_b)
    if a is None or b is None:
        return None
    claves = sorted(set(a) & set(b))
    if len(claves) < 5:
        return None
    dif = np.array([a[k] - b[k] for k in claves])
    d_bar = dif.mean()
    s_d = dif.std(ddof=1)
    p_perm, p_min = permutacion_signos(dif, rng)
    ic = ic_bootstrap(dif, rng)
    return {
        "contraste": f"{variante_a} - {variante_b}",
        "n": len(claves), "d_bar": d_bar, "ic95": ic,
        "d_z": d_bar / s_d if s_d > 0 else float("inf"),
        "p_perm": p_perm, "p_min": p_min,
        "favor": int((dif > 0).sum()),
    }


def holm(resultados):
    orden = sorted(resultados, key=lambda r: r["p_perm"])
    m = len(orden)
    ajustados, prev = {}, 0.0
    for i, r in enumerate(orden):
        prev = max(prev, min((m - i) * r["p_perm"], 1.0))
        ajustados[r["contraste"]] = prev
    return ajustados


def potencia_aprox(d_z, n):
    from statistics import NormalDist
    nd = NormalDist()
    return 1 - nd.cdf(1.959964 - d_z * math.sqrt(n)) + nd.cdf(-1.959964 - d_z * math.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--confirmatorio", action="store_true")
    args = ap.parse_args()

    tabla = tabla_lg(cargar_registros(args.json))
    print("Variantes encontradas:", ", ".join(sorted(tabla)))
    for v in sorted(tabla):
        print(f"  {v}: {len(tabla[v])} corridas")

    rng = np.random.default_rng(SEMILLA_BOOT)
    pares = [("D2_clean5_zpd2", "FS1_zpd4"),
             ("D2_clean5_zpd2", "FS0_raw13"),
             ("FS1_zpd4", "FS0_raw13")]
    resultados = [c for c in (contraste(tabla, a, b, rng) for a, b in pares) if c]
    if not resultados:
        sys.exit("No se encontro ningun contraste con corridas pareadas suficientes.")

    p_holm = holm(resultados)
    etiqueta = "CONFIRMATORIO (particiones virgenes)" if args.confirmatorio else "EXPLORATORIO"
    print(f"\n=== Contraste pareado [{etiqueta}] ===")
    for r in resultados:
        print(f"{r['contraste']}: n={r['n']}, d_bar={r['d_bar']:+.4f}, "
              f"IC95% [{r['ic95'][0]:+.4f}, {r['ic95'][1]:+.4f}], "
              f"d_z={r['d_z']:.2f}, p_perm={r['p_perm']:.5f} "
              f"(p minimo {r['p_min']:.5f}), {r['favor']}/{r['n']} a favor, "
              f"p_Holm={p_holm[r['contraste']]:.5f}")
        print(f"    potencia alcanzada con este n y d_z: {potencia_aprox(r['d_z'], r['n']):.2f}")

    d_ref = next((r for r in resultados if "FS1_zpd4" in r['contraste']
                  and "D2_clean5_zpd2" in r['contraste']), resultados[0])
    for n_obj in (22, 24, 30):
        print(f"Potencia proyectada con n={n_obj} y d_z={d_ref['d_z']:.2f}: "
              f"{potencia_aprox(d_ref['d_z'], n_obj):.2f}")


if __name__ == "__main__":
    main()
