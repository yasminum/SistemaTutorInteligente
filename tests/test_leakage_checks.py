# tests/test_leakage_checks.py - Suite automatizada de pruebas anti-fuga (L1 a L5) y reproducibilidad
import unittest
from src.leakage_checks import run_all_checks
from src.config import MASTER_SEED, EVAL_SEEDS, set_global_seed


class TestLeakageAndReproducibility(unittest.TestCase):
    def test_l1_to_l5_zero_leakage(self):
        rows = run_all_checks()
        self.assertTrue(all(r["resultado"] == "PASS" for r in rows), f"Fallo en pruebas anti-fuga: {rows}")

    def test_seeds_configured(self):
        set_global_seed(MASTER_SEED)
        self.assertEqual(MASTER_SEED, 42)
        self.assertEqual(EVAL_SEEDS, list(range(10)))


if __name__ == "__main__":
    unittest.main()
