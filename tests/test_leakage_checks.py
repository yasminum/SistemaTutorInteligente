# tests/test_leakage_checks.py - Suite automatizada de pruebas anti-fuga (L1 a L5) y reproducibilidad
import unittest
from src.leakage_checks import run_all_checks
from src.experiment import CFG
from src.simulator import ItemBank


class TestLeakageAndReproducibility(unittest.TestCase):
    def test_l1_to_l5_zero_leakage(self):
        rows = run_all_checks()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["resultado"] == "PASS" for r in rows), f"Fallo en pruebas anti-fuga: {rows}")

    def test_cfg_and_item_bank_determinism(self):
        self.assertEqual(CFG["n_students"], 100)
        self.assertEqual(CFG["k_folds"], 5)
        self.assertEqual(CFG["fold_seed"], 2026)
        self.assertEqual(CFG["bank_seed"], 2026)
        self.assertEqual(CFG["seeds"], [0, 1])
        b1 = ItemBank(CFG["bank_seed"]).sha256()
        b2 = ItemBank(CFG["bank_seed"]).sha256()
        self.assertEqual(b1, b2)
        self.assertTrue(b1.startswith("07ed76f0bee0a21d"))


if __name__ == "__main__":
    unittest.main()
