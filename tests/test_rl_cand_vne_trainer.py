import unittest
import torch
from algorithms.rl_cand_vne.trainer import Trainer
from algorithms.rl_cand_vne.policy_network import PolicyNetwork


class TestTrainer(unittest.TestCase):
    def _fake_episode(self, pn, reward, committed_snode_indices=None):
        vnode_feats = torch.randn(2, 5)
        vn_adj = torch.eye(2)
        dip = [
            [(torch.randn(3, 5), torch.eye(3), torch.tensor([50.0, 50.0, 50.0]))],
            [(torch.randn(3, 5), torch.eye(3), torch.tensor([50.0, 50.0, 50.0]))],
        ]
        out = pn(vnode_feats, vn_adj, dip, cpu_demands=[1.0, 1.0], sample=True)
        return {
            "domain_log_probs": out["domain_log_probs"],
            "snode_log_probs_per_vnode": out["snode_log_probs_per_vnode"],
            "reward": reward,
            "committed_snode_indices": committed_snode_indices,
            "success": committed_snode_indices is not None,
        }

    def test_baseline_tracks_recent_rewards(self):
        pn = PolicyNetwork(hidden=16, K=2)
        trainer = Trainer(pn, lr=1e-3, lam_sup=1.0, baseline_window=3)
        for r in [1.0, 2.0, 3.0]:
            ep = self._fake_episode(pn, r, committed_snode_indices=None)
            trainer.record(**ep)
        self.assertAlmostEqual(trainer.baseline(), 2.0, places=5)

    def test_update_produces_finite_loss_and_grads(self):
        torch.manual_seed(0)
        pn = PolicyNetwork(hidden=16, K=2)
        trainer = Trainer(pn, lr=1e-3, lam_sup=1.0, baseline_window=100)
        ep1 = self._fake_episode(pn, reward=0.8, committed_snode_indices=[0, 1])
        ep2 = self._fake_episode(pn, reward=-2.0, committed_snode_indices=None)
        trainer.record(**ep1)
        trainer.record(**ep2)

        # Zero grads explicitly to detect that update actually writes grads.
        for p in pn.parameters():
            p.grad = None
        metrics = trainer.update()

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss_total"])))
        any_grad = any(
            (p.grad is not None and p.grad.abs().sum().item() > 0)
            for p in pn.parameters()
        )
        self.assertTrue(any_grad, "at least one parameter should have nonzero grad")
        self.assertEqual(len(trainer.buffer), 0)


if __name__ == "__main__":
    unittest.main()
