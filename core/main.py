import secrets
from pathlib import Path
from core.universe import UniverseBS
from core.sampling import SamplerConfig
from core.train_config import TrainConfig
from core.trainer import DistributionTrainer
from core.distribution_by_mc import make_and_save_chart
from core.actors import BSDeltaHedge


def main():
    folder = secrets.token_urlsafe(16)  # this is to be used to assign a folder name as an experiment token
    folder = "F BS"
    # train config must be _done first to set the data type for sensitive calculation first (no great)
    c = TrainConfig(hidden=(8, 8, 8), optimizer="lbfgs_graph", lr=1e-3,
                    batch_size=512, max_epochs=10000, max_time_sec=3600, loss_tol_sqrt=1e-4,
                    model_dir='experiment/' + folder, eval_pairs=[(0, 0.0), (20, 0.0), (40, 0.0)],
                    mc_paths=10000, full_batch=True, show_chart=True, cast_64=True)
    u = UniverseBS(sigma=0.3, T=0.5, P=60, K=1.0)
    s = SamplerConfig(N=2 ** 14, x0=0.0, a=0.5, b=4.0, c=1 / 52, r0=0.02, r1=0.002)
    a = BSDeltaHedge()
    trainer = DistributionTrainer(u, s, c, a)
    trainer.build()
    model = trainer.train()

    # todo: we will check the following function later
    # make_and_save_chart(model, u, cfg, Path(cfg.model_dir) / cfg.chart_pdf)

    print("Done. Model:", Path(c.model_dir) / "model.keras")
