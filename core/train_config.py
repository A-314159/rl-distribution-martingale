import tensorflow as tf
import utilities.tensorflow_config as tf_cfg


class TrainConfig:
    def __init__(
            self,
            optimizer: str = "adam",  # adam|sgd|rmsprop|gn|lm|lbfgs
            batch_size: int = 2 ** 10,
            full_batch: bool = False,
            max_epochs: int = 100000,
            max_time_sec: int = 7200,
            loss_tol_sqrt: float = 1e-4,
            hidden: tuple = (32, 32),
            activation: str = "tanh",
            model_dir: str = "no name",
            log_csv: str = "training_log.csv",
            eval_pairs: list = None,
            mc_paths: int = 20000,
            # gauss-newton / levenberg–marquardt
            gn_iters: int = 10,
            gn_damping: float = 1e-2,
            gn_verbose: bool = False,

            # limited memory BFGS
            lbfgs_iters: int = 50,
            lbfgs_tol: float = 1e-6,
            lbfgs_verbose: bool = False,

            # gradient-descent style optimizers
            lr: float = 1e-3,

            # blending between hint and bellman
            anneal_beta_period: int = 1000000,

            # distribution chart
            show_chart: bool = False,  # False: output to file
            chart_pdf: str = "distribution.pdf",

            # cast_up to FP64 for sensitive calculations
            cast_64: bool = True,

            # LBFGS setting
            memory=20,
            line_search="nonmonotone_armijo",
            y_sign_mode="normal",
            memory_update="fifo",
            armijo_c1=1e-4, armijo_window=5, backtrack_factor=0.5,
            max_evals_per_iter=20, wolfe_c2=0.9,
            powell_damping=False,
            init_scaling="bb", init_gamma=1.0,
            eps_curv=1e-12,
            dtype=tf.float32,  # model dtype (default fp32)
            opt_dtype=None,  # accumulation dtype (None → choose by model dtype)
            debug_checks=True,  # runtime finite checks
            # ---- Armijo engine knobs (defaults preserve current behavior) ----
            armijo_use_cubic=True,  # False → geometric; True → safeguarded cubic
            armijo_use_wolfe=False,  # False → pure Armijo; True → light curvature check
            armijo_step_max=0.0  # 0.0 → no trust-region cap on ||d||
    ):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.full_batch = full_batch
        self.max_epochs = max_epochs
        self.max_time_sec = max_time_sec
        self.loss_tol_sqrt = loss_tol_sqrt
        self.hidden = hidden
        self.activation = activation
        self.model_dir = model_dir
        self.log_csv = log_csv
        self.eval_pairs = eval_pairs
        self.mc_paths = mc_paths
        self.gn_iters = gn_iters
        self.gn_damping = gn_damping
        self.gn_verbose = gn_verbose
        self.lbfgs_iters = lbfgs_iters
        self.lbfgs_tol = lbfgs_tol
        self.lbfgs_verbose = lbfgs_verbose
        self.lr = lr
        self.anneal_beta_period = anneal_beta_period
        self.show_chart = show_chart
        self.chart_pdf = chart_pdf
        self.cast_64 = cast_64

        tf_cfg.SENSITIVE_CALC = tf.float64 if cast_64 else tf.float32

        self.memory = 20
        self.line_search = "armijo"  # "armijo" or "hager_zhang"
        self.y_sign_mode = "auto"  # use this or Powell damping
        self.memory_update = "fifo"
        self.armijo_c1 = 1e-4
        self.armijo_window = 1          # 1 for monotone armijo, >1 otherwise
        self.backtrack_factor = 0.5     # when using geometric backtrack instead of cubic
        self.max_evals_per_iter = 20
        self.wolfe_c2 = 0.9
        self.powell_damping = False
        self.init_scaling = "bb"        # standard choice
        self.init_gamma = 1.0
        self.eps_curv = 1e-12
        self.dtype = tf_cfg.HIGH  # model dtype (default fp32)
        self.opt_dtype = tf_cfg.SENSITIVE_CALC  # accumulation dtype (None → choose by model dtype)
        self.debug_checks = True  # Runtime finite/Nan checks
        # ---- Armijo engine knobs (defaults preserve current behavior) ----
        self.armijo_use_cubic = True  # False → geometric; True → safeguarded cubic
        self.armijo_use_wolfe = False  # False → pure Armijo; True → light curvature check
        self.armijo_step_max = 0.0  # 0.0 → no trust-region cap on ||d||

    def get_config(self) -> dict:
        """Return configuration as a plain dictionary (safe for JSON)."""
        return self.__dict__
        """
        return {
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "full_batch": self.full_batch,
            "max_epochs": self.max_epochs,
            "max_time_sec": self.max_time_sec,
            "loss_tol_sqrt": self.loss_tol_sqrt,
            "hidden": tuple(self.hidden),
            "activation": self.activation,
            "model_dir": self.model_dir,
            "log_csv": self.log_csv,
            "eval_pairs": self.eval_pairs if self.eval_pairs is not None else [],
            "mc_paths": self.mc_paths,
            "gn_iters": self.gn_iters,
            "gn_damping": self.gn_damping,
            "gn_verbose": self.gn_verbose,
            "lbfgs_iters": self.lbfgs_iters,
            "lbfgs_tol": self.lbfgs_tol,
            "lbfgs_verbose": self.lbfgs_verbose,
            "lr": self.lr,
            "anneal_beta_period": self.anneal_beta_period,
            "show_chart": self.show_chart,
            "chart_pdf": self.chart_pdf,
            "cast_64": self.cast_64,
        }
        """

    def __repr__(self):
        return f"TrainConfig({self.get_config()})"
