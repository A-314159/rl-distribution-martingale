import utilities.tensorflow_config as cfg
import os
from utilities.scan_tf_graph_readiness import run_scan

if __name__ == "__main__":

    # ----------------------------------------------------
    # Perform some static checks if there are common code mistakes that either:
    #      a) prevent full GPU efficiency.
    #      b) will generate wrong results without raising exception. (such as if condition not performed)
    #
    # Optional, it does not need to be run each time. (Only when the code needs to be checked)
    # ----------------------------------------------------
    scan = False
    if os.environ.get("TRAIN_SKIP_SCAN") == "0" or scan:
        print("🔍 Running TensorFlow static scanner...")
        issues = run_scan(
            paths=["core", "utilities", "optimizers"],
            excludes=["**/__main__.py", "**/scan_tf_graph_readiness.py"],
            scan_graph=True,  # scan for compatibility with graph and autodiff
            scan_broadcast=False,  # scan for potential issues of broadcasting: typically (N,) versus (N,1)
            max_call_depth=5,
            disable_rules=None,  # List of rules to be disabled during scanning, e.g. ["print_in_compiled"],
            return_format="json",
            json_file="scan_report.json",
            extra_decorators=["tf_compile"],
            fail_on_warning=False  # set True to stop on findings
        )
        print("Scanner completed.")

    # ----------------------------------------------------
    # Configure tensorflow for best speed, debugging or reproducibility
    # ----------------------------------------------------

    cfg.configure()

    print("----------------------------------------------------------------------------")
    print("Training will begin.")
    print("In the run console, enter: q to save and quit training, c to show distribution chart")
    print("----------------------------------------------------------------------------")

    # ----------------------------------------------------
    # note: The import below must be done after call to configure()
    # ----------------------------------------------------
    from core.main import main as training_main

    training_main()
