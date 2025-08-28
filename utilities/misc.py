from tensorflow.keras import mixed_precision
import tensorflow as tf
from utilities.tensorflow_config import tf_compile
import sys, threading, csv, json

# ---------------------------------------
# Normal cdf
# ---------------------------------------

sqrt_2 = 1.4142135623730950488016


@tf_compile
def cdf(x):
    return (tf.math.erf(x / sqrt_2) + 1.) / 2


# ---------------------------------------
# Others
# ---------------------------------------

def cast_all(*variable_list, dtype):
    r = []
    for b in variable_list:
        r.append(tf.cast(b, dtype))
    return tuple(r)


def set_attributes(object, param_dictionary):
    # Override attribute values if in the parameter dictionary
    for k, v in param_dictionary.items():
        if not hasattr(object, k): raise Exception('Unknown attribute: %s', k)
        setattr(object, k, v)


def jsonable(obj):
    if hasattr(obj, "get_config"): return obj.get_config()
    if isinstance(obj, (list, tuple)): return [jsonable(x) for x in obj]
    if isinstance(obj, dict): return {k: jsonable(v) for k, v in obj.items()}
    try:
        _ = json.dumps(obj);
        return obj
    except Exception:
        return str(obj)


# ---------------------------------------
# HotKeys: listening to input in python console
# ---------------------------------------

class HotKeys:
    def __init__(self):
        self.stop = False
        self.show_chart = False
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        try:
            for line in sys.stdin:
                cmd = line.strip().lower()
                if cmd == "q":
                    self.stop = True
                elif cmd == "c":
                    self.show_chart = True
        except Exception:
            pass


def to_csv(csv_path, mode, data):
    with open(csv_path, mode, newline="") as f:
        csv.writer(f).writerow(data)
