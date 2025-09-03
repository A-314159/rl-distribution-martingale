#!/usr/bin/env python3
"""
TensorFlow GPU-readiness + Autodiff static scanner
- severity aware
- call chains
- per-file summary
- optional broadcasting heuristics
- max call depth
- rule toggles (disable or enable-only)
- presets (minimal / default / strict)
- decorator @broadcast_ok to skip broadcasting checks per function (and its callees)
- decorator @no_grad_ok to skip autodiff checks per function (and its callees)
- undecorated or @requires_grad => autodiff checks ON

Severities:
  ✖✖✖  CRITICAL  (likely silent wrong semantics if it "works")
  ✖✖    ERROR     (likely to raise in graph/GPU mode or kill grads)
  ✖      WARN     (likely to run but inefficient / suboptimal)
  •       REMARK   (informational; e.g., tf.stop_gradient)
  ≈       BCAST    (potential broadcasting/rank-drop; optional scan)

Broadcasting heuristics (optional or per-function skip via @broadcast_ok):
  - tf.reduce_* without keepdims=True
  - tf.squeeze without axis=
  - tf.reshape(..., [-1])
  - Integer indexing (x[:, 0], x[0], …) that drops an axis

CLI examples:
  python utilities/scan_tf_graph_readiness.py core utilities optimizers \
    --exclude "**/__main__.py" \
    --exclude "**/scan_tf_graph_readiness.py" \
    --preset default \
    --json --json-file scan_report.json
"""

from __future__ import annotations
import ast, sys, pathlib, fnmatch, argparse, json, os
from typing import Dict, Set, List, Optional, Union, cast, Tuple
from collections import defaultdict

# -------------------- Severities & symbols ----------------------------

SEV_CRIT = "CRITICAL"  # ✖✖✖
SEV_ERR = "ERROR"  # ✖✖
SEV_WARN = "WARN"  # ✖
SEV_NOTE = "REMARK"  # •
SEV_BCST = "BCAST"  # ≈

SEV_ORDER = [SEV_CRIT, SEV_ERR, SEV_WARN, SEV_NOTE, SEV_BCST]
SEV_SYM = {SEV_CRIT: "✖✖✖", SEV_ERR: "✖✖", SEV_WARN: "✖", SEV_NOTE: "•", SEV_BCST: "≈"}


def sev_symbol(sev: str) -> str:
    return SEV_SYM.get(sev, "?")


# -------------------- ANSI colors ----------------------------

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "", "dumb")


COLORS = {
    SEV_CRIT: "\033[1;31m",  # bright red
    SEV_ERR: "\033[31m",  # red
    SEV_WARN: "\033[33m",  # yellow
    SEV_NOTE: "\033[36m",  # cyan
    SEV_BCST: "\033[35m",  # magenta
    "path": "\033[90m",  # gray
    "reset": "\033[0m",
}


def colorize(s: str, sev: Optional[str] = None, kind: Optional[str] = None, enable: bool = True) -> str:
    if not enable:
        return s
    if sev in COLORS:
        return f"{COLORS[sev]}{s}{COLORS['reset']}"
    if kind in COLORS:
        return f"{COLORS[kind]}{s}{COLORS['reset']}"
    return s


# -------------------- Issue model ----------------------------

class Issue:
    def __init__(self, path: str, line: int, col: int, sev: str, msg: str,
                 call_chain: Optional[List[str]] = None,
                 kind: str = "graph",
                 rule: str = "unknown"):
        self.path, self.line, self.col, self.sev, self.msg = path, line, col, sev, msg
        self.call_chain = list(call_chain or [])
        self.kind = kind  # "graph" | "broadcast"
        self.rule = rule  # rule identifier

    def __str__(self):
        chain = f"  (via: {' → '.join(self.call_chain)})" if self.call_chain else ""
        return f"{self.path}:{self.line}:{self.col}  {sev_symbol(self.sev)} {self.msg}{chain}"

    def as_dict(self):
        return {
            "path": self.path,
            "line": self.line,
            "col": self.col,
            "severity": self.sev,
            "message": self.msg,
            "call_chain": self.call_chain,
            "kind": self.kind,
            "rule": self.rule,
        }


# -------------------- Rule catalogs -------------------------

# Rule IDs for toggles
# Graph rules
R_PRINT = "print_in_compiled"
R_BUILTIN_ON_TENSOR = "builtin_on_tensor"
R_LEN_ON_TENSOR = "len_on_tensor"
R_ATTR_NUMPY = "attr_numpy_item_tolist"
R_NUMPY_CONVERT = "numpy_convert"
R_BOOL_AND_OR = "python_and_or_on_tensors"
R_TRY_EXCEPT = "try_except_in_compiled"
R_FOR_ITER = "range_enumerate_zip_loop"
R_WHILE_LOOP = "while_loop_compiled"

# Autodiff rules
R_TF_PYFUNC = "tf_py_or_numpy_function"
R_TF_DISCRETE = "tf_discrete_op_on_grad_path"
R_TF_CAST_INT = "tf_cast_to_nonfloat"
R_STOP_GRAD = "tf_stop_gradient_remark"

# Broadcast rules
R_REDUCE_KEEPDIMS = "reduce_without_keepdims"
R_SQUEEZE_NO_AXIS = "squeeze_without_axis"
R_RESHAPE_FLATTEN = "reshape_flatten_to_rank1"
R_INDEX_DROPS_AXIS = "integer_index_drops_axis"

# GPU graph-compat specifics
BAD_ATTR_CALLS = {"numpy", "item", "tolist"}  # -> ERROR
BAD_BUILTINS = {"float", "int", "bool", "list"}  # -> ERROR when on tensors
LEN_BUILTIN = {"len"}  # -> ERROR when on tensors (dynamic) / risky
BAD_PRINT = {"print"}  # -> WARN (use tf.print)
BAD_NP_FUNCS = {"array", "asarray", "asanyarray", "ascontiguousarray"}  # -> WARN (host copies)
BAD_BOOL_OPS = {"And", "Or"}  # -> CRITICAL (Python 'and/or' on tensors)
BAD_TRY = True  # -> ERROR (unsupported in AutoGraph)
BAD_FOR_ITERS = {"range", "enumerate", "zip"}  # -> WARN (prefer tf.range / vectorize)

# Autodiff: names under tf.<name>
GRAD_BLOCKING_TF_FUNCS_ERR = {"py_function", "numpy_function"}  # kill grads
GRAD_DISCRETE_TF_FUNCS_CRIT = {
    "argmax", "argmin", "round", "floor", "ceil", "rint", "sign",
    "unique", "argsort", "sort",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "logical_and", "logical_or", "logical_not",
}
STOP_GRAD_NAME = "stop_gradient"  # -> REMARK always

NON_FLOAT_DTYPES = {
    "tf.int8", "tf.int16", "tf.int32", "tf.int64", "tf.uint8", "tf.uint16", "tf.uint32", "tf.uint64",
    "tf.bool", "tf.string"
}

# Treat as compiled if decorated with these (extend with --decorator)
DEFAULT_COMPILED_DECORATORS = {
    "tf.function", "tf_function", "tf_function_jit",
    "tf.compile",
    "tf_compile",
}

# Autodiff decorators
GRAD_REQUIRED_DECOS = {"requires_grad"}  # (policy: undecorated => strict anyway)
GRAD_EXEMPT_DECOS = {"no_grad_ok"}

# Broadcast decorators
BCAST_EXEMPT_DECOS = {"broadcast_ok"}  # skip broadcasting checks inside this function and its callees


# --------------------- Scanner --------------------------------

class FileScanner(ast.NodeVisitor):
    def __init__(self, path: pathlib.Path,
                 compiled_decorators: Set[str],
                 symtab: Dict[str, ast.AST],
                 numpy_aliases: Set[str],
                 scan_graph: bool,
                 scan_broadcast: bool,
                 max_call_depth: Optional[int],
                 disabled_rules: Set[str],
                 enable_only_rules: Optional[Set[str]]):
        self.path = path
        self.compiled_decorators = compiled_decorators
        self.symtab = symtab
        self.numpy_aliases = numpy_aliases
        self.scan_graph = scan_graph
        self.scan_broadcast_global = scan_broadcast
        self.max_call_depth = max_call_depth
        self.disabled_rules = disabled_rules
        self.enable_only_rules = enable_only_rules

        self.in_compiled_stack: List[bool] = [False]  # GPU graph checks active?
        self.grad_mode_stack: List[str] = ["strict"]  # "strict" | "skip"
        self.bcast_mode_stack: List[str] = ["on"]  # "on" | "skip" (per-function broadcast checks)
        self.current_function: List[str] = []
        self._call_stack: List[str] = []
        self.readable_stack: List[str] = []  # for call-chain
        self.issues: List[Issue] = []

    # ---------- helpers

    def _rule_enabled(self, rule: str) -> bool:
        if self.enable_only_rules is not None:
            return rule in self.enable_only_rules
        return rule not in self.disabled_rules

    def _emit(self, node: ast.AST, sev: str, msg: str, rule: str, kind: str = "graph"):
        if not self._rule_enabled(rule):
            return
        self.issues.append(
            Issue(
                str(self.path),
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0),
                sev,
                msg,
                call_chain=list(self.readable_stack),
                kind=kind,
                rule=rule,
            )
        )

    def _decorator_name(self, dec: Union[ast.AST, ast.expr]) -> Optional[str]:
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Attribute):
            parts = []
            cur: Union[ast.Attribute, ast.Name, ast.expr] = dec
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                parts.reverse()
                return ".".join(parts)
        if isinstance(dec, ast.Call):
            return self._decorator_name(cast(ast.AST, dec.func))
        return None

    def _is_compiled_decorator(self, dec: Union[ast.AST, ast.expr]) -> bool:
        name = self._decorator_name(dec)
        if not name:
            return False
        normalized = name.replace("tensorflow.", "tf.")
        return (normalized in self.compiled_decorators) or (normalized.split(".")[-1] in self.compiled_decorators)

    def _has_decorator(self, node: ast.FunctionDef, names: Set[str]) -> bool:
        for d in node.decorator_list:
            nm = self._decorator_name(d) or ""
            if nm in names or nm.split(".")[-1] in names:
                return True
        return False

    def _dtype_is_nonfloat(self, expr: ast.AST) -> Optional[str]:
        # Resolve "tf.int32" etc.
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "tf":
            dtype_name = f"tf.{expr.attr}"
            if dtype_name in NON_FLOAT_DTYPES:
                return dtype_name
        return None

    def _current_depth(self) -> int:
        # depth 0 = top-level function; deeper for callees
        return max(0, len(self.readable_stack) - 1)

    # ---------- visits

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # GPU-graph (compiled) context
        is_compiled_here = any(self._is_compiled_decorator(d) for d in node.decorator_list)
        effective_compiled = is_compiled_here or self.in_compiled_stack[-1]

        # Autodiff policy (per requirement)
        if self._has_decorator(node, GRAD_EXEMPT_DECOS):
            grad_mode = "skip"
        else:
            grad_mode = "strict"  # requires_grad OR undecorated -> strict

        # Broadcast policy: explicit @broadcast_ok => skip; else inherit "on"/"skip" (but also respect global toggle)
        if self._has_decorator(node, BCAST_EXEMPT_DECOS):
            bcast_mode = "skip"
        else:
            bcast_mode = self.bcast_mode_stack[-1]  # inherit

        self.in_compiled_stack.append(effective_compiled)
        self.grad_mode_stack.append(grad_mode)
        self.bcast_mode_stack.append(bcast_mode)

        self.current_function.append(node.name)
        self.readable_stack.append(node.name)

        self.generic_visit(node)

        self.readable_stack.pop()
        self.current_function.pop()
        self.bcast_mode_stack.pop()
        self.grad_mode_stack.pop()
        self.in_compiled_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Try(self, node: ast.Try):
        if self.in_compiled_stack[-1] and BAD_TRY and self._rule_enabled(R_TRY_EXCEPT):
            # graph rule (only if graph scanning enabled globally)
            if self.scan_graph:
                self._emit(node, SEV_ERR, "try/except inside compiled function is unsupported by AutoGraph",
                           R_TRY_EXCEPT)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        if self.in_compiled_stack[-1] and type(node.op).__name__ in BAD_BOOL_OPS and self._rule_enabled(R_BOOL_AND_OR):
            if self.scan_graph:
                self._emit(node, SEV_CRIT,
                           "Python 'and/or' on tensors in compiled code; use tf.logical_and / tf.logical_or",
                           R_BOOL_AND_OR)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self.in_compiled_stack[-1] and self.scan_graph:
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                fname = node.iter.func.id
                if fname in BAD_FOR_ITERS and self._rule_enabled(R_FOR_ITER):
                    self._emit(node, SEV_WARN,
                               f"{fname}(...) loop in compiled function; prefer tf.range or vectorize / tf.while_loop",
                               R_FOR_ITER)
        self.generic_visit(node)

    def _is_scalar_index(self, slc: ast.AST) -> bool:
        """
        Returns True for integer indexing that drops a dimension:
          x[0], x[:, 0], x[..., 0]
        Handles py>=3.9 (expr) and legacy py<3.9 (ast.Index).
        """
        # py>=3.9: slice is an expr, e.g. Constant, Tuple, Slice
        if isinstance(slc, ast.Constant) and isinstance(slc.value, int):
            return True
        if isinstance(slc, ast.Tuple):
            # any component is an int constant (e.g., x[:, 0] becomes Tuple of Slice, Constant)
            return any(isinstance(elt, ast.Constant) and isinstance(elt.value, int) for elt in slc.elts)
        if isinstance(slc, ast.Slice):
            return False  # pure slices don't drop rank

        # py<3.9 compatibility: ast.Index may exist
        Index = getattr(ast, "Index", None)
        if Index is not None and isinstance(slc, Index):
            val = slc.value
            if isinstance(val, ast.Constant) and isinstance(val.value, int):
                return True

        return False

    def visit_While(self, node: ast.While):
        if self.in_compiled_stack[-1] and self.scan_graph and self._rule_enabled(R_WHILE_LOOP):
            self._emit(node, SEV_WARN,
                       "while loop in compiled function: ensure tensor-based condition (AutoGraph -> tf.while_loop)",
                       R_WHILE_LOOP)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if self.scan_broadcast_global and self.bcast_mode_stack[-1] == "on" and self._rule_enabled(R_INDEX_DROPS_AXIS):
            if self._is_scalar_index(node.slice):  # <-- no union-type warning now
                self._emit(
                    node,
                    SEV_BCST,
                    "Integer indexing drops an axis: x[:, 0] → (N,) not (N,1). Use slices (0:1) or tf.expand_dims.",
                    R_INDEX_DROPS_AXIS,
                    kind="broadcast",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        in_compiled = self.in_compiled_stack[-1]
        grad_mode = self.grad_mode_stack[-1]  # "strict" | "skip"
        bcast_on = (self.scan_broadcast_global and self.bcast_mode_stack[-1] == "on")

        # --- GPU graph-compat checks ---
        if self.scan_graph and in_compiled:
            # print(...)
            if isinstance(node.func, ast.Name) and node.func.id in BAD_PRINT and self._rule_enabled(R_PRINT):
                self._emit(node, SEV_WARN, "print() in compiled function; use tf.print (sparingly)", R_PRINT)

            # float/int/bool/list(...)
            if isinstance(node.func, ast.Name) and node.func.id in BAD_BUILTINS and self._rule_enabled(
                    R_BUILTIN_ON_TENSOR):
                self._emit(node, SEV_ERR,
                           f"{node.func.id}() on tensors inside compiled function; use tensor ops (tf.cast, tf.shape, etc.)",
                           R_BUILTIN_ON_TENSOR)

            # len(...)
            if isinstance(node.func, ast.Name) and node.func.id in LEN_BUILTIN and self._rule_enabled(R_LEN_ON_TENSOR):
                self._emit(node, SEV_ERR,
                           "len(tensor) in compiled function risks trace-time const or error; use tf.shape(tensor)[i]",
                           R_LEN_ON_TENSOR)

            # x.numpy() / x.item() / x.tolist()
            if isinstance(node.func, ast.Attribute) and node.func.attr in BAD_ATTR_CALLS and self._rule_enabled(
                    R_ATTR_NUMPY):
                self._emit(node, SEV_ERR,
                           f".{node.func.attr}() pulls tensors to host in compiled code; breaks GPU graph execution",
                           R_ATTR_NUMPY)

            # numpy conversions
            if isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base,
                              ast.Name) and base.id in self.numpy_aliases and node.func.attr in BAD_NP_FUNCS and self._rule_enabled(
                        R_NUMPY_CONVERT):
                    self._emit(node, SEV_WARN,
                               f"np.{node.func.attr}(...) in compiled function; avoids fusion and triggers host copies",
                               R_NUMPY_CONVERT)
            elif isinstance(node.func, ast.Name):
                if node.func.id in BAD_NP_FUNCS and "numpy" in self.numpy_aliases and self._rule_enabled(
                        R_NUMPY_CONVERT):
                    self._emit(node, SEV_WARN,
                               f"numpy {node.func.id}(...) in compiled function; avoid NumPy conversions in hot path",
                               R_NUMPY_CONVERT)

        # --- Autodiff checks (skip if @no_grad_ok) ---
        if self.scan_graph and grad_mode == "strict":
            # tf.<name>(...)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value,
                                                                   ast.Name) and node.func.value.id == "tf":
                tfname = node.func.attr

                if tfname in GRAD_BLOCKING_TF_FUNCS_ERR and self._rule_enabled(R_TF_PYFUNC):
                    self._emit(node, SEV_ERR, f"tf.{tfname} kills gradients; avoid in grad-required code", R_TF_PYFUNC)

                elif tfname in GRAD_DISCRETE_TF_FUNCS_CRIT and self._rule_enabled(R_TF_DISCRETE):
                    self._emit(node, SEV_CRIT,
                               f"tf.{tfname} is discrete/non-differentiable; ensure its output is not on the gradient path",
                               R_TF_DISCRETE)

                elif tfname == "cast" and self._rule_enabled(R_TF_CAST_INT):
                    # check dtype kw / positional
                    cast_dtype = None
                    for kw in node.keywords:
                        if kw.arg == "dtype":
                            cast_dtype = self._dtype_is_nonfloat(cast(ast.AST, kw.value))
                            break
                    if cast_dtype is None and len(node.args) >= 2:
                        cast_dtype = self._dtype_is_nonfloat(cast(ast.AST, node.args[1]))
                    if cast_dtype is not None:
                        self._emit(node, SEV_ERR,
                                   f"tf.cast(..., {cast_dtype}) stops gradients (non-float) in grad-required code",
                                   R_TF_CAST_INT)

                elif tfname == STOP_GRAD_NAME and self._rule_enabled(R_STOP_GRAD):
                    self._emit(node, SEV_NOTE, "tf.stop_gradient present (informational). Ensure this is intentional.",
                               R_STOP_GRAD)

        # --- Broadcasting heuristics (optional & per-function skippable) ---
        if bcast_on and isinstance(node.func, ast.Attribute) and isinstance(node.func.value,
                                                                            ast.Name) and node.func.value.id == "tf":
            tfname = node.func.attr

            # 1) Reductions without keepdims=True
            if tfname.startswith("reduce_") and self._rule_enabled(R_REDUCE_KEEPDIMS):
                has_keepdims = any(kw.arg == "keepdims" for kw in node.keywords)
                keepdims_is_true = any(
                    kw.arg == "keepdims" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in
                    node.keywords)
                if not has_keepdims or not keepdims_is_true:
                    self._emit(node, SEV_BCST,
                               f"tf.{tfname} without keepdims=True may drop axes (N,)->(N,), consider keepdims=True",
                               R_REDUCE_KEEPDIMS, kind="broadcast")

            # 2) squeeze without explicit axis
            elif tfname == "squeeze" and self._rule_enabled(R_SQUEEZE_NO_AXIS):
                has_axis = any(kw.arg == "axis" for kw in node.keywords) or (len(node.args) >= 2)
                if not has_axis:
                    self._emit(node, SEV_BCST,
                               "tf.squeeze without axis removes all singleton dims; can turn (N,1) into (N,)",
                               R_SQUEEZE_NO_AXIS, kind="broadcast")

            # 3) reshape(..., [-1]) forces rank-1
            elif tfname == "reshape" and self._rule_enabled(R_RESHAPE_FLATTEN):
                shape_arg = None
                if len(node.args) >= 2:
                    shape_arg = node.args[1]
                else:
                    for kw in node.keywords:
                        if kw.arg == "shape":
                            shape_arg = kw.value
                            break
                if isinstance(shape_arg, (ast.List, ast.Tuple)) and len(shape_arg.elts) == 1:
                    self._emit(node, SEV_BCST,
                               "tf.reshape(..., [-1]) produces rank-1 (N,); if you expect a column vector, consider [-1, 1]",
                               R_RESHAPE_FLATTEN, kind="broadcast")

        # --- Recursive scan of user-defined callees as compiled & same modes ---
        callee_name = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id

        if (callee_name and callee_name in self.symtab and callee_name not in self._call_stack):
            # Max depth guard
            if self.max_call_depth is not None and self._current_depth() + 1 > self.max_call_depth:
                return

            self._call_stack.append(callee_name)
            callee_def = cast(ast.FunctionDef, self.symtab[callee_name])

            # Push same contexts for callee
            self.in_compiled_stack.append(in_compiled)

            # Respect @no_grad_ok on the callee; otherwise inherit grad_mode
            if self._has_decorator(callee_def, GRAD_EXEMPT_DECOS):
                self.grad_mode_stack.append("skip")
            else:
                self.grad_mode_stack.append(grad_mode)

            # Respect @broadcast_ok on the callee; otherwise inherit current bcast mode
            if self._has_decorator(callee_def, BCAST_EXEMPT_DECOS):
                self.bcast_mode_stack.append("skip")
            else:
                self.bcast_mode_stack.append(self.bcast_mode_stack[-1])

            self.current_function.append(callee_name)
            self.readable_stack.append(callee_name)

            self.generic_visit(callee_def)

            self.readable_stack.pop()
            self.current_function.pop()
            self.bcast_mode_stack.pop()
            self.grad_mode_stack.pop()
            self.in_compiled_stack.pop()
            self._call_stack.pop()

        self.generic_visit(node)


# --------------------- Utilities -----------------------------------------

def build_symbol_table(tree: ast.AST) -> Dict[str, ast.FunctionDef]:
    sym: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            sym[node.name] = node
    return sym


def collect_numpy_aliases(tree: ast.AST) -> Set[str]:
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy":
                    aliases.add(a.asname or "numpy")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy":
                aliases.add("numpy")
    return aliases


def scan_file(path: pathlib.Path, compiled_decorators: Set[str],
              scan_graph: bool, scan_broadcast: bool,
              max_call_depth: Optional[int],
              disabled_rules: Set[str],
              enable_only_rules: Optional[Set[str]]) -> List[Issue]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [Issue(str(path), 1, 0, SEV_ERR, f"Could not read file: {e}", [], "graph", rule="read_error")]
    try:
        tree = ast.parse(text, filename=str(path))
    except Exception as e:
        return [Issue(str(path), 1, 0, SEV_ERR, f"Parse error: {e}", [], "graph", rule="parse_error")]

    symtab = build_symbol_table(tree)
    np_aliases = collect_numpy_aliases(tree)
    scanner = FileScanner(path, compiled_decorators, symtab, np_aliases,
                          scan_graph=scan_graph, scan_broadcast=scan_broadcast,
                          max_call_depth=max_call_depth,
                          disabled_rules=disabled_rules,
                          enable_only_rules=enable_only_rules)
    scanner.visit(tree)
    return scanner.issues


# --------------------- Reporting -----------------------------------------

def summarize_issues(issues: List[Issue]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    total = {sev: 0 for sev in SEV_ORDER}
    per_file: Dict[str, Dict[str, int]] = defaultdict(lambda: {sev: 0 for sev in SEV_ORDER})
    for it in issues:
        total[it.sev] = total.get(it.sev, 0) + 1
        per_file[it.path][it.sev] = per_file[it.path].get(it.sev, 0) + 1
    return total, per_file


def split_by_kind(issues: List[Issue]) -> Tuple[List[Issue], List[Issue]]:
    graph = [i for i in issues if i.kind == "graph"]
    bcast = [i for i in issues if i.kind == "broadcast"]
    return graph, bcast


def print_text_report(issues: List[Issue], use_color: bool):
    graph_issues, bcast_issues = split_by_kind(issues)

    def _print_section(title: str, items: List[Issue]):
        if not items:
            print(colorize(f"{title}: none", kind="path", enable=use_color))
            return
        items_sorted = sorted(items, key=lambda x: (
        x.path, x.line, x.col, SEV_ORDER.index(x.sev) if x.sev in SEV_ORDER else 999))
        by_file: Dict[str, List[Issue]] = defaultdict(list)
        for it in items_sorted:
            by_file[it.path].append(it)

        print(colorize(f"\n{title}", kind="path", enable=use_color))
        total, per_file = summarize_issues(items_sorted)
        for fpath, lst in by_file.items():
            print(colorize(f"\n{fpath}", kind="path", enable=use_color))
            for it in lst:
                sev = colorize(sev_symbol(it.sev), sev=it.sev, enable=use_color)
                line = colorize(f"{it.line}:{it.col}", kind="path", enable=use_color)
                chain = f"  (via: {' → '.join(it.call_chain)})" if it.call_chain else ""
                print(f"  {line}  {sev} {it.msg}{chain}  [{it.rule}]")
            pf = per_file[fpath]
            parts = []
            for sev in SEV_ORDER:
                cnt = pf.get(sev, 0)
                if cnt:
                    parts.append(f"{colorize(sev, sev=sev, enable=use_color)}={cnt}")
            if parts:
                print("  -- summary:", ", ".join(parts))

        print("\nSection summary:")
        parts = []
        for sev in SEV_ORDER:
            cnt = total.get(sev, 0)
            if cnt:
                parts.append(f"{colorize(sev, sev=sev, enable=use_color)}={cnt}")
        if parts:
            print(" ", ", ".join(parts))
        print(f"  Total issues: {sum(total.values())}")

    # Print sections separately
    _print_section("Graph/Autodiff issues", graph_issues)
    if bcast_issues:
        _print_section("Potential broadcasting/rank-drop issues", bcast_issues)

    if not graph_issues and not bcast_issues:
        print(colorize("\n✅ No issues found.", kind="path", enable=use_color))


def make_json_payload(issues: List[Issue]) -> dict:
    graph_issues, bcast_issues = split_by_kind(issues)
    totals_graph, per_file_graph = summarize_issues(graph_issues)
    totals_bcast, per_file_bcast = summarize_issues(bcast_issues)
    return {
        "summary": {
            "graph": {
                "total": sum(totals_graph.values()),
                "by_severity": totals_graph
            },
            "broadcast": {
                "total": sum(totals_bcast.values()),
                "by_severity": totals_bcast
            },
            "overall_total": len(issues)
        },
        "per_file": {
            "graph": per_file_graph,
            "broadcast": per_file_bcast
        },
        "issues": [it.as_dict() for it in issues]
    }


# --------------------- Presets -----------------------------------------

PRESETS = {
    # Keep only the most dangerous graph/autodiff killers.
    "minimal": dict(
        scan_graph=True,
        scan_broadcast=False,
        max_call_depth=1,
        enable_only_rules={
            R_TF_PYFUNC, R_TF_DISCRETE, R_TF_CAST_INT,
            R_BOOL_AND_OR, R_ATTR_NUMPY
        }
    ),
    # Good default: all graph+autodiff tips, no broadcasting noise by default.
    "default": dict(
        scan_graph=True,
        scan_broadcast=False,
        max_call_depth=None,
        enable_only_rules=None,  # i.e., use all rules except those you disable manually
    ),
    # Strict: everything on, including broadcasting, unlimited depth.
    "strict": dict(
        scan_graph=True,
        scan_broadcast=True,
        max_call_depth=None,
        enable_only_rules=None,
    ),
}


# --------------------- Public API -----------------------------------------

def run_scan(paths: List[str],
             excludes: Optional[List[str]] = None,
             extra_decorators: Optional[List[str]] = None,
             fail_on_warning: bool = False,
             return_format: str = "text",  # "text" | "json"
             json_file: Optional[str] = None,
             color: Optional[bool] = None,
             scan_graph: bool = True,
             scan_broadcast: bool = False,
             max_call_depth: Optional[int] = None,
             disable_rules: Optional[List[str]] = None,
             enable_only_rules: Optional[List[str]] = None,
             preset: Optional[str] = None) -> List[dict]:
    """
    Programmatic API. Returns issues as list of dicts.
    """
    excludes = set(excludes or [])
    compiled_decorators = set(DEFAULT_COMPILED_DECORATORS)
    if extra_decorators:
        compiled_decorators.update(extra_decorators)

    # Apply preset defaults (if any)
    if preset:
        p = PRESETS.get(preset)
        if not p:
            raise ValueError(f"Unknown preset '{preset}'. Valid: {list(PRESETS.keys())}")
        scan_graph = p.get("scan_graph", scan_graph)
        scan_broadcast = p.get("scan_broadcast", scan_broadcast)
        max_call_depth = p.get("max_call_depth", max_call_depth)
        preset_enable_only = p.get("enable_only_rules", None)
    else:
        preset_enable_only = None

    # Merge rule toggles: enable-only (preset or user) takes precedence over disables.
    disabled_rules_set = set(disable_rules or [])
    enable_only_rules_set: Optional[Set[str]] = None
    if enable_only_rules:
        enable_only_rules_set = set(enable_only_rules)
    elif preset_enable_only:
        enable_only_rules_set = set(preset_enable_only)

    # expand targets
    files: List[pathlib.Path] = []
    for pth in paths:
        path = pathlib.Path(pth)
        if path.is_file():
            files.append(path)
        else:
            files.extend(path.rglob("*.py"))

    def excluded(f: pathlib.Path) -> bool:
        s = str(f)
        return any(fnmatch.fnmatch(s, pat) for pat in excludes)

    issues: List[Issue] = []
    for f in files:
        if excluded(f):
            continue
        issues.extend(
            scan_file(
                f, compiled_decorators,
                scan_graph=scan_graph, scan_broadcast=scan_broadcast,
                max_call_depth=max_call_depth,
                disabled_rules=disabled_rules_set,
                enable_only_rules=enable_only_rules_set
            )
        )

    # Output
    if return_format == "json":
        payload = make_json_payload(issues)
        print(json.dumps(payload, indent=2))
        if json_file:
            pathlib.Path(json_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        if color is None:
            color_enabled = _supports_color()
        else:
            color_enabled = bool(color)
        print_text_report(issues, use_color=color_enabled)

    # Fail CI only on graph/autodiff WARN+ (not broadcast) by default.
    if fail_on_warning and any(it.sev in (SEV_CRIT, SEV_ERR, SEV_WARN) and it.kind == "graph" for it in issues):
        sys.exit(2)

    return [it.as_dict() for it in issues]


# --------------------- CLI -----------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan")
    ap.add_argument("--exclude", action="append", default=[],
                    help='Glob pattern to exclude (repeatable), e.g., "**/__main__.py"')
    ap.add_argument("--decorator", action="append", default=[],
                    help="Additional decorator names treated as compiled (e.g., mylib.tf_jit)")
    ap.add_argument("--fail-on-warning", action="store_true",
                    help="Exit non-zero if any CRITICAL/ERROR/WARN (graph) issues are found")
    ap.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    ap.add_argument("--json-file", type=str, default=None, help="Also write JSON report to this file path")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI color in text output")

    # Scan toggles
    ap.add_argument("--scan-graph", dest="scan_graph", action="store_true", help="Enable graph/autodiff checks")
    ap.add_argument("--no-scan-graph", dest="scan_graph", action="store_false", help="Disable graph/autodiff checks")
    ap.add_argument("--scan-broadcast", dest="scan_broadcast", action="store_true",
                    help="Enable broadcasting/rank-drop heuristics")
    ap.add_argument("--no-scan-broadcast", dest="scan_broadcast", action="store_false",
                    help="Disable broadcasting/rank-drop heuristics")
    ap.set_defaults(scan_graph=True, scan_broadcast=False)

    # Depth & rules
    ap.add_argument("--max-call-depth", type=int, default=None,
                    help="Limit recursion into helper functions (0=only decorated, 1=one level, default=unlimited)")
    ap.add_argument("--disable-rule", action="append", default=[],
                    help="Disable a specific rule by id (repeatable). e.g., --disable-rule print_in_compiled")
    ap.add_argument("--enable-only-rule", action="append", default=None,
                    help="Only enable these rules (repeatable). If set, disables all others.")

    # Preset
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                    help="Apply a preset of settings: minimal | default | strict")

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return_format = "json" if args.json else "text"
    color = False if args.no_color else None
    run_scan(paths=args.paths,
             excludes=args.exclude,
             extra_decorators=args.decorator,
             fail_on_warning=args.fail_on_warning,
             return_format=return_format,
             json_file=args.json_file,
             color=color,
             scan_graph=args.scan_graph,
             scan_broadcast=args.scan_broadcast,
             max_call_depth=args.max_call_depth,
             disable_rules=args.disable_rule,
             enable_only_rules=args.enable_only_rule,
             preset=args.preset)


if __name__ == "__main__":
    main()
