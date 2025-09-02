#!/usr/bin/env python3
"""
Severity-aware static scanner for TensorFlow graph-compat / GPU efficiency.

Severities:
  ✖✖✖  CRITICAL  (likely silent wrong semantics if it "works")
  ✖✖    ERROR     (likely to raise in graph/GPU mode)
  ✖      WARN     (likely to run but inefficient / suboptimal)

It treats functions as "compiled" if they are decorated with any of:
  - @tf.function / @tf.function(...)
  - @tf_compile / @tf_compile(...)
  - @tf.compile (TF 2.17+)
You can add more via --decorator.

It also recursively scans helper functions **called from compiled ones**.

CLI:
  python scan_tf_graph_readiness.py core utilities optimizers \
    --exclude "**/__main__.py" \
    --exclude "**/scan_tf_graph_readiness.py" \
    --fail-on-warning

Programmatic use:
  from utilities.scan_tf_graph_readiness import run_scan
  issues = run_scan(paths=["core","utilities","optimizers"],
                    excludes=["**/__main__.py","**/scan_tf_graph_readiness.py"],
                    extra_decorators=None,
                    fail_on_warning=False)
  # 'issues' is a list of dicts with file/line/col/severity/message.
"""

import ast, sys, pathlib, fnmatch, argparse, json
from typing import Dict, Set, List, Optional, cast

# -------------------- Rule catalog & severities -------------------------------

SEV_CRIT = "CRITICAL"  # ✖✖✖
SEV_ERR = "ERROR"  # ✖✖
SEV_WARN = "WARN"  # ✖

BAD_ATTR_CALLS = {"numpy", "item", "tolist"}  # -> ERROR
BAD_BUILTINS = {"float", "int", "bool", "list"}  # -> ERROR when on tensors
LEN_BUILTIN = {"len"}  # -> ERROR when on tensors (dynamic) / risky
BAD_PRINT = {"print"}  # -> WARN (use tf.print)
BAD_NP_FUNCS = {"array", "asarray", "asanyarray", "ascontiguousarray"}  # -> WARN (host copies)
BAD_BOOL_OPS = {"And", "Or"}  # -> CRITICAL (Python 'and/or' on tensors)
BAD_TRY = True  # -> ERROR (not supported in AutoGraph)
BAD_FOR_ITERS = {"range", "enumerate", "zip"}  # -> WARN (prefer tf.range / vectorize)

# Treat as compiled if decorated with these (you can extend via CLI)
DEFAULT_COMPILED_DECORATORS = {
    "tf.function", "tf_function", "tf_function_jit",
    "tf.compile"  # TF 2.17+
}


# --------------------- Scanner ------------------------------------------------

def sev_symbol(sev: str) -> str:
    return {"CRITICAL": "✖✖✖", "ERROR": "✖✖", "WARN": "✖"}[sev]


class Issue:
    def __init__(self, path: str, line: int, col: int, sev: str, msg: str):
        self.path, self.line, self.col, self.sev, self.msg = path, line, col, sev, msg

    def __str__(self):
        return f"{self.path}:{self.line}:{self.col}  {sev_symbol(self.sev)} {self.msg}"

    def as_dict(self):
        return {"path": self.path, "line": self.line, "col": self.col, "severity": self.sev, "message": self.msg}


class FileScanner(ast.NodeVisitor):
    def __init__(self, path: pathlib.Path, compiled_decorators: Set[str], symtab: Dict[str, ast.AST],
                 exclude_globs: List[str], numpy_aliases: Set[str]):
        self.path = path
        self.compiled_decorators = compiled_decorators
        self.symtab = symtab
        self.exclude_globs = exclude_globs
        self.numpy_aliases = numpy_aliases

        self.in_compiled_stack = [False]
        self.current_function: List[str] = []
        self._call_stack: List[str] = []
        self.issues: List[Issue] = []

    # ---- helpers
    def warn(self, node: ast.AST, sev: str, msg: str):
        self.issues.append(Issue(str(self.path), getattr(node, "lineno", 1), getattr(node, "col_offset", 0), sev, msg))

    def _decorator_name(self, dec: ast.AST) -> Optional[str]:
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Attribute):
            parts = []
            cur = dec
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

    def _is_compiled_decorator(self, dec: ast.AST) -> bool:
        name = self._decorator_name(dec)
        if not name:
            return False
        normalized = name.replace("tensorflow.", "tf.")
        return (normalized in self.compiled_decorators) or (normalized.split(".")[-1] in self.compiled_decorators)

    # ---- visits
    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_compiled_here = any(self._is_compiled_decorator(cast(ast.AST, d)) for d in node.decorator_list)
        effective_compiled = is_compiled_here or self.in_compiled_stack[-1]
        self.in_compiled_stack.append(effective_compiled)
        self.current_function.append(node.name)
        self.generic_visit(node)
        self.current_function.pop()
        self.in_compiled_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Try(self, node: ast.Try):
        if self.in_compiled_stack[-1] and BAD_TRY:
            self.warn(node, SEV_ERR, "try/except inside compiled function is unsupported by AutoGraph")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        if self.in_compiled_stack[-1] and type(node.op).__name__ in BAD_BOOL_OPS:
            self.warn(node, SEV_CRIT, "Python 'and/or' on tensors in compiled code; use tf.logical_and / tf.logical_or")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self.in_compiled_stack[-1]:
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                fname = node.iter.func.id
                if fname in BAD_FOR_ITERS:
                    self.warn(node, SEV_WARN,
                              f"{fname}(...) loop in compiled function; prefer tf.range or vectorize / tf.while_loop")
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        if self.in_compiled_stack[-1]:
            self.warn(node, SEV_WARN,
                      "while loop in compiled function: ensure condition is tensor-based (AutoGraph -> tf.while_loop)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        in_compiled = self.in_compiled_stack[-1]

        # print(...)
        if in_compiled and isinstance(node.func, ast.Name) and node.func.id in BAD_PRINT:
            self.warn(node, SEV_WARN, "print() in compiled function; use tf.print (sparingly)")

        # float/int/bool/list(...)
        if in_compiled and isinstance(node.func, ast.Name) and node.func.id in BAD_BUILTINS:
            self.warn(node, SEV_ERR,
                      f"{node.func.id}() on tensors inside compiled function; use tensor ops (tf.cast, tf.shape, etc.)")

        # len(...)
        if in_compiled and isinstance(node.func, ast.Name) and node.func.id in LEN_BUILTIN:
            self.warn(node, SEV_ERR,
                      "len(tensor) in compiled function risks trace-time const or error; use tf.shape(tensor)[i]")

        # x.numpy() / x.item() / x.tolist()
        if in_compiled and isinstance(node.func, ast.Attribute) and node.func.attr in BAD_ATTR_CALLS:
            self.warn(node, SEV_ERR,
                      f".{node.func.attr}() pulls tensors to host in compiled code; breaks GPU graph execution")

        # numpy conversions
        if in_compiled:
            if isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in self.numpy_aliases and node.func.attr in BAD_NP_FUNCS:
                    self.warn(node, SEV_WARN,
                              f"np.{node.func.attr}(...) in compiled function; avoids fusion and triggers host copies")
            elif isinstance(node.func, ast.Name):
                if node.func.id in BAD_NP_FUNCS and "numpy" in self.numpy_aliases:
                    self.warn(node, SEV_WARN,
                              f"numpy {node.func.id}(...) in compiled function; avoid NumPy conversions in hot path")

        # Recursively scan user-defined callees as compiled
        if in_compiled:
            callee_name = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            # (method calls can't be resolved reliably statically)
            if callee_name and callee_name in self.symtab:
                if callee_name not in self._call_stack:
                    self._call_stack.append(callee_name)
                    callee_def = self.symtab[callee_name]
                    self.in_compiled_stack.append(True)
                    self.current_function.append(callee_name)
                    self.generic_visit(callee_def)
                    self.current_function.pop()
                    self.in_compiled_stack.pop()
                    self._call_stack.pop()

        self.generic_visit(node)


# --------------------- Top-level API -----------------------------------------

def build_symbol_table(tree: ast.AST) -> Dict[str, ast.FunctionDef]:
    sym = {}
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


def scan_file(path: pathlib.Path, compiled_decorators: Set[str], exclude_globs: List[str]) -> List[Issue]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [Issue(str(path), 1, 0, SEV_ERR, f"Could not read file: {e}")]
    try:
        tree = ast.parse(text, filename=str(path))
    except Exception as e:
        return [Issue(str(path), 1, 0, SEV_ERR, f"Parse error: {e}")]

    symtab = build_symbol_table(tree)
    np_aliases = collect_numpy_aliases(tree)
    scanner = FileScanner(path, compiled_decorators, symtab, [], np_aliases)
    scanner.visit(tree)
    return scanner.issues


def run_scan(paths: List[str],
             excludes: Optional[List[str]] = None,
             extra_decorators: Optional[List[str]] = None,
             fail_on_warning: bool = False,
             return_format: str = "text",  # "text" | "json"
             json_file: Optional[str] = None) -> List[dict]:
    excludes = excludes or []
    compiled_decorators = set(DEFAULT_COMPILED_DECORATORS)
    if extra_decorators:
        compiled_decorators.update(extra_decorators)

    # expand targets
    files: List[pathlib.Path] = []
    for p in paths:
        path = pathlib.Path(p)
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
        issues.extend(scan_file(f, compiled_decorators, excludes))

    issues_json = [it.as_dict() for it in issues]

    # Output
    if return_format == "json":
        payload = {
            "summary": {
                "total": len(issues_json),
                "by_severity": {
                    "CRITICAL": sum(1 for i in issues_json if i["severity"] == "CRITICAL"),
                    "ERROR": sum(1 for i in issues_json if i["severity"] == "ERROR"),
                    "WARN": sum(1 for i in issues_json if i["severity"] == "WARN"),
                }
            },
            "issues": issues_json
        }
        print(json.dumps(payload, indent=2))
        if json_file:
            pathlib.Path(json_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        if issues:
            for it in issues:
                print(str(it))
            print(f"⚠️  Found {len(issues)} potential issues.")
        else:
            print("✅ No issues found.")

    if fail_on_warning and issues:
        sys.exit(2)

    return issues_json


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan")
    ap.add_argument("--exclude", action="append", default=[],
                    help='Glob pattern to exclude (repeatable), e.g., "**/__main__.py"')
    ap.add_argument("--decorator", action="append", default=[],
                    help="Additional decorator names treated as compiled (e.g., mylib.tf_jit)")
    ap.add_argument("--fail-on-warning", action="store_true",
                    help="Exit non-zero if any warnings are found")
    ap.add_argument("--json", action="store_true",
                    help="Print JSON report to stdout")
    ap.add_argument("--json-file", type=str, default=None,
                    help="Also write JSON report to this file path")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_scan(paths=args.paths,
             excludes=args.exclude,
             extra_decorators=args.decorator,
             fail_on_warning=args.fail_on_warning,
             return_format=("json" if args.json else "text"),
             json_file=args.json_file)


if __name__ == "__main__":
    main()
