import ast
import argparse
from pathlib import Path
from typing import Set, Dict, List, Tuple
import yaml
import sys
import traceback


class ConfigUsageAnalyzer(ast.NodeVisitor):
    """AST visitor that tracks config key access patterns."""

    def __init__(self, config_var_name: str = "CONFIG"):
        self.config_var_name = config_var_name
        self.used_keys: Dict[Tuple[str, ...], Set[str]] = {}
        self.var_to_path: Dict[str, Tuple[str, ...]] = {}
        self.visit_depth = 0
        self.max_depth = 0

    def generic_visit(self, node):
        """Override to track recursion depth."""
        self.visit_depth += 1
        self.max_depth = max(self.max_depth, self.visit_depth)

        if self.visit_depth > 900:  # Safety check
            print(f"WARNING: Deep recursion at depth {self.visit_depth}")
            print(f"Node type: {node.__class__.__name__}")
            self.visit_depth -= 1
            return

        try:
            super().generic_visit(node)
        finally:
            self.visit_depth -= 1

    def visit_Assign(self, node: ast.Assign):
        """Track assignments that create intermediate config variables."""
        if isinstance(node.value, ast.Subscript):
            path = self._extract_config_path(node.value)
            if path is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.var_to_path[target.id] = path

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        """Track subscript access: obj["key"]."""
        path = self._extract_config_path(node)
        if path is not None:
            if len(path) > 1:
                parent_path = path[:-1]
                key = path[-1]
                if parent_path not in self.used_keys:
                    self.used_keys[parent_path] = set()
                self.used_keys[parent_path].add(key)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Track .get() calls: obj.get("key", default)."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            # Check if this is called on a config variable
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if var_name in self.var_to_path:
                    # This is something like: API_CFG.get("key", default)
                    path = self.var_to_path[var_name]
                    if node.args and isinstance(node.args[0], ast.Constant):
                        key = node.args[0].value
                        if path not in self.used_keys:
                            self.used_keys[path] = set()
                        self.used_keys[path].add(key)
            # NEW: Handle chained access like CONFIG["deployment"].get("backup_enabled")
            elif isinstance(node.func.value, ast.Subscript):
                path = self._extract_config_path(node.func.value)
                if path is not None and node.args and isinstance(node.args[0], ast.Constant):
                    key = node.args[0].value
                    if path not in self.used_keys:
                        self.used_keys[path] = set()
                    self.used_keys[path].add(key)

        self.generic_visit(node)

    def _extract_config_path(self, node: ast.Subscript, depth: int = 0) -> Tuple[str, ...] | None:
        """Extract the full path from a subscript chain."""
        if depth > 50:  # Prevent infinite recursion
            print(f"WARNING: Deep subscript chain detected (depth {depth})")
            return None

        path_parts = []
        current = node

        # Iterative instead of recursive
        chain_depth = 0
        while isinstance(current, ast.Subscript):
            chain_depth += 1
            if chain_depth > 50:
                print(f"WARNING: Very deep subscript chain (>{chain_depth} levels)")
                return None

            if isinstance(current.slice, ast.Constant):
                path_parts.append(current.slice.value)
            else:
                return None
            current = current.value

        if isinstance(current, ast.Name):
            if current.id == self.config_var_name:
                return tuple(reversed(path_parts))
            elif current.id in self.var_to_path:
                base_path = self.var_to_path[current.id]
                return base_path + tuple(reversed(path_parts))

        return None


def flatten_config_keys(config: dict, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Set[str]]:
    """Flatten config - iterative version to avoid recursion."""
    result = {}

    # Use a stack instead of recursion
    stack = [(config, prefix)]
    seen = set()

    while stack:
        current_config, current_prefix = stack.pop()

        if not isinstance(current_config, dict):
            continue

        # Cycle detection
        config_id = id(current_config)
        if config_id in seen:
            continue
        seen.add(config_id)

        # Record keys at this level
        keys = set(current_config.keys())
        result[current_prefix] = keys

        # Add children to stack
        for key, value in current_config.items():
            if isinstance(value, dict):
                stack.append((value, current_prefix + (key,)))

    return result


def analyze_file(filepath: Path, config_var_name: str = "CONFIG") -> ConfigUsageAnalyzer:
    """Analyze a single Python file for config usage."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except UnicodeDecodeError:
        # Skip files that aren't valid UTF-8 (likely binary or generated files)
        return ConfigUsageAnalyzer(config_var_name)
    except SyntaxError as e:
        print(f"Warning: Syntax error in {filepath}: {e}")
        return ConfigUsageAnalyzer(config_var_name)
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return ConfigUsageAnalyzer(config_var_name)

    analyzer = ConfigUsageAnalyzer(config_var_name)
    analyzer.visit(tree)
    return analyzer


def analyze_directory(
        directory: Path,
        config_var_name: str = "CONFIG",
        exclude_patterns: Set[str] = None
) -> ConfigUsageAnalyzer:
    """Analyze all Python files in a directory recursively."""
    combined = ConfigUsageAnalyzer(config_var_name)

    # Default exclusions
    if exclude_patterns is None:
        exclude_patterns = {
            '.venv', 'venv', 'env',
            '.git', '.svn', '.hg',
            'node_modules',
            '__pycache__', '.pytest_cache', '.mypy_cache',
            '.cache', '.uv', '.ruff_cache',
            'build', 'dist', '.eggs', '*.egg-info',
            '.tox', '.nox'
        }

    def should_exclude(path: Path) -> bool:
        """Check if path should be excluded."""
        parts = path.parts
        for pattern in exclude_patterns:
            if pattern.startswith('*.'):
                # Suffix match
                if path.name.endswith(pattern[1:]):
                    return True
            else:
                # Directory name match
                if pattern in parts:
                    return True
        return False

    py_files = [f for f in directory.rglob("*.py") if not should_exclude(f)]
    print(f"Found {len(py_files)} Python files (after exclusions)")

    for py_file in py_files:
        analyzer = analyze_file(py_file, config_var_name)

        # Merge results
        for path, keys in analyzer.used_keys.items():
            if path not in combined.used_keys:
                combined.used_keys[path] = set()
            combined.used_keys[path].update(keys)

        combined.var_to_path.update(analyzer.var_to_path)

    return combined


def find_unused_keys(
        config: dict,
        used_keys: Dict[Tuple[str, ...], Set[str]],
        ignore_patterns: Set[str] = None,
        ignore_parents: bool = True
) -> List[Tuple[List[str], str]]:
    """Find config keys that are defined but never used."""
    all_keys = flatten_config_keys(config)
    unused = []
    ignore_patterns = ignore_patterns or set()

    for path, defined_keys in all_keys.items():
        if not defined_keys:
            continue

        accessed_keys = used_keys.get(path, set())
        unaccessed = defined_keys - accessed_keys

        for key in sorted(unaccessed):
            full_path = list(path) + [key]
            full_key_str = ".".join(full_path)

            # Check ignore patterns
            if _matches_ignore_pattern(full_key_str, ignore_patterns):
                continue

            # Check if this is a parent key with accessed children
            if ignore_parents:
                child_path = tuple(full_path)
                if child_path in used_keys and used_keys[child_path]:
                    continue

            unused.append((list(path), key))

    return unused


def _matches_ignore_pattern(key: str, patterns: Set[str]) -> bool:
    """Check if a key matches any ignore pattern."""
    for pattern in patterns:
        if pattern == key:
            return True
        if "*" in pattern:
            pattern_parts = pattern.split("*")
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                if key.startswith(prefix) and key.endswith(suffix):
                    return True
    return False


def format_path(path: List[str], key: str) -> str:
    """Format a config path for display."""
    if path:
        return f"{'.'.join(path)}.{key}"
    return key


def main():
    parser = argparse.ArgumentParser(description="Config linter with debug output")
    parser.add_argument("config", type=Path, help="Path to config.yml file")
    parser.add_argument("source", type=Path, help="Path to Python source (file or directory)")
    parser.add_argument("--var-name", default="CONFIG", help="Config variable name (default: CONFIG)")
    parser.add_argument("--ignore", action="append", help="Ignore keys matching pattern")
    parser.add_argument("--exclude", action="append", help="Exclude directories/patterns from scanning")
    parser.add_argument("--no-default-excludes", action="store_true", help="Don't exclude .venv, node_modules, etc.")
    parser.add_argument("--no-ignore-parents", action="store_true", help="Report parent keys as unused")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show used keys as well")
    parser.add_argument("--recursion-limit", type=int, default=1000, help="Set Python recursion limit")

    args = parser.parse_args()

    # Set recursion limit
    sys.setrecursionlimit(args.recursion_limit)
    print(f"Recursion limit: {args.recursion_limit}")
    print()

    # Load config
    print("Loading config...")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Loaded: {args.config}")
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return 1

    print()

    # Prepare exclude patterns
    exclude_patterns = None if args.no_default_excludes else None  # Use default
    if args.exclude:
        if exclude_patterns is None:
            exclude_patterns = {
                '.venv', 'venv', 'env',
                '.git', '.svn', '.hg',
                'node_modules',
                '__pycache__', '.pytest_cache', '.mypy_cache',
                'build', 'dist', '.eggs', '*.egg-info',
                '.tox', '.nox'
            }
        exclude_patterns.update(args.exclude)

    # Analyze source
    print("Analyzing source code...")
    try:
        if args.source.is_file():
            analyzer = analyze_file(args.source, args.var_name)
        else:
            analyzer = analyze_directory(args.source, args.var_name, exclude_patterns)
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        traceback.print_exc()
        return 1

    print()

    # Find unused keys
    ignore_patterns = set(args.ignore) if args.ignore else set()
    unused = find_unused_keys(
        config,
        analyzer.used_keys,
        ignore_patterns=ignore_patterns,
        ignore_parents=not args.no_ignore_parents
    )

    # Report results
    print(f"Configuration Linter Results")
    print(f"{'=' * 60}")
    print(f"Config file: {args.config}")
    print(f"Source: {args.source}")
    print(f"Config variable: {args.var_name}")
    if ignore_patterns:
        print(f"Ignoring: {', '.join(sorted(ignore_patterns))}")
    print()

    if args.verbose:
        print("Used keys:")
        if analyzer.used_keys:
            for path, keys in sorted(analyzer.used_keys.items()):
                path_str = ".".join(path) if path else "(root)"
                for key in sorted(keys):
                    print(f"  ✓ {path_str}.{key}")
        else:
            print("  (none detected)")
        print()

    print(f"Unused keys: {len(unused)}")
    if unused:
        for path, key in unused:
            print(f"  ✗ {format_path(path, key)}")
        print()
        return 1
    else:
        print("  All config keys are used! ✓")
        print()
        return 0


if __name__ == "__main__":
    exit(main())