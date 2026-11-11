"""一键跑全部测试：契约测试 + 端到端流水线测试。

用法：
    E:\\123\\runtime\\python\\python.exe tests\\run_tests.py
或双击项目根目录的 ``run_tests.bat``。

退出码 0 = 全通过，1 = 有失败（供 CI / 人肉判断用）。
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

MODULES = [
    ("契约测试（四件套 / LLM / 解耦 / 工程约束 / 算法）", "test_contracts"),
    ("端到端测试（输入 → 产物落盘 → 质检通过）", "test_pipeline"),
]


def run_module(title: str, module_name: str) -> tuple[int, int]:
    print(f"\n=== {title} ===")
    module = importlib.import_module(module_name)
    tests = [
        obj
        for name, obj in sorted(vars(module).items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = failed = 0
    for fn in tests:
        started = time.perf_counter()
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}  ({time.perf_counter() - started:.1f}s)")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"  -> {passed}/{passed + failed} 通过")
    return passed, failed


def main() -> int:
    total_passed = total_failed = 0
    for title, module_name in MODULES:
        passed, failed = run_module(title, module_name)
        total_passed += passed
        total_failed += failed
    print("\n" + "=" * 60)
    if total_failed == 0:
        print(f"全部通过：{total_passed} 项")
    else:
        print(f"失败 {total_failed} 项 / 通过 {total_passed} 项")
    print("=" * 60)
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())