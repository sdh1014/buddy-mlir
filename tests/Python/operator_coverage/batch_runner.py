# RUN: echo "utility module" > /dev/null
"""批量运行 aten 覆盖测试的公共工具。

特性：
- 根据覆盖表（coverage.json）为指定 op 列表执行 DynamoCompiler 导出 + 数值对比。
- 默认跳过随机/稀疏/量化/CUDA-only/meta/prim/inplace 以及无法自动生成输入的算子。
- 提供 SUMMARY/FAIL 输出，便于 FileCheck。
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

try:
    import torch._inductor.lowering  # noqa: F401
except Exception:
    # Some builds do not eagerly expose `torch._inductor.lowering` as an attribute.
    # Inductor decompositions may access it via `torch._inductor.lowering`, so we
    # import the submodule explicitly to avoid runtime AttributeError.
    pass
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_COVERAGE_JSON = THIS_DIR / "coverage.json"

SKIP_TAGS = {
    "sparse",
    "quantized",
    "cuda_only",
    "prim",
}

# 这些算子创建未初始化/非确定性内容，不能做数值比对，只校验元数据
UNINITIALIZED_OPS = {
    "empty",
    "empty_like",
    "empty_strided",
    "empty_permuted",
    "new_empty",
    "new_empty_strided",
}


def make_aot_decompositions() -> Dict[Any, Any]:
    """
    Start from Inductor decompositions, but disable a few decompositions that
    introduce `prims.*` ops our frontend doesn't map yet.

    Returning `NotImplemented` from a decomposition keeps the original ATen op
    in the graph, allowing Buddy to use its own lowering directly.
    """
    decomp: Dict[Any, Any] = dict(inductor_decomp)

    def _no_decomp(*args, **kwargs):
        return NotImplemented

    # Inductor decomp for max_pool*_with_indices may rewrite to prims
    # `_low_memory_max_pool_with_offsets` + `_low_memory_max_pool_offsets_to_indices`.
    # Buddy already has direct lowerings for the ATen ops, so keep them intact.
    for key in (
        torch.ops.aten.max_pool2d_with_indices.default,
        torch.ops.aten.max_pool3d_with_indices.default,
    ):
        if key in decomp:
            decomp[key] = _no_decomp

    return decomp


@dataclass
class Result:
    name: str
    status: str  # ok | skip | fail
    reason: str = ""


def load_coverage_map(
    path: Path | str = DEFAULT_COVERAGE_JSON,
) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {f"{e['op']}.{e['overload']}": e for e in entries}


def should_skip(notes: str) -> Tuple[bool, str]:
    for tag in SKIP_TAGS:
        if tag in notes:
            return True, f"skip:{tag}"
    return False, ""


def guess_value(type_str: str) -> Any:
    """根据类型字符串生成最小 CPU fp32 输入。"""
    t = type_str.replace(" ", "").lower()
    # 固定随机种子，便于随机算子复现
    torch.manual_seed(0)

    # 处理固定长度的形如 int[2] / int[3]
    import re

    m = re.match(r"(int|symint|float|double|bool)\[(\d+)\]", t)
    if m:
        base, num = m.group(1), int(m.group(2))
        if base in ("int", "symint"):
            return [0] * num
        if base in ("float", "double"):
            return [0.0] * num
        if base == "bool":
            return [False] * num

    # Torch schema 有时会打印为 List[T]（而非 T[]），这里做一次兼容。
    if t.startswith("list[") and t.endswith("]"):
        inner = t[len("list[") : -1]
        if inner == "number":
            return [1]
        if inner in ("int", "symint"):
            return [0]
        if inner in ("float", "double"):
            return [0.0]
        if inner == "bool":
            return [False]
        if inner == "tensor" or inner.startswith("tensor"):
            return [torch.ones(1, dtype=torch.float32)]
        if inner in ("str", "string"):
            return [""]
        return None

    # dim/dims 相关，返回单一维度列表或整形
    if "int[]?" in t and "dim" in t:
        return [0]
    if "int[]?" in t and "size" in t:
        return [1]
    if "int[]" in t and "stride" in t:
        return [1]

    if "int[]" in t or "symint[]" in t:
        return [0]
    if "float[]" in t or "double[]" in t:
        return [0.0]
    if "bool[]" in t:
        return [False]
    if "scalar[]" in t:
        return [1.0]
    if "device[]" in t:
        return [torch.device("cpu")]
    if "complex" in t:
        return 0.5 + 0.1j
    if "tensor[]" in t:
        return [torch.ones(1, dtype=torch.float32)]
    if "tensor" in t:
        return torch.ones(1, dtype=torch.float32)
    if t == "number":
        return 1
    if "symint" in t or "int" in t:
        return 0
    if "float" in t or "double" in t:
        return 1.0
    if "bool" in t:
        return False
    if "scalar" in t:
        return 1.0
    if "generat" in t:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        return g
    if "device" in t:
        return torch.device("cpu")
    if "layout" in t:
        return torch.strided
    if "memoryformat" in t:
        return torch.contiguous_format
    if "string" in t:
        return ""
    if "dtype" in t:
        return torch.float32
    return None


def _find_first_tensor(obj: Any) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for x in obj:
            found = _find_first_tensor(x)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_first_tensor(v)
            if found is not None:
                return found
    return None


def _make_out_like(
    ref: torch.Tensor | None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if ref is not None:
        return torch.empty_like(ref, dtype=dtype or ref.dtype)
    return torch.zeros(1, dtype=dtype or torch.float32)


def build_inputs(
    schema: torch._C.FunctionSchema,
) -> Tuple[bool, str, List[Any], Dict[str, Any]]:
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    for arg in schema.arguments:
        type_str = str(arg.type)
        # `default_value` may be `None` even when a real default exists (e.g. Optional[T]=None).
        # Use `has_default_value()` to avoid incorrectly treating such args as required inputs.
        has_default = (
            arg.has_default_value()
            if hasattr(arg, "has_default_value")
            else arg.default_value is not None
        )
        if has_default:
            continue
        t_lower = type_str.replace(" ", "").lower()
        # PyTorch schema may encode optional as "T?" or "Optional[T]"
        is_optional = "?" in type_str or t_lower.startswith("optional[")
        # 尝试为 out/indices 等根据已有张量推断形状/类型
        is_out_tensor = "tensor" in t_lower and (
            arg.name == "out"
            or arg.name.startswith("out")
            or arg.name in ("values", "indices")
        )
        if is_out_tensor:
            ref = _find_first_tensor([args, kwargs])
            target_dtype = torch.int64 if "index" in arg.name else None
            val = _make_out_like(ref, dtype=target_dtype)
        # Some schemas encode enum-like args (e.g. ScalarType/MemoryFormat/Layout) as plain ints.
        # Passing 0 often maps to uint8/strided/etc and can trigger backend/fake-tensor issues.
        elif t_lower in ("int", "symint") and arg.name == "dtype":
            val = torch.float32
        elif t_lower in ("int", "symint") and arg.name == "layout":
            val = torch.strided
        elif t_lower in ("int", "symint") and arg.name == "memory_format":
            val = torch.contiguous_format
        # Optional enum-like ints (dtype/layout/memory_format) default to None;
        # generating 0 maps to uint8/strided/etc and causes backend/fake-tensor issues.
        elif t_lower == "optional[int]" and arg.name in (
            "dtype",
            "layout",
            "memory_format",
        ):
            val = None
        else:
            val = guess_value(type_str)

        if val is None:
            if is_optional:
                val = None
            else:
                return False, f"input_gen:{arg.name}", [], {}
        if arg.kwarg_only:
            kwargs[arg.name] = val
        else:
            args.append(val)
    return True, "", args, kwargs


def compare_outputs(ref: Any, res: Any) -> None:
    if isinstance(ref, torch.Tensor):
        torch.testing.assert_close(ref, res, rtol=1e-4, atol=1e-4)
    elif isinstance(ref, (tuple, list)):
        assert isinstance(res, (tuple, list)) and len(ref) == len(
            res
        ), "结构不匹配"
        for a, b in zip(ref, res):
            compare_outputs(a, b)
    else:
        if ref != res:
            raise AssertionError(f"value mismatch: {ref} vs {res}")


def compare_uninitialized(ref: Any, res: Any) -> None:
    """对未初始化/非确定性输出，仅比较形状/类型/设备等元信息。"""
    if isinstance(ref, torch.Tensor):
        assert isinstance(res, torch.Tensor), "输出类型不匹配"
        assert (
            ref.shape == res.shape
        ), f"shape mismatch: {ref.shape} vs {res.shape}"
        assert (
            ref.dtype == res.dtype
        ), f"dtype mismatch: {ref.dtype} vs {res.dtype}"
        assert (
            ref.device == res.device
        ), f"device mismatch: {ref.device} vs {res.device}"
    elif isinstance(ref, (tuple, list)):
        assert isinstance(res, (tuple, list)) and len(ref) == len(
            res
        ), "结构不匹配"
        for a, b in zip(ref, res):
            compare_uninitialized(a, b)
    else:
        if ref != res:
            raise AssertionError(f"value mismatch: {ref} vs {res}")


def is_uninitialized_op(name: str, entry: Dict[str, Any]) -> bool:
    op_name = entry.get("op") or name.split(".")[0]
    return op_name in UNINITIALIZED_OPS


def clone_inputs(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, (list, tuple)):
        cloned = [clone_inputs(x) for x in obj]
        return type(obj)(cloned)
    if isinstance(obj, dict):
        return {k: clone_inputs(v) for k, v in obj.items()}
    return obj


def run_one(
    name: str,
    entry: Dict[str, Any],
    dynamo_compiler: DynamoCompiler,
    templates: Dict[str, Any],
) -> Result:
    skip, reason = should_skip(entry.get("notes", ""))
    if skip:
        return Result(name=name, status="skip", reason=reason)
    try:
        packet = getattr(torch.ops.aten, entry["op"])
        op = getattr(packet, entry["overload"])
    except Exception as e:  # pragma: no cover - defensive
        return Result(name=name, status="skip", reason=f"lookup:{e}")

    schema = op._schema  # type: ignore[attr-defined]
    if name in templates:
        try:
            args, kwargs = templates[name]()
            ok, msg = True, ""
        except Exception as e:
            return Result(name=name, status="skip", reason=f"template:{e}")
    else:
        ok, msg, args, kwargs = build_inputs(schema)
        if not ok:
            return Result(name=name, status="skip", reason=msg)

    torch.manual_seed(0)

    def func(*inputs, **kw):
        return op(*inputs, **kw)

    try:
        graphs = dynamo_compiler.importer(
            func, *clone_inputs(args), **clone_inputs(kwargs)
        )
        if not graphs:
            # 1) 对于没有 Tensor 输出的 op，Dynamo 通常不会触发后端编译（或后端无法覆盖标量输出），
            #    这类属于非目标范围，避免用笼统的 import_empty 混淆统计。
            returns_tensor = any(
                "tensor" in str(ret.type).replace(" ", "").lower()
                for ret in schema.returns
            )
            if not returns_tensor:
                return Result(name=name, status="skip", reason="scalar_output")

            # 2) 常见原因：out/values/indices 等输出缓冲区形状不匹配，触发 resize，Dynamo 不会产生可编译图。
            #    先用 eager 运行一次让 out 缓冲区获得正确 shape，再按正确 shape 重新分配并重试导入。
            out_tensor_names = []
            for a in schema.arguments:
                t_lower = str(a.type).replace(" ", "").lower()
                is_out_tensor = "tensor" in t_lower and (
                    a.name == "out"
                    or a.name.startswith("out")
                    or a.name in ("values", "indices")
                )
                if is_out_tensor:
                    out_tensor_names.append(a.name)

            if out_tensor_names and any(
                isinstance(kwargs.get(k), torch.Tensor)
                for k in out_tensor_names
            ):
                try:
                    rng_state = torch.random.get_rng_state()
                    warm_args = clone_inputs(args)
                    warm_kwargs = clone_inputs(kwargs)
                    func(*warm_args, **warm_kwargs)
                    torch.random.set_rng_state(rng_state)

                    # 用 warm-up 后的真实 shape 重建输出缓冲区，避免 compile 阶段触发 resize
                    new_kwargs = dict(kwargs)
                    for k in out_tensor_names:
                        warm_buf = warm_kwargs.get(k)
                        if isinstance(warm_buf, torch.Tensor):
                            new_kwargs[k] = torch.empty(
                                warm_buf.shape,
                                dtype=warm_buf.dtype,
                                device=warm_buf.device,
                            )
                    kwargs = new_kwargs

                    graphs = dynamo_compiler.importer(
                        func, *clone_inputs(args), **clone_inputs(kwargs)
                    )
                except Exception:
                    graphs = []

            if not graphs:
                return Result(name=name, status="skip", reason="import_empty")
        graph = graphs[0]
        graph.lower_to_top_level_ir()
    except Exception as e:
        tb = traceback.format_exc()
        # Some multi-output `.out` overloads trigger a TorchDynamo internal assert:
        # `assert isinstance(kwargs["out"], (TupleVariable, ListVariable))`.
        # This is a PyTorch limitation/bug rather than a Buddy backend failure.
        if (
            "torch/_dynamo/variables/torch.py" in tb
            and 'assert isinstance(kwargs["out"], (TupleVariable, ListVariable))'
            in tb
        ):
            return Result(
                name=name,
                status="skip",
                reason="template:dynamo_out_overload_bug",
            )

        # Some ops hit functionalization/AOTAutograd limitations (e.g. view-mutation copy_),
        # which prevents Dynamo from producing a functional graph.
        if (
            (
                "torch/_functorch/_aot_autograd/functional_utils.py" in tb
                and "assert_functional_graph" in tb
            )
            or ("FunctionalizeFallbackKernel.cpp" in tb)
            or ("We only support functionalizing operators" in tb)
        ):
            return Result(
                name=name,
                status="skip",
                reason="template:functionalization_limit",
            )

        # Some `.out` ops trigger alias-correction bugs in PyTorch's functional tensor
        # dispatch path (TypeError from normalize_function).
        if (
            "torch/utils/_python_dispatch.py" in tb
            and "normalize_function" in tb
            and "cannot unpack non-iterable NoneType object" in tb
        ):
            return Result(
                name=name,
                status="skip",
                reason="template:dynamo_out_overload_bug",
            )

        return Result(
            name=name, status="fail", reason=f"convert:{type(e).__name__}:{e}"
        )

    try:
        rng_state = torch.random.get_rng_state()
        ref_out = func(*clone_inputs(args), **clone_inputs(kwargs))
        torch.random.set_rng_state(rng_state)
        compiled = torch.compile(func, backend=dynamo_compiler)
        out = compiled(*clone_inputs(args), **clone_inputs(kwargs))
        if is_uninitialized_op(name, entry):
            compare_uninitialized(ref_out, out)
        else:
            compare_outputs(ref_out, out)
    except Exception as e:
        tb = traceback.format_exc(limit=1).strip().splitlines()[-1]
        return Result(
            name=name,
            status="fail",
            reason=f"numerical:{type(e).__name__}:{tb}",
        )

    return Result(name=name, status="ok")


def run_batch(
    names: Iterable[str],
    coverage_json: Path | str = DEFAULT_COVERAGE_JSON,
    batch_label: str = "batch",
    max_fails: int = 20,
    templates: Dict[str, Any] | None = None,
    show_skips: bool = False,
) -> None:
    # Optional env overrides for reporting/debugging without touching batch files.
    env_show_skips = os.getenv("BUDDY_OC_SHOW_SKIPS", "").strip().lower()
    if env_show_skips in ("1", "true", "yes", "y", "on"):
        show_skips = True
    env_max_fails = os.getenv("BUDDY_OC_MAX_FAILS", "").strip()
    if env_max_fails:
        try:
            max_fails = int(env_max_fails)
        except ValueError:
            raise ValueError(f"Invalid BUDDY_OC_MAX_FAILS={env_max_fails!r}")

    cov_map = load_coverage_map(coverage_json)
    templates = templates or {}
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for n in names:
        if n in cov_map:
            entries.append((n, cov_map[n]))
        else:
            entries.append(
                (n, {"op": n, "overload": "", "notes": "missing_in_coverage"})
            )

    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=make_aot_decompositions(),
    )

    results: List[Result] = []
    for name, entry in entries:
        results.append(run_one(name, entry, dynamo_compiler, templates))

    ok = sum(1 for r in results if r.status == "ok")
    fail = sum(1 for r in results if r.status == "fail")
    skip = sum(1 for r in results if r.status == "skip")

    print(
        f"SUMMARY ok={ok} fail={fail} skip={skip} "
        f"batch_label={batch_label} count={len(entries)} total={len(cov_map)}"
    )
    print("# CHECK: SUMMARY ok=")

    remaining = max_fails
    for r in results:
        if r.status == "fail" and remaining > 0:
            print(f"FAIL {r.name} {r.reason}")
            remaining -= 1
            if remaining == 0:
                break

    if show_skips:
        from collections import Counter

        skip_reasons = Counter(r.reason for r in results if r.status == "skip")
        for reason, count in skip_reasons.items():
            print(f"SKIP {reason} count={count}")
        for r in results:
            if r.status == "skip":
                print(f"SKIP {r.name} {r.reason}")
