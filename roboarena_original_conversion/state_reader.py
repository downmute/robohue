"""Non-executing reader for this source's object-array envelope and numeric lists.

Pickle GLOBAL/REDUCE/BUILD are interpreted as inert symbols. No callable named by
the file is imported or executed. Unsupported operations or structures fail closed.
"""

import math
import pickletools
import zipfile
from dataclasses import dataclass

import numpy as np


@dataclass
class Symbol:
    name: str
    args: object = None
    state: object = None


ALLOWED_GLOBALS = {
    "numpy.core.multiarray _reconstruct",
    "numpy._core.multiarray _reconstruct",
    "numpy ndarray",
    "numpy dtype",
}


def decode_data_pickle(payload):
    stack, memo = [], {}
    marker = object()

    def marked():
        index = next(
            (i for i in range(len(stack) - 1, -1, -1) if stack[i] is marker), None
        )
        if index is None:
            raise ValueError("Malformed source pickle mark")
        items = stack[index + 1 :]
        del stack[index:]
        return items

    for count, (opcode, arg, _) in enumerate(pickletools.genops(payload), 1):
        if count > 2_000_000:
            raise ValueError("Source state exceeds operation limit")
        op = opcode.name
        if op in ("PROTO", "FRAME"):
            continue
        if op == "MARK":
            stack.append(marker)
        elif op in (
            "BININT",
            "BININT1",
            "BININT2",
            "LONG1",
            "LONG4",
            "INT",
            "LONG",
            "BINFLOAT",
            "FLOAT",
            "BINUNICODE",
            "SHORT_BINUNICODE",
            "UNICODE",
            "SHORT_BINBYTES",
            "BINBYTES",
            "BINBYTES8",
        ):
            stack.append(arg)
        elif op in ("NONE", "NEWTRUE", "NEWFALSE"):
            stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[op])
        elif op == "GLOBAL":
            if arg not in ALLOWED_GLOBALS:
                raise ValueError("Unsupported source global: " + str(arg))
            stack.append(Symbol(arg))
        elif op == "STACK_GLOBAL":
            name, module = stack.pop(), stack.pop()
            symbol = str(module) + " " + str(name)
            if symbol not in ALLOWED_GLOBALS:
                raise ValueError("Unsupported source global: " + symbol)
            stack.append(Symbol(symbol))
        elif op in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[int(arg)] = stack[-1]
        elif op == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif op in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo[int(arg)])
        elif op == "EMPTY_LIST":
            stack.append([])
        elif op == "EMPTY_DICT":
            stack.append({})
        elif op == "EMPTY_TUPLE":
            stack.append(())
        elif op == "TUPLE":
            stack.append(tuple(marked()))
        elif op in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(op[-1])
            values = tuple(stack[-n:])
            del stack[-n:]
            stack.append(values)
        elif op == "APPENDS":
            values = marked()
            if type(stack[-1]) is not list:
                raise ValueError("Source append target is not a list")
            stack[-1].extend(values)
        elif op == "APPEND":
            value = stack.pop()
            if type(stack[-1]) is not list:
                raise ValueError("Source append target is not a list")
            stack[-1].append(value)
        elif op == "SETITEMS":
            values = marked()
            if type(stack[-1]) is not dict or len(values) % 2:
                raise ValueError("Malformed source state dictionary")
            stack[-1].update(zip(values[::2], values[1::2]))
        elif op == "SETITEM":
            value, key = stack.pop(), stack.pop()
            if type(stack[-1]) is not dict:
                raise ValueError("Source item target is not a dictionary")
            stack[-1][key] = value
        elif op == "REDUCE":
            args, symbol = stack.pop(), stack.pop()
            if not isinstance(symbol, Symbol) or symbol.name not in ALLOWED_GLOBALS:
                raise ValueError("Unsupported source constructor")
            stack.append(Symbol(symbol.name, args=args))
        elif op == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Symbol):
                raise ValueError("Unsupported source object state")
            stack[-1].state = state
        elif op == "STOP":
            if len(stack) != 1 or not isinstance(stack[0], Symbol):
                raise ValueError("Malformed source array envelope")
            root = stack[0]
            if root.name not in {
                "numpy.core.multiarray _reconstruct",
                "numpy._core.multiarray _reconstruct",
            }:
                raise ValueError("Expected source ndarray envelope")
            if (
                not isinstance(root.state, tuple)
                or len(root.state) != 5
                or type(root.state[-1]) is not list
            ):
                raise ValueError("Unsupported ndarray state")
            return root.state[-1]
        else:
            raise ValueError("Unsupported source pickle opcode: " + op)
    raise ValueError("Incomplete source object array")


def read_source_state(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["data.npy"]:
            raise ValueError(
                "Expected source data.npy array; no fields silently dropped"
            )
        info = archive.getinfo("data.npy")
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError("Source state array exceeds supported size")
        with archive.open("data.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Unsupported source NPY version")
            if len(shape) != 1 or not dtype.hasobject:
                raise ValueError(
                    "Expected one-dimensional original source object array"
                )
            rows = decode_data_pickle(stream.read())
    if len(rows) != shape[0] or not rows or any(type(r) is not dict for r in rows):
        raise ValueError("Source state row count/type does not match its header")
    return rows


def inspect_state(path):
    rows = read_source_state(path)
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Source state fields vary; mapping requires review")
    shapes = {}
    nonfinite = 0
    for key in sorted(keys):
        first = rows[0][key]
        if type(first) is not list:
            raise ValueError("Expected source numeric lists: " + key)
        width = len(first)
        for row in rows:
            values = row[key]
            if (
                type(values) is not list
                or len(values) != width
                or any(type(v) not in (int, float) for v in values)
            ):
                raise ValueError("Source field shape/type varies: " + key)
            nonfinite += sum(not math.isfinite(v) for v in values)
        shapes[key] = [len(rows), width]
    return {
        "array_key": "data",
        "sample_count": len(rows),
        "field_shapes": shapes,
        "nonfinite_value_count": nonfinite,
        "reader": "state_reader.read_source_state",
        "physical_timestamps_supplied": False,
        "timebase": "source_state_sample_index",
    }
