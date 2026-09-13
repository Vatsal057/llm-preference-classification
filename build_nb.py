"""Convert notebook_src.py (jupytext percent format) into a Kaggle-ready .ipynb."""
import json
import sys

SRC = "notebook_src.py"
OUT = "lmsys_revision_v2.ipynb"


def parse(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = buf[:]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            return
        if kind == "markdown":
            body = [l[2:] if l.startswith("# ") else (l[1:] if l == "#" else l) for l in body]
        cells.append((kind, "\n".join(body)))

    for line in lines:
        s = line.strip()
        if s.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in s else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def build(cells):
    nb_cells = []
    for kind, src in cells:
        c = {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if kind == "code":
            c["execution_count"] = None
            c["outputs"] = []
        nb_cells.append(c)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    cells = parse(SRC)
    code = "\n\n".join(s for k, s in cells if k == "code")
    compile(code, "<notebook>", "exec")  # hard syntax gate
    nb = build(cells)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"{OUT}: {len(cells)} cells "
          f"({sum(1 for k,_ in cells if k=='code')} code, "
          f"{sum(1 for k,_ in cells if k=='markdown')} markdown), "
          f"{len(code.splitlines())} lines of code -- syntax OK")
