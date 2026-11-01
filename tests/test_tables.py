"""tables.save writes what it's given, one sheet per table."""

import numpy as np
from openpyxl import load_workbook

import tables


def test_save_writes_one_sheet_per_table(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "RESULTS_DIR", str(tmp_path))
    rows = [{"Advisor": "Tax", "t": np.float64(1.23456), "Approved": np.int64(5)}]
    path = tables.save("demo", {"First": rows, "Second": []})
    book = load_workbook(path)
    assert book.sheetnames == ["First", "Second"]
    assert [c.value for c in book["First"][2]] == ["Tax", 1.2346, 5]
