"""Save result tables as Excel files, which are easier to read than wide terminal output
(the owner's request, 1 Nov). Every script that prints a table also calls save().

    tables.save("compare_gbm_groups", {"Adoption test": rows})   # rows: a list of dicts

Writes results/<name>.xlsx (git-ignored; rerunning replaces it): one sheet per table, the
header row frozen, filters on, columns sized to fit. Returns the path.
"""

import os

from openpyxl import Workbook
from openpyxl.utils import get_column_letter

import validity

RESULTS_DIR = os.path.join(validity.PS2_ROOT, "results")


def _cell(value):
    """NumPy numbers become plain ones (openpyxl can't write them); floats keep 4 decimals."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, list | tuple):  # e.g. a margin per advisor
        value = ", ".join(map(str, value))
    return round(value, 4) if isinstance(value, float) else value


def save(name, sheets):
    """{sheet title: [row dict, ...]} -> results/<name>.xlsx. Returns the path."""
    book = Workbook()
    book.remove(book.active)
    for title, rows in sheets.items():
        sheet = book.create_sheet(title[:31])  # Excel's limit
        header = list(rows[0]) if rows else []
        sheet.append(header)
        for row in rows:
            sheet.append([_cell(row[key]) for key in header])
        sheet.freeze_panes = "A2"
        if rows:
            sheet.auto_filter.ref = sheet.dimensions
        for i, key in enumerate(header, 1):
            width = max(len(str(key)), *(len(str(_cell(row[key]))) for row in rows))
            sheet.column_dimensions[get_column_letter(i)].width = min(width + 2, 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.xlsx")
    book.save(path)
    return path
