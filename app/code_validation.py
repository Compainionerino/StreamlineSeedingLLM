from __future__ import annotations

import ast
from dataclasses import dataclass


ALLOWED_IMPORT_ROOTS = {"vtk", "vtkmodules", "math", "numpy", "typing"}
FORBIDDEN_IMPORT_PREFIXES = {
    "vtkmodules.qt",
}
FORBIDDEN_IMPORT_ROOTS = {
    "PySide6",
    "PyQt5",
    "PyQt6",
    "tkinter",
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "pathlib",
    "shutil",
    "sys",
    "builtins",
    "importlib",
}
FORBIDDEN_CALL_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
}
FORBIDDEN_ATTR_CALLS = {
    "Start",
    "Initialize",
    "Render",
    "show",
    "Write",
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "startfile",
    "urlopen",
    "request",
}
FORBIDDEN_VTK_CLASSES = {
    "QApplication",
    "QWidget",
    "QVTKRenderWindowInteractor",
    "vtkRenderWindow",
    "vtkRenderWindowInteractor",
    "vtkXRenderWindowInteractor",
    "vtkWin32RenderWindowInteractor",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def _import_root(module_name: str | None) -> str:
    if not module_name:
        return ""
    return module_name.split(".", 1)[0]


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_forbidden_import(module_name: str | None) -> bool:
    if not module_name:
        return False
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def validate_generated_code(code: str) -> ValidationResult:
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(False, [f"Syntax error: {exc}"])

    has_create_visualization = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "create_visualization":
            has_create_visualization = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _import_root(alias.name)
                if _is_forbidden_import(alias.name):
                    errors.append(f"Forbidden import: {alias.name}")
                elif root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(f"Forbidden import: {alias.name}")
                elif root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Import is not allowed: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            root = _import_root(node.module)
            if _is_forbidden_import(node.module):
                errors.append(f"Forbidden import: {node.module}")
            elif root in FORBIDDEN_IMPORT_ROOTS:
                errors.append(f"Forbidden import: {node.module}")
            elif root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"Import is not allowed: {node.module}")

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_CALL_NAMES:
                errors.append(f"Forbidden call: {call_name}()")
            if call_name in FORBIDDEN_ATTR_CALLS:
                errors.append(f"Forbidden method call: {call_name}()")
            if call_name in FORBIDDEN_VTK_CLASSES:
                errors.append(f"Generated code must not create {call_name}.")

    if not has_create_visualization:
        errors.append("Generated code must define create_visualization(dataset_path, metadata, user_request).")

    return ValidationResult(ok=not errors, errors=errors)


def ensure_valid_code(code: str) -> None:
    result = validate_generated_code(code)
    if not result.ok:
        raise ValueError("Generated code failed validation:\n" + "\n".join(result.errors))
