from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_VTK_EXTENSIONS = {".vtk", ".vti", ".vtu", ".vtp", ".vts", ".vtr"}


class DatasetMetadataError(RuntimeError):
    """Raised when VTK dataset metadata cannot be extracted."""


@dataclass(frozen=True)
class ArrayMetadata:
    name: str
    association: str
    components: int
    tuples: int
    data_type: str
    component_ranges: list[tuple[float, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetMetadata:
    path: str
    file_name: str
    file_extension: str
    dataset_class: str
    dataset_type: str
    bounds: tuple[float, float, float, float, float, float]
    dimensions: tuple[int, int, int] | None
    point_count: int
    cell_count: int
    point_arrays: list[ArrayMetadata]
    cell_arrays: list[ArrayMetadata]
    active_point_scalars: str | None
    active_point_vectors: str | None
    active_cell_scalars: str | None
    active_cell_vectors: str | None
    hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_vtk() -> Any:
    try:
        import vtk  # type: ignore
    except ImportError as exc:
        raise DatasetMetadataError(
            "VTK is required for dataset metadata extraction. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return vtk


def _reader_for_path(path: Path) -> Any:
    vtk = _require_vtk()
    suffix = path.suffix.lower()
    readers = {
        ".vtk": vtk.vtkDataSetReader,
        ".vti": vtk.vtkXMLImageDataReader,
        ".vtu": vtk.vtkXMLUnstructuredGridReader,
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".vts": vtk.vtkXMLStructuredGridReader,
        ".vtr": vtk.vtkXMLRectilinearGridReader,
    }
    if suffix not in readers:
        raise DatasetMetadataError(
            f"Unsupported dataset extension {suffix!r}. Supported: "
            + ", ".join(sorted(SUPPORTED_VTK_EXTENSIONS))
        )
    reader = readers[suffix]()
    reader.SetFileName(str(path))
    return reader


def read_vtk_dataset(path: str | Path) -> Any:
    source = Path(path)
    if not source.exists():
        raise DatasetMetadataError(f"Dataset does not exist: {source}")
    reader = _reader_for_path(source)
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None:
        raise DatasetMetadataError(f"VTK reader returned no dataset for {source}")
    return dataset


def _active_name(attributes: Any, getter: str) -> str | None:
    array = getattr(attributes, getter)()
    if array is None:
        return None
    name = array.GetName()
    return name if name else None


def _array_metadata(attributes: Any, association: str) -> list[ArrayMetadata]:
    arrays: list[ArrayMetadata] = []
    for index in range(attributes.GetNumberOfArrays()):
        array = attributes.GetArray(index)
        if array is None:
            continue
        name = array.GetName() or f"unnamed_{association}_{index}"
        components = int(array.GetNumberOfComponents())
        component_ranges: list[tuple[float, float]] = []
        for component in range(components):
            try:
                value_range = array.GetRange(component)
            except Exception:
                value_range = (0.0, 0.0)
            component_ranges.append((float(value_range[0]), float(value_range[1])))
        arrays.append(
            ArrayMetadata(
                name=name,
                association=association,
                components=components,
                tuples=int(array.GetNumberOfTuples()),
                data_type=str(array.GetDataTypeAsString()),
                component_ranges=component_ranges,
            )
        )
    return arrays


def _dimensions(dataset: Any) -> tuple[int, int, int] | None:
    if not hasattr(dataset, "GetDimensions"):
        return None
    dimensions = dataset.GetDimensions()
    if not dimensions:
        return None
    return tuple(int(value) for value in dimensions)


def _spatial_dimension(bounds: tuple[float, float, float, float, float, float]) -> int:
    extents = [
        abs(bounds[1] - bounds[0]),
        abs(bounds[3] - bounds[2]),
        abs(bounds[5] - bounds[4]),
    ]
    return sum(1 for extent in extents if extent > 1e-12)


def _has_component_count(arrays: list[ArrayMetadata], count: int) -> bool:
    return any(array.components == count for array in arrays)


def _infer_hints(
    *,
    dataset_class: str,
    dimensions: tuple[int, int, int] | None,
    bounds: tuple[float, float, float, float, float, float],
    point_arrays: list[ArrayMetadata],
    cell_arrays: list[ArrayMetadata],
) -> list[str]:
    hints: set[str] = set()
    all_arrays = [*point_arrays, *cell_arrays]

    if _has_component_count(all_arrays, 1):
        hints.add("scalar_field")
    if any(array.components in {2, 3} for array in all_arrays):
        hints.add("vector_field")

    if "PolyData" in dataset_class:
        hints.add("surface")
    if "UnstructuredGrid" in dataset_class:
        hints.add("unstructured_grid")
    if "ImageData" in dataset_class or "StructuredGrid" in dataset_class or "RectilinearGrid" in dataset_class:
        if dimensions and sum(1 for value in dimensions if value > 1) >= 3:
            hints.add("volume")

    spatial_dim = _spatial_dimension(bounds)
    if spatial_dim >= 3:
        hints.add("3d")
    elif spatial_dim == 2:
        hints.add("2d")

    if "volume" in hints:
        hints.add("3d")

    return sorted(hints)


def extract_dataset_metadata(path: str | Path) -> DatasetMetadata:
    source = Path(path)
    dataset = read_vtk_dataset(source)
    point_data = dataset.GetPointData()
    cell_data = dataset.GetCellData()
    point_arrays = _array_metadata(point_data, "point")
    cell_arrays = _array_metadata(cell_data, "cell")
    bounds = tuple(float(value) for value in dataset.GetBounds())
    dimensions = _dimensions(dataset)
    dataset_class = dataset.GetClassName()

    return DatasetMetadata(
        path=str(source.resolve()),
        file_name=source.name,
        file_extension=source.suffix.lower(),
        dataset_class=dataset_class,
        dataset_type=dataset_class.replace("vtk", ""),
        bounds=bounds,  # type: ignore[arg-type]
        dimensions=dimensions,
        point_count=int(dataset.GetNumberOfPoints()),
        cell_count=int(dataset.GetNumberOfCells()),
        point_arrays=point_arrays,
        cell_arrays=cell_arrays,
        active_point_scalars=_active_name(point_data, "GetScalars"),
        active_point_vectors=_active_name(point_data, "GetVectors"),
        active_cell_scalars=_active_name(cell_data, "GetScalars"),
        active_cell_vectors=_active_name(cell_data, "GetVectors"),
        hints=_infer_hints(
            dataset_class=dataset_class,
            dimensions=dimensions,
            bounds=bounds,  # type: ignore[arg-type]
            point_arrays=point_arrays,
            cell_arrays=cell_arrays,
        ),
    )


def format_metadata_summary(metadata: DatasetMetadata | dict[str, Any]) -> str:
    payload = metadata.to_dict() if isinstance(metadata, DatasetMetadata) else metadata
    lines = [
        f"File: {payload.get('file_name')}",
        f"Dataset: {payload.get('dataset_type')} ({payload.get('dataset_class')})",
        f"Points/cells: {payload.get('point_count')} / {payload.get('cell_count')}",
        f"Dimensions: {payload.get('dimensions')}",
        f"Bounds: {payload.get('bounds')}",
        f"Hints: {', '.join(payload.get('hints') or [])}",
    ]

    for label, arrays in (("Point arrays", payload.get("point_arrays")), ("Cell arrays", payload.get("cell_arrays"))):
        if not arrays:
            continue
        parts = []
        for item in arrays:
            if isinstance(item, dict):
                parts.append(f"{item.get('name')}[{item.get('components')}]")
            else:
                parts.append(f"{item.name}[{item.components}]")
        lines.append(f"{label}: " + ", ".join(parts))

    return "\n".join(lines)

