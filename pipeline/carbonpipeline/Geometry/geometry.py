from enum import Enum
from typing import Tuple


class GeometryType(Enum):
    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"
    UNKNOWN = "Unknown"


class Geometry:
    def __init__(self, data = None):
        self.original_depth = self._get_depth(data)
        self.type_signature = self._get_type_signature(data)
        self.geom_type = self._infer_geom_type(self.original_depth)  # infer before flatten

        self.data, self.flattened_depth = self._flatten_to_max3(data)  
        self.rect_region: list[float] | list[int] = [] # List of [float] || [int]

    def validate_coordinates(self):
       self._validation(self.data)

    def _validation(self, data):
        if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
            if len(data) != 2:
                raise ValueError(f"Invalid coordinate pair. Expected 2 elements but received {len(data)}")
            return
        elif isinstance(data, list):
            for item in data:
                self._validation(item)
        else:
            raise TypeError(f"Invalid element in coordinate structure: {self.data}")
    
    def _flatten_to_max3(self, lst) -> Tuple[list, int]:
        depth = self._get_depth(lst)
        result = lst
        
        # Don't flatten MultiPolygon structures
        if depth == 4 and self.geom_type == GeometryType.MULTIPOLYGON:
            return result, depth
        
        # Don't flatten Polygon structures  
        if depth == 3 and self.geom_type == GeometryType.POLYGON:
            return result, depth
        
        # Only flatten if we have excessive depth (>4) and it's not a recognized geometry type
        while depth > 4:
            result = [item for sub in result for item in sub]
            depth -= 1
        return result, depth

    def _get_depth(self, lst):
        if isinstance(lst, list) and lst:
            return 1 + max(self._get_depth(item) for item in lst)
        if isinstance(lst, list):
            return 1
        else:
            return 0

    def _get_type_signature(self, lst):
        if isinstance(lst, list):
            if lst:
                return "[" + self._get_type_signature(lst[0]) + "]"
            else:
                return "[?]"
        else:
            return type(lst).__name__

    @staticmethod
    def _infer_geom_type(depth):
        if depth == 1:
            return GeometryType.POINT
        elif depth == 2:
            return GeometryType.LINESTRING
        elif depth == 3:
            return GeometryType.POLYGON
        elif depth == 4:
            return GeometryType.MULTIPOLYGON
        else:
            return GeometryType.UNKNOWN
        
    def __repr__(self):
        return (f"Geometry={self.geom_type}, "
                f"original_depth={self.original_depth}, "
                f"flattened_depth={self.flattened_depth}, "
                f"signature={self.type_signature}")
