from typing import List, Union
from .geometry import Geometry, GeometryType

Coord = List[float]          # [lon, lat] ou [lat, lon]
Ring = List[Coord]           # [[..., ...], ...]
Polygon = List[Ring]         # [outer_ring, hole1, ...]
MultiPolygon = List[Polygon] # [[ring...], [ring...]], ...

class GeometryProcessor:
    @staticmethod
    def process_geometry(geometry: Geometry) -> List[float]:
        """
        Returns a rectangular region [N, W, S, E] (ERA5 style).

        - POINT       -> [N, W, S, E] based on a small offset around the point
        - POLYGON     -> [N, W, S, E] bounding box of the outer ring
        - MULTIPOLYGON -> [N, W, S, E] bounding box covering all polygons
        """
        match geometry.geom_type:
            case GeometryType.POINT:
                region = GeometryProcessor._get_point_outer_bounds(geometry.data)

            case GeometryType.POLYGON:
                region = GeometryProcessor._get_rect_region_covering_polygon(geometry.data)

            case GeometryType.MULTIPOLYGON:
                region = GeometryProcessor._get_polys_region(geometry.data)

            case GeometryType.UNKNOWN:
                raise TypeError("Unsupported geometry depth/type.")

        # Validation
        if len(region) != 4:
            raise ValueError("Bounding box must have length 4 [N, W, S, E].")

        return region

    # --------------------------
    # Helpers
    # --------------------------

    @staticmethod
    def _infer_lonlat_indices(ring: Ring) -> tuple[int, int]:
        """
        Determines if coordinates are [lon, lat] (GeoJSON standard)
        or [lat, lon].

        Heuristic:
        - If all first values are within [-180, 180] and second values within [-90, 90],
          assume [lon, lat].
        - Otherwise assume [lat, lon].
        """
        def looks_like_lon_lat(pt: Coord) -> bool:
            return len(pt) >= 2 and abs(pt[0]) <= 180 and abs(pt[1]) <= 90

        if ring and all(isinstance(p, list) and len(p) >= 2 for p in ring):
            return (0, 1) if all(looks_like_lon_lat(p) for p in ring) else (1, 0)
        raise ValueError("Ring malformed: expected list of [x, y] coordinates.")

    @staticmethod
    def _normalize_outer_ring(poly_or_ring: Union[Ring, Polygon]) -> Ring:
        """
        Accepts:
          - A ring: [[x, y], ...]
          - A GeoJSON polygon: [[[x, y], ...], [hole1], ...]

        Returns the outer ring only.
        """
        if not poly_or_ring:
            raise ValueError("Empty polygon/ring.")

        first = poly_or_ring[0]
        # If first element is a point (numeric), we have a ring
        if isinstance(first, list) and len(first) >= 2 and not isinstance(first[0], list):
            ring = poly_or_ring  # type: ignore[assignment]
        else:
            # Otherwise, it’s a polygon (list of rings) -> take outer ring
            ring = poly_or_ring[0]  # type: ignore[index]

        if not ring or not isinstance(ring[0], list) or len(ring[0]) < 2:
            raise ValueError("Outer ring malformed.")

        return ring  # type: ignore[return-value]

    @staticmethod
    def _get_point_outer_bounds(point: List[float]) -> List[float]:
        """
        Build a small rectangle around a point.

        Returns [N, W, S, E] where:
          - N/S = lat +/- offset
          - W/E = lon +/- offset
        """
        print(point)
        if not isinstance(point, list) or len(point) < 2:
            raise ValueError("Point malformed: expected [lat, lon] or [lon, lat].")

        lat, lon = point[0], point[1]

        offset = 0.125  # degrees
        N = lat + offset
        S = lat - offset
        W = lon - offset
        E = lon + offset
        return [N, W, S, E]

    @staticmethod
    def _ensure_min_bbox_size(region: List[float], min_delta: float = 0.251) -> List[float]:
        """
        Ensure that a bounding box [N, W, S, E] has at least `min_delta`
        degrees difference in both latitude and longitude.

        If the span is smaller, the box is expanded symmetrically around its center
        to meet the minimum required size.
        """
        N, W, S, E = region
        lat_delta = abs(N - S)
        lon_delta = abs(E - W)

        # Adjust latitude span
        if lat_delta < min_delta:
            center_lat = (N + S) / 2
            half = min_delta / 2
            N = center_lat + half
            S = center_lat - half

        # Adjust longitude span
        if lon_delta < min_delta:
            center_lon = (E + W) / 2
            half = min_delta / 2
            E = center_lon + half
            W = center_lon - half

        return [N, W, S, E]

    @staticmethod
    def _get_rect_region_covering_polygon(poly_or_ring: Union[Ring, Polygon]) -> List[float]:
        """
        Compute the ERA5 bounding box [N, W, S, E] for a Polygon.
        Accepts either a ring or a GeoJSON polygon (outer ring is used).
        """
        ring = GeometryProcessor._normalize_outer_ring(poly_or_ring)
        lon_i, lat_i = GeometryProcessor._infer_lonlat_indices(ring)

        lons = [p[lon_i] for p in ring]
        lats = [p[lat_i] for p in ring]

        N = max(lats)
        S = min(lats)
        W = min(lons)
        E = max(lons)

        region = [N, W, S, E]

        return GeometryProcessor._ensure_min_bbox_size(region)

    @staticmethod
    def _get_polys_region(polys: MultiPolygon) -> List[float]:
        """
        Compute a single ERA5 bounding box [N, W, S, E] that covers
        all polygons in a MultiPolygon.
        Each polygon is represented as [outer_ring, holes...].
        """
        if not isinstance(polys, list) or not polys:
            raise ValueError("MultiPolygon malformed or empty.")

        all_lons = []
        all_lats = []

        for poly in polys:
            ring = GeometryProcessor._normalize_outer_ring(poly)
            lon_i, lat_i = GeometryProcessor._infer_lonlat_indices(ring)

            lons = [p[lon_i] for p in ring]
            lats = [p[lat_i] for p in ring]

            all_lons.extend(lons)
            all_lats.extend(lats)

        N = max(all_lats)
        S = min(all_lats)
        W = min(all_lons)
        E = max(all_lons)

        region = [N, W, S, E]
        return GeometryProcessor._ensure_min_bbox_size(region)
