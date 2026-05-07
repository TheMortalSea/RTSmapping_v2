

Working in progress



# Post inference processing


## Vectorization:

Feedback: rasterio.features.shapes or gdal_polygonize produces "stair-step" polygons. Apply a simplification algorithm (e.g., Douglas-Peucker) during vectorization to reduce file size and make the map look cleaner.


## Merging strategy: Maximum

For pixels covered by multiple tiles, take the maximum probability:
```
P_merged(x, y) = max(P_tile1(x, y), P_tile2(x, y), ...)
```

**Rationale**: Consistent with the detection philosophy—if any tile view detects RTS, include it.



## Scale Fusion

Combine predictions across scales using **pixel-wise maximum**:

```
P_final = max(P_1.0, P_0.5)
```

**Rationale**: If any scale confidently detects RTS, include it. Maximum operation is conservative toward detection while individual scale thresholds control precision.


## Prediction Merging

### Overlap Handling

Adjacent (4-neighbours) tiles overlap by 50%. The overlapping regions have multiple predictions that must be merged.
- can adjust to 25% overlap if computation resource is limited (estimate total tile numbers and gpu-hour for both 25% and 50%)


### Merging Procedure

**Option A: On-the-fly merging (memory-efficient)**
1. Create output raster for region with NoData fill
2. For each tile prediction:
   - Read overlapping region from output
   - Compute pixel-wise maximum with new prediction
   - Write merged result back
3. Advantage: Low memory; Disadvantage: Many I/O operations

**Option B: Batch merging (faster)**
1. Accumulate all tile predictions in memory (or memory-mapped file)
2. Apply reduction (maximum) across overlapping tiles
3. Write final merged raster
4. Advantage: Faster; Disadvantage: High memory for large regions

**Recommendation**: Use Option B for manageable regions (e.g., per Arctic subregion), Option A for full pan-arctic if memory-constrained.

### Area and Perimeter Calculation

EPSG:3857 distorts areas at high latitudes (~13x inflation at 74°N). All area and perimeter measurements must use **geodesic calculations** (e.g., `pyproj.Geod.geometry_area_perimeter`) or reproject to a local equal-area CRS — never compute directly from EPSG:3857 coordinates.

### Output Chunking

For pan-arctic scale, produce merged outputs per region rather than single global raster:
- Easier to manage and distribute
- Enables parallel processing
- Allows region-specific quality control

