#%%bash
export BUCKET="your-gcs-bucket-name"

# prefix up to the parent folder containing grid org (no folders for grid axis values)
export INPUT_PREFIX="input_prefix/"
export POLYGON_GEOJSON_BLOB="path/to/your/polygons.geojson"

# Root file for outputs (folder where data check script is run)
export DATA_ROOT="data-root"

export METADATA_SUBREGIONS="path/to/your/subregions.geojson"

# local colab dir for intermediate files (will be deleted at the end of the run)
export WORK_DIR="/content/tile_work"
export MAX_WORKERS="2"

# export TEST_LIMIT="10"

echo "Environment set:"
echo "  BUCKET             = $BUCKET"
echo "  INPUT_PREFIX       = $INPUT_PREFIX"
echo "  POLYGON_GEOJSON_BLOB = $POLYGON_GEOJSON_BLOB"
echo "  DATA_ROOT          = $DATA_ROOT"
echo "  METADATA_SUBREGIONS= $METADATA_SUBREGIONS"
echo "  WORK_DIR           = $WORK_DIR"
echo "  MAX_WORKERS        = $MAX_WORKERS"
echo "  TEST_LIMIT         = ${TEST_LIMIT:-'(not set — full run)'}"