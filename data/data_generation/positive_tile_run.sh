# this is meant to run in a colab environment
%%bash
export BUCKET="bucket"
export INPUT_PREFIX="tiles/input/"
export OUTPUT_PREFIX="tiles/output/"
export POSITIVE_GEOJSON_BLOB="training_labels"
export IGNRORE_GEOJSON_BLOB="ignore_labels"
export WORK_DIR="/content/work"
export MAX_WORKERS=4
# export TEST_LIMIT=10

python /content/RTSmapping_v2/data/data_generation/positive_tile_creation.py