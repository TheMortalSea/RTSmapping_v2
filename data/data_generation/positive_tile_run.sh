#!/bin/bash
# runs the positive tile generation process, this serves as a template or placeholder for paths

export BUCKET="my-bucket"
export INPUT_PREFIX="tiles/input/"
export OUTPUT_PREFIX="tiles/output/"
export POSITIVE_GEOJSON_BLOB="labels/positive.geojson"
export IGNRORE_GEOJSON_BLOB="labels/ignore.geojson"
export WORK_DIR="/content/work"   #this is the colab working directory chnge as needed
export MAX_WORKERS=4
# export TEST_LIMIT=10    # uncomment to test with a small batch

python "$(dirname "$0")/reprocess.py"