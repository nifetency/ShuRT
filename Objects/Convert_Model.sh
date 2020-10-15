#!/bin/bash
#SPDX-License-Identifier: Apache-2.0

#Set below Model Parameter
export ONNX_MODEL_NAME=$1
#-------
source /opt/intel/openvino/bin/setupvars.sh

export MODEL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MODEL_OPT_DIR=$INTEL_CVSDK_DIR/deployment_tools
export ONNX_PATH=$MODEL_DIR/$ONNX_MODEL_NAME
export OUTPUT_DIR=$MODEL_DIR

#python3 $MODEL_OPT_DIR/model_optimizer/mo.py  --data_type FP32 --input_model $ONNX_PATH --output_dir $OUTPUT_DIR/FP32
python3 $MODEL_OPT_DIR/model_optimizer/mo.py  --data_type FP16 --input_model $ONNX_PATH --output_dir $OUTPUT_DIR/FP16
