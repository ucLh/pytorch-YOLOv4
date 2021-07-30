cfgFile=cfg/yolov4.cfg
imageFile=data/dog.jpg
if [ ! -z "$1" ]
  then
    weightFile="$1"
  else
    echo "You must provide weight file as the first argument"
    exit 1
fi
if [ ! -z "$2" ]
  then
    batchSize="$2"
  else
    echo "You must provide batch size as the second argument"
    exit 1
fi
# Convert to onnx
python3 demo_darknet2onnx.py ${cfgFile} ${weightFile} ${imageFile} ${batchSize}

# Simplify
outName=yolov4_${batchSize}_3_608_608_static.onnx
python3 -m onnxsim ${outName} ${outName}

# Append NMS
confThresh=0.01
NMSThresh=0.3
python3 insert_nms.py --model_name ${outName} --conf_thresh ${confThresh} --nms_thresh ${NMSThresh}
rm ${outName}