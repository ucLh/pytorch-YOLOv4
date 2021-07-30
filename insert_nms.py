import argparse
import onnx_graphsurgeon as gs
import onnx
import numpy as np
import sys


def append_nms(graph, num_classes, scoreThreshold, iouThreshold, keepTopK):
    out_tensors = graph.outputs
    bs = out_tensors[0].shape[0]
    print(bs)

    nms_attrs_effdet = {'shareLocation': True,
                 'backgroundLabelId': -1,
                 'numClasses': num_classes,
                 'topK': 1024,
                 'keepTopK': keepTopK,
                 'scoreThreshold': scoreThreshold,
                 'iouThreshold': iouThreshold,
                 'isNormalized': False,
                 'clipBoxes': False}

    nms_attrs_yolo = {'shareLocation': True,
                 'backgroundLabelId': -1,
                 'numClasses': num_classes,
                 'topK': 1024,
                 'keepTopK': keepTopK,
                 'scoreThreshold': scoreThreshold,
                 'iouThreshold': iouThreshold,
                 'isNormalized': True,
                 'clipBoxes': True}

    nms_attrs = nms_attrs_yolo

    nms_num_detections = gs.Variable(name="nms_num_detections", dtype=np.int32, shape=(bs, 1))
    nms_boxes = gs.Variable(name="nms_boxes", dtype=np.float32, shape=(bs, keepTopK, 4))
    nms_scores = gs.Variable(name="nms_scores", dtype=np.float32, shape=(bs, keepTopK))
    nms_classes = gs.Variable(name="nms_classes", dtype=np.float32, shape=(bs, keepTopK))

    nms = gs.Node(op="BatchedNMS_TRT", attrs=nms_attrs, inputs=out_tensors, outputs=[nms_num_detections, nms_boxes, nms_scores, nms_classes])
    graph.nodes.append(nms)
    graph.outputs = [nms_num_detections, nms_boxes, nms_scores, nms_classes]

    return graph


def add_nms_to_onnx(model_file, num_classes, confidenceThreshold=0.4, nmsThreshold=0.6, keepTopK=100, opset=11):
    graph = gs.import_onnx(onnx.load(model_file))
    
    graph = append_nms(graph, num_classes, confidenceThreshold, nmsThreshold, keepTopK)
    
    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort().fold_constants().cleanup()

    # Export the onnx graph from graphsurgeon
    out_name = model_file[:-5]+'_nms.onnx'
    onnx.save_model(gs.export_onnx(graph), out_name)

    print("Saving the ONNX model to {}".format(out_name))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='Name of a model',
                        default='yolov4_1_3_608_608_static.onnx')
    parser.add_argument('--conf_thresh', type=float,
                        help='Confidence threshold',
                        default=0.01)
    parser.add_argument('--nms_thresh', type=float,
                        help='NMS threshold',
                        default=0.3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    add_nms_to_onnx(args.model_name, 80, confidenceThreshold=args.conf_thresh,
                    nmsThreshold=args.nms_thresh, keepTopK=100, opset=11)
    # add_nms_to_onnx(model_file, 90, confidenceThreshold=0.1, nmsThreshold=0.4, keepTopK=100, opset=11)