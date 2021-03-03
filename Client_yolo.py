import argparse
import cv2
import time
import numpy as np
import logging
from multiprocessing import Process
from gabriel_protocol import gabriel_pb2
from gabriel_client.websocket_client import WebsocketClient
from gabriel_client.websocket_client import ProducerWrapper
from gabriel_client.opencv_adapter import OpencvAdapter

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

DEFAULT_SOURCE_NAME = 'roundtrip123'


def preprocess(frame):
    return frame


def produce_extras():
    return None


def consume_frame(frame, _):
    cv2.imshow('Image from server', frame)
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_name', nargs='?', default=DEFAULT_SOURCE_NAME)
    args = parser.parse_args()

    # cv2.namedWindow("Window")
    capture = cv2.VideoCapture(0)
    # cv2.namedWindow("Window")

    # while True:
    #   ret, frame = capture.read()
    #   cv2.imshow("Window", frame)

    opencv_adapter = OpencvAdapter(preprocess, produce_extras, consume_frame, capture, args.source_name)

    client = WebsocketClient(args.source_name, 9099, opencv_adapter.get_producer_wrappers(), opencv_adapter.consumer)
    client.launch()

    # This breaks on 'q' key
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #  break

    # video_capture.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()