from tritonclient.utils import *
import tritonclient.http as httpclient

import argparse
import numpy as np
import os
import sys
import time
from uuid import uuid4
import glob

from monai.apps.utils import download_and_extract
from monai.utils.type_conversion import convert_to_numpy

model_name = "monai_bdc"

if __name__ == "__main__":
    with httpclient.InferenceServerClient("localhost:8000") as client:
        image_bytes = b""
        jpeg_file = "/home/ubuntu/monai/bdc/sample_data/A/sample_A1.jpg"
        with open(jpeg_file, "rb") as f:
            image_bytes = f.read()

        image_data = np.array([image_bytes], dtype=np.bytes_)
        inputs = [
            httpclient.InferInput("IMAGE", image_data.shape, np_to_triton_dtype(image_data.dtype))
        ]
        inputs[0].set_data_from_numpy(image_data)
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT"),
        ]
        inference_start_time = time.time() * 1000
        response = client.infer(
            model_name,
            inputs,
            request_id=str(uuid4().hex),
            outputs=outputs,
        )
        inference_time = time.time() * 1000 - inference_start_time
        result = response.get_response()
        print(
            "Classification result for `{}`: {}.  (Inference time: {:6.0f} ms)".format(
                jpeg_file,
                response.as_numpy("OUTPUT").astype(str)[0],
                inference_time,
            )
        )
