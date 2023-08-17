import os
import sys
import json
from tempfile import NamedTemporaryFile

import torch
import numpy as np
import monai

from monai.data import DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, Activationsd,ToTensor, Lambda
from monai.inferers.inferer import SimpleInferer
from monai.networks.nets import TorchVisionFCModel

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        model_path = "/mnt/pytorch/model.pt"
        self.inference_device = torch.device("cuda:0")
        self.preprocessing = Compose(
            [
                LoadImaged(keys="image", image_only=True),
                EnsureChannelFirstd(keys="image", channel_dim=2),
                ScaleIntensityd(keys="image"),
                Resized(keys="image", spatial_size=[299, 299])
            ]
        )
        self.postprocessing = Compose(
            [
                Activationsd(keys="pred", sigmoid=True)
            ]
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network_def = TorchVisionFCModel(model_name='inception_v3', num_classes=4, pool=None, use_conv=False, bias=True, pretrained=True)
        self.network = network_def.to(self.device)
        self.inferer = SimpleInferer()
        state_dict = torch.load(model_path)
        self.network.load_state_dict(state_dict, strict=True)

    def execute(self, requests):
        responses = []
        print("#############################starting request #########################")
        for request in requests:
            # get the input by name (as configured in config.pbtxt)
            image = pb_utils.get_input_tensor_by_name(request, "IMAGE")
            label = pb_utils.get_input_tensor_by_name(request, "LABEL")
            label_text = label.as_numpy().astype(np.bytes_).tobytes().decode()
            print("Label: " + label_text)
            tmpFile = NamedTemporaryFile(delete=False, suffix=".jpg")
            tmpFile.seek(0)
            tmpFile.write(image.as_numpy().astype(np.bytes_).tobytes())
            tmpFile.close()
            data_list = [{"image": tmpFile.name, "label": [1,0,0,0]}]
            data = self.preprocessing(data_list)
            dataloader = DataLoader(dataset=data, batch_size=4, shuffle=False, num_workers=4)
            preds = []
            self.network.eval()
            with torch.no_grad():
                for batch in dataloader:
                    pred = self.inferer(batch['image'].to(device=self.inference_device), self.network)
                    pred = pred.softmax(-1)
                    preds.append(pred.detach().cpu().numpy())
            output0_tensor = pb_utils.Tensor("OUTPUT", np.array([preds]))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output0_tensor],
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """
        `finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass