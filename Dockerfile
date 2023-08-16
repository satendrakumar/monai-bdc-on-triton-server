FROM nvcr.io/nvidia/tritonserver:23.06-py3

RUN mkdir -p /mnt/models/monai_bdc/1

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY models/monai_bdc/config.pbtxt /mnt/models/monai_bdc
COPY models/monai_bdc/1/model.py /mnt/models/monai_bdc/1
COPY pytorch/model.pt /mnt/models/pytorch


ENTRYPOINT [ "tritonserver", "--model-repository=/mnt/models", "--log-verbose=1"]