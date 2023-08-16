FROM nvcr.io/nvidia/tritonserver:23.06-py3

# create model directory in container
RUN mkdir -p /models/monai_bdc/1
# install project-specific dependencies
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN rm requirements.txt

# copy contents of model project into model repo in container image
COPY models/monai_bdc/config.pbtxt /models/monai_bdc
COPY models/monai_bdc/1/model.py /models/monai_bdc/1



ENTRYPOINT [ "tritonserver", "--model-repository=/models"]
# ENTRYPOINT [ "tritonserver", "--model-repository=/models", "--log-verbose=1"]