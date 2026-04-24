import warnings

import requests
import urllib3
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# IAM sample image (FKI host has had expired TLS certs; verify first, then dev fallback).
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
try:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
except requests.exceptions.SSLError:
    warnings.warn(
        "TLS certificate verification failed for the sample image URL "
        "(the server certificate may be expired). Retrying without verification — "
        "use a local image file if you need proper TLS.",
        UserWarning,
        stacklevel=1,
    )
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    resp = requests.get(url, stream=True, timeout=30, verify=False)
    resp.raise_for_status()

image = Image.open(resp.raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)