# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import io
from PIL import Image
import rectification as rec
import numpy as np
import cv2

app = FastAPI()

def process_image(input_image: Image.Image) -> Image.Image:
    return rec.rectification(input_image)

@app.post("/api/process_image")
async def handle_image(file: UploadFile = File(...)):
    # 1. 读取上传的图片
    image_data = await file.read()
    # input_image = Image.open(io.BytesIO(image_data))
    input_image = np.frombuffer(image_data, np.uint8)
    cv_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)

    # 2. 调用你的处理函数
    output_image = process_image(cv_image)

    # 3. 将返回的图片转为字节流
    _, img_encoded = cv2.imencode(".png", output_image)
    img_bytes = img_encoded.tobytes()

    # 返回图片
    return Response(content=img_bytes, media_type="image/png")