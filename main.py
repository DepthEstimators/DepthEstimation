#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:45:22 2021

@author: spatikaganesh
https://fastapi.tiangolo.com/
"""

from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import os
import shutil
import shlex
import test_simple

demo_image_path = "./assets/test_image_disp.jpeg"
app = FastAPI()
model_name = "mono+stereo_640x192"

@app.get("/")
def read_root():
    return {"Hello" : "World"}

@app.get("/items/{item_id}") 
def read_item(item_id:int, q:Optional[str]=None):
    return {"item_id": item_id, "q":q}

@app.get("/demoimage")
def get_test_simple():
    return FileResponse(demo_image_path, media_type="image/jpeg")

@app.post("/predict/single")
async def predict_single(image: UploadFile = File(...)):
    extension = image.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format" + str(image.filename.split(".")[-1])
    image_path = "uploadSingle." + image.filename.split(".")[-1] 
    args_str = "--image_path "+ image_path +" --model_name mono+stereo_640x192 --pred_metric_depth --no_cuda"
    #args_arr = [image_path, "mono+stereo_640x192"]
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    args = test_simple.parse_args(shlex.split(args_str))
    test_simple.test_simple(args)
    
    dest_image_path = "uploadSingle_disp.jpeg";
    
    return FileResponse(dest_image_path, media_type="image/jpeg")
    
    