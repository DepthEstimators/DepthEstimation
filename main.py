#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:45:22 2021

@author: spatikaganesh
https://fastapi.tiangolo.com/
"""

from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Response, Request, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
#from app.api import viz
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import shlex
import test_simple
import ProcessVideoFiles as video_test_simple
from fastapi.templating import Jinja2Templates

demo_image_path = "./assets/test_image_disp.jpeg"
app = FastAPI(title='Depth Estimation API', description="API with produces disparity map with depth estimation given input images, video files.",
              version='0.1', docs_url='/',)
model_name = "mono+stereo_640x192"
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/test", StaticFiles(directory="test"), name="test")
app.add_middleware(
CORSMiddleware,
allow_origins=['*'],
allow_credentials=True,
allow_methods=['*'],
allow_headers=['*'],
)
templates = Jinja2Templates(directory="templates/")

@app.get("/")
def read_root():
    return {"Hello" : "World"}

@app.get("/predict/single")
def predict_single(request: Request):
    result = None
    return templates.TemplateResponse('single_predict.html', context={'request': request, 'result': result})


@app.get("/demoimage")
def get_test_simple(request:Request):
    result = demo_image_path
    return templates.TemplateResponse('imageTest.html',  context={'request': request, 'result': result})

@app.post("/predict/single")
async def predict_single(request:Request, imageFile: UploadFile = File(...)):
    extension = imageFile.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format"
    image_path = "./assets/uploadSingle." + imageFile.filename.split(".")[-1]
    args_str = "--image_path " + image_path + " --model_name mono+stereo_640x192 --pred_metric_depth --no_cuda"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(imageFile.file, buffer)

    args = test_simple.parse_args(shlex.split(args_str))
    test_simple.test_simple(args)

    result = "../assets/uploadSingle_disp.jpeg"
    return templates.TemplateResponse('single_predict.html', context={'request': request, 'result': result})

@app.get("/predict/video")
def predict_single(request: Request):
    result = None
    return templates.TemplateResponse('video_predict.html', context={'request': request, 'result': result})

@app.post("/predict/video")
async def predict_video(request:Request, videoFile: UploadFile = File(...)):
    extension = videoFile.filename.split(".")[-1] in "mp4"
    if not extension:
        return "Video must be mp4 format"
    video_path = "./assets/uploadVideo.mp4"
    args_str = "--input_path " + video_path + " --model_name mono+stereo_640x192 --pred_metric_depth --no_cuda"
    dest_video_path = "./test"
    args_str = args_str + " --output_path " + dest_video_path
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(videoFile.file, buffer)

    args = video_test_simple.parse_args(shlex.split(args_str))
    video_test_simple.test_simple(args)

    source = "../test/uploadVideo.mp4"
    return templates.TemplateResponse('video_predict.html', context={'request': request,  'source':source})

if __name__ == '__main__':
    uvicorn.run(app)




