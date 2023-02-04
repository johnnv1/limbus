"""Basic example defining components and connecting them."""
from typing import List, Any
import asyncio

from limbus.core import (
    Component,
    Params,
    InputParams,
    OutputParams,
    ComponentState,
    VerboseMode,
)
from limbus.core.pipeline import Pipeline

from kornia.io import ImageLoadType, load_image
from kornia.contrib import FaceDetector, FaceDetectorResult
from kornia.filters import gaussian_blur2d
from kornia.color import rgb_to_bgr, bgr_to_rgb
from kornia import tensor_to_image, image_to_tensor
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import matplotlib

matplotlib.use("TkAgg")
# TODO: review/redo
# TODO: Move to a APP class

# --------------------------
# Plan
# 0. setup model
# 1. load image/frame
# 2. preprocess
# 3. Run detection
# 4. pos-process
# --------------------------


class Model(Component):
    def __init__(self, name, device, dtype):
        super().__init__(name)
        self.model = FaceDetector().to(device=device, dtype=dtype)
        self._device = device
        self._dtype = dtype

    @staticmethod
    def register_inputs(inputs: Params) -> None:
        inputs.declare("frame", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: Params) -> None:
        outputs.declare("detections", list)

    async def forward(self) -> ComponentState:
        frame = await asyncio.gather(self._inputs.frame.receive())
        # frame = frame[0] # FIXME: For some reason frame here is a list not a tensor
        print(f"{self._name}: Frame received: {type(frame)}")
        frame = frame[0].to(device=self._device, dtype=self._dtype)
        print(f"{self._name}: Frame shape: {frame.shape}")
        with torch.no_grad():
            await self._outputs.detections.send(self.model(frame))

        return ComponentState.OK


class Loader(Component):
    def __init__(self, name: str, frame_path: str):
        super().__init__(name)
        self._frame_path = frame_path

    @staticmethod
    def register_outputs(outputs: Params) -> None:
        outputs.declare("frame", torch.Tensor)
        outputs.declare("img_visu", np.array)

    @staticmethod
    def load(path) -> torch.Tensor:
        # frame = load_image(self._frame_path, ImageLoadType.RGB32)[None, ...]
        img_raw = cv2.imread(path, cv2.IMREAD_COLOR)
        img = image_to_tensor(img_raw, keepdim=False)
        frame = bgr_to_rgb(img)
        return frame

    async def forward(self) -> ComponentState:
        frame = self.load(self._frame_path)
        print(f"{self._name}: Frame loaded: {type(frame)}")
        await self._outputs.frame.send(frame)
        await self._outputs.img_visu.send(tensor_to_image(frame))
        return ComponentState.OK


class Preprocess(Component):
    @staticmethod
    def register_inputs(inputs: Params) -> None:
        inputs.declare("frame", torch.Tensor)

    @staticmethod
    def register_outputs(outputs: Params) -> None:
        outputs.declare("frame", torch.Tensor)

    def preprocess(self, frame) -> torch.Tensor:
        # Do something
        return frame

    async def forward(self) -> ComponentState:
        frame = await self._inputs.frame.receive()
        print(f"{self._name}: Frame received: {type(frame)}")
        await self._outputs.frame.send(self.preprocess(frame))

        return ComponentState.OK


class Posprocess(Component):
    @staticmethod
    def register_inputs(inputs: Params) -> None:
        inputs.declare("detections", list)
        inputs.declare("frame", torch.Tensor)
        inputs.declare("img_visu", np.array)

    @staticmethod
    def register_outputs(outputs: Params) -> None:
        outputs.declare("out", np.ndarray)

    @staticmethod
    def posprocess(detections, frame, img_visu) -> torch.Tensor:
        dets = [FaceDetectorResult(o) for o in detections]
        for det in dets:
            top_left = det.top_left.int().tolist()
            bottom_right = det.bottom_right.int().tolist()
            scores = det.score.tolist()
            for score, tp, br in zip(scores, top_left, bottom_right):
                x1, y1 = tp
                x2, y2 = br
            # if score < 0.7:
            #     continue  # skip detection with low score
            roi = frame[..., y1:y2, x1:x2]
            roi = gaussian_blur2d(roi, (21, 21), (35.0, 35.0))
            roi = rgb_to_bgr(roi)
            img_visu[y1:y2, x1:x2] = tensor_to_image(roi)
        return img_visu

    async def forward(self) -> ComponentState:
        dets, frame, img_visu = await asyncio.gather(
            self._inputs.detections.receive(),
            self._inputs.frame.receive(),
            self._inputs.img_visu.receive(),
        )
        print(f"{self._name}: Frame received: {type(frame)}")
        await self._outputs.out.send(self.posprocess(dets, frame, img_visu))
        return ComponentState.OK


class Show(Component):
    @staticmethod
    def register_inputs(inputs: Params) -> None:
        inputs.declare("result", np.ndarray)

    async def forward(self) -> ComponentState:
        result = await self._inputs.result.receive()

        plt.figure(figsize=(8, 8))
        plt.imshow(result)
        plt.axis("off")
        plt.show()
        return ComponentState.OK


model = Model("FaceDetectorA", torch.device("cuda"), torch.float32)
data0 = Loader("FrameA", "crowd.jpg")
preprocess = Preprocess("Prepare data")
posprocess = Posprocess("Prepare results")
show = Show("Show result")

data0.outputs.frame >> preprocess.inputs.frame

preprocess.outputs.frame >> model.inputs.frame

preprocess.outputs.frame >> posprocess.inputs.frame
data0.outputs.img_visu >> posprocess.inputs.img_visu
model.outputs.detections >> posprocess.inputs.detections

posprocess.outputs.out >> show.inputs.result
# create and run the pipeline
# ---------------------------
engine: Pipeline = Pipeline()
engine.add_nodes([preprocess, model, posprocess, show])

# there are several states for each component, with this verbose mode we can see them
# engine.set_verbose_mode(VerboseMode.COMPONENT)
# run all teh components at least once (since there is an accumulator, some components will be run more than once)
engine.run(1)
