import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

from proto.video_service.video_model_pb2 import VideoFrame
from utils.model.video import VideoModel
from utils.tools.LoggingFormatter import LoggerManager
from utils.tools.gRPCManager import GrpcManager

router = APIRouter(
    prefix="/api/object_detection",
    tags=["object_detection"],
    responses={404: {"description": "Not found"}},
)

logger = LoggerManager(logger_name="object_detection").get_logger()
model = YOLO('model/yolov8n.pt')


def get_grpc_manager():
    """
    获取gRPC管理器
    :return: gRPC管理器
    """
    return GrpcManager()


async def process_object_detection_in_frame(frame, model):
    """
    在给定帧上进行目标检测和处理
    :param frame: 要处理的图像帧
    :param model: 用于目标检测的模型
    :return: 经过处理的图像帧的字节数据
    """
    # 在帧上进行目标检测
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, verbose=False)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode('.jpg', annotated_frame)
    return img_bytes.tobytes()


async def process_video(video_model: VideoModel, grpc_manager) -> str | None:
    """
    处理视频
    :param video_model: 视频实体
    :param grpc_manager: gRPC管理器
    :return:
    """
    global model
    cap = cv2.VideoCapture(video_model.path + video_model.filename)
    video_model.fps = cap.get(cv2.CAP_PROP_FPS) > 100 and 30 or cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video FPS: {video_model.fps}")
    if not cap.isOpened():
        logger.error(f"打开文件失败: {video_model.path + video_model.filename}")
        return
    async with grpc_manager.get_stub('video_pre_service') as stub:
        async def request_generator():
            """
            请求生成器
            :return: 视频处理的请求
            """
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _, img = cv2.imencode('.jpg', frame)
                img_bytes = img.tobytes()
                yield VideoFrame(data=img_bytes, is_final=False, video_id=video_model.id)
            yield VideoFrame(is_final=True, fps=video_model.fps, video_id=video_model.id)

        try:
            response_stream = stub.ProcessVideo(request_generator())
            async for response in response_stream:
                response_frame = np.frombuffer(response.data, dtype=np.uint8)
                response_frame = cv2.imdecode(response_frame, cv2.IMREAD_COLOR)

                # 人脸检测
                face_pro_img_bytes = await process_object_detection_in_frame(response_frame, model)
                video_model.data.append(face_pro_img_bytes)
        except Exception as e:
            # gRPC服务调用失败，不进行预处理，目前返回原视频路径
            logger.error(f"RPC预处理视频失败: {e}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                face_pro_img_bytes = await process_object_detection_in_frame(frame, model)
                video_model.data.append(face_pro_img_bytes)
        finally:
            logger.info("视频流处理完成")
            await video_model.save()
            cap.release()
            return video_model.path + video_model.filename


@router.post("/test")
async def upload_video(video: UploadFile = File(...), grpc_manager: GrpcManager = Depends(get_grpc_manager)):
    # 传递视频文件路径给处理函数
    video_model = await VideoModel.http_video_save(video)
    file_name = await process_video(video_model, grpc_manager)
    response = FileResponse(file_name, media_type="video/mp4", filename="video.mp4",
                            background=BackgroundTask(lambda: os.remove(file_name)))
    return response
