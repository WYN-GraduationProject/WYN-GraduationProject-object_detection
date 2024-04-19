import cv2
import grpc
import numpy as np
from ultralytics import YOLO

from proto.video_service.video_model_pb2 import ProcessedVideoFrame
from proto.video_service.video_service_pb2_grpc import VideoServiceServicer
from utils.model.video import VideoModel
from utils.tools.LoggingFormatter import LoggerManager

logger = LoggerManager(logger_name="VideoFaceDetect").get_logger()
model = YOLO('model/yolov8n.pt')


async def process_object_detection_in_frame(frame):
    """
    在给定帧上进行目标检测和处理
    :param frame: 要处理的图像帧
    :param model: 用于目标检测的模型
    :return: 经过处理的图像帧的字节数据
    """
    global model
    # 在帧上进行目标检测
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, verbose=False)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode('.jpg', annotated_frame)
    return img_bytes.tobytes()


class VideoObjectDetect(VideoServiceServicer):
    async def ObjectDetection(self, request_iterator, context):
        """
        目标检测
        :param request_iterator: 流式请求迭代器
        :param context: rpc请求上下文
        """
        video = VideoModel("", "", "", [], 30)
        try:
            async for request in request_iterator:
                video.id = request.video_id
                video.filename = f"{video.id}.mp4"
                video.path = "video_data/object_detection"
                if request.is_final:
                    video.fps = request.fps
                    logger.info("该视频{}的视频帧率为：{}".format(video.id, video.fps))
                    logger.info("接收到来自客户端的结束帧...")
                    break

                np_arr = np.frombuffer(request.data, np.uint8)

                if np_arr.size == 0:
                    logger.error("没有图像数据.")
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "没有图像数据.")

                # 将numpy数组转换为OpenCV图像
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    logger.error("无法解码帧数据.")
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "无法解码帧数据.")

                # 目标检测
                obj_pro_img_bytes = await process_object_detection_in_frame(img)
                video.data.append(obj_pro_img_bytes)
                # 返回处理后的图像数据
                yield ProcessedVideoFrame(data=obj_pro_img_bytes, video_path=video.path + "/" + video.filename)
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            context.abort(grpc.StatusCode.INTERNAL, "人脸检测失败")
