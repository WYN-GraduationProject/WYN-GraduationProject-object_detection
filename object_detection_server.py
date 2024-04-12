import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.tools.LoggingFormatter import LoggerManager
from utils.tools.NacosManager import NacosManager, NacosServerUtils
from web.object_detection import object_detection_api


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global nacos_serverutils
    nacos_manager = NacosManager()
    nacos_serverutils = nacos_manager.get_server_utils("object-detection-service", "0.0.0.0", 8002)
    await nacos_serverutils.register_service()
    asyncio.create_task(nacos_serverutils.beat(10))
    try:
        yield
    finally:
        await nacos_serverutils.deregister_service()


app = FastAPI(lifespan=app_lifespan)

# 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

app.include_router(object_detection_api.router)

logger = LoggerManager(logger_name="object_detection_server").get_logger()
nacos_logger = logging.getLogger('nacos.client')
nacos_logger.setLevel(logging.WARNING)
nacos_serverutils: NacosServerUtils = None  # 定义变量以便在事件处理器中引用

if __name__ == "__main__":
    logger.info("服务启动...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
