# 车道线识别话题
# 话题名称: lane_recognition_data  
# 话题接口: car_interfaces/msg/LaneRecognitionInterface.msg

float64             timestamp       # 时间戳
uint8               id              # 摄像头ID
float32             centeroffset    # 车道中心线偏移距离，默认0
sensor_msgs/Image   resultimage     # 融合后的车道线识别图像
float32             process_time    # 进程处理时间
