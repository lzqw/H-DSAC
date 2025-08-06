# 相机图像数据话题
# 话题名称: camera_ori_data
# 话题接口: car_interfaces/msg/CameraOriInterface.msg

float64             timestamp       # 时间戳
uint8               id              # 摄像头ID
sensor_msgs/Image   imagedata       # 相机图像
float32             process_time    # 进程处理时间
