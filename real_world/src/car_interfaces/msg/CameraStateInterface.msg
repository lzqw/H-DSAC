# 相机状态话题
# 话题名称：camera_state_data
# 话题接口：car_interfaces/msg/CameraStateInterface.msg

float64     timestamp       # 时间戳
uint8       id              # 摄像头ID
uint8       state           # 设备状态，0：状态正常，1：状态异常
uint8       error           # 错误码；0：无错误，1：未连接设备
