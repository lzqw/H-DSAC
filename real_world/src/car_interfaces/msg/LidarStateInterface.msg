# 激光雷达状态话题
# 话题名称：lidar_state_data
# 话题接口：car_interfaces/msg/LidarStateInterface.msg

float64 timestamp       # 时间戳
uint8   id              # 激光雷达ID
uint8   state           # 设备状态，0：状态正常，1：状态异常
uint8   error           # 错误码；1：通讯错误，2：通讯接受异常，3：校验错误
