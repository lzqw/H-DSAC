# 惯导设备状态话题
# 话题名称：imu_state_data
# 话题接口：car_interfaces/msg/ImuStateInterface.msg

float64     timestamp       # 时间戳
uint8       id              # 导航设备ID
uint8       state           # 设备状态，0：状态正常，1：状态异常
uint8       error           # 错误码；待补充

