# 超声波雷达状态话题
# 话题名称：sonic_state_data
# 话题接口：car_interfaces/msg/SonicStateInterface.msg

float64 timestamp       # 时间戳
uint8   id              # 毫米波雷达ID
uint8   state           # 设备状态，0：状态正常，1：状态异常
uint8   error           # 错误码；1:数据传输错误
