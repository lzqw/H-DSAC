# 惯导设备状态话题
# 话题名称：gps_state_data  
# 话题接口：car_interfaces/msg/GpsStateInterface.msg

float64     timestamp       # 时间戳
uint8       id              # 导航设备ID
uint8       state           # 设备状态，0：状态正常，1：状态异常
uint8       error           # 错误码；1:打开串口错误，2:读取共享内存错误，3:写入共享内存错误，4:读取串口数据错误，5:GPS数据校验错误，6:非组合导航模式，7:无GPS消息，8:无车辆消息，9:陀螺错误，10:加表错误
