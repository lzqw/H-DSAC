# 底盘状态话题
# 话题名称：can_state_data
# 话题接口：car_interfaces/msg/CanStateInterface.msg（自定义）

float64     timestamp       # 时间戳
uint8       id              # CANID
uint8       state           # 设备状态，0：状态正常，1：状态异常
uint8       error           # 错误码,  0: 无错误，1：库函数错误，2：打开设备错误，3：初始化CAN0通道错误，4：清空CAN0缓冲区错误，5：启动CAN0通道错误，6：初始化CAN1通道错误，7：清空CAN1缓冲区错误，8：启动CAN1通道错误
