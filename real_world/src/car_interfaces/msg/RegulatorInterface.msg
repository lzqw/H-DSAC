# 决策状态机数据话题
# 话题名称: regulator_data
# 话题接口: car_interfaces/msg/RegulatorInterface.msg

float64     timestamp       # 数据时间戳
bool        stop            # 是否停车，0无意义，1是，监管者停车优先级最高
float32     process_time    # 进程处理时间

# 依据：超声波雷达障碍物获取/故障时-停车，导航数据+车道线识别数据均丢失时-停车，
