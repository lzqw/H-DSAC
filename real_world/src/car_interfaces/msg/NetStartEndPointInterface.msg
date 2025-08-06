# 全局路径规划话题
# 话题名称: net_start_end_point_data
# 话题接口: car_interfaces/msg/NetStartEndPointInterface.msg

float64     timestamp       # 时间戳
float64[]   startpoint      # 起点位置，先经后纬
float64[]   endpoint        # 终点位置，先经后纬
float32     process_time    # 进程处理时间
