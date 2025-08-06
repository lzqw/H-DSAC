# 激光雷达数据话题
# 话题名称：lidar_ori_data  
# 话题接口：car_interfaces/msg/LidarOriInterface.msg

float64     timestamp       # 时间戳
uint8       id              # 激光雷达ID
uint32      number          # 点云数量
float32[]   pointcloud      # 点云数据
float32     process_time    # 进程处理时间

# 其中pointcloud的float32[]为一维数组
# 储存按照[ranges, angle_h, angle_v, x, y, z, intensities,  ranges, angle_h, angle_v, x, y, z, intensities，……](激光雷达一圈点云数据)
# 7个长度数据为一组
# float ranges          测量的距离数据[米]，精度为0.01
# float angle_h         水平角度[弧度]，精度为0.01
# float angle_v         垂直角度[弧度]，精度为0.01
# float x               笛卡尔坐标系x轴坐标[米]，精度为0.01
# float y               笛卡尔坐标系y轴坐标[米]，精度为0.01
# float z               笛卡尔坐标系z轴坐标[米]，精度为0.01
# float intensities     强度数据
