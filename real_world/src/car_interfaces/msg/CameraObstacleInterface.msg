# 障碍物识别话题
# 话题名称: camera_obstacle_data  
# 话题接口: car_interfaces/msg/CameraObstacleInterface.msg

float64             timestamp       # 时间戳
uint8               id              # 摄像机ID
uint16              number          # 障碍物数量
float32[]           obstacledata    # 障碍物数据
sensor_msgs/Image   resultimage     # 障碍物检测结果图像
float32             process_time    # 进程处理时间

# 其中obstacledata格式float32[]为一维数组
# 储存按照[category ,length ,width ,height , x  ,y  ,z  ,v  ,latv ,category  ,length ,width ,height , x ,y ,z ,v ,latv ,……]存储
# 9个长度数据为一组
# int   category     # 障碍物类别， 0点目标；1小汽车；2卡车/客车；3行人；4摩托车/电动车；5自行车；6宽大目标（如墙面）；7reserved保留； 8traffic light； 9stop sign
# float length       # 障碍物长度m，精度为0.01
# float width       # 障碍物宽度m，精度为0.01
# float height       # 障碍物高度m，精度为0.01
# float x     # 障碍物的x坐标[米]，精度为0.01
# float y   # 障碍物的y坐标[米]，精度为0.01
# float z    # 障碍物的z坐标[米]，精度为0.01
# float v=0    # 障碍物径向速度m/s，精度为0.01
# float latv=0  # 障碍物的横向速度m/s，精度为0.01
# 注:当障碍物类别存在而x,y,z值均为0，说明程序出现错误（主动式）
