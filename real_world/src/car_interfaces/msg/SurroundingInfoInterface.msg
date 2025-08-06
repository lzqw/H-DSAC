# 融合数据话题
# 话题名称：fusion_data  
# 话题接口：car_interfaces/msg/FusionInterface.msg

float64     timestamp               # 时间戳
float32[]   surroundinginfo         # 交通标志数据
float32[]   path_rfu                # 路径规划数据
float32     turn_signals            #转向 -1：左转，0：直行，1：右转
float32     process_time            # 进程处理时间
float32     carlength               # 车长m
float32     carwidth                # 车宽m
float32     carheight               # 车高m
float32     carspeed                # 车辆速度，单位为m/s
float32     steerangle              # 车辆转角，左转为正，右转为负
uint8       throttle_percentage     # 油门踏板开度： 取值0～100
uint8       braking_percentage      # 制动踏板开度： 取值0～100
float32     braketq                 # 制动量（-50-50nm）
uint8       car_run_mode            # 车辆运行模式： 0： ACU控制  1：自动驾驶  2： 急停模式
int8        gearpos                 # 车辆档位信号（01：驻车，02：空挡（N），03：前进（D），04：后退（R），05：无效）
float32     error_yaw                # 车辆偏航角度
float32     error_distance            # 车辆偏航角速度
# 其中signdata格式float32[]为一维数组
# 储存按照[category ,length ,width ,height , x  ,y  ,z  ,v  ,latv ,category  ,length ,width ,height , x ,y ,z ,v ,latv ,……]存储
# 9个长度数据为一组
# int   category    交通标志类别，需自行定义类别，务必做好记录
# float length      交通标志长度m，精度为0.01
# float width       交通标志宽度m，精度为0.01
# float height      交通标志高度m，精度为0.01
# float x           交通标志的x坐标[米]，精度为0.01
# float y           交通标志的y坐标[米]，精度为0.01
# float z           交通标志的z坐标[米]，精度为0.01
# float v=0         交通标志径向速度m/s，默认为0，精度为0.01
# float latv=0      交通标志的横向速度m/s，默认为0，精度为0.01

# 其中lightdata格式float32[]为一维数组
# 储存按照[x,y,z,leftcolor,straightcolor,rightcolor,……]存储
# 6个长度数据为一组
# float x               红绿灯x坐标[米]，精度为0.01
# float y               红绿灯y坐标[米]，精度为0.01
# float z               红绿灯z坐标[米]，精度为0.01
# int   leftcolor       左转灯状态，01：红，02：黄，03：绿，04：关闭，关闭时代表没有左转灯
# int   straightcolor   直行灯状态，01：红，02：黄，03：绿，04：关闭，
# int   rightcolor      右转灯状态，01：红，02：黄，03：绿，04：关闭，关闭时代表没有右转灯
# 一般情况下，如果左转/右转/左右转和直行灯同时存在，无论什么颜色状态，这两个灯都是常亮的。如果信号灯只有直行灯，那用左/右转灯关闭代表当前路口只有直行灯，直行灯绿色时既可以直行也可以左转。

# 其中obstacledata格式float32[]为一维数组
# 储存按照[category ,length ,width ,height , x  ,y  ,z  ,v  ,latv ,category  ,length ,width ,height , x ,y ,z ,v ,latv ,……]存储
# 9个长度数据为一组
# int   category    障碍物类别， 0点目标；1小汽车；2卡车/客车；3行人；4摩托车/电动车；5自行车；6宽大目标（如墙面）；7reserved保留； 8traffic light； 9stop sign
# float length      障碍物长度m，精度为0.01
# float width       障碍物宽度m，精度为0.01
# float height      障碍物高度m，精度为0.01
# float x           障碍物的x坐标[米]，精度为0.01
# float y           障碍物的y坐标[米]，精度为0.01
# float z           障碍物的z坐标[米]，精度为0.01
# float v=0         障碍物径向速度m/s，精度为0.01
# float latv=0      障碍物的横向速度m/s，精度为0.01
