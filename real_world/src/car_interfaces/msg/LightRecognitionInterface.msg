# 红绿灯识别话题
# 话题名称: light_recognition_data
# 话题接口: car_interfaces/msg/LightRecognitionInterface.msg

float64             timestamp       # 时间戳
uint8               id              # 摄像头ID
float32[]           lightdata       # 红绿灯数据
sensor_msgs/Image   resultimage     # 红绿灯识别结果图像
float32             process_time    # 进程处理时间

# 其中lightdata格式float32[]为一维数组
# 储存按照[x,y,z,leftcolor,straightcolor,rightcolor,……]存储
# 6个长度数据为一组
# float x               红绿灯x坐标[米]，精度为0.01
# float y               红绿灯y坐标[米]，精度为0.01
# float z               红绿灯z坐标[米]，精度为0.01
# int   leftcolor       左转灯状态，01：红，02：黄，03：绿，04：关闭，关闭时代表没有左转灯
# int   straightcolor   直行灯状态，01：红，02：黄，03：绿，04：关闭，
# int   rightcolor      右转灯状态，01：红，02：黄，03：绿，04：关闭，关闭时代表没有右转灯

