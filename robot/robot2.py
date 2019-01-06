import argparse
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

width, height = 640, 480  # 设置屏幕上模拟窗口的宽度和高度
weight_path = 'genes.npz'

class robot:
    def __init__(self,vmax,R,L,wheel_width):
        # 初始化位置和速度
        self.R = R
        self.L = L
        self.wheel_width = wheel_width
        self.vmax = vmax
        self.time = 0
        self.pos = [width / 2.0, height/ 2.0] + 10 * np.random.rand(2)
        self.angles = math.pi/2
        self.pos_sensor_right,self.pos_sensor_left = self.sensorPostion(self.angles,self.pos)
        self.pos_wheel_right, self.pos_wheel_left = self.wheelPostion(self.angles,self.pos)
        self.gene = []
        loader=np.load(weight_path)
        self.gene.append(loader['arr_0'])
        self.gene.append(loader['arr_1'])

    # 输入机器人的位置和角度，返回sensor的位置
    def sensorPostion(self,angle,pos):
        x=pos[0]
        y=pos[1]

        a=angle-math.pi/4
        b=angle+math.pi/4

        left=[]
        x_left = x+math.cos(a)*self.L
        y_left = y+math.sin(a)*self.L
        left.append(x_left)
        left.append(y_left)

        right=[]
        x_right = x+math.cos(b)*self.L
        y_right = y+math.sin(b)*self.L
        right.append(x_right)
        right.append(y_right)

        return np.array(left),np.array(right)

    def wheelPostion(self,angle,pos):

        x = pos[0]
        y = pos[1]
        a = angle - math.pi / 2
        b = angle + math.pi / 2

        left = []
        x_left = x + math.cos(a) * (self.L+self.wheel_width)
        y_left = y + math.sin(a) * (self.L+self.wheel_width)
        left.append(x_left)
        left.append(y_left)

        x1 = left[0] + math.cos(angle)*self.R
        y1 = left[1] + math.sin(angle)*self.R
        x2 = left[0] - math.cos(angle)*self.R
        y2 = left[1] - math.sin(angle)*self.R

        right = []
        x_right = x + math.cos(b) * (self.L+self.wheel_width)
        y_right = y + math.sin(b) * (self.L+self.wheel_width)
        right.append(x_right)
        right.append(y_right)

        x3 = right[0] + math.cos(angle)*self.R
        y3 = right[1] + math.sin(angle)*self.R
        x4 = right[0] - math.cos(angle)*self.R
        y4 = right[1] - math.sin(angle)*self.R

        return np.array([x1,x2,y1,y2]), np.array([x3,x4,y3,y4])

    # 求两点之间的距离
    def distance(self,a,b):
        return np.sqrt(np.sum(np.square(a - b)))

    # 求sensor感受到的角度,light是一个坐标值,返回的是左右两个sensor到光点的连线角度
    def sense(self,light):
        vector1 = light - self.pos_sensor_left
        vector2 = light - self.pos_sensor_right
        a_left = (math.atan2(vector1[1], vector1[0]) + 2 * math.pi) % (2 * math.pi) - self.angles
        a_right = (math.atan2(vector2[1], vector2[0]) + 2 * math.pi) % (2 * math.pi) - self.angles
        a_left = (a_left + 2 * math.pi) % (2 * math.pi)
        a_right = (a_right + 2 * math.pi) % (2 * math.pi)
        if (a_left > math.pi):
            a_left = a_left - 2 * math.pi
        if (a_right > math.pi):
            a_right = a_right - 2 * math.pi
        return a_left, a_right

    # 通过角度来求两个轮子的转速


    def softmax(self,x):
        # exps = np.exp(x-np.max(x))
        exp1 = np.exp(x[0]-x.mean())
        exp2 = np.exp(x[1]-x.max())

        x[0] = exp1/(exp1+exp2)
        x[1] = exp2/(exp1+exp2)
        # return  exps/np.sum(exps)
        return x

    def fowrd(self,x, gene):
        # max = x.max()
        # min = x.min()
        x1 = np.dot( x , gene[0])
        # x1=relu(x1)
        max = x1.max()
        min = x1.min()
        x2 = np.dot((x1 - min) / (max - min), gene[1])
        x2 = self.softmax(x2)
        return x2

    # 输入数据是基因->权值,
    #               x->当前位置,角度，光源的位置
    #
    # 传出的数据是接下来的v1，和v2的比，如何
    def gene2presatation(self, position2):
        gene = self.gene
        a_left, a_right = self.sense(position2)
        x = np.append(a_left, a_right)
        # x = np.append(x,self.angles)
        y = self.fowrd(x, gene)
        v_left, v_right = self.vmax * y[0], self.vmax * y[1]
        return v_left, v_right


    def update(self,light):


        v_left,v_right=self.gene2presatation(light)

        # 转弯时的圆弧的半径
        if v_left>v_right:
            r=2*self.L*v_right/(v_left-v_right)
        elif v_left<v_right:
            r = 2*self.L * v_left / (v_right - v_left)
        else:
            r=0

        # 中心线速度,然后根据左右速度大小求得角速度
        v=(v_left+v_right)/2
        if v_left>v_right:
            w=-v/(self.L+r)
        elif v_left<v_right:
            w=v/(self.L+r)
        else:
            w=0

        # 更新angle
        self.angles = self.angles + w
        self.angles = self.angles%(2*math.pi)


        # 求pos的变化量
        if v_right!=v_left:
            if v_right> v_left:
                y_angle = math.sin(w) * (self.L + r)
                x_angle = -(1 - math.cos(w)) * (self.L + r)
            else:
                x_angle = (1 + math.cos(w+math.pi))* (self.L + r)

                y_angle = math.sin(w+math.pi) * (self.L + r)

            x = y_angle*math.cos(self.angles)+x_angle*math.cos(self.angles-math.pi/2)
            y = y_angle*math.sin(self.angles)+x_angle*math.sin(self.angles-math.pi/2)

        # 走直线时，
        else:
            x = math.cos(self.angles)*v
            y = math.sin(self.angles)*v

        delt_pos = []
        delt_pos.append(x)
        delt_pos.append(y)
        # 更新postion
        if(self.distance(self.pos,light)<=self.L):
            self.pos = self.pos
            if (self.time != 0):
                print(self.time)
                self.time = 0
        else:
            self.pos = self.pos +np.array(delt_pos)
            self.pos_sensor_right, self.pos_sensor_left = self.sensorPostion(self.angles, self.pos)
            self.pos_wheel_right, self.pos_wheel_left = self.wheelPostion(self.angles, self.pos)
            self.time += 1


def tick(frameNum, pts, sensor1, sensor2, wheel1, wheel2, robot, light_s, light):
    # print frameNum
    """update function for animation"""
    robot.update(light)
    # update data
    pts.set_data(robot.pos[0],robot.pos[1])
    sensor1.set_data(robot.pos_sensor_left[0],robot.pos_sensor_left[1])
    sensor2.set_data(robot.pos_sensor_right[0],robot.pos_sensor_right[1])
    wheel1.set_data(robot.pos_wheel_left[0:2],robot.pos_wheel_left[2:4])
    wheel2.set_data(robot.pos_wheel_right[0:2],robot.pos_wheel_right[2:4])
    light_s.set_data(light[0], light[1])


    return pts, sensor1 ,sensor2, wheel1, wheel2, light_s

# main() function
def main():
    # use sys.argv if needed
    print('starting boids...')

    parser = argparse.ArgumentParser(description="Implementing Craig Reynold's Boids...")
    # add arguments
    parser.add_argument('--num-boids', dest='N', required=False)
    args = parser.parse_args()

    vmax = 10
    R=25
    L=50
    wheel_width = 1

    # light = (600, 480)
    # light = (0, 480)

    light = (0,0)
    # light = (600, 0)
    # create boids
    boids = robot(vmax=vmax,R=R,L=L,wheel_width=wheel_width)

    # setup plot
    fig = plt.figure(facecolor='blue')
    ax = plt.axes(xlim=(0, width), ylim=(0, height), facecolor='green')


    # 改成椭圆
    # ax.add_patch(patches.Ellipse((1, 1), R, wheel_width, boids.angles))
    pts, = ax.plot([], [], markersize=L,
                   c='k', marker='o', ls='None')

    light_s, = ax.plot([], [], markersize=7,
                       c='r', marker='o', ls='None')

    wheel1, =ax.plot([], [], linewidth=6.0)

    wheel2, =ax.plot([], [], linewidth=6.0)

    sensor1, = ax.plot([], [], markersize=4,
                       c='r', marker='o', ls='None')
    sensor2, =ax.plot([], [], markersize=4,
                      c='r', marker='o', ls='None')

    anim = animation.FuncAnimation(fig, tick, fargs=(pts, sensor1, sensor2, wheel1, wheel2, boids, light_s, light),
                                   interval=50)
    # ells = patches.Ellipse((1, 1),  R, wheel_width, boids.angles)


    # add a "button press" event handler
    # cid = fig.canvas.mpl_connect('button_press_event', boids.buttonPress)

    plt.show()


# call main
if __name__ == '__main__':
    main()