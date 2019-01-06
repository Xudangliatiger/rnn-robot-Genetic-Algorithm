import numpy as np
import math
import matplotlib.pyplot as plt

generation_num = 20
environment_num = 100
population = 1000
vmax= 1
L = 50
lr = 0.01
weight_path = None


def sense(light,pos_sensor_left,pos_sensor_right,angles):
    vector1 = light - pos_sensor_left
    vector2 = light - pos_sensor_right
    a_left = (math.atan2(vector1[1], vector1[0]) + 2 * math.pi) % (2 * math.pi) - angles
    a_right = (math.atan2(vector2[1], vector2[0]) + 2 * math.pi) % (2 * math.pi) - angles
    a_left =  (a_left+2 * math.pi) % (2 * math.pi)
    a_right = (a_right + 2 * math.pi) % (2 * math.pi)
    if(a_left>math.pi):
        a_left =a_left-2*math.pi
    if (a_right > math.pi):
        a_right = a_right - 2 * math.pi
    return a_left, a_right

def sensorPostion(angle,pos):
    x=pos[0]
    y=pos[1]
    a=angle-math.pi/4
    b=angle+math.pi/4

    left=[]
    x_left = x+math.cos(a)*L
    y_left = y+math.sin(a)*L
    left.append(x_left)
    left.append(y_left)

    right=[]
    x_right = x+math.cos(b)*L
    y_right = y+math.sin(b)*L
    right.append(x_right)
    right.append(y_right)

    return np.array(left),np.array(right)
def softmax(x):
    # exps = np.exp(x-np.max(x))
    exp1 = np.exp(x[0])
    exp2 = np.exp(x[1])
    x[0] = exp1 / (exp1 + exp2)
    x[1] = exp2 / (exp1 + exp2)
    # return  exps/np.sum(exps)
    return x
def fowrd(x,gene):
    # max = x.max()
    # min = x.min()
    x1=np.dot(x,gene[0])
    max = x1.max()
    min = x1.min()
    x2=np.dot((x1-min)/(max-min),gene[1])
    x2=softmax(x2)
    return x2

def gene2presatation(gene,position1,position2,angle):
    pos_sensor_left, pos_sensor_right = sensorPostion(angle,position1)
    a_left, a_right = sense(position2,pos_sensor_left,pos_sensor_right,angles=angle)
    x= np.append(a_left, a_right)
    # x= np.append(x,angle)
    y=fowrd(x,gene)
    return y
    # 适应情况，应该是用表现型来计算的。。。所以传入的参数是表现型和环境数据，而这个fitness会不会有不同的表现呢
def nextPosition(y,position1,angle):
    v_left, v_right = vmax*y[0],vmax*y[1]
    # 转弯时的圆弧的半径
    if v_left > v_right:
        r = 2 * L * v_right / (v_left - v_right)
    elif v_left < v_right:
        r = 2 * L * v_left / (v_right - v_left)
    else:
        r = 0
    # 中心线速度,然后根据左右速度大小求得角速度
    v = (v_left + v_right) / 2
    if v_left > v_right:
        w = -v / (L + r)
    elif v_left < v_right:
        w = v / (L + r)
    else:
        w = 0
    # 更新angle
    angle1 = angle + w
    angle1 = angle1 % (2 * math.pi)
    # 求pos的变化量
    if v_right != v_left:
        if v_right > v_left:
            y_angle = math.sin(w) * (L + r)
            x_angle = -(1 - math.cos(w)) * (L + r)
        else:
            x_angle = (1 + math.cos(w + math.pi)) * (L + r)
            y_angle = math.sin(w + math.pi) * (L + r)
        x_ = y_angle * math.cos(angle1) + x_angle * math.cos(angle1 - math.pi / 2)
        y_ = y_angle * math.sin(angle1) + x_angle * math.sin(angle1 - math.pi / 2)
    # 走直线时，
    else:
        x_ = math.cos(angle1) * v
        y_ = math.sin(angle1) * v
    delt_pos = []
    delt_pos.append(x_)
    delt_pos.append(y_)
    # 更新 postion
    position = position1 + np.array(delt_pos)
    return angle1,position

def fitness(gene,evironment):
    position1, position2, angle = evironment[0],evironment[1],evironment[2]
    y = gene2presatation(gene,position1, position2,angle)
    angle2,position3 = nextPosition(y,position1,angle)
    #     # 距离减少的百分比
    delt_distance = (np.sqrt(np.sum(np.square(position1 - position2)))-np.sqrt(np.sum(np.square(position3 - position2))))/np.sqrt(np.sum(np.square(position1 - position2)))
        # 角度减少的百分比
    vector1 = position2 - position1
    vector2 = position2 - position3
    #
    angle1_light = (math.atan2(vector1[1],vector1[0])+2*math.pi)%(2*math.pi)
    angle2_light = (math.atan2(vector2[1],vector2[0])+2*math.pi)%(2*math.pi)
    #
    delt_angle1 = math.fabs(angle1_light-angle)
    delt_angle2 = math.fabs(angle2_light-angle2)
    if (delt_angle1>math.pi):
        delt_angle1 = 2*math.pi - delt_angle1
    if (delt_angle2>math.pi):
        delt_angle2 = 2*math.pi - delt_angle2
    delt_angle = (delt_angle1-delt_angle2)/delt_angle1
    # 两个百分比作为score，但是在前期，两者肯定不在一个数量级上。需要手动调参...
    score = delt_angle

    return score

# 初始化环境，多种环境在一次挑选中



def genesort(generation,environment):
    # 首先要计算fitness，然后对每个gene来排序，然后把垃圾的基因给删除了
    # ppt 里面写的要计算出每一种情况下的fitness
    # 输出为排序之后的分数加基因二位列表
    fitscore = []
    for i in generation:
        #每一次gene在范围内的变化
        temp = []
        for j in environment:
            temp.append(fitness(i,j))
        fitscore.append(np.mean(temp))

    z = list(zip(generation, fitscore))

    z.sort(key=lambda x: x[1],reverse=True)
    # 对列表，排序，里面后90%清空
    return z



def mutation(genes):
    # 对基因的进化写一个算法 ，每一个值要增加多少合适呢。。。感觉这个和学习率差不多。。
    # 首先要前1%的不变，后面的都改变。。。
    random1 = np.random.randn(2, 4)
    random2 = np.random.randn(4, 2)
    newgenes =[]
    newgenes.append(lr*random1/np.abs(random1)+genes[0])
    newgenes.append(lr*random2/np.abs(random2)+genes[1])
    return  newgenes
def newpopulation(generation):
    # 新的人口就是前面
    # top k%，精英阶级直接给留下来，1%
    # 剩下的才进化
    elite_num = population*0.1
    newgeneration = []
    for i in range(population):
        if i<elite_num:
            newgeneration.append(generation[i])
        if i>=elite_num and i<900:
            newgeneration.append(mutation(generation[i]))
        elif i>=900:
            newgeneration.append(mutation(generation[i-900]))
    return  newgeneration

def initVironment():
    environment = []
    for i in range(environment_num):
        temp=[]
        a=np.random.rand(2)
        a= np.array([640,480])*a
        temp.append(a)
        b = np.random.rand(2)
        b = np.array([640, 480]) * b
        temp.append(b)
        c = np.random.rand()
        c = 2*math.pi*c
        temp.append(c)
        environment.append(temp)
    return environment

def save(genes):
    np.savez('genes.npz', genes[0][0],genes[0][1])

if __name__ == '__main__':




    if(weight_path == None):
        #初始化第一代
        generation = []
        for i in  range(population):
            weitght=[]
            weight0 = np.random.randn(2, 4)
            weitght.append(weight0)
            weight1 = np.random.randn(4, 2)
            weitght.append(weight1)
            generation.append(weitght)
    else:
        generation = []
        loader = np.load(weight_path)
        generation.append(loader['arr_0'])
        generation.append(loader['arr_1'])

    environment = initVironment()
    temp = []
    #进化
    #是否需要打印fitness的变化情况
    for i in range(generation_num):


        z=genesort(generation,environment)
        score =np.array([x[1] for x in z])
        generation_sorted = [x[0] for x in z]
        print('i is ---------------',i)
        print(score.max())
        # print(generation_sorted[0])
        temp.append(score.mean())
        print(score.mean())
        generation=newpopulation(generation_sorted)
        save(generation)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('generation')
    ax.set_ylabel('fitness')


    ax.plot(temp)
    plt.show()

