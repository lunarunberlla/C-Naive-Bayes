import numpy as np


######定以一个Naive_Bayes类，来完成作业一和作业二。我们设置他的Mode='Homework_1'/'Homework_2'#############

class Naive_Bayes:

    def __init__(self,Mode):
        self.Mode=Mode

    def Gauss(self,avarge,variance,x):  #高斯分布的概率密度函数，avarge=均值，variance=方差
        ###############由于高斯分布是一种连续型随机变量，所以他在莫一个的概率值为0。所以我们这里用，其概率密度函数值来代替概率值。############
        return 1/(np.sqrt(2*np.pi)*variance)*np.exp(-(x-avarge)**2/(2*variance**2))


    def DataSet(self):
        if self.Mode=='Homework_1':
            FilePath='./heightandweight.csv'
            DataSetAll=open(FilePath).read()
            DataSetAll=DataSetAll.split('\n')
            DataSetAll=DataSetAll[1:len(DataSetAll)-1]
            Data=[ i.split(',') for i in DataSetAll]
            self.Data=Data
            ####Data=[['6', '180', '12', '男'], ['5.92', '190', '11', '男'], ['5.98', '170', '12', '男'], ['5.92', '165', '10', '男'],
            # ['5', '100', '6', '女'], ['5.5', '150', '8', '女'], ['5.42', '130', '7', '女'], ['5.75', '150', '9', '女']]
            return Data
        elif self.Mode=='Homework_2':
            FilePath='./西瓜数据3.0.csv'
            DataSetAll = open(FilePath).read()
            DataSetAll = DataSetAll.split('\n')
            DataSetAll = DataSetAll[1:len(DataSetAll) - 1]
            Data = [i.split(',') for i in DataSetAll]
            self.Data=Data
            #####self.Data=[['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.46', '1'], ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '0.774', '0.376', '1'],.....

    def Feature_scaling_No(self):   ########未缩放的###############
        ####由数据可以看出来，这些属性值都是连续分布的######
        ####我们需要计算他们每一个属性分别的均值和方差######
        _TrainData=self.Data
        Parameter_Yes=[]  ##存放特征为正例的均值和方差
        Parameter_No=[]   ##存放特征为负例的均值和方差

        ########################################################################################################
        ###############计算男的均值和方差############################################################################
        Middle = []  # 处理数据的中间变量
        for item in np.where(np.array(_TrainData)[:,len(_TrainData[0])-1]=='男')[0]:
            Middle.append([float(i) for i in _TrainData[item][0:len(_TrainData[item])-1]])
        ####Middle=[[6.0, 180.0, 12.0], [5.92, 190.0, 11.0], [5.98, 170.0, 12.0], [5.92, 165.0, 10.0]]###

        for item in range(len(Middle[0])):
            F=[] #中间变量
            F.append((np.sum(np.array(Middle)[:,item]))/len(Middle))
            #for i in np.array(Middle)[:,item]:
            #   (i-(np.sum(np.array(Middle)[:,item]))/len(Middle))
            #计算方差
            F.append((np.sum(np.sqrt((np.array(Middle)[:,item]-(np.sum(np.array(Middle)[:,item]))/len(Middle))*(np.array(Middle)[:,item]-(np.sum(np.array(Middle)[:,item]))/len(Middle)))))/len(Middle))
            Parameter_Yes.append(F)
        #Parameter_Yes=[[5.955, 0.03500000000000014], [176.25, 8.75], [11.25, 0.75]]
        #########################################################################################################
        #########################################################################################################
        ##############计算女的均值和方差############################################################################
        Middle = []  # 处理数据的中间变量
        for item in np.where(np.array(_TrainData)[:, len(_TrainData[0]) - 1] == '女')[0]:
            Middle.append([float(i) for i in _TrainData[item][0:len(_TrainData[item]) - 1]])
        ####Middle=[[6.0, 180.0, 12.0], [5.92, 190.0, 11.0], [5.98, 170.0, 12.0], [5.92, 165.0, 10.0]]###

        for item in range(len(Middle[0])):
            F = []  # 中间变量
            F.append((np.sum(np.array(Middle)[:, item])) / len(Middle))
            # for i in np.array(Middle)[:,item]:
            #   (i-(np.sum(np.array(Middle)[:,item]))/len(Middle))
            # 计算方差
            F.append((np.sum(np.sqrt((np.array(Middle)[:, item] - (np.sum(np.array(Middle)[:, item])) / len(Middle)) * (
                        np.array(Middle)[:, item] - (np.sum(np.array(Middle)[:, item])) / len(Middle))))) / len(Middle))
            Parameter_No.append(F)
        # Parameter_No=[[5.4175, 0.20874999999999977], [132.5, 17.5], [7.5, 1.0]]

        return Parameter_No,Parameter_Yes

    def Feature_scaling_Yes(self): #############缩放的####################
        ####由数据可以看出来，这些属性值都是连续分布的######
        ####我们需要计算他们每一个属性分别的均值和方差######
        _TrainData=self.Data
        Parameter_Yes=[]  ##存放特征为正例的均值和方差
        Parameter_No=[]   ##存放特征为负例的均值和方差

        ########################################################################################################
        ###############计算男的均值和方差############################################################################
        Middle = []  # 处理数据的中间变量
        for item in np.where(np.array(_TrainData)[:,len(_TrainData[0])-1]=='男')[0]:
            Middle.append([float(i) for i in _TrainData[item][0:len(_TrainData[item])-1]])
        ####Middle=[[6.0, 180.0, 12.0], [5.92, 190.0, 11.0], [5.98, 170.0, 12.0], [5.92, 165.0, 10.0]]###
        ############################################特征缩放##########################################################
        middle=[]  ###储存缩放后的特征信息
        for item in range(len(Middle[0])):
            #print(float(np.max(np.array(Middle)[:,item])))
            #print(np.array(Middle)[:,item].tolist())
            middle.append((((np.array(Middle)[:,item])/(float(np.max(np.array(Middle)[:,item])))).tolist()))
        middle = (np.array(middle).T.tolist())
        #middle=[[1.0, 0.9473684210526315, 1.0], [0.9866666666666667, 1.0, 0.9166666666666666],
        #      [0.9966666666666667, 0.8947368421052632, 1.0], [0.9866666666666667, 0.868421052631579, 0.8333333333333334]]
        #
        #########################################计算均值和方差#############################################################
        Middle=middle
        for item in range(len(Middle[0])):
            F=[] #中间变量
            F.append((np.sum(np.array(Middle)[:,item]))/len(Middle))
            #for i in np.array(Middle)[:,item]:
            #   (i-(np.sum(np.array(Middle)[:,item]))/len(Middle))
            #计算方差
            F.append((np.sum(np.sqrt((np.array(Middle)[:,item]-(np.sum(np.array(Middle)[:,item]))/len(Middle))*(np.array(Middle)[:,item]-(np.sum(np.array(Middle)[:,item]))/len(Middle)))))/len(Middle))
            Parameter_Yes.append(F)

        #Parameter_Yes=[[0.9925, 0.005833333333333329], [0.9276315789473684, 0.046052631578947345], [0.9375, 0.0625]]
        #########################################################################################################
        #########################################################################################################
        ##############计算女的均值和方差############################################################################
        Middle = []  # 处理数据的中间变量
        for item in np.where(np.array(_TrainData)[:, len(_TrainData[0]) - 1] == '女')[0]:
            Middle.append([float(i) for i in _TrainData[item][0:len(_TrainData[item]) - 1]])
        ####Middle=[[6.0, 180.0, 12.0], [5.92, 190.0, 11.0], [5.98, 170.0, 12.0], [5.92, 165.0, 10.0]]###
        middle = []  ###储存缩放后的特征信息
        for item in range(len(Middle[0])):
             #print(float(np.max(np.array(Middle)[:,item])))
             #print(np.array(Middle)[:,item].tolist())
             #print(((np.array(Middle)[:, item]) / (float(np.max(np.array(Middle)[:, item])))).tolist())
             middle.append(((np.array(Middle)[:, item]) / (float(np.max(np.array(Middle)[:, item])))).tolist())
        middle=(np.array(middle).T.tolist())
        #####################################计算均值和方差########################################################
        Middle=middle
        for item in range(len(Middle[0])):
            F = []  # 中间变量
            F.append((np.sum(np.array(Middle)[:, item])) / len(Middle))
            # for i in np.array(Middle)[:,item]:
            #   (i-(np.sum(np.array(Middle)[:,item]))/len(Middle))
            # 计算方差
            F.append((np.sum(np.sqrt((np.array(Middle)[:, item] - (np.sum(np.array(Middle)[:, item])) / len(Middle)) * (
                        np.array(Middle)[:, item] - (np.sum(np.array(Middle)[:, item])) / len(Middle))))) / len(Middle))
            Parameter_No.append(F)

        # Parameter_No=[[0.9421739130434783, 0.036304347826086936], [0.8833333333333333, 0.11666666666666667], [0.8333333333333333, 0.1111111111111111]]
        return Parameter_No,Parameter_Yes


    def HomeWork1_Testing(self):


        #假定我们现在给定测试数据集[6,180,12,'男']
        ###########测试未进行数据缩放的####################
        ##########N0是女，Yes是男########################
        ParamNo,ParamYes=Naive_Bayes.Feature_scaling_No(self)
        ######计算数据集[6,180,12,'男']是男的概率##########
        #####P=P（男）*P（6|男）*P(180|男)*P（12|男）######
        count=0
        PYes=1
        for i in [6,180,12]:
            PYes=PYes*Naive_Bayes.Gauss(self,ParamYes[count][0],ParamYes[count][1],i)
            count=count+1
        ######计算数据集[6,180,12,'男']是女的概率##########
        #####P=P（女）*P（6|女）*P(180|女)*P（12|女）######
        count = 0
        PNo = 1
        for i in [6, 180, 12]:
            PNo = PNo * Naive_Bayes.Gauss(self, ParamNo[count][0], ParamNo[count][1], i)
            count = count + 1

        if PNo>PYes:
            print("未缩放的测试：")
            print('男人的概率为：{} 女人的概率为：{} 男女概率比为：{}'.format(PYes, PNo, int(PYes / PNo)))
            print('所以该人为女性')
            print('------------------------------------------------------------')
        else:
            print("未缩放的测试：")
            print('男人的概率为：{} 女人的概率为：{} 男女概率比为：{}'.format(PYes, PNo,int(PYes/PNo)))
            print('所以该人为男性')
            print('------------------------------------------------------------')
        ###########测试进行数据缩放的####################
        ##########N0是女，Yes是男########################
        ParamNo, ParamYes = Naive_Bayes.Feature_scaling_Yes(self)
        ######计算数据集[6,180,12,'男']是男的概率##########
        #####P=P（男）*P（6|男）*P(180|男)*P（12|男）######
        ######因为只有一个测试数据，所以特征都变为了[1,1,1]####
        count = 0
        PYes = 1
        for i in [1, 1, 1]:
            PYes = PYes * Naive_Bayes.Gauss(self, ParamYes[count][0], ParamYes[count][1], i)
            count = count + 1
        ######计算数据集[6,180,12,'男']是女的概率##########
        #####P=P（女）*P（6|女）*P(180|女)*P（12|女）######
        count = 0
        PNo = 1
        for i in [1, 1, 1]:
            PNo = PNo * Naive_Bayes.Gauss(self, ParamNo[count][0], ParamNo[count][1], i)
            count = count + 1

        if PNo > PYes:
            print("缩放的测试：")
            print('男人的概率为：{} 女人的概率为：{} 男女概率比为：{}'.format(PYes, PNo, int(PYes / PNo)))
            print('所以该人为女性')
            print('------------------------------------------------------------')
        else:
            print("缩放的测试：")
            print('男人的概率为：{} 女人的概率为：{} 男女概率比为：{}'.format(PYes, PNo, int(PYes / PNo)))
            print('所以该人为男性')
            print('------------------------------------------------------------')
    def isFloat(x):  #####判断一个字符串是否可以转变为浮点型数值数据###########
        try:
            float(x)
            if str(x) in ['inf', 'infinity', 'INF', 'INFINITY', 'True', 'NAN', 'nan', 'False', '-inf', '-INF',
                          '-INFINITY', '-infinity', 'NaN', 'Nan']:
                return False
            else:
                return True
        except:
            return False

    def Laplace(self):
        Data=self.Data
        #Data=[['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '0.697', '0.46', '1'], ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '0.774', '0.376', '1'],.....
        _ZERO=[]
        _ONE=[]
        #######################数据处理，将每一个属性的所有属性值按照划分提取出来#######################################
        for item in np.where(np.array(Data)[:,-1]=='1')[0]:
            _ONE.append([float(i) if Naive_Bayes.isFloat(i) else i for i in Data[item][0:len(Data[0])-1]])
        _ONETrain=[]
        for value in np.array(_ONE).T:
            _ONETrain.append([float(i) if Naive_Bayes.isFloat(i) else i for i in value])
        ###################################计算概率值#############################################################
        ParOneCot=[]
        ParOneDit={}
        ParOneDitList=[]
        apha=0
        for value in _ONETrain:
            if Naive_Bayes.isFloat(value[0]): ################################连续型的利用高斯公式计算,储存均值和方差################
                F=[]
                F.append(np.sum(value)/len(value))        #加入均值
                F.append(np.sum(np.sqrt((value-np.sum(value)/len(value))**2))/len(value)) #加入方差
                ParOneCot.append(F)                       #储存参数的链表
            else:
                for item in sorted(set(value)):
                    ParOneDit.update({item:(value.count(item))/(len(value)+len(set(np.array(Data)[:,apha])))})  #######拉普拉斯修正
            ParOneDitList.append(list(set(np.array(Data)[:,apha])))
            apha=apha+1
        ParOneDitList.append((len(_ONETrain[0])))  #####观察次数的长度
        ParOneDit.update({'one':(len(_ONETrain)+1)/(len(Data)+2)})

        #ParOneDit={'乌黑': 0.45454545454545453, '浅白': 0.18181818181818182, '青绿': 0.36363636363636365, '稍蜷': 0.4, '蜷缩': 0.6, '沉闷': 0.3, '浊响': 0.7, '清晰': 0.8, '稍糊': 0.2, '凹陷': 0.6, '稍凹': 0.4, '硬滑': 0.7, '软粘': 0.3}
        ##########################################计算为0的参数###########################################################################################################
            #######################数据处理，将每一个属性的所有属性值按照划分提取出来#######################################
        _ONE=[]
        for item in np.where(np.array(Data)[:, -1] == '0')[0]:
            _ONE.append([float(i) if Naive_Bayes.isFloat(i) else i for i in Data[item][0:len(Data[0]) - 1]])
        _ONETrain = []
        for value in np.array(_ONE).T:
            _ONETrain.append([float(i) if Naive_Bayes.isFloat(i) else i for i in value])
        ###################################计算概率值#############################################################
        ParZeroCot = []
        ParZeroDit = {}
        ParZeroDitList=[]
        apha=0
        for value in _ONETrain:
            if Naive_Bayes.isFloat(value[0]):  ################################连续型的利用高斯公式计算,储存均值和方差################
                F = []
                F.append(np.sum(value) / len(value))  # 加入均值
                F.append(np.sum(np.sqrt((value - np.sum(value) / len(value)) ** 2)) / len(value))  # 加入方差
                ParZeroCot.append(F)  # 储存参数的链表
            else:
                for item in sorted(set(value)):
                    ParZeroDit.update({item: (value.count(item) ) / (len(value)+len(set(np.array(Data)[:,apha]))) })
            apha=apha+1
            ParZeroDitList.append(list(set(np.array(Data)[:,apha])))
        ParZeroDitList.append(len(_ONETrain[0]))   #####观察次数的长度
        ParZeroDit.update({'zero':(len(_ONETrain)+1)/(len(Data)+2)})

        return ParZeroCot,ParZeroDit,ParZeroDitList,ParOneCot,ParOneDit,ParOneDitList

    def HomeWork2_Testing(self):
        ParZeroCot, ParZeroDit, ParZeroDitList, ParOneCot, ParOneDit, ParOneDitList=Naive_Bayes.Laplace(self)
        #################################################################################################
        ############测试数据集为['青绿'，'蜷缩'，'浊响'，'清晰'，'凹陷'，'硬滑'，0.697，0.460]####################
        ############P(是|X)=P(是|青绿)P(是|蜷缩)P(是|浊响)P(是|清晰)P(是|凹陷)P(是|硬滑)P(是|0.697)P(是|0.460)P(是)
        ############P(否|X)=P(否|青绿)P(否|蜷缩)P(否|浊响)P(否|清晰)P(否|凹陷)P(否|硬滑)P(否|0.697)P(否|0.460)P(否)
        ###计算P(是|X)
        _TestData=['青绿','蜷缩','浊响','清晰','凹陷','硬滑',0.697,0.460]
        POne=1
        count=0
        for i in _TestData:
            if Naive_Bayes.isFloat(i):
                    POne=POne*Naive_Bayes.Gauss(self,float(ParOneCot[0][0]),float(ParOneCot[0][1]),float(i))
            elif i in ParOneDit:
                    POne=POne*ParOneDit.get(i)

            else:
                POne=POne*(1/(ParOneDitList[-1] +len(ParOneDitList[count])))
            count=count+1
        POne=POne*ParOneDit.get(list(ParOneDit.keys())[-1])
        #================================================================================================###
        ###计算P(是|X)
        _TestData = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
        PZero = 1
        count = 0
        for i in _TestData:
            if Naive_Bayes.isFloat(i):
                PZero = PZero * Naive_Bayes.Gauss(self, float(ParZeroCot[0][0]), float(ParZeroCot[0][1]), float(i))
            elif i in ParZeroDit:
                PZero = PZero * ParZeroDit.get(i)

            else:
                PZero = PZero * (1 / (ParZeroDitList[-1] + len(ParZeroDitList[count])))
            count = count + 1
        PZero = PZero * ParZeroDit.get(list(ParZeroDit.keys())[-1])
        #===============================================================================================##
        if POne>PZero:
            print('------------------------------------------------------------')
            print("进行了拉普拉斯平滑的：")
            print("POne:{} > PZero:{}".format(POne,PZero))
            print("该瓜是好瓜")
            print('------------------------------------------------------------')
        else:
            print('------------------------------------------------------------')
            print("进行了拉普拉斯平滑的：")
            print("POne:{} > PZero:{}".format(POne, PZero))
            print("该瓜是坏瓜")
            print('------------------------------------------------------------')

    def Testing(self):
        if self.Mode=='Homework_1':
            Naive_Bayes.DataSet(self)
            Naive_Bayes.HomeWork1_Testing(self)
        elif self.Mode=='Homework_2':
            Naive_Bayes.DataSet(self)
            Naive_Bayes.HomeWork2_Testing(self)

A=Naive_Bayes(Mode='Homework_2')
A.Testing()



