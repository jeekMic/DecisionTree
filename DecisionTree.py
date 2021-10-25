
# coding = gbk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # 标准化模块，这里暂时用不到
from sklearn.model_selection import train_test_split  # 数据集划分库
from sklearn.feature_extraction import DictVectorizer  # 特征向量化
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 决策树分类和dot导出模块
import os
import xml.etree.ElementTree as ET  # 读取XML的工具
import graphviz
import pydotplus as pdp

'''
编号  土种       肥力等级  有机质1  碱解氮1  有效磷1  速效钾1
1  厚腐冲积草甸土    低     28.6     136     14.9    133
2  厚腐冲积草甸土    低     26.4     118     16.5    116
3  厚腐冲积草甸土    低     29.0     143     15.0    116
4  厚腐冲积草甸土    低     29.4     129     15.4    108
5  厚腐冲积草甸土    低     23.9     140     21.1    124
分析数据：
特征集：土种、有机质1、碱解氮1、有效磷1、速效钾1
目标集：肥力等级
因此我们需要做的就是根据特征集来形成一个决策树，来达到最优的目标集

'''

'''决策树模型'''


class SoilPrediction(object):
    def __init__(self, xmlPath):
        super(SoilPrediction, self).__init__()
        # 1、数据文件的地址
        self.file = xmlPath
        # 2、读取的数据
        self.data = None
        # 3、特征集
        self.x = None
        # 4、目标集
        self.y = None
        # 5、划分的数据集
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        # 6、转换器
        self.transfer = None
        # 7、决策树
        self.estimator = None
        # 8、决策树标签
        self.label = None

    def dataProcessing(self):
        assert self.data is not None, 'please read data first'
        """
        数据预处理
        :return:返回处理好的数据
        """
        self.x = self.data[['土种', '有机质1', '碱解氮1', '有效磷1', '速效钾1']]
        self.y = self.data['肥力等级']

    def dataSetPartitioning(self):
        """
        数据集划分
        random_state 随机数种子，保证每次的一样
        :return: 返回划分好的数据集
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, random_state=2)

    def featureEngineering(self):
        """特征工程"""
        # 创建一个转换器
        self.transfer = DictVectorizer(sparse=False)  # 构建的是一个稀疏矩阵 使用parse=False效率性能更高
        # max_depth 最大的深度是5
        self.x_train = self.transfer.fit_transform(self.x_train.to_dict(orient='records'))
        self.x_test = self.transfer.fit_transform(self.x_test.to_dict(orient='records'))
        # print(self.y_train)

    def decisionTree(self):
        """决策树"""
        # criterion 选择的算法是基尼指数

        self.estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5)
        # 运算
        self.estimator.fit(self.x_train, self.y_train)
        # 评估模型
        pre = self.estimator.predict(self.x_test)
        score = self.estimator.score(self.x_test, self.y_test)
        print(self.transfer.get_feature_names())
        print(len(self.transfer.get_feature_names()))
        export_graphviz(self.estimator,
                        out_file="tree2.dot",
                        rounded=True,
                        feature_names=self.transfer.get_feature_names())
    def drawTree(self):
        # 方法一:
        # os.system('dot -Tpng tree2.dot -o tree2.png')

       # 方法二:
        with open("tree2.dot", encoding='utf-8') as f:
            # 注意，要转换成自己电脑适配的字体，要不然，中文会乱码
            dot_graph = f.read().replace('helvetica', 'SimHei')
        graph = pdp.graph_from_dot_data(dot_graph)
        graph.write_png("demo.png")  # 这里也可以转pdf
        f.close()

        #方法二:
        # 通过 graphviz 将dot文件转化为pdf
        # with open("tree2.dot", encoding='utf-8') as f:
        #     dot_graph = f.read()
        # graph = graphviz.Source(dot_graph.replace("helvetica", "FangSong"))
        # graph.save("demo1")
        # graph.view()

    def readData(self):
        """读取xls文件"""
        self.data = pd.read_excel(io=self.file, sheet_name=3)
        self.data.replace('高', 1, inplace=True)
        self.data.replace('中', 2, inplace=True)
        self.data.replace('低', 3, inplace=True)
        feature_names = ['土种', '肥力等级', '碱解氮1', '有效磷1', '速效钾1']
        # ball_data_onehot = pd.get_dummies(self.data, columns=feature_names)
        # self.label = list(ball_data_onehot.columns)
        # print(self.label)


if __name__ == '__main__':
    model = SoilPrediction("data2012to2013.xls")
    model.readData()
    model.dataProcessing()
    model.dataSetPartitioning()
    model.featureEngineering()
    model.decisionTree()
    model.drawTree()
