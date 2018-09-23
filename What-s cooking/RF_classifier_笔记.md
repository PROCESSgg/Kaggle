一、json文件的读取

  给出的数据使用json格式存取，所以涉及到json库，使用方法是：
  
    import json
    with open(path) as f:
        data = json.load(f)
  先导入json库，使用文件流，打开json文件所在路径，load函数实现。该处返回的是一个list格式数据。
  
二、数据特征
  
  给出的训练数据分为'id','cuisine','ingreadients'三种，'cuisine'是label，'ingreadients'是特征；
  
  首先，分类特征是'cuisine'是字符类型，所以先把它转化为数字类型，使用的是LabelEncoder()，会将字符型数据，按字典序转化为对应数字索引。要注意在预测之后，要将数值型输出，转化为字符型，这里就用到了LabelEncoder().inverse_transform()。
  
  然后，离散特征'ingreadients'也是字符型，刚开始考虑使用现成的one-hot工具，可是需要填入array类型数据，这个特征是list型，直接转化为array时，遇到了这个问题，就是array的每一行并不等长，也就是说每个菜品，使用的材料数目不同，那就要填充；而且看了one-hot工具的介绍，发现比如数值型的[0,1,2]，以fit字典[1,2,3,0]来转化，成了[0001 1000 0100]为了方便看才加了空格，这样的话，经过one-hot编码之后，每一行会变得很长，扩大了好几倍，所以干脆自己写。每个样本的原材料长度设为所有原材料的长度，也就是词典库长度，将array对应位置如果被使用就设为1。
  
三、训练模型
  
  将训练数据特征和标签放入RandomForestClassifier()中，第一次使用默认参数，accuracy=0.68，第二次将决策树的数目n_estimators设置为20，默认为10，accuracy=0.7，当我再将n_estimators设为30，accuracy又下降了。
  
四、输出CSV文件

提交要求是CSV文件，所以先将test的id，和模型对于测试数据的输出，添加到pd.dataFrame中，这里有一个坑，刚开始添加顺序一直都是第一列'cuisine'第二列'id'，从网上查资料得知，如果不用column=指定顺序的话，默认是字典序。在将dataFrame输出为CSV文件。
