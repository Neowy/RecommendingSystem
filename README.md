# RecommendingSystem
该文件夹下包含两个文件：
  1. Reco_v1.py 这是gym接口内的文件,对象为RecommendingSystem.
  2. Recoingym.py 这是gym接口外的文件,对象为Q_learning,可以不断学习更新agent的Q_value,并
    据此获得最佳策略.
一. 问题描述：
  参考论文: A reinforcement learning approach to personalized learning recommendation systems 
  其中的study III,现在有11个课程,共有11项技能,一些技能之间有着偏序关系.每次选择一个课程,要求在学习
  6个课程之后获得最多的技能.
二. 对环境的说明:
  1. 环境中的几个变量:
      self.alpha,记录当前状态,即agent已经掌握了多少技能.
      self.D 一个矩阵,记录多项技能之间的偏序关系,即记录在掌握某个技能之前需要掌握哪些其他技能.
      self.MS 一个矩阵,记录课程和技能之间的偏序关系.
  2. env的class中有函数step,即执行一个给定的action之后,返回reward,和next_alpha等变量.同时在
      env内部更新self.alpha.
三. 对训练过程的说明
  1. 为了能用矩阵q_value记录价值函数,需要将状态转化为hash数字,方便q_value数组进行索引.
  2. 训练过程共走1000个轮回(可调),即1000个完整序列. 序列中每步结束都会更新q_value矩阵.
  
