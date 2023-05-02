# QAOA

## 简介

QAOA（Quantum Approximate Optimization Algorithm）是一种用于求解优化问题的量子算法。它是由Farhi、Goldstone和Gutmann在2014年提出的，其基本思想是将优化问题转化为一个哈密顿量，并使用量子计算机来模拟该哈密顿量的演化过程，从而得到最优解的近似值。

在QAOA算法中，优化问题的目标函数被编码为一个哈密顿量，其中每个变量对应于哈密顿量中的一个项。然后，使用一组参数化的量子门来模拟该哈密顿量的演化过程，通过调整这些参数以最大化期望的目标函数值，从而得到最优解的近似值。

QAOA算法的主要优点是它可以在量子计算机上实现，并且在一些特定的优化问题上表现得比经典算法更好。然而，QAOA算法的实际效果还取决于许多因素，例如优化问题的复杂度和哈密顿量的参数化方式等。

## 最大切问题

> 最大切问题是一个图论中的问题，通常指的是在一个连通无向图中，找到一条边，使得将图沿着这条边割开后得到的两个连通分量的节点数之差最大。
>
> 最大切问题可以被看作是最小割问题的对偶问题。最小割问题是找到一组边，将图分成两个不相交的连通分量，并且这些边的权重之和最小。而最大切问题则是找到一条边，将图分成两个不相交的连通分量，并且这条边的权重最大。

## 使用qiskit实现逻辑

1.创建一个简单的图

使用NetworkX库创建一个包含4个节点和5条边的无向图，并使用matplotlib.pyplot库绘制该图。

2.计算Max-Cut问题的权重矩阵

对于给定的无向图，计算其Max-Cut问题的权重矩阵，即一个大小为4x4的矩阵，其中weights[i,j]表示节点i和节点j之间的边的权重。在这个例子中，所有边的权重都是1。

3.定义QuadraticProgram对象

使用Qiskit Optimization库的Maxcut模块将权重矩阵转化为一个二次规划(Quadratic Programming)问题。这里的二次规划问题是将最大割问题转化为一个二次优化问题，并且在这个问题中，目标函数的系数是权重矩阵，即maxcut.to_quadratic_program()会返回一个包含权重矩阵信息的QuadraticProgram对象。

4.将二次规划问题转换为QUBO

使用Qiskit Optimization库的LinearEqualityToPenalty和QuadraticProgramToQubo转换器，将二次规划问题转换为QUBO问题(Quadratic Unconstrained Binary Optimization)，即将问题转化为只包含二次项和一次项的式子，并且所有变量的取值只能是0或1。LinearEqualityToPenalty转换器将线性等式转换为惩罚项，QuadraticProgramToQubo转换器将QuadraticProgram对象转换为QUBO问题。

5.获得QUBO的Ising模型

将QUBO问题转换为Ising模型，即将二次项和一次项分别对应为矩阵的对角线和向量，计算得到Ising模型的哈密顿量和偏置项。

6.使用QAOA求解Max-Cut问题

使用Qiskit库的QAOA算法(Quantum Approximate Optimization Algorithm)求解Max-Cut问题。QAOA算法是一种近似优化算法，使用量子电路近似地求解最优解。在这个例子中，设置reps=2表示使用深度为2的QAOA电路，quantum_instance=backend表示使用statevector_simulator作为量子仿真器。

7.提取结果

将求解得到的QUBO问题的最优解转换为Max-Cut问题的解，并输出最优解和最优值。

## 🔥代码

```py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumCircuit
from qiskit_optimization.applications import Maxcut
from qiskit.utils import algorithm_globals
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo, LinearEqualityToPenalty

# 创建一个简单的图
num_nodes = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
graph = nx.Graph(edges)
nx.draw(graph, with_labels=True)
plt.show()

# 计算Max-Cut问题的权重矩阵
weights = np.zeros([num_nodes, num_nodes])
for i, j in graph.edges():
    weights[i, j] = 1
    weights[j, i] = 1

# 定义QuadraticProgram对象
maxcut = Maxcut(weights)
qp = maxcut.to_quadratic_program()

# 将二次规划问题转换为QUBO
linear2penalty = LinearEqualityToPenalty()
qp_wo_linear = linear2penalty.convert(qp)
qp_to_qubo = QuadraticProgramToQubo()
qubo = qp_to_qubo.convert(qp_wo_linear)

# 获得QUBO的Ising模型
qubit_op, offset = qubo.to_ising()

# 使用QAOA求解Max-Cut问题
seed = 123
algorithm_globals.random_seed = seed
backend = Aer.get_backend("statevector_simulator")
qaoa = QAOA(reps=2, quantum_instance=backend)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qubo)

# 提取结果
x = result.x
print(f"Optimal solution: {x}")
print(f"Optimal value: {maxcut.interpret(x)}")
```

