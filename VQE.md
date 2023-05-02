# VQE

## 介绍

> 何为哈密顿量？
>
> 哈密顿量是物理学中一个重要的概念，用于描述一个物理系统的总能量。在量子力学中，哈密顿量可以看作是一个厄米（Hermitian）算符，它对应着一个物理系统的总能量算符。哈密顿量的本征值对应着物理系统的能量量子态，而哈密顿量的本征态则对应着物理系统的能量本征态。
>
> 在量子计算中，哈密顿量通常用于描述量子比特和量子门之间的相互作用。例如，可以将哈密顿量表示为一组基本的量子门操作，从而将一个量子算法转化为一个基于量子门操作的电路。在量子化学中，哈密顿量也被用来描述分子的总能量，从而可以通过求解哈密顿量的本征值和本征态来计算分子的性质，如分子的基态能量、分子结构等。
>
> VQE是Variational Quantum Eigensolver的缩写。它是一种量子算法，用于近似计算特定分子或材料的基态能量。VQE算法被设计为在量子计算机上运行，它可以用来解决经典计算机难以有效解决的问题。
>
> VQE的基本思想是，使用量子计算的能力来构建一个量子态，通过量子态来计算一个期望值，这个期望值可以代表哈密顿量的本征值。然后通过经典优化算法来最小化这个期望值，从而得到哈密顿量的最小本征值。



## 实现逻辑

1. 导入所需的库和模块。

2. 定义哈密顿量（Hamiltonian） H。

3. 使用NumPyMinimumEigensolver（一种经典算法）计算哈密顿量 H 的最小特征值，作为后面 VQE 结果的基准。

4. 定义量子神经网络（QNN），在这里选择了 `TwoLocal` 电路作为 QNN。

5. 设置优化器，这里使用了 COBYLA。

6. 设置后端，这里选择了 Qiskit Aer 的状态向量模拟器。

7. 使用 VQE 算法计算哈密顿量 H 的最小特征值。

8. 将计算出的 NumPy 和 VQE 的最小特征值打印到屏幕上。

   ### 量子电路图

   ![image-20230501214710873](https://raw.githubusercontent.com/lijianye521/images/master/image-20230501214710873.png)

> 这个电路包括了一系列参数化的单量子比特旋转门和两量子比特 CNOT 门。这些门通过堆叠、重复或连接在一起的方式组成了电路结构。具体来说，这个电路包含了：
>
> - 初始 Hadamard 门用于在 |0> 和 |1> 之间产生叠加态；
> - 一系列的 RX、RY 和 RZ 门，这些门可以旋转量子比特的 Bloch 球坐标系中的角度；
> - CNOT 门用于在量子比特之间产生纠缠态；
> - 与初始 Hadamard 门对称的一系列 RX、RY 和 RZ 门，这些门用于对量子比特进行反向旋转，使得它们回到经典态。

## 代码

```py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal

# 定义 Hamiltonian
H = (0.5 * Z ^ I) + (0.5 * I ^ Z) + (0.5 * Z ^ Z) + (0.5 * X ^ X)

# 使用 NumPy 求解最小特征值（用作基准）
numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(H)
print("NumPy 最小特征值: ", result.eigenvalue.real)

# 定义量子神经网络（QNN）
qnn = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz", entanglement="linear", reps=3)

# 使用 matplotlib 绘制量子电路
qnn.draw("mpl")
plt.show()

# 设置优化器
optimizer = COBYLA(maxiter=500)

# 设置后端
backend = Aer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(backend, shots=1000)

# 使用 VQE 求解最小特征值
vqe = VQE(ansatz=qnn, optimizer=optimizer, quantum_instance=quantum_instance)
vqe_result = vqe.compute_minimum_eigenvalue(H)
print("VQE 最小特征值: ", vqe_result.eigenvalue.real)

```

以上代码定义了一个简单的 Hamiltonian，并使用 NumPy 求解器作为基准。接下来，我们定义一个量子神经网络（QNN）作为 VQE 的 ansatz，并设置一个优化器（这里使用了 COBYLA）。最后，我们使用 Qiskit 的 VQE 算法求解 Hamiltonian 的最小特征值。