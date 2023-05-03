# Qiskit使用

## 💿环境安装

首先安装anaconda，然后依次输入以下命令

```shell
conda create -n qiskit_env python=3.8
conda activate qiskit_env
pip install qiskit
```

下载jupyter，并使用qiskit_env的环境

```shell
conda install ipykernel
python -m ipykernel install --user --name=qiskit_env
jupyter notebook
```

## ⛏检测环境是否成功

```python
import qiskit
#输出qiskit的版本
qiskit.__qiskit_version__
```

需要用api去调用IBM公司的量子计算机，自行去https://quantum-computing.ibm.com/这个网站注册账号，复制key

```python
from qiskit import IBMQ
#access the ibm quantum device
IBMQ.save_account('your key')
IBMQ.load_account()
```

> 但是好像你需要额外下载一个库，因为这是好久之前的教程，需要在qiskit_env中运行下面命令
>
> pip install qiskit_ibm_provider

## 🧪Hello World

```py
#调用库
from qiskit import *
#创建两个量子寄存器和两个经典寄存器
qr = QuantumRegister(2)
cr = ClassicalRegister(2)
#build a circuit
circuit=QuantumCircuit(qr,cr)
#调用绘图包
import matplotlib
%matplotlib inline
circuit.draw()
```

### 👀解释

> 在量子计算中，**|0⟩ 和 |1⟩** 分别代表一个量子比特的两个基本状态。量子比特（也称为量子位或 qubit）是量子计算的基本单位，类似于经典计算中的比特（bit）。然而，与经典比特不同，量子比特可以同时处于 |0⟩ 和 |1⟩ 状态的叠加态。
>
> 这里的 |0⟩ 和 |1⟩ 是用 Dirac 符号（也称为 bra-ket 符号）表示的。Dirac 符号在量子力学和量子计算中广泛使用，它提供了一种简洁的表示复杂量子态的方法。在这种表示法中，|...⟩ 符号表示一个量子态，称为 ket。
>
> 量子比特的状态可以表示为 |0⟩ 和 |1⟩ 状态的线性组合，如下所示：
>
> ```markdown
> |ψ⟩ = α |0⟩ + β |1⟩
> 
> ```
>
> 
>
> Hadamard 门（H门）和 CNOT 门是量子计算中使用的两种基本量子逻辑门。以下是关于这两种门的简要解释：
>
> 1. **Hadamard 门（H门）：**
>
> Hadamard 门是一种单量子比特门，它将一个量子比特的基态（|0⟩ 或 |1⟩）映射到一个处于叠加态的量子比特。具体而言，Hadamard 门将 |0⟩ 映射到 (|0⟩ + |1⟩)/√2，将 |1⟩ 映射到 (|0⟩ - |1⟩)/√2。Hadamard 门的矩阵表示形式如下：
>
> ```markdown
> H = (1/√2) * | 1  1 |
>               | 1 -1 |
> ```
>
> 通过将一个量子比特的基态变换为叠加态，Hadamard 门允许我们在量子计算中利用叠加原理。
>
> 1. **CNOT 门：**
>
> CNOT 门（Controlled NOT 门）是一种双量子比特门，它对两个量子比特执行条件性操作。CNOT 门有一个控制量子比特和一个目标量子比特。当控制量子比特处于 |1⟩ 状态时，CNOT 门对目标量子比特执行 NOT 操作（即翻转目标量子比特的状态）。如果控制量子比特处于 |0⟩ 状态，目标量子比特保持不变。CNOT 门的矩阵表示形式如下：
>
> ```markdown
> CNOT = | 1 0 0 0 |
>        | 0 1 0 0 |
>        | 0 0 0 1 |
>        | 0 0 1 0 |
> 
> ```
>
> CNOT 门是一种非常重要的量子逻辑门，因为它可以用来生成和操作纠缠态，纠缠态是量子计算中的另一个关键概念。
>
> 这两种量子门与经典逻辑门的一个重要区别是，它们都是可逆的，即它们可以在不丢失信息的情况下相互撤销。这使得量子计算可以在更大的计算空间中进行操作，实现经典计算难以完成的任务。

### 👻人话

就是把两个基态的量子比特（这里设为q0，q1）放在电路上，让第一个量子比特处于叠加态，然后加一个CNOT门去控制另一个量子比特 ，这样，q0就能控制q1的状态，简称，我怎么变你怎么变。注意测量的时候我们肯定测不出来叠加态，量子比特会根据测量结果立即塌缩为相应的基态。在测量之前，量子比特存在于两个状态的叠加，但测量过程将其强制为一个确定的状态。

**如何测量？**

测量结果将存储在经典寄存器中。通过执行量子电路并获取测量结果，我们可以观察到量子比特处于 0 或 1 的概率分布。

## 🕋代码

全部代码在仓库里 ，但把key抹掉了，用自己的去

执行完会发现在模拟器中理想情况下的贝尔态电路真正执行起来还是会出现小概率的01和10 为啥呢，可能是科技不够先进，量子比特很容易受到外界环境影响的呢



![ch5jpfqccl2b15q3naeg_histogram (1)](https://raw.githubusercontent.com/lijianye521/images/master/ch5jpfqccl2b15q3naeg_histogram%20(1).png)
