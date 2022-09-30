from tqdm import tqdm

N = 10
# 该情况下将<AU1编号为1,>P1编号为2...,<VSVG编号为23
q_1_3_3 = '>LoxP,<AU1,<LoxP,>P1,>LoxN,<AU5,<LoxN,>P2,>Lox2272,<B,<Lox2272,>P3,' \
          '>LoxP,<E1,<LoxP,>P4,>LoxN,<E2,<LoxN,>P5,>Lox2272,<Flag,<Lox2272,>P6,' \
          '>LoxP,<HA,<LoxP,>P7,>LoxN,<OLLAS,<LoxN,>P8,>Lox2272,<SI,<Lox2272,>P9,' \
          '>LoxP,<Strep2,<LoxP,>P10,>LoxN,<V5,<LoxN,>P11,>Lox2272,<VSVG,<Lox2272'
# 进一步地，把问题list(1 2 3 ... 22 23)，每个数字代表一个非Lox开头的字符，比如1代表<AU1,且用-1代表>AU1
result = {0: list()}
result[0].append(tuple(range(1, 24)))
# 同向重组相当于去掉下标i...(i+6)/i...(i+12)/i...(i+18)的元素
# 对于反向重组，假设编号为i的元素为第一个反向重组元素，
# 若编号为奇数（下标偶数），比如<AU1，则可能的情况包括 i本身反向/i...(i+6)反向/i...(i+12)反向/...
# 若编号为偶数（下标奇数），比如>P1，则可能的情况包括i...(i+4)反向/i...(i+10)反向/...
# 例如对<E2，其编号为9为奇数，则需要反转的有<E2/<E2~<OLLAS/<E2~<V5

for n in range(1, N):
    result[n] = list()
    for l in tqdm(result[n - 1]):
        # 同向重组,去掉下标连续的6/12/18个元素
        for i in range(0, len(l)):  # 遍历l中的下标
            for j in range(5, len(l)-i, 6):  # 遍历5/11/17等可能的下标增量
                result[n].append(l[:i]+l[i+j+1:])
        # 反向重组，根据奇偶
        for i in range(1, len(l), 2):  # 遍历编号为偶数（数组下标为奇数）的情况
            for j in range(4, len(l)-i, 6):
                result[n].append(l[:i] + tuple([-item for item in reversed(l[i:i+j+1])]) + l[i+j+1:])
        for i in range(0, len(l), 2):  # 遍历编号为奇数（数组下标为偶数）的情况
            for j in range(6, len(l)-i, 6):
                result[n].append(l[:i] + tuple([-item for item in reversed(l[i:i + j + 1])]) + l[i + j + 1:])
    result[n] = list(set(result[n]))
    result[n].sort()
    print(f"n={n},有{len(result[n])}种不同重组")


