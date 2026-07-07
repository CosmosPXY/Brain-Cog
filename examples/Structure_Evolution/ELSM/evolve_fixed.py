import time
import threading
from threading import Thread
import os
import networkx as nx
import numpy as np
from population import *
import nsganet as engine
from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import logging
from model import *
from spikes import calc_f2
from multiprocessing import Process, Pool
from datetime import datetime
import argparse
import torch
from timm.utils import setup_default_logging

print("[INIT] ========== 程序启动 ==========")
print(f"[INIT] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[INIT] Python版本: {__import__('sys').version}")
print(f"[INIT] PyTorch版本: {torch.__version__}")
print(f"[INIT] CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INIT] CUDA设备数: {torch.cuda.device_count()}")

_logger = logging.getLogger('')
config_parser = parser = argparse.ArgumentParser(description='Evolution Config', add_help=False)

parser = argparse.ArgumentParser(description='SNN Evolving')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--seed', type=int, default=68, metavar='S')
parser.add_argument('--datapath', default='/data/', type=str, metavar='PATH')
parser.add_argument('--output', default='/data/LSM/Eresult/new', type=str, metavar='PATH')
R_mode="TEST"
if R_mode =="TEST" :
    parser.add_argument('--liquid-size', type=int, default=5)  # 5×5 = 25个变量
    parser.add_argument('--pop-size', type=int, default=2)  # 种群2个
    parser.add_argument('--up', type=int, default=20)  # 约束参数调整
    parser.add_argument('--low', type=int, default=5)  # 约束参数调整
    parser.add_argument('--n_offspring', type=int, default=1)  # 1个后代
    parser.add_argument('--n_gens', type=int, default=1)  # 仅1代
    parser.add_argument('--arand', type=float, default=10)  # 调整为10
    parser.add_argument('--brand', type=float, default=1.0)  # 调整为1.0

else:   
    parser.add_argument('--liquid-size', type=int, default=8000)
    parser.add_argument('--pop-size', type=int, default=20)
    parser.add_argument('--up', type=int, default=32000000)
    parser.add_argument('--low', type=int, default=320000)
    parser.add_argument('--n_offspring', type=int, default=200)
    parser.add_argument('--n_gens', type=int, default=2000)
    parser.add_argument('--arand', type=float, default=285)
    parser.add_argument('--brand', type=float, default=1.8)


def _parse_args():
    print("[ARGS] 解析命令行参数...")
    args_config, remaining = config_parser.parse_known_args()
    args = parser.parse_args(remaining)
    print(f"[ARGS] device={args.device}, seed={args.seed}")
    print(f"[ARGS] output={args.output}")
    print(f"[ARGS] liquid_size={args.liquid_size}, pop_size={args.pop_size}")
    print(f"[ARGS] n_offspring={args.n_offspring}, n_gens={args.n_gens}")
    return args


def calc_f1(dirs):
    print(f"[CALC_F1] 处理文件: {dirs}")
    try:
        ci = []
        G = nx.read_gpickle(dirs)
        print(f"[CALC_F1] 加载图成功, 节点数: {G.number_of_nodes()}")

        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component)
        print(f"[CALC_F1] 最大连通分量节点数: {G.number_of_nodes()}")

        for u in G.nodes:
            ci.append(nx.clustering(G, u))
        a = sum(ci)

        print("[CALC_F1] 开始计算最短路径...")
        print(f"[CALC_F1] 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        path = nx.average_shortest_path_length(G)
        print("[CALC_F1] 完成计算")
        print(f"[CALC_F1] 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        print(f"[CALC_F1] 聚类系数和: {a}, 平均最短路径: {path}")
        return a, path
    except Exception as e:
        print(f"[CALC_F1] 错误: {e}")
        raise


def mul_f1(pop, steps, rootdir):
    print(f"[MUL_F1] 开始多进程计算 (pop={pop}, steps={steps})")
    result = []
    for i in range(0, pop, steps):
        print(f"[MUL_F1] 处理批次 {i}-{i + steps}")
        p = Pool(steps)
        dirs = [os.path.join(rootdir, str(j) + '.pkl') for j in range(i, i + steps)]
        print(f"[MUL_F1] 文件列表: {dirs}")
        ret = p.map(calc_f1, dirs)
        result.extend(ret)
        print(f"[MUL_F1] 批次结果: {ret}")
        p.close()
        p.join()
    print(f"[MUL_F1] 多进程计算完成, 总结果数: {len(result)}")
    return result


class Evolve(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, args, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None):
        print(f"[EVOLVE] 初始化 Evolve 类")
        print(f"[EVOLVE] n_var={n_var}, n_obj={n_obj}, n_constr={n_constr}")
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int64)
        self.xl = lb if lb is not None else np.zeros(n_var, dtype=np.int64)
        self.xu = ub if ub is not None else np.ones(n_var, dtype=np.int64)
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.args = args
        print(f"[EVOLVE] 下界 shape: {self.xl.shape}, 上界 shape: {self.xu.shape}")

    def _evaluate(self, x, out, *args, **kwargs):
        print(f"[EVALUATE] 评估开始, 样本数: {x.shape[0]}")

        objs = np.full((x.shape[0], self.n_obj), np.nan)
        g1 = np.full((x.shape[0]), np.nan)
        g2 = np.full((x.shape[0]), np.nan)

        gen_dir = os.path.join(self.args.output, 'generaion' + str(kwargs['algorithm'].n_gen))
        print(f"[EVALUATE] 生成目录: {gen_dir}")
        os.makedirs(gen_dir, exist_ok=True)

        # np.save(os.path.join(gen_dir,"x.npy"),x)
        lsms = x.reshape(x.shape[0], self.args.liquid_size, self.args.liquid_size)
        print(f"[EVALUATE] 重塑形状: {lsms.shape}")

        for i in range(x.shape[0]):
            temp_G = nx.Graph(lsms[i])
            nx.write_gpickle(temp_G, os.path.join(gen_dir, str(i) + ".pkl"))
        print(f"[EVALUATE] 已保存 {x.shape[0]} 个图")

        print(f"[EVALUATE] 开始多进程计算 f1...")
        self.ob1 = mul_f1(pop=x.shape[0], steps=10, rootdir=gen_dir)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print(f'\n[EVALUATE] ========== 网络 {arch_id} ==========')
            _logger.info('Network= {}'.format(arch_id))
            genome = x[i, :]

            g1[i] = genome.sum() - self.args.up
            g2[i] = self.args.low - genome.sum()
            lsmm = genome.reshape(self.args.liquid_size, self.args.liquid_size)
            small_coe_a, small_coe_b = self.ob1[i]

            print(f"[EVALUATE] 转换到 CUDA 设备...")
            lsmm = torch.tensor(lsmm, device='cuda:%d' % self.args.device).float()
            crit = calc_f2(lsmm, 'cuda:%d' % self.args.device)
            objs[i, 1] = abs(crit - 1)
            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = -(small_coe_a / self.args.arand) / (small_coe_b / self.args.brand)

            _logger.info('small word= {}'.format(objs[i, 0]))
            _logger.info('criticality= {}'.format(objs[i, 1]))
            print(f"[EVALUATE] 小世界性: {objs[i, 0]}, 临界性: {objs[i, 1]}")

            self._n_evaluated += 1

        out["F"] = objs
        out["G"] = np.column_stack([g1, g2])
        print(f"[EVALUATE] 评估完成, 已评估总数: {self._n_evaluated}")
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    print(f"\n[CALLBACK] ========== 代数回调 ==========")
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    _logger.info("generation = {}".format(gen))
    print(f"[CALLBACK] 代数: {gen}")

    _logger.info("population error1: best = {}, mean = {}, "
                 "median1 = {}, worst1 = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                    np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    print(f"[CALLBACK] 目标1 - 最优: {np.min(pop_obj[:, 0]):.6f}, 平均: {np.mean(pop_obj[:, 0]):.6f}")
    _logger.info('Best1 Genome id= {}'.format(np.argmin(pop_obj[:, 0])))

    _logger.info("population error2: best = {}, mean = {}, "
                 "median2 = {}, worst2 = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                    np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))
    print(f"[CALLBACK] 目标2 - 最优: {np.min(pop_obj[:, 1]):.6f}, 平均: {np.mean(pop_obj[:, 1]):.6f}")
    _logger.info('Best2 Genome id= {}'.format(np.argmin(pop_obj[:, 1])))

    if gen % 20 == 0:
        print(f"[CALLBACK] 保存第 {gen} 代的最优基因组...")
        best_sid = np.argmin(pop_obj[:, 0])
        best_sname = '-'.join([
            'gen' + str(gen),
            's' + str(float('%.4f' % pop_obj[best_sid, 0])),
            'c' + str(float('%.4f' % pop_obj[best_sid, 1])),
        ])
        best_cid = np.argmin(pop_obj[:, 1])
        best_cname = '-'.join([
            'gen' + str(gen),
            's' + str(float('%.4f' % pop_obj[best_cid, 0])),
            'c' + str(float('%.4f' % pop_obj[best_cid, 1])),
        ])

        save_dir = '/data/save/genome'
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, best_sname + datetime.now().strftime("%Y%m%d-%H%M%S")),
                pop_var[np.argmin(pop_obj[:, 0])])
        np.save(os.path.join(save_dir, best_cname + datetime.now().strftime("%Y%m%d-%H%M%S")),
                pop_var[np.argmin(pop_obj[:, 1])])
        print(f"[CALLBACK] 已保存: {best_sname}, {best_cname}")


if __name__ == '__main__':
    print("[MAIN] 主程序开始")
    args = _parse_args()

    out_base_dir = os.path.join(args.output, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"[MAIN] 创建输出目录: {out_base_dir}")
    os.makedirs(out_base_dir, exist_ok=True)
    args.output = out_base_dir
    setup_default_logging(log_path=os.path.join(out_base_dir, 'log.txt'))
    print(f"[MAIN] 日志已设置")

    n_var = args.liquid_size * args.liquid_size
    print(f"[MAIN] 设定变量数: {n_var}")

    lb = np.zeros(n_var, dtype=np.int64)
    ub = np.ones(n_var, dtype=np.int64)
    print(f"[MAIN] 边界设定: lb shape={lb.shape}, ub shape={ub.shape}")

    print(f"[MAIN] 初始化 Evolve 问题类...")
    kkk = Evolve(args,
                 n_var=n_var,
                 n_obj=2,
                 n_constr=2,
                 lb=lb,
                 ub=ub)

    print(f"[MAIN] 初始化优化方法...")
    method = engine.nsganet(pop_size=args.pop_size,
                            sampling=RandomSampling(var_type='custom'),
                            mutation=BinaryBitflipMutation(),
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    print(f"[MAIN] 开始优化...")
    kres = minimize(kkk,
                    method,
                    callback=do_every_generations,
                    termination=('n_gen', args.n_gens))

    print(f"\n[MAIN] ========== 优化完成 ==========")
    print(f"[MAIN] 结果已保存到: {args.output}")
