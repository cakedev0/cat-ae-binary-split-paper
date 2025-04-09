import numpy as np
from itertools import product, compress
from matplotlib.ticker import MultipleLocator,FuncFormatter
import os
import random
import heapq
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
import time
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt


def sklearn_solution(S):
    start = time.time()

    S = sorted(S, key=np.median)
    X = np.repeat(np.arange(len(S)), list(map(len, S)))[:, np.newaxis]
    y = np.hstack(S)
    model = DecisionTreeRegressor(max_depth=1,criterion='absolute_error')
    model.fit(X, y)

    ans = np.sum(np.abs(y - model.predict(X)))
    duration = time.time()-start

    return ans,duration

def lightgbm_solution(S):
    start = time.time()
    S_sorted = sorted(S, key=np.median)
    X = np.repeat(np.arange(len(S_sorted)), list(map(len, S_sorted)))[:, np.newaxis]
    y = np.hstack(S_sorted)
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        num_leaves=2,
        max_depth=1,
        n_estimators=1
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    ans = np.sum(np.abs(y - y_pred))
    duration = time.time()-start
    return ans,duration

def new_median_heuristic(S):
    S = sorted(S, key=np.median)
    K = len(S)
    res = []
    heapmx = []
    heapres = []
    nn = 0
    sum1 = 0
    sum2 = 0
    for st in S:
        nn += len(st)
        for x in st:
            if len(heapmx)==0 :
                heapq.heappush(heapmx, x)
                sum1+=x
            else :
                now=heapmx[0]
                if x <= now:
                    heapq.heappush(heapres,-x)
                    sum2+=x
                else:
                    sum2 += heapmx[0]
                    sum1 -= heapmx[0]
                    sum1 += x
                    heapq.heappush(heapres,-heapmx[0])
                    heapq.heapreplace(heapmx, x)
        mid = (nn+1)//2
        while len(heapmx) < mid:
            vl = -heapq.heappop(heapres)
            sum2 -= vl
            sum1 += vl
            heapq.heappush(heapmx,vl)
        res.append(sum1-heapmx[0]*len(heapmx)+heapmx[0]*len(heapres)-sum2)
    heapmx = []
    heapres = []
    nn = 0
    sum1 = 0 
    sum2 = 0
    for i in range(K-1,0,-1):
        st = S[i]
        nn += len(st)
        for x in st:
            if len(heapmx)==0 :
                heapq.heappush(heapmx, x)
                sum1+=x
            else :
                now=heapmx[0]
                if x <= now:
                    heapq.heappush(heapres,-x)
                    sum2+=x
                else:
                    sum2 += heapmx[0]
                    sum1 -= heapmx[0]
                    sum1 += x
                    heapq.heappush(heapres,-heapmx[0])
                    heapq.heapreplace(heapmx, x)
        mid = (nn+1)//2
        while len(heapmx) < mid:
            vl = -heapq.heappop(heapres)
            sum2 -= vl
            sum1 += vl
            heapq.heappush(heapmx,vl)
        res[i-1]+=(sum1-heapmx[0]*len(heapmx)+heapmx[0]*len(heapres)-sum2)
    return min(res)

def our_solution(S):
    os.system("binary_split.exe < tmp_data.txt")
    with open('cpp_result.txt', 'r') as f:
        for line in f:
            numbers = line.split()
            num1 = float(numbers[0])
            num2 = float(numbers[1])
            return num1,num2
    #os.system("v1.exe")

result_file=open("./result.txt","w")

def benchmark(S, cases, N, K, feature_name):
    time = []
    ans = []
    print(f'feature: {feature_name}, data size: {N}, categories: {K}',file=result_file)
    for name, func in cases:
        if name=='sklearn':
            if N>100000: #dataset to large for sklearn to run, time set to 1000
                result = new_median_heuristic(S)
                duration = 1000 
            else:
                result,duration = func(S)
            #result,duration = func(S)
            ans.append(result)
            time.append(duration)
            print(f'{name} : time: {duration}, result: {result}',file=result_file)
            print(f'{N} {K} {duration} {result}')
        else:
            result,duration = func(S)
            time.append(duration)
            ans.append(result)
            print(f'{name} : time: {duration}, result: {result}',file=result_file)
            print(f'{N} {K} {duration} {result}')
    return ans,time

def load_data():
    data, meta = arff.loadarff('dataset_42729.arff') # name of dataset file
    df = pd.DataFrame(data)

    # name of features of dataset
    group_by_feature = ['RatecodeID','DOLocationID','PULocationID','passenger_count','extra','mta_tax','tolls_amount','total_amount','lpep_pickup_datetime_day','lpep_pickup_datetime_hour']

    # name of target feature
    target_feature = 'tip_amount'  
    S = []
    for feature in group_by_feature:
        grouped = df.groupby(feature)
        S.append([feature,[group[target_feature].tolist() for group_name, group in grouped if not group.empty]])
    return S
    
def gen_data(S,n): # subsample S to size of n
    sizes = [len(sublist) for sublist in S]
    total_size = sum(sizes)
    
    if n > total_size:
        raise ValueError("m cannot be larger than the total number of elements in S.")
    
    proportions = [size / total_size for size in sizes]
    num_samples = [int(p * n) for p in proportions]
    
    #num_samples[-1] += n - sum(num_samples)
    diff = n - sum(num_samples)  
    for i in reversed(range(len(S))):  
        if diff == 0:
            break
        if num_samples[i] < sizes[i]:  
            additional = min(diff, sizes[i] - num_samples[i])
            num_samples[i] += additional
            diff -= additional

    result = [
        random.sample(sublist, min(num, len(sublist))) if num > 0 else []
        for sublist, num in zip(S, num_samples)
    ]
    result = [ss for ss in result if len(ss)>0]
    nn = 0
    for ss in result:
        nn+=len(ss)
    return result,nn
    # if n < initial_count:
    #     data = [[] for _ in range(len(S))]
    #     for i in range(n):
    #         chosen_index = random.choices(range(len(S)), weights=probabilities, k=1)[0]
    #         data[chosen_index].append(random.choice(S[chosen_index])) 
    #     data = [s for s in data if len(s)>0]
    #     return data
    

def draw_figure(list_n,sklearn_time,our_time,lightgbm_time,test_name):
    fig, ax = plt.subplots()
    tim_limit = max(max(our_time),max(lightgbm_time))*2
    ax.set_ylim(0, tim_limit)
    ax.plot(list_n, sklearn_time, label='sklearn', color='blue', linestyle='-')
    ax.plot(list_n, our_time, label='our', color='red', linestyle='-')
    ax.plot(list_n, lightgbm_time, label='lightgbm', color='green', linestyle='-')


    x_major_interval = 1e6  
    x_minor_interval = 1e6 / 5  

    y_major_interval = 1  
    y_minor_interval = 0.2 

    ax.xaxis.set_major_locator(MultipleLocator(x_major_interval))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_interval))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}")) 

    ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_interval))

    # 添加标签
    ax.set_xlabel(r"$N \times 10^6$")
    ax.set_ylabel("Running time(second)")

    # 显示图像
    plt.legend()
    plt.savefig('feature_{}.png'.format(test_name))

def truncate_dataframe(df, decimals):
    factor = 10 ** decimals
    return df.apply(lambda x: np.trunc(x * factor) / factor)

def draw_table(list_n,list_k,sklearn_ans,sklearn_time,our_ans,our_time,lightgbm_time,lightgbm_ans,test_name):
    data = {
    "n": list_n,                # n: data size
    "k": list_k,                # k: category number
    "sklearn Time": sklearn_time, 
    "sklearn Accuracy":sklearn_ans, 
    "lightgbm Time": lightgbm_time, 
    "lightgbm Accuracy": lightgbm_ans, 
    "our Time": our_time, 
    "our Result": our_ans, 
    }
    df = pd.DataFrame(data)

    integer_columns = ["n", "k"]
    for col in integer_columns:
        df[col] = df[col].apply(lambda x: str(x))

    df["our Result"] = df["our Result"].round(15)
    df["sklearn Time"] = df["sklearn Time"].round(6)
    df["sklearn Accuracy"] = truncate_dataframe(df["sklearn Accuracy"], 9)
    df["lightgbm Time"] = df["lightgbm Time"].round(6)
    df["lightgbm Accuracy"] = truncate_dataframe(df["lightgbm Accuracy"], 9)
    df["our Time"] = df["our Time"].round(6)

    # export to Markdown table
    markdown_table = df.to_markdown(index=False)
    with open('feature_{}.md'.format(test_name), "w") as md_file:
        md_file.write(markdown_table)

    # export to Excel file
    df.to_excel('feature_{}.xlsx'.format(test_name), index=False)

    # data = {
    # "n": list_n,                # 数据规模参数 n
    # "sklearn Time": sklearn_time, # 算法1运行时间
    # "lightgbm Time": lightgbm_time, # 算法2运行时间
    # "our Time": our_time, # 算法3运行时间
    # }
    # df = pd.DataFrame(data)
    # integer_columns = ["n"]
    # for col in integer_columns:
    #     df[col] = df[col].apply(lambda x: str(x))
    # df["sklearn Time"] = df["sklearn Time"].round(6)
    # df["lightgbm Time"] = df["lightgbm Time"].round(6)
    # df["our Time"] = df["our Time"].round(6)
    # markdown_table = df.to_markdown(index=False)
    # with open('feature_{}_time.md'.format(test_name), "w") as md_file:
    #     md_file.write(markdown_table)

def draw_feature_table(feature_name,list_n,list_k,sklearn_ans,sklearn_time,our_ans,our_time,lightgbm_time,lightgbm_ans):
    test_name = "taxi_tip(42729)" # name of dataset
    data = {

    "feature":feature_name,
    "n": list_n,                # n: data size 
    "k": list_k,                # k: category number
    "sklearn Time": sklearn_time, 
    "sklearn Accuracy":sklearn_ans,
    "lightgbm Time": lightgbm_time, 
    "lightgbm Accuracy": lightgbm_ans,
    "our Time": our_time, 
    "our Result": our_ans, 
    }
    df = pd.DataFrame(data)

    integer_columns = ["n", "k"]
    for col in integer_columns:
        df[col] = df[col].apply(lambda x: str(x))

    df["our Result"] = df["our Result"].round(15)
    df["sklearn Time"] = df["sklearn Time"].round(6)
    df["sklearn Accuracy"] = truncate_dataframe(df["sklearn Accuracy"], 9)
    df["lightgbm Time"] = df["lightgbm Time"].round(6)
    df["lightgbm Accuracy"] = truncate_dataframe(df["lightgbm Accuracy"], 9)
    df["our Time"] = df["our Time"].round(6)

    # export to Markdown table
    markdown_table = df.to_markdown(index=False)
    with open('{}.md'.format(test_name), "w") as md_file:
        md_file.write(markdown_table)

    # export to Excel file
    df.to_excel('{}.xlsx'.format(test_name), index=False)    

def main():
    seed = int(time.time())  
    np.random.seed(seed)
    test_case = 0
    test_set = load_data()
    #test_n = [0,10**5,10**6,5*(10**6),10**7]
    #test_n = []
    #test_n = [0.02*i for i in range(1,51)]
    #test_n = np.repeat(test_n,3)
    feature_name = []
    list_n = []
    list_k = []
    sklearn_time = []
    sklearn_ans = []
    our_time = []
    our_ans = []
    lightgbm_time = []
    lightgbm_ans = []
    for name,preS in test_set:
        # lightgbm_solution(preS)
        # list_n = []
        # list_k = []
        # sklearn_time = []
        # sklearn_ans = []
        # our_time = []
        # our_ans = []
        # lightgbm_time = []
        # lightgbm_ans = []
        test_case += 1
        NN = 0
        for ss in preS:
            NN += len(ss)    
        # now_n = [x for x in test_n if x<=NN]
        now_n = [NN]
        # now_n = [int(x*NN) for x in test_n]
        # now_n.extend(range(1000,10000,1000))
        # now_n.extend(range(10000,100000,10000))
        # now_n = sorted(now_n)
        for n in now_n:
            S,n = gen_data(preS,n)
            k = len(S)
            feature_name.append(name)
            list_n.append(n)
            list_k.append(k)
            
            input_file=open("./tmp_data.txt","w")
            #input_file=open("./data.txt","w")
            print(f'{n} {k}',file=input_file)
            for E in S:
                print(np.size(E),file=input_file)
                for x in E:
                    print(x,file=input_file,end=' ')
                print(file=input_file)
            input_file.close()
            
            ans,duration = benchmark(S, [
                #['brute_force', brute_force_solution],
                #['smawk', smawk_base_solution],
                ['sklearn', sklearn_solution],
                ['our',our_solution],
                ['lightgbm', lightgbm_solution],
                ],
                n,k,name)
            sklearn_time.append(duration[0])
            our_time.append(duration[1])
            lightgbm_time.append(duration[2])
            sklearn_ans.append(ans[1]/ans[0])
            our_ans.append(ans[1])
            lightgbm_ans.append(ans[1]/ans[2])
        #draw_figure(list_n,sklearn_time,our_time,lightgbm_time,name)
        #draw_table(list_n,list_k,sklearn_ans,sklearn_time,our_ans,our_time,lightgbm_time,lightgbm_ans,name)
        #plt.show()
    draw_feature_table(feature_name,list_n,list_k,sklearn_ans,sklearn_time,our_ans,our_time,lightgbm_time,lightgbm_ans)

if __name__ == "__main__":
    main()