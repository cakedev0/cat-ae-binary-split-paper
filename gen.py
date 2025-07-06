import numpy as np 
from matplotlib.ticker import MultipleLocator, FuncFormatter
import os
import random
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
import time
import pandas as pd
from scipy.io import arff
#import arff
#import heapq
#from itertools import product, compress


def sklearn_solution(S, N):
    # Compute the solution using sklearn's DecisionTreeRegressor for a given set S with N elements.
    # Returns the minimum Mean Absolute Error (MAE) and the computation time.

    if N > 100000:  # If the dataset is too large for sklearn, return 0 and time set to 1000s.
        return 0, 1000
    else:
        # Use sklearn DecisionTreeRegressor with depth=1 and absolute error loss
        start = time.time()
        S = sorted(S, key=np.median)
        X = np.repeat(np.arange(len(S)), list(map(len, S)))[:, np.newaxis]
        y = np.hstack(S)
        model = DecisionTreeRegressor(max_depth=1, criterion='absolute_error')
        model.fit(X, y)
        ans = np.sum(np.abs(y - model.predict(X)))
        duration = time.time() - start
        return ans, duration

def lightgbm_solution(S, N):
    # Compute solution using LightGBM with single decision stump (max_depth=1, num_leaves=2).
    # Returns MAE and time consumed.
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
    duration = time.time() - start
    return ans, duration



def our_solution(S, N):
    # Run the external C++ binary "binary_split.exe" with current dataset stored in tmp_data.txt.
    # Results are read from cpp_result.txt.
    # Returns the MAE and computation time from our custom algorithm.

    k = len(S)
    input_file = open("./tmp_data.txt", "w")
    print(f'{N} {k}', file=input_file)
    for E in S:
        print(np.size(E), file=input_file)
        for x in E:
            print(x, file=input_file, end=' ')
        print(file=input_file)
    input_file.close()

    os.system("binary_split.exe < tmp_data.txt")
    with open('cpp_result.txt', 'r') as f:
        for line in f:
            numbers = line.split()
            num1 = float(numbers[1])
            num2 = float(numbers[3])
            return num1, num2


result_file = open("./result.txt", "w")

def benchmark(S, cases, N, K, feature_name):
    # Run benchmarking for multiple methods and write results to result.txt
    
    time = []
    ans = []
    print(f'feature: {feature_name}, data size: {N}, categories: {K}', file=result_file)
    for name, func in cases:
        result, duration = func(S, N)
        time.append(duration)
        ans.append(result)
        print(f'{name} time: {duration}, result: {result}', file=result_file)
        print(f'{N} {K} {duration} {result}')
    print('', file=result_file)
    return ans, time



def load_data():
    #data = pd.read_csv('dataset_40753.csv')  # Input CSV file
    data, meta = arff.loadarff('dataset_42225.arff')  # Input ARFF file
    #print(len(data))
    df = pd.DataFrame(data)

    # Features to group by
    group_by_feature = ['carat','color','table','x']
    target_feature = 'price'  # Target variable

    S = []
    for feature in group_by_feature:
        grouped = df.groupby(feature)
        S.append([feature, [group[target_feature].tolist() for group_name, group in grouped if not group.empty]])
    return S



def gen_data(S, n):

    # Subsample data S to total size n while maintaining feature-wise distribution.
    # Returns a list of lists, where each sublist corresponds to a category and contains totally n elements

    sizes = [len(sublist) for sublist in S]
    total_size = sum(sizes)

    if n > total_size:
        raise ValueError("n cannot be larger than the total number of elements in S.")

    proportions = [size / total_size for size in sizes]
    num_samples = [int(p * n) for p in proportions]

    # Ensure total count matches requested size
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
    result = [ss for ss in result if len(ss) > 0]
    nn = sum(len(ss) for ss in result)
    return result, nn
    # if n < initial_count:
    #     data = [[] for _ in range(len(S))]
    #     for i in range(n):
    #         chosen_index = random.choices(range(len(S)), weights=probabilities, k=1)[0]
    #         data[chosen_index].append(random.choice(S[chosen_index])) 
    #     data = [s for s in data if len(s)>0]
    #     return data


def draw_figure(list_n, sklearn_time, our_time, lightgbm_time, test_name):
    #Draws a line plot comparing the running times of three algorithms: sklearn, our solution, and LightGBM.
    #Each point on the x-axis represents a different data size (N), and the corresponding y-value is the time taken.
    #The plot is saved as a PNG file named 'feature_{test_name}.png'.

    fig, ax = plt.subplots()

    tim_limit = max(max(our_time), max(lightgbm_time)) * 2
    ax.set_ylim(0, tim_limit)

    # Plot running time curves for each method with distinct colors and labels
    ax.plot(list_n, sklearn_time, label='sklearn', color='blue', linestyle='-')
    ax.plot(list_n, our_time, label='our', color='red', linestyle='-')
    ax.plot(list_n, lightgbm_time, label='lightgbm', color='green', linestyle='-')

    x_major_interval = 1e6            # major tick every 1 million
    x_minor_interval = 1e6 / 5        # minor tick every 0.2 million

    y_major_interval = 1             # major tick every 1 second
    y_minor_interval = 0.2           # minor tick every 0.2 seconds

    ax.xaxis.set_major_locator(MultipleLocator(x_major_interval))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_interval))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}"))  # convert x ticks to millions

    ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_interval))

    ax.set_xlabel(r"$N \times 10^6$")
    ax.set_ylabel("Running time (seconds)")

    plt.legend()

    plt.savefig(f'feature_{test_name}.png')



def truncate_dataframe(df, decimals):
    factor = 10 ** decimals
    return df.apply(lambda x: np.trunc(x * factor) / factor)


def draw_table(list_n, list_k, sklearn_ans, sklearn_time, our_ans, our_time, lightgbm_time, lightgbm_ans, test_name):
    #Generates a comparative performance table for a single feature, including running times and accuracy of three methods: sklearn, LightGBM, and our custom method.
    #The results are saved both as a Markdown file and an Excel spreadsheet, named using the feature name.

    data = {
        "n": list_n,
        "k": list_k,
        "sklearn Time": sklearn_time,
        "sklearn Accuracy": sklearn_ans,
        "lightgbm Time": lightgbm_time,
        "lightgbm Accuracy": lightgbm_ans,
        "our Time": our_time,
        "our Result": our_ans,
    }
    df = pd.DataFrame(data)
    for col in ["n", "k"]:
        df[col] = df[col].astype(str)

    df["our Result"] = df["our Result"].round(15)
    df["sklearn Time"] = df["sklearn Time"].round(6)
    df["sklearn Accuracy"] = truncate_dataframe(df["sklearn Accuracy"], 9)
    df["lightgbm Time"] = df["lightgbm Time"].round(6)
    df["lightgbm Accuracy"] = truncate_dataframe(df["lightgbm Accuracy"], 9)
    df["our Time"] = df["our Time"].round(6)

    markdown_table = df.to_markdown(index=False)
    with open(f'feature_{test_name}.md', "w") as md_file:
        md_file.write(markdown_table)

    df.to_excel(f'feature_{test_name}.xlsx', index=False)


def draw_feature_table(feature_name, list_n, list_k, sklearn_ans, sklearn_time, our_ans, our_time, lightgbm_time, lightgbm_ans):
    
    # Generates a summary table comparing the performance of sklearn, LightGBM, and our custom method across multiple features on the full dataset.
    # The table includes running time and accuracy metrics for each method, and is saved as both a Markdown file and an Excel file named after the dataset.
    
    test_name = "diamonds_42225"  # Name of the dataset used for testing
    data = {
        "feature": feature_name,
        "n": list_n,
        "k": list_k,
        "sklearn Time": sklearn_time,
        "sklearn Accuracy": sklearn_ans,
        "lightgbm Time": lightgbm_time,
        "lightgbm Accuracy": lightgbm_ans,
        "our Time": our_time,
        "our Result": our_ans,
    }
    df = pd.DataFrame(data)
    for col in ["n", "k"]:
        df[col] = df[col].astype(str)

    df["our Result"] = df["our Result"].round(15)
    df["sklearn Time"] = df["sklearn Time"].round(6)
    df["sklearn Accuracy"] = truncate_dataframe(df["sklearn Accuracy"], 9)
    df["lightgbm Time"] = df["lightgbm Time"].round(6)
    df["lightgbm Accuracy"] = truncate_dataframe(df["lightgbm Accuracy"], 9)
    df["our Time"] = df["our Time"].round(6)

    markdown_table = df.to_markdown(index=False)
    with open(f'{test_name}.md', "w") as md_file:
        md_file.write(markdown_table)

    df.to_excel(f'{test_name}.xlsx', index=False)


# Entry point of the script. Loads data, runs experiments, and saves summary results.
def main():
    seed = int(time.time())
    np.random.seed(seed)
    test_case = 0
    test_set = load_data()
    feature_name = []
    list_n = []
    list_k = []
    sklearn_time = []
    sklearn_ans = []
    our_time = []
    our_ans = []
    lightgbm_time = []
    lightgbm_ans = []
    for name, preS in test_set:
        test_case += 1
        NN = sum(len(ss) for ss in preS)

        #now_n = [NN*i//10 for i in range(1,11)]
        now_n = [NN]  # Adjust this to try different data sizes
        for n in now_n:
            S, n = gen_data(preS, n)
            k = len(S)
            feature_name.append(name)
            list_n.append(n)
            list_k.append(k)

            ans, duration = benchmark(S, [
                ['sklearn', sklearn_solution],
                ['our', our_solution],
                ['lightgbm', lightgbm_solution],
            ], n, k, name)
            sklearn_time.append(duration[0])
            our_time.append(duration[1])
            lightgbm_time.append(duration[2])
            sklearn_ans.append(ans[1] / ans[0])
            our_ans.append(ans[1])
            lightgbm_ans.append(ans[1] / ans[2])
        #draw_figure(list_n,sklearn_time,our_time,lightgbm_time,name)
        #draw_table(list_n,list_k,sklearn_ans,sklearn_time,our_ans,our_time,lightgbm_time,lightgbm_ans,name)
        #plt.show()
    draw_feature_table(feature_name, list_n, list_k, sklearn_ans, sklearn_time, our_ans, our_time, lightgbm_time, lightgbm_ans)


if __name__ == "__main__":
    main()
