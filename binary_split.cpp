#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/hash_policy.hpp>
#include <chrono>

using namespace std;
using namespace __gnu_pbds;

#define ll long long

const int MAX_COORD_COUNT = 20000005;
const double EPSILON = 1e-8;
const double POSITIVE_INF = 1e18;

//coordinate of index i
double coord_list[MAX_COORD_COUNT];


double value_accum[MAX_COORD_COUNT], slope_accum[MAX_COORD_COUNT];

int coord_count = 0;
gp_hash_table<double, int> coord_to_index;

struct PointData {
    double value;
    double index;
    double slope;

    PointData(double _value = 0, double _index = 0, double _slope = 0)
        : value(_value), index(_index), slope(_slope) {}
};

struct PiecewiseFunction {
    vector<double> x_coords;
    vector<double> slopes;
    vector<double> func_values;

    /*
    store the piecewise linear function f
    x_coords[i] : x coordinate of i-th breakpoint
    slopes[i] : the slope after the x_coords[i]
    func_values[i]: f(x_coords[i])
    */

    double query_min(double left, double right) {// return the minimal value in [l,r]
        double result = POSITIVE_INF;
        for (int i = 0; i < x_coords.size(); ++i) {
            if (left <= x_coords[i] && x_coords[i] <= right)
                result = min(result, func_values[i]);
            if (i && x_coords[i - 1] < left && left <= x_coords[i])
                result = min(result, func_values[i - 1] + slopes[i - 1] * (left - x_coords[i - 1]));
            if (i && x_coords[i - 1] < right && right <= x_coords[i])
                result = min(result, func_values[i - 1] + slopes[i - 1] * (right - x_coords[i - 1]));
        }
        return result;
    }

    double evaluate(double x) const {// return value of f(x)
        if (x_coords.empty()) return 0;
        if (x < x_coords[0]) return func_values[0] - (slopes[0] - 2) * (x_coords[0] - x);
        int left = 0, right = x_coords.size() - 1, result = 0;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (x_coords[mid] <= x) result = mid, left = mid + 1;
            else right = mid - 1;
        }
        return func_values[result] + slopes[result] * (x - x_coords[result]);
    }

    //return f(x),index of x'(maxinum coord index that <= x),slopes[x'];
    PointData query(double x) const {
        if (x < x_coords[0])
            return PointData(func_values[0] - (slopes[0] - 2) * (x_coords[0] - x), 0, slopes[0] - 2);
        int left = 0, right = x_coords.size() - 1, result = 0;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (x_coords[mid] <= x) result = mid, left = mid + 1;
            else right = mid - 1;
        }
        return PointData(func_values[result] + slopes[result] * (x - x_coords[result]), result + 1, slopes[result]);
    }

    // return f(a),f'(a),the slope of f'(a)
    PointData find_critical_point(double x) const {
        PointData tmp = query(x);
        ll total_size = x_coords.size();
        ll left = tmp.index - 1, right = total_size - 1, res = left;
        if (left < 0) left = 0, res = 0;
        if (left + 1 < total_size && func_values[left] > tmp.value)
            ++left, ++res;
        while (left <= right) {
            ll mid = (left + right) / 2;
            if (func_values[mid] <= tmp.value) res = mid, left = mid + 1;
            else right = mid - 1;
        }
        if (slopes[res] == 0) return PointData(POSITIVE_INF, tmp.value, slopes[res]);
        double fa = x_coords[res] + (tmp.value - func_values[res]) / slopes[res];
        return PointData(fa, tmp.value, slopes[res]);
    }
};

//build the piecewise linear function on pointset
PiecewiseFunction build_piecewise_function(const vector<double> &a) {
    PiecewiseFunction f;
    double initial_value = 0;
    double slope = a.size();
    for (double x : a) f.x_coords.emplace_back(x);
    for (double x : a) initial_value += x - a[0];
    f.func_values.emplace_back(initial_value);
    slope = -slope + 2;
    f.slopes.emplace_back(slope);
    for (int i = 1; i < a.size(); ++i) {
        initial_value += (a[i] - a[i - 1]) * slope;
        slope += 2;
        f.func_values.emplace_back(initial_value);
        f.slopes.emplace_back(slope);
    }
    return f;
}

//Reduce the point in function to coordinate in [l,r]
void reduce_function_domain(PiecewiseFunction &f, double left, double right) {
    if (f.x_coords.empty()) return;
    if (f.x_coords.back() <= left) {
        double last_x = f.x_coords.back(), last_f = f.func_values.back(), last_slope = f.slopes.back();
        f.x_coords = {left};
        f.func_values = {last_f + last_slope * (left - last_x)};
        f.slopes = {last_slope};
        return;
    }
    int i = -1;
    while (i + 1 < f.x_coords.size() && f.x_coords[i + 1] <= left) ++i;
    i--;
    if (i >= 0) {
        f.x_coords.erase(f.x_coords.begin(), f.x_coords.begin() + i + 1);
        f.func_values.erase(f.func_values.begin(), f.func_values.begin() + i + 1);
        f.slopes.erase(f.slopes.begin(), f.slopes.begin() + i + 1);
    }
    while (f.x_coords.size() > 1 && right < f.x_coords.back()) {
        f.x_coords.pop_back();
        f.func_values.pop_back();
        f.slopes.pop_back();
    }
}

vector<PiecewiseFunction> function_pool;

//merge func_indices into base_func in range[v[l],v[r]]
void merge_functions(PiecewiseFunction &res, const PiecewiseFunction &base_func, const vector<int> &func_indices, int left, int right) {
    double last_slope;
    int pos;

    //update value and slope in array to merge functions
    for (const int &idx : func_indices) {
        const PiecewiseFunction *f = &function_pool[idx];
        PointData pd = f->query(coord_list[left]);
        value_accum[left] += pd.value;
        slope_accum[left] += pd.slope;
        last_slope = pd.slope;
        for (int i = round(pd.index); i < f->x_coords.size(); ++i) {
            if (f->x_coords[i] > coord_list[right]) break;
            pos = coord_to_index[f->x_coords[i]];
            slope_accum[pos] += f->slopes[i] - last_slope;
            last_slope = f->slopes[i];
        }
    }

    if (!base_func.x_coords.empty()) {
        PointData pd = base_func.query(coord_list[left]);
        value_accum[left] += pd.value;
        slope_accum[left] += pd.slope;
        last_slope = pd.slope;
        for (int i = round(pd.index); i < base_func.x_coords.size(); ++i) {
            if (base_func.x_coords[i] > coord_list[right]) break;
            pos = coord_to_index[base_func.x_coords[i]];
            slope_accum[pos] += base_func.slopes[i] - last_slope;
            last_slope = base_func.slopes[i];
        }
    }

    res.x_coords.clear();
    res.func_values.clear();
    res.slopes.clear();

    double acc = 0, slope = 0;
    double prev_coord = coord_list[left];
    for (int i = left; i <= right; ++i) {
        if (slope_accum[i] || value_accum[i]) {
            acc += slope * (coord_list[i] - prev_coord) + value_accum[i];
            slope += slope_accum[i];
            res.x_coords.emplace_back(coord_list[i]);
            res.func_values.emplace_back(acc);
            res.slopes.emplace_back(slope);
            prev_coord = coord_list[i];
        }
    }

    for (int i = left; i <= right; ++i)
        value_accum[i] = slope_accum[i] = 0;

}

pair<double, int> calculate_minimum_in_row(int row_idx, int lb, int rb, const PiecewiseFunction &EA, const PiecewiseFunction &EB, const vector<int> &F) {
    //calc the minimal value of F in (a,[lb,rb])
    //update value and slope 
    double min_val = POSITIVE_INF;
    int min_col = -1;
    if (lb < row_idx) lb = row_idx;

    for (const int &idx : F) {
        PiecewiseFunction *f = &function_pool[idx];
        PointData pd = f->find_critical_point(coord_list[row_idx]);
        double fa = pd.value;
        if (fa <= coord_list[lb]) {
            value_accum[lb] += pd.index;
        } else {
            //when f(b) <= f(a) choose b
            PointData pd_lb = f->query(coord_list[lb]);
            value_accum[lb] += pd_lb.value;
            slope_accum[lb] += pd_lb.slope;
            for (int i = pd_lb.index; i < f->x_coords.size(); ++i) {
                if (f->x_coords[i] > coord_list[rb] || f->x_coords[i] > fa) break;
                int pos = coord_to_index[f->x_coords[i]];
                slope_accum[pos] += 2;
            }
            // when f(a) < f(b),choose a
            if (coord_list[rb] >= fa) {
                int l = lb, r = rb, res = -1;
                while (l <= r) {
                    int mid = (l + r) / 2;
                    if (fa <= coord_list[mid]) res = mid, r = mid - 1;
                    else l = mid + 1;
                }
                double override_val = f->query(coord_list[res]).value;
                value_accum[res] = value_accum[res] - override_val + pd.index;
                slope_accum[res] -= pd.slope;
            }
        }
    }
    //calc value in each coordinate
    double acc = 0, slope = 0;
    double val_a = EA.evaluate(coord_list[row_idx]);
    for (int b = lb; b <= rb; ++b) {
        double val_b = EB.evaluate(coord_list[b]);
        acc += slope * (coord_list[b] - coord_list[b - 1]) + value_accum[b];
        double res_val = val_a + val_b + acc;
        slope += slope_accum[b];
        if (res_val < min_val) min_val = res_val, min_col = b;
    }

    for (int i = lb; i <= rb; ++i)
        slope_accum[i] = value_accum[i] = 0;

    return {min_val, min_col};
}


struct AnswerType {
    double value;
    int row, col;
    AnswerType(double _val = POSITIVE_INF, int _row = 0, int _col = 0)
        : value(_val), row(_row), col(_col) {}

    bool operator<(const AnswerType &other) const {
        return value < other.value;
    }
};

//divide and conquer in ([la,ra],[lb,rb])
//EA,EB: the sum of function choose a,b
//F: set of funtions

AnswerType divide_and_conquer(int la, int ra, int lb, int rb, PiecewiseFunction &EA, PiecewiseFunction &EB, vector<int> &F) {
    if (la > ra) return AnswerType();

    reduce_function_domain(EA, coord_list[la], coord_list[ra]);
    reduce_function_domain(EB, coord_list[lb], coord_list[rb]);
    for (int idx : F)
        reduce_function_domain(function_pool[idx], coord_list[min(la, lb)], coord_list[max(ra, rb)]);

    if (la == ra) {
        auto res = calculate_minimum_in_row(la, lb, rb, EA, EB, F);
        return AnswerType(res.first, la, res.second);
    }

    if (F.empty()) {
        double val = EA.query_min(coord_list[la], coord_list[ra]) + EB.query_min(coord_list[lb], coord_list[rb]);
        return AnswerType(val, la, lb);
    }

    int mid = (la + ra) / 2;
    auto [min_val, min_pos] = calculate_minimum_in_row(mid, lb, rb, EA, EB, F);
    vector<int> left_set, right_set;
    PiecewiseFunction EA_right, EB_left;

    for (int idx : F) {
        if (function_pool[idx].evaluate(coord_list[mid]) <= function_pool[idx].evaluate(coord_list[min_pos]))
            left_set.push_back(idx);
        else
            right_set.push_back(idx);
    }

    merge_functions(EA_right, EA, left_set, mid + 1, ra);
    merge_functions(EB_left, EB, right_set, lb, min_pos);

    AnswerType ans(min_val, mid, min_pos);
    AnswerType left_ans = divide_and_conquer(la, mid - 1, lb, min_pos, EA, EB_left, left_set);
    AnswerType right_ans = divide_and_conquer(mid + 1, ra, min_pos, rb, EA_right, EB, right_set);

    return min(ans, min(left_ans, right_ans));
}

void solve(const vector<vector<double>> &point_sets) {
    for (const auto &ps : point_sets)
        for (const auto &x : ps)
            coord_list[++coord_count] = x;


    auto start_time = std::chrono::high_resolution_clock::now();
    sort(coord_list + 1, coord_list + coord_count + 1);
    coord_count = unique(coord_list + 1, coord_list + coord_count + 1) - coord_list - 1;

    for (int i = 1; i <= coord_count; ++i)
        coord_to_index[coord_list[i]] = i;

    vector<int> func_indices;
    for (const auto &ps : point_sets) {
        func_indices.emplace_back(function_pool.size());
        function_pool.emplace_back(build_piecewise_function(ps));
    }

    PiecewiseFunction EA, EB;
    AnswerType result = divide_and_conquer(1, coord_count, 1, coord_count, EA, EB, func_indices);

    auto end_time = chrono::high_resolution_clock::now();
    //printf("Our_result: %.10lf Our_time: %.10lf\n",result.value,((double)(end_time-start_time).count()/1000000000));

    FILE *output = fopen("cpp_result.txt", "w");
    fprintf(output, "Our_result: %.10lf Our_time: %.10lf\n",result.value,((double)(end_time-start_time).count()/1000000000));
    //fprintf(output, "Our_result: %.10lf\\n", result.value);
    fclose(output);
}

int main() {
    //freopen("tmp_data.txt","r",stdin);
    int n, k;
    scanf("%d%d", &n, &k);
    vector<vector<double>> point_sets(k);
    for (int i = 0; i < k; ++i) {
        int m; scanf("%d", &m);
        point_sets[i].resize(m);
        for (int j = 0; j < m; ++j) {
            scanf("%lf", &point_sets[i][j]);
        }
        sort(point_sets[i].begin(), point_sets[i].end());
    }
    solve(point_sets);
    return 0;
}
