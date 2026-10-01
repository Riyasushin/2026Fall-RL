#include <ctime>
#include <random>
#include <utility>
#include <iostream>
#include <algorithm>

using namespace std;

class JackCarRental{
    static const int
        MAX_CAR_1,
        MAX_CAR_2,
        MOVE_LIMIT;
    static const double
        MOVE_COST,
        RENT_PRICE,
        MEAN_REQUEST_1,
        MEAN_REQUEST_2,
        MEAN_RETURN_1,
        MEAN_RETURN_2;
    static poisson_distribution<int>
        request_1, request_2, return_1, return_2;
    public:
        typedef pair<int, int> State;
        bool verbose;
        State state(){
            return make_pair(car_1, car_2);
        }
        void set_state(int car_1, int car_2){
            if (verbose){
                cout << "State set to (" << car_1 << ", " << car_2 << ")" << endl;
            }
            this->car_1 = car_1;
            this->car_2 = car_2;
        }
        void reset(){
            if (verbose){
                cout << "Environment reset." << endl;
            }
            day = 0;
            car_1 = 0;
            car_2 = 0;
        }
        pair<State, double> step(int action){
            day ++;
            if (verbose){
                cout << "\nDay: " << day 
                    << " State: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            double reward = state_transition(action);
            if (verbose){
                cout << "\tReward: " << reward << endl;
            }
            return make_pair(state(), reward);
        }
        int sample_action(){
            int action_low = max(-car_2, -MOVE_LIMIT);
            int action_high = min(car_1, MOVE_LIMIT);
            uniform_int_distribution<int> random_action(action_low, action_high);
            int action = random_action(e);
            if (verbose){
                cout << "\tAction " << action 
                    << " sampled from uniform[" << action_low << ", " << action_high << "]" << endl;
            }
            return action;
        }
        JackCarRental(int car_1=0, int car_2=0, bool verbose=false){
            this->day = 0;
            this->verbose = verbose;
            set_state(car_1, car_2);
            e.seed(time(nullptr));
        }
    private:
        int car_1, car_2, day;
        default_random_engine e;
        
        double state_transition(int action){
            car_1 = min(car_1 - action, MAX_CAR_1);
            car_2 = min(car_2 + action, MAX_CAR_2);
            double total_move_cost = abs(action) * MOVE_COST;
            if (verbose){
                cout << "\tMove: (" << -action << ", " << action 
                    << "), cost: " << total_move_cost << endl; 
                cout << "\tAfter movement, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            int req_1 = this->request_1(e);
            int req_2 = request_2(e);
            if (verbose){
                cout << "\tRental request: (" << req_1 << ", " << req_2 << ")" << endl; 
            } 
            int rent_1 = min(car_1, req_1);
            int rent_2 = min(car_2, req_2);
            double total_income = (rent_1 + rent_2) * RENT_PRICE;
            car_1 -= rent_1;
            car_2 -= rent_2;
            if (verbose){
                cout << "\tRent: (" << rent_1 << ", " << rent_2 
                    << "), income: " << total_income << endl;
                cout << "\tAfter rent, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            int ret_1 = return_1(e);
            int ret_2 = return_2(e);
            if (verbose){
                cout << "\tCars to return: (" << ret_1 << ", " << ret_2 << ")" << endl;
            }
            car_1 = min(car_1 + ret_1, MAX_CAR_1);
            car_2 = min(car_2 + ret_2, MAX_CAR_2);
            if (verbose){
                cout << "\tAfter return, state: (" << car_1 << ", " << car_2 << ")" << endl;
            }
            return total_income - total_move_cost;
        }
};

const int
    JackCarRental::MAX_CAR_1 = 20,
    JackCarRental::MAX_CAR_2 = 20,
    JackCarRental::MOVE_LIMIT = 5;
const double 
    JackCarRental::MOVE_COST = 2.0,
    JackCarRental::RENT_PRICE = 10.0,
    JackCarRental::MEAN_REQUEST_1 = 3.0,
    JackCarRental::MEAN_REQUEST_2 = 4.0,
    JackCarRental::MEAN_RETURN_1 = 3.0,
    JackCarRental::MEAN_RETURN_2 = 2.0;
poisson_distribution<int> 
    JackCarRental::request_1(JackCarRental::MEAN_REQUEST_1),
    JackCarRental::request_2(JackCarRental::MEAN_REQUEST_2),
    JackCarRental::return_1(JackCarRental::MEAN_RETURN_1),
    JackCarRental::return_2(JackCarRental::MEAN_RETURN_2);

#include <cmath>
#include <chrono>
#include <iomanip>
#include <vector>
#include <fstream>
#include <string>

// ===================== Policy Iteration Solver =====================
// Sutton & Barto Figure 4.2 (Jack's Car Rental), gamma = 0.9
// Note: solver does not call the sampling environment above.

static const int MAX_CARS = 20;
static const int MOVE_LIMIT = 5;
static const int POISSON_TRUNC = 11;  // P(n>=11) negligible for lambda<=4
static const double MOVE_COST = 2.0;
static const double RENT_PRICE = 10.0;
static const double GAMMA = 0.9;
static const double THETA = 1e-6;

static const double MEAN_REQ[2] = {3.0, 4.0};
static const double MEAN_RET[2] = {3.0, 2.0};

static double poisson_pmf(int n, double lambda) {
    // e^{-lambda} lambda^n / n!
    double p = exp(-lambda);
    for (int i = 1; i <= n; ++i) {
        p *= lambda / i;
    }
    return p;
}

// For a single location with `cars` available after overnight move:
// expected rental reward and next-state distribution P[next_cars]
struct LocModel {
    double expected_rent_reward = 0.0;
    double next_prob[MAX_CARS + 1] = {0.0};
};

static LocModel build_loc_model(int cars, double mean_req, double mean_ret) {
    LocModel m;
    double req_p[POISSON_TRUNC + 1];
    double ret_p[POISSON_TRUNC + 1];
    double req_tail = 1.0;
    double ret_tail = 1.0;
    for (int n = 0; n < POISSON_TRUNC; ++n) {
        req_p[n] = poisson_pmf(n, mean_req);
        ret_p[n] = poisson_pmf(n, mean_ret);
        req_tail -= req_p[n];
        ret_tail -= ret_p[n];
    }
    // put remaining mass on the truncation point (rare events)
    req_p[POISSON_TRUNC] = max(0.0, req_tail);
    ret_p[POISSON_TRUNC] = max(0.0, ret_tail);

    for (int req = 0; req <= POISSON_TRUNC; ++req) {
        int rented = min(cars, req);
        m.expected_rent_reward += req_p[req] * rented * RENT_PRICE;
        int left = cars - rented;
        for (int ret = 0; ret <= POISSON_TRUNC; ++ret) {
            int nxt = min(left + ret, MAX_CARS);
            m.next_prob[nxt] += req_p[req] * ret_p[ret];
        }
    }
    return m;
}

class PolicyIteration {
public:
    double V[MAX_CARS + 1][MAX_CARS + 1];
    int policy[MAX_CARS + 1][MAX_CARS + 1];
    LocModel loc1[MAX_CARS + 1];
    LocModel loc2[MAX_CARS + 1];
    int iterations = 0;

    PolicyIteration() {
        for (int i = 0; i <= MAX_CARS; ++i) {
            for (int j = 0; j <= MAX_CARS; ++j) {
                V[i][j] = 0.0;
                policy[i][j] = 0;
            }
            loc1[i] = build_loc_model(i, MEAN_REQ[0], MEAN_RET[0]);
            loc2[i] = build_loc_model(i, MEAN_REQ[1], MEAN_RET[1]);
        }
    }

    // Expected return of taking action `a` in state (n1, n2) under current V
    double action_value(int n1, int n2, int a) const {
        // Feasibility: cannot move more cars than available
        if (a > n1 || -a > n2) return -1e18;
        if (a > MOVE_LIMIT || a < -MOVE_LIMIT) return -1e18;

        int cars1 = min(n1 - a, MAX_CARS);
        int cars2 = min(n2 + a, MAX_CARS);
        double move_cost = abs(a) * MOVE_COST;

        const LocModel& m1 = loc1[cars1];
        const LocModel& m2 = loc2[cars2];
        double q = m1.expected_rent_reward + m2.expected_rent_reward - move_cost;

        for (int i = 0; i <= MAX_CARS; ++i) {
            if (m1.next_prob[i] == 0.0) continue;
            for (int j = 0; j <= MAX_CARS; ++j) {
                if (m2.next_prob[j] == 0.0) continue;
                q += m1.next_prob[i] * m2.next_prob[j] * GAMMA * V[i][j];
            }
        }
        return q;
    }

    void policy_evaluation() {
        while (true) {
            double delta = 0.0;
            for (int n1 = 0; n1 <= MAX_CARS; ++n1) {
                for (int n2 = 0; n2 <= MAX_CARS; ++n2) {
                    double v_old = V[n1][n2];
                    V[n1][n2] = action_value(n1, n2, policy[n1][n2]);
                    delta = max(delta, fabs(V[n1][n2] - v_old));
                }
            }
            if (delta < THETA) break;
        }
    }

    bool policy_improvement() {
        bool stable = true;
        for (int n1 = 0; n1 <= MAX_CARS; ++n1) {
            for (int n2 = 0; n2 <= MAX_CARS; ++n2) {
                int old_a = policy[n1][n2];
                int best_a = old_a;
                double best_q = -1e18;
                int a_low = max(-n2, -MOVE_LIMIT);
                int a_high = min(n1, MOVE_LIMIT);
                for (int a = a_low; a <= a_high; ++a) {
                    double q = action_value(n1, n2, a);
                    // Prefer smaller |a| on ties to match common reference policies
                    if (q > best_q + 1e-9 ||
                        (fabs(q - best_q) <= 1e-9 && abs(a) < abs(best_a))) {
                        best_q = q;
                        best_a = a;
                    }
                }
                policy[n1][n2] = best_a;
                if (best_a != old_a) stable = false;
            }
        }
        return stable;
    }

    void solve() {
        iterations = 0;
        while (true) {
            ++iterations;
            policy_evaluation();
            if (policy_improvement()) break;
        }
    }

    void print_policy(ostream& os) const {
        os << "Optimal policy pi* (rows = cars at loc1/A 0..20, cols = cars at loc2/B 0..20)\n";
        os << "Positive = move A->B, negative = move B->A\n\n";
        os << "    ";
        for (int n2 = 0; n2 <= MAX_CARS; ++n2) os << setw(3) << n2;
        os << "\n";
        for (int n1 = MAX_CARS; n1 >= 0; --n1) {
            os << setw(3) << n1 << " ";
            for (int n2 = 0; n2 <= MAX_CARS; ++n2) {
                os << setw(3) << policy[n1][n2];
            }
            os << "\n";
        }
    }

    void print_value(ostream& os) const {
        os << "\nState-value function V* (rounded):\n";
        os << "    ";
        for (int n2 = 0; n2 <= MAX_CARS; ++n2) os << setw(7) << n2;
        os << "\n";
        for (int n1 = MAX_CARS; n1 >= 0; --n1) {
            os << setw(3) << n1 << " ";
            for (int n2 = 0; n2 <= MAX_CARS; ++n2) {
                os << setw(7) << fixed << setprecision(1) << V[n1][n2];
            }
            os << "\n";
        }
    }

    void save_csv(const string& policy_path, const string& value_path) const {
        ofstream pf(policy_path);
        ofstream vf(value_path);
        for (int n1 = 0; n1 <= MAX_CARS; ++n1) {
            for (int n2 = 0; n2 <= MAX_CARS; ++n2) {
                pf << policy[n1][n2] << (n2 == MAX_CARS ? "" : ",");
                vf << V[n1][n2] << (n2 == MAX_CARS ? "" : ",");
            }
            pf << "\n";
            vf << "\n";
        }
    }
};

int main() {
    auto t0 = chrono::steady_clock::now();
    PolicyIteration solver;
    solver.solve();
    auto t1 = chrono::steady_clock::now();
    double seconds = chrono::duration<double>(t1 - t0).count();

    cout << "Policy iteration converged in " << solver.iterations
         << " improvement step(s), " << fixed << setprecision(3)
         << seconds << " seconds.\n\n";
    solver.print_policy(cout);
    solver.print_value(cout);

    ofstream out("policy_result.txt");
    out << "Policy iteration converged in " << solver.iterations
        << " improvement step(s), " << fixed << setprecision(3)
        << seconds << " seconds.\n\n";
    solver.print_policy(out);
    solver.print_value(out);
    out.close();

    solver.save_csv("policy.csv", "value.csv");
    cout << "\nSaved policy_result.txt, policy.csv, value.csv\n";
    return 0;
}
