#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

#include "tictactoe.hpp"

using namespace std;

class TicTacToePolicyBase {
public:
    virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
    virtual ~TicTacToePolicyBase() = default;
};

// The O branch deliberately remains the supplied first-empty-cell policy.
// The X branch uses the state-value table learned below.
class TicTacToePolicyDefault : public TicTacToePolicyBase {
public:
    struct TrainingStats {
        int wins = 0;
        int draws = 0;
        int losses = 0;
    };

    explicit TicTacToePolicyDefault(uint32_t seed = 20260921)
        : rng(seed) {}

    TrainingStats train(int episodes, double exploration, double learning_rate) {
        state_values.clear();
        TrainingStats stats;
        uniform_real_distribution<double> probability(0.0, 1.0);

        for (int episode = 0; episode < episodes; ++episode) {
            TicTacToe::State state;
            vector<uint32_t> visited;

            while (!terminal(state)) {
                visited.push_back(state_key(state));
                vector<TicTacToe::Action> actions = state.action_space();
                TicTacToe::Action action;
                if (state.turn == TicTacToe::PLAYER_X) {
                    action = choose_training_action(state, actions,
                                                    exploration, probability);
                } else {
                    // Fixed opponent: O always selects the first empty cell.
                    action = actions.front();
                }
                state.put(action);
            }
            visited.push_back(state_key(state));

            int return_value = terminal_return(state);
            if (return_value > 0) {
                ++stats.wins;
            } else if (return_value < 0) {
                ++stats.losses;
            } else {
                ++stats.draws;
            }
            for (uint32_t key : visited) {
                update_value(key, return_value, learning_rate);
            }
        }
        return stats;
    }

    TicTacToe::Action operator()(const TicTacToe::State& state) const override {
        vector<TicTacToe::Action> actions = state.action_space();
        if (actions.empty()) {
            return TicTacToe::Action();
        }
        if (state.turn == TicTacToe::PLAYER_O) {
            // Keep the supplied default opponent unchanged.
            return actions.front();
        }

        // Greedy control with respect to the learned state values.  Ties are
        // resolved by the first action, making the demonstration reproducible.
        TicTacToe::Action best_action = actions.front();
        double best_value = -numeric_limits<double>::infinity();
        for (const TicTacToe::Action& action : actions) {
            TicTacToe::State next = state;
            next.put(action);
            double value = state_value(next);
            if (value > best_value + 1e-12) {
                best_value = value;
                best_action = action;
            }
        }
        return best_action;
    }

    size_t learned_state_count() const {
        return state_values.size();
    }

private:
    struct Accumulator {
        double value = 0.0;
        uint64_t visits = 0;
    };

    unordered_map<uint32_t, Accumulator> state_values;
    mutable mt19937 rng;

    static uint32_t state_key(const TicTacToe::State& state) {
        return static_cast<uint32_t>(state.board) |
               (static_cast<uint32_t>(state.turn) << 18);
    }

    static bool terminal(const TicTacToe::State& state) {
        return state.test_win() || state.full();
    }

    static int terminal_return(const TicTacToe::State& state) {
        if (state.test_win()) {
            return state.turn == TicTacToe::PLAYER_O
                       ? 1
                       : -1;
        }
        return 0;
    }

    double state_value(const TicTacToe::State& state) const {
        if (state.test_win()) {
            return state.turn == TicTacToe::PLAYER_O ? 1.0 : -1.0;
        }
        if (state.full()) {
            return 0.0;
        }
        auto found = state_values.find(state_key(state));
        if (found == state_values.end() || found->second.visits == 0) {
            return 0.0;
        }
        return found->second.value;
    }

    TicTacToe::Action choose_training_action(
        const TicTacToe::State& state,
        const vector<TicTacToe::Action>& actions,
        double exploration,
        uniform_real_distribution<double>& probability) const {
        if (probability(rng) < exploration) {
            uniform_int_distribution<size_t> index(0, actions.size() - 1);
            return actions[index(rng)];
        }

        TicTacToe::Action best_action = actions.front();
        double best_value = -numeric_limits<double>::infinity();
        for (const TicTacToe::Action& action : actions) {
            TicTacToe::State next = state;
            next.put(action);
            double value = state_value(next);
            if (value > best_value + 1e-12) {
                best_value = value;
                best_action = action;
            }
        }
        return best_action;
    }

    void update_value(uint32_t key, int return_value, double learning_rate) {
        Accumulator& accumulator = state_values[key];
        const double target = static_cast<double>(return_value);
        accumulator.value += learning_rate * (target - accumulator.value);
        ++accumulator.visits;
    }
};

static int play_against_default(const TicTacToePolicyDefault& policy) {
    TicTacToe env(false);
    while (!env.done()) {
        TicTacToe::State state = env.get_state();
        env.step(policy(state));
    }
    return env.winner();
}

int main() {
    constexpr int episodes = 120000;
    constexpr double exploration = 0.25;
    constexpr double learning_rate = 0.08;

    TicTacToePolicyDefault policy;
    TicTacToePolicyDefault::TrainingStats stats =
        policy.train(episodes, exploration, learning_rate);

    cout << "State-value training completed" << endl;
    cout << "  episodes: " << episodes << endl;
    cout << "  exploration epsilon: " << exploration << endl;
    cout << "  learning rate alpha: " << learning_rate << endl;
    cout << "  learned states: " << policy.learned_state_count() << endl;
    cout << "  exploration outcomes (X/O/draw): " << stats.wins << "/"
         << stats.losses << "/" << stats.draws << endl;

    int verification_winner = play_against_default(policy);
    cout << "Verification winner: "
         << TicTacToe::PLAYER_NAME[verification_winner] << endl;
    cout << "--- final learned-policy vs default-policy game ---" << endl;

    TicTacToe env(true);
    while (!env.done()) {
        TicTacToe::State state = env.get_state();
        TicTacToe::Action action = policy(state);
        cout << (state.turn == TicTacToe::PLAYER_X
                     ? "Learned X action: ("
                     : "Default O action: (")
             << action.x << "," << action.y << ")" << endl;
        env.step(action);
    }

    // done() already printed the winner in verbose mode; suppress the
    // duplicate diagnostic while still obtaining the result explicitly.
    env.verbose = false;
    int winner = env.winner();
    cout << "Final result: "
         << (winner == TicTacToe::PLAYER_X
                 ? "X wins"
                 : winner == TicTacToe::PLAYER_O ? "O wins" : "Draw")
         << endl;
    return 0;
}
