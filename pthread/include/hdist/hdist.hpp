#pragma once

#include <vector>
#include <cmath>
#include <thread>
#define MIN_PER_PROC 1

#define debugx

void workload_distributor(int *displs, int *scounts, int bodies, int THREAD)
{
    int offset = 0, cur_rank = 0, worker_proc = 0, remain = 0;
    // [1] Each proc get jobs to do
    if (bodies >= THREAD * MIN_PER_PROC)
    {
        offset = 0;
        for (cur_rank = 0; cur_rank < THREAD; ++cur_rank)
        {
            displs[cur_rank] = offset;
            scounts[cur_rank] = std::ceil(((float)bodies - cur_rank) / THREAD);
            offset += scounts[cur_rank];
        }
    }
    // [2] Some proc has no job to do
    else
    {
        offset = 0;
        worker_proc = std::ceil((float)bodies / MIN_PER_PROC);
        remain = bodies % MIN_PER_PROC;
        for (cur_rank = 0; cur_rank < worker_proc - 1; ++cur_rank)
        {
            displs[cur_rank] = offset;
            scounts[cur_rank] = MIN_PER_PROC;
            offset += MIN_PER_PROC;
        }
        displs[cur_rank] = offset;
        scounts[cur_rank] = remain == 0 ? MIN_PER_PROC : remain;
        offset += scounts[cur_rank];
        cur_rank++;
        for (; cur_rank < THREAD; ++cur_rank)
        {
            displs[cur_rank] = offset;
            scounts[cur_rank] = 0;
        }
    }
}

namespace hdist {
    /* 2 algo to compute */

    /*
    * Jacobi: t[i,j]' = 0.25 * (t[i-1,j]+t[i,j-1]+t[i+1,j]+t[i,j+1])
    * Sor: t[i,j]'= t[i,j] + (t[i-1,j]+t[i,j-1]+t[i+1,j]+t[i,j+1] - 4*t[i,j])/w
    * w = 2: converge faster
    * w < 2: diverge
    */
    enum class Algorithm : int {
        Jacobi = 0, 
        Sor = 1
    };

    struct State {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 2.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        bool operator==(const State &that) const = default;
    };

    struct Alt {
    };

    constexpr static inline Alt alt{};

    struct Grid {
        std::vector<double> data0, data1;
        size_t current_buffer = 0;
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                : data0(size * size), data1(size * size), length(size) {
            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->operator[]({i, j}) = border_temp;
                    } else if (i == x && j == y) {
                        this->operator[]({i, j}) = source_temp;
                    } else {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        std::vector<double> &get_current_buffer() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        double &operator[](std::pair<size_t, size_t> index) {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index) {
            return current_buffer == 1 ? data0[std::get<1>(index) * length + std::get<2>(index)] : data1[
                    std::get<1>(index) * length + std::get<2>(index)];
        }

        void switch_buffer() {
            current_buffer = !current_buffer;
        }
    };


    double update_single(size_t i, size_t j, Grid &grid, const State &state) {
        double temp;
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1) {
            temp = state.border_temp;
        } else if (i == state.source_x && j == state.source_y) {
            temp = state.source_temp;
        } else {
            auto sum = (grid[{i + 1, j}] + grid[{i - 1, j}] + grid[{i, j + 1}] + grid[{i, j - 1}]);
            switch (state.algo) {
                case Algorithm::Jacobi:
                    temp = 0.25 * sum;
                    break;
                case Algorithm::Sor:
                    temp = grid[{i, j}] + (1.0 / state.sor_constant) * (sum - 4.0 * grid[{i, j}]);
                    break;
            }
        }
        return temp;
    }

    void calculate(const State &state, Grid &grid, int localoffset, int localsize, int k) {
        
        #ifdef debug
        // printf("offset: %d size: %d\n", localoffset, localsize);
        #endif
        switch (state.algo) {
            case Algorithm::Jacobi:
                for (size_t i = localoffset; i < localoffset + localsize; ++i) {
                    for (size_t j = 0; j < state.room_size; ++j) {
                        auto temp = update_single(i, j, grid, state);
                        #ifdef debug
                        // std::thread::id this_id = std::this_thread::get_id();
                        // printf("before %zu %zu %f\n", i, j,temp);
                        #endif
                        grid[{alt, i, j}] = temp;
                        #ifdef debug
                        // printf("after %zu %zu %f\n", i, j, grid[{alt, i, j}]);
                        #endif
                    }
                }
                // grid.switch_buffer();
                break;
            case Algorithm::Sor:
                // odd-even turn
                for (size_t i = localoffset; i < localoffset + localsize; i++) {
                    for (size_t j = 0; j < state.room_size; j++) {
                        if (k == ((i + j) & 1)) {
                            auto temp = update_single(i, j, grid, state);
                            grid[{alt, i, j}] = temp;
                        } else {
                            grid[{alt, i, j}] = grid[{i, j}];
                        }
                    }
                }
                //grid.switch_buffer();
                
        }
    };


} // namespace hdist