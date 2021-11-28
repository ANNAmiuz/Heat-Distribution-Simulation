#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <stdio.h>

#define debug

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

ImColor temp_to_color(double temp)
{
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0); // RGB 0-255
    return {value, 0, 255 - value};
}

int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    if (argc != 6)
    {
        std::cerr << "usage: " << argv[0] << " <SIZE> <Iteration_Limit> <GRAPH:0/1> <ALGO> <THREAD>" << std::endl;
        return 0;
    }
    int THREAD = std::atoi(argv[5]);
    int al_choice = std::atoi(argv[4]);
    bool isGraph = std::atoi(argv[3]);
    int iteration_limit = std::atoi(argv[2]);
    int size = std::atoi(argv[1]);

    bool isStart = false;
    bool first = true;
    bool finished = false;

    int iteration_num = 0;
    double speed = 0;
    size_t workload = 0, duration = 0;
    static std::chrono::high_resolution_clock::time_point before, after;

    static hdist::State current_state, last_state;
    static const char *algo_list[2] = {"jacobi", "sor"};
    current_state.room_size = size;
    current_state.source_x = size / 2;
    current_state.source_y = size / 2;
    current_state.algo = al_choice == 0 ? hdist::Algorithm::Jacobi : hdist::Algorithm::Sor;

    auto grid = hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};

    // mpi preparations

    int rank, wsize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    // each proc calculate their global offset and data size
    int *displs, *scounts;
    displs = (int *)malloc(wsize * sizeof(int));
    scounts = (int *)malloc(wsize * sizeof(int));
    Workload mywork = workload_distributor(displs, scounts, rank, size, wsize);

#ifdef debug
    // printf("rank %d localoffset: %d localsize: %d\n", rank, mywork.localoffset, mywork.localsize);
    // if (rank == 0) {
    //     for (int i = 0; i < wsize; ++i){
    //         printf("[%d] displs: %d, scounts: %d\n", i, displs[i], scounts[i]);
    //     }
    // }
#endif

#ifdef debug
    auto grid1 = hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};
#endif
#ifdef debug
    // for (int i = 0; i < current_state.room_size; i++)
    // {
    //     for (int j = 0; j < current_state.room_size; j++)
    //     {
    //         if (grid[{i, j}] != grid1[{i, j}])
    //         {
    //             printf("GRID0: [%d %d] %f %f\n", i, j, grid.data0[i * current_state.room_size + j], grid.data1[i * current_state.room_size + j]);
    //             printf("GRID1: [%d %d] %f %f\n", i, j, grid1.data0[i * current_state.room_size + j], grid1.data1[i * current_state.room_size + j]);
    //         }
    //     }
    // }
#endif

    int k_turn = 0; // for Sor method

    if (isGraph)
    {
        if (rank == 0)
        {
            graphic::GraphicContext context{"Assignment 4"};
            context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                        {
                auto io = ImGui::GetIO();
                ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
                ImGui::SetNextWindowSize(io.DisplaySize);
                ImGui::Begin("Assignment 4", nullptr,
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                ImDrawList *draw_list = ImGui::GetWindowDrawList();
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                ImGui::Text("Computation speed average %f", speed);

                if (ImGui::Button("Start") && isStart == false)
                {
                    isStart = true;
                }

                {
                    // if (isStart == false) {
                    //     ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
                    //     ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
                    //     ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100, "%f");
                    //     ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100, "%f");
                    //     ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
                    //     ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
                    //     ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
                    //     ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

                    //     if (current_state.algo == hdist::Algorithm::Sor) {
                    //         ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
                    //     }

                    //     if (ImGui::Button("Start") && isStart == false) {
                    //         isStart = true;
                    //     }

                    //     if (current_state.room_size != last_state.room_size) {
                    //         grid = hdist::Grid{
                    //                 static_cast<size_t>(current_state.room_size),
                    //                 current_state.border_temp,
                    //                 current_state.source_temp,
                    //                 static_cast<size_t>(current_state.source_x),
                    //                 static_cast<size_t>(current_state.source_y)};
                    //         first = true;
                    //     }

                    //     if (current_state != last_state) {
                    //         last_state = current_state;
                    //         finished = false;
                    //     }
                    // }
                }

                if (first)
                {
                    first = false;
                    finished = false;
                }

                if (!finished && isStart == true)
                {
                    before = std::chrono::high_resolution_clock::now();

                    // calculate entry

                    if (current_state.algo == hdist::Algorithm::Jacobi)
                    {
                        MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                        MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                    grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                    grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    }
                    else if (current_state.algo == hdist::Algorithm::Sor)
                    {
                        MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                        MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                    grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                    grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        k_turn = !k_turn;
                        MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                        MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                    grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                    grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        k_turn = !k_turn;
                    }

                    after = std::chrono::high_resolution_clock::now();
                    iteration_num++;
                    if (iteration_num == iteration_limit)
                        finished = true;
                    duration += duration_cast<std::chrono::nanoseconds>(after - before).count();
                    workload += current_state.room_size * current_state.room_size;
                    speed = static_cast<double>(workload) / static_cast<double>(duration) * 1e9;
                    if (finished)
                        std::cout << "speed: " << speed << " bodies per second average in last " << iteration_limit << " iterations" << std::endl;

#ifdef debug
                    int k_turn1 = 0;
                    Workload mywork1;
                    mywork1.localoffset = 0;
                    mywork1.localsize = current_state.room_size;
                    if (current_state.algo == hdist::Algorithm::Jacobi)
                        hdist::calculate(current_state, grid1, mywork1, k_turn1, 1);
                    else if (current_state.algo == hdist::Algorithm::Sor)
                    {
                        hdist::calculate(current_state, grid1, mywork1, k_turn1, 1); // calculate entry of standard
                        k_turn1 = !k_turn1;
                        hdist::calculate(current_state, grid1, mywork1, k_turn1, 1);
                        k_turn1 = !k_turn1;
                    }
                    for (int i = 0; i < current_state.room_size; ++i)
                    {
                        for (int j = 0; j < current_state.room_size; ++j)
                        {
                            double ans = grid1[{i, j}];
                            double myres = grid[{i, j}];
                            if (ans != myres)
                                std::cout << "ans: " << ans << " myres: " << myres << std::endl;
                        }
                    }
#endif

                    // use current_state and grid vector to display GUI

                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
                    for (size_t i = 0; i < current_state.room_size; ++i)
                    {
                        for (size_t j = 0; j < current_state.room_size; ++j)
                        {
                            auto temp = grid[{i, j}];
                            auto color = temp_to_color(temp);
                            draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                            y += current_state.block_size;
                        }
                        x += current_state.block_size;
                        y = p.y + current_state.block_size;
                    }
#ifdef debug
                    for (int i = 0; i < current_state.room_size; ++i)
                    {
                        for (int j = 0; j < current_state.room_size; ++j)
                        {
                            double ans = grid1[{i, j}];
                            double myres = grid[{i, j}];
                            if (ans != myres)
                                std::cout << i << " " << j << "ans: " << ans << " myres: " << myres << std::endl;
                        }
                    }
#endif

                    if (finished)
                    {
                        MPI_Finalize();
                        std::exit(0);
                    }
                    ImGui::End();
            } });
        }

        else if (rank != 0)
        {
            while (finished == false)
            {
                // calculate entry
                if (current_state.algo == hdist::Algorithm::Jacobi)
                {
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                }
                else if (current_state.algo == hdist::Algorithm::Sor)
                {
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    k_turn = !k_turn;
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    k_turn = !k_turn;
                }
                iteration_num++;
                if (iteration_num == iteration_limit)
                    finished = true;
            }
            MPI_Finalize();
        }
    }

    else if (isGraph == false)
    {
        while (finished == false)
        {
            if (rank == 0)
            {
                before = std::chrono::high_resolution_clock::now();
                if (current_state.algo == hdist::Algorithm::Jacobi)
                {
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                }
                else if (current_state.algo == hdist::Algorithm::Sor)
                {
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    #ifdef debug
                    // printf("mpi k: %d\n", k_turn);
                    // for (int i = 0; i < current_state.room_size; ++i)
                    // {
                    //     for (int j = 0; j < current_state.room_size; ++j)
                    //     {
                    //         double myres = grid[{i, j}];
                    //         printf("[%d %d]: %f\n", i, j, myres);
                    //     }
                    // }
                    #endif
                    k_turn = !k_turn;
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    #ifdef debug
                    // printf("mpi k: %d\n", k_turn);
                    // for (int i = 0; i < current_state.room_size; ++i)
                    // {
                    //     for (int j = 0; j < current_state.room_size; ++j)
                    //     {
                    //         double myres = grid[{i, j}];
                    //         printf("[%d %d]: %f\n", i, j, myres);
                    //     }
                    // }
                    #endif
                    k_turn = !k_turn;
                }
                after = std::chrono::high_resolution_clock::now();
                iteration_num++;

                if (iteration_num == iteration_limit)
                    finished = true;
                duration += duration_cast<std::chrono::nanoseconds>(after - before).count();
                workload += current_state.room_size * current_state.room_size;
                speed = static_cast<double>(workload) / static_cast<double>(duration) * 1e9;
                if (finished)
                    std::cout << "speed: " << speed << " bodies per second average in last " << iteration_limit << " iterations" << std::endl;
#ifdef debug
                int k_turn1 = 0;
                Workload mywork1;
                mywork1.localoffset = 0;
                mywork1.localsize = current_state.room_size;
                if (current_state.algo == hdist::Algorithm::Jacobi)
                    hdist::calculate(current_state, grid1, mywork1, k_turn1, 1);
                else if (current_state.algo == hdist::Algorithm::Sor)
                {
                    hdist::calculate(current_state, grid1, mywork1, k_turn1, 1); // calculate entry of standard
                    k_turn1 = !k_turn1;
                    hdist::calculate(current_state, grid1, mywork1, k_turn1, 1);
                    k_turn1 = !k_turn1;
                }
                for (int i = 0; i < current_state.room_size; ++i)
                {
                    for (int j = 0; j < current_state.room_size; ++j)
                    {
                        double ans = grid1[{i, j}];
                        double myres = grid[{i, j}];
                        if (ans != myres)
                            std::cout << "ans: " << ans << " myres: " << myres << std::endl;
                    }
                }
#endif
            }
            else if (rank != 0)
            {
                if (current_state.algo == hdist::Algorithm::Jacobi)
                {
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                }
                else if (current_state.algo == hdist::Algorithm::Sor)
                {
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    k_turn = !k_turn;
                    MPI_Bcast(grid.data0.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Bcast(grid.data1.data(), current_state.room_size * current_state.room_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    hdist::calculate(current_state, grid, mywork, k_turn, THREAD); // calculate entry
                    MPI_Gatherv(grid.data0.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data0.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(grid.data1.data() + mywork.localoffset * current_state.room_size, mywork.localsize * current_state.room_size, MPI_DOUBLE,
                                grid.data1.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    k_turn = !k_turn;
                }
                iteration_num++;
                if (iteration_num == iteration_limit)
                    finished = true;
            }
        }
        MPI_Finalize();
    }
}
