#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>

#define DEBUG

// hello test now
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
        std::cerr << "usage: " << argv[0] << " <SIZE> <Iteration_Limit> <GRAPH:0/1> <BLOCK_SIZE> <ALGO> " << std::endl;
        return 0;
    }
    int al_choice = std::atoi(argv[5]);
    size_t nelem = std::atoi(argv[4]);
    bool isGraph = std::atoi(argv[3]);
    int iteration_limit = std::atoi(argv[2]);
    int size = std::atoi(argv[1]);

    nelem = nelem <= size ? nelem : size;

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

#ifdef DEBUG
    auto grid1 = hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};
#endif

    /* CUDA memory: */
    double *data0_d, *data1_d;

    if (isGraph)
    {
        graphic::GraphicContext context{"Assignment 4"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                    {

        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 4", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::Text("Computation speed average %f", speed);

        if (isStart == false) {
            ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
            ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
            ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100, "%f");
            ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100, "%f");
            ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
            ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
            ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
            ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

            if (current_state.algo == hdist::Algorithm::Sor) {
                ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
            }

            if (ImGui::Button("Start") && isStart == false) {
                isStart = true;
            }
        
            if (current_state.room_size != last_state.room_size) {
                grid = hdist::Grid{
                        static_cast<size_t>(current_state.room_size),
                        current_state.border_temp,
                        current_state.source_temp,
                        static_cast<size_t>(current_state.source_x),
                        static_cast<size_t>(current_state.source_y)};
                first = true;
            }

            if (current_state != last_state) {
                last_state = current_state;
                finished = false;
            }
        }


        if (first) {
            first = false;
            finished = false;
        }

        // do cuda memory allocation: once start, never change
        if (!isStart) {
            host_call_memory_copy_to_device(current_state.room_size, grid.data0.data(), grid.data1.data(), &data0_d, &data1_d);
        }
    
        
        if (!finished && isStart == true) {
            before = std::chrono::high_resolution_clock::now();
            int algo_choice = current_state.algo == hdist::Algorithm::Jacobi? 0 : 1;
            host_call_calculate_entry(current_state.room_size, current_state.block_size, current_state.source_x, current_state.source_y, 
            current_state.source_temp, current_state.border_temp, current_state.tolerance, current_state.sor_constant, algo_choice,
            data0_d, data1_d, nelem); // calculate entry
            host_call_memory_copy_from_device(current_state.room_size, grid.data0.data(), grid.data1.data(), data0_d, data1_d);
            after = std::chrono::high_resolution_clock::now();
            iteration_num++;
            if (iteration_num == iteration_limit) finished = true;
            duration += duration_cast<std::chrono::nanoseconds>(after - before).count();
            workload += current_state.room_size * current_state.room_size;
            speed = static_cast<double>(workload) / static_cast<double>(duration) * 1e9;
            if (finished)
                std::cout << "speed: " << speed << " bodies per second average in last "<<iteration_limit <<" iterations" << std::endl;

#ifdef DEBUG
            hdist::calculate(current_state, grid1); // calculate entry of standard
#endif
        }
        
        // use current_state and grid vector to display GUI
        
        const ImVec2 p = ImGui::GetCursorScreenPos();
        grid.switch_buffer();
        if (al_choice==1) grid.switch_buffer();
        float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
        for (size_t i = 0; i < current_state.room_size; ++i) {
            for (size_t j = 0; j < current_state.room_size; ++j) {
                auto temp = grid[{i, j}];
                auto color = temp_to_color(temp);
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                y += current_state.block_size;
            }
            x += current_state.block_size;
            y = p.y + current_state.block_size;
        }

#ifdef DEBUG
        for (int i = 0; i < current_state.room_size; ++i) {
                for (int j = 0; j < current_state.room_size; ++j) {
                    double ans = grid1[{i,j}];
                    double myres = grid[{i,j}];
                    if (ans != myres)
                        std::cout<<"ans: "<<ans<<" myres: "<<myres<<std::endl;
                }
            }
#endif
        
        if (finished) {
            host_call_CUDA_clean(data0_d, data1_d);
            std::exit(0);
        }
        ImGui::End(); });
    }

    else if (isGraph == false)
    {
        host_call_memory_copy_to_device(current_state.room_size, grid.data0.data(), grid.data1.data(), &data0_d, &data1_d);
        while (finished == false)
        {
            before = std::chrono::high_resolution_clock::now();

            int algo_choice = current_state.algo == hdist::Algorithm::Jacobi ? 0 : 1;

            host_call_calculate_entry(current_state.room_size, current_state.block_size, current_state.source_x, current_state.source_y,
                                      current_state.source_temp, current_state.border_temp, current_state.tolerance, current_state.sor_constant, algo_choice,
                                      data0_d, data1_d, nelem); // calculate entry
            host_call_memory_copy_from_device(current_state.room_size, grid.data0.data(), grid.data1.data(), data0_d, data1_d);
            grid.switch_buffer();
            if (al_choice == 1)
                grid.switch_buffer();

            after = std::chrono::high_resolution_clock::now();
            iteration_num++;
            if (iteration_num == iteration_limit)
                finished = true;
            duration += duration_cast<std::chrono::nanoseconds>(after - before).count();
            workload += current_state.room_size * current_state.room_size;
            speed = static_cast<double>(workload) / static_cast<double>(duration) * 1e9;

#ifdef DEBUG
            hdist::calculate(current_state, grid1); // calculate entry of standard
            for (int i = 0; i < current_state.room_size; ++i)
            {
                for (int j = 0; j < current_state.room_size; ++j)
                {
                    double ans = grid1[{i, j}];
                    double myres = grid[{i, j}];
                    if (ans != myres)
                        printf("[%d %d] ans: %f myres: %f\n", i, j, ans, myres);
                }
            }
#endif

            if (finished)
            {
                host_call_CUDA_clean(data0_d, data1_d);
                std::cout << "speed: " << speed << " bodies per second average in last " << iteration_limit << " iterations" << std::endl;
            }
        }
    }
}
