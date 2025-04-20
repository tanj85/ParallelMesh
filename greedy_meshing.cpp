#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <omp.h>

struct Rect { int x, y, width, height, val; };

// 1) load flat grid + dims
std::vector<int> readGridFromFile(const std::string& filename,
                                  int& width, int& height)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<int> grid;
    width = height = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int v;
        while (iss >> v) row.push_back(v);
        if (row.empty()) continue;
        if (width == 0) width = (int)row.size();
        for (int x: row) grid.push_back(x);
        height++;
    }
    return grid;
}

// 2) greedyMeshChunk(appends into rects)
void greedyMeshChunk(const int* grid,
                     int totalWidth,
                     int x0, int y0,
                     int w, int h,
                     char* visited,
                     std::vector<Rect>& rects)
{
    rects.reserve(rects.size() + (w*h)/8);
    for (int dy = 0; dy < h; ++dy) {
        int gy = y0 + dy;
        for (int dx = 0; dx < w; ++dx) {
            int gx = x0 + dx;
            int idx = gy*totalWidth + gx;
            if (visited[idx] || grid[idx] == 0) continue;
            int val = grid[idx];

            // expand width
            int W = 0;
            while (dx+W < w) {
                int i2 = gy*totalWidth + (gx+W);
                if (grid[i2]!=val || visited[i2]) break;
                ++W;
            }

            // expand height
            int H = 1;
            bool ok;
            do {
                ok = true;
                if (gy+H >= y0+h) break;
                for (int xx = 0; xx < W; ++xx) {
                    int i2 = (gy+H)*totalWidth + (gx+xx);
                    if (grid[i2]!=val || visited[i2]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) ++H;
            } while (ok);

            // mark visited
            for (int yy = 0; yy < H; ++yy)
                for (int xx = 0; xx < W; ++xx)
                    visited[(gy+yy)*totalWidth + (gx+xx)] = 1;

            rects.push_back({gx, gy, W, H, val});
        }
    }
}

// 3) recursiveGreedyMesh with local leaf buffers + critical merge
// void recursiveGreedyMesh(const int* grid,
//                         int totalWidth, int totalHeight,
//                         int x0, int y0, int w, int h,
//                         int chunkSize,
//                         std::vector<Rect>& output,
//                         char* visited)
// {
//     if (w <= chunkSize && h <= chunkSize) {
//         // leaf: do local, then merge under lock
//         std::vector<Rect> local;
//         greedyMeshChunk(grid, totalWidth, x0, y0, w, h, visited, local);
//         #pragma omp critical
//         output.insert(output.end(), local.begin(), local.end());
//         return;
//     }

//     int midW = w/2, midH = h/2;
//     int area = w*h;
//     int threads = omp_get_max_threads();
//     int totalCells = totalWidth * totalHeight;
//     int min_grain = totalCells / (threads * 4);
//     bool do_parallel = (area >= min_grain);

//     if (do_parallel) {
//         #pragma omp task shared(output, visited)
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0,      y0,      midW,   midH,
//                             chunkSize, output, visited);

//         #pragma omp task shared(output, visited)
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0+midW, y0,      w-midW, midH,
//                             chunkSize, output, visited);

//         #pragma omp task shared(output, visited)
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0,      y0+midH, midW,   h-midH,
//                             chunkSize, output, visited);

//         #pragma omp task shared(output, visited)
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0+midW, y0+midH, w-midW, h-midH,
//                             chunkSize, output, visited);

//         #pragma omp taskwait
//     } else {
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0,      y0,      midW,   midH,
//                             chunkSize, output, visited);
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0+midW, y0,      w-midW, midH,
//                             chunkSize, output, visited);
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0,      y0+midH, midW,   h-midH,
//                             chunkSize, output, visited);
//         recursiveGreedyMesh(grid, totalWidth, totalHeight,
//                             x0+midW, y0+midH, w-midW, h-midH,
//                             chunkSize, output, visited);
//     }
// }

// 4) top‐level parallel wrapper
// std::vector<Rect> recursiveParallelMeshing(const std::vector<int>& grid,
//                                            int width, int height,
//                                            int chunkSize)
// {
//     std::vector<Rect> result;
//     std::vector<char> visited(width * height, 0);

//     #pragma omp parallel
//     {
//         #pragma omp single nowait
//         recursiveGreedyMesh(grid.data(),
//                             width, height,
//                             0, 0, width, height,
//                             chunkSize,
//                             result,
//                             visited.data());
//     }
//     return result;
// }

std::vector<Rect> parallelMeshing(const std::vector<int>& grid,
                                                int width, int height,
                                                int task_num)
{
    std::vector<Rect> result;
    std::vector<char> visited(width * height, 0);

    // Compute rows and cols of tiling (trying to make tiles square)
    int rows = std::sqrt(task_num);
    int cols = (task_num + rows - 1) / rows;
    task_num = rows * cols; // Ensure full grid is covered

    std::vector<std::vector<Rect>> task_rects(task_num);

    #pragma omp parallel for collapse(2)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int task_id = r * cols + c;

            int x0 = (c * width) / cols;
            int x1 = ((c + 1) * width) / cols;
            int y0 = (r * height) / rows;
            int y1 = ((r + 1) * height) / rows;

            int w = x1 - x0;
            int h = y1 - y0;

            greedyMeshChunk(grid.data(), width, x0, y0, w, h,
                            visited.data(), task_rects[task_id]);
        }
    }

    for (auto& r : task_rects)
        result.insert(result.end(), r.begin(), r.end());

    return result;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./greedy_meshing <input_file> <task_num>\n";
        return 1;
    }

    int width, height;
    auto grid = readGridFromFile(argv[1], width, height);
    int task_num = std::stoi(argv[2]);

    std::vector<char> vis0(width * height, 0);
    std::vector<Rect> serial;
    auto t0 = std::chrono::high_resolution_clock::now();
    greedyMeshChunk(grid.data(), width, 0, 0, width, height, vis0.data(), serial);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dur1 = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Serial runtime: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    t0 = std::chrono::high_resolution_clock::now();
    auto parallel = parallelMeshing(grid, width, height, task_num);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel runtime: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::cout << "Speedup: " << dur1 / std::chrono::duration<double>(t1 - t0).count() << std::endl;

    return 0;
}